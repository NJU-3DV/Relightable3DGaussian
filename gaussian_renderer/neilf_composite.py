import math
import torch
import numpy as np
import torch.nn.functional as F
from arguments import OptimizationParams
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel
from scene.derect_light_sh import DirectLightEnv
from utils.sh_utils import eval_sh, eval_sh_coef
from utils.graphics_utils import fibonacci_sphere_sampling
from .r3dg_rasterization import GaussianRasterizationSettings, GaussianRasterizer


def render_view(viewpoint_camera: Camera, pc: GaussianModel, pipe, bg_color: torch.Tensor,
                scaling_modifier=1.0, override_color=None, is_training=False, 
                dict_params=None, bake=False):
    
    direct_light_env_light = dict_params.get("env_light")
    gamma_transform = dict_params.get("gamma")
    sample_num = dict_params.get("sample_num")

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    intrinsic = viewpoint_camera.intrinsics

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        cx=float(intrinsic[0, 2]),
        cy=float(intrinsic[1, 2]),
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        backward_geometry=True,
        computer_pseudo_normal=True,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.compute_SHs_python:
            dir_pp_normalized = F.normalize(viewpoint_camera.camera_center.repeat(means3D.shape[0], 1) - means3D,
                                            dim=-1)
            shs_view = pc.get_shs.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_shs
    else:
        colors_precomp = override_color

    base_color = pc.get_base_color
    roughness = pc.get_roughness
    metallic = pc.get_metallic
    normal = pc.get_normal
    visibility = pc.get_visibility
    incidents = pc.get_incidents
    viewdirs = F.normalize(viewpoint_camera.camera_center - means3D, dim=-1)

    # process by chunks to save memory
    # TODO: rewrite in CUDA
    chunk_size = base_color.shape[0] // ((sample_num - 1) // 8 + 1)
    brdf_color_chunks = []
    extra_results_chunks = []
    for offset in range(0, base_color.shape[0], chunk_size):
        brdf_color, extra_results = rendering_equation_python(
            base_color[offset:offset+chunk_size],
            roughness[offset:offset+chunk_size],
            metallic[offset:offset+chunk_size],
            normal.detach()[offset:offset+chunk_size],
            viewdirs[offset:offset+chunk_size],
            incidents[offset:offset+chunk_size],
            is_training, direct_light_env_light,
            visibility[offset:offset+chunk_size], sample_num, bake,
            visibility_precompute=None if bake else pc._visibility_tracing[offset:offset+chunk_size])
        
        brdf_color_chunks.append(brdf_color)
        extra_results_chunks.append(extra_results)
        torch.cuda.empty_cache()
    brdf_color = torch.cat(brdf_color_chunks, dim=0)
    extra_results = {k: torch.cat([x[k] for x in extra_results_chunks], dim=0) for k in extra_results_chunks[0].keys()}

    features = torch.cat([brdf_color, normal, base_color, roughness, metallic,
                          extra_results["incident_lights"],
                          extra_results["local_incident_lights"],
                          extra_results["global_incident_lights"],
                          extra_results["incident_visibility"]], dim=-1)
    
    (num_rendered, num_contrib, rendered_image, rendered_opacity, rendered_depth,
     rendered_feature, rendered_pseudo_normal, rendered_surface_xyz, radii) = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
        features=features,
    )
    feature_dict = {}
    rendered_pbr, rendered_normal, rendered_base_color, rendered_roughness, rendered_metallic, \
        rendered_light, rendered_local_light, rendered_global_light, rendered_visibility \
        = rendered_feature.split([3, 3, 3, 1, 1, 3, 3, 3, 1], dim=0)

    feature_dict.update({"base_color": rendered_base_color,
                         "roughness": rendered_roughness,
                         "metallic": rendered_metallic,
                         "lights": rendered_light,
                         "local lights": rendered_local_light,
                         "global lights": rendered_global_light,
                         "visibility": rendered_visibility,
                         })

    pbr = rendered_pbr
    rendered_pbr = pbr + (1 - rendered_opacity) * bg_color[:, None, None]
    
    # HDR out radiance to LDR
    val_gamma = 0
    if gamma_transform is not None:
        rendered_pbr = gamma_transform.hdr2ldr(rendered_pbr)
        val_gamma = gamma_transform.gamma.item()

    results = {"render": rendered_image,
               "pbr": rendered_pbr,
               "normal": rendered_normal,
               "pseudo_normal": rendered_pseudo_normal,
               "surface_xyz": rendered_surface_xyz,
               "opacity": rendered_opacity,
               "depth": rendered_depth,
               "viewspace_points": screenspace_points,
               "visibility_filter": radii > 0,
               "radii": radii,
               "num_rendered": num_rendered,
               "num_contrib": num_contrib}
    results.update(feature_dict)
    results["hdr"] = viewpoint_camera.hdr
    results["val_gamma"] = val_gamma

    if not is_training:
        directions = viewpoint_camera.get_world_directions()
        if isinstance(direct_light_env_light, DirectLightEnv):
            shs_view_direct = direct_light_env_light.get_env_shs.transpose(1, 2).unsqueeze(1)
            env = torch.clamp_min(eval_sh(direct_light_env_light.sh_degree, shs_view_direct, directions.permute(1, 2, 0)) + 0.5, 0).permute(2, 0, 1)
        else:
            env = direct_light_env_light.direct_light(directions.permute(1, 2, 0)).permute(2, 0, 1)
        results["render"] = rendered_image + (1 - rendered_opacity) * env
        results["pbr_env"] = pbr + (1 - rendered_opacity) * env

    return results


def render_neilf_composite(viewpoint_camera: Camera, pc: GaussianModel, pipe, bg_color: torch.Tensor,
                 scaling_modifier=1.0, override_color=None, opt: OptimizationParams = False,
                 is_training=False, dict_params=None, bake=False):
    """
    Render the scene.
    Background tensor (bg_color) must be on GPU!
    """
    results = render_view(viewpoint_camera, pc, pipe, bg_color,
                          scaling_modifier, override_color,
                          is_training, dict_params, bake)

    return results


def rendering_equation_python(base_color, roughness, metallic, normals, viewdirs, incidents, 
                              is_training=False, direct_light_env_light=None, visibility=None, 
                              sample_num=24, bake=False, visibility_precompute=None):
    
    incident_dirs, incident_areas = sample_incident_rays(normals, is_training, sample_num)

    base_color = base_color.unsqueeze(-2).contiguous()
    roughness = roughness.unsqueeze(-2).contiguous()
    metallic = metallic.unsqueeze(-2).contiguous()
    normals = normals.unsqueeze(-2).contiguous()
    viewdirs = viewdirs.unsqueeze(-2).contiguous()

    deg = int(np.sqrt(visibility.shape[1]) - 1)
    incident_dirs_coef = eval_sh_coef(deg, incident_dirs).unsqueeze(2)
    shs_view = incidents.transpose(1, 2).view(base_color.shape[0], 1, 3, -1)
    
    shs_visibility = visibility.transpose(1, 2).view(base_color.shape[0], 1, 1, -1)
    local_incident_lights = torch.clamp_min((incident_dirs_coef[..., :shs_view.shape[-1]] * shs_view).sum(-1), 0)
    if direct_light_env_light is not None:
        if isinstance(direct_light_env_light, DirectLightEnv):
            shs_view_direct = direct_light_env_light.get_env_shs.transpose(1, 2).unsqueeze(1)
            global_incident_lights = torch.clamp_min(
                (incident_dirs_coef[..., :shs_view_direct.shape[-1]] * shs_view_direct).sum(-1) + 0.5, 0)
        else:
            global_incident_lights = direct_light_env_light.direct_light(incident_dirs)
    else:
        global_incident_lights = torch.zeros_like(local_incident_lights, requires_grad=False)

    if bake:
        incident_visibility = torch.clamp(
            (incident_dirs_coef[..., :shs_visibility.shape[-1]] * shs_visibility).sum(-1) + 0.5, 0, 1)
    else:
        if visibility_precompute is not None:
            incident_visibility = visibility_precompute
        else:
            raise ValueError("visibility should be pre-computed.")

    global_incident_lights = global_incident_lights * incident_visibility
    incident_lights = local_incident_lights + global_incident_lights

    def _dot(a, b):
        return (a * b).sum(dim=-1, keepdim=True)  # [H, W, 1, 1]

    def _f_diffuse(base_color, metallic):
        return (1 - metallic) * base_color / np.pi  # [H, W, 1, 3]

    def _f_specular(h_d_n, h_d_o, n_d_i, n_d_o, base_color, roughness, metallic):
        # used in SG, wrongly normalized
        def _d_sg(r, cos):
            r2 = (r * r).clamp(min=1e-7)
            amp = 1 / (r2 * np.pi)
            sharp = 2 / r2
            return amp * torch.exp(sharp * (cos - 1))

        D = _d_sg(roughness, h_d_n)
       
        # Fresnel term F
        F_0 = 0.04 * (1 - metallic) + base_color * metallic  # [H, W, 1, 3]
        F = F_0 + (1.0 - F_0) * ((1.0 - h_d_o) ** 5)  # [H, W, S, 3]

        # geometry term V, we use V = G / (4 * cos * cos) here
        def _v_schlick_ggx(r, cos):
            r2 = ((1 + r) ** 2) / 8
            return 0.5 / (cos * (1 - r2) + r2).clamp(min=1e-7)

        V = _v_schlick_ggx(roughness, n_d_i) * _v_schlick_ggx(roughness, n_d_o)  # [H, W, S, 1]

        return D * F * V

    # half vector and all cosines
    half_dirs = incident_dirs + viewdirs
    half_dirs = F.normalize(half_dirs, dim=-1)

    h_d_n = _dot(half_dirs, normals).clamp(min=0)
    h_d_o = _dot(half_dirs, viewdirs).clamp(min=0)
    n_d_i = _dot(normals, incident_dirs).clamp(min=0)
    n_d_o = _dot(normals, viewdirs).clamp(min=0)
    
    f_d = _f_diffuse(base_color, metallic)
    f_s = _f_specular(h_d_n, h_d_o, n_d_i, n_d_o, base_color, roughness, metallic)

    transport = incident_lights * incident_areas * n_d_i
    rgb_d = (f_d * transport).mean(dim=-2)
    rgb_s = (f_s * transport).mean(dim=-2)
    rgb = rgb_d + rgb_s

    extra_results = {
        "incident_lights": incident_lights.mean(dim=-2),
        "local_incident_lights": local_incident_lights.mean(dim=-2),
        "global_incident_lights": global_incident_lights.mean(dim=-2),
        "incident_visibility": incident_visibility.mean(dim=-2),
    }
    
    return rgb, extra_results


def sample_incident_rays(normals, is_training=False, sample_num=24):
    if is_training:
        incident_dirs, incident_areas = fibonacci_sphere_sampling(
            normals, sample_num, random_rotate=True)
    else:
        incident_dirs, incident_areas = fibonacci_sphere_sampling(
            normals, sample_num, random_rotate=False)

    return incident_dirs, incident_areas  # [N, S, 3], [N, S, 1]
