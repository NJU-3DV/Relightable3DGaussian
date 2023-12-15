import math
import torch
import numpy as np
import torch.nn.functional as F
from arguments import OptimizationParams
from bvh import RayTracer
from scene.gaussian_model import GaussianModel
from scene.cameras import Camera
from utils.sh_utils import eval_sh, eval_sh_coef
from utils.loss_utils import ssim, bilateral_smooth_loss
from utils.image_utils import psnr
from utils.graphics_utils import fibonacci_sphere_sampling
from .r3dg_rasterization import GaussianRasterizationSettings, GaussianRasterizer, RenderEquation, \
    RenderEquation_complex


def render_view(viewpoint_camera: Camera, pc: GaussianModel, pipe, bg_color: torch.Tensor,
                scaling_modifier=1.0, override_color=None, is_training=False, dict_params=None):
    direct_light_env_light = dict_params.get("env_light")
    gamma_transform = dict_params.get("gamma")

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
    incidents = pc.get_incidents  # incident shs
    viewdirs = F.normalize(viewpoint_camera.camera_center - means3D, dim=-1)

    if pipe.compute_neilf_python:
        brdf_color, extra_results = rendering_equation_python(
            base_color, roughness, metallic, normal.detach(), viewdirs, incidents,
            is_training, direct_light_env_light, visibility, sample_num=dict_params['sample_num'])
    else:
        brdf_color, extra_results = rendering_equation(
            base_color, roughness, metallic, normal.detach(), viewdirs, incidents,
            is_training, direct_light_env_light, visibility, sample_num=dict_params['sample_num'])

    if is_training:
        features = torch.cat([brdf_color, normal, base_color, roughness, metallic], dim=-1)
    else:
        features = torch.cat([brdf_color, normal, base_color, roughness, metallic,
                              extra_results["incident_lights"].mean(-2),
                              extra_results["local_incident_lights"].mean(-2),
                              extra_results["global_incident_lights"].mean(-2),
                              extra_results["incident_visibility"].mean(-2)], dim=-1)

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
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
    if is_training:
        rendered_pbr, rendered_normal, rendered_base_color, \
            rendered_roughness, rendered_metallic \
            = rendered_feature.split([3, 3, 3, 1, 1], dim=0)
        feature_dict.update({"base_color": rendered_base_color,
                             "roughness": rendered_roughness,
                             "metallic": rendered_metallic,
                             })
    else:
        rendered_pbr, rendered_normal, rendered_base_color, rendered_roughness, rendered_metallic, \
            rendered_light, rendered_local_light, rendered_global_light, rendered_visibility \
            = rendered_feature.split([3, 3, 3, 1, 1, 3, 3, 3, 1], dim=0)

        feature_dict.update({"base_color": rendered_base_color,
                             "roughness": rendered_roughness,
                             "metallic": rendered_metallic,
                             "lights": rendered_light,
                             "local_lights": rendered_local_light,
                             "global_lights": rendered_global_light,
                             "visibility": rendered_visibility,
                             })

    pbr = rendered_pbr
    rendered_pbr = pbr + (1 - rendered_opacity) * bg_color[:, None, None]

    val_gamma = 0
    if gamma_transform is not None:
        rendered_pbr = gamma_transform.hdr2ldr(rendered_pbr)
        val_gamma = gamma_transform.gamma.item()

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
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
               "num_contrib": num_contrib
               }

    results.update(feature_dict)
    results["diffuse_light"] = extra_results["diffuse_light"]
    results["hdr"] = viewpoint_camera.hdr
    results["val_gamma"] = val_gamma

    return results


def calculate_loss(viewpoint_camera, pc, results, opt):
    tb_dict = {
        "num_points": pc.get_xyz.shape[0],
    }
    rendered_image = results["render"]
    rendered_depth = results["depth"]
    rendered_normal = results["normal"]
    rendered_pbr = results["pbr"]
    rendered_opacity = results["opacity"]
    rendered_base_color = results["base_color"]
    rendered_metallic = results["metallic"]
    rendered_roughness = results["roughness"]

    gt_image = viewpoint_camera.original_image.cuda()
    Ll1 = F.l1_loss(rendered_image, gt_image)
    ssim_val = ssim(rendered_image, gt_image)
    tb_dict["l1"] = Ll1.item()
    tb_dict["psnr"] = psnr(rendered_image, gt_image).mean().item()
    tb_dict["ssim"] = ssim_val.item()
    loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_val)

    if opt.lambda_pbr > 0:
        Ll1_pbr = F.l1_loss(rendered_pbr, gt_image)
        ssim_val_pbr = ssim(rendered_pbr, gt_image)
        tb_dict["l1_pbr"] = Ll1_pbr.item()
        tb_dict["ssim_pbr"] = ssim_val_pbr.item()
        tb_dict["psnr_pbr"] = psnr(rendered_pbr, gt_image).mean().item()
        loss_pbr = (1.0 - opt.lambda_dssim) * Ll1_pbr + opt.lambda_dssim * (1.0 - ssim_val_pbr)
        loss = loss + opt.lambda_pbr * loss_pbr

    if opt.lambda_depth > 0:
        gt_depth = viewpoint_camera.depth.cuda()
        image_mask = viewpoint_camera.image_mask.cuda().bool()
        depth_mask = gt_depth > 0
        sur_mask = torch.logical_xor(image_mask, depth_mask)

        loss_depth = F.l1_loss(rendered_depth[~sur_mask], gt_depth[~sur_mask])
        tb_dict["loss_depth"] = loss_depth.item()
        loss = loss + opt.lambda_depth * loss_depth

    if opt.lambda_mask_entropy > 0:
        o = rendered_opacity.clamp(1e-6, 1 - 1e-6)
        image_mask = viewpoint_camera.image_mask.cuda()
        loss_mask_entropy = -(image_mask * torch.log(o) + (1 - image_mask) * torch.log(1 - o)).mean()
        tb_dict["loss_mask_entropy"] = loss_mask_entropy.item()
        loss = loss + opt.lambda_mask_entropy * loss_mask_entropy

    if opt.lambda_normal_render_depth > 0:
        normal_pseudo = results['pseudo_normal']
        image_mask = viewpoint_camera.image_mask.cuda()
        loss_normal_render_depth = F.mse_loss(
            rendered_normal * image_mask, normal_pseudo.detach() * image_mask)
        tb_dict["loss_normal_render_depth"] = loss_normal_render_depth.item()
        loss = loss + opt.lambda_normal_render_depth * loss_normal_render_depth

    if opt.lambda_normal_mvs_depth > 0:
        gt_depth = viewpoint_camera.depth.cuda()
        depth_mask = (gt_depth > 0).float()
        mvs_normal = viewpoint_camera.normal.cuda()
        loss_normal_mvs_depth = F.mse_loss(
            rendered_normal * depth_mask, mvs_normal * depth_mask)
        tb_dict["loss_normal_mvs_depth"] = loss_normal_mvs_depth.item()
        loss = loss + opt.lambda_normal_mvs_depth * loss_normal_mvs_depth

    if opt.lambda_light > 0:
        diffuse_light = results["diffuse_light"]
        mean_light = diffuse_light.mean(-1, keepdim=True).expand_as(diffuse_light)
        loss_light = F.l1_loss(diffuse_light, mean_light)
        tb_dict["loss_light"] = loss_light.item()
        loss = loss + opt.lambda_light * loss_light

    if opt.lambda_base_color > 0:
        image_mask = viewpoint_camera.image_mask.cuda()
        value_img = torch.max(gt_image * image_mask, dim=0, keepdim=True)[0]
        shallow_enhance = gt_image * image_mask
        shallow_enhance = 1 - (1 - shallow_enhance) * (1 - shallow_enhance)

        specular_enhance = gt_image * image_mask
        specular_enhance = specular_enhance * specular_enhance

        k = 5
        specular_weight = 1 / (1 + torch.exp(-k * (value_img - 0.5)))
        target_img = (specular_weight * specular_enhance + (1 - specular_weight) * shallow_enhance)

        loss_base_color = F.l1_loss(target_img, rendered_base_color)
        tb_dict["loss_base_color"] = loss_base_color.item()
        lambda_base_color = opt.lambda_base_color  # * max(0, 1 - float(dict_params["iteration"]) / (opt.base_color_guide_iter_num+1e-8))
        tb_dict["lambda_base_color"] = lambda_base_color

        loss = loss + lambda_base_color * loss_base_color

    if opt.lambda_base_color_smooth > 0:
        image_mask = viewpoint_camera.image_mask.cuda()
        loss_base_color_smooth = bilateral_smooth_loss(rendered_base_color, gt_image, image_mask)
        tb_dict["loss_base_color_smooth"] = loss_base_color_smooth.item()
        loss = loss + opt.lambda_base_color_smooth * loss_base_color_smooth

    if opt.lambda_metallic_smooth > 0:
        image_mask = viewpoint_camera.image_mask.cuda()
        loss_metallic_smooth = bilateral_smooth_loss(rendered_metallic, gt_image, image_mask)
        tb_dict["loss_metallic_smooth"] = loss_metallic_smooth.item()
        loss = loss + opt.lambda_metallic_smooth * loss_metallic_smooth

    if opt.lambda_roughness_smooth > 0:
        image_mask = viewpoint_camera.image_mask.cuda()
        loss_roughness_smooth = bilateral_smooth_loss(rendered_roughness, gt_image, image_mask)
        tb_dict["loss_roughness_smooth"] = loss_roughness_smooth.item()
        loss = loss + opt.lambda_roughness_smooth * loss_roughness_smooth

    if opt.lambda_visibility > 0:
        num = 10000
        means3D = pc.get_xyz
        visibility = pc.get_visibility
        normal = pc.get_normal
        opacity = pc.get_opacity

        rand_idx = torch.randperm(means3D.shape[0])[:num]
        rand_visibility_shs_view = visibility.transpose(1, 2).view(-1, 1, 4 ** 2)[rand_idx]
        rand_rays_o = means3D[rand_idx]
        rand_rays_d = torch.randn_like(rand_rays_o)
        cov_inv = pc.get_inverse_covariance()
        rand_normal = normal[rand_idx]
        mask = (rand_rays_d * rand_normal).sum(-1) < 0
        rand_rays_d[mask] *= -1
        sample_sh2vis = eval_sh(3, rand_visibility_shs_view, rand_rays_d)
        sample_vis = torch.clamp(sample_sh2vis + 0.5, 0.0, 1.0)
        raytracer = RayTracer(means3D, pc.get_scaling, pc.get_rotation)
        trace_results = raytracer.trace_visibility(
            rand_rays_o, rand_rays_d, means3D,
            cov_inv, opacity, normal)

        rand_ray_visibility = trace_results["visibility"]
        loss_visibility = F.l1_loss(rand_ray_visibility, sample_vis)
        tb_dict["loss_visibility"] = loss_visibility.item()
        loss = loss + opt.lambda_visibility * loss_visibility

    tb_dict["loss"] = loss.item()

    return loss, tb_dict


def render_neilf(viewpoint_camera: Camera, pc: GaussianModel, pipe, bg_color: torch.Tensor,
                 scaling_modifier=1.0, override_color=None, opt: OptimizationParams = False,
                 is_training=False, dict_params=None):
    """
    Render the scene.
    Background tensor (bg_color) must be on GPU!
    """
    results = render_view(viewpoint_camera, pc, pipe, bg_color,
                          scaling_modifier, override_color, is_training, dict_params)

    if is_training:
        loss, tb_dict = calculate_loss(viewpoint_camera, pc, results, opt)
        results["tb_dict"] = tb_dict
        results["loss"] = loss

    return results


def rendering_equation(base_color, roughness, metallic, normals, viewdirs,
                       incidents, is_training=False, direct_light_env_light=None,
                       visibility=None, sample_num=24):
    if is_training:
        pbr, _, diffuse_light = RenderEquation(
            base_color,
            roughness,
            metallic,
            normals,
            viewdirs,
            incidents,
            direct_light_env_light.get_env_shs,
            visibility,
            sample_num=sample_num,
            is_training=True,
            debug=False,
        )

        extra_results = {
            "pbr": pbr,
            "diffuse_light": diffuse_light,
        }
    else:
        (pbr, _, incident_lights, local_incident_lights,
         global_incident_lights, incident_visibility, diffuse_light,
         local_diffuse_light, accum, rgb_d, rgb_s) = RenderEquation_complex(
            base_color,
            roughness,
            metallic,
            normals,
            viewdirs,
            incidents,
            direct_light_env_light.get_env_shs,
            visibility,
            sample_num=sample_num,
        )

        extra_results = {
            "incident_lights": incident_lights,
            "local_incident_lights": local_incident_lights,
            "global_incident_lights": global_incident_lights,
            "incident_visibility": incident_visibility,
            "diffuse_light": diffuse_light,
            "local_diffuse_light": local_diffuse_light,
            "accum": accum,
            "rgb_d": rgb_d,
            "rgb_s": rgb_s,
            "pbr": pbr,
        }

    return pbr, extra_results


def sample_incident_rays(normals, is_training=False, sample_num=24):
    if is_training:
        incident_dirs, incident_areas = fibonacci_sphere_sampling(
            normals, sample_num, random_rotate=True)
    else:
        incident_dirs, incident_areas = fibonacci_sphere_sampling(
            normals, sample_num, random_rotate=False)

    return incident_dirs, incident_areas  # [N, S, 3], [N, S, 1]


def rendering_equation_python(base_color, roughness, metallic, normals, viewdirs,
                              incidents, is_training=False, direct_light_env_light=None,
                              visibility=None, sample_num=24):
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
        shs_view_direct = direct_light_env_light.get_env_shs.transpose(1, 2).unsqueeze(1)
        global_incident_lights = torch.clamp_min(
            (incident_dirs_coef[..., :shs_view_direct.shape[-1]] * shs_view_direct).sum(-1) + 0.5, 0)
    else:
        global_incident_lights = torch.zeros_like(local_incident_lights, requires_grad=False)

    incident_visibility = torch.clamp(
        (incident_dirs_coef[..., :shs_visibility.shape[-1]] * shs_visibility).sum(-1) + 0.5, 0, 1)
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

    half_dirs = incident_dirs + viewdirs
    half_dirs = F.normalize(half_dirs, dim=-1)

    h_d_n = _dot(half_dirs, normals).clamp(min=0)
    h_d_o = _dot(half_dirs, viewdirs).clamp(min=0)
    n_d_i = _dot(normals, incident_dirs).clamp(min=0)
    n_d_o = _dot(normals, viewdirs).clamp(min=0)
    f_d = _f_diffuse(base_color, metallic)
    f_s = _f_specular(h_d_n, h_d_o, n_d_i, n_d_o, base_color, roughness, metallic)

    transport = incident_lights * incident_areas * n_d_i  # （num_pts, num_sample, 3)
    rgb_d = (f_d * transport).mean(dim=-2)
    rgb_s = (f_s * transport).mean(dim=-2)
    pbr = rgb_d + rgb_s
    diffuse_light = transport.mean(dim=-2)

    extra_results = {
        "incident_dirs": incident_dirs,
        "incident_lights": incident_lights,
        "local_incident_lights": local_incident_lights,
        "global_incident_lights": global_incident_lights,
        "incident_visibility": incident_visibility,
        "diffuse_light": diffuse_light,
    }

    return pbr, extra_results
