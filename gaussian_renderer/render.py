
# This is script for 3D Gaussian Splatting rendering

import math
import torch
import torch.nn.functional as F
from arguments import OptimizationParams
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.loss_utils import ssim, first_order_edge_aware_loss, second_order_edge_aware_loss, \
    bilateral_smooth_loss, tv_loss
from utils.image_utils import psnr
from .r3dg_rasterization import GaussianRasterizationSettings, GaussianRasterizer


def render_view(camera: Camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, 
                scaling_modifier, override_color, computer_pseudo_normal=True):
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means

    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(camera.FoVx * 0.5)
    tanfovy = math.tan(camera.FoVy * 0.5)
    intrinsic = camera.intrinsics
    raster_settings = GaussianRasterizationSettings(
        image_height=int(camera.image_height),
        image_width=int(camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        cx=float(intrinsic[0, 2]),
        cy=float(intrinsic[1, 2]),
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=camera.world_view_transform,
        projmatrix=camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=camera.camera_center,
        prefiltered=False,
        backward_geometry=True,
        computer_pseudo_normal=computer_pseudo_normal,
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
            shs_view = pc.get_shs.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = (pc.get_xyz - camera.camera_center.repeat(pc.get_shs.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_shs
    else:
        colors_precomp = override_color

    normals = pc.get_normal
    
    dir_pp = (pc.get_xyz - camera.camera_center.repeat(pc.get_shs.shape[0], 1))
    dir_pp_normalized = F.normalize(dir_pp, dim=-1)
    
    xyz_homo = torch.cat([means3D, torch.ones_like(means3D[:, :1])], dim=-1)
    depths = (xyz_homo @ camera.world_view_transform)[:, 2:3]
    depths2 = depths.square()
    features = torch.cat([normals, depths, depths2], dim=-1)
    
    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    (num_rendered, num_contrib, rendered_image, rendered_opacity, rendered_depth,
     rendered_feature, rendered_pseudo_normal, rendered_surface_xyz, weights, radii) = rasterizer(
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
     
    mask = num_contrib > 0
    rendered_feature = rendered_feature / rendered_opacity.clamp_min(1e-5) * mask
    # rendered_depth = rendered_depth / rendered_opacity.clamp_min(1e-5) * mask
    
    rendered_normal, rendered_depth, rendered_depth2 = torch.split(rendered_feature, [3, 1, 1], dim=0)
    
    rendered_var = rendered_depth2 - rendered_depth.square()

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    results = {"render": rendered_image,
               "opacity": rendered_opacity,
               "depth": rendered_depth,
               "depth_var": rendered_var,
               "normal": rendered_normal,
               "pseudo_normal": rendered_pseudo_normal,
               "surface_xyz": rendered_surface_xyz,
               "viewspace_points": screenspace_points,
               "visibility_filter": radii > 0,
               "radii": radii,
               "num_rendered": num_rendered,
               "num_contrib": num_contrib,
               "opacities": opacity,
               "normals": normals,
               "directions": dir_pp_normalized,
               "weights": weights}
    
    return results

def calculate_loss(viewpoint_camera, pc, render_pkg, opt, iteration):
    tb_dict = {
        "num_points": pc.get_xyz.shape[0],
    }
    
    rendered_image = render_pkg["render"]
    rendered_opacity = render_pkg["opacity"]
    rendered_depth = render_pkg["depth"]
    rendered_normal = render_pkg["normal"]
    visibility_filter = render_pkg["visibility_filter"]
    gt_image = viewpoint_camera.original_image.cuda()
    image_mask = viewpoint_camera.image_mask.cuda()

    Ll1 = F.l1_loss(rendered_image, gt_image)
    ssim_val = ssim(rendered_image, gt_image)
    tb_dict["loss_l1"] = Ll1.item()
    tb_dict["psnr"] = psnr(rendered_image, gt_image).mean().item()
    tb_dict["ssim"] = ssim_val.item()
    loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_val)

    if opt.lambda_mask_entropy > 0:
        o = rendered_opacity.clamp(1e-6, 1 - 1e-6)
        loss_mask_entropy = -(image_mask * torch.log(o) + (1-image_mask) * torch.log(1 - o)).mean()
        tb_dict["loss_mask_entropy"] = loss_mask_entropy.item()
        loss = loss + opt.lambda_mask_entropy * loss_mask_entropy

    if opt.lambda_normal_render_depth > 0:
        normal_pseudo = render_pkg['pseudo_normal']
        loss_normal_render_depth = F.mse_loss(
            rendered_normal * image_mask, normal_pseudo.detach() * image_mask)
        tb_dict["loss_normal_render_depth"] = loss_normal_render_depth.item()
        loss = loss + opt.lambda_normal_render_depth * loss_normal_render_depth

    if opt.lambda_normal_smooth > 0:
        loss_normal_smooth = first_order_edge_aware_loss(rendered_normal, gt_image)
        tb_dict["loss_normal_smooth"] = loss_normal_smooth.item()
        lambda_normal_smooth = opt.lambda_normal_smooth
        loss = loss + lambda_normal_smooth * loss_normal_smooth
    
    if opt.lambda_depth_smooth > 0:
        loss_depth_smooth = first_order_edge_aware_loss(rendered_depth, gt_image)
        tb_dict["loss_depth_smooth"] = loss_depth_smooth.item()
        lambda_depth_smooth = opt.lambda_depth_smooth
        loss = loss + lambda_depth_smooth * loss_depth_smooth
        
    if opt.lambda_point_entropy > 0:
        ws = render_pkg["weights"]
        vis_opacities = render_pkg["opacities"]
        loss_point_entropy = (ws * (
                        - vis_opacities * torch.log(vis_opacities + 1e-10)
                        - (1 - vis_opacities) * torch.log(1 - vis_opacities + 1e-10)
                        )).mean()
        tb_dict["loss_normal_smooth"] = loss_point_entropy.item()
        loss = loss + opt.lambda_point_entropy * loss_point_entropy
        
    if opt.lambda_orientation > 0 and iteration > opt.lambda_orientation_from_iter:
        ws = render_pkg["weights"].clamp_max(1)
        normals = render_pkg["normals"]
        directions = render_pkg["directions"]
        loss_orientation = (ws * (normals * directions).sum(-1, keepdim=True).clamp_min(0.0)).mean()
        tb_dict["loss_orientation"] = loss_orientation.item()
        loss = loss + opt.lambda_orientation * loss_orientation
    
    if opt.lambda_depth_var > 0:
        depth_var = render_pkg["depth_var"]
        loss_depth_var = depth_var.clamp_min(1e-6).sqrt().mean()
        tb_dict["loss_depth_var"] = loss_depth_var.item()
        lambda_depth_var = opt.lambda_depth_var * min(math.pow(10, iteration / 5000), 100)
        # lambda_depth_var = opt.lambda_depth_var
        loss = loss + lambda_depth_var * loss_depth_var
    
    
    if opt.lambda_surface > 0:
        center, _ = torch.median(pc.get_xyz, dim=0)
        loss_surface = torch.exp(-(pc.get_xyz - center[None, ...]).abs().mean())
        
        tb_dict["loss_surface"] = loss_surface.item()
        loss = loss + opt.lambda_surface * loss_surface
        
    if opt.lambda_scaling > 0:
        scaling = pc.get_scaling
        scaling_loss = (scaling - scaling.mean(dim=-1, keepdim=True)).abs().sum(-1).mean()
        lambda_scaling = opt.lambda_scaling - 0.99 * opt.lambda_scaling * min(1, 4 * iteration / opt.iterations)
        loss = loss + lambda_scaling * scaling_loss
    
    tb_dict["loss"] = loss.item()
    
    return loss, tb_dict

def render(viewpoint_camera: Camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, 
           scaling_modifier=1.0,override_color=None, opt: OptimizationParams = None, 
           is_training=False, dict_params=None, iteration=0):
    """
    Render the scene.
    Background tensor (bg_color) must be on GPU!
    """
    results = render_view(viewpoint_camera, pc, pipe, bg_color, scaling_modifier, override_color,
                          computer_pseudo_normal=True if opt is not None and opt.lambda_normal_render_depth>0 else False)

    if is_training:
        loss, tb_dict = calculate_loss(viewpoint_camera, pc, results, opt, iteration)
        results["tb_dict"] = tb_dict
        results["loss"] = loss
    
    return results
