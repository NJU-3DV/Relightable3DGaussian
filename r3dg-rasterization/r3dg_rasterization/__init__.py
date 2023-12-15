import os
from typing import NamedTuple
import torch.nn as nn
import torch
from utils.system_utils import Timing

from r3dg_rasterization import _C


def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)


def rasterize_gaussians(
        means3D,
        means2D,
        features,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        features,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    )


class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx,
            means3D,
            means2D,
            features,
            sh,
            colors_precomp,
            opacities,
            scales,
            rotations,
            cov3Ds_precomp,
            raster_settings,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg,
            means3D,
            features,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.cx,
            raster_settings.cy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.computer_pseudo_normal,
            raster_settings.debug
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args)  # Copy them before they can be corrupted
            try:
                num_rendered, num_contrib, color, opacity, depth, feature, normal, surface_xyz, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(
                    *args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, num_contrib, color, opacity, depth, feature, normal, surface_xyz, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(
                *args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, means3D, features, scales, rotations, cov3Ds_precomp,
                              radii, sh, geomBuffer, binningBuffer, imgBuffer)
        return num_rendered, num_contrib, color, opacity, depth, feature, normal, surface_xyz, radii

    @staticmethod
    def backward(ctx, grad_num_rendered, grad_num_contrib, grad_out_color, grad_out_opacity, grad_out_depth,
                 grad_out_feature, grad_out_normal, grad_out_surface_xyz, grad_out_radii):
        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, means3D, features, scales, rotations, cov3Ds_precomp, radii, sh, \
            geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D,
                features,
                radii,
                colors_precomp,
                scales,
                rotations,
                raster_settings.scale_modifier,
                cov3Ds_precomp,
                raster_settings.viewmatrix,
                raster_settings.projmatrix,
                raster_settings.tanfovx,
                raster_settings.tanfovy,
                grad_out_color,
                grad_out_opacity,
                grad_out_depth,
                grad_out_feature,
                sh,
                raster_settings.sh_degree,
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                raster_settings.backward_geometry,
                raster_settings.debug)
        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args)  # Copy them before they can be corrupted
            try:
                grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_features, grad_cov3Ds_precomp, \
                    grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
            grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_features, grad_cov3Ds_precomp, \
                grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_features,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
        )

        return grads


class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int
    tanfovx: float
    tanfovy: float
    cx: float
    cy: float
    bg: torch.Tensor
    scale_modifier: float
    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    sh_degree: int
    campos: torch.Tensor
    prefiltered: bool
    backward_geometry: bool
    computer_pseudo_normal: bool
    debug: bool


class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)

        return visible

    def forward(self, means3D, means2D, opacities, shs=None, colors_precomp=None,
                scales=None, rotations=None, cov3D_precomp=None, features=None):

        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')

        if ((scales is None or rotations is None) and cov3D_precomp is None) or (
                (scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')

        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        if features is None:
            features = torch.empty_like(means3D[..., :0])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            features,
            shs,
            colors_precomp,
            opacities,
            scales,
            rotations,
            cov3D_precomp,
            raster_settings,
        )


class _RenderEquation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, base_color, roughness, metallic, normals,
                viewdirs, incidents_shs, direct_shs, visibility_shs,
                sample_num, is_training, debug=False):
        pbr, incident_dirs, diffuse_light = _C.render_equation_forward(
            base_color, roughness, metallic, normals,
            viewdirs, incidents_shs, direct_shs, visibility_shs,
            sample_num, is_training, debug)

        ctx.sample_num = sample_num
        ctx.debug = debug
        ctx.save_for_backward(base_color, roughness, metallic, normals,
                              viewdirs, incidents_shs, direct_shs, visibility_shs, incident_dirs)
        return pbr, incident_dirs, diffuse_light

    @staticmethod
    def backward(ctx, grad_pbr, grad_incident_dirs, grad_diffuse_light):
        base_color, roughness, metallic, normals, \
            viewdirs, incidents_shs, direct_shs, visibility_shs, incident_dirs = ctx.saved_tensors
        debug = ctx.debug
        sample_num = ctx.sample_num

        (dL_dbase_color, dL_droughness, dL_dmetallic, dL_dnormals, dL_dviewdirs,
         dL_dincidents_shs, dL_ddirect_shs, dL_dvisibility_shs) = _C.render_equation_backward(
            base_color, roughness, metallic, normals,
            viewdirs, incidents_shs, direct_shs,
            visibility_shs, sample_num, incident_dirs, grad_pbr, grad_diffuse_light, debug)
        grads = (
            dL_dbase_color,
            dL_droughness,
            dL_dmetallic,
            dL_dnormals,
            dL_dviewdirs,
            dL_dincidents_shs,
            dL_ddirect_shs,
            dL_dvisibility_shs,
            None,
            None,
            None,
        )
        return grads


def RenderEquation_complex(base_color, roughness, metallic, normals,
                   viewdirs, incidents_shs, direct_shs, visibility_shs,
                   sample_num):
    return _C.render_equation_forward_complex(
        base_color, roughness, metallic, normals,
        viewdirs, incidents_shs, direct_shs, visibility_shs,
        sample_num)

def RenderEquation(base_color, roughness, metallic, normals,
                   viewdirs, incidents_shs, direct_shs, visibility_shs,
                   sample_num, is_training, debug=False):
    return _RenderEquation.apply(
            base_color, roughness, metallic, normals,
            viewdirs, incidents_shs, direct_shs, visibility_shs,
            sample_num, is_training, debug)
