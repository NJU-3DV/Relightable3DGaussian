import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from utils.general_utils import rotation_to_quaternion, quaternion_multiply
from utils.sh_utils import RGB2SH, eval_sh
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from arguments import OptimizationParams
from tqdm import tqdm
from bvh import RayTracer
from utils.graphics_utils import fibonacci_sphere_sampling


def sample_incident_rays(normals, is_training=False, sample_num=24):
    if is_training:
        incident_dirs, incident_areas = fibonacci_sphere_sampling(
            normals, sample_num, random_rotate=True)
    else:
        incident_dirs, incident_areas = fibonacci_sphere_sampling(
            normals, sample_num, random_rotate=False)

    return incident_dirs, incident_areas  # [N, S, 3], [N, S, 1]

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.normal_activation = lambda x: torch.nn.functional.normalize(x, dim=-1, eps=1e-3)
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

        if self.use_pbr:
            self.base_color_activation = lambda x: torch.sigmoid(x) * 0.77 + 0.03
            self.roughness_activation = lambda x: torch.sigmoid(x) * 0.9 + 0.09
            self.inverse_roughness_activation = lambda y: inverse_sigmoid((y-0.09) / 0.9)

    def __init__(self, sh_degree: int, render_type='render'):
        self.render_type = render_type
        self.use_pbr = render_type in ['neilf']
        self.active_sh_degree = 3
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._normal = torch.empty(0)  # normal
        self._shs_dc = torch.empty(0)  # output radiance
        self._shs_rest = torch.empty(0)  # output radiance
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._visibility_tracing = None
        self.max_radii2D = torch.empty(0)
        self.weights_accum = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.normal_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0

        self.setup_functions()
        self.transform = {}
        if self.use_pbr:
            self._base_color = torch.empty(0)
            self._roughness = torch.empty(0)
            self._incidents_dc = torch.empty(0)
            self._incidents_rest = torch.empty(0)
            self._visibility_dc = torch.empty(0)
            self._visibility_rest = torch.empty(0)
        self.base_color_scale = torch.ones(3, dtype=torch.float, device="cuda")

    @torch.no_grad()
    def set_transform(self, rotation=None, center=None, scale=None, offset=None, transform=None):
        if transform is not None:
            scale = transform[:3, :3].norm(dim=-1)
            self._scaling.data = self.scaling_inverse_activation(self.get_scaling * scale)
            xyz_homo = torch.cat([self._xyz.data, torch.ones_like(self._xyz[:, :1])], dim=-1)
            self._xyz.data = (xyz_homo @ transform.T)[:, :3]
            rotation = transform[:3, :3] / scale[:, None]
            self._normal.data = self._normal.data @ rotation.T
            rotation_q = rotation_to_quaternion(rotation[None])
            self._rotation.data = quaternion_multiply(rotation_q, self._rotation.data)
            return

        if center is not None:
            self._xyz.data = self._xyz.data - center
        if rotation is not None:
            self._xyz.data = (self._xyz.data @ rotation.T)
            self._normal.data = self._normal.data @ rotation.T
            rotation_q = rotation_to_quaternion(rotation[None])
            self._rotation.data = quaternion_multiply(rotation_q, self._rotation.data)
        if scale is not None:
            self._xyz.data = self._xyz.data * scale
            self._scaling.data = self.scaling_inverse_activation(self.get_scaling * scale)
        if offset is not None:
            self._xyz.data = self._xyz.data + offset

    def capture(self):
        captured_list = [
            self.active_sh_degree,
            self._xyz,
            self._normal,
            self._shs_dc,
            self._shs_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.weights_accum,
            self.xyz_gradient_accum,
            self.normal_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        ]
        if self.use_pbr:
            captured_list.extend([
                self._base_color,
                self._roughness,
                self._incidents_dc,
                self._incidents_rest,
                self._visibility_dc,
                self._visibility_rest,
            ])

        return captured_list

    def restore(self, model_args, training_args,
                is_training=False, restore_optimizer=True):
        (self.active_sh_degree,
         self._xyz,
         self._normal,
         self._shs_dc,
         self._shs_rest,
         self._scaling,
         self._rotation,
         self._opacity,
         self.max_radii2D,
         weights_accum,
         xyz_gradient_accum,
         normal_gradient_accum,
         denom,
         opt_dict,
         self.spatial_lr_scale) = model_args[:15]
        if len(model_args) > 15 and self.use_pbr:
            (self._base_color,
             self._roughness,
             self._incidents_dc,
             self._incidents_rest,
             self._visibility_dc,
             self._visibility_rest) = model_args[15:]

        if is_training:
            self.training_setup(training_args)
            self.weights_accum = weights_accum
            self.xyz_gradient_accum = xyz_gradient_accum
            self.normal_gradient_accum = normal_gradient_accum
            self.denom = denom
            if restore_optimizer:
                # TODO automatically match the opt_dict
                try:
                    self.optimizer.load_state_dict(opt_dict)
                except:
                    pass

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_normal(self):
        return self.normal_activation(self._normal)

    @property
    def get_shs(self):
        """SH"""
        shs_dc = self._shs_dc
        shs_rest = self._shs_rest
        return torch.cat((shs_dc, shs_rest), dim=1)

    @property
    def get_incidents(self):
        """SH"""
        incidents_dc = self._incidents_dc
        incidents_rest = self._incidents_rest
        return torch.cat((incidents_dc, incidents_rest), dim=1)

    @property
    def get_visibility(self):
        """SH"""
        visibility_dc = self._visibility_dc
        visibility_rest = self._visibility_rest
        return torch.cat((visibility_dc, visibility_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_base_color(self):
        return self.base_color_activation(self._base_color) * self.base_color_scale[None, :]

    @property
    def get_roughness(self):
        return self.roughness_activation(self._roughness)

    @property
    def get_brdf(self):
        return torch.cat([self.get_base_color, self.get_roughness], dim=-1)

    def get_by_names(self, names):
        if len(names) == 0:
            return None
        fs = []
        for name in names:
            fs.append(getattr(self, "get_" + name))
        return torch.cat(fs, dim=1)

    def split_by_names(self, features, names):
        results = {}
        last_idx = 0
        for name in names:
            current_shape = getattr(self, "_" + name).shape[1]
            results[name] = features[last_idx:last_idx + current_shape]
            last_idx += getattr(self, "_" + name).shape[1]
        return results

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling,
                                          scaling_modifier,
                                          self.get_rotation)

    def get_inverse_covariance(self, scaling_modifier=1):
        return self.covariance_activation(1 / self.get_scaling,
                                          1 / scaling_modifier,
                                          self.get_rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    @property
    def attribute_names(self):
        attribute_names = ['xyz', 'normal', 'shs_dc', 'shs_rest', 'scaling', 'rotation', 'opacity']
        if self.use_pbr:
            attribute_names.extend(['base_color', 'roughness',
                                    'incidents_dc', 'incidents_rest',
                                    'visibility_dc', 'visibility_rest'])
        return attribute_names

    def finetune_visibility(self, iterations=1000):
        visibility_sh_lr = 1e-2
        optimizer = torch.optim.Adam([
            {'params': [self._visibility_dc], 'lr': visibility_sh_lr},
            {'params': [self._visibility_rest], 'lr': visibility_sh_lr}
        ])
        means3D = self.get_xyz
        opacity = self.get_opacity[:, 0]
        scaling = self.get_scaling
        rotation = self.get_rotation
        normal = self.get_normal
        cov_inv = self.get_inverse_covariance()
        tbar = tqdm(range(iterations), desc="Finetuning visibility shs")
        raytracer = RayTracer(means3D, scaling, rotation)
        visibility_shs_view = self.get_visibility.transpose(1, 2)
        vis_sh_degree = np.sqrt(visibility_shs_view.shape[-1]) - 1
        rays_o = means3D
        for iteration in tbar:
            rays_d = torch.randn_like(rays_o)
            rays_d = F.normalize(rays_d, dim=-1)
            mask = (rays_d * normal).sum(-1) < 0
            rays_d[mask] *= -1
            sample_sh2vis = eval_sh(vis_sh_degree, visibility_shs_view, rays_d)
            sample_vis = torch.clamp(sample_sh2vis + 0.5, 0.0, 1.0)
            trace_results = raytracer.trace_visibility(
                rays_o,
                rays_d,
                means3D,
                cov_inv,
                opacity,
                normal)
            visibility = trace_results["visibility"]
            loss = F.l1_loss(visibility, sample_vis)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    
    @torch.no_grad()
    def update_visibility(self, sample_num):
        raytracer = RayTracer(self.get_xyz, self.get_scaling, self.get_rotation)
        gaussians_xyz = self.get_xyz
        gaussians_inverse_covariance = self.get_inverse_covariance()
        gaussians_opacity = self.get_opacity[:, 0]
        gaussians_normal = self.get_normal
        incident_visibility_results = []
        incident_dirs_results = []
        incident_areas_results = []
        chunk_size = gaussians_xyz.shape[0] // ((sample_num - 1) // 24 + 1)
        for offset in tqdm(range(0, gaussians_xyz.shape[0], chunk_size), "Update visibility with raytracing."):
            incident_dirs, incident_areas = sample_incident_rays(gaussians_normal[offset:offset + chunk_size], False,
                                                    sample_num)
            trace_results = raytracer.trace_visibility(
                gaussians_xyz[offset:offset + chunk_size, None].expand_as(incident_dirs),
                incident_dirs,
                gaussians_xyz,
                gaussians_inverse_covariance,
                gaussians_opacity,
                gaussians_normal)
            incident_visibility = trace_results["visibility"]
            incident_visibility_results.append(incident_visibility)
            incident_dirs_results.append(incident_dirs)
            incident_areas_results.append(incident_areas)
        incident_visibility_result = torch.cat(incident_visibility_results, dim=0)
        incident_dirs_result = torch.cat(incident_dirs_results, dim=0)
        incident_areas_result = torch.cat(incident_areas_results, dim=0)
        self._visibility_tracing = incident_visibility_result
        self._incident_dirs = incident_dirs_result
        self._incident_areas = incident_areas_result

    @classmethod
    def create_from_gaussians(cls, gaussians_list, dataset):
        assert len(gaussians_list) > 0
        sh_degree = max(g.max_sh_degree for g in gaussians_list)
        gaussians = GaussianModel(sh_degree=sh_degree,
                                  render_type=gaussians_list[0].render_type)
        attribute_names = gaussians.attribute_names
        for attribute_name in attribute_names:
            setattr(gaussians, "_" + attribute_name,
                    nn.Parameter(torch.cat([getattr(g, "_" + attribute_name).data for g in gaussians_list],
                                           dim=0).requires_grad_(True)))

        return gaussians

    def create_from_ckpt(self, checkpoint_path, restore_optimizer=False):
        (model_args, first_iter) = torch.load(checkpoint_path)

        (self.active_sh_degree,
         self._xyz,
         self._normal,
         self._shs_dc,
         self._shs_rest,
         self._scaling,
         self._rotation,
         self._opacity,
         self.max_radii2D,
         weights_accum,
         xyz_gradient_accum,
         normal_gradient_accum,
         denom,
         opt_dict,
         self.spatial_lr_scale) = model_args[:15]

        self.weights_accum = weights_accum
        self.normal_gradient_accum = normal_gradient_accum
        self.denom = denom

        if self.use_pbr:
            if len(model_args) > 15:
                (self._base_color,
                 self._roughness,
                 self._incidents_dc,
                 self._incidents_rest,
                 self._visibility_dc,
                 self._visibility_rest) = model_args[15:]
            else:
                self._base_color = nn.Parameter(torch.zeros_like(self._xyz).requires_grad_(True))
                roughness = torch.zeros_like(self._xyz[..., :1])
                # roughness = self.inverse_roughness_activation(torch.full((self._xyz.shape[0], 1), 0.9, dtype=torch.float, device="cuda"))
                self._roughness = nn.Parameter(roughness.requires_grad_(True))
                incidents = torch.zeros((self._xyz.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()

                self._incidents_dc = nn.Parameter(
                    incidents[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
                self._incidents_rest = nn.Parameter(
                    incidents[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))

                visibility = torch.zeros((self._xyz.shape[0], 1, 4 ** 2)).float().cuda()
                self._visibility_dc = nn.Parameter(
                    visibility[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
                self._visibility_rest = nn.Parameter(
                    visibility[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))

        if restore_optimizer:
            # TODO automatically match the opt_dict
            try:
                self.optimizer.load_state_dict(opt_dict)
            except:
                print("Not loading optimizer state_dict!")

        return first_iter

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_normal = torch.tensor(np.asarray(pcd.normals)).float().cuda()
        fused_color = torch.tensor(np.asarray(pcd.colors)).float().cuda()
        shs = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        shs[:, :3, 0] = RGB2SH(fused_color)
        shs[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._normal = nn.Parameter(fused_normal.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._shs_dc = nn.Parameter(shs[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._shs_rest = nn.Parameter(shs[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        if self.use_pbr:
            base_color = torch.zeros_like(fused_point_cloud)
            roughness = torch.zeros((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")

            self._base_color = nn.Parameter(base_color.requires_grad_(True))
            self._roughness = nn.Parameter(roughness.requires_grad_(True))

            incidents = torch.zeros((self._xyz.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
            self._incidents_dc = nn.Parameter(incidents[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
            self._incidents_rest = nn.Parameter(incidents[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))

            visibility = torch.zeros((self._xyz.shape[0], 1, 4 ** 2)).float().cuda()
            self._visibility_dc = nn.Parameter(visibility[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
            self._visibility_rest = nn.Parameter(visibility[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))

    def training_setup(self, training_args: OptimizationParams):
        self.percent_dense = training_args.percent_dense
        self.weights_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.normal_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._normal], 'lr': training_args.normal_lr, "name": "normal"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._shs_dc], 'lr': training_args.sh_lr, "name": "f_dc"},
            {'params': [self._shs_rest], 'lr': training_args.sh_lr / 20.0, "name": "f_rest"}
        ]

        if self.use_pbr:
            if training_args.light_rest_lr < 0:
                training_args.light_rest_lr = training_args.light_lr / 20.0
            if training_args.visibility_rest_lr < 0:
                training_args.visibility_rest_lr = training_args.visibility_lr / 20.0

            l.extend([
                {'params': [self._base_color], 'lr': training_args.base_color_lr, "name": "base_color"},
                {'params': [self._roughness], 'lr': training_args.roughness_lr, "name": "roughness"},
                {'params': [self._incidents_dc], 'lr': training_args.light_lr, "name": "incidents_dc"},
                {'params': [self._incidents_rest], 'lr': training_args.light_rest_lr, "name": "incidents_rest"},
                {'params': [self._visibility_dc], 'lr': training_args.visibility_lr, "name": "visibility_dc"},
                {'params': [self._visibility_rest], 'lr': training_args.visibility_rest_lr, "name": "visibility_rest"},
            ])
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final * self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def step(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

    def update_learning_rate(self, iteration):
        """ Learning rate scheduling per step """
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._shs_dc.shape[1] * self._shs_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._shs_rest.shape[1] * self._shs_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        if self.use_pbr:
            for i in range(self._base_color.shape[1]):
                l.append('base_color_{}'.format(i))
            l.append('roughness')
            for i in range(self._incidents_dc.shape[1] * self._incidents_dc.shape[2]):
                l.append('incidents_dc_{}'.format(i))
            for i in range(self._incidents_rest.shape[1] * self._incidents_rest.shape[2]):
                l.append('incidents_rest_{}'.format(i))
            for i in range(self._visibility_dc.shape[1] * self._visibility_dc.shape[2]):
                l.append('visibility_dc_{}'.format(i))
            for i in range(self._visibility_rest.shape[1] * self._visibility_rest.shape[2]):
                l.append('visibility_rest_{}'.format(i))

        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normal = self._normal.detach().cpu().numpy()
        sh_dc = self._shs_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        sh_rest = self._shs_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        attributes_list = [xyz, normal, sh_dc, sh_rest, opacities, scale, rotation]
        if self.use_pbr:
            attributes_list.extend([
                self._base_color.detach().cpu().numpy(),
                self._roughness.detach().cpu().numpy(),
                self._incidents_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy(),
                self._incidents_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy(),
                self._visibility_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy(),
                self._visibility_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy(),
            ])

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(attributes_list, axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        normal = np.stack((np.asarray(plydata.elements[0]["nx"]),
                           np.asarray(plydata.elements[0]["ny"]),
                           np.asarray(plydata.elements[0]["nz"])), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        shs_dc = np.zeros((xyz.shape[0], 3, 1))
        shs_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        shs_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        shs_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        shs_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            shs_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        shs_extra = shs_extra.reshape((shs_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._normal = nn.Parameter(torch.tensor(normal, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._shs_dc = nn.Parameter(torch.tensor(
            shs_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._shs_rest = nn.Parameter(torch.tensor(
            shs_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

        if self.use_pbr:
            base_color_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("base_color")]
            base_color_names = sorted(base_color_names, key=lambda x: int(x.split('_')[-1]))
            base_color = np.zeros((xyz.shape[0], len(base_color_names)))
            for idx, attr_name in enumerate(base_color_names):
                base_color[:, idx] = np.asarray(plydata.elements[0][attr_name])

            roughness = np.asarray(plydata.elements[0]["roughness"])[..., np.newaxis]

            self._base_color = nn.Parameter(
                torch.tensor(base_color, dtype=torch.float, device="cuda").requires_grad_(True))
            self._roughness = nn.Parameter(
                torch.tensor(roughness, dtype=torch.float, device="cuda").requires_grad_(True))

            incidents_dc = np.zeros((xyz.shape[0], 3, 1))
            incidents_dc[:, 0, 0] = np.asarray(plydata.elements[0]["incidents_dc_0"])
            incidents_dc[:, 1, 0] = np.asarray(plydata.elements[0]["incidents_dc_1"])
            incidents_dc[:, 2, 0] = np.asarray(plydata.elements[0]["incidents_dc_2"])
            extra_incidents_names = [p.name for p in plydata.elements[0].properties if
                                     p.name.startswith("incidents_rest_")]
            extra_incidents_names = sorted(extra_incidents_names, key=lambda x: int(x.split('_')[-1]))
            assert len(extra_incidents_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
            incidents_extra = np.zeros((xyz.shape[0], len(extra_incidents_names)))
            for idx, attr_name in enumerate(extra_incidents_names):
                incidents_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
            # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
            incidents_extra = incidents_extra.reshape((incidents_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))
            self._incidents_dc = nn.Parameter(torch.tensor(incidents_dc, dtype=torch.float, device="cuda").transpose(
                1, 2).contiguous().requires_grad_(True))
            self._incidents_rest = nn.Parameter(
                torch.tensor(incidents_extra, dtype=torch.float, device="cuda").transpose(
                    1, 2).contiguous().requires_grad_(True))

            visibility_dc = np.zeros((xyz.shape[0], 1, 1))
            visibility_dc[:, 0, 0] = np.asarray(plydata.elements[0]["visibility_dc_0"])
            extra_visibility_names = [p.name for p in plydata.elements[0].properties if
                                      p.name.startswith("visibility_rest_")]
            extra_visibility_names = sorted(extra_visibility_names, key=lambda x: int(x.split('_')[-1]))
            assert len(extra_visibility_names) == 4 ** 2 - 1
            visibility_extra = np.zeros((xyz.shape[0], len(extra_visibility_names)))
            for idx, attr_name in enumerate(extra_visibility_names):
                visibility_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
            # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
            visibility_extra = visibility_extra.reshape((visibility_extra.shape[0], 1, 4 ** 2 - 1))
            self._visibility_dc = nn.Parameter(torch.tensor(visibility_dc, dtype=torch.float, device="cuda").transpose(
                1, 2).contiguous().requires_grad_(True))
            self._visibility_rest = nn.Parameter(
                torch.tensor(visibility_extra, dtype=torch.float, device="cuda").transpose(
                    1, 2).contiguous().requires_grad_(True))

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._normal = optimizable_tensors["normal"]
        self._shs_dc = optimizable_tensors["f_dc"]
        self._shs_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.weights_accum = self.weights_accum[valid_points_mask]
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.normal_gradient_accum = self.normal_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

        if self.use_pbr:
            self._base_color = optimizable_tensors["base_color"]
            self._roughness = optimizable_tensors["roughness"]
            self._incidents_dc = optimizable_tensors["incidents_dc"]
            self._incidents_rest = optimizable_tensors["incidents_rest"]
            self._visibility_dc = optimizable_tensors["visibility_dc"]
            self._visibility_rest = optimizable_tensors["visibility_rest"]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]

                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_normal, new_shs_dc, new_shs_rest, new_opacities, new_scaling,
                              new_rotation, new_base_color=None, new_roughness=None,
                              new_incidents_dc=None, new_incidents_rest=None,
                              new_visibility_dc=None, new_visibility_rest=None):
        d = {"xyz": new_xyz,
             "normal": new_normal,
             "rotation": new_rotation,
             "scaling": new_scaling,
             "opacity": new_opacities,
             "f_dc": new_shs_dc,
             "f_rest": new_shs_rest}

        if self.use_pbr:
            d.update({
                "base_color": new_base_color,
                "roughness": new_roughness,
                "incidents_dc": new_incidents_dc,
                "incidents_rest": new_incidents_rest,
                "visibility_dc": new_visibility_dc,
                "visibility_rest": new_visibility_rest,
            })

        optimizable_tensors = self.cat_tensors_to_optimizer(d)

        self._xyz = optimizable_tensors["xyz"]
        self._normal = optimizable_tensors["normal"]
        self._rotation = optimizable_tensors["rotation"]
        self._scaling = optimizable_tensors["scaling"]
        self._opacity = optimizable_tensors["opacity"]
        self._shs_dc = optimizable_tensors["f_dc"]
        self._shs_rest = optimizable_tensors["f_rest"]

        self.weights_accum = torch.cat([self.weights_accum, torch.ones((new_xyz.shape[0], 1), device="cuda")], dim=0)
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.normal_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        if self.use_pbr:
            self._base_color = optimizable_tensors["base_color"]
            self._roughness = optimizable_tensors["roughness"]
            self._incidents_dc = optimizable_tensors["incidents_dc"]
            self._incidents_rest = optimizable_tensors["incidents_rest"]
            self._visibility_dc = optimizable_tensors["visibility_dc"]
            self._visibility_rest = optimizable_tensors["visibility_rest"]

    def densify_and_split(self, grads, grad_threshold, scene_extent, grads_normal, grad_normal_threshold, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        padded_grad_normal = torch.zeros((n_init_points), device="cuda")
        padded_grad_normal[:grads_normal.shape[0]] = grads_normal.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask_normal = torch.where(padded_grad_normal >= grad_normal_threshold, True, False)
        print("densify_and_split_normal:", selected_pts_mask_normal.sum().item(), "/", self.get_xyz.shape[0])

        selected_pts_mask = torch.logical_or(selected_pts_mask, selected_pts_mask_normal)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask, torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent)
        # print("densify_and_split:", selected_pts_mask.sum().item(), "/", self.get_xyz.shape[0])

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)  # (N, 3)
        means = torch.zeros((stds.size(0), 3), device="cuda")  # (N, 3)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)

        new_normal = self._normal[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_shs_dc = self._shs_dc[selected_pts_mask].repeat(N, 1, 1)
        new_shs_rest = self._shs_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        args = [new_xyz, new_normal, new_shs_dc, new_shs_rest, new_opacity, new_scaling, new_rotation]
        if self.use_pbr:
            new_base_color = self._base_color[selected_pts_mask].repeat(N, 1)
            new_roughness = self._roughness[selected_pts_mask].repeat(N, 1)
            new_incidents_dc = self._incidents_dc[selected_pts_mask].repeat(N, 1, 1)
            new_incidents_rest = self._incidents_rest[selected_pts_mask].repeat(N, 1, 1)
            new_visibility_dc = self._visibility_dc[selected_pts_mask].repeat(N, 1, 1)
            new_visibility_rest = self._visibility_rest[selected_pts_mask].repeat(N, 1, 1)
            args.extend([
                new_base_color,
                new_roughness,
                new_incidents_dc,
                new_incidents_rest,
                new_visibility_dc,
                new_visibility_rest,
            ])

        self.densification_postfix(*args)

        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent, grads_normal, grad_normal_threshold):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask_normal = torch.where(torch.norm(grads_normal, dim=-1) >= grad_normal_threshold, True, False)
        # print("densify_and_clone_normal:", selected_pts_mask_normal.sum().item(), "/", self.get_xyz.shape[0])
        selected_pts_mask = torch.logical_or(selected_pts_mask, selected_pts_mask_normal)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values <= self.percent_dense * scene_extent)
        # print("densify_and_clone:", selected_pts_mask.sum().item(), "/", self.get_xyz.shape[0])

        new_xyz = self._xyz[selected_pts_mask]
        new_normal = self._normal[selected_pts_mask]
        new_shs_dc = self._shs_dc[selected_pts_mask]
        new_shs_rest = self._shs_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        args = [new_xyz, new_normal, new_shs_dc, new_shs_rest, new_opacities,
                new_scaling, new_rotation]
        if self.use_pbr:
            new_base_color = self._base_color[selected_pts_mask]
            new_roughness = self._roughness[selected_pts_mask]
            new_incidents_dc = self._incidents_dc[selected_pts_mask]
            new_incidents_rest = self._incidents_rest[selected_pts_mask]
            new_visibility_dc = self._visibility_dc[selected_pts_mask]
            new_visibility_rest = self._visibility_rest[selected_pts_mask]

            args.extend([
                new_base_color,
                new_roughness,
                new_incidents_dc,
                new_incidents_rest,
                new_visibility_dc,
                new_visibility_rest,
            ])

        self.densification_postfix(*args)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, max_grad_normal, weights_threshold=1e-4):
        # print(self.xyz_gradient_accum.shape)
        grads = self.xyz_gradient_accum / self.denom
        grads_normal = self.normal_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        grads_normal[grads_normal.isnan()] = 0.0

        # if self._xyz.shape[0] < 1000000:
        self.densify_and_clone(grads, max_grad, extent, grads_normal, max_grad_normal)
        self.densify_and_split(grads, max_grad, extent, grads_normal, max_grad_normal)
        # self.densify_and_compact()

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        weight_mask = self.weights_accum[:, 0] < weights_threshold
        prune_mask = torch.logical_or(weight_mask, prune_mask)
        print("weights_accum:", weight_mask.sum().item(), "/", self.get_xyz.shape[0])
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        self.prune_points(prune_mask)
        self.weights_accum.data[:] = 0.0

        torch.cuda.empty_cache()

    def prune(self, min_opacity, extent, max_screen_size, weights_threshold=1e-4):
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        weight_mask = self.weights_accum[:, 0] < weights_threshold
        prune_mask = torch.logical_or(weight_mask, prune_mask)
        print("weights_accum:", weight_mask.sum().item(), "/", self.get_xyz.shape[0])
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        self.prune_points(prune_mask)
        self.weights_accum.data[:] = 0.0

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter, weights):
        self.weights_accum += weights
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1,
                                                             keepdim=True)
        self.normal_gradient_accum[update_filter] += torch.norm(
            self._normal.grad[update_filter], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
