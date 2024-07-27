import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, getProjectionMatrixCenterShift


class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, fx, fy, cx, cy, image, image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device="cuda",
                 height=None, width=None, depth=None, normal=None, image_mask=None):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device")
            self.data_device = torch.device("cuda")

        if image is not None:
            self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
            self.image_width = self.original_image.shape[2]
            self.image_height = self.original_image.shape[1]
        else:
            self.image_width = width
            self.image_height = height

        if depth is not None:
            self.depth = depth
        else:
            self.depth = torch.zeros((1, self.image_height, self.image_width), dtype=torch.float32, device=data_device)

        if normal is not None:
            self.normal = normal
        else:
            self.normal = torch.zeros((3, self.image_height, self.image_width), dtype=torch.float32, device=data_device)

        if image_mask is not None:
            self.image_mask = image_mask
        else:
            self.image_mask = torch.ones_like(self.depth)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()

        if self.fx is None:
            self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx,
                                                         fovY=self.FoVy).transpose(0, 1).cuda()
        else:
            self.projection_matrix = getProjectionMatrixCenterShift(
                self.znear, self.zfar, cx, cy, fx, fy, self.image_width, self.image_height).transpose(0, 1).cuda()

        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.c2w = self.world_view_transform.transpose(0, 1).inverse()
        self.intrinsics = self.get_intrinsics()
        self.extrinsics = self.get_extrinsics()
        self.proj_matrix = self.get_proj_matrix()

    def get_world_directions(self):
        """not used, bug fixed, when the ppx is not in the center"""
        v, u = torch.meshgrid(torch.arange(self.image_height, device='cuda'),
                              torch.arange(self.image_width, device='cuda'), indexing="ij")
        focal_x = self.intrinsics[0, 0]
        focal_y = self.intrinsics[1, 1]
        directions = torch.stack([(u - self.intrinsics[0, 2]) / focal_x,
                                  (v - self.intrinsics[1, 2]) / focal_y,
                                  torch.ones_like(u)], dim=0)
        directions = F.normalize(directions, dim=0)
        directions = (self.c2w[:3, :3] @ directions.reshape(3, -1)).reshape(3, self.image_height, self.image_width)

        return directions

    def get_primary_axis(self):
        p_axis = torch.zeros([3], dtype=torch.float32).cuda()
        p_axis[2] = 1
        p_axis_world = self.c2w[:3, :3] @ p_axis
        return p_axis_world

    def get_intrinsics(self):
        if self.fx is None:
            focal_x = self.image_width / (2 * np.tan(self.FoVx * 0.5))
            focal_y = self.image_height / (2 * np.tan(self.FoVy * 0.5))

            return torch.tensor([[focal_x, 0, self.image_width / 2],
                                 [0, focal_y, self.image_height / 2],
                                 [0, 0, 1]], device='cuda', dtype=torch.float32)
        else:
            return torch.tensor([[self.fx, 0, self.cx],
                                 [0, self.fy, self.cy],
                                 [0, 0, 1]], device='cuda', dtype=torch.float32)

    def get_extrinsics(self):
        Rt = np.eye(4, dtype=np.float32)
        Rt[:3, :3] = self.R.transpose()
        Rt[:3, 3] = self.T

        return torch.from_numpy(Rt).float().cuda()

    def get_proj_matrix(self):
        eK_mat = torch.eye(4, dtype=self.intrinsics.dtype, device=self.intrinsics.device)
        eK_mat[0:3, 0:3] = self.intrinsics
        return torch.bmm(eK_mat.unsqueeze(0), self.extrinsics.unsqueeze(0)).squeeze(0)

    def get_rotation(self):
        return torch.from_numpy(self.R.T).float().cuda()

    @staticmethod
    def create_for_gui():
        return Camera(0, np.eye(3), np.zeros(3), 50, 50, None, None, "gui", "gui")
