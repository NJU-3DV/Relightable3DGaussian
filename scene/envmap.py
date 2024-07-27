
import torch
import torch.nn.functional as F
import numpy as np
import nvdiffrast.torch as dr
import imageio
import pyexr
from utils.graphics_utils import srgb_to_rgb
imageio.plugins.freeimage.download()

class EnvLight(torch.nn.Module):
    def __init__(self, path=None, scale=1.0):
        super().__init__()
        self.device = "cuda"  # only supports cuda
        self.scale = scale  # scale of the hdr values
        self.to_opengl = torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=torch.float32, device="cuda")

        self.envmap = self.load(path, scale=self.scale, device=self.device)
        self.transform = None

    @staticmethod
    def load(envmap_path, scale, device):
        if not envmap_path.endswith(".exr"):
            image = srgb_to_rgb(imageio.imread(envmap_path)[:, :, :3] / 255)
        else:
            # load latlong env map from file
            image = pyexr.open(envmap_path).get()[:, :, :3]

        image = image * scale

        env_map_torch = torch.tensor(image, dtype=torch.float32, device=device, requires_grad=False)

        return env_map_torch

    def direct_light(self, dirs, transform=None):
        shape = dirs.shape
        dirs = dirs.reshape(-1, 3)

        if transform is not None:
            dirs = dirs @ transform.T
        elif self.transform is not None:
            dirs = dirs @ self.transform.T

        envir_map =  self.envmap.permute(2, 0, 1).unsqueeze(0) # [1, 3, H, W]
        phi = torch.arccos(dirs[:, 2]).reshape(-1) - 1e-6
        theta = torch.atan2(dirs[:, 1], dirs[:, 0]).reshape(-1)
        # normalize to [-1, 1]
        query_y = (phi / np.pi) * 2 - 1
        query_x = - theta / np.pi
        grid = torch.stack((query_x, query_y)).permute(1, 0).unsqueeze(0).unsqueeze(0)
        light_rgbs = F.grid_sample(envir_map, grid, align_corners=True).squeeze().permute(1, 0).reshape(-1, 3)
    
        return light_rgbs.reshape(*shape)
