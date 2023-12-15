import cv2
import torch
import numpy as np
import nvdiffrast.torch as dr


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
        # load latlong env map from file
        image = cv2.imread(envmap_path, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if image.dtype != np.float32:
            image = image.astype(np.float32) / 255

        image = image * scale
        env_map_torch = torch.tensor(image, dtype=torch.float32, device=device, requires_grad=False)

        return env_map_torch

    def direct_light(self, dirs, transform=None):
        """infer light from env_map directly"""
        shape = dirs.shape
        dirs = dirs.reshape(-1, 3)

        if transform is not None:
            dirs = dirs @ transform.T
        elif self.transform is not None:
            dirs = dirs @ self.transform.T

        v = dirs @ self.to_opengl.T
        tu = torch.atan2(v[..., 0:1], -v[..., 2:3]) / (2 * np.pi) + 0.5
        tv = torch.acos(torch.clamp(v[..., 1:2], min=-1, max=1)) / np.pi
        texcoord = torch.cat((tu, tv), dim=-1)

        light = dr.texture(self.envmap[None, ...], texcoord[None, None, ...], filter_mode='linear')[0, 0]

        return light.reshape(*shape)
