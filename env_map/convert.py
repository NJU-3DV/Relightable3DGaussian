import numpy as np
import torch
import pyexr
import cv2

def rgb_to_srgb(img, clip=True):
    # hdr img
    if isinstance(img, np.ndarray):
        assert len(img.shape) == 3, img.shape
        assert img.shape[2] == 3, img.shape
        img = np.where(img > 0.0031308, np.power(np.maximum(img, 0.0031308), 1.0 / 2.4) * 1.055 - 0.055, 12.92 * img)
        if clip:
            img = np.clip(img, 0.0, 1.0)
        return img
    elif isinstance(img, torch.Tensor):
        assert len(img.shape) == 3, img.shape
        assert img.shape[0] == 3, img.shape
        img = torch.where(img > 0.0031308, torch.pow(torch.max(img, torch.tensor(0.0031308)), 1.0 / 2.4) * 1.055 - 0.055, 12.92 * img)
        if clip:
            img = torch.clamp(img, 0.0, 1.0)
        return img
    else:
        raise TypeError("Unsupported input type. Supported types are numpy.ndarray and torch.Tensor.")

exr_path = "envmap3.exr"
img = rgb_to_srgb(pyexr.open(exr_path).get())
cv2.imwrite(exr_path.replace('.exr', '.png'), (img*255).astype(np.uint8)[:, :, [2, 1, 0]])