
import re
import pyexr
import numpy as np
import imageio as imageio
from utils.graphics_utils import srgb_to_rgb, rgb_to_srgb

def load_pfm(file: str):
    color = None
    width = None
    height = None
    scale = None
    endian = None
    with open(file, 'rb') as f:
        header = f.readline().rstrip()
        if header == b'PF':
            color = True
        elif header == b'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')
        dim_match = re.match(br'^(\d+)\s(\d+)\s$', f.readline())
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')
        scale = float(f.readline().rstrip())
        if scale < 0:  # little-endian
            endian = '<'
            scale = -scale
        else:
            endian = '>'  # big-endian
        data = np.fromfile(f, endian + 'f')
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = data[::-1, ...]  # cv2.flip(data, 0)

    return np.ascontiguousarray(data)

def load_img_rgb(path):
    
    if path.endswith(".exr"):
        exr_file = pyexr.open(path)
        img = exr_file.get()
        img[..., 0:3] = rgb_to_srgb(img[..., 0:3], clip=False)
    else:
        img = imageio.imread(path)
        img = img / 255
        # img[..., 0:3] = srgb_to_rgb(img[..., 0:3])
    return img

def load_mask_bool(mask_file):
    mask = imageio.imread(mask_file, mode='L')
    mask = mask.astype(np.float32)
    mask[mask > 0.5] = 1.0

    return mask

def load_depth(tiff_file):
    return imageio.imread(tiff_file, mode='L')

def save_render_orb(file_path_wo_ext, data):
    exr_file = file_path_wo_ext + ".exr"
    pyexr.write(exr_file, data)
    
    png_file = file_path_wo_ext + ".png"
    data = rgb_to_srgb(data) * 255
    imageio.imwrite(png_file, data.astype(np.uint8))

def save_depth_orb(file_path_wo_ext, data):
    data = data[..., 0]
    
    exr_file = file_path_wo_ext + ".exr"
    pyexr.write(exr_file, data)
    
    png_file = file_path_wo_ext + ".png"
    
    mask = data != 0
    data[mask] = (data[mask] - np.min(data[mask])) / (np.max(data[mask])- np.min(data[mask]))
    data = data * 255
    imageio.imwrite(png_file, data.astype(np.uint8))

def save_normal_orb(file_path_wo_ext, data):
    exr_file = file_path_wo_ext + ".exr"
    pyexr.write(exr_file, data)
    
    png_file = file_path_wo_ext + ".png"
    
    data = data * 0.5 + 0.5
    data = data * 255
    imageio.imwrite(png_file, data.astype(np.uint8))

def save_albedo_orb(file_path_wo_ext, data):
    exr_file = file_path_wo_ext + ".exr"
    pyexr.write(exr_file, data)
    
    png_file = file_path_wo_ext + ".png"
    data = np.clip(data, 0.0, 1.0) * 255
    imageio.imwrite(png_file, data.astype(np.uint8))

def save_roughness_orb(file_path_wo_ext, data):
    data = data[..., 0]

    exr_file = file_path_wo_ext + ".exr"
    pyexr.write(exr_file, data)
    
    png_file = file_path_wo_ext + ".png"
    data = np.clip(data, 0.0, 1.0) * 255
    imageio.imwrite(png_file, data.astype(np.uint8))
