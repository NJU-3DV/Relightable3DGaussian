import re
import sys
import json
import os

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def load_pair(file: str):
    with open(file) as f:
        lines = f.readlines()
    n_cam = int(lines[0])
    pairs = []
    for i in range(1, 1+2*n_cam, 2):
        pair = []
        pair_str = lines[i+1].strip().split(' ')
        n_pair = int(pair_str[0])
        for j in range(1, 1+2*n_pair, 2):
            pair.append(int(pair_str[j]))
        pairs.append(pair)
    return pairs


def load_cam(file: str, max_d, interval_scale=1):
    """ read camera txt file """
    cam = np.zeros((2, 4, 4))
    with open(file) as f:
        words = f.read().split()
    # read extrinsic
    for i in range(0, 4):
        for j in range(0, 4):
            extrinsic_index = 4 * i + j + 1
            cam[0][i][j] = words[extrinsic_index]

    # read intrinsic
    for i in range(0, 3):
        for j in range(0, 3):
            intrinsic_index = 3 * i + j + 18
            cam[1][i][j] = words[intrinsic_index]

    if len(words) == 29:
        cam[1][3][0] = words[27]
        cam[1][3][1] = float(words[28]) * interval_scale
        cam[1][3][2] = max_d
        cam[1][3][3] = cam[1][3][0] + cam[1][3][1] * (cam[1][3][2] - 1)
    elif len(words) == 30:
        cam[1][3][0] = words[27]
        cam[1][3][1] = float(words[28]) * interval_scale
        cam[1][3][2] = words[29]
        cam[1][3][3] = cam[1][3][0] + cam[1][3][1] * (cam[1][3][2] - 1)
    elif len(words) == 31:
        cam[1][3][0] = words[27]
        cam[1][3][1] = float(words[28]) * interval_scale
        cam[1][3][2] = words[29]
        cam[1][3][3] = words[30]
    else:
        cam[1][3][0] = 0
        cam[1][3][1] = 0
        cam[1][3][2] = 0
        cam[1][3][3] = 0

    return cam


def cam_adjust_max_d(cam, max_d):
    cam = cam.copy()
    interval_scale = cam[1][3][2] / max_d
    cam[1][3][1] *= interval_scale
    cam[1][3][2] = max_d
    return cam


def write_cam(file: str, cam):
    content = f"""
extrinsic
{cam[0][0][0]} {cam[0][0][1]} {cam[0][0][2]} {cam[0][0][3]}
{cam[0][1][0]} {cam[0][1][1]} {cam[0][1][2]} {cam[0][1][3]}
{cam[0][2][0]} {cam[0][2][1]} {cam[0][2][2]} {cam[0][2][3]}
{cam[0][3][0]} {cam[0][3][1]} {cam[0][3][2]} {cam[0][3][3]}

intrinsic
{cam[1][0][0]} {cam[1][0][1]} {cam[1][0][2]}
{cam[1][1][0]} {cam[1][1][1]} {cam[1][1][2]}
{cam[1][2][0]} {cam[1][2][1]} {cam[1][2][2]}

{cam[1][3][0]} {cam[1][3][1]} {cam[1][3][2]} {cam[1][3][3]}
"""
    with open(file, 'w') as f:
        f.write(content.strip())


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
    return data


def write_pfm(file: str, image, scale=1):
    with open(file, 'wb') as f:
        color = None

        if image.dtype.name != 'float32':
            raise Exception('Image dtype must be float32.')

        image = np.flipud(image)

        if len(image.shape) == 3 and image.shape[2] == 3: # color image
            color = True
        elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
            color = False
        else:
            raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

        f.write(b'PF\n' if color else b'Pf\n')
        f.write(b'%d %d\n' % (image.shape[1], image.shape[0]))

        endian = image.dtype.byteorder

        if endian == '<' or endian == '=' and sys.byteorder == 'little':
            scale = -scale

        f.write(b'%f\n' % scale)

        image.tofile(f)


def save_model(obj, save_dir: str, job_name: str, global_step: int, max_keep: int):
    os.makedirs(os.path.join(save_dir, job_name), exist_ok=True)
    record_file = os.path.join(save_dir, job_name, 'record')
    cktp_file = os.path.join(save_dir, job_name, f'{global_step}.tar')
    if not os.path.exists(record_file):
        with open(record_file, 'w+') as f:
            json.dump([], f)
    with open(record_file, 'r') as f:
        record = json.load(f)
    record.append(global_step)
    if len(record) > max_keep:
        old = record[0]
        record = record[1:]
        os.remove(os.path.join(save_dir, job_name, f'{old}.tar'))
    torch.save(obj, cktp_file)
    with open(record_file, 'w') as f:
        json.dump(record, f)


def load_model(model: nn.Module, load_path: str, load_step: int):
    if load_step is None:
        model.load_state_dict(torch.load(load_path)['state_dict'])
        return 0
    else:
        if load_step == -1:
            record_file = os.path.join(load_path, 'record')
            with open(record_file, 'r') as f:
                record = json.load(f)
            if len(record) == 0:
                raise Exception('no latest model.')
            load_step = record[-1]
        cktp_file = os.path.join(load_path, f'{load_step}.tar')
        model.load_state_dict(torch.load(cktp_file)['state_dict'], strict=True)
        return torch.load(cktp_file)['global_step']


def subplot_map(plt_map):
    h = len(plt_map)
    w = len(plt_map[0])
    for i in range(h):
        for j in range(w):
            if plt_map[i][j] is not None:
                plt.subplot(h, w, i*w+j+1)
                plt.imshow(plt_map[i][j])
