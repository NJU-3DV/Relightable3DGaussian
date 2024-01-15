from typing import List, Union, Tuple, Dict

import numpy as np
import cv2
import random

import torch
# import tensorflow as tf


def recursive_apply(obj: Union[List, Dict], func):
    assert type(obj) == dict or type(obj) == list
    idx_iter = obj if type(obj) == dict else range(len(obj))
    for k in idx_iter:
        if type(obj[k]) == list or type(obj[k]) == dict:
            recursive_apply(obj[k], func)
        else:
            obj[k] = func(obj[k])


def center_image(img):
    """ normalize image input """
    img = img.astype(np.float32)
    var = np.var(img, axis=(0,1), keepdims=True)
    mean = np.mean(img, axis=(0,1), keepdims=True)
    return (img - mean) / (np.sqrt(var) + 0.00000001)


def image_net_center(img):
    stats = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    img = img.astype(np.float32)
    img /= 255.
    std = np.array([[stats['std'][::-1]]], dtype=np.float32)  # RGB to BGR
    mean = np.array([[stats['mean'][::-1]]], dtype=np.float32)
    return (img - mean) / (std + 0.00000001)


def image_net_center_inv(img):
    stats = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    std = np.array([[stats['std'][::-1]]], dtype=np.float32)  # RGB to BGR
    mean = np.array([[stats['mean'][::-1]]], dtype=np.float32)
    return ((img * std + mean)*255).astype(np.uint8)


def mask_depth_image(depth_image, min_depth, max_depth):
    """ mask out-of-range pixel to zero """
    # print ('mask min max', min_depth, max_depth)
    ret, depth_image = cv2.threshold(depth_image, min_depth, 100000, cv2.THRESH_TOZERO)
    ret, depth_image = cv2.threshold(depth_image, max_depth, 100000, cv2.THRESH_TOZERO_INV)
    depth_image = np.expand_dims(depth_image, 2)
    return depth_image


def to_channel_first(images: List[np.ndarray]):
    return [np.transpose(img, [2, 0, 1]) for img in images]


def scale_camera(cam: Union[np.ndarray, torch.Tensor], scale: Union[Tuple, float]=1):
    """ resize input in order to produce sampled depth map """
    if type(scale) != tuple:
        scale = (scale, scale)
    if type(cam) == np.ndarray:
        new_cam = np.copy(cam)
        # focal:
        new_cam[1, 0, 0] = cam[1, 0, 0] * scale[0]
        new_cam[1, 1, 1] = cam[1, 1, 1] * scale[1]
        # principle point:
        new_cam[1, 0, 2] = cam[1, 0, 2] * scale[0]
        new_cam[1, 1, 2] = cam[1, 1, 2] * scale[1]
    elif type(cam) == torch.Tensor:
        new_cam = cam.clone()
        # focal:
        new_cam[..., 1, 0, 0] = cam[..., 1, 0, 0] * scale[0]
        new_cam[..., 1, 1, 1] = cam[..., 1, 1, 1] * scale[1]
        # principle point:
        new_cam[..., 1, 0, 2] = cam[..., 1, 0, 2] * scale[0]
        new_cam[..., 1, 1, 2] = cam[..., 1, 1, 2] * scale[1]
    # elif type(cam) == tf.Tensor:
    #     scale_tensor = np.ones((1, 2, 4, 4))
    #     scale_tensor[0, 1, 0, 0] = scale[0]
    #     scale_tensor[0, 1, 1, 1] = scale[1]
    #     scale_tensor[0, 1, 0, 2] = scale[0]
    #     scale_tensor[0, 1, 1, 2] = scale[1]
    #     new_cam = cam * scale_tensor
    else:
        raise TypeError
    return new_cam


def crop_camera(cam: Union[np.ndarray, torch.Tensor], start: Union[Tuple, float]=0):
    if type(start) != tuple:
        start = (start, start)
    if type(cam) == np.ndarray:
        new_cam = np.copy(cam)
        # principle point:
        new_cam[1, 0, 2] = cam[1, 0, 2] - start[0]
        new_cam[1, 1, 2] = cam[1, 1, 2] - start[1]
    elif type(cam) == torch.Tensor:
        new_cam = cam.clone()
        # principle point:
        new_cam[..., 1, 0, 2] = cam[..., 1, 0, 2] - start[0]
        new_cam[..., 1, 1, 2] = cam[..., 1, 1, 2] - start[1]
    else:
        raise TypeError
    return new_cam


def resize(sample: List[np.ndarray], width, height):
    ref, ref_cam, srcs, srcs_cam, gt, masks = sample[:6]
    h_o, w_o = ref.shape[:2]

    ref_resized, *srcs_resized, gt_resized = [
        cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR) if w_o != width or h_o != height else img
        for img in [ref] + srcs + [gt[..., 0]]
    ]
    masks_resized = [
        np.expand_dims(cv2.resize(img[..., 0], (width, height), interpolation=cv2.INTER_NEAREST) if w_o != width or h_o != height else img[..., 0], -1)
        for img in masks
    ]
    gt_resized = np.expand_dims(gt_resized, -1)
    ref_cam_scaled, *srcs_cam_scaled = [
        scale_camera(cam, (width/w_o, height/h_o)) if w_o != width or h_o != height else cam
        for cam in [ref_cam] + srcs_cam
    ]

    return ref_resized, ref_cam_scaled, srcs_resized, srcs_cam_scaled, gt_resized, masks_resized


def random_crop(sample: List[np.ndarray], width, height, seed: Union[Tuple, int]=None):
    ref, ref_cam, srcs, srcs_cam, gt, masks = sample[:6]
    h_o, w_o = ref.shape[:2]
    if seed is not None and type(seed) == int:
        seed = (seed, seed)
    if seed is not None: random.seed(seed[0])
    start_w = random.randint(0, w_o - width)
    if seed is not None: random.seed(seed[1])
    start_h = random.randint(0, h_o - height)
    finish_w = start_w + width
    finish_h = start_h + height
    ref_cropped, *srcs_cropped, gt_cropped = [
        arr[start_h:finish_h, start_w:finish_w]
        for arr in [ref] + srcs + [gt]
    ]
    masks_cropped = [
        arr[start_h:finish_h, start_w:finish_w]
        for arr in masks
    ]
    ref_cam_cropped, *srcs_cam_cropped = [
        crop_camera(cam, (start_w, start_h))
        for cam in [ref_cam] + srcs_cam
    ]
    return ref_cropped, ref_cam_cropped, srcs_cropped, srcs_cam_cropped, gt_cropped, masks_cropped


def center_crop(sample: List[np.ndarray], width, height):
    ref, ref_cam, srcs, srcs_cam, gt, masks = sample[:6]
    h_o, w_o = ref.shape[:2]
    start_w = (w_o - width)//2
    start_h = (h_o - height)//2
    finish_w = start_w + width
    finish_h = start_h + height
    ref_cropped, *srcs_cropped, gt_cropped = [
        arr[start_h:finish_h, start_w:finish_w] if w_o != width or h_o != height else arr
        for arr in [ref] + srcs + [gt]
    ]
    masks_cropped = [
        arr[start_h:finish_h, start_w:finish_w] if w_o != width or h_o != height else arr
        for arr in masks
    ]
    ref_cam_cropped, *srcs_cam_cropped = [
        crop_camera(cam, (start_w, start_h)) if w_o != width or h_o != height else cam
        for cam in [ref_cam] + srcs_cam
    ]
    return ref_cropped, ref_cam_cropped, srcs_cropped, srcs_cam_cropped, gt_cropped, masks_cropped


def random_brightness(img: np.ndarray, max_abs_change=50):
    dv = np.random.randint(-max_abs_change, max_abs_change)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = v.astype(np.int32)
    v = np.clip(v + dv, 0, 255).astype(np.uint8)
    hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def random_contrast(img: np.ndarray, strength_range=[0.3, 1.5]):
    lo, hi = strength_range
    strength = np.random.rand() * (hi - lo) + lo
    img = img.astype(np.int32)
    img = (img - 128) * strength + 128
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def motion_blur(img: np.ndarray, max_kernel_size=3):
        # Either vertial, hozirontal or diagonal blur
        mode = np.random.choice(['h', 'v', 'diag_down', 'diag_up'])
        ksize = np.random.randint(0, (max_kernel_size+1)/2)*2 + 1  # make sure is odd
        center = int((ksize-1)/2)
        kernel = np.zeros((ksize, ksize))
        if mode == 'h':
            kernel[center, :] = 1.
        elif mode == 'v':
            kernel[:, center] = 1.
        elif mode == 'diag_down':
            kernel = np.eye(ksize)
        elif mode == 'diag_up':
            kernel = np.flip(np.eye(ksize), 0)
        var = ksize * ksize / 16.
        grid = np.repeat(np.arange(ksize)[:, np.newaxis], ksize, axis=-1)
        gaussian = np.exp(-(np.square(grid-center)+np.square(grid.T-center))/(2.*var))
        kernel *= gaussian
        kernel /= np.sum(kernel)
        img = cv2.filter2D(img, -1, kernel)
        return img
