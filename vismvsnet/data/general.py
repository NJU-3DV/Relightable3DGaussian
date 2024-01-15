import os

import cv2
import numpy as np
import torch.utils.data as data

from utils.io_utils import load_cam, load_pfm, load_pair, cam_adjust_max_d
from utils.preproc import to_channel_first, resize, center_crop, image_net_center as center_image
from data.data_utils import dict_collate


class MyDataset(data.Dataset):

    def __init__(self, root, num_src, read, transforms):
        self.root = root
        self.num_src = num_src
        self.read = read
        self.transforms = transforms
        self.pair = load_pair(os.path.join(self.root, f'pair.txt'))

        self.names = [os.path.splitext(d)[0] for d in os.listdir(f"{self.root}/images")]
        self.names = sorted(self.names)

    def __len__(self):
        return len(self.pair)

    def __getitem__(self, i):
        ref_idx = i
        src_idxs = self.pair[ref_idx][:self.num_src]

        ref, *srcs = [os.path.join(self.root, f'images/{self.names[idx]}.png') for idx in [ref_idx] + src_idxs]
        ref_cam, *srcs_cam = [os.path.join(self.root, f'cams/{self.names[idx]}_cam.txt') for idx in [ref_idx] + src_idxs]
        skip = 0

        sample = self.read({'ref':ref, 'ref_cam':ref_cam, 'srcs':srcs, 'srcs_cam':srcs_cam, 'skip':skip})
        for t in self.transforms:
            sample = t(sample)
        return sample


def read(filenames, max_d, interval_scale):
    ref_name, ref_cam_name, srcs_name, srcs_cam_name, skip = [filenames[attr] for attr in ['ref', 'ref_cam', 'srcs', 'srcs_cam', 'skip']]
    ref, *srcs = [cv2.imread(fn) for fn in [ref_name] + srcs_name]
    ref_cam, *srcs_cam = [load_cam(fn, max_d, interval_scale) for fn in [ref_cam_name] + srcs_cam_name]
    gt = np.zeros((ref.shape[0], ref.shape[1], 1))
    masks = [np.zeros((ref.shape[0], ref.shape[1], 1)) for i in range(len(srcs))]
    return {
        'ref': ref,
        'ref_cam': ref_cam,
        'srcs': srcs,
        'srcs_cam': srcs_cam,
        'gt': gt,
        'masks': masks,
        'skip': skip,
        'name': os.path.splitext(os.path.basename(ref_name))[0]
    }


def val_preproc(sample, preproc_args):
    ref, ref_cam, srcs, srcs_cam, gt, masks, skip = [sample[attr] for attr in ['ref', 'ref_cam', 'srcs', 'srcs_cam', 'gt', 'masks', 'skip']]

    ref, *srcs = [center_image(img) for img in [ref] + srcs]
    ref, ref_cam, srcs, srcs_cam, gt, masks = resize([ref, ref_cam, srcs, srcs_cam, gt, masks], preproc_args['resize_width'], preproc_args['resize_height'])
    ref, ref_cam, srcs, srcs_cam, gt, masks = center_crop([ref, ref_cam, srcs, srcs_cam, gt, masks], preproc_args['crop_width'], preproc_args['crop_height'])
    ref, *srcs, gt = to_channel_first([ref] + srcs + [gt])
    masks = to_channel_first(masks)

    srcs, srcs_cam, masks = [np.stack(arr_list, axis=0) for arr_list in [srcs, srcs_cam, masks]]

    return {
        'ref': ref,  # 3hw
        'ref_cam': ref_cam,  # 244
        'srcs': srcs,  # v3hw
        'srcs_cam': srcs_cam,  # v244
        'gt': gt,  # 1hw
        'masks': masks,  # v1hw
        'skip': skip,  # scalar
        'name': sample["name"]
    }


def get_val_loader(root, num_src, preproc_args):
    dataset = MyDataset(
        root, num_src,
        read=lambda filenames: read(filenames, preproc_args['max_d'], preproc_args['interval_scale']),
        transforms=[lambda sample: val_preproc(sample, preproc_args)]
    )
    loader = data.DataLoader(dataset, 1, collate_fn=dict_collate, shuffle=False)
    return dataset, loader
