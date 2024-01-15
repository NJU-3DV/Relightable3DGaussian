import argparse
import os
#os.environ['SCAN'] = '0'
import shutil
import sys
import json
import itertools
sys.path.append('.')
import importlib

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import matplotlib.pyplot as plt
# from apex import amp

# from core.model_cas import Model, Loss
from utils.preproc import to_channel_first, resize, random_crop, recursive_apply, image_net_center_inv, scale_camera
# from data.tnt_training import get_val_loader
from utils.io_utils import load_model, subplot_map, write_cam, write_pfm

'''
python test.py --data_root /home/gj/Project/3DG_material/datasets/nerf/lego/colmap --resize '800,800 --crop '800,800'
'''

parser = argparse.ArgumentParser()

parser.add_argument('--data_root', type=str, help='The root dir of the data.')
parser.add_argument('--dataset_name', type=str, default='general', help='The name of the dataset. Should be identical to the dataloader source file. e.g. blended refers to data/blended.py.')
parser.add_argument('--model_name', type=str, default='model_cas', help='The name of the model. Should be identical to the model source file. e.g. model_cas refers to core/model_cas.py.')

parser.add_argument('--num_src', type=int, default=7, help='The number of source views.')
parser.add_argument('--max_d', type=int, default=256, help='The standard max depth number.')
parser.add_argument('--interval_scale', type=float, default=1., help='The standard interval scale.')
parser.add_argument('--cas_depth_num', type=str, default='64,32,16', help='The depth number for each stage.')
parser.add_argument('--cas_interv_scale', type=str, default='4,2,1', help='The interval scale for each stage.')
parser.add_argument('--resize', type=str, default='1600,1200', help='The size of the preprocessed input resized from the original one.')
parser.add_argument('--crop', type=str, default='1600,1200', help='The size of the preprocessed input cropped from the resized one.')

parser.add_argument('--mode', type=str, default='soft', choices=['soft', 'hard', 'uwta', 'maxpool', 'average'], help='The fusion strategy.')
parser.add_argument('--occ_guide', action='store_true', default=False, help='Deprecated')

parser.add_argument('--load_path', type=str, default="./pretrained_model/vis", help='The dir of the folder containing the pretrained checkpoints.')
parser.add_argument('--load_step', type=int, default=-1, help='The step to load. -1 for the latest one.')

parser.add_argument('--show_result', action='store_true', default=False, help='Set to show the results.')
parser.add_argument('--write_result', action='store_false', default=True, help='Set to save the results.')
parser.add_argument('--result_dir', default=None, type=str, help='The dir to save the results.')

args = parser.parse_args()

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True

    [resize_width, resize_height], [crop_width, crop_height] = [[int(v) for v in arg_str.split(',')] for arg_str in [args.resize, args.crop]]
    cas_depth_num = [int(v) for v in args.cas_depth_num.split(',')]
    cas_interv_scale = [float(v) for v in args.cas_interv_scale.split(',')]

    Model = importlib.import_module(f'core.{args.model_name}').Model
    Loss = importlib.import_module(f'core.{args.model_name}').Loss
    get_val_loader = importlib.import_module(f'data.{args.dataset_name}').get_val_loader

    dataset, loader = get_val_loader(
        args.data_root, args.num_src,
        {
            'interval_scale': args.interval_scale,
            'max_d': args.max_d,
            'resize_width': resize_width,
            'resize_height': resize_height,
            'crop_width': crop_width,
            'crop_height': crop_height
        }
    )

    model = Model()
    model.cuda()
    # model = amp.initialize(model, opt_level='O0')
    model = nn.DataParallel(model)
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters() if p.requires_grad])))

    load_model(model, args.load_path, args.load_step)
    print(f'load {os.path.join(args.load_path, str(args.load_step))}')
    model.eval()

    pbar = tqdm.tqdm(enumerate(loader), dynamic_ncols=True, total=len(loader))
    # pbar = itertools.product(range(num_scan), range(num_ref), range(num_view))
    for i, sample in pbar:
        if sample.get('skip') is not None and np.any(sample['skip']): raise ValueError()

        ref, ref_cam, srcs, srcs_cam, gt, masks = [sample[attr] for attr in ['ref', 'ref_cam', 'srcs', 'srcs_cam', 'gt', 'masks']]
        sample_name = sample.pop("name")[0]
       
        recursive_apply(sample, lambda x: torch.from_numpy(x).float().cuda())
        ref_t, ref_cam_t, srcs_t, srcs_cam_t, gt_t, masks_t = [sample[attr] for attr in ['ref', 'ref_cam', 'srcs', 'srcs_cam', 'gt', 'masks']]

        with torch.no_grad():
            # est_depth, prob_map, pair_results = model([ref_t, ref_cam_t, srcs_t, srcs_cam_t], args.max_d, upsample=True, mem=True, mode=args.mode)  #MVS
            outputs, refined_depth, prob_maps = model(sample, cas_depth_num, cas_interv_scale, mode=args.mode)
            [[est_depth_1, pair_results_1], [est_depth_2, pair_results_2], [est_depth_3, pair_results]] = outputs
            # est_depth = model([ref_t, ref_cam_t, srcs_t, srcs_cam_t, gt_t], args.max_d)
        # est_depth, prob_map = [arr.clone().cpu().data.numpy() for arr in [refined_depth, prob_map]]
        est_depth, *prob_maps = [arr.clone().cpu().data.numpy() for arr in [refined_depth] + prob_maps]
        recursive_apply(pair_results, lambda x: x.clone().cpu().data.numpy())  #MVS

        pbar.set_description(f'{est_depth.shape}')

        if (i % 49 == 0 or True) and (args.show_result or args.write_result):
            if args.show_result:
                # plt_map = [
                #     [est_depth[0, 0], prob_maps[2][0, 0], None],
                #     [ref[0, 0], srcs[0, 0, 0], srcs[0, 1, 0]],
                #     [prob_maps[0][0, 0], pair_results[0][0][0, 0], pair_results[1][0][0, 0]],  #MVS
                #     [prob_maps[1][0, 0], pair_results[0][1][0][0, 0], pair_results[1][1][0][0, 0]],
                # ]
                plt_map = [
                    [est_depth[0, 0], est_depth_1.cpu().data.numpy()[0, 0], est_depth_2.cpu().data.numpy()[0, 0]],
                    [ref[0, 0], srcs[0, 0, 0], srcs[0, 1, 0]],
                    [prob_maps[0][0, 0], prob_maps[1][0, 0], prob_maps[2][0, 0]]
                ]
                subplot_map(plt_map)
                plt.show()
            if args.write_result:
                ref_o = np.transpose(ref[0], [1, 2, 0])
                ref_o = image_net_center_inv(ref_o)
                ref_o = cv2.resize(ref_o, (ref_o.shape[1]//2, ref_o.shape[0]//2), interpolation=cv2.INTER_LINEAR)
                ref_cam_o = ref_cam[0]
                ref_cam_o = scale_camera(ref_cam_o, .5)
                est_depth_o = est_depth[0, 0]
                prob_maps_o = [prob_map[0, 0] for prob_map in prob_maps]

                if args.result_dir is None:
                    args.result_dir = f"{args.data_root}/vis_mvsnet"

                if not os.path.exists(args.result_dir):
                    os.mkdir(args.result_dir)
                
                cv2.imwrite(os.path.join(args.result_dir, f'{sample_name}.jpg'), ref_o)
                write_cam(os.path.join(args.result_dir, f'cam_{sample_name}_flow3.txt'), ref_cam_o)
                write_pfm(os.path.join(args.result_dir, f'{sample_name}_flow3.pfm'), est_depth_o)
                for stage_i, prob_map_o in enumerate(prob_maps_o):
                    write_pfm(os.path.join(args.result_dir, f'{sample_name}_flow{stage_i+1}_prob.pfm'), prob_map_o)

        # del pair_results, est_depth
