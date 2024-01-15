import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import os
import cv2

from core.homography import get_pixel_grids
from core.nn_utils import bin_op_reduce
from utils.io_utils import load_cam, load_pfm
from utils.preproc import recursive_apply

import matplotlib.pyplot as plt

def load_pair(file: str, min_views: int=None):
    with open(file) as f:
        lines = f.readlines()
    n_cam = int(lines[0])
    pairs = {}
    img_ids = []
    for i in range(1, 1+2*n_cam, 2):
        pair = []
        score = []
        img_id = lines[i].strip()
        pair_str = lines[i+1].strip().split(' ')
        n_pair = int(pair_str[0])
        if min_views is not None and n_pair < min_views: continue
        for j in range(1, 1+2*n_pair, 2):
            pair.append(pair_str[j])
            score.append(float(pair_str[j+1]))
        img_ids.append(img_id)
        pairs[img_id] = {'id': img_id, 'index': i//2, 'pair': pair, 'score': score}
    pairs['id_list'] = img_ids
    return pairs


def idx_img2cam(idx_img_homo, depth, cam):  # nhw31, n1hw -> nhw41
    idx_cam = cam[:,1:2,:3,:3].unsqueeze(1).inverse() @ idx_img_homo  # nhw31
    idx_cam = idx_cam / (idx_cam[...,-1:,:]+1e-9) * depth.permute(0,2,3,1).unsqueeze(4)  # nhw31
    idx_cam_homo = torch.cat([idx_cam, torch.ones_like(idx_cam[...,-1:,:])], dim=-2)  # nhw41
    # FIXME: out-of-range is 0,0,0,1, will have valid coordinate in world
    return idx_cam_homo


def idx_cam2world(idx_cam_homo, cam):  # nhw41 -> nhw41
    idx_world_homo =  cam[:,0:1,...].unsqueeze(1).inverse() @ idx_cam_homo  # nhw41
    idx_world_homo = idx_world_homo / (idx_world_homo[...,-1:,:]+1e-9)  # nhw41
    return idx_world_homo


def idx_world2cam(idx_world_homo, cam):  # nhw41 -> nhw41
    idx_cam_homo = cam[:,0:1,...].unsqueeze(1) @ idx_world_homo  # nhw41
    idx_cam_homo = idx_cam_homo / (idx_cam_homo[...,-1:,:]+1e-9)   # nhw41
    return idx_cam_homo


def idx_cam2img(idx_cam_homo, cam):  # nhw41 -> nhw31
    idx_cam = idx_cam_homo[...,:3,:] / (idx_cam_homo[...,3:4,:]+1e-9)  # nhw31
    idx_img_homo = cam[:,1:2,:3,:3].unsqueeze(1) @ idx_cam  # nhw31
    idx_img_homo = idx_img_homo / (idx_img_homo[...,-1:,:]+1e-9)
    return idx_img_homo


def project_img(src_img, dst_depth, src_cam, dst_cam, height=None, width=None):  # nchw, n1hw -> nchw, n1hw
    if height is None: height = src_img.size()[-2]
    if width is None: width = src_img.size()[-1]
    dst_idx_img_homo = get_pixel_grids(height, width).unsqueeze(0)  # nhw31
    dst_idx_cam_homo = idx_img2cam(dst_idx_img_homo, dst_depth, dst_cam)  # nhw41
    dst_idx_world_homo = idx_cam2world(dst_idx_cam_homo, dst_cam)  # nhw41
    dst2src_idx_cam_homo = idx_world2cam(dst_idx_world_homo, src_cam)  # nhw41
    dst2src_idx_img_homo = idx_cam2img(dst2src_idx_cam_homo, src_cam)  # nhw31
    warp_coord = dst2src_idx_img_homo[...,:2,0]  # nhw2
    warp_coord[..., 0] /= width
    warp_coord[..., 1] /= height
    warp_coord = (warp_coord*2-1).clamp(-1.1, 1.1)  # nhw2
    in_range = bin_op_reduce([-1<=warp_coord[...,0], warp_coord[...,0]<=1, -1<=warp_coord[...,1], warp_coord[...,1]<=1], torch.min).to(src_img.dtype).unsqueeze(1)  # n1hw
    warped_img = F.grid_sample(src_img, warp_coord, mode='bilinear', padding_mode='zeros', align_corners=False)
    return warped_img, in_range


def prob_filter(ref_probs, pthresh, greater=True):  # n3hw -> n1hw
    cmpr = lambda x, y: x > y if greater else lambda x, y: x < y
    masks = cmpr(ref_probs, torch.Tensor(pthresh).to(ref_probs.dtype).to(ref_probs.device).view(1,3,1,1)).to(ref_probs.dtype)
    mask = (masks.sum(dim=1, keepdim=True) >= (len(pthresh)-0.1))
    return mask


def get_reproj(ref_depth, srcs_depth, ref_cam, srcs_cam):  # n1hw, nv1hw -> nv3hw, nv1hw
    n, v, _, h, w = srcs_depth.size()
    srcs_depth_f = srcs_depth.view(n*v, 1, h, w)
    srcs_valid_f = (srcs_depth_f > 1e-9).to(srcs_depth_f.dtype)
    srcs_cam_f = srcs_cam.view(n*v, 2, 4, 4)
    ref_depth_r = ref_depth.unsqueeze(1).repeat(1,v,1,1,1).view(n*v, 1, h, w)
    ref_cam_r = ref_cam.unsqueeze(1).repeat(1,v,1,1,1).view(n*v, 2, 4, 4)
    idx_img = get_pixel_grids(h, w).unsqueeze(0)  # 1hw31

    srcs_idx_cam = idx_img2cam(idx_img, srcs_depth_f, srcs_cam_f)  # Nhw41
    srcs_idx_world = idx_cam2world(srcs_idx_cam, srcs_cam_f)  # Nhw41
    srcs2ref_idx_cam = idx_world2cam(srcs_idx_world, ref_cam_r)  # Nhw41
    srcs2ref_idx_img = idx_cam2img(srcs2ref_idx_cam, ref_cam_r)  # Nhw31
    srcs2ref_xydv = torch.cat([srcs2ref_idx_img[...,:2,0], srcs2ref_idx_cam[...,2:3,0], srcs_valid_f.permute(0,2,3,1)], dim=-1).permute(0,3,1,2)  # N4hw

    reproj_xydv_f, in_range_f= project_img(srcs2ref_xydv, ref_depth_r, srcs_cam_f, ref_cam_r)  # N4hw, N1hw
    reproj_xyd = reproj_xydv_f.view(n,v,4,h,w)[:,:,:3]
    in_range = (in_range_f * reproj_xydv_f[:,3:]).view(n,v,1,h,w)
    return reproj_xyd, in_range


def vis_filter(ref_depth, reproj_xyd, in_range, img_dist_thresh, depth_thresh, vthresh):
    n, v, _, h, w = reproj_xyd.size()
    xy = get_pixel_grids(h, w).permute(3,2,0,1).unsqueeze(1)[:,:,:2]  # 112hw
    dist_masks = (reproj_xyd[:,:,:2,:,:] - xy).norm(dim=2, keepdim=True) < img_dist_thresh  # nv1hw
    depth_masks = (ref_depth.unsqueeze(1) - reproj_xyd[:,:,2:,:,:]).abs() < (torch.max(ref_depth.unsqueeze(1), reproj_xyd[:,:,2:,:,:])*depth_thresh)  # nv1hw
    masks = in_range * dist_masks.to(ref_depth.dtype) * depth_masks.to(ref_depth.dtype)  # nv1hw
    mask = masks.sum(dim=1) >= (vthresh-1.1)  # n1hw
    return masks, mask


def med_fusion(ref_depth, reproj_xyd, masks, mask):
    all_d = torch.cat([reproj_xyd[:,:,2:,:,:]*masks, ref_depth.unsqueeze(1)], dim=1)  # n(v+1)1hw
    valid_num = masks.sum(dim=1, keepdim=True) + 1  # n11hw
    gather_idx = (valid_num // 2).long()  # n11hw
    med = all_d.sort(dim=1, descending=True)[0].gather(dim=1, index=gather_idx).squeeze(1)  # n1hw
    return med * mask

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str)
    parser.add_argument('--pair', type=str)
    parser.add_argument('--view', type=int, default=5)
    parser.add_argument('--vthresh', type=int, default=2)
    parser.add_argument('--pthresh', type=str, default='.6,.6,.6')
    parser.add_argument('--cam_scale', type=float, default=1)
    parser.add_argument('--out_dir', type=str)
    args = parser.parse_args()

    pthresh = [float(v) for v in args.pthresh.split(',')]
    num_src = args.view
    pair = load_pair(args.pair, min_views=num_src)
    n_views = len(pair['id_list'])

    views = {}

    file_names = [os.path.splitext(d)[0] for d in os.listdir(args.data) if os.path.splitext(d)[-1]==".jpg"]
    file_names = sorted(file_names)
    for i, id in tqdm(enumerate(pair['id_list']), 'load data', n_views):
        image = cv2.imread(f'{args.data}/{file_names[i]}.jpg').transpose(2,0,1)[::-1]
        cam = load_cam(f'{args.data}/cam_{file_names[i]}_flow3.txt', 256, 1)
        depth = np.expand_dims(load_pfm(f'{args.data}/{file_names[i]}_flow3.pfm'), axis=0)
        probs = np.stack([load_pfm(f'{args.data}/{file_names[i]}_flow{k+1}_prob.pfm') for k in range(3)], axis=0)
        
        views[id] = {
            'image': image,  # 13hw (after next step)
            'cam': cam,  # 1244
            'org_depth': depth,
            'depth': depth,  # 11hw
            'prob': probs,  # 13hw
        }
        recursive_apply(views[id], lambda arr: torch.from_numpy(np.ascontiguousarray(arr)).float().unsqueeze(0))
    
    for i, id in tqdm(enumerate(pair['id_list']), 'prob filter', n_views):
        views[id]['mask'] = prob_filter(views[id]['prob'].cuda(), pthresh).cpu()  # 11hw bool
        views[id]['depth'] *= views[id]['mask']
    
    update = {}
    for i, id in tqdm(enumerate(pair['id_list']), 'vis filter', n_views):
        srcs_id = pair[id]['pair'][:args.view]
        ref_depth_g, ref_cam_g = views[id]['depth'].cuda(), views[id]['cam'].cuda()
        srcs_depth_g, srcs_cam_g = [torch.stack([views[loop_id][attr] for loop_id in srcs_id], dim=1).cuda() for attr in ['depth', 'cam']]

        reproj_xyd_g, in_range_g = get_reproj(ref_depth_g, srcs_depth_g, ref_cam_g, srcs_cam_g)
        vis_masks_g, vis_mask_g = vis_filter(ref_depth_g, reproj_xyd_g, in_range_g, 1, 0.01, args.vthresh)

        update[id] = {
            'mask': vis_mask_g.cpu()
        }
        del ref_depth_g, ref_cam_g, srcs_depth_g, srcs_cam_g, reproj_xyd_g, in_range_g, vis_masks_g, vis_mask_g
    
    # update = {}
    # for i, id in tqdm(enumerate(pair['id_list']), 'vis filter and med fusion', n_views):
    #     srcs_id = pair[id]['pair'][:args.view]
    #     ref_depth_g, ref_cam_g = views[id]['depth'].cuda(), views[id]['cam'].cuda()
    #     srcs_depth_g, srcs_cam_g = [torch.stack([views[loop_id][attr] for loop_id in srcs_id], dim=1).cuda() for attr in ['depth', 'cam']]

    #     reproj_xyd_g, in_range_g = get_reproj(ref_depth_g, srcs_depth_g, ref_cam_g, srcs_cam_g)
    #     vis_masks_g, vis_mask_g = vis_filter(ref_depth_g, reproj_xyd_g, in_range_g, 1, 0.01, args.vthresh)

    #     ref_depth_med_g = med_fusion(ref_depth_g, reproj_xyd_g, vis_masks_g, vis_mask_g)

    #     update[id] = {
    #         'depth': ref_depth_med_g.cpu(),
    #         'mask': vis_mask_g.cpu()
    #     }
    #     del ref_depth_g, ref_cam_g, srcs_depth_g, srcs_cam_g, reproj_xyd_g, in_range_g, vis_masks_g, vis_mask_g, ref_depth_med_g
    # for i, id in enumerate(pair['id_list']):
    #     views[id]['mask'] = views[id]['mask'] & update[id]['mask']
    #     views[id]['depth'] = update[id]['depth'] * views[id]['mask']
    
    out_depth_dir = f"{args.out_dir}/depths" 
    out_mask_dir = f"{args.out_dir}/masks" 

    if not os.path.exists(out_depth_dir):
        os.makedirs(out_depth_dir)
    if not os.path.exists(out_mask_dir):
        os.makedirs(out_mask_dir)
    
    for i, id in enumerate(pair['id_list']):
        views[id]['mask'] = views[id]['mask'] & update[id]['mask']
        views[id]['depth'] *= views[id]['mask']

        # print(views[id]["depth"][0, 0].cpu().numpy().dtype)
        depth = views[id]["org_depth"][0, 0].cpu().numpy()
        mask = views[id]["mask"][0, 0].cpu().numpy() # mask (H, W)

        # print(depth.shape, mask.shape)
        depth = depth * mask
        depth = cv2.resize(depth, dsize=None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        
        mask = mask.astype(np.uint8)*255
        mask = cv2.resize(mask, dsize=None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)

        cv2.imwrite(f"{out_depth_dir}/{file_names[i]}.tiff", depth)
        cv2.imwrite(f"{out_mask_dir}/{file_names[i]}.png", mask)

