import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Callable, Any
import numpy as np
import itertools
import time
import matplotlib.pyplot as plt

from core.nn_utils import ListModule, UNet, multi_dims, CSPN, soft_argmin, entropy, StackCGRU, hourglass, bin_op_reduce, groupwise_correlation
from utils.preproc import scale_camera, recursive_apply
from core.homography import get_pixel_grids, get_homographies, homography_warping, interpolate
from utils.utils import NanError

cpg = 8


class FeatExt(nn.Module):

    def __init__(self):
        super(FeatExt, self).__init__()
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, 16, 5, 2, 2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.unet = UNet(16, 2, 1, 2, [], [32, 64, 128], [], '2d', 2)
        self.final_conv_1 = nn.Conv2d(128, 32, 3, 1, 1, bias=False)
        self.final_conv_2 = nn.Conv2d(64, 32, 3, 1, 1, bias=False)
        self.final_conv_3 = nn.Conv2d(32, 32, 3, 1, 1, bias=False)

    def forward(self, x):
        out = self.init_conv(x)
        out1, out2, out3 = self.unet(out, multi_scale=3)
        return self.final_conv_1(out1), self.final_conv_2(out2), self.final_conv_3(out3)


class Reg(nn.Module):

    def __init__(self):
        super(Reg, self).__init__()
        self.init_conv = lambda x: x
        self.unet = UNet(8, 1, 0, 4, [], [8, 16], [], 'reg1', dim=3)

    def forward(self, x):
        init = self.init_conv(x)
        out = self.unet(init)
        return out


class RegPair(nn.Module):

    def __init__(self):
        super(RegPair, self).__init__()
        self.final_conv = nn.Conv3d(8, 1, 3, 1, 1, bias=False)

    def forward(self, x):
        out = self.final_conv(x)
        return out


class RegFuse(nn.Module):

    def __init__(self):
        super(RegFuse, self).__init__()
        self.init_conv = lambda x: x
        self.unet = UNet(8, 1, 0, 4, [], [8, 16], [], 'reg2', dim=3)
        self.final_conv = nn.Conv3d(8, 1, 3, 1, 1, bias=False)

    def forward(self, x):
        init = self.init_conv(x)
        out = self.unet(init)
        out = self.final_conv(out)
        return out


class UncertNet(nn.Module):

    def __init__(self, num_heads=1):
        super(UncertNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )
        self.head_convs = ListModule([nn.Conv2d(8, 1, 3, 1, 1, bias=False) for _ in range(num_heads)])

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out += x
        outs = [conv(out) for conv in self.head_convs]
        return outs


class GNRefine(nn.Module):
    def __init__(self):
        super(GNRefine, self).__init__()
        sobel = torch.Tensor(
            [[
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]
            ], 
            [
                [-1, -2, -1],
                [0, 0, 0],
                [1, 2, 1]
            ]]).to(torch.float32).cuda().repeat(32,1,1).unsqueeze(1)  # 2133
        self.sobel_conv = nn.Conv2d(32, 64, 3, padding=1, groups=32, bias=False)
        self.sobel_conv.weight.requires_grad = False
        self.sobel_conv.weight[...] = sobel
    
    def forward(self, init_d, ref_feat, ref_cam, srcs_feat, srcs_cam, s_scale):
        """
             d Fi     d p2     d p3
        J = ------ * ------ * ------
             d p2     d p3     d d
             c*2      2*3      3*1
        """
        n, c, h, w = ref_feat.size()
        ref_cam_scaled, *srcs_cam_scaled = [scale_camera(cam, 1/s_scale) for cam in [ref_cam]+srcs_cam]
        J_list = []
        r_list = []
        for src_feat, src_cam_scaled in zip(srcs_feat, srcs_cam_scaled):
            H = get_homographies(ref_cam_scaled, src_cam_scaled, 1, init_d.detach(), torch.ones(n,1,1,1, dtype=torch.float32).cuda()).squeeze(1)  # nhw33
            # copied from homography.py
            with torch.no_grad():
                pixel_grids = get_pixel_grids(*src_feat.size()[-2:]).unsqueeze(0)  # 1hw31
                warped_homo_coord = (H @ pixel_grids).squeeze(-1)  # nhw3
                warped_coord = warped_homo_coord[..., :2] / (warped_homo_coord[..., 2:3] + 1e-9)  # nhw2
            warped = interpolate(src_feat, warped_coord)
            residual = (warped - ref_feat).permute(0,2,3,1).unsqueeze(-1)  # nhwc1
            src_grad = self.sobel_conv(src_feat)  # n c*2 hw
            src_grad_warped = interpolate(src_grad, warped_coord).permute(0, 2, 3, 1).reshape(n, h, w, c, 2)  # nhwc2
            d3to2_1 = torch.eye(2, dtype=torch.float32).cuda().view(1,1,1,2,2) / (warped_homo_coord[...,-1].view(n,h,w,1,1)+1e-9)  # nhw22
            d3to2_2 = warped_coord.unsqueeze(-1) / (warped_homo_coord[...,-1].view(n,h,w,1,1)+1e-9)  # nhw21
            d3to2 = torch.cat([d3to2_1, d3to2_2], dim=-1)  # nhw23
            Ki = src_cam_scaled[:, 1, :3, :3].reshape(-1,1,1,3,3)  # n1133
            K0 = ref_cam_scaled[:, 1, :3, :3].reshape(-1,1,1,3,3)
            Ri = src_cam_scaled[:, 0, :3, :3].reshape(-1,1,1,3,3)
            R0 = ref_cam_scaled[:, 0, :3, :3].reshape(-1,1,1,3,3)
            # dptod = Ki @ Ri @ R0.inverse() @ K0.inverse() @ pixel_grids  # nhw31
            dptod = (Ki @ Ri @ R0.inverse() @ K0.inverse() - H) @ pixel_grids / init_d.detach().view(n,h,w,1,1)
            Ji = src_grad_warped @ d3to2 @ dptod  # nhwc1
            r_list.append(residual)
            J_list.append(Ji)
        J, r = [torch.cat(l, dim=-2) for l in [J_list, r_list]]
        delta = (- (J.transpose(-1, -2) @ r) / (J.transpose(-1, -2) @ J + 1e-9)).reshape(n,1,h,w)
        if (delta != delta).any():
            raise NanError
        # delta = delta.clamp(-1, 1)
        # plt.imshow(delta[0,0,...].clone().cpu().data.numpy())
        # plt.show()
        refined_d = init_d + delta
        return refined_d


class SingleStage(nn.Module):

    def __init__(self):
        super(SingleStage, self).__init__()
        # self.feat_ext = FeatExt()
        self.reg = Reg()
        self.reg_fuse = RegFuse()
        self.reg_pair = RegPair()  #MVS
        self.uncert_net = UncertNet(2)  #MVS

    def build_cost_volume(self, ref, ref_cam, src, src_cam, depth_num, depth_start, depth_interval, s_scale, d_scale):
        ref_cam_scaled, src_cam_scaled = [scale_camera(cam, 1 / s_scale) for cam in [ref_cam, src_cam]]
        Hs = get_homographies(ref_cam_scaled, src_cam_scaled, depth_num//d_scale, depth_start, depth_interval*d_scale)
        # ndhw33
        src_nd_c_h_w = src.unsqueeze(1).repeat(1, depth_num//d_scale, 1, 1, 1).view(-1, *src.size()[1:])  # n*d chw
        warped_src_nd_c_h_w = homography_warping(src_nd_c_h_w, Hs.view(-1, *Hs.size()[2:]))  # n*d chw
        warped_src = warped_src_nd_c_h_w.view(-1, depth_num//d_scale, *src.size()[1:]).transpose(1, 2)  # ncdhw
        return warped_src

    def build_cost_maps(self, ref, ref_cam, source, source_cam, depth_num, depth_start, depth_interval, scale):
        ref_cam_scaled, source_cam_scaled = [scale_camera(cam, 1 / scale) for cam in [ref_cam, source_cam]]
        Hs = get_homographies(ref_cam_scaled, source_cam_scaled, depth_num, depth_start, depth_interval)

        cost_maps = []
        for d in range(depth_num):
            H = Hs[:, d, ...]
            warped_source = homography_warping(source, H)
            cost_maps.append(torch.cat([ref, warped_source], dim=1))
        return cost_maps

    def forward_mem(self, sample, depth_num, upsample=False, mode='soft'):  # obsolete
        ref, ref_cam, srcs, srcs_cam = sample
        depth_start = ref_cam[:, 1:2, 3:4, 0:1]  # n111
        depth_interval = ref_cam[:, 1:2, 3:4, 1:2]  # n111
        srcs = [srcs[:, i, ...] for i in range(srcs.size()[1])]
        srcs_cam = [srcs_cam[:, i, ...] for i in range(srcs_cam.size()[1])]

        s_scale = 4
        upsample_scale = 2
        d_scale = 1
        interm_scale = 2

        ref_feat = self.feat_ext(ref)
        ref_ncdhw = ref_feat.unsqueeze(2).repeat(1, 1, depth_num // d_scale, 1, 1)

        pair_results = []

        if mode == 'soft':
            weight_sum = torch.zeros((ref_ncdhw.size()[0], 1, 1, ref_ncdhw.size()[3]//interm_scale, ref_ncdhw.size()[4]//interm_scale)).to(ref_ncdhw.dtype).cuda()
        if mode == 'hard':
            weight_sum = torch.zeros((ref_ncdhw.size()[0], 1, 1, ref_ncdhw.size()[3]//interm_scale, ref_ncdhw.size()[4]//interm_scale)).to(ref_ncdhw.dtype).cuda()
        if mode == 'average':
            pass
        if mode == 'uwta':
            min_weight = None
        if mode == 'maxpool':
            init = True

        fused_interm = torch.zeros((ref_ncdhw.size()[0], 16, ref_ncdhw.size()[2]//interm_scale, ref_ncdhw.size()[3]//interm_scale, ref_ncdhw.size()[4]//interm_scale)).to(ref_ncdhw.dtype).cuda()
        for src, src_cam in zip(srcs, srcs_cam):
            src_feat = self.feat_ext(src)
            warped_src = self.build_cost_volume(ref_feat, ref_cam, src_feat, src_cam, depth_num, depth_start, depth_interval, s_scale, d_scale)
            cost_volume = groupwise_correlation(ref_ncdhw, warped_src, 8, 1)
            interm = self.reg(cost_volume)
            score_volume = self.reg_pair(interm)  # n1dhw
            if d_scale != 1: score_volume = F.interpolate(score_volume, scale_factor=(d_scale, 1, 1), mode='trilinear', align_corners=False)
            score_volume = score_volume.squeeze(1)  # ndhw
            prob_volume, est_depth_class = soft_argmin(score_volume, dim=1, keepdim=True)
            est_depth = est_depth_class * depth_interval * interm_scale + depth_start
            entropy_ = entropy(prob_volume, dim=1, keepdim=True)
            heads = self.uncert_net(entropy_)
            pair_results.append([est_depth, heads])

            if mode == 'soft':
                weight = (-heads[0]).exp().unsqueeze(2)
                weight_sum += weight
                fused_interm += interm * weight
            if mode == 'hard':
                weight = (heads[0]<0).to(interm.dtype).unsqueeze(2) + 1e-4
                weight_sum += weight
                fused_interm += interm * weight
            if mode == 'average':
                weight = None
                fused_interm += interm
            if mode == 'uwta':
                weight = heads[0].unsqueeze(2)
                if min_weight is None:
                    min_weight = weight
                    mask = torch.ones_like(min_weight).to(interm.dtype).cuda()
                else:
                    mask = (weight<min_weight).to(interm.dtype)
                    min_weight = weight * mask + min_weight * (1 - mask)
                fused_interm = interm * mask + fused_interm * (1 - mask)
            if mode == 'maxpool':
                weight = None
                if init:
                    fused_interm += interm
                    init = False
                else:
                    fused_interm = torch.max(fused_interm, interm)

            # if not self.training:
            #     del weight, prob_volume, est_depth_class, score_volume, interm, cost_volume, warped_src, src_feat

        if mode == 'soft':
            fused_interm /= weight_sum
        if mode == 'hard':
            fused_interm /= weight_sum
        if mode == 'average':
            fused_interm /= len(srcs)
        if mode == 'uwta':
            pass
        if mode == 'maxpool':
            pass

        score_volume = self.reg_fuse(fused_interm)  # n1dhw
        if d_scale != 1: score_volume = F.interpolate(score_volume, scale_factor=(d_scale, 1, 1), mode='trilinear', align_corners=False)
        score_volume = score_volume.squeeze(1)  # ndhw
        if upsample:
            score_volume = F.interpolate(score_volume, scale_factor=upsample_scale, mode='bilinear', align_corners=False)
        prob_volume, est_depth_class, prob_map = soft_argmin(score_volume, dim=1, keepdim=True, window=2)
        est_depth = est_depth_class * depth_interval + depth_start

        return est_depth, prob_map, pair_results

    def forward(self, sample, depth_num, upsample=False, mem=False, mode='soft', depth_start_override=None, depth_interval_override=None, s_scale=1):
        if mem:
            raise NotImplementedError

        ref_feat, ref_cam, srcs_feat, srcs_cam = sample
        depth_start = ref_cam[:, 1:2, 3:4, 0:1] if depth_start_override is None else depth_start_override  # n111 or n1hw
        depth_interval = ref_cam[:, 1:2, 3:4, 1:2] if depth_interval_override is None else depth_interval_override  # n111

        upsample_scale = 1
        d_scale = 1
        interm_scale = 1
        
        ref_ncdhw = ref_feat.unsqueeze(2).repeat(1, 1, depth_num//d_scale, 1, 1)
        pair_results = []  #MVS

        if mode == 'soft':
            weight_sum = torch.zeros((ref_ncdhw.size()[0], 1, 1, ref_ncdhw.size()[3]//interm_scale, ref_ncdhw.size()[4]//interm_scale)).to(ref_ncdhw.dtype).cuda()
        if mode == 'hard':
            weight_sum = torch.zeros((ref_ncdhw.size()[0], 1, 1, ref_ncdhw.size()[3]//interm_scale, ref_ncdhw.size()[4]//interm_scale)).to(ref_ncdhw.dtype).cuda()
        if mode == 'average':
            pass
        if mode == 'uwta':
            min_weight = None
        if mode == 'maxpool':
            init = True
        fused_interm = torch.zeros((ref_ncdhw.size()[0], 8, ref_ncdhw.size()[2]//interm_scale, ref_ncdhw.size()[3]//interm_scale, ref_ncdhw.size()[4]//interm_scale)).to(ref_ncdhw.dtype).cuda()

        for src_feat, src_cam in zip(srcs_feat, srcs_cam):
            warped_src = self.build_cost_volume(ref_feat, ref_cam, src_feat, src_cam, depth_num, depth_start, depth_interval, s_scale, d_scale)
            cost_volume = groupwise_correlation(ref_ncdhw, warped_src, 8, 1)
            interm = self.reg(cost_volume)
            # if not self.training: del cost_volume
            score_volume = self.reg_pair(interm)  # n1dhw
            if d_scale != 1: score_volume = F.interpolate(score_volume, scale_factor=(d_scale,1,1), mode='trilinear', align_corners=False)
            score_volume = score_volume.squeeze(1)  # ndhw
            prob_volume, est_depth_class = soft_argmin(score_volume, dim=1, keepdim=True)
            est_depth = est_depth_class * depth_interval * interm_scale + depth_start
            entropy_ = entropy(prob_volume, dim=1, keepdim=True)
            heads = self.uncert_net(entropy_)
            pair_results.append([est_depth, heads])
            # if not self.training: del score_volume, prob_volume

            if mode == 'soft':
                weight = (-heads[0]).exp().unsqueeze(2)
                weight_sum = weight_sum + weight
                fused_interm = fused_interm + interm * weight
            if mode == 'hard':
                weight = (heads[0]<0).to(interm.dtype).unsqueeze(2) + 1e-4
                weight_sum = weight_sum + weight
                fused_interm = fused_interm + interm * weight
            if mode == 'average':
                weight = None
                fused_interm = fused_interm + interm
            if mode == 'uwta':
                weight = heads[0].unsqueeze(2)
                if min_weight is None:
                    min_weight = weight
                    mask = torch.ones_like(min_weight).to(interm.dtype).cuda()
                else:
                    mask = (weight<min_weight).to(interm.dtype)
                    min_weight = weight * mask + min_weight * (1 - mask)
                fused_interm = interm * mask + fused_interm * (1 - mask)
            if mode == 'maxpool':
                weight = None
                if init:
                    fused_interm = fused_interm + interm
                    init = False
                else:
                    fused_interm = torch.max(fused_interm, interm)
            
            if not self.training:
                del weight, prob_volume, est_depth_class, score_volume, interm, cost_volume, warped_src
        
        if mode == 'soft':
            fused_interm /= weight_sum
        if mode == 'hard':
            fused_interm /= weight_sum
        if mode == 'average':
            fused_interm /= len(srcs_feat)
        if mode == 'uwta':
            pass
        if mode == 'maxpool':
            pass

        score_volume = self.reg_fuse(fused_interm)  # n1dhw
        if d_scale != 1: score_volume = F.interpolate(score_volume, scale_factor=(d_scale,1,1), mode='trilinear', align_corners=False)
        score_volume = score_volume.squeeze(1)  # ndhw
        if upsample:
            score_volume = F.interpolate(score_volume, scale_factor=upsample_scale, mode='bilinear', align_corners=False)

        prob_volume, est_depth_class, prob_map = soft_argmin(score_volume, dim=1, keepdim=True, window=2)
        est_depth = est_depth_class * depth_interval + depth_start

        # entropy_ = entropy(prob_volume, dim=1, keepdim=True)
        # uncert = self.uncert_net(entropy_)
        # uncert = torch.cuda.FloatTensor(*est_depth.size()).zero_()

        # if upsample and est_depth.size() != gt.size():
        #     final_size = gt.size()
        #     size = est_depth.size()
        #     # est_depth, uncert = [
        #     #     F.interpolate(img, size=(final_size[2], final_size[3]), mode='bilinear', align_corners=False)
        #     #     for img in [est_depth, uncert]
        #     # ]
        #     est_depth = F.interpolate(est_depth, size=(final_size[2], final_size[3]), mode='bilinear', align_corners=False)

        return est_depth, prob_map, pair_results  #MVS


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.feat_ext = FeatExt()
        self.stage1 = SingleStage()
        self.stage2 = SingleStage()
        self.stage3 = SingleStage()
        self.refine = GNRefine()
    
    def forward(self, sample, depth_nums, interval_scales, upsample=False, mem=False, mode='soft'):
        ref, ref_cam, srcs, srcs_cam = [sample[attr] for attr in ['ref', 'ref_cam', 'srcs', 'srcs_cam']]
        depth_start = ref_cam[:, 1:2, 3:4, 0:1]  # n111
        depth_interval = ref_cam[:, 1:2, 3:4, 1:2]  # n111
        srcs_cam = [srcs_cam[:, i, ...] for i in range(srcs_cam.size()[1])]

        n, v, c, h, w = srcs.size()
        img_pack = torch.cat([ref, srcs.transpose(0, 1).reshape(v*n, c, h, w)])
        feat_pack_1, feat_pack_2, feat_pack_3 = self.feat_ext(img_pack)

        ref_feat_1, *srcs_feat_1 = [feat_pack_1[i*n:(i+1)*n] for i in range(v+1)]
        est_depth_1, prob_map_1, pair_results_1 = self.stage1([ref_feat_1, ref_cam, srcs_feat_1, srcs_cam], depth_num=depth_nums[0], upsample=False, mem=mem, mode=mode, depth_start_override=None, depth_interval_override=depth_interval*interval_scales[0], s_scale=8)
        prob_map_1_up = F.interpolate(prob_map_1, scale_factor=4, mode='bilinear', align_corners=False)

        ref_feat_2, *srcs_feat_2 = [feat_pack_2[i*n:(i+1)*n] for i in range(v+1)]
        depth_start_2 = F.interpolate(est_depth_1.detach(), size=(ref_feat_2.size()[2], ref_feat_2.size()[3]), mode='bilinear', align_corners=False) - depth_nums[1] * depth_interval * interval_scales[1] / 2
        est_depth_2, prob_map_2, pair_results_2 = self.stage2([ref_feat_2, ref_cam, srcs_feat_2, srcs_cam], depth_num=depth_nums[1], upsample=False, mem=mem, mode=mode, depth_start_override=depth_start_2, depth_interval_override=depth_interval*interval_scales[1], s_scale=4)
        prob_map_2_up = F.interpolate(prob_map_2, scale_factor=2, mode='bilinear', align_corners=False)

        ref_feat_3, *srcs_feat_3 = [feat_pack_3[i*n:(i+1)*n] for i in range(v+1)]
        depth_start_3 = F.interpolate(est_depth_2.detach(), size=(ref_feat_3.size()[2], ref_feat_3.size()[3]), mode='bilinear', align_corners=False) - depth_nums[2] * depth_interval * interval_scales[2] / 2
        est_depth_3, prob_map_3, pair_results_3 = self.stage3([ref_feat_3, ref_cam, srcs_feat_3, srcs_cam], depth_num=depth_nums[2], upsample=upsample, mem=mem, mode=mode, depth_start_override=depth_start_3, depth_interval_override=depth_interval*interval_scales[2], s_scale=2)

        # refined_depth = self.refine(est_depth_3, ref_feat_3, ref_cam, srcs_feat_3, srcs_cam, 2)
        refined_depth = est_depth_3

        return [[est_depth_1, pair_results_1], [est_depth_2, pair_results_2], [est_depth_3, pair_results_3]], refined_depth, [prob_map_1_up, prob_map_2_up, prob_map_3]


class Loss(nn.Module):  # TODO

    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, outputs, gt, masks, ref_cam, max_d, occ_guide=False, mode='soft'):  #MVS
        outputs, refined_depth = outputs

        depth_start = ref_cam[:, 1:2, 3:4, 0:1]  # n111
        depth_interval = ref_cam[:, 1:2, 3:4, 1:2]  # n111
        depth_end = depth_start + (max_d - 2) * depth_interval  # strict range
        masks = [masks[:, i, ...] for i in range(masks.size()[1])]

        stage_losses = []
        stats = []
        for est_depth, pair_results in outputs:
            gt_downsized = F.interpolate(gt, size=(est_depth.size()[2], est_depth.size()[3]), mode='bilinear', align_corners=False)
            masks_downsized = [
                F.interpolate(mask, size=(est_depth.size()[2], est_depth.size()[3]), mode='nearest')
                for mask in masks
            ]
            in_range = torch.min((gt_downsized >= depth_start), (gt_downsized <= depth_end))
            masks_valid = [torch.min((mask > 50), in_range) for mask in masks_downsized]  # mask and in_range
            masks_overlap = [torch.min((mask > 200), in_range) for mask in masks_downsized]
            union_overlap = bin_op_reduce(masks_overlap, torch.max)  # A(B+C)=AB+AC
            valid = union_overlap if occ_guide else in_range

            same_size = est_depth.size()[2]==pair_results[0][0].size()[2] and est_depth.size()[3]==pair_results[0][0].size()[3]
            gt_interm = F.interpolate(gt, size=(pair_results[0][0].size()[2], pair_results[0][0].size()[3]), mode='bilinear', align_corners=False) if not same_size else gt_downsized
            masks_interm = [
                F.interpolate(mask, size=(pair_results[0][0].size()[2], pair_results[0][0].size()[3]), mode='nearest')
                for mask in masks
            ] if not same_size else masks_downsized
            in_range_interm = torch.min((gt_interm >= depth_start), (gt_interm <= depth_end)) if not same_size else in_range
            masks_valid_interm = [torch.min((mask > 50), in_range_interm) for mask in masks_interm] if not same_size else masks_valid  # mask and in_range
            masks_overlap_interm = [torch.min((mask > 200), in_range_interm) for mask in masks_interm] if not same_size else masks_overlap
            union_overlap_interm = bin_op_reduce(masks_overlap_interm, torch.max) if not same_size else union_overlap  # A(B+C)=AB+AC
            valid_interm = (union_overlap_interm if occ_guide else in_range_interm) if not same_size else valid

            abs_err = (est_depth - gt_downsized).abs()
            abs_err_scaled = abs_err / depth_interval
            pair_abs_err = [(est - gt_interm).abs() for est in [est for est, (uncert, occ) in pair_results]]
            pair_abs_err_scaled = [err / depth_interval for err in pair_abs_err]

            l1 = abs_err_scaled[valid].mean()

            # ===== pair l1 =====
            if occ_guide:
                pair_l1_losses = [
                    err[mask_overlap].mean()
                    for err, mask_overlap in zip(pair_abs_err_scaled, masks_overlap_interm)
                ]
            else:
                pair_l1_losses = [
                    err[in_range_interm].mean()
                    for err in pair_abs_err_scaled
                ]
            pair_l1_loss = sum(pair_l1_losses) / len(pair_l1_losses)

            # ===== uncert =====
            if mode in ['soft', 'hard', 'uwta']:
                if occ_guide:
                    uncert_losses = [
                        (err[mask_valid] * (-uncert[mask_valid]).exp() + uncert[mask_valid]).mean()
                        for err, (est, (uncert, occ)), mask_valid, mask_overlap in zip(pair_abs_err_scaled, pair_results, masks_valid_interm, masks_overlap_interm)
                    ]
                else:
                    uncert_losses = [
                        (err[in_range_interm] * (-uncert[in_range_interm]).exp() + uncert[in_range_interm]).mean()
                        for err, (est, (uncert, occ)) in zip(pair_abs_err_scaled, pair_results)
                    ]
                uncert_loss = sum(uncert_losses) / len(uncert_losses)

            # ===== logistic =====
            if occ_guide and mode in ['soft', 'hard', 'uwta']:
                logistic_losses = [
                    nn.SoftMarginLoss()(occ[mask_valid], -mask_overlap[mask_valid].to(gt.dtype)*2+1)
                    for (est, (uncert, occ)), mask_valid, mask_overlap in zip(pair_results, masks_valid_interm, masks_overlap_interm)
                ]
                logistic_loss = sum(logistic_losses) / len(logistic_losses)

            less1 = (abs_err_scaled[valid] < 1.).to(gt.dtype).mean()
            less3 = (abs_err_scaled[valid] < 3.).to(gt.dtype).mean()

            pair_loss = pair_l1_loss
            if mode in ['soft', 'hard', 'uwta']:
                pair_loss = pair_loss + uncert_loss
                if occ_guide:
                    pair_loss = pair_loss + logistic_loss
            loss = l1 + pair_loss
            stage_losses.append(loss)
            stats.append((l1, less1, less3))
        
        abs_err = (refined_depth - gt_downsized).abs()
        abs_err_scaled = abs_err / depth_interval
        l1 = abs_err_scaled[valid].mean()
        less1 = (abs_err_scaled[valid] < 1.).to(gt.dtype).mean()
        less3 = (abs_err_scaled[valid] < 3.).to(gt.dtype).mean()
        
        loss = stage_losses[0]*0.5 + stage_losses[1]*1.0 + stage_losses[2]*2.0# + l1*2.0

        return loss, pair_loss, less1, less3, l1, stats, abs_err_scaled, valid
