#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render_fn_dict
from torchvision.utils import save_image
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene, GaussianModel
from scene.direct_light_map import DirectLightMap
from lpipsPyTorch import lpips
from utils.loss_utils import ssim
from utils.image_utils import psnr


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, pbr_kwargs=None):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    normal_path = os.path.join(model_path, name, "ours_{}".format(iteration), "normal")
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(normal_path, exist_ok=True)
    if gaussians.use_pbr:
        base_color_path = os.path.join(model_path, name, "ours_{}".format(iteration), "base_color")
        roughness_path = os.path.join(model_path, name, "ours_{}".format(iteration), "roughness")
        lights_path = os.path.join(model_path, name, "ours_{}".format(iteration), "lights")
        local_lights_path = os.path.join(model_path, name, "ours_{}".format(iteration), "local_lights")
        global_lights_path = os.path.join(model_path, name, "ours_{}".format(iteration), "global_lights")
        visibility_path = os.path.join(model_path, name, "ours_{}".format(iteration), "visibility")
        makedirs(base_color_path, exist_ok=True)
        makedirs(roughness_path, exist_ok=True)
        makedirs(lights_path, exist_ok=True)
        makedirs(local_lights_path, exist_ok=True)
        makedirs(global_lights_path, exist_ok=True)
        makedirs(visibility_path, exist_ok=True)
    
    psnr_test = 0.0
    ssim_test = 0.0
    lpips_test = 0.0
    render_fn = render_fn_dict[args.type]
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        results = render_fn(view, gaussians, pipeline, background, dict_params=pbr_kwargs)
        gt = view.original_image[0:3, :, :]
        save_image(results["render"], os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        save_image(results["normal"] * 0.5 + 0.5,  os.path.join(normal_path, '{0:05d}'.format(idx) + ".png"))
        
        if gaussians.use_pbr:
            save_image(results["pbr"], os.path.join(base_color_path, '{0:05d}'.format(idx) + ".png"))
            save_image(results["base_color"], os.path.join(base_color_path, '{0:05d}'.format(idx) + ".png"))
            save_image(results["roughness"], os.path.join(roughness_path, '{0:05d}'.format(idx) + ".png"))
            save_image(results["lights"], os.path.join(lights_path, '{0:05d}'.format(idx) + ".png"))
            save_image(results["local_lights"], os.path.join(local_lights_path, '{0:05d}'.format(idx) + ".png"))
            save_image(results["global_lights"], os.path.join(global_lights_path, '{0:05d}'.format(idx) + ".png"))
            save_image(results["visibility"], os.path.join(visibility_path, '{0:05d}'.format(idx) + ".png"))
        
        img = results["pbr"] if gaussians.use_pbr else results["render"]
        with torch.no_grad():
            psnr_test += psnr(img, gt).mean().double()
            ssim_test += ssim(img, gt).mean().double()
            lpips_test += lpips(img, gt, net_type='vgg').mean().double()

    psnr_test /= len(views)
    ssim_test /= len(views)
    lpips_test /= len(views)
    with open(os.path.join(model_path, f"metric_{name}.txt"), "w") as f:
        f.write(f"psnr: {psnr_test}\n")
        f.write(f"ssim: {ssim_test}\n")
        f.write(f"lpips: {lpips_test}\n")
    print("\nEvaluating {}: PSNR {} SSIM {} LPIPS {}".format(name, psnr_test, ssim_test, lpips_test))

def render_sets(dataset : ModelParams, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, render_type=args.type)
        scene = Scene(dataset, gaussians, shuffle=False)
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        if args.checkpoint:
            print("Create Gaussians from checkpoint {}".format(args.checkpoint))
            iteration = gaussians.create_from_ckpt(args.checkpoint, restore_optimizer=True)
        elif scene.loaded_iter:
            gaussians.load_ply(os.path.join(dataset.model_path,
                                            "point_cloud",
                                            "iteration_" + str(scene.loaded_iter),
                                            "point_cloud.ply"))
            iteration = scene.loaded_iter
        else:
            gaussians.create_from_pcd(scene.scene_info.point_cloud, scene.cameras_extent)
            iteration = scene.loaded_iter

        pbr_kwargs = dict()
        if iteration is not None and gaussians.use_pbr:
            gaussians.update_visibility(args.sample_num)
        
            pbr_kwargs['sample_num'] = args.sample_num
            print("Using global incident light for regularization.")
            direct_env_light = DirectLightMap(args.env_resolution)
            
            if args.checkpoint:
                env_checkpoint = os.path.dirname(args.checkpoint) + "/env_light_" + os.path.basename(args.checkpoint)
                print("Trying to load global incident light from ", env_checkpoint)
                if os.path.exists(env_checkpoint):
                    direct_env_light.create_from_ckpt(env_checkpoint, restore_optimizer=True)
                    print("Successfully loaded!")
                else:
                    print("Failed to load!")
                pbr_kwargs["env_light"] = direct_env_light

        
        if not skip_train:
             render_set(dataset.model_path, "train", iteration, scene.getTrainCameras(), gaussians, pipeline, background, pbr_kwargs)

        if not skip_test:
             render_set(dataset.model_path, "test", iteration, scene.getTestCameras(), gaussians, pipeline, background, pbr_kwargs)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('-t', '--type', choices=['render', 'normal', 'neilf'], default='render')
    parser.add_argument("-c", "--checkpoint", type=str, default=None)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), pipeline.extract(args), args.skip_train, args.skip_test)