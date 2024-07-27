import json
import os
from gaussian_renderer import render_fn_dict
import numpy as np
import torch
from scene import GaussianModel
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from scene.cameras import Camera
from scene.envmap import EnvLight
from utils.graphics_utils import focal2fov, fov2focal
from torchvision.utils import save_image
from tqdm import tqdm
from lpipsPyTorch import lpips
from utils.loss_utils import ssim
from utils.image_utils import psnr
from scene.utils import load_img_rgb
import warnings
warnings.filterwarnings("ignore")


def load_json_config(json_file):
    if not os.path.exists(json_file):
        return None

    with open(json_file, 'r', encoding='UTF-8') as f:
        load_dict = json.load(f)

    return load_dict


if __name__ == '__main__':
    # Set up command line argument parser
    parser = ArgumentParser(description="Composition and Relighting for Relightable 3D Gaussian")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("-c", "--checkpoint", type=str, default=None)
    parser.add_argument('-e', '--envmap_path', default=None, help="Env map path")
    parser.add_argument('-bg', "--background_color", type=float, default=1,
                        help="If set, use it as background color")
    args = get_combined_args(parser)
    dataset = model.extract(args)
    pipe = pipeline.extract(args)

    # load gaussians
    gaussians = GaussianModel(model.sh_degree, render_type="neilf")
    
    if args.checkpoint:
        print("Create Gaussians from checkpoint {}".format(args.checkpoint))
        iteration = gaussians.create_from_ckpt(args.checkpoint, restore_optimizer=False)
    else:
        raise NotImplementedError
        
    # deal with each item
    test_transforms_file = os.path.join(args.source_path, "transforms_test.json")
    contents = load_json_config(test_transforms_file)

    fovx = contents["camera_angle_x"]
    frames = contents["frames"]

    task_dict = {
        "env6": {
            "capture_list": ["pbr", "pbr_env", "base_color", "roughness"],
            "envmap_path": "env_map/envmap6.exr",
        },
        "env12": {
            "capture_list": ["pbr", "pbr_env", "base_color", "roughness"],
            "envmap_path": "env_map/envmap12.exr",
        }
    }

    bg = 1 if dataset.white_background else 0
    background = torch.tensor([bg, bg, bg], dtype=torch.float32, device="cuda")
    render_fn = render_fn_dict['neilf']
    gaussians.update_visibility(args.sample_num)
    
    results_dir = os.path.join(args.model_path, "test_rli")
    task_names = ['env6', 'env12']
    for task_name in task_names:
        task_dir = os.path.join(results_dir, task_name)
        os.makedirs(task_dir, exist_ok=True)
        light = EnvLight(path=task_dict[task_name]["envmap_path"], scale=1)

        render_kwargs = {
            "pc": gaussians,
            "pipe": pipe,
            "bg_color": background,
            "is_training": False,
            "dict_params": {
                "env_light": light,
                "sample_num": args.sample_num,
            },
        }
        
        if "/air_baloons/" in args.model_path:
            gaussians.base_color_scale = torch.tensor([1.3746, 0.6428, 0.7279], dtype=torch.float32, device="cuda")
        elif "/chair/" in args.model_path:
            gaussians.base_color_scale = torch.tensor([1.8865, 1.9675, 1.7410], dtype=torch.float32, device="cuda")
        elif "/hotdog/" in args.model_path:
            gaussians.base_color_scale = torch.tensor([2.6734, 2.0917, 1.2587], dtype=torch.float32, device="cuda")
        elif "/jugs/" in args.model_path:
            # gaussians.base_color_scale = torch.tensor([1.1916, 0.9296, 0.5684], dtype=torch.float32, device="cuda")
            gaussians.base_color_scale = torch.tensor([1.0044, 0.9253, 0.7648], dtype=torch.float32, device="cuda")
        else:
            raise NotImplementedError
        
        psnr_pbr = 0.0
        ssim_pbr = 0.0
        lpips_pbr = 0.0
        
        psnr_albedo = 0.0
        ssim_albedo = 0.0
        lpips_albedo = 0.0
        
        mse_roughness = 0.0
        
        capture_list = task_dict[task_name]["capture_list"]
        for capture_type in capture_list:
            capture_type_dir = os.path.join(task_dir, capture_type)
            os.makedirs(capture_type_dir, exist_ok=True)

        os.makedirs(os.path.join(task_dir, "gt"), exist_ok=True)
        os.makedirs(os.path.join(task_dir, "gt_albedo"), exist_ok=True)
        os.makedirs(os.path.join(task_dir, "gt_roughness"), exist_ok=True)
        os.makedirs(os.path.join(task_dir, "gt_pbr_env"), exist_ok=True)
        envname = os.path.splitext(os.path.basename(task_dict[task_name]["envmap_path"]))[0]
            
        for idx, frame in enumerate(tqdm(frames, leave=False)):
            image_path = os.path.join(args.source_path, "test_rli/" + envname + "_" +  frame["file_path"].split("/")[-1] + ".png")
            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_rgba = load_img_rgb(image_path)
            image = image_rgba[..., :3]
            mask = image_rgba[..., 3:]

            gt_image = torch.from_numpy(image).permute(2, 0, 1).float().cuda()
            mask = torch.from_numpy(mask).permute(2, 0, 1).float().cuda()
            
            H = image.shape[0]
            W = image.shape[1]
            fovy = focal2fov(fov2focal(fovx, W), H)

            custom_cam = Camera(colmap_id=0, R=R, T=T,
                                FoVx=fovx, FoVy=fovy, fx=None, fy=None, cx=None, cy=None,
                                image=torch.zeros(3, H, W), image_name=None, uid=0)

            with torch.no_grad():
                render_pkg = render_fn(viewpoint_camera=custom_cam, **render_kwargs)

            for capture_type in capture_list:
                if capture_type == "normal":
                    render_pkg[capture_type] = render_pkg[capture_type] * 0.5 + 0.5
                    render_pkg[capture_type] = render_pkg[capture_type] * mask + (1 - mask) * bg
                elif capture_type in ["roughness", "diffuse", "specular", "lights", "local_lights", "global_lights", "visibility"]:
                    render_pkg[capture_type] = render_pkg[capture_type] * mask + (1 - mask) * bg
                elif capture_type in ["base_color"]:
                    render_pkg[capture_type] = render_pkg[capture_type] * mask + (1 - mask) * bg
                elif capture_type in ["pbr"]:
                    render_pkg[capture_type] = render_pkg["pbr"] * mask + (1 - mask) * bg
                elif capture_type in ["pbr_env"]:
                    render_pkg[capture_type] = render_pkg["pbr"] * mask + (1 - mask) * render_pkg["env_only"]
                save_image(render_pkg[capture_type], os.path.join(task_dir, capture_type, f"{idx}.png"))
                
            gt_image = gt_image * mask + bg * (1 - mask)
            save_image(gt_image, os.path.join(task_dir, "gt", f"{idx}.png"))
            albedo_path = os.path.join(args.source_path, "test/" + frame["file_path"].split("/")[-1] + "_albedo.png")
            albedo_rgba = load_img_rgb(albedo_path)
            gt_albedo = torch.from_numpy(albedo_rgba[..., :3]).permute(2, 0, 1).float().cuda()
            gt_albedo = gt_albedo * mask + bg * (1 - mask)
            save_image(gt_albedo, os.path.join(task_dir, "gt_albedo", f"{idx}.png"))
            
            roughness_path = os.path.join(args.source_path, "test/" + frame["file_path"].split("/")[-1] + "_rough.png")
            roughness_rgba = load_img_rgb(roughness_path)
            gt_roughness = torch.from_numpy(roughness_rgba[..., :3]).permute(2, 0, 1).float().cuda()
            gt_roughness = gt_roughness * mask + bg * (1 - mask)
            save_image(gt_roughness, os.path.join(task_dir, "gt_roughness", f"{idx}.png"))
            
            gt_image_env = gt_image * mask + render_pkg["env_only"] * (1 - mask)
            save_image(gt_image_env, os.path.join(task_dir, "gt_pbr_env", f"{idx}.png"))
            
            with torch.no_grad():
                psnr_pbr += psnr(render_pkg['pbr'], gt_image).mean().double()
                ssim_pbr += ssim(render_pkg['pbr'], gt_image).mean().double()
                lpips_pbr += lpips(render_pkg['pbr'], gt_image, net_type='vgg').mean().double()
                
                psnr_albedo += psnr(render_pkg['base_color'], gt_albedo).mean().double()
                ssim_albedo += ssim(render_pkg['base_color'], gt_albedo).mean().double()
                lpips_albedo += lpips(render_pkg['base_color'], gt_albedo, net_type='vgg').mean().double()
                
                mse_roughness += ((render_pkg['roughness'] - gt_roughness)**2).mean().double()
            
            if idx == 0:
                albedo_scale = (gt_albedo / render_pkg['base_color'].clamp(1e-6, 1))[:, mask[0] > 0].median(dim=1).values
                print("Albedo scale:", albedo_scale)

        psnr_pbr /= len(frames)
        ssim_pbr /= len(frames)
        lpips_pbr /= len(frames)
        
        psnr_albedo /= len(frames)
        ssim_albedo /= len(frames)
        lpips_albedo /= len(frames)
        
        mse_roughness /= len(frames)
        
        with open(os.path.join(task_dir, f"metric.txt"), "w") as f:
            f.write(f"psnr_pbr: {psnr_pbr}\n")
            f.write(f"ssim_pbr: {ssim_pbr}\n")
            f.write(f"lpips_pbr: {lpips_pbr}\n")
            f.write(f"psnr_albedo: {psnr_albedo}\n")
            f.write(f"ssim_albedo: {ssim_albedo}\n")
            f.write(f"lpips_albedo: {lpips_albedo}\n")
            f.write(f"mse_roughness: {mse_roughness}\n")
            
        print("\nEvaluating {}: PSNR_PBR {} SSIM_PBR {} LPIPS_PBR {}".format(task_name, psnr_pbr, ssim_pbr, lpips_pbr))
        print("\nEvaluating {}: PSNR_ALBEDO {} SSIM_ALBEDO {} LPIPS_ALBEDO {}".format(task_name, psnr_albedo, ssim_albedo, lpips_albedo))
        print("\nEvaluating {}: MSE_ROUGHNESS {}".format(task_name, mse_roughness))