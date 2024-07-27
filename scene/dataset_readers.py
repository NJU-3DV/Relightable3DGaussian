import re
import os
import sys
import json
import numpy as np

from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from tqdm import tqdm
import cv2
from scene.utils import load_img_rgb, load_mask_bool, load_depth, load_pfm

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    FovY: np.array = None
    FovX: np.array = None
    fx: np.array = None
    fy: np.array = None
    cx: np.array = None
    cy: np.array = None
    normal: np.array = None
    depth: np.array = None
    image_mask: np.array = None

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, debug=False):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[0]
            ppx = intr.params[1]
            ppy = intr.params[2]

            Fovx = focal2fov(focal_length_x, width)
            FovY = focal2fov(focal_length_y, height)

        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            ppx = intr.params[2]
            ppy = intr.params[3]

            Fovx = focal2fov(focal_length_x, width)
            FovY = focal2fov(focal_length_y, height)

        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = load_img_rgb(image_path)

        mask_path = os.path.join(os.path.dirname(images_folder), "masks", os.path.basename(extr.name))
        mask = 1.0 - load_mask_bool(mask_path) / 255
        image = image * mask
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovX=Fovx, FovY=FovY, fx=focal_length_x, fy=focal_length_y, cx=ppx,
                              cy=ppy, image=image, image_path=image_path, image_name=image_name, width=width, height=height,
                              image_mask=mask)
        cam_infos.append(cam_info)

        if debug and idx >= 5:
            break
    sys.stdout.write('\n')
    return cam_infos


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T

    if colors.dtype == np.uint8:
        colors = colors.astype(np.float32)
        colors /= 255.0

    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    if np.all(normals == 0):
        print("random init normal")
        normals = np.random.random(normals.shape)

    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb, normals=None):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    if normals is None:
        normals = np.random.randn(*xyz.shape)
        normals /= np.linalg.norm(normals, axis=-1, keepdims=True)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def readColmapSceneInfo(path, images, eval, llffhold=8, debug=False):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images is None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,
                                           images_folder=os.path.join(path, reading_dir),
                                           debug=debug)
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    if "DTU" in path and not debug:
        test_indexes = [2, 12, 17, 30, 34]
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx not in test_indexes]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx in test_indexes]
    elif eval and not debug:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png", debug=False):
    cam_infos = []

    read_mvs = False
    mvs_dir = f"{path}/extra"
    if os.path.exists(mvs_dir) and "train" not in transformsfile:
        print("Loading mvs as geometry constraint.")
        read_mvs = True

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(tqdm(frames, leave=False)):
            image_path = os.path.join(path, frame["file_path"] + extension)
            image_name = Path(image_path).stem

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image = load_img_rgb(image_path)
            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

            image_mask = np.ones_like(image[..., 0])
            if image.shape[-1] == 4:
                image_mask = image[:, :, 3]
                image = image[:, :, :3] * image[:, :, 3:4] + bg * (1 - image[:, :, 3:4])

            # read depth and mask
            depth = None
            normal = None
            if read_mvs:
                depth_path = os.path.join(mvs_dir + "/depths/", os.path.basename(frame["file_path"]) + ".tiff")
                normal_path = os.path.join(mvs_dir + "/normals/", os.path.basename(frame["file_path"]) + ".pfm")

                depth = load_depth(depth_path)
                normal = load_pfm(normal_path)

                depth = depth * image_mask
                normal = normal * image_mask[..., np.newaxis]

            fovy = focal2fov(fov2focal(fovx, image.shape[0]), image.shape[1])
            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=fovy, FovX=fovx, image=image, image_mask=image_mask,
                                        image_path=image_path, depth=depth, normal=normal, image_name=image_name,
                                        width=image.shape[1], height=image.shape[0]))

            if debug and idx >= 5:
                break

    return cam_infos


def readNerfSyntheticInfo(path, white_background, eval, extension=".png", debug=False):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension, debug=debug)
    if eval:
        print("Reading Test Transforms")
        test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension,
                                                   debug=debug)
    else:
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        normals = np.random.randn(*xyz.shape)
        normals /= np.linalg.norm(normals, axis=-1, keepdims=True)

        storePly(ply_path, xyz, SH2RGB(shs) * 255, normals)

    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)

    return scene_info


def loadCamsFromScene(path, valid_list, white_background, debug):
    with open(f'{path}/sfm_scene.json') as f:
        sfm_scene = json.load(f)

    # load bbox transform
    bbox_transform = np.array(sfm_scene['bbox']['transform']).reshape(4, 4)
    bbox_transform = bbox_transform.copy()
    bbox_transform[[0, 1, 2], [0, 1, 2]] = bbox_transform[[0, 1, 2], [0, 1, 2]].max() / 2
    bbox_inv = np.linalg.inv(bbox_transform)

    # meta info
    image_list = sfm_scene['image_path']['file_paths']

    # camera parameters
    train_cam_infos = []
    test_cam_infos = []
    camera_info_list = sfm_scene['camera_track_map']['images']
    for i, (index, camera_info) in enumerate(camera_info_list.items()):
        # flg == 2 stands for valid camera 
        if camera_info['flg'] == 2:
            intrinsic = np.zeros((4, 4))
            intrinsic[0, 0] = camera_info['camera']['intrinsic']['focal'][0]
            intrinsic[1, 1] = camera_info['camera']['intrinsic']['focal'][1]
            intrinsic[0, 2] = camera_info['camera']['intrinsic']['ppt'][0]
            intrinsic[1, 2] = camera_info['camera']['intrinsic']['ppt'][1]
            intrinsic[2, 2] = intrinsic[3, 3] = 1

            extrinsic = np.array(camera_info['camera']['extrinsic']).reshape(4, 4)
            c2w = np.linalg.inv(extrinsic)
            c2w[:3, 3] = (c2w[:4, 3] @ bbox_inv.T)[:3]
            extrinsic = np.linalg.inv(c2w)

            R = np.transpose(extrinsic[:3, :3])
            T = extrinsic[:3, 3]

            focal_length_x = camera_info['camera']['intrinsic']['focal'][0]
            focal_length_y = camera_info['camera']['intrinsic']['focal'][1]
            ppx = camera_info['camera']['intrinsic']['ppt'][0]
            ppy = camera_info['camera']['intrinsic']['ppt'][1]

            image_path = os.path.join(path, image_list[index])
            image_name = Path(image_path).stem

            image = load_img_rgb(image_path)
            mask_path = os.path.join(path + "/pmasks/", os.path.basename(image_list[index]).replace(os.path.splitext(image_list[index])[-1], ".png"))
            
            if os.path.exists(mask_path):
                img_mask = load_mask_bool(mask_path)
                # if pmask is available, mask the image for PSNR
                image *= img_mask[..., np.newaxis]
            else:
                img_mask = np.ones_like(image[:, :, 0])

            fovx = focal2fov(focal_length_x, image.shape[1])
            fovy = focal2fov(focal_length_y, image.shape[0])
            if int(index) in valid_list:
                image *= img_mask[..., np.newaxis]
                test_cam_infos.append(CameraInfo(uid=index, R=R, T=T, FovY=fovy, FovX=fovx, fx=focal_length_x,
                                                 fy=focal_length_y, cx=ppx, cy=ppy, image=image,
                                                 image_path=image_path, image_name=image_name,
                                                 image_mask=img_mask,
                                                 width=image.shape[1], height=image.shape[0]))
            else:
                image *= img_mask[..., np.newaxis]
                train_cam_infos.append(CameraInfo(uid=index, R=R, T=T, FovY=fovy, FovX=fovx, fx=focal_length_x,
                                                  fy=focal_length_y, cx=ppx, cy=ppy, image=image,
                                                  image_path=image_path, image_name=image_name,
                                                  image_mask=img_mask,
                                                  width=image.shape[1], height=image.shape[0]))
        if debug and i >= 5:
            break

    return train_cam_infos, test_cam_infos, bbox_transform


def readNeILFInfo(path, white_background, eval, debug=False):
    validation_indexes = []
    # if "data_dtu" in path:
    if True:
        if eval:
            validation_indexes = [2, 12, 17, 30, 34]
    else:
        raise NotImplementedError

    print("Reading Training transforms")
    if eval:
        print("Reading Test transforms")

    train_cam_infos, test_cam_infos, bbx_trans = loadCamsFromScene(
        f'{path}/inputs', validation_indexes, white_background, debug)

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = f'{path}/inputs/model/sparse_bbx_scale.ply'
    # if not os.path.exists(ply_path):
    org_ply_path = f'{path}/inputs/model/sparse.ply'

    # scale sparse.ply
    pcd = fetchPly(org_ply_path)
    inv_scale_mat = np.linalg.inv(bbx_trans)  # [4, 4]
    points = pcd.points
    xyz = (np.concatenate([points, np.ones_like(points[:, :1])], axis=-1) @ inv_scale_mat.T)[:, :3]
    normals = pcd.normals
    colors = pcd.colors

    storePly(ply_path, xyz, colors * 255, normals)

    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms2(path, transformsfile, white_background, 
                               extension=".png", benchmark_size = 512, debug=False):
    cam_infos = []
    
    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(tqdm(frames, leave=False)):
            if os.path.exists(os.path.join(path, frame["file_path"] + '.png')):
                image_path = os.path.join(path, frame["file_path"] + '.png')
            else:
                image_path = os.path.join(path, frame["file_path"] + '.exr')
                
            mask_item = frame["file_path"].replace("test", "test_mask").replace("train", "train_mask")
            if os.path.exists(os.path.join(path, mask_item + '.png')):
                mask_path = os.path.join(path, mask_item + '.png')
            else:
                mask_path = os.path.join(path, mask_item + '.exr')
            
            image_name = Path(image_path).stem

            c2w = np.array(frame["transform_matrix"])
            c2w[:3, 1:3] *= -1

            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])
            T = w2c[:3, 3]

            image = load_img_rgb(image_path)
            mask = load_mask_bool(mask_path).astype(np.float32)
            image = cv2.resize(image, (benchmark_size, benchmark_size), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (benchmark_size, benchmark_size), interpolation=cv2.INTER_AREA)

            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])
            image = image * mask[..., None] + bg * (1 - mask[..., None])

            fovy = focal2fov(fov2focal(fovx, image.shape[0]), image.shape[1])
            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=fovy, FovX=fovx, image=image, image_mask=mask,
                                        image_path=image_path, depth=None, normal=None, image_name=image_name,
                                        width=image.shape[1], height=image.shape[0]))

            if debug and idx >= 5:
                break

    return cam_infos


def readStanfordORBInfo(path, white_background, eval, extension=".exr", benchmark_size = 512, debug=False):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms2(path, "transforms_train.json", white_background, 
                                                 extension, benchmark_size, debug=debug)
    if eval:
        print("Reading Test Transforms")
        test_cam_infos = readCamerasFromTransforms2(path, "transforms_test.json", white_background, 
                                                    extension, benchmark_size, debug=debug)
    else:
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if os.path.exists(ply_path):
        os.remove(ply_path)
        
    # Since this data set has no colmap data, we start with random points
    num_pts = 100_000
    print(f"Generating random point cloud ({num_pts})...")

    # We create random points inside the bounds of the synthetic Blender scenes
    xyz = np.random.random((num_pts, 3)) * 1 - 0.5
    # print(np.min(xyz), np.max(xyz))
    shs = np.random.random((num_pts, 3)) / 255.0
    normals = np.random.randn(*xyz.shape)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)

    storePly(ply_path, xyz, SH2RGB(shs) * 255, normals)

    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)

    return scene_info

def readCamerasFromTransforms3(path, transformsfile, white_background, extension=".png", debug=False):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(tqdm(frames, leave=False)):
            image_path = os.path.join(path, frame["file_path"] + extension)
            mask_path = image_path.replace("_rgb.exr", "_mask.png")
            image_name = Path(image_path).stem

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            bg = 1 if white_background else 0
            
            image = load_img_rgb(image_path)
            mask = load_mask_bool(mask_path).astype(np.float32)

            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])
            image = image[..., :3] * mask[..., None] + bg * (1 - mask[..., None])

            fovy = focal2fov(fov2focal(fovx, image.shape[0]), image.shape[1])
            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=fovy, FovX=fovx, image=image, image_mask=mask,
                                        image_path=image_path, image_name=image_name,
                                        width=image.shape[1], height=image.shape[0]))

            if debug and idx >= 5:
                break

    return cam_infos


def readSynthetic4RelightInfo(path, white_background, eval, debug=False):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms3(path, "transforms_train.json", white_background, "_rgb.exr", debug=debug)
    if eval:
        print("Reading Test Transforms")
        test_cam_infos = readCamerasFromTransforms3(path, "transforms_test.json", white_background, "_rgba.png", debug=debug)
    else:
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        normals = np.random.randn(*xyz.shape)
        normals /= np.linalg.norm(normals, axis=-1, keepdims=True)

        storePly(ply_path, xyz, SH2RGB(shs) * 255, normals)

    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)

    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo,
    "Synthetic4Relight": readSynthetic4RelightInfo,
    "NeILF": readNeILFInfo,
    "StanfordORB": readStanfordORBInfo,
}
