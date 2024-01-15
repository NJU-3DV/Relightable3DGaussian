
import os
import json
import imageio
import numpy as np
import logging
import pyexr
import math


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def mkdir_if_not_exist(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

class ViewSelection:
    def __init__(self, data_root, json_file):
        self.image_dir = data_root
        self.json_file = json_file 
        
        self.info_list = self.load_scene_json(self.image_dir, self.json_file)
        self.info_list = sorted(self.info_list.items(), key=lambda v:v[1]['image_name'])
        
        self.colmap_workspace = f"{data_root}/colmap"
        self.colmap_image_dir = f"{data_root}/colmap/images"
        self.colmap_manual_model_dir = f"{data_root}/colmap/manual" 
        self.colmap_sparse_dir = f"{data_root}/colmap/sparse" 
        mkdir_if_not_exist(self.colmap_workspace)
        mkdir_if_not_exist(self.colmap_image_dir)
        mkdir_if_not_exist(self.colmap_manual_model_dir)
        mkdir_if_not_exist(self.colmap_sparse_dir)
        
    def load_scene_json(self, path, json_file, extension=".png"):
        with open(json_file, "r") as f:
            contents = json.load(f)
            fovx = contents["camera_angle_x"]

            frames = contents["frames"]

            info_list = dict()
            for idx, frame in enumerate(frames):
                image_path = os.path.join(path, frame["file_path"] + extension)
                image_name = os.path.basename(image_path)

                # NeRF 'transform_matrix' is a camera-to-world transform
                c2w = np.array(frame["transform_matrix"])
                # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
                c2w[:3, 1:3] *= -1

                # get the world-to-camera transform and set R, T
                w2c = np.linalg.inv(c2w)
                R = w2c[:3, :3]
                T = w2c[:3, 3]

                img, is_hdr = self.load_img(image_path)

                img_width = img.shape[1]
                img_height = img.shape[0]

                fovy = focal2fov(fov2focal(fovx, img_width), img_height)

                intrinsic = np.zeros((4, 4))
                focal_x = img_width / (2 * np.tan(fovx * 0.5))
                focal_y = img_height / (2 * np.tan(fovy * 0.5))

                intrinsic[0, 0] = focal_x
                intrinsic[1, 1] = focal_y
                intrinsic[0, 2] = img_width / 2
                intrinsic[1, 2] = img_height / 2
                intrinsic[2, 2] = intrinsic[3, 3] = 1

                extrinsic = np.zeros((4, 4))
                extrinsic[:3, :3] = R
                extrinsic[:3, 3] = T
                extrinsic[3, 3] = 1

                info_list[int(idx)] = {
                    "intrinsic": intrinsic,
                    "extrinsic": extrinsic,
                    "image_path": image_path,
                    "size": [img_width, img_height],
                    "image_name": image_name
                }
        
        return info_list
    
    @ staticmethod
    def load_img(path):
        if path.endswith(".exr"):
            exr_file = pyexr.open(path)
            # print(exr_file.channels)
            all_data = exr_file.get()
            img = all_data[..., 0:3]

            if "A" in exr_file.channels:    
                mask = np.clip(all_data[..., 3:4], 0, 1)
                img = img * mask
            
            img = np.nan_to_num(img)
            hdr = True
        else:  # LDR image
            img = imageio.imread(path)
            img = img / 255
            # img[..., 0:3] = srgb_to_rgb_np(img[..., 0:3])
            hdr = False
        return img, hdr
    
    @ staticmethod
    def hdr2ldr(img, scale=0.666667):
        img = img * scale
        # img = 1 - np.exp(-3.0543 * img)  # Filmic
        img = (img * (2.51 * img + 0.03)) / (img * (2.43 * img + 0.59) + 0.14)  # ACES
        return img
    
    def create_known_camera(self):
        image_txt_file = f"{self.colmap_manual_model_dir}/images.txt"
        camera_txt_file = f"{self.colmap_manual_model_dir}/cameras.txt"
        points_txt_file = f"{self.colmap_manual_model_dir}/points3D.txt"

        with open(image_txt_file, "w") as f:
            text = "# Image list with two lines of data per image:\n"
            text += "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
            text += "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
            text += "# Number of images: {}, mean observations per image: 0\n".format(self.info_list.__len__())

            def rotmat2qvec(R):
                Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
                K = np.array([
                    [Rxx - Ryy - Rzz, 0, 0, 0],
                    [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                    [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                    [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
                eigvals, eigvecs = np.linalg.eigh(K)
                qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
                if qvec[0] < 0:
                    qvec *= -1
                return qvec
            
            colmap_id = 0
            # print(self.info_list)
            for info in self.info_list:
                # print(info[1]["image_path"])
                v = info[1]["extrinsic"]
                qvec = rotmat2qvec(v[0:3, 0:3])
                tvec = v[0:3, -1]

                img_id = colmap_id + 1

                dst_name = os.path.basename(info[1]["image_path"])
                dst_name = dst_name.replace(os.path.splitext(dst_name)[-1], ".png")

                text += "{0} {1} {2} {3} {4} {5} {6} {7} {8} {9}\n".format(
                    img_id, qvec[0], qvec[1], qvec[2], qvec[3], tvec[0], 
                    tvec[1], tvec[2], img_id, dst_name)
                text += "\n"

                colmap_id += 1
            
            f.write(text)
        
        """
        # Camera list with one line of data per camera:
        #   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
        # Number of cameras: 49
        """
        with open(camera_txt_file, "w") as f:
            text = "# Camera list with one line of data per camera:\n"
            text += "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
            text += "# Number of cameras: {}\n".format(self.info_list.__len__())

            colmap_id = 0
            for info in self.info_list:
                v = info[1]["intrinsic"]
                img_id = colmap_id + 1
                # print(v)
                text += "{} PINHOLE {} {} {} {} {} {}\n".format(
                    img_id, info[1]["size"][0], info[1]["size"][1], v[0][0], v[1][1], v[0][2], v[1][2])
                
                colmap_id += 1

            f.write(text)
        
        with open(points_txt_file, "w") as f:
            pass

    def run_colmap(self):
        # prepare data
        for info in self.info_list:
            src_img_file = info[1]["image_path"]
            
            img, is_hdr = self.load_img(src_img_file)
            if is_hdr:
                img = self.hdr2ldr(img)

            dst_name = os.path.basename(src_img_file)
            dst_name = dst_name.replace(os.path.splitext(dst_name)[-1], ".png")
            dst_img_file = f"{self.colmap_image_dir}/{dst_name}"

            img = (img * 255).astype(np.uint8)
            imageio.imwrite(dst_img_file, img)

        # run feature extraction
        feat_extracton_cmd = "colmap feature_extractor "\
            "--database_path " + self.colmap_workspace + "/database.db \
            --image_path " + self.colmap_image_dir + " \
            --ImageReader.camera_model PINHOLE \
            --SiftExtraction.use_gpu 1"

        exit_code = os.system(feat_extracton_cmd)
        if exit_code != 0:
            logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
            exit(exit_code)
        
        # run feature matching
        feat_matching_cmd = "colmap exhaustive_matcher " \
            "--database_path " + self.colmap_workspace + "/database.db \
            --SiftMatching.use_gpu 1"
        exit_code = os.system(feat_matching_cmd)
        if exit_code != 0:
            logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
            exit(exit_code)
        
        # create model manually
        self.create_known_camera()

        triangulator_cmd = "colmap point_triangulator "\
            "--database_path " + self.colmap_workspace + "/database.db \
            --image_path " + self.colmap_image_dir + \
            " --input_path " + self.colmap_manual_model_dir + \
            " --output_path " + self.colmap_sparse_dir
        exit_code = os.system(triangulator_cmd)
        if exit_code != 0:
            logging.error(f"Point triangulator failed with code {exit_code}. Exiting.")
            exit(exit_code)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, help='The root dir of the data.')
    parser.add_argument('--transform_json_file', type=str, help='The root dir of the data.')
    args = parser.parse_args()

    data_root = args.data_root
    json_file = args.transform_json_file
    view_select = ViewSelection(data_root, json_file)
    view_select.run_colmap()
    
    '''
    python prepare_nerf.py --data_root /home/gj/Project/3DG_material/datasets/nerf/lego --transform_json_file /home/gj/Project/3DG_material/datasets/nerf/lego/transforms_train.json
    '''