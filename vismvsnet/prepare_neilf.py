
import os
import json
import imageio
import numpy as np
import logging
import pyexr
import struct
from plyfile import PlyData, PlyElement


def mkdir_if_not_exist(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_points3D_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """

    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]

        xyzs = np.empty((num_points, 3))
        rgbs = np.empty((num_points, 3))
        errors = np.empty((num_points, 1))

        for p_id in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd")
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(
                fid, num_bytes=8 * track_length,
                format_char_sequence="ii" * track_length)
            xyzs[p_id] = xyz
            rgbs[p_id] = rgb
            errors[p_id] = error
    return xyzs, rgbs, errors

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

class ViewSelection:
    def __init__(self, data_root):
        self.image_dir = f"{data_root}/inputs/images"
        self.model_dir = f"{data_root}/inputs/model"
        self.json_file = f"{data_root}/inputs/sfm_scene.json"
        
        self.valid_list = []
        if "data_dtu" in data_root:
            self.valid_list = [2, 12, 17, 30, 34]
        elif "data_hdr" in data_root:
            self.valid_list = [9, 18, 30, 41, 50, 62, 73, 82, 94]
        elif "data_synthetic" in data_root:
            self.valid_list = [9, 18, 30, 41, 50, 62, 73, 82, 94]
        
        self.info_list = self.load_scene_json(self.json_file)        
        for v in self.valid_list:
            self.info_list.pop(v)
        
        self.info_list = sorted(self.info_list.items(), key=lambda v:v[0])
        # for i in range(self.info_list.__len__()):
        #     print(self.info_list[i][1]["image_path"])
        
        self.colmap_workspace = f"{data_root}/colmap"
        self.colmap_image_dir = f"{data_root}/colmap/images"
        self.colmap_manual_model_dir = f"{data_root}/colmap/manual" 
        self.colmap_sparse_dir = f"{data_root}/colmap/sparse" 
        mkdir_if_not_exist(self.colmap_workspace)
        mkdir_if_not_exist(self.colmap_image_dir)
        mkdir_if_not_exist(self.colmap_manual_model_dir)
        mkdir_if_not_exist(self.colmap_sparse_dir)
        
    @ staticmethod
    def load_scene_json(json_file):
        with open(json_file, "r") as f:
            sfm_scene = json.load(f)
    
        # load bbox transform
        # bbox_transform = np.array(sfm_scene['bbox']['transform']).reshape(4, 4)

        # meta info
        image_list = sfm_scene['image_path']['file_paths']

        # camera parameters
        info_list = dict()
        camera_info_list = sfm_scene['camera_track_map']['images']
        for i, (index, camera_info) in enumerate(camera_info_list.items()):
            # flg == 2 stands for valid camera 
            if camera_info['flg'] == 2:
                size = camera_info["size"]
                intrinsic = np.zeros((4, 4))
                intrinsic[0, 0] = camera_info['camera']['intrinsic']['focal'][0]
                intrinsic[1, 1] = camera_info['camera']['intrinsic']['focal'][1]
                intrinsic[0, 2] = camera_info['camera']['intrinsic']['ppt'][0]
                intrinsic[1, 2] = camera_info['camera']['intrinsic']['ppt'][1]
                intrinsic[2, 2] = intrinsic[3, 3] = 1
                extrinsic = np.array(camera_info['camera']['extrinsic']).reshape(4, 4)

                image_path = os.path.join(os.path.dirname(json_file), image_list[index])

                info_list[int(index)] = {
                    "intrinsic": intrinsic,
                    "extrinsic": extrinsic,
                    "image_path": image_path,
                    "size": size,
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
        
        # bin to ply
        bin_path = self.colmap_sparse_dir + "/points3D.bin"
        output_path = self.model_dir + "/sparse.ply"
        xyz, rgb, _ = read_points3D_binary(bin_path)
        storePly(output_path, xyz, rgb)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, help='The root dir of the data.')
    args = parser.parse_args()

    data_root = args.data_root
    view_select = ViewSelection(data_root)
    view_select.run_colmap()