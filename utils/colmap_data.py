import os
from utils.colmap_utils import read_cameras_binary, read_images_binary, read_points3D_binary
from utils.matrix import quaternion_to_rotation_vectorized
import numpy as np

class COLMAP_Data:
    def __init__(self, root_dir):
        self.root_dir = root_dir

        self.cameras_file = os.path.join(self.root_dir, "cameras.bin")
        self.images_file = os.path.join(self.root_dir, "images.bin")
        self.points3D_file = os.path.join(self.root_dir, "points3D.bin")

        self.cameras_data = read_cameras_binary(self.cameras_file)
        self.images_data = read_images_binary(self.images_file)
        self.points3D_data = read_points3D_binary(self.points3D_file)

        self.points, self.colors = self.get_points_colors()

        self.frame_names = [v['name'] for v in self.images_data.values()]

        self.width, self.height = self.cameras_data[1]["width"], self.cameras_data[1]["height"]

        self.K, self.distortion = self.get_intrinsics() # K: (3, 3), distortion: (4,)
        self.E = self.get_extrinsics()

    def get_intrinsics(self):
        """
        instrincis from COLMAP: [fx, fy, cx, cy, s, k1, k2, k3]

        K = [fx 0 cx
            0 fy cy
            0 0 1]
        
        distortion = [s, k1, k2, k3]
        """
        intrinsics = np.array(self.cameras_data[1]["params"])
        K = np.array([[intrinsics[0], 0, intrinsics[2]],
                    [0, intrinsics[1], intrinsics[3]],
                    [0, 0, 1]])
        distortion = intrinsics[4:]
        return K, distortion
    
    def get_extrinsics(self):
        """
        E = [R t
            0 1]

        returns {frame_name: E}
        Extrinsic: camera to world
        """
        qvecs = np.array([v['qvec'] for v in self.images_data.values()])
        tvecs = np.array([v['tvec'] for v in self.images_data.values()])

        rotation = quaternion_to_rotation_vectorized(qvecs)
        w2c = np.zeros((len(self.images_data), 4, 4))
        w2c[:, :3, :3] = rotation
        w2c[:, :3, 3] = tvecs
        w2c[:, 3, 3] = 1
        c2w = np.linalg.inv(w2c).transpose(0, 2, 1)

        E = {k: c2w[i] for i, k in enumerate(self.frame_names)}
        return E

    def get_points_colors(self):
        """
            Not noticing significant difference between sparse / dense (from original gaussian-splatting repository)
        """
        points = np.array([v["xyz"] for v in self.points3D_data.values()])
        colors = np.array([v["rgb"] / 255.0 for v in self.points3D_data.values()])
        return points, colors

