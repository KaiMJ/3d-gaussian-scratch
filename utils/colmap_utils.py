import struct
import numpy as np


def read_cameras_binary(file_path):
    """
    Read COLMAP camera binary file (cameras.bin).
    Returns a dictionary mapping camera_id to a dictionary containing camera parameters.
    """
    cameras = {}
    with open(file_path, "rb") as f:
        num_cameras = struct.unpack("Q", f.read(8))[0]
        for _ in range(num_cameras):
            camera_id = struct.unpack("i", f.read(4))[0]
            model_id = struct.unpack("i", f.read(4))[0]
            width = struct.unpack("Q", f.read(8))[0]
            height = struct.unpack("Q", f.read(8))[0]
            params = []
            num_params = {
                0: 3,  # SIMPLE_PINHOLE
                1: 4,  # PINHOLE
                2: 5,  # SIMPLE_RADIAL
                3: 6,  # RADIAL
                4: 8,  # OPENCV
                5: 8,  # OPENCV_FISHEYE
                6: 12,  # FULL_OPENCV
                7: 3,  # SIMPLE_RADIAL_FISHEYE
                8: 4,  # RADIAL_FISHEYE
                9: 5,  # THIN_PRISM_FISHEYE
            }[model_id]
            for _ in range(num_params):
                params.append(struct.unpack("d", f.read(8))[0])

            cameras[camera_id] = {
                "model_id": model_id,
                "width": width,
                "height": height,
                "params": params
            }
    return cameras


def read_points3D_binary(file_path):
    points3D = {}
    with open(file_path, "rb") as f:
        num_points = struct.unpack("Q", f.read(8))[0]
        for _ in range(num_points):
            point_id = struct.unpack("Q", f.read(8))[0]
            xyz = struct.unpack("ddd", f.read(24))  # X, Y, Z
            rgb = struct.unpack("BBB", f.read(3))   # R, G, B
            error = struct.unpack("d", f.read(8))[0]
            track_length = struct.unpack("Q", f.read(8))[0]
            track = []
            for _ in range(track_length):
                image_id, feature_id = struct.unpack("ii", f.read(8))
                track.append((image_id, feature_id))
            points3D[point_id] = {
                "xyz": np.array(xyz),
                "rgb": np.array(rgb),
                "error": error,
                "track": track
            }
    return points3D


def read_images_binary(images_file):
    images = {}
    
    with open(images_file, "rb") as fid:
        num_images = struct.unpack("Q", fid.read(8))[0]
        
        for _ in range(num_images):
            image_id = struct.unpack("i", fid.read(4))[0]
            qw, qx, qy, qz = struct.unpack("dddd", fid.read(32))
            tx, ty, tz = struct.unpack("ddd", fid.read(24))
            camera_id = struct.unpack("i", fid.read(4))[0]
            
            name = ""
            name_char = struct.unpack("c", fid.read(1))[0]
            while name_char != b"\x00":
                name += name_char.decode("utf-8")
                name_char = struct.unpack("c", fid.read(1))[0]
            
            num_points2D = struct.unpack("Q", fid.read(8))[0]
            points2D = []
            for _ in range(num_points2D):
                x, y = struct.unpack("dd", fid.read(16))
                point3D_id = struct.unpack("q", fid.read(8))[0]
                points2D.append([x, y, point3D_id])

            images[image_id] = {
                "qvec": [qw, qx, qy, qz],
                "tvec": [tx, ty, tz],
                "camera_id": camera_id,
                "name": name,
                "points2D": points2D
            }
            
    return images


def qvec2rotmat(qvec):
    """Convert quaternion to rotation matrix"""
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


def create_transforms_json(cameras_data, images_data):
    """Convert COLMAP data to transforms.json format"""
    frames = []

    for image_id, image_data in images_data.items():
        camera = cameras_data[image_data["camera_id"]]

        # Get rotation and translation
        R = qvec2rotmat(image_data["qvec"])
        t = image_data["tvec"]

        # Convert to 4x4 transformation matrix
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = R
        transform_matrix[:3, 3] = t

        # Get camera intrinsics
        fx, fy, cx, cy = camera["params"][:4]  # Assuming PINHOLE camera model

        frame = {
            "file_path": image_data["name"],
            "transform_matrix": transform_matrix.tolist(),
            "camera_angle_x": 2 * np.arctan(camera["width"] / (2 * fx)),
            "camera_angle_y": 2 * np.arctan(camera["height"] / (2 * fy)),
            "fl_x": fx,
            "fl_y": fy,
            "cx": cx,
            "cy": cy,
            "w": camera["width"],
            "h": camera["height"],
        }
        frames.append(frame)

    transforms_dict = {
        "camera_angle_x": frames[0]["camera_angle_x"],
        "camera_angle_y": frames[0]["camera_angle_y"],
        "fl_x": frames[0]["fl_x"],
        "fl_y": frames[0]["fl_y"],
        "cx": frames[0]["cx"],
        "cy": frames[0]["cy"],
        "w": frames[0]["w"],
        "h": frames[0]["h"],
        "frames": frames
    }

    return transforms_dict
