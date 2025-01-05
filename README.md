# 3D Gaussian Splatting from Scratch

References
- https://huggingface.co/datasets/DL3DV/DL3DV-Benchmark
- https://github.com/google-research/multinerf/blob/main/scripts/local_colmap_and_resize.sh
- https://colmap.github.io/cli.html#cli


## 1. Download Dataset (DL3DV-Benchmark)

<details>
<summary>Download Dataset</summary>

First try running without --hash to get .csv file.
Then $hash = scene_hash in .csv file

```bash
python utils/download.py --odir data --subset hash --only_level4 --hash $hash
```
</details>

## 2. Run COLMAP

<details>
<summary>Run with docker</summary>

```bash

docker pull colmap/colmap # cuda=12.6
docker pull colmap/colmap:20240723.601 # cuda<=12.4

docker run --gpus all -it -v $(pwd)/data:/data colmap/colmap:20240723.601
```
</details>

<details>
<summary>Run COLMAP in Docker shell</summary>

```bash
USE_GPU=1
# Dataset path that contains images/
DATASET_PATH=/data
CAMERA=${2:-OPENCV} # or OPENCV_FISHEYE

### Feature extraction
colmap feature_extractor \
    --database_path "$DATASET_PATH"/database.db \
    --image_path "$DATASET_PATH"/images \
    --ImageReader.single_camera 1 \
    --ImageReader.camera_model "$CAMERA" \
    --SiftExtraction.use_gpu "$USE_GPU"


### Feature matching
colmap exhaustive_matcher \
    --database_path "$DATASET_PATH"/database.db \
    --SiftMatching.use_gpu "$USE_GPU"

### Bundle adjustment
mkdir -p "$DATASET_PATH"/sparse
colmap mapper \
    --database_path "$DATASET_PATH"/database.db \
    --image_path "$DATASET_PATH"/images \
    --output_path "$DATASET_PATH"/sparse \
    --Mapper.ba_global_function_tolerance=0.000001
```
</details>


<details>
<summary>Optional Dense Reconstruction (not necessary for this dataset)</summary>

```bash
colmap image_undistorter \
    --image_path $DATASET_PATH/images \
    --input_path $DATASET_PATH/sparse/0 \
    --output_path $DATASET_PATH/dense \
    --output_type COLMAP \
    --max_image_size 2000

colmap patch_match_stereo \
    --workspace_path $DATASET_PATH/dense \
    --workspace_format COLMAP \
    --PatchMatchStereo.geom_consistency true

colmap stereo_fusion \
    --workspace_path $DATASET_PATH/dense \
    --workspace_format COLMAP \
    --input_type geometric \
    --output_path $DATASET_PATH/dense/fused.ply

colmap poisson_mesher \
    --input_path $DATASET_PATH/dense/fused.ply \
    --output_path $DATASET_PATH/dense/meshed-poisson.ply

colmap delaunay_mesher \
    --input_path $DATASET_PATH/dense \
    --output_path $DATASET_PATH/dense/meshed-delaunay.ply
```

</details>

<details>
<summary>Output format of COLMAP</summary>

### cameras.bin
Contains camera intrinsic parameters for each camera:
| Parameter | Description |
|-----------|-------------|
| Camera ID | Unique identifier for each camera |
| Model Type | Camera model (PINHOLE, SIMPLE_RADIAL, etc.) |
| Dimensions | Image width and height |
| Calibration | Focal length, principal point, distortion coefficients |

### images.bin
Stores camera extrinsic parameters (poses) for each image:
| Parameter | Description |
|-----------|-------------|
| Image ID | Unique identifier for each image |
| Rotation | Camera orientation as quaternion |
| Translation | Camera position vector |
| Camera ID | Reference to corresponding camera in cameras.bin |
| Filename | Name of the source image file |

### points3D.bin
Contains the reconstructed 3D point cloud data:
| Parameter | Description |
|-----------|-------------|
| Point ID | Unique identifier for each 3D point |
| Coordinates | XYZ position in world space |
| Color | RGB color values |
| Error | Reprojection error |
| Track Length | Number of images observing this point |
| Track Info | List of image IDs and feature IDs that observed this point |

</details>