import os
import subprocess
import shutil
import numpy as np
import random
from pathlib import Path

INPUT_IMAGE_FOLDER = r""
OUTPUT_DATASET_FOLDER = r""
COLMAP_EXE_PATH = r"COLMAP.bat"  # Path to COLMAP.bat

COLMAP_WORKSPACE = os.path.join(OUTPUT_DATASET_FOLDER, "colmap_workspace")
COLMAP_DB_PATH = os.path.join(COLMAP_WORKSPACE, "database.db")
COLMAP_SPARSE_PATH = os.path.join(COLMAP_WORKSPACE, "sparse")
COLMAP_SPARSE_0_PATH = os.path.join(COLMAP_SPARSE_PATH, "0")

IMGS_FOLDER = os.path.join(OUTPUT_DATASET_FOLDER, "imgs")
TRAIN_FOLDER = os.path.join(OUTPUT_DATASET_FOLDER, "train")
TEST_FOLDER = os.path.join(OUTPUT_DATASET_FOLDER, "test")

TRAIN_INTRINSICS_FOLDER = os.path.join(TRAIN_FOLDER, "intrinsics")
TRAIN_POSES_FOLDER = os.path.join(TRAIN_FOLDER, "pose")

TEST_INTRINSICS_FOLDER = os.path.join(TEST_FOLDER, "intrinsics")
TEST_POSES_FOLDER = os.path.join(TEST_FOLDER, "pose")

TRAIN_TEST_SPLIT = 0.8 

def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def run_colmap_feature_extractor():
    """Run COLMAP feature extraction."""
    print("Running COLMAP feature extraction...")
    cmd = [
        COLMAP_EXE_PATH, "feature_extractor",
        "--database_path", COLMAP_DB_PATH,
        "--image_path", INPUT_IMAGE_FOLDER,
        "--ImageReader.single_camera", "1",  # Treat all images as from the same camera
        "--SiftExtraction.use_gpu", "1",
        "--SiftExtraction.max_num_features", "100000"     # Enable GPU acceleration
    ]
    subprocess.run(cmd, check=True)

def run_colmap_matcher():
    """Run COLMAP exhaustive matcher."""
    print("Running COLMAP matcher...")
    cmd = [
        COLMAP_EXE_PATH, "exhaustive_matcher",
        "--database_path", COLMAP_DB_PATH,
        "--SiftMatching.use_gpu", "1"       # Enable GPU acceleration
    ]
    subprocess.run(cmd, check=True)

def run_colmap_mapper():
    print("Running COLMAP mapper...")
    ensure_directory(COLMAP_SPARSE_PATH)
    cmd = [
        COLMAP_EXE_PATH, "mapper",
        "--database_path", COLMAP_DB_PATH,
        "--image_path", INPUT_IMAGE_FOLDER,
        "--output_path", COLMAP_SPARSE_PATH
    ]
    subprocess.run(cmd, check=True)

def run_colmap_model_converter():
    """Convert COLMAP model from binary to text format."""
    print("Converting COLMAP model to text format...")
    cmd = [
        COLMAP_EXE_PATH, "model_converter",
        "--input_path", COLMAP_SPARSE_0_PATH,
        "--output_path", COLMAP_SPARSE_0_PATH,
        "--output_type", "TXT"
    ]
    subprocess.run(cmd, check=True)

def quaternion_to_rotation_matrix(q):
    w, x, y, z = q
    R = np.zeros((3, 3))
    
    R[0, 0] = 1 - 2*y*y - 2*z*z
    R[0, 1] = 2*x*y - 2*z*w
    R[0, 2] = 2*x*z + 2*y*w
    
    R[1, 0] = 2*x*y + 2*z*w
    R[1, 1] = 1 - 2*x*x - 2*z*z
    R[1, 2] = 2*y*z - 2*x*w
    
    R[2, 0] = 2*x*z - 2*y*w
    R[2, 1] = 2*y*z + 2*x*w
    R[2, 2] = 1 - 2*x*x - 2*y*y
    
    return R

def read_colmap_model():
    print("Reading COLMAP model...")
    
    cameras_txt = os.path.join(COLMAP_SPARSE_0_PATH, "cameras.txt")
    images_txt = os.path.join(COLMAP_SPARSE_0_PATH, "images.txt")
    
    if not (os.path.exists(cameras_txt) and os.path.exists(images_txt)):
        print("Text format files not found. Attempting to convert model...")
        run_colmap_model_converter()
    
    cameras = {}
    with open(cameras_txt, "r") as f:
        for line in f:
            if line.startswith("#") or line.strip() == "":
                continue
            parts = line.strip().split()
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(p) for p in parts[4:]]
            cameras[camera_id] = {
                "model": model,
                "width": width,
                "height": height,
                "params": params
            }
    
    images = {}
    with open(images_txt, "r") as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line == "" or line.startswith("#"):
                i += 1
                continue
                
            parts = line.split()
            if len(parts) < 9:
                i += 1
                continue
                
            image_id = int(parts[0])
            qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
            camera_id = int(parts[8])
            image_name = parts[9]
            
            q = np.array([qw, qx, qy, qz])
            R = quaternion_to_rotation_matrix(q)
            
            # Construct 4x4 transformation matrix
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = np.array([tx, ty, tz])
            
            flip = np.array([
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1]
            ])
            T = flip @ T
            
            images[image_id] = {
                "name": image_name,
                "camera_id": camera_id,
                "transform_matrix": T
            }
            
            i += 2
    
    return cameras, images

def validate_images(images):
    input_images = set(os.listdir(INPUT_IMAGE_FOLDER))
    processed_images = set(img["name"] for img in images.values())
    
    missing_images = input_images - processed_images
    if missing_images:
        print(f"Warning: {len(missing_images)} images were not processed by COLMAP:")
        for img in missing_images:
            print(f" - {img}")
    else:
        print("All images were processed successfully.")

def create_fox_dataset(cameras, images):
    print("Creating dataset...")
    
    ensure_directory(IMGS_FOLDER)
    ensure_directory(TRAIN_FOLDER)
    ensure_directory(TEST_FOLDER)
    ensure_directory(TRAIN_INTRINSICS_FOLDER)
    ensure_directory(TRAIN_POSES_FOLDER)
    ensure_directory(TEST_INTRINSICS_FOLDER)
    ensure_directory(TEST_POSES_FOLDER)
    
    image_list = list(images.values())
    random.shuffle(image_list)
    
    split_idx = int(len(image_list) * TRAIN_TEST_SPLIT)
    train_images = image_list[:split_idx]
    test_images = image_list[split_idx:]
    
    for idx, img_data in enumerate(train_images):
        image_name = img_data["name"]
        base_name = f"train_{idx}"
        camera_id = img_data["camera_id"]
        camera_data = cameras[camera_id]
        
        src_path = os.path.join(INPUT_IMAGE_FOLDER, image_name)
        dst_path = os.path.join(IMGS_FOLDER, f"{base_name}.png")
        shutil.copy2(src_path, dst_path)
        
        write_intrinsics(os.path.join(TRAIN_INTRINSICS_FOLDER, f"{base_name}.txt"), camera_data)
        
        write_pose(os.path.join(TRAIN_POSES_FOLDER, f"{base_name}.txt"), img_data["transform_matrix"])
    
    for idx, img_data in enumerate(test_images):
        image_name = img_data["name"]
        base_name = f"test_{idx}"
        camera_id = img_data["camera_id"]
        camera_data = cameras[camera_id]
        
        src_path = os.path.join(INPUT_IMAGE_FOLDER, image_name)
        dst_path = os.path.join(IMGS_FOLDER, f"{base_name}.png")
        shutil.copy2(src_path, dst_path)
        
        write_intrinsics(os.path.join(TEST_INTRINSICS_FOLDER, f"{base_name}.txt"), camera_data)
        
        write_pose(os.path.join(TEST_POSES_FOLDER, f"{base_name}.txt"), img_data["transform_matrix"])

def write_intrinsics(file_path, camera_data):
    with open(file_path, "w") as f:
        intrinsic_matrix = get_intrinsic_matrix(camera_data)
        for value in intrinsic_matrix.flatten():
            f.write(f"{value}\n")

def write_pose(file_path, transform_matrix):
    with open(file_path, "w") as f:
        for value in transform_matrix.flatten():
            f.write(f"{value}\n")

def get_intrinsic_matrix(camera_data):
    width, height = camera_data["width"], camera_data["height"]
    if camera_data["model"] == "SIMPLE_RADIAL":
        focal = camera_data["params"][0]
        cx, cy = camera_data["params"][1], camera_data["params"][2]
        k1 = camera_data["params"][3]
    else:
        focal = max(width, height)
        cx, cy = width / 2, height / 2
        k1 = 0.0
    
    intrinsic_matrix = np.array([
        [focal, 0.0, cx, 0.0],
        [0.0, focal, cy, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    return intrinsic_matrix

def main():
    print(f"Processing images from {INPUT_IMAGE_FOLDER}")
    print(f"Output dataset will be created at {OUTPUT_DATASET_FOLDER}")
    
    ensure_directory(OUTPUT_DATASET_FOLDER)
    ensure_directory(COLMAP_WORKSPACE)
    
    run_colmap_feature_extractor()
    run_colmap_matcher()
    run_colmap_mapper()
    run_colmap_model_converter()  
    
    cameras, images = read_colmap_model()
    
    validate_images(images)
    
    create_fox_dataset(cameras, images)
    
    print(f"Dataset created successfully at {OUTPUT_DATASET_FOLDER}")
    print("Dataset structure matches the specified format.")

if __name__ == "__main__":
    main()