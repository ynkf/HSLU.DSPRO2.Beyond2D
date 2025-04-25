import numpy as np
import os
from pathlib import Path

def quaternion_to_rotation_matrix(q):
    """Convert quaternion to rotation matrix"""
    qw, qx, qy, qz = q
    R = np.zeros((3, 3))
    
    R[0, 0] = 1 - 2*qy**2 - 2*qz**2
    R[0, 1] = 2*qx*qy - 2*qz*qw
    R[0, 2] = 2*qx*qz + 2*qy*qw
    
    R[1, 0] = 2*qx*qy + 2*qz*qw
    R[1, 1] = 1 - 2*qx**2 - 2*qz**2
    R[1, 2] = 2*qy*qz - 2*qx*qw
    
    R[2, 0] = 2*qx*qz - 2*qy*qw
    R[2, 1] = 2*qy*qz + 2*qx*qw
    R[2, 2] = 1 - 2*qx**2 - 2*qy**2
    
    return R

def process_kaggle_dataset(cameras_file, images_file, output_dir):
    # Load camera intrinsics
    camera_dict = {}
    with open(cameras_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) < 8:
                continue
                
            camera_id = parts[0]
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            
            if model == 'SIMPLE_RADIAL':
                focal = float(parts[4])
                cx = float(parts[5])
                cy = float(parts[6])
                distortion = float(parts[7])
                
                # Create K matrix
                K = np.array([
                    [focal, 0, cx],
                    [0, focal, cy],
                    [0, 0, 1]
                ])
                
                camera_dict[camera_id] = {
                    'K': K,
                    'width': width,
                    'height': height
                }
    
    # Write global K.txt file
    if len(camera_dict) > 0:
        # Just use the first camera's K matrix for simplicity
        first_cam_id = list(camera_dict.keys())[0]
        K = camera_dict[first_cam_id]['K']
        
        with open(os.path.join(output_dir, 'K.txt'), 'w') as f:
            for i in range(3):
                f.write(' '.join(map(str, K[i])) + ' \n')
    
    # Process image extrinsics
    is_image_header = True
    current_image_id = None
    
    with open(images_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
                
            parts = line.strip().split()
            
            if is_image_header and len(parts) >= 10:
                # This is an image header line
                image_id = parts[0]
                qw, qx, qy, qz = map(float, parts[1:5])
                tx, ty, tz = map(float, parts[5:8])
                camera_id = parts[8]
                image_name = parts[9]
                
                if camera_id not in camera_dict:
                    print(f"Warning: Camera {camera_id} not found for image {image_id}")
                    continue
                    
                # Get camera intrinsics
                K = camera_dict[camera_id]['K']
                width = camera_dict[camera_id]['width']
                height = camera_dict[camera_id]['height']
                
                # Convert quaternion to rotation matrix
                R = quaternion_to_rotation_matrix((qw, qx, qy, qz))
                
                # Create output file path
                output_path = os.path.join(output_dir, f"{image_name.replace('.png', '.jpg')}.camera")
                
                # Write .jpg.camera file
                with open(output_path, 'w') as out_f:
                    # Intrinsic parameters
                    for i in range(3):
                        out_f.write(' '.join(map(str, K[i])) + ' \n')
                    
                    # Zero line
                    out_f.write('0 0 0\n')
                    
                    # Rotation matrix
                    for i in range(3):
                        out_f.write(' '.join(map(str, R[i])) + ' \n')
                    
                    # Translation vector
                    out_f.write(f"{tx} {ty} {tz} \n")
                    
                    # Image dimensions
                    out_f.write(f"{width} {height}")
                
                is_image_header = False
                current_image_id = image_id
            else:
                # This is a feature line, we're ignoring feature points for now
                is_image_header = True
                current_image_id = None

# Usage
input_dir = 'C:\\Users\\fabia\\Desktop\\dioscuri\\sfm'
output_dir = 'C:\\Users\\fabia\\Desktop\\test_1\\test_conv_scripts'

cameras_file = os.path.join(input_dir, 'cameras.txt')
images_file = os.path.join(input_dir, 'images.txt')

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

process_kaggle_dataset(cameras_file, images_file, output_dir)