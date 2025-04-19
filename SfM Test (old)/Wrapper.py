import cv2
import numpy as np
from func import *
import glob
import matplotlib.pyplot as plt

def main():
    # File paths
    image_paths = []
    image_paths.extend(glob.glob('SfM Test\\Data\\Fountain\\*.jpg'))
    image_paths.extend(glob.glob('SfM Test\\Data\\*.png'))
    image_paths.sort()
    calibration_path = 'SfM Test\\Data\\K.txt'

    print(f"Found {len(image_paths)} images")
    
    if len(image_paths) < 2:
        print("Not enough images found in the Data directory. At least two images are required.")
        return

    # Load images and calibration data
    try:
        images = load_images(image_paths)
        print(f"Successfully loaded {len(images)} images")
    except FileNotFoundError as e:
        print(e)
        return

    try:
        K = read_calibration(calibration_path)
        print("Successfully loaded calibration data")
    except Exception as e:
        print(f"Error reading calibration file: {e}")
        return

    print(f"Intrinsic Matrix K:\n{K}\n")

    # Initialize list to hold all 3D points and colors
    all_points_3d = []
    all_colors = []

    # Initialize global rotation and translation
    R_total = np.eye(3)
    t_total = np.zeros((3, 1))

    # Store the previous pose
    previous_R = R_total.copy()
    previous_t = t_total.copy()

    for i in range(len(images) - 1):
        img1 = images[i]
        img2 = images[i + 1]

        print(f"Processing image pair {i + 1} and {i + 2}...")

        # Feature matching
        try:
            src_pts, dst_pts, img_matches = feature_matching_and_display(img1, img2)
            print(f"Feature matching completed. Found {len(src_pts)} matching points.\n")
        except ValueError as e:
            print(f"Feature matching error: {e}")
            continue

        # Recover relative pose
        try:
            R_rel, t_rel, mask_pose = recover_camera_pose(src_pts, dst_pts, K)
            print("Camera pose recovered.\n")
        except ValueError as e:
            print(f"Camera pose recovery error: {e}")
            continue

        # Important: We're keeping the scale information now!
        # Do not normalize t_rel to preserve scale consistency
        # t_rel = t_rel / np.linalg.norm(t_rel)

        # Update global pose
        R_total = R_rel @ previous_R
        t_total = previous_t + previous_R @ t_rel

        # Triangulate points using global poses
        try:
            points_3d = triangulate_points(src_pts, dst_pts, previous_R, previous_t, R_total, t_total, K)
            print(f"Triangulation completed. Generated {points_3d.shape[1]} 3D points.\n")
        except Exception as e:
            print(f"Triangulation error: {e}")
            continue

        # Perform bundle adjustment
        try:
            points_3d_refined, R_total, t_total = bundle_adjustment(points_3d, src_pts, dst_pts, K, R_total, t_total)
            if points_3d_refined is not None:
                points_3d = points_3d_refined
                print("Bundle adjustment completed.\n")
        except Exception as e:
            print(f"Bundle adjustment error: {e}")

        # Extract colors for the matched points in img1
        colors = extract_colors_from_image(img1, src_pts.reshape(-1, 2))

        # Apply lenient filtering to current point set (using False for aggressive flag)
        filtered_points, filtered_colors = filter_outliers(points_3d.T, colors, aggressive=False)
        print(f"Filtered out {points_3d.T.shape[0] - filtered_points.shape[0]} outlier points")
        
        if filtered_points.shape[0] > 0:
            # Append the filtered triangulated points and their colors
            all_points_3d.append(filtered_points)
            all_colors.append(filtered_colors)
        else:
            print("Warning: No points remained after filtering for this image pair.")
            # In case of extreme filtering, keep at least some points
            # Take a random subset of the original points to preserve some structure
            if points_3d.shape[1] > 10:
                indices = np.random.choice(points_3d.shape[1], min(100, points_3d.shape[1]), replace=False)
                all_points_3d.append(points_3d.T[indices])
                all_colors.append(colors[indices])
                print(f"Added {len(indices)} random points to preserve structure.")

        # Update previous pose
        previous_R = R_total.copy()
        previous_t = t_total.copy()

    if not all_points_3d or all(len(pts) == 0 for pts in all_points_3d):
        print("No 3D points were reconstructed. Exiting.")
        return

    # Concatenate all 3D points and colors
    all_points_3d = np.concatenate(all_points_3d, axis=0)
    all_colors = np.concatenate(all_colors, axis=0)

    print(f"Total 3D points reconstructed: {all_points_3d.shape[0]}")
    print(f"Point cloud shape: {all_points_3d.shape}")
    print(f"Color data shape: {all_colors.shape}\n")

    # Apply very lenient global filtering to remove only the most extreme outliers
    final_points, final_colors = filter_outliers(all_points_3d, all_colors, aggressive=False)
    print(f"Final point cloud after global filtering: {final_points.shape[0]} points")

    # Visualize the point cloud
    visualize_point_cloud(final_points)

    # Save both filtered and unfiltered point clouds
    save_point_cloud_as_ply(final_points, final_colors, filename="output_filtered.ply")
    save_point_cloud_as_ply(all_points_3d, all_colors, filename="output_unfiltered.ply")

    print("Structure from Motion pipeline completed successfully.")

if __name__ == '__main__':
    main()