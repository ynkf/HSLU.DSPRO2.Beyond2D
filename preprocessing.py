import cv2
import numpy as np
import os
from tqdm import tqdm
from scipy.ndimage import binary_closing, binary_opening, binary_erosion
from multiprocessing import Pool, cpu_count

BASE_DATA_PATH = './data/DeepFurniture/uncompressed_data/scenes'
ORIGINAL_DEPTH_FILENAME = 'depth.png'
CORRECTED_DEPTH_FILENAME = 'depth_corrected.png'

THRESHOLD_PERCENTILE = 95 # 95 means top 5% of depth values.
CLEANING_ITERATIONS = 5
EROSION_ITERATIONS = 10

def process_folder(folder_name):
    try:
        folder_path = os.path.join(BASE_DATA_PATH, folder_name)
        depth_path = os.path.join(folder_path, ORIGINAL_DEPTH_FILENAME)
        corrected_depth_path = os.path.join(folder_path, CORRECTED_DEPTH_FILENAME)

        if not os.path.exists(depth_path):
            return

        depth_map = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

        if depth_map is None or depth_map.size == 0:
            return

        valid_depths = depth_map[depth_map > 0]
        if valid_depths.size == 0:
            cv2.imwrite(corrected_depth_path, depth_map)
            return

        depth_threshold = np.percentile(valid_depths, THRESHOLD_PERCENTILE)
        max_value = np.max(valid_depths)
        extreme_depth_mask = (depth_map > depth_threshold) | (depth_map == max_value)

        temp_mask = binary_closing(extreme_depth_mask, iterations=CLEANING_ITERATIONS)
        window_mask = binary_opening(temp_mask, iterations=CLEANING_ITERATIONS).astype(bool)

        if not np.any(window_mask):
            cv2.imwrite(corrected_depth_path, depth_map)
            return

        # Correction
        eroded_mask = binary_erosion(window_mask, iterations=EROSION_ITERATIONS)
        boundary_mask = window_mask & ~eroded_mask

        valid_wall_pixels = depth_map[boundary_mask & (depth_map > 0)]

        if len(valid_wall_pixels) == 0:
            corrected_depth = depth_map
        else:
            wall_depth_value = np.median(valid_wall_pixels)
            corrected_depth = depth_map.copy()
            corrected_depth[window_mask] = wall_depth_value

        cv2.imwrite(corrected_depth_path, corrected_depth)
        return f"Processed {folder_name}"
    except Exception as e:
        return f"Error processing {folder_name}: {e}"


if __name__ == '__main__':
    folder_names = [f for f in os.listdir(BASE_DATA_PATH) if os.path.isdir(os.path.join(BASE_DATA_PATH, f))]

    # Use almost all available CPU cores to process images in parallel
    num_processes = max(1, cpu_count() - 1)

    print(f"Found {len(folder_names)} scene folders. Starting processing with {num_processes} parallel processes...")

    with Pool(processes=num_processes) as pool:
        list(tqdm(pool.imap_unordered(process_folder, folder_names), total=len(folder_names)))

    print(f"\n✅ Finished. Check for '{CORRECTED_DEPTH_FILENAME}' in each subfolder.")