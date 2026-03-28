import os
import numpy as np
from pathlib import Path

SELECTED_INDICES = np.concatenate([np.array([2, 3, 4, 5]), np.arange(138, 180)])
NUM_RELEVANT_POINTS = len(SELECTED_INDICES)

def filter_keypoints(input_dir, output_dir):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    processed_count = 0
    error_count = 0
    
    for npy_file in input_path.glob('*.npy'):
        try:
            keypoints = np.load(npy_file)
            frames, flat_keypoints = keypoints.shape
            
            keypoints = keypoints.reshape(frames, flat_keypoints // 3, 3)
            relevant_keypoints = keypoints[:, SELECTED_INDICES, :]
            
            filtered_flat_keypoints = NUM_RELEVANT_POINTS * 3
            relevant_keypoints = relevant_keypoints.reshape(frames, filtered_flat_keypoints)
            
            output_file_path = output_path / npy_file.name
            np.save(output_file_path, relevant_keypoints)
            processed_count += 1
            
        except Exception as e:
            error_count += 1

    print(f"[SUCCESS] Directory {input_dir} processed.")
    print(f"[INFO] Files saved: {processed_count} | Errors: {error_count}")
    print(f"[INFO] New feature size: {filtered_flat_keypoints} (from {flat_keypoints})")
    print("-" * 50)

if __name__ == '__main__':
    train_input = '/mnt/c/Workstudy/CV/keypoints_test_540'
    train_output = '/mnt/c/Workstudy/CV/keypoints_test_filtered'
    print(f"[INFO] Processing Train dataset...")
    filter_keypoints(train_input, train_output)
    
    val_input = '/mnt/c/Workstudy/CV/keypoints_val_540'
    val_output = '/mnt/c/Workstudy/CV/keypoints_val_filtered'
    print(f"[INFO] Processing Validation dataset...")
    filter_keypoints(val_input, val_output)