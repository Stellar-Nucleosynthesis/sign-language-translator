import os
import numpy as np
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path='reduce_npy.env')
TRAIN_INPUT = os.getenv("TRAIN_INPUT")
TRAIN_OUTPUT = os.getenv("TRAIN_OUTPUT")
TEST_INPUT = os.getenv("TEST_INPUT")
TEST_OUTPUT = os.getenv("TEST_OUTPUT")
VAL_INPUT = os.getenv("VAL_INPUT")
VAL_OUTPUT = os.getenv("VAL_OUTPUT")

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
            
        except Exception:
            error_count += 1

    print(f"[SUCCESS] Directory {input_dir} processed.")
    print(f"[INFO] Files saved: {processed_count} | Errors: {error_count}")
    print(f"[INFO] New feature size: {filtered_flat_keypoints} (from {flat_keypoints})")
    print("-" * 50)

if __name__ == '__main__':
    print(f"[INFO] Processing Train dataset...")
    filter_keypoints(TRAIN_INPUT, TRAIN_OUTPUT)

    print(f"[INFO] Processing Test dataset...")
    filter_keypoints(TEST_INPUT, TEST_OUTPUT)

    print(f"[INFO] Processing Validation dataset...")
    filter_keypoints(VAL_INPUT, VAL_OUTPUT)