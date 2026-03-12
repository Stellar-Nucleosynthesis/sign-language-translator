import numpy as np
from pathlib import Path
from collections import Counter

def analyze_npy_shapes(folder):
    folder = Path(folder)

    shape_counter = Counter()
    total_files = 0
    failed_files = []

    for npy_file in folder.glob("*.npy"):
        try:
            arr = np.load(npy_file)
            shape_counter[arr.shape] += 1
            total_files += 1
        except Exception as e:
            failed_files.append((npy_file, str(e)))

    print(f"Total .npy files processed: {total_files}\n")

    print("Shapes distribution:")
    for shape, count in shape_counter.most_common():
        print(f"{shape}: {count}")

    if failed_files:
        print("\nFiles that failed to load:")
        for f, err in failed_files:
            print(f"{f} -> {err}")

if __name__ == "__main__":
    folder_path = "folder/"
    analyze_npy_shapes(folder_path)