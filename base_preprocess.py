import os
import subprocess
import hashlib
import json
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys

def get_file_hash(filepath):
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()

def process_video(input_path, output_path):
    command = [
        'ffmpeg',
        '-y',
        '-hwaccel', 'cuda',
        '-i', str(input_path),
        '-vf', 'fps=10,scale=256:256',
        '-c:v', 'h264_nvenc',
        '-preset', 'fast',
        str(output_path)
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_path

def balance_classes(json_path, target_dir):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    labels = [item.get('text', item.get('label', 'unknown')) for item in data]
    counts = Counter(labels)
    
    total_samples = len(labels)
    unique_classes = len(counts)
    
    if unique_classes == 0:
        return
        
    max_allowed = int(total_samples / unique_classes) + 2
    class_tracker = {label: 0 for label in counts.keys()}

    for video_path in Path(target_dir).glob('*.mp4'):
        match = re.search(r'(\d+)\.mp4$', video_path.name)
        if not match:
            continue
            
        idx = int(match.group(1))
        if idx >= len(data):
            continue
            
        label = data[idx].get('text', data[idx].get('label', 'unknown'))
        
        if class_tracker[label] < max_allowed:
            class_tracker[label] += 1
        else:
            try:
                os.remove(video_path)
            except OSError:
                pass

def main(json_path, input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Hashing files to find duplicates...")
    seen_hashes = set()
    unique_files = []
    all_files = list(input_dir.glob('*.mp4'))

    for i, filepath in enumerate(all_files, 1):
        file_hash = get_file_hash(filepath)
        if file_hash not in seen_hashes:
            seen_hashes.add(file_hash)
            unique_files.append(filepath)
        sys.stdout.write(f"\rHashed: {i}/{len(all_files)}")
        sys.stdout.flush()

    total_unique = len(unique_files)
    print(f"\nFound {total_unique} unique videos. Starting FFMPEG processing...")

    processed = 0

    with ProcessPoolExecutor() as executor:
        futures = []
        for filepath in unique_files:
            out_path = output_dir / f"processed_{filepath.name}"
            futures.append(executor.submit(process_video, filepath, out_path))

        for future in as_completed(futures):
            future.result()
            processed += 1
            sys.stdout.write(f"\rProcessed: {processed}/{total_unique}")
            sys.stdout.flush()

    print("\nBalancing classes...")
    balance_classes(json_path, output_dir)
    print("Done!")

if __name__ == '__main__':
    main(
        '/mnt/c/Workstudy/CV/data/annotations/MSASL_val.json',
        '/mnt/c/Workstudy/CV/data/videos_val',
        '/mnt/c/Workstudy/CV/processed_videos_val'
    )
