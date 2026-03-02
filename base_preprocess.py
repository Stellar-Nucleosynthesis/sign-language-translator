import os
import subprocess
import hashlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

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

def main(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seen_hashes = set()
    unique_files = []

    for filepath in input_dir.glob('*.mp4'):
        file_hash = get_file_hash(filepath)
        if file_hash not in seen_hashes:
            seen_hashes.add(file_hash)
            unique_files.append(filepath)

    with ProcessPoolExecutor() as executor:
        futures = []
        for filepath in unique_files:
            out_path = output_dir / f"processed_{filepath.name}"
            futures.append(executor.submit(process_video, filepath, out_path))

        for future in futures:
            future.result()

if __name__ == '__main__':
    main('/mnt/input_path', '/mnt/output_path')
