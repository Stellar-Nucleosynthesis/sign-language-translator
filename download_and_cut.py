import json
import os
import subprocess
import concurrent.futures
import threading
from yt_dlp import YoutubeDL

DATASET_JSON = "/mnt/c/Workstudy/CV/data/annotations/MSASL_test.json"
RAW_VIDEO_DIR = "raw_videos"
CLIP_DIR = "videos_test"

os.makedirs(RAW_VIDEO_DIR, exist_ok=True)
os.makedirs(CLIP_DIR, exist_ok=True)

YDL_OPTS = {
    "format": "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4]",
    "merge_output_format": "mp4",
    "quiet": True,
    "no_warnings": True,
    "retries": 3,
}

cleanup_lock = threading.Lock()
counter_lock = threading.Lock()

ok = 0
dead = 0

def cleanup_raw_videos():
    with cleanup_lock:
        files = [os.path.join(RAW_VIDEO_DIR, f) for f in os.listdir(RAW_VIDEO_DIR) if f.endswith('.mp4')]
        if len(files) > 50:
            files.sort(key=os.path.getmtime)
            for f in files[:-50]:
                try:
                    os.remove(f)
                except OSError:
                    pass

def download_video(url, out_path):
    if os.path.exists(out_path):
        return True

    ydl_opts = YDL_OPTS | {"outtmpl": out_path}

    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return os.path.exists(out_path)
    except Exception:
        return False

def cut_clip(video_path, start, end, out_path):
    if not os.path.exists(video_path):
        return False

    cmd = [
        "ffmpeg",
        "-hwaccel", "cuda",
        "-y",
        "-i", video_path,
        "-ss", str(start),
        "-to", str(end),
        "-c:v", "h264_nvenc",
        "-an",
        out_path
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    return result.returncode == 0 and os.path.exists(out_path)

def process_video(idx, sample):
    global ok, dead
    
    clip_id = f"{idx:05d}"
    raw_path = f"{RAW_VIDEO_DIR}/{clip_id}.mp4"
    clip_path = f"{CLIP_DIR}/{clip_id}.mp4"

    if os.path.exists(clip_path):
        with counter_lock:
            ok += 1
        return

    success = download_video(sample["url"], raw_path)
    
    if not success:
        with counter_lock:
            dead += 1
        return

    clipped = cut_clip(
        raw_path,
        sample["start_time"],
        sample["end_time"],
        clip_path
    )

    cleanup_raw_videos()

    with counter_lock:
        if clipped:
            ok += 1
        else:
            dead += 1
        
        total_processed = ok + dead
        if total_processed % 50 == 0:
            print(f"Processed: {total_processed} | OK: {ok} | DEAD: {dead}")

def main(start_index=0, max_workers=16):
    with open(DATASET_JSON) as f:
        samples = json.load(f)

    samples_to_process = samples[start_index:]

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for idx, sample in enumerate(samples_to_process, start=start_index):
            futures.append(executor.submit(process_video, idx, sample))
        
        concurrent.futures.wait(futures)

    print("\n=== DONE ===")
    print(f"Total processed in this run: {len(samples_to_process)}")
    print(f"OK:    {ok}")
    print(f"DEAD:  {dead}")

if __name__ == "__main__":
    main(start_index=0, max_workers=16)
