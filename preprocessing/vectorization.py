import os

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
import concurrent.futures

from dotenv import load_dotenv

load_dotenv(dotenv_path='vectorization.env')
SOURCE_DIR = os.getenv("SOURCE_DIR")
TARGET_DIR = os.getenv("TARGET_DIR")

pose_indices = [11, 12, 13, 14, 15, 16]
face_indices = list(range(132))

def extract_landmarks(landmarks, indices):
    output = np.zeros((len(indices), 3))

    if landmarks:
        for i, idx in enumerate(indices):
            res = landmarks.landmark[idx]
            output[i] = np.array([res.x, res.y, res.z])

    return output

def extract_keypoints(results):
    pose = extract_landmarks(results.pose_landmarks, pose_indices)
    face = extract_landmarks(results.face_landmarks, face_indices)

    lh = np.zeros((21, 3))
    if results.left_hand_landmarks:
        for i, res in enumerate(results.left_hand_landmarks.landmark):
            lh[i] = np.array([res.x, res.y, res.z])

    rh = np.zeros((21, 3))
    if results.right_hand_landmarks:
        for i, res in enumerate(results.right_hand_landmarks.landmark):
            rh[i] = np.array([res.x, res.y, res.z])

    return np.concatenate([pose.flatten(), face.flatten(), lh.flatten(), rh.flatten()])

def process_and_save(args):
    video_file, target_dir = args
    target_path = Path(target_dir)
    npy_path = target_path / f"{video_file.stem}.npy"
    
    if npy_path.exists():
        return f"Skipped: {npy_path.name}"

    mp_holistic = mp.solutions.holistic
    cap = cv2.VideoCapture(str(video_file))
    frames_data = []
    target_frames = 30
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            
            keypoints = extract_keypoints(results)
            frames_data.append(keypoints)
            
    cap.release()

    frames_data = np.array(frames_data)
    
    if len(frames_data) == 0:
        final_data = np.zeros((target_frames, 540), dtype=np.float32)
    elif len(frames_data) < target_frames:
        padding = np.zeros((target_frames - len(frames_data), 540), dtype=np.float32)
        final_data = np.vstack((frames_data, padding))
    else:
        indices = np.linspace(0, len(frames_data) - 1, target_frames).astype(int)
        final_data = frames_data[indices]

    np.save(npy_path, final_data.astype(np.float32))
    return f"Saved: {npy_path.name}"

def vectorize_dataset(source_dir, target_dir):
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    if not source_path.exists():
        print(f"Error: Directory {source_dir} does not exist!")
        return

    video_files = list(source_path.rglob('*.mp4')) + list(source_path.rglob('*.MP4'))
    total_videos = len(video_files)

    print(f"Found {total_videos} videos in {source_dir}")

    if total_videos == 0:
        return

    num_workers = 4
    print(f"Starting multiprocessing with {num_workers} cores...")

    tasks = [(video_file, target_dir) for video_file in video_files]

    completed = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        for result in executor.map(process_and_save, tasks):
            completed += 1
            if completed % 50 == 0 or completed == total_videos:
                print(f"[{completed}/{total_videos}] {result}")

if __name__ == '__main__':
    vectorize_dataset(SOURCE_DIR, TARGET_DIR)
