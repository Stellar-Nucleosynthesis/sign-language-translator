import cv2
import mediapipe as mp
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys
import traceback

global_holistic = None

def init_worker():
    global global_holistic
    mp_holistic = mp.solutions.holistic
    global_holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1
    )

def process_video_for_keypoints(video_path, output_dir):
    global global_holistic

    cap = cv2.VideoCapture(str(video_path))
    sequence_data = []

    if not cap.isOpened():
        return False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = global_holistic.process(frame_rgb)

        frame_keypoints = []

        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                frame_keypoints.extend([lm.x, lm.y, lm.z])
        else:
            frame_keypoints.extend([0.0] * 63)

        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                frame_keypoints.extend([lm.x, lm.y, lm.z])
        else:
            frame_keypoints.extend([0.0] * 63)

        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                frame_keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])
        else:
            frame_keypoints.extend([0.0] * 132)

        if results.face_landmarks:
            for lm in results.face_landmarks.landmark:
                frame_keypoints.extend([lm.x, lm.y, lm.z])
        else:
            frame_keypoints.extend([0.0] * 1404)

        sequence_data.append(frame_keypoints)

    cap.release()

    if len(sequence_data) > 0:
        out_file = Path(output_dir) / f"{Path(video_path).stem}.npy"
        np.save(out_file, np.array(sequence_data, dtype=np.float32))

    return True

def main(input_dir, output_dir, max_workers=4):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_files = list(input_dir.glob('*.mp4'))
    total_videos = len(video_files)

    if total_videos == 0:
        print(f"No videos found in {input_dir}.")
        return

    processed = 0
    print(f"Starting processing of {total_videos} videos...")

    with ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker) as executor:
        futures = {executor.submit(process_video_for_keypoints, vf, output_dir): vf for vf in video_files}

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"\nError processing video:")
                traceback.print_exc()

            processed += 1
            percentage = int((processed / total_videos) * 100)
            sys.stdout.write(f"\rProcessed: {processed} / {total_videos} [{percentage}%]")
            sys.stdout.flush()

    print("\nFinished!")

if __name__ == '__main__':
    input_directory = '/mnt/input_dir'
    output_directory = '/mnt/output_dir'
    main(input_directory, output_directory, max_workers=4)
