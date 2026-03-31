import json
import random
import re
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

class MSASLDataset(Dataset):
    def __init__(self, json_path, npy_dir, label_mapping, oversample=False, augment=False):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.raw_data = json.load(f)

        self.npy_dir = Path(npy_dir)
        self.valid_samples = []
        self.augment = augment

        print(f"Scanning {self.npy_dir.name}...")

        for npy_file in self.npy_dir.glob('*.npy'):
            match = re.search(r'(\d+)\.npy$', npy_file.name)
            if not match:
                continue

            idx = int(match.group(1))
            if idx >= len(self.raw_data):
                continue

            raw_label = self.raw_data[idx].get('text', self.raw_data[idx].get('label', 'unknown'))
            label_text = str(raw_label).lower().strip()

            if label_text in label_mapping:
                self.valid_samples.append({
                    'file_path': npy_file,
                    'label_idx': label_mapping[label_text]
                })

        if len(self.valid_samples) == 0:
            raise ValueError(f"Dataset is empty! Could not find any valid labels in {self.npy_dir.name}.")

        if oversample:
            class_counts = defaultdict(list)
            for sample in self.valid_samples:
                class_counts[sample['label_idx']].append(sample)

            max_count = max(len(samples) for samples in class_counts.values())
            balanced_samples = []

            for label_idx, samples in class_counts.items():
                balanced_samples.extend(samples)
                shortage = max_count - len(samples)
                if shortage > 0:
                    balanced_samples.extend(random.choices(samples, k=shortage))

            self.valid_samples = balanced_samples

        print(f"-> Ready {len(self.valid_samples)} samples in {self.npy_dir.name}\n")

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        sample = self.valid_samples[idx]
        keypoints = np.load(sample['file_path'])

        target_frames = 30
        current_frames = keypoints.shape[0]

        if current_frames < target_frames:
            padding = np.zeros((target_frames - current_frames, keypoints.shape[1]), dtype=np.float32)
            keypoints = np.vstack((keypoints, padding))
        elif current_frames > target_frames:
            if self.augment:
                start_max = current_frames - target_frames
                start_idx = random.randint(0, start_max)
                keypoints = keypoints[start_idx:start_idx + target_frames, :]
            else:
                indices = np.linspace(0, current_frames - 1, target_frames).astype(int)
                keypoints = keypoints[indices]

        keypoints = keypoints.reshape(target_frames, 180, 3)

        centers = (keypoints[:, 0, :] + keypoints[:, 1, :]) / 2.0
        keypoints = keypoints - centers[:, np.newaxis, :]

        shoulder_dist = np.linalg.norm(keypoints[:, 0, :] - keypoints[:, 1, :], axis=-1, keepdims=True)
        shoulder_dist = np.where(shoulder_dist < 1e-5, 1.0, shoulder_dist)
        keypoints = keypoints / shoulder_dist[:, np.newaxis, :]

        if self.augment:
            jitter = np.random.normal(0, 0.02, keypoints.shape)
            keypoints = keypoints + jitter

        keypoints = keypoints.reshape(target_frames, 540)

        tensor_data = torch.FloatTensor(keypoints)
        label = torch.tensor(sample['label_idx'], dtype=torch.long)

        return tensor_data, label