import json
import re
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

class KinematicDataset(Dataset):
    def __init__(self, json_path, npy_dir, word_to_idx=None, is_train=False):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.raw_data = json.load(f)
        self.npy_dir = Path(npy_dir)
        self.is_train = is_train
        self.valid_samples = []

        self.word_to_idx = word_to_idx if word_to_idx is not None else {}
        if word_to_idx is None:
            idx_counter = 0
            for item in self.raw_data:
                label_text = str(item.get('text', item.get('label', 'unknown'))).lower().strip()
                if label_text not in self.word_to_idx:
                    self.word_to_idx[label_text] = idx_counter
                    idx_counter += 1

        for npy_file in self.npy_dir.glob('*.npy'):
            match = re.search(r'(\d+)\.npy$', npy_file.name)
            if not match:
                continue
            idx = int(match.group(1))
            if idx >= len(self.raw_data):
                continue
            raw_label = self.raw_data[idx].get('text', self.raw_data[idx].get('label', 'unknown'))
            label_text = str(raw_label).lower().strip()

            if label_text in self.word_to_idx:
                try:
                    _ = np.load(npy_file, mmap_mode='r')
                    self.valid_samples.append({
                        'file_path': npy_file,
                        'label_idx': self.word_to_idx[label_text],
                        'label_text': label_text
                    })
                except Exception:
                    pass

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        sample = self.valid_samples[idx]
        try:
            keypoints = np.load(sample['file_path'])
        except Exception:
            keypoints = np.zeros((30, 138), dtype=np.float32)

        target_frames = 30
        current_frames = keypoints.shape[0]

        if current_frames < target_frames:
            padding = np.zeros((target_frames - current_frames, keypoints.shape[1]), dtype=np.float32)
            keypoints = np.vstack((keypoints, padding))
        elif current_frames > target_frames:
            indices = np.linspace(0, current_frames - 1, target_frames).astype(int)
            keypoints = keypoints[indices]

        keypoints = keypoints.reshape(target_frames, 46, 3)
        centers = (keypoints[:, 0, :] + keypoints[:, 1, :]) / 2.0
        keypoints = keypoints - centers[:, np.newaxis, :]
        shoulder_dist = np.linalg.norm(keypoints[:, 0, :] - keypoints[:, 1, :], axis=-1, keepdims=True)
        shoulder_dist = np.where(shoulder_dist < 1e-5, 1.0, shoulder_dist)
        keypoints = keypoints / shoulder_dist[:, np.newaxis, :]

        if self.is_train:
            scale = np.random.uniform(0.9, 1.1)
            keypoints = keypoints * scale
            shift = np.random.uniform(-0.05, 0.05, (1, 1, 3)).astype(np.float32)
            keypoints = keypoints + shift
            noise = np.random.normal(0, 0.005, keypoints.shape).astype(np.float32)
            keypoints = keypoints + noise

        keypoints = keypoints.reshape(target_frames, 138)
        tensor_data = torch.FloatTensor(keypoints)
        return tensor_data, sample['label_idx'], sample['label_text']