import os
import json
import re
import time
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

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

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, x):
        scores = self.attention(x).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), x).squeeze(1)
        return context

class KinematicEncoder(nn.Module):
    def __init__(self, input_size=138, hidden_size=256, latent_size=256, num_classes=800, dropout=0.5):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=3, padding=1)
        self.bn_conv = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        self.attention = Attention(hidden_size * 2)
        
        self.embed = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, latent_size),
            nn.BatchNorm1d(latent_size),
            nn.ReLU()
        )
        self.classifier = nn.Linear(latent_size, num_classes)

    def get_embedding(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.bn_conv(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        out = self.attention(out)
        return self.embed(out)

    def forward(self, x):
        emb = self.get_embedding(x)
        return self.classifier(emb)

def run_prototypical_test():
    json_train = '/mnt/c/Workstudy/CV/data/annotations/MSASL_train.json'
    dir_train = '/mnt/c/Workstudy/CV/keypoints_train_filtered'
    json_test = '/mnt/c/Workstudy/CV/data/annotations/MSASL_test.json'
    dir_test = '/mnt/c/Workstudy/CV/keypoints_test_filtered'
    
    model_save_path = '/mnt/c/Workstudy/CV/encoder.pth'
    prototypes_save_path = '/mnt/c/Workstudy/CV/prototypes.pt'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}\n")
    
    train_dataset = KinematicDataset(json_train, dir_train, is_train=True)
    test_dataset = KinematicDataset(json_test, dir_test, word_to_idx=train_dataset.word_to_idx, is_train=False)
    
    num_classes = len(train_dataset.word_to_idx)
    model = KinematicEncoder(num_classes=num_classes).to(device)
    
    if os.path.exists(model_save_path) and os.path.exists(prototypes_save_path):
        print("[INFO] Loading saved model and prototypes...")
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        
        saved_data = torch.load(prototypes_save_path, map_location=device)
        prototype_tensor = saved_data['tensor']
        words_list = saved_data['words']
        model.eval()
    else:
        print("[INFO] Saved model not found. Training from scratch...")
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        
        num_epochs = 40
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1:02d}/{num_epochs}]", leave=False)
            for inputs, labels, _ in progress_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                acc = 100 * correct / total
                progress_bar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{acc:.1f}%"})
                
            print(f"Epoch [{epoch+1:02d}/{num_epochs}] | CE Loss: {running_loss/len(train_loader):.4f} | Train Acc: {acc:.2f}%")
            
        torch.save(model.state_dict(), model_save_path)
        print("\n=== Extracting Prototypes ===")
        model.eval()
        word_embeddings = defaultdict(list)
        train_eval_dataset = KinematicDataset(json_train, dir_train, word_to_idx=train_dataset.word_to_idx, is_train=False)
        seq_loader = DataLoader(train_eval_dataset, batch_size=128, shuffle=False, num_workers=4)
        
        with torch.no_grad():
            for inputs, _, text_labels in tqdm(seq_loader, desc="Encoding vectors"):
                inputs = inputs.to(device)
                latent_vectors = model.get_embedding(inputs).cpu().numpy()
                for vec, label in zip(latent_vectors, text_labels):
                    word_embeddings[label].append(vec)
                    
        words_list = []
        prototype_matrix = []
        for word, vecs in word_embeddings.items():
            mean_vec = np.mean(vecs, axis=0)
            words_list.append(word)
            prototype_matrix.append(mean_vec)
            
        prototype_tensor = torch.tensor(np.array(prototype_matrix)).to(device)
        
        torch.save({'tensor': prototype_tensor, 'words': words_list}, prototypes_save_path)
        print("[SUCCESS] Model and Prototypes saved!")

    print("\n=== EVALUATION ON TEST SET ===")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    total_samples = 0
    correct_top1 = 0
    correct_top5 = 0
    
    print_limit = 50
    print(f"\n{'TRUE WORD':<15} | {'TOP-1 PRED':<15} | {'CONF %':<8} | {'TOP-2 PRED':<15} | {'CONF %':<8} | {'STATUS'}")
    print("-" * 85)

    with torch.no_grad():
        for inputs, _, true_words in tqdm(test_loader, desc="Testing", leave=False):
            inputs = inputs.to(device)
            true_word = true_words[0]
            
            embedding = model.get_embedding(inputs) 
            distances = torch.cdist(embedding, prototype_tensor, p=2.0)
            
            probs = torch.softmax(-distances, dim=1)
            
            top5_probs, top5_indices = torch.topk(probs, 5, dim=1, largest=True)
            
            total_samples += 1
            
            top1_word = words_list[top5_indices[0, 0].item()]
            top1_conf = top5_probs[0, 0].item() * 100
            
            top2_word = words_list[top5_indices[0, 1].item()]
            top2_conf = top5_probs[0, 1].item() * 100
            
            top5_words = [words_list[idx.item()] for idx in top5_indices[0]]
            
            if top1_word == true_word:
                correct_top1 += 1
                status = "✅"
            else:
                status = "❌"
                
            if true_word in top5_words:
                correct_top5 += 1
                
            if total_samples <= print_limit:
                print(f"{true_word:<15} | {top1_word:<15} | {top1_conf:>5.1f}%   | {top2_word:<15} | {top2_conf:>5.1f}%   | {status}")

    top1_acc = (correct_top1 / total_samples) * 100 if total_samples > 0 else 0
    top5_acc = (correct_top5 / total_samples) * 100 if total_samples > 0 else 0

    print("\n" + "="*50)
    print("FINAL TEST SET RESULTS")
    print("="*50)
    print(f"Total Test Samples Evaluated : {total_samples}")
    print(f"Top-1 Accuracy (Exact match) : {top1_acc:.2f}%")
    print(f"Top-5 Accuracy (In top 5)    : {top5_acc:.2f}%")
    print("="*50)

if __name__ == '__main__':
    run_prototypical_test()