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
from smart_gestures.gestures.lstm_model import LSTMModel

def get_global_label_mapping(json_paths):
    all_labels = set()
    for jp in json_paths:
        if not os.path.exists(jp):
            continue
        with open(jp, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                label = item.get('text', item.get('label', 'unknown'))
                all_labels.add(str(label).lower().strip())
    
    sorted_labels = sorted(list(all_labels))
    return {label: idx for idx, label in enumerate(sorted_labels)}

class MSASLDataset(Dataset):
    def __init__(self, json_path, npy_dir, label_mapping):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.raw_data = json.load(f)
        
        self.npy_dir = Path(npy_dir)
        self.valid_samples = []
        
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

        print(f"-> Found {len(self.valid_samples)} matching samples in {self.npy_dir.name}\n")
        
        if len(self.valid_samples) == 0:
            raise ValueError(f"Dataset is empty! Could not find any valid labels in {self.npy_dir.name}.")

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        sample = self.valid_samples[idx]
        keypoints = np.load(sample['file_path'])
        
        target_frames = 30
        current_frames = keypoints.shape[0]
        actual_length = min(current_frames, target_frames)
        
        if current_frames < target_frames:
            padding = np.zeros((target_frames - current_frames, keypoints.shape[1]), dtype=np.float32)
            keypoints = np.vstack((keypoints, padding))
        elif current_frames > target_frames:
            keypoints = keypoints[:target_frames, :]
            
        tensor_data = torch.FloatTensor(keypoints)
        label = torch.tensor(sample['label_idx'], dtype=torch.long)
        seq_length = torch.tensor(actual_length, dtype=torch.long)
        
        return tensor_data, label, seq_length

def format_time(seconds):
    return str(datetime.timedelta(seconds=int(seconds)))

def evaluate_loader(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels, seq_lengths in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            seq_lengths = seq_lengths.cpu()
            
            outputs = model(inputs, seq_lengths)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    acc = 100 * correct / total if total > 0 else 0
    avg_loss = running_loss / len(loader) if len(loader) > 0 else 0
    return avg_loss, acc

def train_model():
    json_train = '/mnt/c/Workstudy/CV/data/annotations/MSASL_train.json'
    json_val = '/mnt/c/Workstudy/CV/data/annotations/MSASL_val.json'
    dir_train = '/mnt/c/Workstudy/CV/keypoints_train'
    dir_val = '/mnt/c/Workstudy/CV/keypoints_val'
    checkpoint_dir = Path('/mnt/c/Workstudy/CV/checkpoints')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    label_mapping = get_global_label_mapping([json_train, json_val])
    num_classes = len(label_mapping)
    
    print(f"Prepared mapping for {num_classes} unique classes.\n")

    with open(checkpoint_dir / 'label_mapping.json', 'w', encoding='utf-8') as f:
        json.dump(label_mapping, f, ensure_ascii=False, indent=4)

    train_dataset = MSASLDataset(json_train, dir_train, label_mapping)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

    val_dataset = MSASLDataset(json_val, dir_val, label_mapping)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    num_epochs = 150
    start_epoch = 0

    wrapper = LSTMModel()
    
    if isinstance(wrapper, nn.Module):
        base_model = wrapper
    elif hasattr(wrapper, 'model') and isinstance(wrapper.model, nn.Module):
        base_model = wrapper.model
    else:
        found = False
        for attr_name in dir(wrapper):
            attr = getattr(wrapper, attr_name)
            if isinstance(attr, nn.Module):
                base_model = attr
                found = True
                break
        if not found:
            raise AttributeError("Cannot extract PyTorch nn.Module from smart_gestures LSTMModel.")

    modules = list(base_model.named_modules())
    linear_layers = [m for m in modules if isinstance(m[1], nn.Linear)]
    
    if linear_layers:
        last_name, last_module = linear_layers[-1]
        parts = last_name.split('.')
        parent = base_model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], nn.Linear(last_module.in_features, num_classes))
    else:
        raise ValueError("Could not find a Linear layer to replace in the pre-trained model.")

    model = base_model.to(device)

    for param in model.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    start_time = time.time()

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels, seq_lengths in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            seq_lengths = seq_lengths.cpu()

            optimizer.zero_grad()
            outputs = model(inputs, seq_lengths)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        val_loss, val_acc = evaluate_loader(model, val_loader, criterion, device)

        elapsed_time = time.time() - start_time
        time_per_epoch = elapsed_time / ((epoch - start_epoch) + 1)
        remaining_epochs = num_epochs - (epoch + 1)
        eta = time_per_epoch * remaining_epochs

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
              f"Elapsed: {format_time(elapsed_time)} ETA: {format_time(eta)}")

        if (epoch + 1) % 5 == 0:
            checkpoint_path = checkpoint_dir / f'model_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)

    final_model_path = checkpoint_dir / 'model_final.pth'
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, final_model_path)

if __name__ == '__main__':
    train_model()