import os
import json
import time
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import numpy as np
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from pathlib import Path

from datasets.msasl_dataset import MSASLDataset

load_dotenv(dotenv_path="transformer_model.env")
JSON_TRAIN = os.getenv("JSON_TRAIN")
JSON_VAL = os.getenv("JSON_VAL")
DIR_TRAIN = os.getenv("DIR_TRAIN")
DIR_VAL = os.getenv("DIR_VAL")
CHECKPOINT_DIR = Path(os.getenv("CHECKPOINT_DIR"))
FINAL_MODEL_PATH = CHECKPOINT_DIR / 'model_final.pth'
LABEL_MAPPING = CHECKPOINT_DIR / 'label_mapping.json'

def get_top_k_label_mapping(json_paths, top_k=100):
    label_counts = Counter()
    for jp in json_paths:
        if not os.path.exists(jp):
            continue
        with open(jp, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                label = str(item.get('text', item.get('label', 'unknown'))).lower().strip()
                label_counts[label] += 1
                
    top_labels = [label for label, count in label_counts.most_common(top_k)]
    top_labels.sort()
    return {label: idx for idx, label in enumerate(top_labels)}

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=30):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].unsqueeze(0)
        return x

class TransformerASLModel(nn.Module):
    def __init__(self, input_size=540, hidden_size=256, num_heads=8, num_layers=4, num_classes=100, dropout=0.5):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size)
        )
        self.pos_encoder = PositionalEncoding(hidden_size)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_size, 
            nhead=num_heads, 
            dim_feedforward=hidden_size * 2, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        x = self.project(x)
        x = self.pos_encoder(x)
        out = self.transformer(x)
        out = torch.mean(out, dim=1)
        out = self.fc(out)
        return out

def format_time(seconds):
    return str(datetime.timedelta(seconds=int(seconds)))

def evaluate_loader(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    acc = 100 * correct / total if total > 0 else 0
    avg_loss = running_loss / len(loader) if len(loader) > 0 else 0
    return avg_loss, acc

def train_model():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    top_k_classes = 100
    label_mapping = get_top_k_label_mapping([JSON_TRAIN, JSON_VAL], top_k=top_k_classes)
    num_classes = len(label_mapping)
    
    print(f"Prepared mapping for top {num_classes} unique classes.\n")

    with open(LABEL_MAPPING, 'w', encoding='utf-8') as f:
        json.dump(label_mapping, f, ensure_ascii=False, indent=4)

    train_dataset = MSASLDataset(JSON_TRAIN, DIR_TRAIN, label_mapping, oversample=True, augment=True)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)

    val_dataset = MSASLDataset(JSON_VAL, DIR_VAL, label_mapping, oversample=False, augment=False)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    num_epochs = 150
    start_epoch = 0

    model = TransformerASLModel(input_size=540, hidden_size=256, num_heads=8, num_layers=4, num_classes=num_classes, dropout=0.6).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)
    
    start_time = time.time()

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
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

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        val_loss, val_acc = evaluate_loader(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        elapsed_time = time.time() - start_time
        time_per_epoch = elapsed_time / ((epoch - start_epoch) + 1)
        remaining_epochs = num_epochs - (epoch + 1)
        eta = time_per_epoch * remaining_epochs

        print(f"Epoch [{epoch+1}/{num_epochs}] LR: {current_lr:.6f} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
              f"Elapsed: {format_time(elapsed_time)} ETA: {format_time(eta)}")

        if (epoch + 1) % 5 == 0:
            checkpoint_path = CHECKPOINT_DIR / f'model_epoch_{epoch + 1}.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)

    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, FINAL_MODEL_PATH)

if __name__ == '__main__':
    train_model()