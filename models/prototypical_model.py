import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from collections import defaultdict
from tqdm import tqdm

from datasets.kinematic_dataset import KinematicDataset

load_dotenv(dotenv_path='prototypical_model.env')
JSON_TRAIN = os.getenv("JSON_TRAIN")
DIR_TRAIN = os.getenv("DIR_TRAIN")
JSON_VAL = os.getenv("JSON_VAL")
DIR_VAL = os.getenv("DIR_VAL")

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

def run_prototypical_pipeline():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}\n")
    
    train_dataset = KinematicDataset(JSON_TRAIN, DIR_TRAIN, is_train=True)
    val_dataset = KinematicDataset(JSON_VAL, DIR_VAL, word_to_idx=train_dataset.word_to_idx, is_train=False)
    
    num_classes = len(train_dataset.word_to_idx)
    print(f"[INFO] Found {num_classes} unique words.")
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    model = KinematicEncoder(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    num_epochs = 40
    print(f"\n=== STEP 1: Training Discriminative Space ({num_epochs} Epochs) ===")
    
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
            
        train_acc = 100 * correct / total
        print(f"Epoch [{epoch+1:02d}/{num_epochs}] | CE Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}%")

    print("\n=== STEP 2: Extracting Prototypes ===")
    model.eval()
    word_embeddings = defaultdict(list)
    
    train_eval_dataset = KinematicDataset(JSON_TRAIN, DIR_TRAIN, word_to_idx=train_dataset.word_to_idx, is_train=False)
    seq_loader = DataLoader(train_eval_dataset, batch_size=128, shuffle=False, num_workers=4)
    
    with torch.no_grad():
        for inputs, _, text_labels in tqdm(seq_loader, desc="Encoding vectors"):
            inputs = inputs.to(device)
            latent_vectors = model.get_embedding(inputs).cpu().numpy()
            for vec, label in zip(latent_vectors, text_labels):
                word_embeddings[label].append(vec)
                
    prototypes = {}
    words_list = []
    prototype_matrix = []
    
    for word, vecs in word_embeddings.items():
        mean_vec = np.mean(vecs, axis=0)
        prototypes[word] = torch.tensor(mean_vec).to(device)
        words_list.append(word)
        prototype_matrix.append(mean_vec)
        
    prototype_tensor = torch.tensor(np.array(prototype_matrix)).to(device)

    print("\n=== STEP 3: Prototypical Inference (Validation) ===")
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
    
    total_samples = 0
    correct_top1 = 0
    correct_top5 = 0
    
    with torch.no_grad():
        for inputs, _, true_words in tqdm(val_loader, desc="Validating"):
            inputs = inputs.to(device)
            batch_embeddings = model.get_embedding(inputs) 
            
            distances = torch.cdist(batch_embeddings, prototype_tensor, p=2.0)
            top5_distances, top5_indices = torch.topk(distances, 5, dim=1, largest=False)
            
            for i in range(len(true_words)):
                true_word = true_words[i]
                total_samples += 1
                
                top1_word = words_list[top5_indices[i, 0].item()]
                top5_words = [words_list[idx.item()] for idx in top5_indices[i]]
                
                if top1_word == true_word:
                    correct_top1 += 1
                if true_word in top5_words:
                    correct_top5 += 1

    top1_acc = (correct_top1 / total_samples) * 100 if total_samples > 0 else 0
    top5_acc = (correct_top5 / total_samples) * 100 if total_samples > 0 else 0

    print("\n" + "="*50)
    print("PROTOTYPICAL NETWORK RESULTS")
    print("="*50)
    print(f"Total Validation Samples: {total_samples}")
    print(f"Top-1 Accuracy (Exact match) : {top1_acc:.2f}%")
    print(f"Top-5 Accuracy (In top 5 guesses) : {top5_acc:.2f}%")
    print("="*50)

if __name__ == '__main__':
    run_prototypical_pipeline()