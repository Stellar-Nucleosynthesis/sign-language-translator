import json
import re
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

class MSASLDataset(Dataset):
    def __init__(self, json_path, npy_dir, label_mapping):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.raw_data = json.load(f)
        
        self.npy_dir = Path(npy_dir)
        self.valid_samples = []
        
        for npy_file in self.npy_dir.glob('*.npy'):
            match = re.search(r'(\d+)\.npy$', npy_file.name)
            if not match:
                continue
                
            idx = int(match.group(1))
            if idx >= len(self.raw_data):
                continue

            label_text = self.raw_data[idx].get('text', self.raw_data[idx].get('label', 'unknown'))
            if label_text in label_mapping:
                self.valid_samples.append({
                    'file_path': npy_file,
                    'label_idx': label_mapping[label_text]
                })

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
            keypoints = keypoints[:target_frames, :]
            
        tensor_data = torch.FloatTensor(keypoints)
        label = torch.tensor(sample['label_idx'], dtype=torch.long)
        return tensor_data, label

class GestureLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GestureLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def evaluate_loader(model, loader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    if total > 0:
        return 100 * correct / total, correct, total
    return 0.0, 0, 0

def test_model():
    json_val = 'MSASL_val.json'
    dir_val = 'val_dir/'
    json_test = 'MSASL_test.json'
    dir_test = 'test_dir/'
    
    checkpoint_dir = Path('checkpoints/')
    mapping_path = checkpoint_dir / 'label_mapping.json'
    model_path = checkpoint_dir / 'model_final.pth'

    if not mapping_path.exists():
        print("Error: label_mapping.json not found in checkpoints!")
        return

    with open(mapping_path, 'r', encoding='utf-8') as f:
        label_mapping = json.load(f)

    num_classes = len(label_mapping)
    
    val_dataset = MSASLDataset(json_val, dir_val, label_mapping)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    test_dataset = MSASLDataset(json_test, dir_test, label_mapping)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Validation samples found: {len(val_dataset)}")
    print(f"Testing samples found: {len(test_dataset)}")

    input_size = 258
    hidden_size = 128
    num_layers = 2

    model = GestureLSTM(input_size, hidden_size, num_layers, num_classes).to(device)

    if not model_path.exists():
        print(f"Error: Checkpoint not found at {model_path}")
        return

    print("Loading model weights...")
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()

    print("Starting evaluation on validation set...")
    val_acc, val_corr, val_tot = evaluate_loader(model, val_loader, device)
    
    print("Starting evaluation on test set...")
    test_acc, test_corr, test_tot = evaluate_loader(model, test_loader, device)

    print("\n=== EVALUATION RESULTS ===")
    print(f"VALIDATION ACCURACY: {val_acc:.2f}% ({val_corr} / {val_tot})")
    print(f"TEST ACCURACY:       {test_acc:.2f}% ({test_corr} / {test_tot})")
    
    diff = val_acc - test_acc
    print(f"DIFFERENCE:          {abs(diff):.2f}%")
    
    if diff > 5.0:
        print("\nWARNING: Validation accuracy is significantly higher than Test accuracy.")
        print("Possible overfitting to the validation set.")
    elif diff < -5.0:
        print("\nNOTE: Test accuracy is higher than Validation accuracy.")
    else:
        print("\nOK: Model generalizes well. No significant overfitting detected.")

if __name__ == '__main__':
    test_model()