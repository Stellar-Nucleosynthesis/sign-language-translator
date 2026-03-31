import json
import os
import torch
import torch.nn as nn
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from pathlib import Path

from datasets.msasl_dataset import MSASLDataset

load_dotenv(dotenv_path='lstm_model.env')
JSON_VAL = os.getenv("JSON_VAL")
DIR_VAL = os.getenv("DIR_VAL")
JSON_TEST = os.getenv("JSON_TEST")
DIR_TEST = os.getenv("DIR_TEST")
CHECKPOINT_DIR = Path(os.getenv("CHECKPOINT_DIR"))
MAPPING_PATH = CHECKPOINT_DIR / 'label_mapping.json'
MODEL_PATH = CHECKPOINT_DIR / 'model_final.pth'

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
    if not MAPPING_PATH.exists():
        print("Error: label_mapping.json not found in checkpoints!")
        return

    with open(MAPPING_PATH, 'r', encoding='utf-8') as f:
        label_mapping = json.load(f)

    num_classes = len(label_mapping)
    
    val_dataset = MSASLDataset(JSON_VAL, DIR_VAL, label_mapping)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    test_dataset = MSASLDataset(JSON_TEST, DIR_TEST, label_mapping)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Validation samples found: {len(val_dataset)}")
    print(f"Testing samples found: {len(test_dataset)}")

    input_size = 258
    hidden_size = 128
    num_layers = 2

    model = GestureLSTM(input_size, hidden_size, num_layers, num_classes).to(device)

    if not MODEL_PATH.exists():
        print(f"Error: Checkpoint not found at {MODEL_PATH}")
        return

    print("Loading model weights...")
    checkpoint = torch.load(MODEL_PATH)
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