import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from collections import defaultdict
from tqdm import tqdm

from prototypical_pipeline import KinematicDataset, KinematicEncoder

load_dotenv(dotenv_path='test_prototypical.env')
JSON_TRAIN = os.getenv("JSON_TRAIN")
DIR_TRAIN = os.getenv("DIR_TRAIN")
JSON_TEST = os.getenv("JSON_TEST")
DIR_TEST = os.getenv("DIR_TEST")
MODEL_SAVE_PATH = os.getenv("MODEL_SAVE_PATH")
PROTOTYPES_SAVE_PATH = os.getenv("PROTOTYPES_SAVE_PATH")

def run_prototypical_test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}\n")
    
    train_dataset = KinematicDataset(JSON_TRAIN, DIR_TRAIN, is_train=True)
    test_dataset = KinematicDataset(JSON_TEST, DIR_TEST, word_to_idx=train_dataset.word_to_idx, is_train=False)
    
    num_classes = len(train_dataset.word_to_idx)
    model = KinematicEncoder(num_classes=num_classes).to(device)
    
    if os.path.exists(MODEL_SAVE_PATH) and os.path.exists(PROTOTYPES_SAVE_PATH):
        print("[INFO] Loading saved model and prototypes...")
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
        
        saved_data = torch.load(PROTOTYPES_SAVE_PATH, map_location=device)
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
            
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print("\n=== Extracting Prototypes ===")
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
                    
        words_list = []
        prototype_matrix = []
        for word, vecs in word_embeddings.items():
            mean_vec = np.mean(vecs, axis=0)
            words_list.append(word)
            prototype_matrix.append(mean_vec)
            
        prototype_tensor = torch.tensor(np.array(prototype_matrix)).to(device)
        
        torch.save({'tensor': prototype_tensor, 'words': words_list}, PROTOTYPES_SAVE_PATH)
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