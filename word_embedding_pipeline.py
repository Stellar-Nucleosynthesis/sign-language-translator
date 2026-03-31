import json
import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

load_dotenv(dotenv_path='prototypical_pipeline.env')
JSON_TRAIN = os.getenv("JSON_TRAIN")
DIR_TRAIN  = os.getenv("DIR_TRAIN")
JSON_VAL   = os.getenv("JSON_VAL")
DIR_VAL    = os.getenv("DIR_VAL")
DICT_PATH  = Path(os.getenv("DICT_PATH"))
TFIDF_PATH = Path(os.getenv("TFIDF_PATH"))
LSI_PATH   = Path(os.getenv("LSI_PATH"))

NUM_CLUSTERS = 30
ROUTER_EPOCHS = 20
LOCAL_EPOCHS = 30


def cluster_vocabulary(word_list, model, n_clusters):
    print(f"[MiniLM] Embedding {len(word_list)} words …")

    vecs = model.encode(
        word_list,
        batch_size=256,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    effective_k = min(n_clusters, len(word_list))

    print(f"[MiniLM] Running KMeans (k={effective_k}) …")
    km = KMeans(
        n_clusters=effective_k,
        random_state=42,
        n_init=10
    )
    labels = km.fit_predict(vecs)

    word_to_cluster = {w: int(c) for w, c in zip(word_list, labels)}
    cluster_to_words = defaultdict(list)

    for w, c in word_to_cluster.items():
        cluster_to_words[c].append(w)

    sizes = [len(v) for v in cluster_to_words.values()]
    print(
        f"[MiniLM + KMeans] {effective_k} clusters | "
        f"min={min(sizes)}, max={max(sizes)}, avg={np.mean(sizes):.1f}"
    )

    return word_to_cluster, cluster_to_words, effective_k

class KinematicDataset(Dataset):
    def __init__(self, json_path, npy_dir, word_to_idx=None, is_train=False):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.raw_data = json.load(f)
        self.npy_dir       = Path(npy_dir)
        self.is_train      = is_train
        self.valid_samples = []

        self.word_to_idx = word_to_idx if word_to_idx is not None else {}
        if word_to_idx is None:
            idx_counter = 0
            for item in self.raw_data:
                label = str(item.get('text', item.get('label', 'unknown'))).lower().strip()
                if label not in self.word_to_idx:
                    self.word_to_idx[label] = idx_counter
                    idx_counter += 1

        for npy_file in self.npy_dir.glob('*.npy'):
            m = re.search(r'(\d+)\.npy$', npy_file.name)
            if not m:
                continue
            idx = int(m.group(1))
            if idx >= len(self.raw_data):
                continue
            raw   = self.raw_data[idx].get('text', self.raw_data[idx].get('label', 'unknown'))
            label = str(raw).lower().strip()
            if label in self.word_to_idx:
                try:
                    np.load(npy_file, mmap_mode='r')
                    self.valid_samples.append({
                        'file_path': npy_file,
                        'label_idx': self.word_to_idx[label],
                        'label_text': label,
                    })
                except Exception:
                    pass

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, i):
        s = self.valid_samples[i]
        try:
            kp = np.load(s['file_path'])
        except Exception:
            kp = np.zeros((30, 138), dtype=np.float32)

        t, f = 30, kp.shape[0]
        if f < t:
            kp = np.vstack((kp, np.zeros((t - f, kp.shape[1]), dtype=np.float32)))
        elif f > t:
            kp = kp[np.linspace(0, f - 1, t).astype(int)]

        kp = kp.reshape(t, 46, 3)
        centers = (kp[:, 0] + kp[:, 1]) / 2.0
        kp -= centers[:, None]
        sd  = np.linalg.norm(kp[:, 0] - kp[:, 1], axis=-1, keepdims=True)
        sd  = np.where(sd < 1e-5, 1.0, sd)
        kp /= sd[:, None]

        if self.is_train:
            kp  = kp * np.random.uniform(0.9, 1.1)
            kp += np.random.uniform(-0.05, 0.05, (1, 1, 3)).astype(np.float32)
            kp += np.random.normal(0, 0.005, kp.shape).astype(np.float32)

        return torch.FloatTensor(kp.reshape(t, 138)), s['label_idx'], s['label_text']


class RouterDataset(Dataset):
    def __init__(self, base: KinematicDataset, word_to_cluster: dict):
        self.base            = base
        self.word_to_cluster = word_to_cluster

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        tensor, _, label_text = self.base[i]
        return tensor, self.word_to_cluster.get(label_text, 0), label_text


class ClusterDataset(Dataset):
    def __init__(self, base: KinematicDataset, cluster_id: int,
                 word_to_cluster: dict, cluster_words: list):
        self.base       = base
        self.cid        = cluster_id
        self.local_idx  = {w: i for i, w in enumerate(sorted(cluster_words))}
        self.indices    = [
            j for j, s in enumerate(base.valid_samples)
            if word_to_cluster.get(s['label_text'], -1) == cluster_id
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        tensor, _, label_text = self.base[self.indices[i]]
        return tensor, self.local_idx[label_text], label_text

    @property
    def idx_to_word(self) -> dict:
        return {v: k for k, v in self.local_idx.items()}

class Attention(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.fc = nn.Linear(h, 1, bias=False)

    def forward(self, x):
        w = torch.softmax(self.fc(x).squeeze(-1), dim=-1)
        return torch.bmm(w.unsqueeze(1), x).squeeze(1)


class KinematicEncoder(nn.Module):
    def __init__(self, input_size=138, hidden_size=256, latent_size=256,
                 num_classes=800, dropout=0.5):
        super().__init__()
        self.conv  = nn.Conv1d(input_size, hidden_size, 3, padding=1)
        self.bn    = nn.BatchNorm1d(hidden_size)
        self.relu  = nn.ReLU()
        self.lstm  = nn.LSTM(hidden_size, hidden_size, num_layers=2,
                             batch_first=True, bidirectional=True, dropout=dropout)
        self.attn  = Attention(hidden_size * 2)
        self.embed = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, latent_size),
            nn.BatchNorm1d(latent_size),
            nn.ReLU(),
        )
        self.head  = nn.Linear(latent_size, num_classes)

    def get_embedding(self, x):
        x = self.relu(self.bn(self.conv(x.permute(0, 2, 1)))).permute(0, 2, 1)
        out, _ = self.lstm(x)
        return self.embed(self.attn(out))

    def forward(self, x):
        return self.head(self.get_embedding(x))


def train_model(model: nn.Module, loader: DataLoader,
                num_epochs: int, device, tag: str) -> nn.Module:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    for epoch in range(num_epochs):
        model.train()
        total_loss = correct = total = 0
        bar = tqdm(loader, desc=f"  {tag} [{epoch+1:02d}/{num_epochs}]", leave=False)

        for x, y, _ in bar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out  = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total      += y.size(0)
            correct    += (out.argmax(1) == y).sum().item()
            bar.set_postfix(loss=f"{loss.item():.4f}",
                            acc=f"{100*correct/total:.1f}%")

        print(f"  {tag} [{epoch+1:02d}/{num_epochs}]  "
              f"loss={total_loss/len(loader):.4f}  "
              f"acc={100*correct/total:.2f}%")

    return model


def run_embedding_clustered_pipeline():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] device = {device}\n")

    print("=== Loading SentenceTransformer (MiniLM) ===")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    print("\n=== Building base training dataset ===")
    train_base = KinematicDataset(JSON_TRAIN, DIR_TRAIN, is_train=True)
    print(f"[INFO] {len(train_base.word_to_idx)} unique words, "
          f"{len(train_base)} training samples")

    print("\n=== Step 0: Vocabulary clustering ===")
    word_list = list(train_base.word_to_idx.keys())
    word_to_cluster, cluster_to_words, n_clusters = cluster_vocabulary(
        word_list, embedder, NUM_CLUSTERS
    )

    print(f"\n=== Step 1: Training router  "
          f"({n_clusters} clusters, {ROUTER_EPOCHS} epochs) ===")
    router_ds     = RouterDataset(train_base, word_to_cluster)
    router_loader = DataLoader(router_ds, batch_size=128, shuffle=True,
                               num_workers=4, drop_last=True)
    router        = KinematicEncoder(num_classes=n_clusters).to(device)
    router        = train_model(router, router_loader, ROUTER_EPOCHS, device, "Router")

    print(f"\n=== Step 2: Training {n_clusters} local cluster models "
          f"({LOCAL_EPOCHS} epochs each) ===")

    local_models:   dict[int, KinematicEncoder] = {}
    local_idx_maps: dict[int, dict] = {}

    for cid in range(n_clusters):
        words = cluster_to_words[cid]
        print(f"\n── Cluster {cid:02d}  ({len(words)} words) "
              f"| sample: {words[:6]}{'...' if len(words) > 6 else ''}")

        ds = ClusterDataset(train_base, cid, word_to_cluster, words)
        if len(ds) == 0:
            print("  (no training samples — skipping)")
            continue

        loader = DataLoader(ds, batch_size=min(64, len(ds)), shuffle=True,
                            num_workers=4, drop_last=(len(ds) > 64))
        model = KinematicEncoder(num_classes=len(words)).to(device)
        model = train_model(model, loader, LOCAL_EPOCHS, device, f"Cluster {cid:02d}")

        local_models[cid] = model
        local_idx_maps[cid] = ds.idx_to_word

    print("\n=== Step 3: Hierarchical validation ===")
    val_base = KinematicDataset(JSON_VAL, DIR_VAL,
                                  word_to_idx=train_base.word_to_idx, is_train=False)
    val_loader = DataLoader(val_base, batch_size=256, shuffle=False, num_workers=4)

    router.eval()
    for m in local_models.values():
        m.eval()

    cluster_buffers: dict[int, list] = defaultdict(list)
    router_correct = total_samples = 0

    with torch.no_grad():
        for x, _, true_words in tqdm(val_loader, desc="Pass A — router"):
            x        = x.to(device)
            pred_cids = router(x).argmax(1).cpu().tolist()
            for tensor_cpu, pred_cid, true_word in zip(x.cpu(), pred_cids, true_words):
                total_samples += 1
                if pred_cid == word_to_cluster.get(true_word, -1):
                    router_correct += 1
                cluster_buffers[pred_cid].append((tensor_cpu, true_word))

    correct_top1 = correct_top5 = 0

    for cid, samples in tqdm(cluster_buffers.items(), desc="Pass B — local models"):
        if cid not in local_models:
            continue

        local_model = local_models[cid]
        idx_to_word = local_idx_maps[cid]
        k = min(5, len(idx_to_word))

        tensors = torch.stack([s[0] for s in samples]).to(device)
        true_words = [s[1] for s in samples]

        with torch.no_grad():
            logits    = local_model(tensors)
            top5_idxs = logits.topk(k, dim=1).indices.cpu().tolist()

        for i, tw in enumerate(true_words):
            top5_words = [idx_to_word.get(j, "") for j in top5_idxs[i]]
            if top5_words[0] == tw:
                correct_top1 += 1
            if tw in top5_words:
                correct_top5 += 1

    r_acc  = 100 * router_correct / total_samples if total_samples else 0.0
    t1_acc = 100 * correct_top1 / total_samples if total_samples else 0.0
    t5_acc = 100 * correct_top5 / total_samples if total_samples else 0.0

    print("\n" + "=" * 55)
    print("  WORD-CLUSTERED HIERARCHICAL RESULTS")
    print("=" * 55)
    print(f"  Validation samples          : {total_samples}")
    print(f"  Router cluster accuracy     : {r_acc:.2f}%")
    print(f"  End-to-end Top-1 accuracy   : {t1_acc:.2f}%")
    print(f"  End-to-end Top-5 accuracy   : {t5_acc:.2f}%")
    print("=" * 55)


if __name__ == '__main__':
    run_embedding_clustered_pipeline()