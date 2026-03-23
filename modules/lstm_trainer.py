"""
LSTM Trainer Module
Trains the ResNet50 + LSTM activity classifier on labeled otter crops.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import os
import json
import time
from pathlib import Path
from typing import Dict

from config import (
    DEVICE, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
    NUM_EPOCHS, EARLY_STOPPING_PATIENCE, SEQUENCE_LENGTH,
    TRAIN_SPLIT, VAL_SPLIT, ACTIVITY_CLASSES, OBJECT_CLASSES,
    ACTIVITY_MODEL_PATH, BEST_MODEL_PATH, MODEL_DIR, IMG_SIZE
)
from modules.activity_classifier import CNNFeatureExtractor, ActivityLSTM


class SequenceDataset(Dataset):
    """Dataset of CNN feature sequences for LSTM training."""

    def __init__(self, features, activity_labels, object_labels, seq_len=SEQUENCE_LENGTH):
        self.features = features
        self.activity_labels = activity_labels
        self.object_labels = object_labels
        self.seq_len = seq_len
        self.num_sequences = max(0, len(features) - seq_len + 1)

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        seq = torch.tensor(self.features[idx:idx + self.seq_len], dtype=torch.float32)
        act = int(self.activity_labels[idx + self.seq_len - 1])
        obj = int(self.object_labels[idx + self.seq_len - 1])
        return seq, act, obj


class LSTMTrainer:
    """Two-stage training: extract CNN features → train LSTM."""

    def __init__(self, device: str = DEVICE):
        self.device = device
        self.cnn = None
        self.lstm = None
        self.history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    def build_models(self):
        self.cnn = CNNFeatureExtractor().to(self.device)
        self.lstm = ActivityLSTM().to(self.device)

    def extract_features(self, crops_dir: str, labels_file: str) -> str:
        """Extract CNN features from labeled crops and cache them."""
        import cv2
        from torchvision import transforms

        cache_path = str(MODEL_DIR / "cached_features.npz")
        if os.path.exists(cache_path):
            print(f"[LSTMTrain] Using cached features: {cache_path}")
            return cache_path

        with open(labels_file) as f:
            labels = json.load(f)

        act_to_idx = {a: i for i, a in enumerate(ACTIVITY_CLASSES)}
        obj_to_idx = {o: i for i, o in enumerate(OBJECT_CLASSES)}

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.cnn.eval()
        all_features, all_act, all_obj = [], [], []

        sorted_names = sorted(labels.keys())
        print(f"[LSTMTrain] Extracting features from {len(sorted_names)} crops...")

        with torch.no_grad():
            batch_imgs = []
            batch_act = []
            batch_obj = []

            for i, name in enumerate(sorted_names):
                img_path = os.path.join(crops_dir, name)
                if not os.path.exists(img_path):
                    continue

                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                label = labels[name]
                batch_imgs.append(transform(img))
                batch_act.append(act_to_idx.get(label["activity"], 0))
                batch_obj.append(obj_to_idx.get(label["object"], len(OBJECT_CLASSES) - 1))

                if len(batch_imgs) == 32 or i == len(sorted_names) - 1:
                    tensor = torch.stack(batch_imgs).to(self.device)
                    feats = self.cnn(tensor).cpu().numpy()
                    all_features.append(feats)
                    all_act.extend(batch_act)
                    all_obj.extend(batch_obj)
                    batch_imgs, batch_act, batch_obj = [], [], []

                    if (i + 1) % 500 == 0:
                        print(f"[LSTMTrain]   {i+1}/{len(sorted_names)} processed")

        features = np.concatenate(all_features, axis=0)
        np.savez(cache_path, features=features,
                 activity_labels=np.array(all_act), object_labels=np.array(all_obj))
        print(f"[LSTMTrain] Cached {len(features)} features to {cache_path}")
        return cache_path

    def train(self, cache_path: str, num_epochs: int = NUM_EPOCHS, lr: float = LEARNING_RATE) -> Dict:
        data = np.load(cache_path)
        dataset = SequenceDataset(data["features"], data["activity_labels"], data["object_labels"])

        total = len(dataset)
        train_n = int(total * TRAIN_SPLIT)
        val_n = int(total * VAL_SPLIT)
        test_n = total - train_n - val_n
        train_set, val_set, test_set = random_split(dataset, [train_n, val_n, test_n])

        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

        print(f"[LSTMTrain] Train: {train_n}, Val: {val_n}, Test: {test_n}")

        act_criterion = nn.CrossEntropyLoss()
        obj_criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.lstm.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)

        best_val_loss = float("inf")
        patience = 0

        for epoch in range(num_epochs):
            # Train
            self.lstm.train()
            t_loss, t_correct, t_total = 0, 0, 0
            for seqs, acts, objs in train_loader:
                seqs, acts, objs = seqs.to(self.device), acts.to(self.device), objs.to(self.device)
                optimizer.zero_grad()
                a_logits, o_logits = self.lstm(seqs)
                loss = act_criterion(a_logits, acts) + obj_criterion(o_logits, objs)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.lstm.parameters(), 1.0)
                optimizer.step()
                t_loss += loss.item() * seqs.size(0)
                t_correct += (a_logits.argmax(1) == acts).sum().item()
                t_total += seqs.size(0)

            # Validate
            self.lstm.eval()
            v_loss, v_correct, v_total = 0, 0, 0
            with torch.no_grad():
                for seqs, acts, objs in val_loader:
                    seqs, acts, objs = seqs.to(self.device), acts.to(self.device), objs.to(self.device)
                    a_logits, o_logits = self.lstm(seqs)
                    loss = act_criterion(a_logits, acts) + obj_criterion(o_logits, objs)
                    v_loss += loss.item() * seqs.size(0)
                    v_correct += (a_logits.argmax(1) == acts).sum().item()
                    v_total += seqs.size(0)

            t_loss /= t_total
            v_loss /= v_total
            scheduler.step(v_loss)

            self.history["train_loss"].append(t_loss)
            self.history["val_loss"].append(v_loss)
            self.history["train_acc"].append(t_correct / t_total)
            self.history["val_acc"].append(v_correct / v_total)

            print(f"[Epoch {epoch+1}/{num_epochs}] "
                  f"Loss: {t_loss:.4f}/{v_loss:.4f} | "
                  f"Acc: {t_correct/t_total:.3f}/{v_correct/v_total:.3f}")

            if v_loss < best_val_loss:
                best_val_loss = v_loss
                patience = 0
                torch.save({
                    "lstm_state_dict": self.lstm.state_dict(),
                    "cnn_state_dict": self.cnn.state_dict(),
                }, str(BEST_MODEL_PATH))
                print(f"  → Best model saved (val_loss: {v_loss:.4f})")
            else:
                patience += 1
                if patience >= EARLY_STOPPING_PATIENCE:
                    print(f"[LSTMTrain] Early stopping at epoch {epoch+1}")
                    break

        with open(str(MODEL_DIR / "training_history.json"), "w") as f:
            json.dump(self.history, f, indent=2)

        return self.history
