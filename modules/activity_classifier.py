"""
Activity Classifier Module — ResNet50 + LSTM
CNN extracts per-frame features, LSTM classifies activity over temporal windows.
Two heads: activity classification + object interaction classification.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import cv2
import os
from typing import Tuple, Optional
from collections import deque
from torchvision import transforms

from config import (
    DEVICE, FEATURE_DIM, FREEZE_BACKBONE_LAYERS,
    LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS, LSTM_DROPOUT,
    SEQUENCE_LENGTH, NUM_ACTIVITY_CLASSES, NUM_OBJECT_CLASSES,
    ACTIVITY_CLASSES, OBJECT_CLASSES, IMG_SIZE, BEST_MODEL_PATH
)


class CNNFeatureExtractor(nn.Module):
    """ResNet50 backbone for feature extraction."""

    def __init__(self, freeze_layers: int = FREEZE_BACKBONE_LAYERS):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.features = nn.Sequential(*list(resnet.children())[:-1])

        layers = list(self.features.children())
        for i, layer in enumerate(layers):
            if i < freeze_layers:
                for param in layer.parameters():
                    param.requires_grad = False

        self.output_dim = FEATURE_DIM

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x).flatten(1)


class ActivityLSTM(nn.Module):
    """LSTM with two classification heads: activity + held object."""

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=FEATURE_DIM,
            hidden_size=LSTM_HIDDEN_SIZE,
            num_layers=LSTM_NUM_LAYERS,
            dropout=LSTM_DROPOUT if LSTM_NUM_LAYERS > 1 else 0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(LSTM_DROPOUT)
        self.shared_fc = nn.Sequential(
            nn.Linear(LSTM_HIDDEN_SIZE, 256), nn.ReLU(), nn.Dropout(LSTM_DROPOUT),
        )
        self.activity_head = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, NUM_ACTIVITY_CLASSES),
        )
        self.object_head = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, NUM_OBJECT_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        lstm_out, _ = self.lstm(x)
        last_hidden = self.dropout(lstm_out[:, -1, :])
        shared = self.shared_fc(last_hidden)
        return self.activity_head(shared), self.object_head(shared)


class ActivityClassifier:
    """
    Full classification pipeline. Extracts CNN features per frame,
    buffers them per otter, and runs LSTM for temporal classification.
    """

    def __init__(self, device: str = DEVICE):
        self.device = device
        self.cnn = None
        self.lstm = None
        self._loaded = False
        self.feature_buffers = {}
        self.max_buffer_size = SEQUENCE_LENGTH

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def load_models(self, model_path: Optional[str] = None):
        if self._loaded:
            return

        print("[Activity] Loading ResNet50 feature extractor...")
        self.cnn = CNNFeatureExtractor().to(self.device)
        self.cnn.eval()

        print("[Activity] Loading LSTM classifier...")
        self.lstm = ActivityLSTM().to(self.device)

        load_path = model_path or str(BEST_MODEL_PATH)
        if os.path.exists(load_path):
            print(f"[Activity] Loading trained weights from {load_path}")
            checkpoint = torch.load(load_path, map_location=self.device, weights_only=False)
            if "lstm_state_dict" in checkpoint:
                self.lstm.load_state_dict(checkpoint["lstm_state_dict"])
            if "cnn_state_dict" in checkpoint:
                self.cnn.load_state_dict(checkpoint["cnn_state_dict"], strict=False)
            print("[Activity] Weights loaded.")
        else:
            print(f"[Activity] No trained weights at {load_path} — predictions will be random.")
            print("[Activity] Train first: python run.py --train-lstm")

        self.lstm.eval()
        self._loaded = True

    @torch.no_grad()
    def extract_features(self, cropped_image: np.ndarray) -> np.ndarray:
        if not self._loaded:
            self.load_models()
        rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb).unsqueeze(0).to(self.device)
        return self.cnn(tensor).cpu().numpy().flatten()

    def update_buffer(self, otter_id: int, features: np.ndarray):
        if otter_id not in self.feature_buffers:
            self.feature_buffers[otter_id] = deque(maxlen=self.max_buffer_size)
        self.feature_buffers[otter_id].append(features)

    @torch.no_grad()
    def classify(self, otter_id: int) -> Tuple[str, float, str, float]:
        if otter_id not in self.feature_buffers:
            return "unknown", 0.0, "none", 0.0

        buffer = self.feature_buffers[otter_id]
        if len(buffer) < 3:
            return "unknown", 0.0, "none", 0.0

        features_list = list(buffer)
        while len(features_list) < SEQUENCE_LENGTH:
            features_list.insert(0, features_list[0])
        features_list = features_list[-SEQUENCE_LENGTH:]

        sequence = torch.tensor(
            np.stack(features_list), dtype=torch.float32
        ).unsqueeze(0).to(self.device)

        act_logits, obj_logits = self.lstm(sequence)
        act_probs = torch.softmax(act_logits, dim=1)[0]
        obj_probs = torch.softmax(obj_logits, dim=1)[0]

        act_idx = act_probs.argmax().item()
        obj_idx = obj_probs.argmax().item()

        return (
            ACTIVITY_CLASSES[act_idx], act_probs[act_idx].item(),
            OBJECT_CLASSES[obj_idx], obj_probs[obj_idx].item(),
        )

    def classify_detections(self, detections) -> None:
        """Process all detections in a frame. Modifies detections in-place."""
        if not self._loaded:
            self.load_models()

        for det in detections:
            if det.cropped_image is None or det.cropped_image.size == 0:
                continue
            features = self.extract_features(det.cropped_image)
            self.update_buffer(det.otter_id, features)
            activity, act_conf, obj, obj_conf = self.classify(det.otter_id)
            det.activity = activity
            det.activity_conf = act_conf
            det.held_object = obj
            det.object_conf = obj_conf

    def reset_buffers(self):
        self.feature_buffers.clear()
