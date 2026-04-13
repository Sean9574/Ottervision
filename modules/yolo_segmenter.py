"""
Ensemble YOLO Segmenter — Runs small model (or nano fallback).
Accepts live settings from the web UI for dynamic conf/imgsz tuning.
"""

import torch
import numpy as np
import cv2
import threading
from typing import List, Dict
from dataclasses import dataclass, field

from config import (
    YOLO_NANO_PATH, YOLO_SMALL_PATH, YOLO_DEVICE,
    YOLO_CONF, YOLO_IMG_SIZE, YOLO_IOU_MERGE
)


@dataclass
class OtterDetection:
    otter_id: int = 0
    bbox: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0, 0]))
    mask: np.ndarray = None
    confidence: float = 0.0
    activity: str = "active"
    held_object: str = "none"
    source: str = ""


class EnsembleSegmenter:
    def __init__(self):
        self.nano = None
        self.small = None
        self._loaded = False

    def load_model(self):
        from ultralytics import YOLO

        device = f"cuda:{YOLO_DEVICE}"

        if YOLO_NANO_PATH.exists():
            print(f"[YOLO] Loading nano: {YOLO_NANO_PATH}")
            self.nano = YOLO(str(YOLO_NANO_PATH))
            if str(YOLO_NANO_PATH).endswith('.pt'):
                self.nano.to(device)

        if YOLO_SMALL_PATH.exists():
            print(f"[YOLO] Loading small: {YOLO_SMALL_PATH}")
            self.small = YOLO(str(YOLO_SMALL_PATH))
            if str(YOLO_SMALL_PATH).endswith('.pt'):
                self.small.to(device)

        if self.nano is None and self.small is None:
            print("[YOLO] ERROR: No models found! Train first with: python run.py --train")
            return

        # Warmup
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        if self.small:
            self.small(dummy, imgsz=YOLO_IMG_SIZE, conf=YOLO_CONF, verbose=False)
        elif self.nano:
            self.nano(dummy, imgsz=YOLO_IMG_SIZE, conf=YOLO_CONF, verbose=False)

        count = sum(1 for m in [self.nano, self.small] if m is not None)
        print(f"[YOLO] Ready: {count} model(s) loaded. Conf: {YOLO_CONF}")
        self._loaded = True

    def _run_model(self, model, frame, tag, conf=None, imgsz=None, half=True, max_det=10):
        """Run a single YOLO model with optional live settings override."""
        if model is None:
            return []

        results = model(
            frame,
            imgsz=imgsz or YOLO_IMG_SIZE,
            conf=conf or YOLO_CONF,
            verbose=False,
            device=f"cuda:{YOLO_DEVICE}",
            half=half,
            max_det=max_det,
        )[0]

        detections = []
        h, w = frame.shape[:2]

        if results.boxes is None:
            return []

        boxes = results.boxes
        masks = results.masks

        for i in range(len(boxes)):
            bbox = boxes.xyxy[i].cpu().numpy().astype(int)
            conf_val = boxes.conf[i].item()

            mask = None
            if masks is not None and i < len(masks):
                mask_data = masks.data[i].cpu().numpy()
                mask = cv2.resize(mask_data, (w, h), interpolation=cv2.INTER_LINEAR)
                mask = (mask > 0.5).astype(np.uint8)

            det = OtterDetection(
                otter_id=0,
                bbox=bbox,
                mask=mask,
                confidence=conf_val,
                source=tag,
            )
            detections.append(det)

        return detections

    def segment_frame(self, frame, conf=None, imgsz=None, half=True, max_det=10):
        """Run the best available model with live settings."""
        if not self._loaded:
            return []

        # Only run the better model (small preferred)
        model = self.small if self.small else self.nano
        if model is None:
            return []

        detections = self._run_model(
            model, frame, "small",
            conf=conf, imgsz=imgsz, half=half, max_det=max_det,
        )

        for idx, det in enumerate(detections):
            det.otter_id = idx

        return detections

    def detections_to_json(self, detections, include_masks=True):
        """Convert detections to JSON with optional polygon extraction."""
        det_list = []
        for det in detections:
            d = {
                "otter_id": det.otter_id,
                "bbox": det.bbox.tolist() if hasattr(det.bbox, 'tolist') else list(det.bbox),
                "confidence": round(det.confidence, 2),
                "activity": det.activity,
                "object": det.held_object,
                "polygon": None,
            }

            if include_masks and det.mask is not None:
                contours, _ = cv2.findContours(det.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    contour = max(contours, key=cv2.contourArea)
                    epsilon = 0.005 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    if len(approx) >= 3:
                        d["polygon"] = approx.reshape(-1, 2).tolist()

            det_list.append(d)
        return det_list

    def is_loaded(self):
        return self._loaded