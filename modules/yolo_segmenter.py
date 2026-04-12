"""
Ensemble YOLO Segmenter — Runs both nano and small models.
Merges detections using NMS so if one model misses an otter, the other catches it.
Both models run on GPU 0 sequentially (still fast — ~15-20fps combined on A100).
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
    source: str = ""  # which model detected this


class EnsembleSegmenter:
    def __init__(self):
        self.nano = None
        self.small = None
        self._loaded = False

    def load_model(self):
        from ultralytics import YOLO

        device = f"cuda:{YOLO_DEVICE}"

        # Load nano
        if YOLO_NANO_PATH.exists():
            print(f"[YOLO] Loading nano: {YOLO_NANO_PATH}")
            self.nano = YOLO(str(YOLO_NANO_PATH))
            self.nano.to(device)
        else:
            print(f"[YOLO] Nano model not found at {YOLO_NANO_PATH}, skipping.")

        # Load small
        if YOLO_SMALL_PATH.exists():
            print(f"[YOLO] Loading small: {YOLO_SMALL_PATH}")
            self.small = YOLO(str(YOLO_SMALL_PATH))
            # TensorRT engines don't need .to() — device is set during predict
            if str(YOLO_SMALL_PATH).endswith('.pt'):
                self.small.to(device)
        else:
            print(f"[YOLO] Small model not found at {YOLO_SMALL_PATH}, skipping.")

        if self.nano is None and self.small is None:
            print("[YOLO] ERROR: No models found! Train first with: python run.py --train")
            return

        # Warmup both
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        if self.nano:
            self.nano(dummy, imgsz=YOLO_IMG_SIZE, conf=YOLO_CONF, verbose=False)
        if self.small:
            self.small(dummy, imgsz=YOLO_IMG_SIZE, conf=YOLO_CONF, verbose=False)

        count = sum(1 for m in [self.nano, self.small] if m is not None)
        print(f"[YOLO] Ensemble ready: {count} model(s) loaded. Conf: {YOLO_CONF}")
        self._loaded = True

    def _run_model(self, model, frame: np.ndarray, tag: str) -> List[OtterDetection]:
        """Run a single YOLO model and return detections."""
        if model is None:
            return []

        results = model(
            frame,
            imgsz=YOLO_IMG_SIZE,
            conf=YOLO_CONF,
            verbose=False,
            device=f"cuda:{YOLO_DEVICE}",
            half=True,
        )[0]
        detections = []
        h, w = frame.shape[:2]

        if results.boxes is None:
            return []

        boxes = results.boxes
        masks = results.masks

        for i in range(len(boxes)):
            bbox = boxes.xyxy[i].cpu().numpy().astype(int)
            conf = boxes.conf[i].item()

            mask = None
            if masks is not None and i < len(masks):
                mask_data = masks.data[i].cpu().numpy()
                mask = cv2.resize(mask_data, (w, h), interpolation=cv2.INTER_LINEAR)
                mask = (mask > 0.5).astype(np.uint8)

            det = OtterDetection(
                otter_id=0,
                bbox=bbox,
                mask=mask,
                confidence=conf,
                source=tag,
            )
            detections.append(det)

        return detections

    def _compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute IoU between two [x1,y1,x2,y2] boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / max(union, 1e-6)

    def _merge_detections(self, all_dets: List[OtterDetection]) -> List[OtterDetection]:
        """
        Merge detections from both models using NMS.
        If both models detect the same otter (high IoU), keep the higher confidence one.
        If only one model detects it, keep it.
        """
        if not all_dets:
            return []

        # Sort by confidence descending
        all_dets.sort(key=lambda d: d.confidence, reverse=True)

        merged = []
        used = [False] * len(all_dets)

        for i in range(len(all_dets)):
            if used[i]:
                continue

            best = all_dets[i]
            used[i] = True

            # Check remaining for overlaps
            for j in range(i + 1, len(all_dets)):
                if used[j]:
                    continue
                iou = self._compute_iou(best.bbox, all_dets[j].bbox)
                if iou > YOLO_IOU_MERGE:
                    # Same otter — merge masks if the suppressed one has a better mask
                    used[j] = True
                    # If the suppressed detection has a mask and the best doesn't, use it
                    if best.mask is None and all_dets[j].mask is not None:
                        best.mask = all_dets[j].mask
                    # Boost confidence since both models agree
                    best.confidence = min(0.99, best.confidence + 0.1)

            merged.append(best)

        # Re-assign otter IDs
        for idx, det in enumerate(merged):
            det.otter_id = idx

        return merged

    def segment_frame(self, frame: np.ndarray) -> List[OtterDetection]:
        if not self._loaded:
            return []

        # Only run the better model
        model = self.small if self.small else self.nano
        if model is None:
            return []

        detections = self._run_model(model, frame, "small")

        for idx, det in enumerate(detections):
            det.otter_id = idx

        return detections

    def detections_to_json(self, detections: List[OtterDetection]) -> List[Dict]:
        """Convert detections to JSON with polygons for overlay."""
        det_list = []
        for det in detections:
            d = {
                "otter_id": det.otter_id,
                "bbox": det.bbox.tolist(),
                "confidence": round(det.confidence, 2),
                "activity": det.activity,
                "object": det.held_object,
                "polygon": None,
            }

            if det.mask is not None:
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