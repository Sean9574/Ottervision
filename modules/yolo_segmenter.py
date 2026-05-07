"""
YOLO Segmenter — YOLOv8s-seg on GPU 0
Single model, fp16, ~10ms inference (100fps) on A100.
Accepts live settings for dynamic conf/imgsz tuning.
"""

import numpy as np
import cv2
from typing import List, Dict
from dataclasses import dataclass, field

from config import YOLO_NANO_PATH, YOLO_SMALL_PATH, YOLO_DEVICE, YOLO_CONF, YOLO_IMG_SIZE


@dataclass
class OtterDetection:
    otter_id: int = 0
    bbox: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0, 0]))
    mask: np.ndarray = None
    confidence: float = 0.0
    activity: str = "active"
    held_object: str = "none"


class EnsembleSegmenter:
    """Loads the best available model (small preferred, nano fallback)."""

    def __init__(self):
        self.model = None
        self._loaded = False

    def load_model(self):
        from ultralytics import YOLO

        device = f"cuda:{YOLO_DEVICE}"

        # Prefer small, fall back to nano
        if YOLO_SMALL_PATH.exists():
            path = str(YOLO_SMALL_PATH)
        elif YOLO_NANO_PATH.exists():
            path = str(YOLO_NANO_PATH)
        else:
            print("[YOLO] ERROR: No model found. Run: python run.py --train")
            return

        print(f"[YOLO] Loading {path} on {device}...")
        self.model = YOLO(path)
        self.model.to(device)

        # Warmup
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        self.model(dummy, imgsz=YOLO_IMG_SIZE, conf=YOLO_CONF, verbose=False, half=True)

        print(f"[YOLO] Ready. Conf: {YOLO_CONF}, imgsz: {YOLO_IMG_SIZE}, fp16: True")
        self._loaded = True

    def segment_frame(self, frame, conf=None, imgsz=None, half=True, max_det=10):
        """Run YOLO on a frame. Returns list of OtterDetection."""
        if not self._loaded or self.model is None:
            return []

        results = self.model(
            frame,
            imgsz=imgsz or YOLO_IMG_SIZE,
            conf=conf or YOLO_CONF,
            verbose=False,
            device=f"cuda:{YOLO_DEVICE}",
            half=half,
            max_det=max_det,
        )[0]

        if results.boxes is None:
            return []

        detections = []
        h, w = frame.shape[:2]

        for i in range(len(results.boxes)):
            bbox = results.boxes.xyxy[i].cpu().numpy().astype(int)
            conf_val = results.boxes.conf[i].item()

            mask = None
            if results.masks is not None and i < len(results.masks):
                mask_data = results.masks.data[i].cpu().numpy()
                mask = cv2.resize(mask_data, (w, h), interpolation=cv2.INTER_LINEAR)
                mask = (mask > 0.5).astype(np.uint8)

            detections.append(OtterDetection(
                otter_id=i, bbox=bbox, mask=mask, confidence=conf_val,
            ))

        return detections

    def detections_to_json(self, detections, include_masks=True):
        """Convert to JSON-serializable format with simplified polygons."""
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
            if include_masks and det.mask is not None:
                contours, _ = cv2.findContours(det.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    contour = max(contours, key=cv2.contourArea)
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    if len(approx) >= 3:
                        d["polygon"] = approx.reshape(-1, 2).tolist()
            det_list.append(d)
        return det_list

    def is_loaded(self):
        return self._loaded