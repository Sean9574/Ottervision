"""
Data Prep Module — GroundingDINO + SAM (OFFLINE ONLY)
Runs on your downloaded training videos to auto-detect and segment otters.
Generates YOLO-format labels + cropped images for training.
This is NEVER used during live inference.
"""

import torch
import numpy as np
import cv2
import os
import json
from PIL import Image
from pathlib import Path
from typing import List, Dict, Tuple

from config import (
    DEVICE, DINO_MODEL_ID, SAM_MODEL_ID,
    DINO_TEXT_PROMPT, DINO_THRESHOLD,
    FRAME_DIR, LABEL_DIR, YOLO_DATASET_DIR,
    PREP_DTYPE, HW_TIER
)


class DataPrepPipeline:
    """
    Heavy model pipeline for generating training data.
    GroundingDINO finds otters → SAM generates masks → export to YOLO format.
    """

    def __init__(self, device: str = DEVICE):
        self.device = device
        self.dino_model = None
        self.dino_processor = None
        self.sam_model = None
        self.sam_processor = None

    def load_models(self):
        """Load GroundingDINO and SAM. Only called during data prep."""
        print(f"[DataPrep] Hardware tier: {HW_TIER.upper()}")
        print(f"[DataPrep] Precision: {'fp16' if PREP_DTYPE == torch.float16 else 'fp32'}")

        print(f"[DataPrep] Loading GroundingDINO...")
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

        self.dino_processor = AutoProcessor.from_pretrained(DINO_MODEL_ID)
        self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            DINO_MODEL_ID, torch_dtype=PREP_DTYPE
        ).to(self.device)
        self.dino_model.eval()
        print("[DataPrep] GroundingDINO loaded.")

        print(f"[DataPrep] Loading SAM ({SAM_MODEL_ID.split('/')[-1]})...")
        from transformers import SamModel, SamProcessor

        self.sam_processor = SamProcessor.from_pretrained(SAM_MODEL_ID)
        self.sam_model = SamModel.from_pretrained(
            SAM_MODEL_ID, torch_dtype=PREP_DTYPE
        ).to(self.device)
        self.sam_model.eval()
        print("[DataPrep] SAM loaded.")

    def unload_models(self):
        """Free GPU memory after data prep is done."""
        del self.dino_model, self.dino_processor
        del self.sam_model, self.sam_processor
        self.dino_model = self.dino_processor = None
        self.sam_model = self.sam_processor = None
        torch.cuda.empty_cache()
        print("[DataPrep] Models unloaded, GPU memory freed.")

    @torch.no_grad()
    def detect_otters(self, frame: np.ndarray) -> List[Dict]:
        """Detect otters in a frame using GroundingDINO."""
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        inputs = self.dino_processor(
            images=pil_image,
            text=DINO_TEXT_PROMPT,
            return_tensors="pt"
        ).to(self.device)

        outputs = self.dino_model(**inputs)

        results = self.dino_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=DINO_THRESHOLD,
            target_sizes=[pil_image.size[::-1]]
        )[0]

        detections = []
        for bbox, score, label in zip(results["boxes"], results["scores"], results["labels"]):
            detections.append({
                "bbox": bbox.cpu().numpy(),
                "confidence": score.item(),
                "label": label,
            })
        return detections

    @torch.no_grad()
    def generate_mask(self, frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """Generate a segmentation mask for a single detection using SAM."""
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        bbox_list = bbox.tolist()
        inputs = self.sam_processor(
            pil_image,
            input_boxes=[[[bbox_list]]],
            return_tensors="pt"
        ).to(self.device)

        outputs = self.sam_model(**inputs)

        predicted_masks = self.sam_processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu()
        )

        mask = predicted_masks[0][0]
        iou_scores = outputs.iou_scores[0][0]
        best_idx = iou_scores.argmax().item()
        return mask[best_idx].numpy().astype(np.uint8)

    def process_frame(self, frame: np.ndarray) -> List[Dict]:
        """Full pipeline: detect otters + generate masks for one frame."""
        detections = self.detect_otters(frame)
        results = []

        for det in detections:
            mask = self.generate_mask(frame, det["bbox"])
            bbox = det["bbox"]
            x1, y1, x2, y2 = map(int, bbox)
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            crop = frame[y1:y2, x1:x2].copy()
            if crop.size == 0:
                continue

            results.append({
                "bbox": [x1, y1, x2, y2],
                "mask": mask,
                "crop": crop,
                "confidence": det["confidence"],
            })

        return results

    def bbox_to_yolo_format(self, bbox, img_w, img_h) -> str:
        """Convert [x1,y1,x2,y2] to YOLO format: class cx cy w h (normalized)."""
        x1, y1, x2, y2 = bbox
        cx = ((x1 + x2) / 2) / img_w
        cy = ((y1 + y2) / 2) / img_h
        bw = (x2 - x1) / img_w
        bh = (y2 - y1) / img_h
        return f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"

    def mask_to_yolo_seg_format(self, mask: np.ndarray, img_w: int, img_h: int) -> str:
        """Convert binary mask to YOLO segmentation format: class x1 y1 x2 y2 ... (normalized polygon)."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return ""

        # Take largest contour
        contour = max(contours, key=cv2.contourArea)

        # Simplify polygon
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) < 3:
            return ""

        # Format: class_id x1 y1 x2 y2 ...
        points = []
        for point in approx:
            px = point[0][0] / img_w
            py = point[0][1] / img_h
            points.extend([f"{px:.6f}", f"{py:.6f}"])

        return "0 " + " ".join(points)

    def process_all_frames(
        self,
        frames_dir: str = str(FRAME_DIR),
        output_dir: str = str(YOLO_DATASET_DIR),
        export_crops: bool = True,
        use_seg: bool = True,
    ) -> int:
        """
        Process all extracted frames: detect otters, generate masks, export YOLO labels.

        Args:
            frames_dir: Directory containing extracted video frames
            output_dir: YOLO dataset output directory
            export_crops: Also save cropped otter images for activity labeling
            use_seg: Export segmentation polygons (True) or just bboxes (False)

        Returns:
            Total number of otter detections
        """
        frames_dir = Path(frames_dir)
        output_dir = Path(output_dir)

        img_train_dir = output_dir / "images" / "train"
        img_val_dir = output_dir / "images" / "val"
        lbl_train_dir = output_dir / "labels" / "train"
        lbl_val_dir = output_dir / "labels" / "val"
        crops_dir = frames_dir.parent / "crops"

        for d in [img_train_dir, img_val_dir, lbl_train_dir, lbl_val_dir]:
            d.mkdir(parents=True, exist_ok=True)
        if export_crops:
            crops_dir.mkdir(parents=True, exist_ok=True)

        frame_files = sorted([
            f for f in os.listdir(frames_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        if not frame_files:
            print("[DataPrep] No frames found. Run --extract-frames first.")
            return 0

        print(f"[DataPrep] Processing {len(frame_files)} frames...")
        total_detections = 0
        crop_index = 0
        crop_metadata = {}

        for i, fname in enumerate(frame_files):
            frame_path = str(frames_dir / fname)
            frame = cv2.imread(frame_path)
            if frame is None:
                continue

            h, w = frame.shape[:2]
            results = self.process_frame(frame)

            # 80/20 train/val split
            is_val = (i % 5 == 0)
            img_dir = img_val_dir if is_val else img_train_dir
            lbl_dir = lbl_val_dir if is_val else lbl_train_dir

            # Copy image to YOLO dataset
            dst_img = img_dir / fname
            if not dst_img.exists():
                cv2.imwrite(str(dst_img), frame)

            # Write YOLO labels
            label_lines = []
            for det in results:
                total_detections += 1

                if use_seg and det["mask"] is not None:
                    line = self.mask_to_yolo_seg_format(det["mask"], w, h)
                else:
                    line = self.bbox_to_yolo_format(det["bbox"], w, h)

                if line:
                    label_lines.append(line)

                # Save crop for activity labeling
                if export_crops and det["crop"].size > 0:
                    crop_name = f"crop_{crop_index:06d}.jpg"
                    cv2.imwrite(str(crops_dir / crop_name), det["crop"])
                    crop_metadata[crop_name] = {
                        "source_frame": fname,
                        "bbox": det["bbox"],
                        "confidence": det["confidence"],
                    }
                    crop_index += 1

            lbl_file = lbl_dir / fname.replace(".jpg", ".txt").replace(".png", ".txt")
            with open(lbl_file, "w") as f:
                f.write("\n".join(label_lines))

            if (i + 1) % 100 == 0:
                print(f"[DataPrep] Processed {i+1}/{len(frame_files)} frames, {total_detections} detections")

        # Save crop metadata
        if export_crops:
            meta_path = crops_dir / "metadata.json"
            with open(meta_path, "w") as f:
                json.dump(crop_metadata, f, indent=2)

        # Write YOLO dataset config
        yaml_content = f"""path: {output_dir}
train: images/train
val: images/val

names:
  0: otter
"""
        with open(output_dir / "dataset.yaml", "w") as f:
            f.write(yaml_content)

        print(f"[DataPrep] Done! {total_detections} otter detections across {len(frame_files)} frames")
        print(f"[DataPrep] YOLO dataset saved to {output_dir}")
        if export_crops:
            print(f"[DataPrep] {crop_index} otter crops saved to {crops_dir}")

        return total_detections
