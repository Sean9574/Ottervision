"""
Dataset Merger — Record live feed, label with DINO+SAM, merge into existing dataset.
Preserves your existing labeled data and adds new frames on top.

Usage:
    python modules/dataset_merger.py --record-minutes 60
    python modules/dataset_merger.py --frames-dir /path/to/existing/frames
"""

import os
import sys
import time
import subprocess
import torch
import torch.multiprocessing as mp
import numpy as np
import cv2
from PIL import Image
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, YOLO_DATASET_DIR, NUM_GPUS

DINO_MODEL_ID = "IDEA-Research/grounding-dino-base"
SAM_MODEL_ID = "facebook/sam-vit-huge"
DINO_TEXT_PROMPT = "otter . sea otter . river otter"
DINO_THRESHOLD = 0.35

LIVE_URL = "https://www.youtube.com/watch?v=_KXHUb0wFRE"


def record_live_feed(output_path: str, minutes: int = 60):
    """Record live feed for specified minutes."""
    if os.path.exists(output_path):
        print(f"[Record] {output_path} already exists, skipping.")
        return True

    print(f"[Record] Recording {minutes} min of live feed...")
    result = subprocess.run(
        ["yt-dlp", "-f", "best[height<=720]", "-g", LIVE_URL],
        capture_output=True, text=True, timeout=30
    )
    if result.returncode != 0:
        print(f"[Record] Error getting stream URL")
        return False

    stream_url = result.stdout.strip().split("\n")[0]
    seconds = minutes * 60

    subprocess.run([
        "ffmpeg", "-i", stream_url,
        "-t", str(seconds),
        "-c", "copy", "-v", "warning",
        output_path
    ], timeout=seconds + 120)

    if os.path.exists(output_path):
        size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"[Record] Done: {size:.0f} MB")
        return True
    return False


def extract_frames(video_path: str, output_dir: str, fps: float = 2.0, prefix: str = "live"):
    """Extract frames from video."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"[Extract] Extracting at {fps} fps...")
    subprocess.run([
        "ffmpeg", "-i", video_path,
        "-vf", f"fps={fps}", "-q:v", "2",
        os.path.join(output_dir, f"{prefix}_%06d.jpg"),
        "-v", "warning"
    ])
    count = len([f for f in os.listdir(output_dir) if f.startswith(prefix) and f.endswith(".jpg")])
    print(f"[Extract] {count} frames extracted")
    return count


def _label_worker(gpu_id, worker_id, frame_list, frames_dir, output_dir):
    """DINO + SAM labeling worker."""
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    tag = f"[GPU{gpu_id}-W{worker_id}]"

    print(f"{tag} Loading models on {device}...")
    from transformers import (
        AutoProcessor, AutoModelForZeroShotObjectDetection,
        SamModel, SamProcessor,
    )

    dino_proc = AutoProcessor.from_pretrained(DINO_MODEL_ID)
    dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_MODEL_ID).to(device)
    dino_model.eval()
    sam_proc = SamProcessor.from_pretrained(SAM_MODEL_ID)
    sam_model = SamModel.from_pretrained(SAM_MODEL_ID).to(device)
    sam_model.eval()

    print(f"{tag} Processing {len(frame_list)} frames.")

    frames_dir = Path(frames_dir)
    output_dir = Path(output_dir)
    img_train = output_dir / "images" / "train"
    img_val = output_dir / "images" / "val"
    lbl_train = output_dir / "labels" / "train"
    lbl_val = output_dir / "labels" / "val"

    local_det = 0
    start_time = time.time()

    for i, (global_idx, fname) in enumerate(frame_list):
        frame = cv2.imread(str(frames_dir / fname))
        if frame is None:
            continue

        h, w = frame.shape[:2]
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        with torch.no_grad():
            inputs = dino_proc(images=pil_image, text=DINO_TEXT_PROMPT, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = dino_model(**inputs)
            results = dino_proc.post_process_grounded_object_detection(
                outputs, inputs["input_ids"],
                threshold=DINO_THRESHOLD,
                target_sizes=[pil_image.size[::-1]]
            )[0]

        label_lines = []
        for bbox in results["boxes"]:
            bbox_np = bbox.cpu().numpy()
            with torch.no_grad():
                sam_inputs = sam_proc(pil_image, input_boxes=[[[bbox_np.tolist()]]], return_tensors="pt")
                sam_inputs = {k: v.to(device) for k, v in sam_inputs.items()}
                sam_out = sam_model(**sam_inputs)
                masks = sam_proc.image_processor.post_process_masks(
                    sam_out.pred_masks.cpu(),
                    sam_inputs["original_sizes"].cpu(),
                    sam_inputs["reshaped_input_sizes"].cpu()
                )
                best_idx = sam_out.iou_scores[0][0].argmax().item()
                binary_mask = masks[0][0][best_idx].numpy().astype(np.uint8)

            local_det += 1
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) < 3:
                continue
            points = []
            for pt in approx:
                points.extend([f"{pt[0][0]/w:.6f}", f"{pt[0][1]/h:.6f}"])
            label_lines.append("0 " + " ".join(points))

        # Skip frames with no detections
        if not label_lines:
            continue

        is_val = (global_idx % 5 == 0)
        img_dir = img_val if is_val else img_train
        lbl_dir = lbl_val if is_val else lbl_train

        dst_img = img_dir / fname
        if not dst_img.exists():
            cv2.imwrite(str(dst_img), frame)
        lbl_name = fname.replace(".jpg", ".txt")
        with open(lbl_dir / lbl_name, "w") as f:
            f.write("\n".join(label_lines))

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            speed = (i + 1) / elapsed
            remaining = (len(frame_list) - i - 1) / max(speed, 0.01) / 3600
            print(f"{tag} {i+1}/{len(frame_list)} | {local_det} det | {speed:.1f} f/s | ~{remaining:.1f}h left")

    del dino_model, dino_proc, sam_model, sam_proc
    torch.cuda.empty_cache()
    print(f"{tag} DONE. {local_det} detections in {(time.time()-start_time)/60:.1f} min")


def run_labeling(frames_dir: str, output_dir: str):
    """Run DINO+SAM across all GPUs."""
    frames = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])
    if not frames:
        print("[Label] No frames found.")
        return

    print(f"[Label] Labeling {len(frames)} frames across {NUM_GPUS} GPU(s)...")

    indexed = list(enumerate(frames))
    num_gpus = max(NUM_GPUS, 1)

    chunks = []
    per_gpu = len(indexed) // num_gpus
    for gpu_id in range(num_gpus):
        start = gpu_id * per_gpu
        end = start + per_gpu if gpu_id < num_gpus - 1 else len(indexed)
        chunks.append((gpu_id, gpu_id, indexed[start:end]))

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    procs = []
    for gpu_id, wid, chunk in chunks:
        p = mp.Process(target=_label_worker, args=(gpu_id, wid, chunk, frames_dir, output_dir))
        procs.append(p)
        p.start()
    for p in procs:
        p.join()


def merge_live_feed(record_minutes: int = 60):
    """Record live feed, extract frames, label, merge into existing dataset."""
    work_dir = DATA_DIR / "live_merge"
    video_path = str(work_dir / "live_recording.mp4")
    frames_dir = str(work_dir / "frames")
    output_dir = str(YOLO_DATASET_DIR)

    os.makedirs(str(work_dir), exist_ok=True)
    for d in ["images/train", "images/val", "labels/train", "labels/val"]:
        os.makedirs(os.path.join(output_dir, d), exist_ok=True)

    # Count existing
    existing_train = len(os.listdir(os.path.join(output_dir, "images/train")))
    existing_val = len(os.listdir(os.path.join(output_dir, "images/val")))
    print(f"[Merge] Existing dataset: {existing_train} train, {existing_val} val")

    # Record
    print(f"\n{'='*60}")
    print(f"STEP 1: Recording {record_minutes} min of live feed")
    print(f"{'='*60}")
    record_live_feed(video_path, record_minutes)

    # Extract
    print(f"\n{'='*60}")
    print(f"STEP 2: Extracting frames")
    print(f"{'='*60}")
    extract_frames(video_path, frames_dir, fps=2.0, prefix="live")

    # Label
    print(f"\n{'='*60}")
    print(f"STEP 3: Labeling with DINO + SAM")
    print(f"{'='*60}")
    run_labeling(frames_dir, output_dir)

    # Write dataset yaml
    yaml_path = os.path.join(output_dir, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"""path: {output_dir}
train: images/train
val: images/val

names:
  0: otter
""")

    # Count final
    final_train = len(os.listdir(os.path.join(output_dir, "images/train")))
    final_val = len(os.listdir(os.path.join(output_dir, "images/val")))
    added_train = final_train - existing_train
    added_val = final_val - existing_val

    print(f"\n{'='*60}")
    print(f"MERGE COMPLETE")
    print(f"{'='*60}")
    print(f"Added:  {added_train} train + {added_val} val images from live feed")
    print(f"Total:  {final_train} train + {final_val} val images")
    print(f"\nNext: python run.py --train")


def label_existing_frames(frames_dir: str):
    """Label existing frames and add to dataset (for re-labeling old frames)."""
    output_dir = str(YOLO_DATASET_DIR)
    for d in ["images/train", "images/val", "labels/train", "labels/val"]:
        os.makedirs(os.path.join(output_dir, d), exist_ok=True)

    existing_train = len(os.listdir(os.path.join(output_dir, "images/train")))
    print(f"[Label] Existing dataset: {existing_train} train images")
    print(f"[Label] Labeling frames from: {frames_dir}")

    run_labeling(frames_dir, output_dir)

    # Write yaml
    with open(os.path.join(output_dir, "dataset.yaml"), "w") as f:
        f.write(f"""path: {output_dir}
train: images/train
val: images/val

names:
  0: otter
""")

    final_train = len(os.listdir(os.path.join(output_dir, "images/train")))
    print(f"[Label] Final dataset: {final_train} train images")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--record-minutes", type=int, default=60, help="Minutes of live feed to record")
    parser.add_argument("--frames-dir", type=str, default="", help="Label existing frames instead of recording")
    args = parser.parse_args()

    if args.frames_dir:
        label_existing_frames(args.frames_dir)
    else:
        merge_live_feed(args.record_minutes)
