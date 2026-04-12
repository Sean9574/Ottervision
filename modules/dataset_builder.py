"""
Dataset Builder — Multi-Source DINO + SAM Auto-Labeling
Supports pre-recorded videos AND live streams (records 30 min of live).
Edit the SOURCES list below and run: python modules/dataset_builder.py
"""

import os
import sys
import time
import random
import subprocess
import torch
import torch.multiprocessing as mp
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, YOLO_DATASET_DIR, NUM_GPUS

# ============================================================
# EDIT THESE — set live=True for live streams
# ============================================================
SOURCES = [
    {"url": "https://www.youtube.com/watch?v=LLh0WOuP89Y", "live": False},
    {"url": "https://www.youtube.com/watch?v=Sij74m_wDT4", "live": False},
    {"url": "https://www.youtube.com/watch?v=9mg9PoFEX2U", "live": True},
]

TOTAL_FRAMES = 50000
FPS = 2.0
WORKERS_PER_GPU = 1
LIVE_RECORD_MINUTES = 30
# ============================================================

DINO_MODEL_ID = "IDEA-Research/grounding-dino-base"
SAM_MODEL_ID = "facebook/sam-vit-huge"
DINO_TEXT_PROMPT = "otter . sea otter . river otter"
DINO_THRESHOLD = 0.35


def download_video(url: str, output_dir: str, index: int, is_live: bool = False) -> str:
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"source_{index:03d}.mp4")

    if os.path.exists(output_path):
        print(f"[Download] Source {index} already exists, skipping.")
        return output_path

    if is_live:
        record_seconds = LIVE_RECORD_MINUTES * 60
        print(f"[Download] Source {index} (LIVE): Recording {LIVE_RECORD_MINUTES} min from {url}")

        # Get the direct stream URL
        result = subprocess.run(
            ["yt-dlp", "-f", "best[height<=720]", "-g", url],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            print(f"[Download] Error getting stream URL: {result.stderr[:200]}")
            return ""

        stream_url = result.stdout.strip().split("\n")[0]

        # Record with ffmpeg for exact duration
        result = subprocess.run([
            "ffmpeg",
            "-i", stream_url,
            "-t", str(record_seconds),
            "-c", "copy",
            "-v", "warning",
            output_path
        ], capture_output=True, text=True, timeout=record_seconds + 120)

        if not os.path.exists(output_path):
            print(f"[Download] Error: Recording failed")
            return ""

        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"[Download] Source {index} (LIVE): Recorded {size_mb:.0f} MB")

    else:
        print(f"[Download] Source {index}: {url}")
        result = subprocess.run([
            "yt-dlp", "-f", "best[height<=720]",
            "-o", output_path,
            "--merge-output-format", "mp4",
            url
        ], capture_output=True, text=True)

        if result.returncode != 0:
            print(f"[Download] Error: {result.stderr[:200]}")
            return ""

    print(f"[Download] Source {index} saved: {output_path}")
    return output_path


def split_video_for_web(video_path: str, web_video_dir: str, segment_minutes: int = 5):
    os.makedirs(web_video_dir, exist_ok=True)
    name = Path(video_path).stem

    existing = [f for f in os.listdir(web_video_dir) if f.startswith(name + "_seg")]
    if existing:
        print(f"[Split] {name} already split ({len(existing)} segments), skipping.")
        return len(existing)

    print(f"[Split] Splitting {name} into {segment_minutes}-min segments...")
    subprocess.run([
        "ffmpeg", "-i", video_path,
        "-c", "copy",
        "-segment_time", str(segment_minutes * 60),
        "-f", "segment",
        "-reset_timestamps", "1",
        os.path.join(web_video_dir, f"{name}_seg%04d.mp4"),
        "-v", "warning"
    ], capture_output=True, text=True)

    segments = [f for f in os.listdir(web_video_dir) if f.startswith(name + "_seg")]
    print(f"[Split] {name}: {len(segments)} segments saved to {web_video_dir}")
    return len(segments)


def extract_frames(video_path: str, output_dir: str, fps: float, max_frames: int, prefix: str) -> int:
    os.makedirs(output_dir, exist_ok=True)

    probe = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", video_path
    ], capture_output=True, text=True)
    duration = float(probe.stdout.strip()) if probe.returncode == 0 else 0

    actual_fps = fps
    if duration > 0 and int(duration * fps) > max_frames:
        actual_fps = max_frames / duration
        print(f"[Extract] Adjusting fps from {fps} to {actual_fps:.2f} to stay within {max_frames} frames")

    print(f"[Extract] {prefix}: extracting at {actual_fps:.2f} fps from {duration:.0f}s video...")

    subprocess.run([
        "ffmpeg", "-i", video_path,
        "-vf", f"fps={actual_fps}",
        "-q:v", "2",
        os.path.join(output_dir, f"{prefix}_%06d.jpg"),
        "-v", "warning"
    ], capture_output=True, text=True)

    count = len([f for f in os.listdir(output_dir) if f.startswith(prefix) and f.endswith(".jpg")])
    print(f"[Extract] {prefix}: {count} frames extracted")
    return count


def _label_worker(gpu_id, worker_id, frame_list, frames_dir, output_dir):
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

    print(f"{tag} Ready. Processing {len(frame_list)} frames.")

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

        is_val = (global_idx % 5 == 0)
        img_dir = img_val if is_val else img_train
        lbl_dir = lbl_val if is_val else lbl_train

        dst_img = img_dir / fname
        if not dst_img.exists():
            cv2.imwrite(str(dst_img), frame)

        lbl_name = fname.replace(".jpg", ".txt").replace(".png", ".txt")
        with open(lbl_dir / lbl_name, "w") as f:
            f.write("\n".join(label_lines))

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            speed = (i + 1) / elapsed
            remaining = (len(frame_list) - i - 1) / max(speed, 0.01) / 3600
            print(f"{tag} {i+1}/{len(frame_list)} | {local_det} det | {speed:.1f} f/s | ~{remaining:.1f}h left")

    del dino_model, dino_proc, sam_model, sam_proc
    torch.cuda.empty_cache()
    elapsed = time.time() - start_time
    print(f"{tag} DONE. {local_det} detections in {elapsed/60:.1f} min")


def build_dataset(
    sources: List[dict],
    total_frames: int = 50000,
    fps: float = 2.0,
    workers_per_gpu: int = 1,
    output_dir: str = str(YOLO_DATASET_DIR),
):
    work_dir = DATA_DIR / "dataset_build"
    video_dir = work_dir / "videos"
    frames_dir = work_dir / "frames"
    output_dir = Path(output_dir)

    for d in [video_dir, frames_dir]:
        d.mkdir(parents=True, exist_ok=True)
    for d in ["images/train", "images/val", "labels/train", "labels/val"]:
        (output_dir / d).mkdir(parents=True, exist_ok=True)

    # Step 1: Download + split
    print(f"\n{'='*60}")
    print(f"STEP 1: Downloading {len(sources)} source(s)")
    print(f"{'='*60}")

    video_paths = []
    for i, src in enumerate(sources):
        is_live = src.get("live", False)
        if is_live:
            print(f"[Download] Source {i} is LIVE — will record {LIVE_RECORD_MINUTES} min")
        path = download_video(src["url"], str(video_dir), i, is_live=is_live)
        if path:
            video_paths.append(path)
            split_video_for_web(path, str(DATA_DIR / "videos"), segment_minutes=5)

    if not video_paths:
        print("[Error] No videos downloaded.")
        return

    # Step 2: Extract frames
    print(f"\n{'='*60}")
    print(f"STEP 2: Extracting {total_frames} frames across {len(video_paths)} source(s)")
    print(f"{'='*60}")

    frames_per_source = total_frames // len(video_paths)
    print(f"Target: {frames_per_source} frames per source")

    for i, vpath in enumerate(video_paths):
        extract_frames(vpath, str(frames_dir), fps, frames_per_source, f"src{i:03d}")

    all_frames = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])

    if len(all_frames) > total_frames:
        random.shuffle(all_frames)
        for f in all_frames[total_frames:]:
            os.remove(str(frames_dir / f))
        all_frames = all_frames[:total_frames]

    print(f"[Extract] Total frames: {len(all_frames)}")
    for i in range(len(video_paths)):
        count = sum(1 for f in all_frames if f.startswith(f"src{i:03d}"))
        print(f"  Source {i}: {count} frames")

    # Step 3: DINO + SAM
    print(f"\n{'='*60}")
    print(f"STEP 3: Auto-labeling {len(all_frames)} frames with GroundingDINO + SAM")
    print(f"{'='*60}")

    indexed = list(enumerate(all_frames))
    num_gpus = max(NUM_GPUS, 1)
    total_workers = num_gpus * workers_per_gpu if NUM_GPUS > 0 else 1

    chunks = []
    per_worker = len(indexed) // total_workers
    wid = 0
    for gpu_id in range(num_gpus):
        for w in range(workers_per_gpu):
            start = wid * per_worker
            end = start + per_worker if wid < total_workers - 1 else len(indexed)
            chunks.append((gpu_id, wid, indexed[start:end]))
            wid += 1

    print(f"Workers: {total_workers} ({workers_per_gpu} per GPU)")
    for gpu_id, wid, chunk in chunks:
        print(f"  Worker {wid}: GPU {gpu_id} — {len(chunk)} frames")

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    start_time = time.time()
    procs = []
    for gpu_id, wid, chunk in chunks:
        p = mp.Process(
            target=_label_worker,
            args=(gpu_id, wid, chunk, str(frames_dir), str(output_dir))
        )
        procs.append(p)
        p.start()

    for p in procs:
        p.join()

    elapsed = time.time() - start_time

    # Step 4: Write config
    with open(output_dir / "dataset.yaml", "w") as f:
        f.write(f"""path: {output_dir}
train: images/train
val: images/val

names:
  0: otter
""")

    train_imgs = len(list((output_dir / "images" / "train").glob("*")))
    val_imgs = len(list((output_dir / "images" / "val").glob("*")))

    print(f"\n{'='*60}")
    print(f"DATASET BUILD COMPLETE")
    print(f"{'='*60}")
    print(f"Sources:      {len(video_paths)} videos ({sum(1 for s in sources if s.get('live'))} live)")
    print(f"Train images: {train_imgs}")
    print(f"Val images:   {val_imgs}")
    print(f"Time:         {elapsed/3600:.1f} hours ({elapsed/60:.0f} min)")
    print(f"Dataset:      {output_dir}")
    print(f"\nNext: python run.py --train")


if __name__ == "__main__":
    build_dataset(
        sources=SOURCES,
        total_frames=TOTAL_FRAMES,
        fps=FPS,
        workers_per_gpu=WORKERS_PER_GPU,
    )