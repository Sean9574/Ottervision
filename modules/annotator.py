"""
Manual Annotation Tool — Draw boxes in browser, SAM refines to masks.
Each --extract-frames run records new footage and adds new frames.
Already-annotated frames are preserved across runs.
No-otter frames are saved as negative examples for training.

Usage:
    python run.py --extract-frames --url "URL" --num-frames 100 --live
    # Open http://localhost:4444/annotate
    python run.py --train-seed
    python run.py --auto-label
"""

import os
import sys
import json
import time
import subprocess
import base64
import random
import shutil
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, YOLO_DATASET_DIR, NUM_GPUS, MODEL_DIR

SEED_FRAMES_DIR = DATA_DIR / "seed_frames"
SEED_LABELS_DIR = DATA_DIR / "seed_labels"
UNLABELED_FRAMES_DIR = DATA_DIR / "unlabeled_frames"

for d in [SEED_FRAMES_DIR, SEED_LABELS_DIR, UNLABELED_FRAMES_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def extract_frames_for_annotation(url: str, num_frames: int = 100, fps: float = 0.5, is_live: bool = False):
    """Extract frames from a video for manual annotation. Additive — each run adds new frames."""
    work_dir = DATA_DIR / "annotator_work"
    work_dir.mkdir(exist_ok=True)

    run_id = int(time.time())
    video_path = str(work_dir / f"recording_{run_id}.mp4")

    if is_live:
        print(f"[Annotator] Recording 1 hour of live feed...")
        result = subprocess.run(
            ["yt-dlp", "-f", "best[height<=720]", "-g", url],
            capture_output=True, text=True, timeout=4000
        )
        if result.returncode != 0:
            print("[Annotator] Error getting stream URL")
            return 0
        stream_url = result.stdout.strip().split("\n")[0]
        subprocess.run([
            "ffmpeg", "-i", stream_url, "-t", "3600",
            "-c", "copy", "-v", "warning", video_path
        ], timeout=15000)
    else:
        print(f"[Annotator] Downloading video...")
        subprocess.run([
            "yt-dlp", "-f", "best[height<=720]",
            "-o", video_path, "--merge-output-format", "mp4", url
        ], capture_output=True, text=True)

    if not os.path.exists(video_path):
        print("[Annotator] Video not available")
        return 0

    file_size = os.path.getsize(video_path) / (1024 * 1024)
    print(f"[Annotator] Recorded: {file_size:.0f} MB")

    temp_frames = str(work_dir / f"temp_{run_id}")
    os.makedirs(temp_frames, exist_ok=True)

    print(f"[Annotator] Extracting frames at {fps} fps...")
    subprocess.run([
        "ffmpeg", "-i", video_path,
        "-vf", f"fps={fps}", "-q:v", "2",
        os.path.join(temp_frames, f"r{run_id}_%06d.jpg"),
        "-v", "warning"
    ])

    extracted = [f for f in os.listdir(temp_frames) if f.endswith(".jpg")]
    print(f"[Annotator] Extracted {len(extracted)} frames from this recording")

    if not extracted:
        print("[Annotator] No frames extracted — video may be corrupt")
        return 0

    existing_seeds = set(os.listdir(str(SEED_FRAMES_DIR)))
    existing_unlabeled = set(os.listdir(str(UNLABELED_FRAMES_DIR)))

    random.shuffle(extracted)
    added_seed = 0
    added_unlabeled = 0

    for i, fname in enumerate(extracted):
        src = os.path.join(temp_frames, fname)
        seed_name = f"seed_{fname}"
        unlabeled_name = f"unlabeled_{fname}"

        if i < num_frames:
            if seed_name not in existing_seeds:
                shutil.copy(src, str(SEED_FRAMES_DIR / seed_name))
                added_seed += 1
        else:
            if unlabeled_name not in existing_unlabeled:
                shutil.copy(src, str(UNLABELED_FRAMES_DIR / unlabeled_name))
                added_unlabeled += 1

    total_seed = len([f for f in os.listdir(str(SEED_FRAMES_DIR)) if f.endswith(".jpg")])
    total_unlabeled = len([f for f in os.listdir(str(UNLABELED_FRAMES_DIR)) if f.endswith(".jpg")])
    annotated = len([f for f in os.listdir(str(SEED_LABELS_DIR)) if f.endswith(".txt")])
    no_otter = len([f for f in os.listdir(str(SEED_LABELS_DIR)) if f.endswith(".nootter")])
    remaining = total_seed - annotated - no_otter

    print(f"\n[Annotator] This run: +{added_seed} seed, +{added_unlabeled} unlabeled")
    print(f"[Annotator] Total seed frames: {total_seed} ({annotated} annotated, {no_otter} no-otter, {remaining} remaining)")
    print(f"[Annotator] Total unlabeled frames: {total_unlabeled}")
    print(f"[Annotator] Start server and go to /annotate")
    return added_seed


def mask_to_yolo_polygon(binary_mask: np.ndarray, img_w: int, img_h: int) -> str:
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return ""
    contour = max(contours, key=cv2.contourArea)
    epsilon = 0.01 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    if len(approx) < 3:
        return ""
    points = []
    for pt in approx:
        points.extend([f"{pt[0][0]/img_w:.6f}", f"{pt[0][1]/img_h:.6f}"])
    return "0 " + " ".join(points)


def compute_iou(box1, box2):
    """IoU between two [x1,y1,x2,y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / max(union, 1e-6)


def auto_label_with_yolo(yolo_conf: float = 0.3, dino_conf: float = 0.35, iou_agree: float = 0.3):
    """
    Consensus auto-labeler: DINO + YOLO seed must BOTH detect an otter
    at the same location. Only agreed detections get SAM masks.
    
    - DINO finds candidates (general object detector)
    - YOLO seed confirms candidates (trained on your perfect labels)
    - Only detections where both agree (IoU > threshold) are kept
    - SAM refines agreed detections to pixel-perfect masks
    - Frames with no agreed detections saved as negative examples
    """
    from ultralytics import YOLO
    from transformers import (
        AutoProcessor, AutoModelForZeroShotObjectDetection,
        SamModel, SamProcessor,
    )
    from PIL import Image

    DINO_MODEL_ID = "IDEA-Research/grounding-dino-base"
    DINO_TEXT_PROMPT = "otter . sea otter . river otter"

    # Load YOLO seed
    yolo_path = str(MODEL_DIR / "otter_yolo_small.pt")
    if not os.path.exists(yolo_path):
        yolo_path = str(MODEL_DIR / "otter_yolo_nano.pt")
    if not os.path.exists(yolo_path):
        print("[AutoLabel] No trained YOLO model found. Train seed model first.")
        return

    frames = sorted([f for f in os.listdir(str(UNLABELED_FRAMES_DIR)) if f.endswith(".jpg")])
    if not frames:
        print("[AutoLabel] No unlabeled frames found.")
        return

    # Load all models
    print(f"[AutoLabel] Loading YOLO seed from {yolo_path} on cuda:0...")
    yolo = YOLO(yolo_path)
    yolo.to("cuda:0")

    print(f"[AutoLabel] Loading GroundingDINO on cuda:0...")
    dino_proc = AutoProcessor.from_pretrained(DINO_MODEL_ID)
    dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_MODEL_ID).to("cuda:0")
    dino_model.eval()

    device_sam = "cuda:1" if NUM_GPUS >= 2 else "cuda:0"
    print(f"[AutoLabel] Loading SAM on {device_sam}...")
    sam_proc = SamProcessor.from_pretrained("facebook/sam-vit-huge")
    sam_model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device_sam)
    sam_model.eval()

    print(f"\n[AutoLabel] Consensus labeling: DINO (conf>{dino_conf}) + YOLO (conf>{yolo_conf}) must agree (IoU>{iou_agree})")
    print(f"[AutoLabel] Processing {len(frames)} frames...\n")

    output_dir = Path(str(YOLO_DATASET_DIR))
    for d in ["images/train", "images/val", "labels/train", "labels/val"]:
        (output_dir / d).mkdir(parents=True, exist_ok=True)

    labeled = 0
    negative = 0
    total_agreed = 0
    total_dino_only = 0
    total_yolo_only = 0
    start_time = time.time()

    for i, fname in enumerate(frames):
        frame_path = str(UNLABELED_FRAMES_DIR / fname)
        frame = cv2.imread(frame_path)
        if frame is None:
            continue

        # Skip dark frames
        if frame.mean() < 30:
            continue

        h, w = frame.shape[:2]
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        # === YOLO detections ===
        yolo_results = yolo(frame, imgsz=960, conf=yolo_conf, verbose=False, device="cuda:0")[0]
        yolo_boxes = []
        if yolo_results.boxes is not None:
            for box in yolo_results.boxes:
                yolo_boxes.append(box.xyxy[0].cpu().numpy().tolist())

        # === DINO detections ===
        dino_boxes = []
        with torch.no_grad():
            inputs = dino_proc(images=pil_image, text=DINO_TEXT_PROMPT, return_tensors="pt")
            inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
            outputs = dino_model(**inputs)
            results = dino_proc.post_process_grounded_object_detection(
                outputs, inputs["input_ids"],
                threshold=dino_conf,
                target_sizes=[pil_image.size[::-1]]
            )[0]
        for bbox, score in zip(results["boxes"], results["scores"]):
            dino_boxes.append(bbox.cpu().numpy().tolist())

        # === Find consensus: boxes where BOTH models agree ===
        agreed_boxes = []
        used_dino = set()
        used_yolo = set()

        for yi, ybox in enumerate(yolo_boxes):
            for di, dbox in enumerate(dino_boxes):
                if di in used_dino:
                    continue
                iou = compute_iou(ybox, dbox)
                if iou > iou_agree:
                    # Both agree — use the average of both boxes for best accuracy
                    avg_box = [
                        (ybox[0] + dbox[0]) / 2,
                        (ybox[1] + dbox[1]) / 2,
                        (ybox[2] + dbox[2]) / 2,
                        (ybox[3] + dbox[3]) / 2,
                    ]
                    agreed_boxes.append(avg_box)
                    used_dino.add(di)
                    used_yolo.add(yi)
                    break

        total_agreed += len(agreed_boxes)
        total_dino_only += len(dino_boxes) - len(used_dino)
        total_yolo_only += len(yolo_boxes) - len(used_yolo)

        is_val = (i % 5 == 0)
        split = "val" if is_val else "train"
        out_name = fname.replace("unlabeled_", "auto_")
        lbl_name = out_name.replace(".jpg", ".txt")

        if not agreed_boxes:
            # No consensus — save as negative example
            cv2.imwrite(str(output_dir / "images" / split / out_name), frame)
            with open(str(output_dir / "labels" / split / lbl_name), "w") as f:
                f.write("")
            negative += 1
        else:
            # Consensus detections — refine with SAM
            label_lines = []
            for bbox in agreed_boxes:
                with torch.no_grad():
                    sam_inputs = sam_proc(pil_image, input_boxes=[[[bbox]]], return_tensors="pt")
                    sam_inputs = {k: v.to(device_sam) for k, v in sam_inputs.items()}
                    sam_out = sam_model(**sam_inputs)
                    masks = sam_proc.image_processor.post_process_masks(
                        sam_out.pred_masks.cpu(),
                        sam_inputs["original_sizes"].cpu(),
                        sam_inputs["reshaped_input_sizes"].cpu()
                    )
                    best_idx = sam_out.iou_scores[0][0].argmax().item()
                    binary_mask = masks[0][0][best_idx].numpy().astype(np.uint8)

                polygon_line = mask_to_yolo_polygon(binary_mask, w, h)
                if polygon_line:
                    label_lines.append(polygon_line)

            if label_lines:
                cv2.imwrite(str(output_dir / "images" / split / out_name), frame)
                with open(str(output_dir / "labels" / split / lbl_name), "w") as f:
                    f.write("\n".join(label_lines))
                labeled += 1
            else:
                # SAM couldn't produce masks — save as negative
                cv2.imwrite(str(output_dir / "images" / split / out_name), frame)
                with open(str(output_dir / "labels" / split / lbl_name), "w") as f:
                    f.write("")
                negative += 1

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            speed = (i + 1) / elapsed
            remaining = (len(frames) - i - 1) / max(speed, 0.01) / 60
            print(f"[AutoLabel] {i+1}/{len(frames)} | "
                  f"agreed:{total_agreed} dino-only:{total_dino_only} yolo-only:{total_yolo_only} | "
                  f"{labeled} labeled {negative} negative | "
                  f"{speed:.1f} f/s | ~{remaining:.0f} min left")

    del yolo, dino_model, dino_proc, sam_model, sam_proc
    torch.cuda.empty_cache()

    with open(output_dir / "dataset.yaml", "w") as f:
        f.write(f"""path: {output_dir}
train: images/train
val: images/val

names:
  0: otter
""")

    train_count = len(list((output_dir / "images" / "train").glob("*")))
    val_count = len(list((output_dir / "images" / "val").glob("*")))

    print(f"\n{'='*60}")
    print(f"CONSENSUS AUTO-LABEL COMPLETE")
    print(f"{'='*60}")
    print(f"  Agreed detections (DINO + YOLO): {total_agreed}")
    print(f"  DINO-only (rejected):            {total_dino_only}")
    print(f"  YOLO-only (rejected):            {total_yolo_only}")
    print(f"  Positive labels saved:           {labeled}")
    print(f"  Negative examples saved:         {negative}")
    print(f"  Total dataset: {train_count} train + {val_count} val")
    print(f"\nNext: python run.py --train")


def add_annotator_routes(app):
    """Add annotation web UI routes to FastAPI."""
    from fastapi import Request
    from fastapi.responses import HTMLResponse, JSONResponse

    sam_model_cache = {"model": None, "processor": None}

    def get_sam(device="cuda:0"):
        if sam_model_cache["model"] is None:
            from transformers import SamModel, SamProcessor
            print("[SAM] Loading for annotation...")
            sam_model_cache["processor"] = SamProcessor.from_pretrained("facebook/sam-vit-huge")
            sam_model_cache["model"] = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
            sam_model_cache["model"].eval()
            print("[SAM] Ready.")
        return sam_model_cache["model"], sam_model_cache["processor"]

    def get_seed_frames():
        return sorted([f for f in os.listdir(str(SEED_FRAMES_DIR)) if f.endswith((".jpg", ".png"))])

    def get_annotation_status():
        frames = get_seed_frames()
        annotated = set()
        for f in os.listdir(str(SEED_LABELS_DIR)):
            if f.endswith(".txt"):
                annotated.add(f.replace(".txt", ".jpg"))
            elif f.endswith(".nootter"):
                annotated.add(f.replace(".nootter", ".jpg"))
        remaining = [f for f in frames if f not in annotated]
        return frames, list(annotated), remaining

    @app.get("/annotate", response_class=HTMLResponse)
    async def annotate_page():
        return ANNOTATE_HTML

    @app.get("/api/annotate/status")
    async def annotate_status():
        frames, done, remaining = get_annotation_status()
        return JSONResponse({"total": len(frames), "done": len(done), "remaining": len(remaining)})

    @app.get("/api/annotate/next")
    async def annotate_next():
        frames, done, remaining = get_annotation_status()
        if not remaining:
            return JSONResponse({"done": True, "total": len(frames), "annotated": len(done)})

        fname = remaining[0]
        img_path = str(SEED_FRAMES_DIR / fname)
        image = cv2.imread(img_path)
        if image is None:
            return JSONResponse({"error": "Could not read image"}, status_code=500)

        h, w = image.shape[:2]
        _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        b64 = base64.b64encode(buffer).decode('utf-8')

        return JSONResponse({
            "done": False, "image": b64, "filename": fname,
            "width": w, "height": h,
            "index": len(done), "total": len(frames), "remaining": len(remaining),
        })

    @app.post("/api/annotate/save")
    async def annotate_save(request: Request):
        body = await request.json()
        fname = body.get("filename", "")
        boxes = body.get("boxes", [])

        if not fname:
            return JSONResponse({"error": "No filename"}, status_code=400)

        img_path = str(SEED_FRAMES_DIR / fname)
        image = cv2.imread(img_path)
        if image is None:
            return JSONResponse({"error": "Image not found"}, status_code=404)

        h, w = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        output_dir = Path(str(YOLO_DATASET_DIR))
        for d in ["images/train", "images/val", "labels/train", "labels/val"]:
            (output_dir / d).mkdir(parents=True, exist_ok=True)

        is_val = random.random() < 0.2
        split = "val" if is_val else "train"
        lbl_name = fname.replace(".jpg", ".txt").replace(".png", ".txt")

        if not boxes:
            # No otters — save as negative example
            marker = str(SEED_LABELS_DIR / fname.replace(".jpg", ".nootter").replace(".png", ".nootter"))
            with open(marker, "w") as f:
                f.write("no otters")

            # Add to dataset with empty label (teaches model what's NOT an otter)
            cv2.imwrite(str(output_dir / "images" / split / fname), image)
            with open(str(output_dir / "labels" / split / lbl_name), "w") as f:
                f.write("")

            with open(output_dir / "dataset.yaml", "w") as f:
                f.write(f"""path: {output_dir}
train: images/train
val: images/val

names:
  0: otter
""")
            return JSONResponse({"status": "saved", "detections": 0, "masks": [], "message": "Saved as negative example"})

        # Has otters — run SAM on each box
        from PIL import Image as PILImage
        pil_image = PILImage.fromarray(image_rgb)
        sam_model, sam_proc = get_sam()
        device = next(sam_model.parameters()).device

        label_lines = []
        mask_b64_list = []

        for box in boxes:
            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
            bbox = [x1, y1, x2, y2]

            with torch.no_grad():
                sam_inputs = sam_proc(pil_image, input_boxes=[[[bbox]]], return_tensors="pt")
                sam_inputs = {k: v.to(device) for k, v in sam_inputs.items()}
                sam_out = sam_model(**sam_inputs)
                masks = sam_proc.image_processor.post_process_masks(
                    sam_out.pred_masks.cpu(),
                    sam_inputs["original_sizes"].cpu(),
                    sam_inputs["reshaped_input_sizes"].cpu()
                )
                best_idx = sam_out.iou_scores[0][0].argmax().item()
                binary_mask = masks[0][0][best_idx].numpy().astype(np.uint8)

            polygon_line = mask_to_yolo_polygon(binary_mask, w, h)
            if polygon_line:
                label_lines.append(polygon_line)

            mask_colored = np.zeros((h, w, 4), dtype=np.uint8)
            mask_colored[binary_mask > 0] = [0, 255, 0, 100]
            _, mbuf = cv2.imencode('.png', mask_colored)
            mask_b64_list.append(base64.b64encode(mbuf).decode('utf-8'))

        # Save to seed labels
        with open(str(SEED_LABELS_DIR / lbl_name), "w") as f:
            f.write("\n".join(label_lines))

        # Save to YOLO dataset
        cv2.imwrite(str(output_dir / "images" / split / fname), image)
        with open(str(output_dir / "labels" / split / lbl_name), "w") as f:
            f.write("\n".join(label_lines))

        with open(output_dir / "dataset.yaml", "w") as f:
            f.write(f"""path: {output_dir}
train: images/train
val: images/val

names:
  0: otter
""")

        return JSONResponse({"status": "saved", "detections": len(label_lines), "masks": mask_b64_list})


ANNOTATE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OtterVision — Annotate</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700&family=Source+Code+Pro:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg:#f5f5f7;--surface:#ffffff;--border:#e5e5e7;
            --text:#1d1d1f;--text-sec:#6e6e73;--text-ter:#aeaeb2;
            --accent:#0071e3;--green:#34c759;--red:#ff3b30;--amber:#ff9f0a;
            --radius:12px;--shadow:0 4px 12px rgba(0,0,0,0.06);
        }
        *{margin:0;padding:0;box-sizing:border-box}
        body{font-family:'Outfit',sans-serif;background:var(--bg);color:var(--text);min-height:100vh;display:flex;flex-direction:column;align-items:center;padding:24px}
        .header{text-align:center;margin-bottom:20px}
        .header h1{font-size:24px;font-weight:700;letter-spacing:-0.3px}
        .header p{color:var(--text-sec);font-size:14px;margin-top:4px}
        .progress-bar{width:100%;max-width:960px;height:6px;background:var(--border);border-radius:3px;margin-bottom:16px;overflow:hidden}
        .progress-fill{height:100%;background:var(--accent);border-radius:3px;transition:width 0.3s}
        .stats-row{display:flex;gap:12px;margin-bottom:16px;font-size:13px;font-weight:600}
        .stat{padding:5px 14px;border-radius:20px;background:var(--surface);border:0.5px solid var(--border)}
        .canvas-container{position:relative;background:var(--surface);border-radius:var(--radius);box-shadow:var(--shadow);border:0.5px solid var(--border);overflow:hidden;max-width:960px;width:100%}
        .canvas-info{padding:10px 16px;border-bottom:0.5px solid var(--border);display:flex;justify-content:space-between;font-size:13px}
        .canvas-info .fname{font-family:'Source Code Pro',monospace;color:var(--text-sec)}
        .canvas-wrap{position:relative;background:#000;cursor:crosshair}
        .canvas-wrap img{width:100%;display:block}
        .canvas-wrap canvas{position:absolute;top:0;left:0;width:100%;height:100%}
        .buttons{display:flex;gap:10px;padding:14px 16px;justify-content:center;border-top:0.5px solid var(--border);flex-wrap:wrap}
        .btn{padding:10px 28px;border-radius:10px;border:none;font-size:14px;font-weight:700;cursor:pointer;font-family:inherit;transition:all 0.15s;display:flex;align-items:center;gap:6px}
        .btn-save{background:var(--green);color:#fff;box-shadow:0 2px 8px rgba(52,199,89,0.3)}
        .btn-save:hover{transform:translateY(-1px)}
        .btn-nootter{background:var(--surface);color:var(--text-sec);border:0.5px solid var(--border)}
        .btn-nootter:hover{background:#fafafa}
        .btn-undo{background:var(--surface);color:var(--amber);border:0.5px solid var(--border)}
        .btn-undo:hover{background:#fffbf5}
        .btn-clear{background:var(--surface);color:var(--red);border:0.5px solid var(--border)}
        .btn-clear:hover{background:#fff5f5}
        .kbd{display:inline-block;padding:1px 6px;border-radius:4px;background:rgba(0,0,0,0.06);font-size:11px;font-family:'Source Code Pro',monospace}
        .instructions{max-width:960px;width:100%;margin-top:16px;padding:16px 20px;background:var(--surface);border-radius:var(--radius);border:0.5px solid var(--border);font-size:13px;color:var(--text-sec);line-height:1.6}
        .instructions strong{color:var(--text)}
        .done-screen{display:none;text-align:center;padding:60px}
        .done-screen h2{font-size:28px;margin-bottom:12px}
        .saving-overlay{display:none;position:absolute;top:0;left:0;right:0;bottom:0;background:rgba(255,255,255,0.8);z-index:50;align-items:center;justify-content:center;font-size:16px;font-weight:600;color:var(--accent)}
    </style>
</head>
<body>
    <div class="header">
        <h1>Annotate Otters</h1>
        <p>Draw boxes around every otter. SAM will refine to pixel-perfect masks.</p>
    </div>
    <div class="progress-bar"><div class="progress-fill" id="progressFill" style="width:0%"></div></div>
    <div class="stats-row">
        <div class="stat" id="statProgress">0 / 0</div>
        <div class="stat" id="statBoxes">0 boxes drawn</div>
        <div class="stat" id="statRemaining">0 remaining</div>
    </div>
    <div class="canvas-container" id="mainContainer">
        <div class="canvas-info">
            <span class="fname" id="fileName">Loading...</span>
            <span id="imgSize">0x0</span>
        </div>
        <div class="canvas-wrap" id="canvasWrap">
            <img id="sourceImage" src="" alt="">
            <canvas id="drawCanvas"></canvas>
            <div class="saving-overlay" id="savingOverlay">Refining with SAM...</div>
        </div>
        <div class="buttons">
            <button class="btn btn-undo" onclick="undoBox()">Undo <span class="kbd">Z</span></button>
            <button class="btn btn-clear" onclick="clearBoxes()">Clear <span class="kbd">C</span></button>
            <button class="btn btn-nootter" onclick="noOtters()">No Otters <span class="kbd">N</span></button>
            <button class="btn btn-save" onclick="saveAndNext()">Save & Next <span class="kbd">Enter</span></button>
        </div>
    </div>
    <div class="done-screen" id="doneScreen">
        <h2>Annotation Complete!</h2>
        <p id="doneSummary"></p>
        <p style="margin-top:16px;color:var(--text-sec)">Next steps:</p>
        <p style="margin-top:8px"><code style="background:var(--surface);padding:8px 16px;border-radius:8px;border:0.5px solid var(--border);font-family:'Source Code Pro',monospace">python run.py --train-seed</code></p>
        <p style="margin-top:8px"><code style="background:var(--surface);padding:8px 16px;border-radius:8px;border:0.5px solid var(--border);font-family:'Source Code Pro',monospace">python run.py --auto-label</code></p>
        <p style="margin-top:8px;font-size:14px;color:var(--text-sec)">Or extract more frames and keep annotating:</p>
        <p style="margin-top:8px"><code style="background:var(--surface);padding:8px 16px;border-radius:8px;border:0.5px solid var(--border);font-family:'Source Code Pro',monospace">python run.py --extract-frames --live --num-frames 100</code></p>
        <p style="margin-top:16px"><a href="/" style="color:var(--accent)">Back to OtterVision</a></p>
    </div>
    <div class="instructions">
        <strong>How to annotate:</strong> Click and drag to draw a bounding box around each otter.
        Draw a box for every otter visible. SAM will snap to the otter's outline.
        If there are no otters, click "No Otters" — this teaches the model what's NOT an otter.
        Press Enter when done with each frame. You can extract more frames and come back anytime.
    </div>
    <script>
        let boxes=[], drawing=false, startX=0, startY=0;
        let currentFilename='', imgW=0, imgH=0;
        let totalFrames=0, doneFrames=0, remainingFrames=0;
        const img=document.getElementById('sourceImage');
        const canvas=document.getElementById('drawCanvas');
        const ctx=canvas.getContext('2d');

        async function loadNext(){
            const r=await fetch('/api/annotate/next');
            const d=await r.json();
            if(d.done){
                document.getElementById('mainContainer').style.display='none';
                document.getElementById('doneScreen').style.display='block';
                document.getElementById('doneSummary').textContent='Annotated '+d.annotated+' of '+d.total+' frames.';
                return;
            }
            currentFilename=d.filename;imgW=d.width;imgH=d.height;
            totalFrames=d.total;doneFrames=d.index;remainingFrames=d.remaining;
            img.src='data:image/jpeg;base64,'+d.image;
            document.getElementById('fileName').textContent=d.filename;
            document.getElementById('imgSize').textContent=d.width+'x'+d.height;
            boxes=[];updateStats();
            img.onload=()=>{resizeCanvas();drawBoxes()};
        }
        function resizeCanvas(){canvas.width=img.clientWidth;canvas.height=img.clientHeight}
        function toImageCoords(cx,cy){const r=canvas.getBoundingClientRect();return[Math.round((cx-r.left)*(imgW/r.width)),Math.round((cy-r.top)*(imgH/r.height))]}
        function toCanvasCoords(ix,iy){const r=canvas.getBoundingClientRect();return[ix*(r.width/imgW),iy*(r.height/imgH)]}

        canvas.addEventListener('mousedown',(e)=>{drawing=true;[startX,startY]=toImageCoords(e.clientX,e.clientY)});
        canvas.addEventListener('mousemove',(e)=>{
            if(!drawing)return;
            const[mx,my]=toImageCoords(e.clientX,e.clientY);
            drawBoxes();
            const[sx,sy]=toCanvasCoords(startX,startY);
            const[ex,ey]=toCanvasCoords(mx,my);
            ctx.strokeStyle='#0071e3';ctx.lineWidth=2;ctx.setLineDash([6,3]);
            ctx.strokeRect(sx,sy,ex-sx,ey-sy);ctx.setLineDash([]);
        });
        canvas.addEventListener('mouseup',(e)=>{
            if(!drawing)return;drawing=false;
            const[endX,endY]=toImageCoords(e.clientX,e.clientY);
            const x1=Math.max(0,Math.min(startX,endX)),y1=Math.max(0,Math.min(startY,endY));
            const x2=Math.min(imgW,Math.max(startX,endX)),y2=Math.min(imgH,Math.max(startY,endY));
            if((x2-x1)>10&&(y2-y1)>10)boxes.push({x1,y1,x2,y2});
            drawBoxes();updateStats();
        });

        function drawBoxes(){
            ctx.clearRect(0,0,canvas.width,canvas.height);
            for(let i=0;i<boxes.length;i++){
                const b=boxes[i];
                const[x1,y1]=toCanvasCoords(b.x1,b.y1);
                const[x2,y2]=toCanvasCoords(b.x2,b.y2);
                ctx.fillStyle='rgba(0,113,227,0.15)';ctx.fillRect(x1,y1,x2-x1,y2-y1);
                ctx.strokeStyle='#0071e3';ctx.lineWidth=2;ctx.strokeRect(x1,y1,x2-x1,y2-y1);
                ctx.fillStyle='#0071e3';ctx.font='600 13px Outfit';
                ctx.fillText('Otter '+(i+1),x1+4,y1-6);
            }
        }
        function updateStats(){
            const pct=totalFrames>0?((doneFrames+1)/totalFrames*100):0;
            document.getElementById('progressFill').style.width=pct+'%';
            document.getElementById('statProgress').textContent=(doneFrames+1)+' / '+totalFrames;
            document.getElementById('statBoxes').textContent=boxes.length+' box'+(boxes.length!==1?'es':'')+' drawn';
            document.getElementById('statRemaining').textContent=remainingFrames+' remaining';
        }
        function undoBox(){boxes.pop();drawBoxes();updateStats()}
        function clearBoxes(){boxes=[];drawBoxes();updateStats()}

        async function saveAndNext(){
            document.getElementById('savingOverlay').style.display='flex';
            try{
                const r=await fetch('/api/annotate/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({filename:currentFilename,boxes:boxes})});
                const d=await r.json();
                if(d.masks&&d.masks.length>0){
                    for(const mb of d.masks){const mi=new Image();mi.src='data:image/png;base64,'+mb;mi.onload=()=>{ctx.drawImage(mi,0,0,canvas.width,canvas.height)}}
                    await new Promise(r=>setTimeout(r,400));
                }
            }catch(e){console.error(e)}
            document.getElementById('savingOverlay').style.display='none';
            await loadNext();
        }
        async function noOtters(){
            document.getElementById('savingOverlay').style.display='flex';
            await fetch('/api/annotate/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({filename:currentFilename,boxes:[]})});
            document.getElementById('savingOverlay').style.display='none';
            await loadNext();
        }
        document.addEventListener('keydown',(e)=>{
            if(e.target.tagName==='INPUT'||e.target.tagName==='TEXTAREA')return;
            if(e.key==='Enter')saveAndNext();
            else if(e.key==='z'||e.key==='Z')undoBox();
            else if(e.key==='c'||e.key==='C')clearBoxes();
            else if(e.key==='n'||e.key==='N')noOtters();
        });
        window.addEventListener('resize',()=>{resizeCanvas();drawBoxes()});
        loadNext();
    </script>
</body>
</html>"""


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--url", type=str, default="https://www.youtube.com/watch?v=_KXHUb0wFRE")
    parser.add_argument("--num-frames", type=int, default=100)
    parser.add_argument("--fps", type=float, default=0.5)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--auto-label", action="store_true")
    args = parser.parse_args()

    if args.extract:
        extract_frames_for_annotation(args.url, args.num_frames, args.fps, args.live)
    elif args.auto_label:
        auto_label_with_yolo()