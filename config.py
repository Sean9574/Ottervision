"""
OtterVision Configuration — Ensemble YOLO + Qwen
"""

import torch
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
VIDEO_DIR = DATA_DIR / "videos"
FRAME_DIR = DATA_DIR / "frames"
YOLO_DATASET_DIR = DATA_DIR / "yolo_dataset"
MODEL_DIR = BASE_DIR / "models"
STATIC_DIR = BASE_DIR / "static"
TEMPLATE_DIR = BASE_DIR / "templates"

for d in [VIDEO_DIR, FRAME_DIR, MODEL_DIR, STATIC_DIR, TEMPLATE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0

if NUM_GPUS > 0:
    names = [torch.cuda.get_device_properties(i).name for i in range(NUM_GPUS)]
    vram = sum(torch.cuda.get_device_properties(i).total_memory / (1024**3) for i in range(NUM_GPUS))
    print(f"[OtterVision] GPUs: {NUM_GPUS} ({', '.join(names)}) | VRAM: {vram:.1f} GB")

# YOLO ensemble — both models on GPU 0
YOLO_NANO_PATH = MODEL_DIR / "otter_yolo_nano.pt"
YOLO_SMALL_PATH = MODEL_DIR / "otter_yolo_small.pt"
YOLO_DEVICE = 0
YOLO_CONF = 0.10
YOLO_IMG_SIZE = 640
YOLO_IOU_MERGE = 0.35  # IoU threshold for merging duplicate detections

# VLM
QA_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

# Video
YOUTUBE_LIVE_URL = "https://www.youtube.com/watch?v=_KXHUb0wFRE"
DISPLAY_FPS = 15

# Server
HOST = "0.0.0.0"
PORT = 4444
WEBSOCKET_FRAME_INTERVAL = 1.0 / DISPLAY_FPS