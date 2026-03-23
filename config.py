"""
OtterVision Configuration
Auto-detects hardware and scales settings accordingly.
"""

import os
from pathlib import Path

import torch

# ============================================================
# PATHS
# ============================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
VIDEO_DIR = DATA_DIR / "videos"
FRAME_DIR = DATA_DIR / "frames"
LABEL_DIR = DATA_DIR / "labels"
YOLO_DATASET_DIR = DATA_DIR / "yolo_dataset"
MODEL_DIR = BASE_DIR / "models"
STATIC_DIR = BASE_DIR / "static"
TEMPLATE_DIR = BASE_DIR / "templates"

for d in [VIDEO_DIR, FRAME_DIR, LABEL_DIR, YOLO_DATASET_DIR, MODEL_DIR, STATIC_DIR, TEMPLATE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# HARDWARE DETECTION — auto-scales everything
# ============================================================
NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0
DEVICE = "cuda" if NUM_GPUS > 0 else "cpu"

GPU_VRAM_GB = 0
GPU_NAMES = []
TOTAL_VRAM_GB = 0

if NUM_GPUS > 0:
    for i in range(NUM_GPUS):
        vram = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        name = torch.cuda.get_device_name(i)
        GPU_NAMES.append(name)
        TOTAL_VRAM_GB += vram
    GPU_VRAM_GB = torch.cuda.get_device_properties(0).total_memory / (1024**3)

CPU_CORES = os.cpu_count() or 4
RAM_GB = 0
try:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemTotal"):
                RAM_GB = int(line.split()[1]) / (1024**2)
                break
except:
    pass

# Determine hardware tier
if TOTAL_VRAM_GB >= 60:
    HW_TIER = "high"       # Dual A100s, multi-GPU workstation
elif TOTAL_VRAM_GB >= 16:
    HW_TIER = "medium"     # RTX 3080/3090/4070+, single A100
elif TOTAL_VRAM_GB >= 6:
    HW_TIER = "low"        # RTX 3060/3070, GTX 1660
else:
    HW_TIER = "cpu"        # No GPU or tiny VRAM

print(f"[OtterVision] Hardware detected:")
print(f"  Device:    {DEVICE}")
print(f"  GPUs:      {NUM_GPUS} ({', '.join(GPU_NAMES) if GPU_NAMES else 'none'})")
print(f"  VRAM:      {TOTAL_VRAM_GB:.1f} GB total")
print(f"  CPU cores: {CPU_CORES}")
print(f"  RAM:       {RAM_GB:.1f} GB")
print(f"  Tier:      {HW_TIER.upper()}")
print()

# ============================================================
# SCALED SETTINGS PER TIER
# ============================================================

# --- Data Prep (GroundingDINO + SAM) ---
DINO_MODEL_ID = "IDEA-Research/grounding-dino-base"
DINO_TEXT_PROMPT = "otter . sea otter . river otter"
DINO_THRESHOLD = 0.35

if HW_TIER == "high":
    SAM_MODEL_ID = "facebook/sam-vit-huge"      # Best masks, ~2.5GB VRAM
    PREP_BATCH_SIZE = 4                           # Process multiple frames at once
    DATALOADER_WORKERS = min(CPU_CORES, 16)
elif HW_TIER == "medium":
    SAM_MODEL_ID = "facebook/sam-vit-large"      # Good masks, ~1.2GB VRAM
    PREP_BATCH_SIZE = 2
    DATALOADER_WORKERS = min(CPU_CORES, 8)
elif HW_TIER == "low":
    SAM_MODEL_ID = "facebook/sam-vit-base"       # Decent masks, ~0.4GB VRAM
    PREP_BATCH_SIZE = 1
    DATALOADER_WORKERS = min(CPU_CORES, 4)
else:  # cpu
    SAM_MODEL_ID = "facebook/sam-vit-base"
    PREP_BATCH_SIZE = 1
    DATALOADER_WORKERS = min(CPU_CORES, 2)

# --- Data Prep precision ---
if HW_TIER in ("high", "medium"):
    PREP_DTYPE = torch.float16                    # Half precision = 2x speed, half VRAM
else:
    PREP_DTYPE = torch.float32                    # Full precision for stability on small GPUs

FRAME_EXTRACTION_FPS = 2

# --- YOLO Training ---
YOLO_MODEL_PATH = MODEL_DIR / "otter_yolo_seg.pt"
YOLO_CONF_THRESHOLD = 0.5
YOLO_IOU_THRESHOLD = 0.45

if HW_TIER == "high":
    YOLO_BASE_MODEL = "yolov8m-seg.pt"           # Medium model, best accuracy
    YOLO_BATCH_SIZE = 32
    YOLO_IMG_SIZE = 800
    YOLO_DEVICE = list(range(NUM_GPUS))           # Use ALL GPUs
elif HW_TIER == "medium":
    YOLO_BASE_MODEL = "yolov8s-seg.pt"           # Small model, good balance
    YOLO_BATCH_SIZE = 16
    YOLO_IMG_SIZE = 640
    YOLO_DEVICE = 0
elif HW_TIER == "low":
    YOLO_BASE_MODEL = "yolov8n-seg.pt"           # Nano model, fast
    YOLO_BATCH_SIZE = 8
    YOLO_IMG_SIZE = 640
    YOLO_DEVICE = 0
else:
    YOLO_BASE_MODEL = "yolov8n-seg.pt"
    YOLO_BATCH_SIZE = 4
    YOLO_IMG_SIZE = 480
    YOLO_DEVICE = "cpu"

# --- Activity Classifier (ResNet50 + LSTM) ---
ACTIVITY_CLASSES = [
    "swimming", "floating", "eating", "grooming",
    "playing", "socializing", "diving", "resting", "exploring",
]
OBJECT_CLASSES = [
    "rock", "shellfish", "fish", "kelp",
    "toy", "other_otter", "food_item", "none",
]
NUM_ACTIVITY_CLASSES = len(ACTIVITY_CLASSES)
NUM_OBJECT_CLASSES = len(OBJECT_CLASSES)

FEATURE_DIM = 2048
FREEZE_BACKBONE_LAYERS = 7
IMG_SIZE = (224, 224)

if HW_TIER == "high":
    LSTM_HIDDEN_SIZE = 1024
    LSTM_NUM_LAYERS = 3
    LSTM_DROPOUT = 0.3
    SEQUENCE_LENGTH = 24                          # Longer temporal window
    BATCH_SIZE = 32
    LEARNING_RATE = 3e-4
elif HW_TIER == "medium":
    LSTM_HIDDEN_SIZE = 512
    LSTM_NUM_LAYERS = 2
    LSTM_DROPOUT = 0.3
    SEQUENCE_LENGTH = 16
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
elif HW_TIER == "low":
    LSTM_HIDDEN_SIZE = 256
    LSTM_NUM_LAYERS = 2
    LSTM_DROPOUT = 0.4
    SEQUENCE_LENGTH = 12
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4
else:
    LSTM_HIDDEN_SIZE = 128
    LSTM_NUM_LAYERS = 1
    LSTM_DROPOUT = 0.5
    SEQUENCE_LENGTH = 8
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 8
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

ACTIVITY_MODEL_PATH = MODEL_DIR / "activity_classifier.pth"
BEST_MODEL_PATH = MODEL_DIR / "best_activity_classifier.pth"

# --- VLM Q&A (LLaVA) ---
if HW_TIER == "high":
    LLAVA_MODEL_ID = "llava-hf/llava-v1.6-34b-hf"     # 34B, best answers
    LLAVA_DTYPE = torch.float16
elif HW_TIER == "medium" and GPU_VRAM_GB >= 12:
    LLAVA_MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"  # 7B, good
    LLAVA_DTYPE = torch.float16
elif HW_TIER == "low":
    LLAVA_MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"
    LLAVA_DTYPE = torch.float16
else:
    LLAVA_MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"
    LLAVA_DTYPE = torch.float32

LLAVA_MAX_NEW_TOKENS = 256
LLAVA_TEMPERATURE = 0.2
LLAVA_SYSTEM_PROMPT = """You are an expert marine biologist observing otters through a live camera feed.
When asked questions about what you see, provide detailed and accurate descriptions of:
- What the otter(s) are doing (swimming, eating, floating, grooming, playing, etc.)
- What objects they are interacting with (rocks, shells, fish, kelp, toys, other otters)
- Their behavior patterns and body language
- How many otters are visible
Keep answers concise but informative. If you cannot clearly see something, say so."""

# --- Live Inference ---
if HW_TIER == "high":
    INFERENCE_FPS = 15
    DISPLAY_FPS = 30
elif HW_TIER == "medium":
    INFERENCE_FPS = 10
    DISPLAY_FPS = 20
elif HW_TIER == "low":
    INFERENCE_FPS = 6
    DISPLAY_FPS = 15
else:
    INFERENCE_FPS = 2
    DISPLAY_FPS = 10

YOUTUBE_LIVE_URL = "https://www.youtube.com/watch?v=9mg9PoFEX2U"
MAX_FRAME_BUFFER = 30

# --- Web Server ---
HOST = "0.0.0.0"
PORT = 8000
WEBSOCKET_FRAME_INTERVAL = 1.0 / DISPLAY_FPS

# --- Visualization ---
ACTIVITY_COLORS = {
    "swimming": (66, 133, 244),
    "floating": (52, 168, 83),
    "eating": (251, 188, 4),
    "grooming": (234, 67, 53),
    "playing": (171, 71, 188),
    "socializing": (0, 188, 212),
    "diving": (63, 81, 181),
    "resting": (121, 134, 203),
    "exploring": (255, 138, 101),
}
MASK_ALPHA = 0.4

# --- Print scaled config summary ---
print(f"[OtterVision] Scaled settings for {HW_TIER.upper()} tier:")
print(f"  SAM model:       {SAM_MODEL_ID.split('/')[-1]}")
print(f"  YOLO model:      {YOLO_BASE_MODEL} (batch {YOLO_BATCH_SIZE}, img {YOLO_IMG_SIZE})")
print(f"  YOLO device:     {YOLO_DEVICE}")
print(f"  LSTM:            hidden={LSTM_HIDDEN_SIZE}, layers={LSTM_NUM_LAYERS}, seq={SEQUENCE_LENGTH}")
print(f"  LLaVA:           {LLAVA_MODEL_ID.split('/')[-1]}")
print(f"  Inference FPS:   {INFERENCE_FPS}")
print(f"  Precision:       {'fp16' if PREP_DTYPE == torch.float16 else 'fp32'}")
print()
