# OtterVision 🦦

**Real-time otter behavior analysis using deep learning.**

## How It Works — Two Phases

### Phase 1: Data Prep & Training (offline, slow, one-time)
```
Training Videos → Extract Frames → GroundingDINO + SAM (auto-labels otters)
                                          ↓
                              YOLO dataset + otter crops
                                    ↓              ↓
                          Train YOLOv8-seg    Label activity/object
                                    ↓              ↓
                          Fast otter detector   Train LSTM classifier
```

### Phase 2: Live Inference (real-time, fast)
```
Live YouTube Feed
       ↓
  YOLOv8-seg (30+ fps) → otter masks + crops
       ↓
  ResNet50 + LSTM → activity + held object
       ↓
  Annotated video stream in browser
       ↓
  (optional) User asks question → LLaVA VLM answers
```

## Quick Start

```bash
# Setup
conda create -n ottervision python=3.11 -y
conda activate ottervision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
sudo apt install ffmpeg  # for video splitting

# Phase 1: Prepare training data
python run.py --download "https://youtube.com/watch?v=XXXXX"  # downloads + splits into 5min segments
python run.py --extract-frames                                 # pulls frames at 2fps
python run.py --prep-data                                      # GroundingDINO + SAM auto-labels (slow)
python run.py --train-yolo --epochs 100                        # train fast detector
python run.py --label                                          # manually label activity/object
python run.py --train-lstm --epochs 50                         # train activity classifier

# Phase 2: Go live
python run.py --serve
# Open http://localhost:8000
```

## Project Structure

```
otter_vision/
├── run.py                    # CLI entry point
├── app.py                    # FastAPI web server
├── config.py                 # All settings
├── requirements.txt
├── modules/
│   ├── data_prep.py          # GroundingDINO + SAM (offline only)
│   ├── live_segmenter.py     # YOLOv8-seg (real-time)
│   ├── activity_classifier.py # ResNet50 + LSTM
│   ├── vlm_qa.py             # LLaVA Q&A (on-demand)
│   ├── video_pipeline.py     # Video handling + YouTube download
│   ├── yolo_trainer.py       # YOLOv8-seg training
│   └── lstm_trainer.py       # LSTM training
├── utils/
│   └── labeling_tool.py      # Manual annotation tool
├── templates/
│   └── index.html            # Web UI
├── data/
│   ├── videos/               # Training videos go here
│   ├── frames/               # Extracted frames
│   ├── crops/                # Cropped otter images
│   ├── labels/               # Activity/object labels
│   └── yolo_dataset/         # Auto-generated YOLO training data
└── models/                   # Trained model weights
```

## Architecture Details

| Component | Model | When Used | Speed |
|-----------|-------|-----------|-------|
| Otter detection (data prep) | GroundingDINO + SAM | Offline only | ~0.5 fps |
| Otter detection (live) | YOLOv8n-seg | Every frame | 30+ fps |
| Activity classification | ResNet50 + LSTM | Every frame | 50+ fps |
| Q&A | LLaVA 7B | On user request | ~3s per question |

## GPU Requirements
- Data prep (Phase 1): 8GB+ VRAM recommended
- Live inference (Phase 2): 4GB+ VRAM is enough (YOLOv8n is tiny)
- LLaVA Q&A: 8GB+ VRAM (only loaded when first question asked)
```
