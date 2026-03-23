"""
OtterVision 🦦 — Main Entry Point

Two-phase system:
  Phase 1 (offline): Download videos → extract frames → auto-label with DINO+SAM → train models
  Phase 2 (live):    Run trained YOLOv8-seg + LSTM on live feed at real-time speeds
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))


def main():
    parser = argparse.ArgumentParser(
        description="OtterVision 🦦 — Real-time otter behavior analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
========================================
  PHASE 1: DATA PREP & TRAINING (offline)
========================================
  Step 1: Download and split training videos
    python run.py --download "https://youtube.com/watch?v=XXXXX"
    python run.py --download "https://youtube.com/watch?v=XXXXX" --segment-minutes 5

  Step 2: Extract frames from training videos
    python run.py --extract-frames
    python run.py --extract-frames --fps 2

  Step 3: Auto-label frames with GroundingDINO + SAM (slow, one-time)
    python run.py --prep-data

  Step 4: (Optional) Manually label otter crops with activity/object
    python run.py --label

  Step 5: Train YOLOv8-seg for fast otter detection
    python run.py --train-yolo --epochs 100

  Step 6: Train LSTM for activity classification
    python run.py --train-lstm --epochs 50

========================================
  PHASE 2: LIVE INFERENCE (real-time)
========================================
  python run.py --serve
  Then open http://localhost:8000
        """,
    )

    # Phase 1: Data prep
    parser.add_argument("--download", type=str, metavar="URL",
                        help="Download a YouTube video and split into segments")
    parser.add_argument("--segment-minutes", type=int, default=5,
                        help="Segment length when splitting (default: 5)")
    parser.add_argument("--extract-frames", action="store_true",
                        help="Extract frames from videos in data/videos/")
    parser.add_argument("--fps", type=float, default=2.0,
                        help="Frames per second for extraction (default: 2)")
    parser.add_argument("--prep-data", action="store_true",
                        help="Run GroundingDINO + SAM to auto-label frames (slow)")
    parser.add_argument("--label", action="store_true",
                        help="Open manual labeling tool for activity/object")

    # Phase 1: Training
    parser.add_argument("--train-yolo", action="store_true",
                        help="Train YOLOv8-seg on auto-labeled data")
    parser.add_argument("--train-lstm", action="store_true",
                        help="Train LSTM activity classifier on labeled crops")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Training epochs (default: 100 for YOLO, 50 for LSTM)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size (default: 16)")

    # Auto mode
    parser.add_argument("--auto", action="store_true",
                        help="Run entire pipeline: extract → prep → train YOLO → label → train LSTM → serve")
    parser.add_argument("--auto-no-label", action="store_true",
                        help="Same as --auto but skip manual labeling (YOLO only, no LSTM activity)")

    # Phase 2: Live
    parser.add_argument("--serve", action="store_true",
                        help="Launch web app with live inference")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")

    args = parser.parse_args()

    # Default to serve
    has_action = any([
        args.download, args.extract_frames, args.prep_data, args.label,
        args.train_yolo, args.train_lstm, args.serve, args.auto, args.auto_no_label
    ])
    if not has_action:
        args.serve = True

    # ---- AUTO MODE ----
    if args.auto or args.auto_no_label:
        from modules.video_pipeline import VideoPipeline
        from modules.data_prep import DataPrepPipeline
        from modules.yolo_trainer import YOLOTrainer
        from config import VIDEO_DIR, FRAME_DIR, YOLO_DATASET_DIR, DATA_DIR, LABEL_DIR

        # Check that videos exist
        video_files = [f for f in os.listdir(str(VIDEO_DIR))
                       if f.endswith((".mp4", ".avi", ".mov", ".mkv", ".webm"))]
        if not video_files:
            print("[OtterVision] No videos found in data/videos/")
            print("[OtterVision] First download a video:")
            print('  python run.py --download "https://youtube.com/watch?v=XXXXX"')
            sys.exit(1)

        print(f"[OtterVision] Found {len(video_files)} videos. Starting auto pipeline...\n")

        # Step 1: Extract frames
        print("=" * 60)
        print("  STEP 1/4: Extracting frames")
        print("=" * 60)
        VideoPipeline.extract_all_videos(str(VIDEO_DIR), str(FRAME_DIR), args.fps)

        # Step 2: Auto-label with GroundingDINO + SAM
        print("\n" + "=" * 60)
        print("  STEP 2/4: Auto-labeling with GroundingDINO + SAM (this takes a while)")
        print("=" * 60)
        pipeline = DataPrepPipeline()
        pipeline.load_models()
        pipeline.process_all_frames(str(FRAME_DIR), str(YOLO_DATASET_DIR))
        pipeline.unload_models()

        # Step 3: Train YOLOv8-seg
        print("\n" + "=" * 60)
        print("  STEP 3/4: Training YOLOv8-seg")
        print("=" * 60)
        yolo_trainer = YOLOTrainer()
        yolo_trainer.train(epochs=args.epochs, batch=args.batch_size)

        # Step 4: Manual labeling + LSTM (unless --auto-no-label)
        if args.auto and not args.auto_no_label:
            crops_dir = str(DATA_DIR / "crops")
            labels_file = str(LABEL_DIR / "activity_labels.json")

            if os.path.exists(crops_dir) and os.listdir(crops_dir):
                print("\n" + "=" * 60)
                print("  STEP 4/4: Manual labeling for activity classification")
                print("  Label the crops, then press 'q' to save and continue.")
                print("=" * 60)

                from utils.labeling_tool import run_labeling_tool
                run_labeling_tool(crops_dir, labels_file)

                # Train LSTM if labels were created
                if os.path.exists(labels_file):
                    print("\n[OtterVision] Training LSTM activity classifier...")
                    from modules.lstm_trainer import LSTMTrainer
                    trainer = LSTMTrainer()
                    trainer.build_models()
                    cache = trainer.extract_features(crops_dir, labels_file)
                    trainer.train(cache, num_epochs=min(args.epochs, 50))
                else:
                    print("[OtterVision] No labels created, skipping LSTM training.")
            else:
                print("[OtterVision] No crops found, skipping activity labeling.")
        else:
            print("\n[OtterVision] Skipping manual labeling (--auto-no-label mode).")
            print("[OtterVision] YOLO will detect otters but activity classification won't be trained.")
            print("[OtterVision] You can label later: python run.py --label && python run.py --train-lstm")

        # Done — launch server
        print("\n" + "=" * 60)
        print("  ALL DONE! Launching web server...")
        print("=" * 60)
        import uvicorn
        print(f"\n[OtterVision] Open http://localhost:{args.port}")
        uvicorn.run("app:app", host=args.host, port=args.port, reload=False)
        return

    # ---- DOWNLOAD ----
    if args.download:
        from modules.video_pipeline import VideoPipeline
        VideoPipeline.download_and_split(args.download, segment_minutes=args.segment_minutes)

    # ---- EXTRACT FRAMES ----
    if args.extract_frames:
        from modules.video_pipeline import VideoPipeline
        from config import VIDEO_DIR, FRAME_DIR
        VideoPipeline.extract_all_videos(str(VIDEO_DIR), str(FRAME_DIR), args.fps)

    # ---- PREP DATA (GroundingDINO + SAM) ----
    if args.prep_data:
        from modules.data_prep import DataPrepPipeline
        from config import FRAME_DIR, YOLO_DATASET_DIR

        pipeline = DataPrepPipeline()
        pipeline.load_models()
        pipeline.process_all_frames(str(FRAME_DIR), str(YOLO_DATASET_DIR))
        pipeline.unload_models()
        print("\n[OtterVision] Data prep complete!")
        print("[OtterVision] Next: python run.py --train-yolo")

    # ---- LABEL ----
    if args.label:
        from utils.labeling_tool import run_labeling_tool
        from config import DATA_DIR, LABEL_DIR

        crops_dir = str(DATA_DIR / "crops")
        labels_file = str(LABEL_DIR / "activity_labels.json")

        if not os.path.exists(crops_dir):
            print("[OtterVision] No crops found. Run --prep-data first.")
            sys.exit(1)

        run_labeling_tool(crops_dir, labels_file)

    # ---- TRAIN YOLO ----
    if args.train_yolo:
        from modules.yolo_trainer import YOLOTrainer

        trainer = YOLOTrainer()
        trainer.train(epochs=args.epochs, batch=args.batch_size)
        print("\n[OtterVision] YOLO training complete!")
        print("[OtterVision] Next: python run.py --label (then --train-lstm)")

    # ---- TRAIN LSTM ----
    if args.train_lstm:
        from modules.lstm_trainer import LSTMTrainer
        from config import DATA_DIR, LABEL_DIR

        crops_dir = str(DATA_DIR / "crops")
        labels_file = str(LABEL_DIR / "activity_labels.json")

        if not os.path.exists(labels_file):
            print("[OtterVision] No activity labels found. Run --label first.")
            sys.exit(1)

        trainer = LSTMTrainer()
        trainer.build_models()
        cache = trainer.extract_features(crops_dir, labels_file)
        lstm_epochs = min(args.epochs, 50)
        trainer.train(cache, num_epochs=lstm_epochs)
        print("\n[OtterVision] LSTM training complete!")
        print("[OtterVision] Ready for live inference: python run.py --serve")

    # ---- SERVE ----
    if args.serve:
        import uvicorn
        print(f"\n[OtterVision] Starting web server: http://localhost:{args.port}")
        uvicorn.run("app:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
