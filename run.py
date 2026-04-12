"""OtterVision — Ensemble YOLO + Qwen + Annotation Pipeline"""

import argparse
import shutil
import os
import glob


def find_best_weights(name):
    paths = [
        f"models/{name}/weights/best.pt",
        f"runs/segment/models/{name}/weights/best.pt",
        f"runs/segment/{name}/weights/best.pt",
    ]
    paths.extend(glob.glob(f"**/{name}/weights/best.pt", recursive=True))
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def main():
    parser = argparse.ArgumentParser(description="OtterVision")
    parser.add_argument("--serve", action="store_true", help="Launch web server")
    parser.add_argument("--train", action="store_true", help="Train both YOLO nano + small")
    parser.add_argument("--train-nano", action="store_true", help="Train only nano")
    parser.add_argument("--train-small", action="store_true", help="Train only small")
    parser.add_argument("--train-seed", action="store_true", help="Quick train on manually annotated seed data")
    parser.add_argument("--auto-label", action="store_true", help="Auto-label unlabeled frames with YOLO + SAM")
    parser.add_argument("--extract-frames", action="store_true", help="Extract frames for annotation")
    parser.add_argument("--merge-live", action="store_true", help="Record live feed and merge into dataset")
    parser.add_argument("--label-frames", type=str, default="", help="Label existing frames dir")
    parser.add_argument("--url", type=str, default="https://www.youtube.com/watch?v=_KXHUb0wFRE")
    parser.add_argument("--num-frames", type=int, default=100)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--record-minutes", type=int, default=60)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=4444)
    args = parser.parse_args()

    if args.extract_frames:
        from modules.annotator import extract_frames_for_annotation
        extract_frames_for_annotation(args.url, args.num_frames, fps=0.5, is_live=args.live)
        return

    if args.auto_label:
        from modules.annotator import auto_label_with_yolo
        auto_label_with_yolo()
        return

    if args.merge_live:
        from modules.dataset_merger import merge_live_feed
        merge_live_feed(args.record_minutes)
        return

    if args.label_frames:
        from modules.dataset_merger import label_existing_frames
        label_existing_frames(args.label_frames)
        return

    if args.train_seed:
        from ultralytics import YOLO

        data_yaml = os.path.join(os.getcwd(), "data/yolo_dataset/dataset.yaml")

        # Write dataset yaml if missing
        if not os.path.exists(data_yaml):
            os.makedirs("data/yolo_dataset", exist_ok=True)
            with open(data_yaml, "w") as f:
                f.write(f"""path: {os.path.join(os.getcwd(), 'data/yolo_dataset')}
train: images/train
val: images/val

names:
  0: otter
""")

        train_count = len(os.listdir("data/yolo_dataset/images/train")) if os.path.exists("data/yolo_dataset/images/train") else 0
        val_count = len(os.listdir("data/yolo_dataset/images/val")) if os.path.exists("data/yolo_dataset/images/val") else 0
        print(f"\n[OtterVision] Seed training: {train_count} train + {val_count} val images")

        if train_count < 5:
            print("[OtterVision] Not enough seed data. Annotate more frames at /annotate")
            return

        # Quick train — fewer epochs, small model, aggressive augmentation
        print("[OtterVision] Training seed model (YOLOv8s-seg)...")
        model = YOLO("yolov8s-seg.pt")
        model.train(
            data=data_yaml,
            epochs=min(100, max(50, train_count)),  # scale epochs to data size
            imgsz=960, batch=min(32, train_count),
            device=0, workers=8, cache=True, amp=True,
            mosaic=1.0, mixup=0.3, copy_paste=0.3, scale=0.5,
            hsv_h=0.02, hsv_s=0.7, hsv_v=0.4,
            flipud=0.5, fliplr=0.5, degrees=15, translate=0.2,
            project="models", name="yolo_seed", exist_ok=True, patience=30,
        )
        best = find_best_weights("yolo_seed")
        if best:
            shutil.copy(best, "models/otter_yolo_small.pt")
            shutil.copy(best, "models/otter_yolo_nano.pt")  # use same for both until full train
            print(f"[OtterVision] Seed model saved!")
            print(f"[OtterVision] Next: python run.py --auto-label")
        else:
            print("[OtterVision] Warning: best.pt not found")
        return

    if args.train or args.train_nano:
        from ultralytics import YOLO
        print("\n[OtterVision] Training YOLOv8n-seg (nano)...")
        model = YOLO("yolov8n-seg.pt")
        model.train(
            data=os.path.join(os.getcwd(), "data/yolo_dataset/dataset.yaml"),
            epochs=args.epochs, imgsz=960, batch=128,
            device=[0, 1], workers=12, cache=True, amp=True, cos_lr=True,
            mosaic=1.0, mixup=0.3, copy_paste=0.2, scale=0.5,
            hsv_h=0.02, hsv_s=0.7, hsv_v=0.4,
            flipud=0.5, fliplr=0.5, degrees=15, translate=0.2,
            project="models", name="yolo_nano_train", exist_ok=True, patience=50,
        )
        best = find_best_weights("yolo_nano_train")
        if best:
            shutil.copy(best, "models/otter_yolo_nano.pt")
            print(f"[OtterVision] Nano saved: models/otter_yolo_nano.pt")

    if args.train or args.train_small:
        from ultralytics import YOLO
        print("\n[OtterVision] Training YOLOv8s-seg (small)...")
        model = YOLO("yolov8s-seg.pt")
        model.train(
            data=os.path.join(os.getcwd(), "data/yolo_dataset/dataset.yaml"),
            epochs=args.epochs, imgsz=960, batch=64,
            device=[0, 1], workers=12, cache=True, amp=True, cos_lr=True,
            mosaic=1.0, mixup=0.3, copy_paste=0.2, scale=0.5,
            hsv_h=0.02, hsv_s=0.7, hsv_v=0.4,
            flipud=0.5, fliplr=0.5, degrees=15, translate=0.2,
            project="models", name="yolo_small_train", exist_ok=True, patience=50,
        )
        best = find_best_weights("yolo_small_train")
        if best:
            shutil.copy(best, "models/otter_yolo_small.pt")
            print(f"[OtterVision] Small saved: models/otter_yolo_small.pt")

    if args.train or args.train_nano or args.train_small:
        print("\n[OtterVision] Training complete! Run --serve to start.")
        return

    # Default: serve
    import uvicorn
    print(f"\n[OtterVision] Starting: http://localhost:{args.port}")
    uvicorn.run("app:app", host=args.host, port=args.port, reload=False, log_level="warning")


if __name__ == "__main__":
    main()