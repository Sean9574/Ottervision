"""OtterVision — CLI for all operations"""

import argparse
import shutil
import os
import glob
import random
import subprocess
import time
from pathlib import Path


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


def build_dataset(url, record_minutes, num_annotate, fps):
    """
    Extract frames from ALL local videos + live feed.
    Puts most into the unlabeled pool for auto-labeling,
    grabs a mixed sample for hand annotation.
    """
    import cv2

    work_dir = Path("data/annotator_work")
    work_dir.mkdir(parents=True, exist_ok=True)
    unlabeled_dir = Path("data/unlabeled_frames")
    unlabeled_dir.mkdir(parents=True, exist_ok=True)
    seed_dir = Path("data/seed_frames")
    seed_dir.mkdir(parents=True, exist_ok=True)

    all_extracted = []  # (source_dir, filename) pairs

    # === Step 1: Extract from all local videos ===
    video_dir = Path("data/videos")
    if video_dir.exists():
        videos = sorted([f for f in video_dir.iterdir() if f.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}])
        if videos:
            temp_local = work_dir / "temp_local"
            temp_local.mkdir(exist_ok=True)
            print(f"\n[BuildDataset] Extracting frames from {len(videos)} local video(s)...")
            for vid in videos:
                name = vid.stem
                print(f"  {vid.name}...", end=" ", flush=True)
                subprocess.run([
                    "ffmpeg", "-i", str(vid), "-vf", f"fps={fps}", "-q:v", "2",
                    str(temp_local / f"{name}_%06d.jpg"), "-v", "warning", "-y"
                ])
                count = len([f for f in temp_local.iterdir() if f.name.startswith(name)])
                print(f"{count} frames")

            local_frames = sorted([f.name for f in temp_local.iterdir() if f.suffix == ".jpg"])
            for f in local_frames:
                all_extracted.append((str(temp_local), f))
            print(f"[BuildDataset] Total from local videos: {len(local_frames)}")
        else:
            print("[BuildDataset] No local videos found in data/videos/")
    else:
        print("[BuildDataset] data/videos/ not found, skipping local videos")

    # === Step 2: Record and extract from live feed ===
    if record_minutes > 0 and url:
        temp_live = work_dir / "temp_live"
        temp_live.mkdir(exist_ok=True)

        print(f"\n[BuildDataset] Recording {record_minutes} min from live feed...")
        try:
            result = subprocess.run(
                ["yt-dlp", "-f", "best[height<=720]", "-g", url],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                stream_url = result.stdout.strip().split("\n")[0]
                seconds = record_minutes * 60
                print(f"[BuildDataset] Extracting frames directly from stream for {record_minutes} min...")
                subprocess.run([
                    "ffmpeg", "-i", stream_url, "-t", str(seconds),
                    "-vf", f"fps={fps}", "-q:v", "2",
                    str(temp_live / "live_%06d.jpg"), "-v", "warning"
                ])
                live_frames = sorted([f.name for f in temp_live.iterdir() if f.suffix == ".jpg"])
                for f in live_frames:
                    all_extracted.append((str(temp_live), f))
                print(f"[BuildDataset] Total from live feed: {len(live_frames)}")
            else:
                print(f"[BuildDataset] Could not get stream URL: {result.stderr[:200]}")
        except Exception as e:
            print(f"[BuildDataset] Live recording failed: {e}")

    if not all_extracted:
        print("[BuildDataset] No frames extracted from any source!")
        return

    # === Step 3: Filter dark frames and move to unlabeled pool ===
    print(f"\n[BuildDataset] Processing {len(all_extracted)} frames...")
    moved = 0
    skipped_dark = 0
    skipped_dup = 0

    for src_dir, fname in all_extracted:
        img = cv2.imread(os.path.join(src_dir, fname))
        if img is None:
            continue
        if img.mean() < 30:
            skipped_dark += 1
            continue
        out_path = unlabeled_dir / f"unlabeled_{fname}"
        if out_path.exists():
            skipped_dup += 1
            continue
        shutil.copy(os.path.join(src_dir, fname), str(out_path))
        moved += 1

    print(f"[BuildDataset] Added {moved} to unlabeled pool ({skipped_dark} dark, {skipped_dup} duplicates)")

    # === Step 4: Sample frames for hand annotation ===
    random.shuffle(all_extracted)
    added_seed = 0
    for src_dir, fname in all_extracted:
        if added_seed >= num_annotate:
            break
        img = cv2.imread(os.path.join(src_dir, fname))
        if img is None or img.mean() < 30:
            continue
        out_path = seed_dir / f"seed_mix_{fname}"
        if out_path.exists():
            continue
        shutil.copy(os.path.join(src_dir, fname), str(out_path))
        added_seed += 1

    total_unlabeled = len([f for f in unlabeled_dir.iterdir() if f.suffix == ".jpg"])
    total_seed = len([f for f in seed_dir.iterdir() if f.suffix == ".jpg"])
    labeled = len([f for f in Path("data/seed_labels").iterdir() if f.suffix in {".txt", ".nootter"}]) if Path("data/seed_labels").exists() else 0

    print(f"\n{'='*50}")
    print(f"DATASET BUILD COMPLETE")
    print(f"{'='*50}")
    print(f"  Unlabeled pool:    {total_unlabeled} frames")
    print(f"  Seed frames:       {total_seed} ({added_seed} new)")
    print(f"  Already labeled:   {labeled}")
    print(f"  Needs annotation:  {total_seed - labeled}")
    print(f"\nNext steps:")
    print(f"  1. python run.py --serve     # then go to /annotate")
    print(f"  2. python run.py --rebuild-and-train")


def rebuild_dataset_from_seeds():
    """Clear old dataset, rebuild from all seed labels (including no-otter negatives)."""
    import cv2

    seed_frames = Path("data/seed_frames")
    seed_labels = Path("data/seed_labels")
    dataset = Path("data/yolo_dataset")

    # Clear
    for split in ["images/train", "images/val", "labels/train", "labels/val"]:
        d = dataset / split
        if d.exists():
            for f in d.iterdir():
                f.unlink()
        d.mkdir(parents=True, exist_ok=True)

    # Copy positive labels (otter annotations)
    pos_count = 0
    txt_files = sorted(seed_labels.glob("*.txt")) if seed_labels.exists() else []
    for f in txt_files:
        name = f.stem
        img = seed_frames / f"{name}.jpg"
        if not img.exists():
            continue
        if f.stat().st_size == 0:
            continue  # skip empty (handled below as nootter)

        split = "val" if random.random() < 0.2 else "train"
        shutil.copy(str(img), str(dataset / "images" / split / f"{name}.jpg"))
        shutil.copy(str(f), str(dataset / "labels" / split / f"{name}.txt"))
        pos_count += 1

    # Copy negative examples (no-otter frames)
    neg_count = 0
    nootter_files = sorted(seed_labels.glob("*.nootter")) if seed_labels.exists() else []
    for f in nootter_files:
        name = f.stem
        img = seed_frames / f"{name}.jpg"
        if not img.exists():
            continue

        split = "val" if random.random() < 0.2 else "train"
        shutil.copy(str(img), str(dataset / "images" / split / f"{name}.jpg"))
        # Empty label = no otters
        (dataset / "labels" / split / f"{name}.txt").touch()
        neg_count += 1

    # Write dataset.yaml
    with open(dataset / "dataset.yaml", "w") as f:
        f.write(f"""path: {dataset.resolve()}
train: images/train
val: images/val

names:
  0: otter
""")

    train_count = len(list((dataset / "images" / "train").glob("*")))
    val_count = len(list((dataset / "images" / "val").glob("*")))

    print(f"\n[Rebuild] Dataset rebuilt from seed labels")
    print(f"  Positive (otter):  {pos_count}")
    print(f"  Negative (empty):  {neg_count}")
    print(f"  Train: {train_count} | Val: {val_count}")

    return train_count


def main():
    parser = argparse.ArgumentParser(description="OtterVision")

    # Data commands
    parser.add_argument("--build-dataset", action="store_true",
                        help="Extract frames from ALL local videos + live feed into unlabeled pool + seed frames")
    parser.add_argument("--extract-frames", action="store_true",
                        help="Extract frames from a single source for annotation")
    parser.add_argument("--rebuild-and-train", action="store_true",
                        help="Full pipeline: rebuild dataset from seeds → train seed → auto-label → full train")

    # Training commands
    parser.add_argument("--train", action="store_true", help="Train both YOLO nano + small")
    parser.add_argument("--train-nano", action="store_true", help="Train only nano")
    parser.add_argument("--train-small", action="store_true", help="Train only small")
    parser.add_argument("--train-seed", action="store_true", help="Quick train on manually annotated seed data")
    parser.add_argument("--auto-label", action="store_true", help="Consensus auto-label (DINO + YOLO)")

    # Other commands
    parser.add_argument("--serve", action="store_true", help="Launch web server")
    parser.add_argument("--merge-live", action="store_true", help="Record live feed and merge into dataset")

    # Options
    parser.add_argument("--url", type=str, default="https://www.youtube.com/watch?v=_KXHUb0wFRE")
    parser.add_argument("--num-frames", type=int, default=100, help="Frames to sample for annotation")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--record-minutes", type=int, default=60, help="Minutes to record from live feed")
    parser.add_argument("--fps", type=float, default=0.5, help="Frame extraction rate")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=4444)
    args = parser.parse_args()

    # ================================================================
    # --build-dataset: Extract from local videos + live feed
    # ================================================================
    if args.build_dataset:
        build_dataset(
            url=args.url,
            record_minutes=args.record_minutes,
            num_annotate=args.num_frames,
            fps=args.fps,
        )
        return

    # ================================================================
    # --extract-frames: Single source extraction (original behavior)
    # ================================================================
    if args.extract_frames:
        from modules.annotator import extract_frames_for_annotation
        extract_frames_for_annotation(args.url, args.num_frames, fps=args.fps, is_live=args.live)
        return

    # ================================================================
    # --rebuild-and-train: Full pipeline in one command
    # ================================================================
    if args.rebuild_and_train:
        print("\n" + "="*60)
        print("OTTERVISION FULL TRAINING PIPELINE")
        print("="*60)

        # Step 1: Rebuild dataset from seed labels
        print("\n[Step 1/4] Rebuilding dataset from seed labels...")
        train_count = rebuild_dataset_from_seeds()

        if train_count < 10:
            print(f"\n[ERROR] Only {train_count} training images. Annotate more at /annotate first.")
            return

        # Step 2: Train seed model
        print("\n[Step 2/4] Training seed model...")
        from ultralytics import YOLO

        data_yaml = os.path.join(os.getcwd(), "data/yolo_dataset/dataset.yaml")
        model = YOLO("yolov8s-seg.pt")
        model.train(
            data=data_yaml,
            epochs=min(100, max(50, train_count)),
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
            shutil.copy(best, "models/otter_yolo_nano.pt")
            print(f"[Step 2/4] Seed model saved!")
        else:
            print("[Step 2/4] ERROR: Seed training failed — best.pt not found")
            return

        # Step 3: Consensus auto-label
        unlabeled_count = len([f for f in os.listdir("data/unlabeled_frames") if f.endswith(".jpg")]) if os.path.exists("data/unlabeled_frames") else 0
        if unlabeled_count > 0:
            print(f"\n[Step 3/4] Consensus auto-labeling {unlabeled_count} frames...")
            from modules.annotator import auto_label_with_yolo
            auto_label_with_yolo()
        else:
            print("\n[Step 3/4] No unlabeled frames — skipping auto-label")

        # Step 4: Full train on everything
        dataset_count = len(os.listdir("data/yolo_dataset/images/train")) if os.path.exists("data/yolo_dataset/images/train") else 0
        print(f"\n[Step 4/4] Full training on {dataset_count} images...")

        # Train small
        model = YOLO("yolov8s-seg.pt")
        model.train(
            data=data_yaml,
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
            print(f"[Step 4/4] Small model saved!")

        # Train nano
        model = YOLO("yolov8n-seg.pt")
        model.train(
            data=data_yaml,
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
            print(f"[Step 4/4] Nano model saved!")

        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        print("Run: python run.py --serve")
        return

    # ================================================================
    # --auto-label
    # ================================================================
    if args.auto_label:
        from modules.annotator import auto_label_with_yolo
        auto_label_with_yolo()
        return

    # ================================================================
    # --merge-live
    # ================================================================
    if args.merge_live:
        from modules.dataset_merger import merge_live_feed
        merge_live_feed(args.record_minutes)
        return

    # ================================================================
    # --train-seed
    # ================================================================
    if args.train_seed:
        rebuild_dataset_from_seeds()

        from ultralytics import YOLO
        data_yaml = os.path.join(os.getcwd(), "data/yolo_dataset/dataset.yaml")

        train_count = len(os.listdir("data/yolo_dataset/images/train")) if os.path.exists("data/yolo_dataset/images/train") else 0
        val_count = len(os.listdir("data/yolo_dataset/images/val")) if os.path.exists("data/yolo_dataset/images/val") else 0
        print(f"\n[OtterVision] Seed training: {train_count} train + {val_count} val images")

        if train_count < 5:
            print("[OtterVision] Not enough seed data. Annotate more frames at /annotate")
            return

        print("[OtterVision] Training seed model (YOLOv8s-seg)...")
        model = YOLO("yolov8s-seg.pt")
        model.train(
            data=data_yaml,
            epochs=min(100, max(50, train_count)),
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
            shutil.copy(best, "models/otter_yolo_nano.pt")
            print(f"[OtterVision] Seed model saved!")
            print(f"[OtterVision] Next: python run.py --auto-label")
        else:
            print("[OtterVision] Warning: best.pt not found")
        return

    # ================================================================
    # --train / --train-nano / --train-small
    # ================================================================
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

    # ================================================================
    # Default: --serve
    # ================================================================
    import uvicorn
    print(f"\n[OtterVision] Starting: http://localhost:{args.port}")
    uvicorn.run("app:app", host=args.host, port=args.port, reload=False, log_level="warning")


if __name__ == "__main__":
    main()