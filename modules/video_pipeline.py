"""
Video Pipeline Module
Handles video input, frame extraction, YouTube downloading and splitting.
"""

import cv2
import numpy as np
import subprocess
import threading
import queue
import time
import os
import math
from pathlib import Path
from typing import Optional, Generator

from config import VIDEO_DIR, FRAME_DIR, MAX_FRAME_BUFFER, FRAME_EXTRACTION_FPS, YOUTUBE_LIVE_URL


class VideoPipeline:
    """Unified video input: local files, YouTube live, webcam."""

    def __init__(self):
        self.cap = None
        self.frame_queue = queue.Queue(maxsize=MAX_FRAME_BUFFER)
        self.running = False
        self.capture_thread = None
        self.source_type = None
        self.fps = 30
        self.frame_count = 0

    def open_local(self, video_path: str) -> bool:
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            print(f"[Video] Failed to open: {video_path}")
            return False
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.source_type = "local"
        print(f"[Video] Opened: {video_path} ({self.fps:.1f} fps)")
        return True

    def open_youtube(self, url: str = YOUTUBE_LIVE_URL) -> bool:
        print(f"[Video] Connecting to YouTube: {url}")
        try:
            result = subprocess.run(
                ["yt-dlp", "-f", "best[height<=720]", "-g", url],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode != 0:
                print(f"[Video] yt-dlp error: {result.stderr}")
                return False

            stream_url = result.stdout.strip()
            if not stream_url:
                return False

            self.cap = cv2.VideoCapture(stream_url)
            if not self.cap.isOpened():
                print("[Video] Failed to open stream.")
                return False

            self.fps = 30
            self.source_type = "youtube"
            print("[Video] YouTube connected.")
            return True
        except FileNotFoundError:
            print("[Video] yt-dlp not found. Install: pip install yt-dlp")
            return False
        except subprocess.TimeoutExpired:
            print("[Video] Timeout connecting to YouTube.")
            return False

    def _capture_loop(self):
        while self.running and self.cap is not None:
            ret, frame = self.cap.read()
            if not ret:
                if self.source_type == "local":
                    self.running = False
                    break
                time.sleep(1)
                continue
            self.frame_count += 1
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self.frame_queue.put(frame)

    def start(self):
        if self.cap is None:
            return
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

    def stop(self):
        self.running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=5)
        if self.cap:
            self.cap.release()
            self.cap = None

    def get_frame(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    # ================================================================
    # DATA UTILITIES
    # ================================================================

    @staticmethod
    def extract_frames_from_video(video_path: str, output_dir: str, fps: float = FRAME_EXTRACTION_FPS) -> int:
        os.makedirs(output_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_interval = max(1, int(video_fps / fps))
        prefix = Path(video_path).stem
        frame_count = 0
        saved = 0

        print(f"[Video] Extracting from {video_path} at {fps} fps...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % frame_interval == 0:
                cv2.imwrite(os.path.join(output_dir, f"{prefix}_{saved:06d}.jpg"), frame)
                saved += 1
                if saved % 200 == 0:
                    print(f"[Video]   {saved} frames extracted...")

        cap.release()
        print(f"[Video] Extracted {saved} frames from {Path(video_path).name}")
        return saved

    @staticmethod
    def extract_all_videos(video_dir: str = str(VIDEO_DIR), output_dir: str = str(FRAME_DIR), fps: float = FRAME_EXTRACTION_FPS) -> int:
        exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}
        total = 0
        for f in sorted(os.listdir(video_dir)):
            if Path(f).suffix.lower() in exts:
                total += VideoPipeline.extract_frames_from_video(
                    os.path.join(video_dir, f), output_dir, fps
                )
        print(f"[Video] Total: {total} frames extracted")
        return total

    @staticmethod
    def download_youtube(url: str, output_dir: str = str(VIDEO_DIR)) -> str:
        """Download a YouTube video."""
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "%(title)s.%(ext)s")
        print(f"[Video] Downloading: {url}")
        subprocess.run([
            "yt-dlp", "-f", "best[height<=720]",
            "-o", output_path,
            "--merge-output-format", "mp4",
            url
        ], check=True)
        print("[Video] Download complete.")
        # Return the most recently modified file
        files = sorted(Path(output_dir).glob("*.mp4"), key=os.path.getmtime, reverse=True)
        return str(files[0]) if files else ""

    @staticmethod
    def split_video(video_path: str, output_dir: str = str(VIDEO_DIR), segment_minutes: int = 5) -> int:
        """Split a video into segments using ffmpeg."""
        os.makedirs(output_dir, exist_ok=True)

        # Get duration
        result = subprocess.run([
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ], capture_output=True, text=True)
        duration = float(result.stdout.strip())

        num_segments = math.ceil(duration / (segment_minutes * 60))
        stem = Path(video_path).stem
        print(f"[Video] Splitting {duration/3600:.1f}h video into {num_segments} x {segment_minutes}min segments...")

        for i in range(num_segments):
            start = i * segment_minutes * 60
            out_path = os.path.join(output_dir, f"{stem}_seg{i:04d}.mp4")
            subprocess.run([
                "ffmpeg", "-y", "-ss", str(start),
                "-i", video_path,
                "-t", str(segment_minutes * 60),
                "-c", "copy", out_path
            ], capture_output=True, check=True)
            print(f"[Video]   Segment {i+1}/{num_segments}")

        # Remove original after splitting
        os.remove(video_path)
        print(f"[Video] Done. {num_segments} segments in {output_dir}")
        return num_segments

    @staticmethod
    def download_and_split(url: str, output_dir: str = str(VIDEO_DIR), segment_minutes: int = 5) -> int:
        """Download a YouTube video and split it into segments."""
        video_path = VideoPipeline.download_youtube(url, output_dir)
        if not video_path:
            print("[Video] Download failed.")
            return 0
        return VideoPipeline.split_video(video_path, output_dir, segment_minutes)
