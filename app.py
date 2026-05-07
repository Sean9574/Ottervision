"""
OtterVision Web Server — Performance-Fixed Version

Key fixes:
- VLM labeling is now asynchronous and no longer blocks the frame loop.
- Local JPEG annotation/encoding is throttled to display/send rate instead of every processed frame.
- Repeated full-frame copies for mask drawing were reduced.
- Shared global state is protected with a lock for safer cross-thread access.
- Boolean settings parsing is fixed ("false" no longer becomes True).
"""

import asyncio
import base64
import time
import cv2
import numpy as np
import subprocess
import threading
import logging
import os
from pathlib import Path
from collections import deque, Counter
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from config import (
    HOST,
    PORT,
    STATIC_DIR,
    TEMPLATE_DIR,
    YOUTUBE_LIVE_URL,
    VIDEO_DIR,
    YOLO_CONF,
    YOLO_IMG_SIZE,
    DISPLAY_FPS,
    WEBSOCKET_FRAME_INTERVAL,
)
from modules.yolo_segmenter import EnsembleSegmenter
from modules.vlm_engine import VLMEngine
from modules.label_reviewer import add_review_routes
from modules.annotator import add_annotator_routes

logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

app = FastAPI(title="OtterVision")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

qa_executor = ThreadPoolExecutor(max_workers=1)
vlm_executor = ThreadPoolExecutor(max_workers=1)

state_lock = threading.Lock()
source_change_event = threading.Event()

VLM_SUBMIT_INTERVAL_SEC = 0.50
VLM_CACHE_MAX_AGE_SEC = 2.0
LOCAL_JPEG_QUALITY = 80

# ============================================================
# LIVE SETTINGS (adjustable from web UI)
# ============================================================

live_settings = {
    "conf": YOLO_CONF,
    "imgsz": YOLO_IMG_SIZE,
    "max_det": 10,
    "half": True,
    "show_masks": True,
    "show_labels": True,
    "vlm_enabled": True,
}

# ============================================================
# SHARED STATE
# ============================================================

state = {
    "yolo": None,
    "vlm": None,

    "running": False,
    "source_type": None,  # "youtube" or "local"
    "source_generation": 0,

    "current_frame": None,   # latest raw frame
    "frame_count": 0,
    "inference_fps": 0.0,

    # Local video mode
    "cv_cap": None,
    "local_fps": 30.0,
    "local_frame_interval": 1.0 / 30.0,
    "local_publish_interval": max(WEBSOCKET_FRAME_INTERVAL, 1.0 / max(DISPLAY_FPS, 1)),
    "last_local_publish": 0.0,
    "local_annotated_jpg": None,   # base64 JPEG string
    "local_det_count": 0,

    # YouTube mode
    "ffmpeg_proc": None,
    "latest_detections": [],

    # Clients
    "ws_clients": 0,

    # Video info
    "video_width": 1280,
    "video_height": 720,

    # Async VLM cache
    "vlm_future": None,
    "vlm_last_submit_ts": 0.0,
    "vlm_cache": {
        "labels": {},
        "ts": 0.0,
        "generation": -1,
        "count": 0,
    },

    "stats": {
        "total_inferences": 0,
        "otters_detected": 0,
        "activity_history": deque(maxlen=500),
        "otter_count_history": deque(maxlen=200),
        "timeline": deque(maxlen=100),
    },
}


# ============================================================
# GENERAL HELPERS
# ============================================================

def _parse_bool(value):
    """Strict bool parsing for settings updates."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "yes", "on"}:
            return True
        if v in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"Invalid boolean string: {value}")
    if isinstance(value, (int, float)):
        return bool(value)
    raise ValueError(f"Invalid boolean value: {value}")


def _safe_video_path(filename: str) -> Path:
    """Prevent path traversal when serving local video files."""
    base = Path(VIDEO_DIR).resolve()
    path = (base / filename).resolve()
    if base != path and base not in path.parents:
        raise ValueError("Invalid filename path")
    return path


def _snapshot_runtime():
    with state_lock:
        return {
            "running": state["running"],
            "source_type": state["source_type"],
            "frame_count": state["frame_count"],
            "inference_fps": state["inference_fps"],
            "local_annotated_jpg": state["local_annotated_jpg"],
            "local_det_count": state["local_det_count"],
            "latest_detections": state["latest_detections"],
            "show_masks": live_settings["show_masks"],
            "show_labels": live_settings["show_labels"],
        }


# ============================================================
# YOUTUBE: ffmpeg stream reader
# ============================================================

def _get_stream_url(url):
    result = subprocess.run(
        ["yt-dlp", "-f", "best[height<=720]", "-g", url],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        return ""
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return lines[0] if lines else ""


def _start_ffmpeg(stream_url):
    w, h = 1280, 720
    try:
        p = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "csv=p=0:s=x",
                stream_url,
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if p.returncode == 0 and "x" in p.stdout.strip():
            first_line = p.stdout.strip().splitlines()[0]
            parts = first_line.split("x")
            if len(parts) == 2:
                w, h = int(parts[0]), int(parts[1])
    except Exception:
        pass

    with state_lock:
        state["video_width"] = w
        state["video_height"] = h

    cmd = [
        "ffmpeg",
        "-fflags", "nobuffer",
        "-flags", "low_delay",
        "-reconnect", "1",
        "-reconnect_streamed", "1",
        "-reconnect_delay_max", "5",
        "-i", stream_url,
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-an",
        "-sn",
        "-v", "warning",
        "-",
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=w * h * 3 * 2,
    )

    with state_lock:
        state["ffmpeg_proc"] = proc

    print(f"[ffmpeg] Live stream {w}x{h}")

    def _log_stderr():
        if proc.stderr is None:
            return
        for line in proc.stderr:
            msg = line.decode("utf-8", errors="replace").strip()
            if msg:
                print(f"[ffmpeg] {msg}")

    threading.Thread(target=_log_stderr, daemon=True).start()


# ============================================================
# VLM ASYNC HELPERS
# ============================================================

def _vlm_inference_task(frame, detection_count, generation):
    labels = state["vlm"].get_activity_labels(frame, detection_count)
    return {
        "labels": labels or {},
        "ts": time.time(),
        "generation": generation,
        "count": detection_count,
    }


def _poll_vlm_result():
    with state_lock:
        future = state["vlm_future"]

    if future is None or not future.done():
        return

    try:
        result = future.result()
    except Exception as e:
        print(f"[VLM] Worker error: {e}")
        with state_lock:
            state["vlm_future"] = None
        return

    with state_lock:
        current_generation = state["source_generation"]
        if result["generation"] == current_generation:
            state["vlm_cache"] = result
        state["vlm_future"] = None


def _maybe_submit_vlm(frame, detection_count, generation):
    """Submit VLM work asynchronously at a limited rate."""
    if not live_settings["vlm_enabled"]:
        return

    if detection_count <= 0:
        return

    now = time.time()
    with state_lock:
        future = state["vlm_future"]
        last_submit = state["vlm_last_submit_ts"]

    if future is not None:
        return

    if now - last_submit < VLM_SUBMIT_INTERVAL_SEC:
        return

    # Copy only when we actually submit to the background worker.
    frame_copy = frame.copy()
    future = vlm_executor.submit(_vlm_inference_task, frame_copy, detection_count, generation)

    with state_lock:
        state["vlm_future"] = future
        state["vlm_last_submit_ts"] = now


def _apply_cached_vlm_labels(detections, generation):
    """Apply latest cached VLM labels without blocking the frame loop."""
    with state_lock:
        cache = dict(state["vlm_cache"])

    labels = cache.get("labels", {})
    cache_ts = cache.get("ts", 0.0)
    cache_generation = cache.get("generation", -1)

    if not labels:
        return
    if cache_generation != generation:
        return
    if time.time() - cache_ts > VLM_CACHE_MAX_AGE_SEC:
        return

    sorted_items = sorted(labels.items())

    for idx, det in enumerate(detections):
        if det.otter_id in labels:
            label = labels[det.otter_id]
            det.activity = label.get("activity", det.activity)
            det.held_object = label.get("object", det.held_object)
        elif idx < len(sorted_items):
            _, label = sorted_items[idx]
            det.activity = label.get("activity", det.activity)
            det.held_object = label.get("object", det.held_object)


# ============================================================
# SHARED HELPERS
# ============================================================

def _update_stats(detections, det_json):
    with state_lock:
        s = state["stats"]
        s["total_inferences"] += 1
        s["otters_detected"] = len(detections)
        s["otter_count_history"].append(len(detections))

        for d in det_json:
            if d.get("activity", "active") != "active":
                s["activity_history"].append(d["activity"])

        if det_json:
            s["timeline"].append({
                "time": time.strftime("%H:%M:%S"),
                "otters": len(det_json),
                "activities": [d.get("activity", "active") for d in det_json],
                "objects": [d.get("object", "none") for d in det_json if d.get("object", "none") != "none"],
            })


def _draw_detections(frame, detections):
    """Draw YOLO detections directly on the frame with fewer full-frame copies."""
    colors = {
        "floating": (52, 199, 89),
        "diving": (88, 86, 214),
        "eating": (255, 159, 10),
        "grooming": (255, 59, 48),
        "playing": (175, 82, 222),
        "socializing": (48, 176, 199),
        "resting": (142, 142, 147),
        "exploring": (255, 103, 35),
        "active": (0, 113, 227),
    }

    h, w = frame.shape[:2]
    overlay = None
    mask_union = None
    contour_ops = []

    if live_settings["show_masks"]:
        overlay = frame.copy()
        mask_union = np.zeros((h, w), dtype=bool)

    for det in detections:
        color = colors.get(det.activity, (0, 113, 227))
        x1, y1, x2, y2 = det.bbox

        used_mask = False
        if live_settings["show_masks"] and det.mask is not None:
            mask = det.mask > 0
            if mask.any():
                overlay[mask] = color
                mask_union |= mask
                mask_u8 = mask.astype(np.uint8)
                contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contour_ops.append((contours, color))
                used_mask = True

        if not used_mask:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    if overlay is not None and mask_union is not None and mask_union.any():
        blended = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
        frame[mask_union] = blended[mask_union]
        for contours, color in contour_ops:
            cv2.drawContours(frame, contours, -1, color, 2)

    if live_settings["show_labels"]:
        for det in detections:
            color = colors.get(det.activity, (0, 113, 227))
            x1, y1, _, _ = det.bbox

            label = f"Otter #{det.otter_id + 1}"
            if det.activity != "active":
                label += f" {det.activity}"
            if det.held_object != "none":
                label += f" [{det.held_object}]"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            lx, ly = max(0, x1), max(th + 8, y1 - 6)
            cv2.rectangle(frame, (lx, ly - th - 6), (lx + tw + 8, ly + 2), color, -1)
            cv2.putText(
                frame,
                label,
                (lx + 4, ly - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    cv2.putText(
        frame,
        f"YOLO: {state['inference_fps']:.0f} fps | {len(detections)} otters",
        (8, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )


def _set_inference_fps(times):
    if len(times) > 1:
        fps = len(times) / max(times[-1] - times[0], 0.001)
        with state_lock:
            state["inference_fps"] = fps


# ============================================================
# YOUTUBE LOOP
# ============================================================

def _youtube_loop():
    print("[YouTube] Inference loop waiting...")
    times = deque(maxlen=60)

    while True:
        with state_lock:
            running = state["running"]
            source_type = state["source_type"]
            proc = state["ffmpeg_proc"]
            w = state["video_width"]
            h = state["video_height"]
            generation = state["source_generation"]

        if not running or source_type != "youtube" or proc is None:
            source_change_event.wait(timeout=0.1)
            source_change_event.clear()
            continue

        frame_size = w * h * 3

        try:
            raw = proc.stdout.read(frame_size) if proc.stdout else b""
        except (ValueError, OSError):
            time.sleep(0.05)
            continue

        if len(raw) != frame_size:
            if proc.poll() is not None:
                print("[YouTube] Stream ended.")
                with state_lock:
                    state["running"] = False
            time.sleep(0.05)
            continue

        # No eager .copy() here. Keep the raw frame reference as-is.
        frame = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3))

        with state_lock:
            state["current_frame"] = frame
            state["frame_count"] += 1

        try:
            detections = state["yolo"].segment_frame(
                frame,
                conf=live_settings["conf"],
                imgsz=live_settings["imgsz"],
                half=live_settings["half"],
                max_det=live_settings["max_det"],
            )

            _poll_vlm_result()
            _maybe_submit_vlm(frame, len(detections), generation)
            _apply_cached_vlm_labels(detections, generation)

            det_json = state["yolo"].detections_to_json(
                detections,
                include_masks=live_settings["show_masks"],
            )

            with state_lock:
                state["latest_detections"] = det_json

            _update_stats(detections, det_json)

            times.append(time.time())
            _set_inference_fps(times)

            with state_lock:
                total = state["stats"]["total_inferences"]
                fps = state["inference_fps"]

            if total > 0 and total % 100 == 0:
                print(f"[YouTube] {total} | {fps:.0f} fps | {len(detections)} otters")

        except Exception as e:
            print(f"[YouTube] Error: {e}")
            time.sleep(0.05)


# ============================================================
# LOCAL VIDEO LOOP
# ============================================================

def _local_loop():
    print("[Local] Inference loop waiting...")
    times = deque(maxlen=60)

    while True:
        with state_lock:
            running = state["running"]
            source_type = state["source_type"]
            cap = state["cv_cap"]
            frame_interval = state["local_frame_interval"]
            publish_interval = state["local_publish_interval"]
            generation = state["source_generation"]
            ws_clients = state["ws_clients"]

        if not running or source_type != "local" or cap is None or not cap.isOpened():
            source_change_event.wait(timeout=0.1)
            source_change_event.clear()
            continue

        t_start = time.time()
        ret, frame = cap.read()

        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        with state_lock:
            state["current_frame"] = frame
            state["frame_count"] += 1

        try:
            detections = state["yolo"].segment_frame(
                frame,
                conf=live_settings["conf"],
                imgsz=live_settings["imgsz"],
                half=live_settings["half"],
                max_det=live_settings["max_det"],
            )

            _poll_vlm_result()
            _maybe_submit_vlm(frame, len(detections), generation)
            _apply_cached_vlm_labels(detections, generation)

            det_json = state["yolo"].detections_to_json(detections, include_masks=False)

            with state_lock:
                state["latest_detections"] = det_json

            _update_stats(detections, det_json)

            times.append(time.time())
            _set_inference_fps(times)

            with state_lock:
                total = state["stats"]["total_inferences"]
                fps = state["inference_fps"]
                last_publish = state["last_local_publish"]

            if total > 0 and total % 100 == 0:
                print(f"[Local] {total} | {fps:.0f} fps | {len(detections)} otters")

            # Only annotate/encode when needed for websocket display.
            now = time.time()
            should_publish = ws_clients > 0 and (now - last_publish) >= publish_interval

            if should_publish:
                annotated = frame.copy()
                _draw_detections(annotated, detections)

                ok, buf = cv2.imencode(
                    ".jpg",
                    annotated,
                    [cv2.IMWRITE_JPEG_QUALITY, LOCAL_JPEG_QUALITY],
                )
                if ok:
                    encoded = base64.b64encode(buf).decode("ascii")
                    with state_lock:
                        state["local_annotated_jpg"] = encoded
                        state["local_det_count"] = len(detections)
                        state["last_local_publish"] = now

        except Exception as e:
            print(f"[Local] Error: {e}")

        elapsed = time.time() - t_start
        wait = frame_interval - elapsed
        if wait > 0:
            time.sleep(wait)


# ============================================================
# STOP / INIT
# ============================================================

def _stop():
    with state_lock:
        proc = state["ffmpeg_proc"]
        cap = state["cv_cap"]

        state["running"] = False
        state["source_type"] = None
        state["source_generation"] += 1

        state["ffmpeg_proc"] = None
        state["cv_cap"] = None

        state["current_frame"] = None
        state["frame_count"] = 0
        state["latest_detections"] = []

        state["local_annotated_jpg"] = None
        state["local_det_count"] = 0
        state["last_local_publish"] = 0.0

        state["vlm_cache"] = {
            "labels": {},
            "ts": 0.0,
            "generation": -1,
            "count": 0,
        }

    if proc:
        try:
            proc.kill()
        except Exception:
            pass

    if cap:
        try:
            cap.release()
        except Exception:
            pass

    source_change_event.set()


def initialize():
    print("[App] Loading YOLO...")
    state["yolo"] = EnsembleSegmenter()
    state["yolo"].load_model()

    print("[App] Initializing VLM (loads lazily on GPU 1)...")
    state["vlm"] = VLMEngine()

    add_review_routes(app)
    add_annotator_routes(app)

    threading.Thread(target=_youtube_loop, daemon=True).start()
    threading.Thread(target=_local_loop, daemon=True).start()

    print("[App] Ready.")
    print("[App] Routes: / (main) | /annotate (label) | /review (check labels)")


# ============================================================
# ROUTES
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/api/videos")
async def list_videos():
    exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    vd = Path(VIDEO_DIR)
    videos = sorted(
        [f.name for f in vd.iterdir() if f.is_file() and f.suffix.lower() in exts]
    ) if vd.exists() else []
    return JSONResponse({"videos": videos})


@app.get("/api/video/{filename:path}")
async def serve_video(filename: str):
    try:
        path = _safe_video_path(filename)
    except ValueError:
        return JSONResponse({"error": "Invalid path"}, status_code=400)

    if not path.exists():
        return JSONResponse({"error": "Not found"}, status_code=404)

    return FileResponse(
        str(path),
        media_type="video/mp4",
        headers={"Accept-Ranges": "bytes"},
    )


@app.post("/api/start")
async def start_stream(request: Request):
    body = await request.json()
    source = body.get("source", "youtube")
    url = body.get("url", YOUTUBE_LIVE_URL)
    filename = body.get("filename", "")

    _stop()
    await asyncio.sleep(0.2)

    if source == "youtube":
        with state_lock:
            state["source_type"] = "youtube"
            generation = state["source_generation"]

        stream_url = await asyncio.get_event_loop().run_in_executor(None, _get_stream_url, url)
        if not stream_url:
            return JSONResponse({"error": "Could not get stream URL"}, status_code=500)

        _start_ffmpeg(stream_url)

        with state_lock:
            state["running"] = True
            width = state["video_width"]
            height = state["video_height"]

        source_change_event.set()

        return JSONResponse({
            "status": "started",
            "mode": "youtube",
            "width": width,
            "height": height,
            "generation": generation,
        })

    if source == "local":
        with state_lock:
            state["source_type"] = "local"

        try:
            path = _safe_video_path(filename)
        except ValueError:
            return JSONResponse({"error": "Invalid filename"}, status_code=400)

        if not path.exists():
            return JSONResponse({"error": f"Not found: {filename}"}, status_code=404)

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return JSONResponse({"error": "Failed to open video"}, status_code=500)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) else 0

        with state_lock:
            state["cv_cap"] = cap
            state["video_width"] = width
            state["video_height"] = height
            state["local_fps"] = fps
            state["local_frame_interval"] = 1.0 / max(fps, 1.0)
            state["local_publish_interval"] = max(
                WEBSOCKET_FRAME_INTERVAL,
                1.0 / max(DISPLAY_FPS, 1),
            )
            state["running"] = True

        print(f"[Local] Opened {filename}: {width}x{height} @ {fps:.0f}fps ({total} frames)")
        source_change_event.set()

        return JSONResponse({
            "status": "started",
            "mode": "local",
            "width": width,
            "height": height,
        })

    return JSONResponse({"error": "Unknown source"}, status_code=400)


@app.post("/api/stop")
async def stop_stream():
    _stop()
    return JSONResponse({"status": "stopped"})


@app.post("/api/ask")
async def ask_question(request: Request):
    body = await request.json()
    question = body.get("question", "")

    if not question.strip():
        return JSONResponse({"answer": "Please ask a question."})

    with state_lock:
        frame = state["current_frame"]
        latest_detections = state["latest_detections"]

    if frame is None:
        return JSONResponse({"answer": "No video frame available."})

    context = "; ".join(
        [f"Otter {d['otter_id'] + 1}: {d.get('activity', 'active')}" for d in latest_detections]
    )

    answer = await asyncio.get_event_loop().run_in_executor(
        qa_executor,
        state["vlm"].ask_detailed,
        frame.copy(),
        question,
        context,
    )

    return JSONResponse({"answer": answer})


@app.get("/api/stats")
async def get_stats():
    with state_lock:
        s = state["stats"]
        payload = {
            "inference_fps": round(state["inference_fps"], 1),
            "total_inferences": s["total_inferences"],
            "otters_detected": s["otters_detected"],
            "activity_distribution": dict(Counter(s["activity_history"])),
            "otter_count_history": list(s["otter_count_history"]),
            "recent_timeline": list(s["timeline"]),
        }
    return JSONResponse(payload)


@app.get("/api/settings")
async def get_settings():
    return JSONResponse(live_settings)


@app.post("/api/settings")
async def update_settings(request: Request):
    body = await request.json()

    valid_keys = {
        "conf": (float, 0.01, 0.99),
        "imgsz": (int, 320, 1280),
        "max_det": (int, 1, 50),
        "half": (bool, None, None),
        "show_masks": (bool, None, None),
        "show_labels": (bool, None, None),
        "vlm_enabled": (bool, None, None),
    }

    changed = {}

    for key, value in body.items():
        if key not in valid_keys:
            continue

        typ, mn, mx = valid_keys[key]

        try:
            if typ is bool:
                parsed = _parse_bool(value)
            else:
                parsed = typ(value)

            if mn is not None:
                parsed = max(mn, parsed)
            if mx is not None:
                parsed = min(mx, parsed)

            live_settings[key] = parsed
            changed[key] = parsed

        except Exception:
            continue

    if changed:
        print(f"[Settings] {changed}")

    return JSONResponse({"status": "ok", "settings": live_settings})


# ============================================================
# WEBSOCKET
# ============================================================

@app.websocket("/ws/overlay")
async def overlay_ws(websocket: WebSocket):
    await websocket.accept()

    with state_lock:
        state["ws_clients"] += 1

    try:
        while True:
            snap = _snapshot_runtime()

            if not snap["running"]:
                await asyncio.sleep(0.25)
                continue

            if snap["source_type"] == "local":
                jpg = snap["local_annotated_jpg"]
                if jpg:
                    await websocket.send_json({
                        "type": "frame",
                        "image": jpg,
                        "otters": snap["local_det_count"],
                        "inference_fps": round(snap["inference_fps"], 1),
                    })
                await asyncio.sleep(WEBSOCKET_FRAME_INTERVAL)

            elif snap["source_type"] == "youtube":
                await websocket.send_json({
                    "type": "detections",
                    "detections": snap["latest_detections"],
                    "inference_fps": round(snap["inference_fps"], 1),
                    "settings": {
                        "show_masks": snap["show_masks"],
                        "show_labels": snap["show_labels"],
                    },
                })
                await asyncio.sleep(WEBSOCKET_FRAME_INTERVAL)

            else:
                await asyncio.sleep(0.1)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WS] Error: {e}")
    finally:
        with state_lock:
            state["ws_clients"] = max(0, state["ws_clients"] - 1)


@app.on_event("startup")
async def startup():
    initialize()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)