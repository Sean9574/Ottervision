"""
OtterVision Web Server — YOLO + Qwen + Annotation
GPU 0: YOLOv8s-seg — segmentation
GPU 1: Qwen2.5-VL-7B — activity labels + Q&A
"""

import asyncio
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

from config import HOST, PORT, STATIC_DIR, TEMPLATE_DIR, YOUTUBE_LIVE_URL, VIDEO_DIR
from modules.yolo_segmenter import EnsembleSegmenter
from modules.vlm_engine import VLMEngine
from modules.label_reviewer import add_review_routes
from modules.annotator import add_annotator_routes

logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

app = FastAPI(title="OtterVision")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))
qa_executor = ThreadPoolExecutor(max_workers=1)

state = {
    "yolo": None, "vlm": None, "running": False,
    "current_frame": None, "latest_detections": [],
    "inference_fps": 0, "frame_count": 0,
    "ffmpeg_proc": None, "video_width": 1280, "video_height": 720,
    "stats": {
        "total_inferences": 0, "otters_detected": 0,
        "activity_history": deque(maxlen=500),
        "otter_count_history": deque(maxlen=200),
        "timeline": deque(maxlen=100),
    }
}


def _get_stream_url(url):
    result = subprocess.run(["yt-dlp", "-f", "best[height<=720]", "-g", url],
                            capture_output=True, text=True, timeout=30)
    return result.stdout.strip().split("\n")[0] if result.returncode == 0 else ""


def _start_ffmpeg(source):
    w, h = 1280, 720
    try:
        p = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                            "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", source],
                           capture_output=True, text=True, timeout=15)
        if p.returncode == 0 and "x" in p.stdout.strip():
            parts = p.stdout.strip().split("\n")[0].split("x")
            w, h = int(parts[0]), int(parts[1])
    except:
        pass
    state["video_width"], state["video_height"] = w, h
    print(f"[ffmpeg] {w}x{h}")
    cmd = ["ffmpeg"]
    if source.startswith("http"):
        cmd += ["-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "5"]
    else:
        cmd += ["-re"]
    cmd += ["-i", source, "-f", "rawvideo", "-pix_fmt", "bgr24", "-an", "-sn", "-v", "warning", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=w*h*3*10)
    state["ffmpeg_proc"] = proc
    def _log():
        for line in proc.stderr:
            l = line.decode("utf-8", errors="replace").strip()
            if l: print(f"[ffmpeg] {l}")
    threading.Thread(target=_log, daemon=True).start()


def _frame_reader():
    """Read frames as fast as possible, just overwrite current_frame with the latest."""
    print("[FrameReader] Waiting...")
    while True:
        if not state["running"]:
            time.sleep(0.1)
            continue
        proc = state["ffmpeg_proc"]
        if proc is None:
            time.sleep(0.1)
            continue
        w, h = state["video_width"], state["video_height"]
        frame_size = w * h * 3
        try:
            raw = proc.stdout.read(frame_size)
        except (ValueError, OSError):
            time.sleep(0.1)
            continue
        if len(raw) != frame_size:
            if proc.poll() is not None:
                print("[FrameReader] Stream ended, waiting...")
                state["running"] = False
                time.sleep(0.5)
                continue
            time.sleep(0.1)
            continue
        state["current_frame"] = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3))
        state["frame_count"] += 1
        if state["frame_count"] == 1:
            print(f"[FrameReader] First frame! {w}x{h}")
        elif state["frame_count"] % 500 == 0:
            print(f"[FrameReader] {state['frame_count']} frames")


def _inference_loop():
    """Grab the latest frame and run YOLO as fast as possible. No waiting, no frame counting."""
    print("[Inference] Waiting...")
    times = deque(maxlen=60)
    while True:
        if not state["running"] or state["current_frame"] is None:
            time.sleep(0.01)
            continue

        frame = state["current_frame"].copy()
        try:
            t0 = time.time()
            detections = state["yolo"].segment_frame(frame)
            t1 = time.time()

            vlm_labels = state["vlm"].get_activity_labels(frame, len(detections))

            for det in detections:
                if det.otter_id in vlm_labels:
                    label = vlm_labels[det.otter_id]
                    det.activity = label["activity"]
                    det.held_object = label["object"]
                elif vlm_labels:
                    sorted_keys = sorted(vlm_labels.keys())
                    if det.otter_id < len(sorted_keys):
                        label = vlm_labels[sorted_keys[det.otter_id]]
                        det.activity = label["activity"]
                        det.held_object = label["object"]

            det_json = state["yolo"].detections_to_json(detections)
            state["latest_detections"] = det_json

            s = state["stats"]
            s["total_inferences"] += 1
            s["otters_detected"] = len(detections)
            s["otter_count_history"].append(len(detections))
            for d in det_json:
                if d["activity"] != "active":
                    s["activity_history"].append(d["activity"])
            if det_json:
                s["timeline"].append({
                    "time": time.strftime("%H:%M:%S"),
                    "otters": len(det_json),
                    "activities": [d["activity"] for d in det_json],
                    "objects": [d["object"] for d in det_json if d["object"] != "none"],
                })

            times.append(time.time())
            if len(times) > 1:
                state["inference_fps"] = len(times) / max(times[-1] - times[0], 0.001)

            yolo_ms = (t1 - t0) * 1000

            if s["total_inferences"] == 1:
                print(f"[Inference] First result! {len(detections)} otters | YOLO: {yolo_ms:.0f}ms")
            elif s["total_inferences"] % 100 == 0:
                print(f"[Inference] {s['total_inferences']} | {state['inference_fps']:.1f} fps | {s['otters_detected']} otters | YOLO: {yolo_ms:.0f}ms")

        except Exception as e:
            print(f"[Inference] Error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(0.2)


def _stop():
    state["running"] = False
    proc = state["ffmpeg_proc"]
    state["ffmpeg_proc"] = None
    if proc:
        try: proc.kill()
        except: pass
    state["current_frame"] = None
    state["frame_count"] = 0
    state["latest_detections"] = []


def initialize():
    print("[App] Loading YOLO...")
    state["yolo"] = EnsembleSegmenter()
    state["yolo"].load_model()
    print("[App] Initializing VLM (loads lazily on GPU 1)...")
    state["vlm"] = VLMEngine()
    add_review_routes(app)
    add_annotator_routes(app)
    threading.Thread(target=_frame_reader, daemon=True).start()
    threading.Thread(target=_inference_loop, daemon=True).start()
    print("[App] Ready.")
    print("[App] Routes: / (main) | /annotate (label) | /review (check labels)")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/api/videos")
async def list_videos():
    exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    vd = Path(VIDEO_DIR)
    videos = sorted([f.name for f in vd.iterdir() if f.suffix.lower() in exts]) if vd.exists() else []
    return JSONResponse({"videos": videos})

@app.get("/api/video/{filename}")
async def serve_video(filename: str):
    path = Path(VIDEO_DIR) / filename
    if not path.exists(): return JSONResponse({"error": "Not found"}, status_code=404)
    return FileResponse(str(path), media_type="video/mp4", headers={"Accept-Ranges": "bytes"})

@app.post("/api/start")
async def start_stream(request: Request):
    body = await request.json()
    source = body.get("source", "youtube")
    url = body.get("url", YOUTUBE_LIVE_URL)
    filename = body.get("filename", "")
    _stop()
    await asyncio.sleep(0.3)
    if source == "youtube":
        stream_url = await asyncio.get_event_loop().run_in_executor(None, _get_stream_url, url)
        if not stream_url: return JSONResponse({"error": "Could not get stream URL"}, status_code=500)
        _start_ffmpeg(stream_url)
    elif source == "local":
        path = str(Path(VIDEO_DIR) / filename)
        if not os.path.exists(path): return JSONResponse({"error": f"Not found: {filename}"}, status_code=404)
        _start_ffmpeg(path)
    state["running"] = True
    return JSONResponse({"status": "started", "width": state["video_width"], "height": state["video_height"]})

@app.post("/api/stop")
async def stop_stream():
    _stop()
    return JSONResponse({"status": "stopped"})

@app.post("/api/ask")
async def ask_question(request: Request):
    body = await request.json()
    question = body.get("question", "")
    frame = state["current_frame"]
    if frame is None: return JSONResponse({"answer": "No video frame available."})
    if not question.strip(): return JSONResponse({"answer": "Please ask a question."})
    context = "; ".join([f"Otter {d['otter_id']+1}: {d['activity']}" for d in state["latest_detections"]])
    print(f"[QA] Q: {question}")
    answer = await asyncio.get_event_loop().run_in_executor(qa_executor, state["vlm"].ask_detailed, frame.copy(), question, context)
    print(f"[QA] A: {answer[:100]}...")
    return JSONResponse({"answer": answer})

@app.get("/api/stats")
async def get_stats():
    s = state["stats"]
    return JSONResponse({
        "inference_fps": round(state["inference_fps"], 1),
        "total_inferences": s["total_inferences"],
        "otters_detected": s["otters_detected"],
        "activity_distribution": dict(Counter(s["activity_history"])),
        "otter_count_history": list(s["otter_count_history"]),
        "recent_timeline": list(s["timeline"]),
    })

@app.websocket("/ws/overlay")
async def overlay_ws(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            if not state["running"]:
                await asyncio.sleep(0.5)
                continue
            await websocket.send_json({
                "type": "detections",
                "detections": state["latest_detections"],
                "inference_fps": round(state["inference_fps"], 1),
            })
            await asyncio.sleep(0.3)
    except WebSocketDisconnect: pass
    except Exception as e: print(f"[WS] Error: {e}")

@app.on_event("startup")
async def startup():
    initialize()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)