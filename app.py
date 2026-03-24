"""
OtterVision Web Server
YOLO segments otters every frame (fast).
LLaVA auto-labels activity every few seconds + detailed Q&A on demand.
"""

import asyncio
import base64
import time
from collections import Counter, deque

import cv2
import numpy as np
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from config import (
    DISPLAY_FPS,
    HOST,
    INFERENCE_FPS,
    PORT,
    STATIC_DIR,
    TEMPLATE_DIR,
    WEBSOCKET_FRAME_INTERVAL,
    YOUTUBE_LIVE_URL,
)
from modules.live_segmenter import LiveSegmenter, OtterDetection
from modules.video_pipeline import VideoPipeline
from modules.vlm_qa import OtterVLM

app = FastAPI(title="OtterVision")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

state = {
    "video": None,
    "segmenter": None,
    "vlm": None,
    "current_frame": None,
    "running": False,
    "stats": {
        "fps": 0,
        "total_frames": 0,
        "otters_detected": 0,
        "activity_history": deque(maxlen=500),
        "object_history": deque(maxlen=500),
        "otter_count_history": deque(maxlen=200),
        "timeline": deque(maxlen=100),
    }
}


def initialize():
    print("[App] Loading models...")

    state["segmenter"] = LiveSegmenter()
    state["segmenter"].load_model()

    # VLM loads lazily on first frame or first question
    state["vlm"] = OtterVLM()

    print("[App] Ready.")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/api/start")
async def start_stream(request: Request):
    body = await request.json()
    source = body.get("source", "youtube")
    url = body.get("url", YOUTUBE_LIVE_URL)
    path = body.get("path", "")

    if state["video"] is not None:
        state["video"].stop()

    state["video"] = VideoPipeline()

    if source == "youtube":
        success = state["video"].open_youtube(url)
    elif source == "local":
        success = state["video"].open_local(path)
    else:
        return JSONResponse({"error": f"Unknown source: {source}"}, status_code=400)

    if not success:
        return JSONResponse({"error": "Failed to open video source"}, status_code=500)

    state["video"].start()
    state["running"] = True
    return JSONResponse({"status": "started", "source": source})


@app.post("/api/stop")
async def stop_stream():
    state["running"] = False
    if state["video"]:
        state["video"].stop()
    return JSONResponse({"status": "stopped"})


@app.post("/api/ask")
async def ask_question(request: Request):
    body = await request.json()
    question = body.get("question", "")
    frame = state["current_frame"]

    if frame is None:
        return JSONResponse({"answer": "No video frame available. Start a stream first."})
    if not question.strip():
        return JSONResponse({"answer": "Please ask a question."})

    vlm = state["vlm"]
    answer = vlm.ask_detailed(frame, question)
    return JSONResponse({"answer": answer})


@app.get("/api/stats")
async def get_stats():
    stats = state["stats"]
    return JSONResponse({
        "fps": round(stats["fps"], 1),
        "total_frames": stats["total_frames"],
        "otters_detected": stats["otters_detected"],
        "activity_distribution": dict(Counter(stats["activity_history"])),
        "object_distribution": dict(Counter(stats["object_history"])),
        "otter_count_history": list(stats["otter_count_history"]),
        "recent_timeline": list(stats["timeline"]),
    })


@app.websocket("/ws/video")
async def video_websocket(websocket: WebSocket):
    await websocket.accept()
    print("[WS] Client connected.")

    frame_count = 0
    last_inference = 0
    inference_interval = 1.0 / INFERENCE_FPS
    fps_times = deque(maxlen=30)

    try:
        while True:
            if not state["running"] or state["video"] is None:
                await asyncio.sleep(0.1)
                continue

            frame = state["video"].get_frame(timeout=0.5)
            if frame is None:
                await asyncio.sleep(0.05)
                continue

            frame_count += 1
            now = time.time()
            state["current_frame"] = frame.copy()
            annotated = frame.copy()

            if now - last_inference >= inference_interval:
                last_inference = now

                try:
                    # 1. YOLO segmentation (fast, every frame)
                    detections = state["segmenter"].segment_frame(frame)

                    # 2. VLM auto-labeling (runs every few seconds in background)
                    vlm = state["vlm"]
                    vlm_labels = vlm.get_activity_labels(frame)

                    # 3. Apply VLM labels to YOLO detections
                    for det in detections:
                        if det.otter_id in vlm_labels:
                            label = vlm_labels[det.otter_id]
                            det.activity = label["activity"]
                            det.activity_conf = 0.9
                            det.held_object = label["object"]
                            det.object_conf = 0.9
                        elif vlm_labels:
                            # If VLM found fewer/more otters than YOLO, try matching by order
                            sorted_keys = sorted(vlm_labels.keys())
                            if det.otter_id < len(sorted_keys):
                                key = sorted_keys[det.otter_id]
                                label = vlm_labels[key]
                                det.activity = label["activity"]
                                det.activity_conf = 0.8
                                det.held_object = label["object"]
                                det.object_conf = 0.8

                    # 4. Draw annotations
                    annotated = state["segmenter"].draw_detections(frame, detections)

                    # Update stats
                    s = state["stats"]
                    s["total_frames"] = frame_count
                    s["otters_detected"] = len(detections)
                    s["otter_count_history"].append(len(detections))

                    for d in detections:
                        if d.activity != "unknown":
                            s["activity_history"].append(d.activity)
                        if d.held_object != "none":
                            s["object_history"].append(d.held_object)

                    if detections:
                        s["timeline"].append({
                            "time": time.strftime("%H:%M:%S"),
                            "otters": len(detections),
                            "activities": [d.activity for d in detections],
                            "objects": [d.held_object for d in detections if d.held_object != "none"],
                        })

                except Exception as e:
                    cv2.putText(annotated, f"Error: {str(e)[:60]}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # FPS
            fps_times.append(now)
            if len(fps_times) > 1:
                state["stats"]["fps"] = len(fps_times) / max(fps_times[-1] - fps_times[0], 0.001)

            cv2.putText(annotated, f"FPS: {state['stats']['fps']:.1f}",
                        (10, annotated.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
            await websocket.send_json({
                "type": "frame",
                "data": base64.b64encode(buf).decode("utf-8"),
                "stats": {
                    "fps": round(state["stats"]["fps"], 1),
                    "otters": state["stats"]["otters_detected"],
                    "frame": frame_count,
                }
            })

            await asyncio.sleep(WEBSOCKET_FRAME_INTERVAL)

    except WebSocketDisconnect:
        print("[WS] Client disconnected.")
    except Exception as e:
        print(f"[WS] Error: {e}")


@app.on_event("startup")
async def startup():
    initialize()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)