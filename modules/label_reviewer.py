"""
Web Label Reviewer — Review YOLO labels in your browser.
Adds routes to the existing FastAPI app.

Import and call `add_review_routes(app)` from app.py,
then open http://localhost:4444/review
"""

import os
import cv2
import numpy as np
import base64
from pathlib import Path
from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse

from config import YOLO_DATASET_DIR


def add_review_routes(app):
    """Add label review endpoints to existing FastAPI app."""

    def get_labeled_images(split):
        img_dir = os.path.join(str(YOLO_DATASET_DIR), "images", split)
        lbl_dir = os.path.join(str(YOLO_DATASET_DIR), "labels", split)
        if not os.path.exists(img_dir):
            return []
        images = sorted([
            f for f in os.listdir(img_dir)
            if f.endswith(('.jpg', '.png'))
        ])
        labeled = []
        for img_name in images:
            lbl_name = img_name.replace('.jpg', '.txt').replace('.png', '.txt')
            lbl_path = os.path.join(lbl_dir, lbl_name)
            if os.path.exists(lbl_path) and os.path.getsize(lbl_path) > 0:
                labeled.append(img_name)
        return labeled

    def render_image_with_labels(img_name, split):
        img_dir = os.path.join(str(YOLO_DATASET_DIR), "images", split)
        lbl_dir = os.path.join(str(YOLO_DATASET_DIR), "labels", split)
        img_path = os.path.join(img_dir, img_name)
        lbl_name = img_name.replace('.jpg', '.txt').replace('.png', '.txt')
        lbl_path = os.path.join(lbl_dir, lbl_name)

        image = cv2.imread(img_path)
        if image is None:
            return None

        h, w = image.shape[:2]
        overlay = image.copy()

        det_count = 0
        if os.path.exists(lbl_path) and os.path.getsize(lbl_path) > 0:
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 7:
                        continue
                    coords = [float(x) for x in parts[1:]]
                    points = []
                    for i in range(0, len(coords), 2):
                        px = int(coords[i] * w)
                        py = int(coords[i + 1] * h)
                        points.append([px, py])
                    if len(points) < 3:
                        continue
                    pts = np.array(points, dtype=np.int32)
                    cv2.fillPoly(overlay, [pts], (0, 255, 0))
                    cv2.polylines(image, [pts], True, (0, 255, 0), 2)
                    x_min, y_min = pts.min(axis=0)
                    x_max, y_max = pts.max(axis=0)
                    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 255), 1)
                    det_count += 1

        image = cv2.addWeighted(overlay, 0.25, image, 0.75, 0)
        _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64 = base64.b64encode(buffer).decode('utf-8')
        return b64, det_count

    @app.get("/review", response_class=HTMLResponse)
    async def review_page():
        return REVIEW_HTML

    @app.get("/api/review/info")
    async def review_info(split: str = "train"):
        labeled = get_labeled_images(split)
        return JSONResponse({"total": len(labeled), "split": split})

    @app.get("/api/review/image")
    async def review_image(index: int = 0, split: str = "train"):
        labeled = get_labeled_images(split)
        if index >= len(labeled) or index < 0:
            return JSONResponse({"done": True, "total": len(labeled)})

        img_name = labeled[index]
        result = render_image_with_labels(img_name, split)
        if result is None:
            return JSONResponse({"error": "Could not read image"}, status_code=500)

        b64, det_count = result
        return JSONResponse({
            "image": b64,
            "filename": img_name,
            "index": index,
            "total": len(labeled),
            "detections": det_count,
            "done": False,
        })

    @app.post("/api/review/delete")
    async def review_delete(request: Request):
        body = await request.json()
        img_name = body.get("filename", "")
        split = body.get("split", "train")

        img_dir = os.path.join(str(YOLO_DATASET_DIR), "images", split)
        lbl_dir = os.path.join(str(YOLO_DATASET_DIR), "labels", split)
        lbl_name = img_name.replace('.jpg', '.txt').replace('.png', '.txt')

        img_path = os.path.join(img_dir, img_name)
        lbl_path = os.path.join(lbl_dir, lbl_name)

        deleted = False
        if os.path.exists(img_path):
            os.remove(img_path)
            deleted = True
        if os.path.exists(lbl_path):
            os.remove(lbl_path)
            deleted = True

        return JSONResponse({"deleted": deleted, "filename": img_name})


REVIEW_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OtterVision — Label Review</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700&family=Source+Code+Pro:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #f5f5f7; --surface: #ffffff; --border: #e5e5e7;
            --text: #1d1d1f; --text-secondary: #6e6e73; --text-tertiary: #aeaeb2;
            --accent: #0071e3; --green: #34c759; --red: #ff3b30;
            --radius: 12px;
            --shadow: 0 4px 12px rgba(0,0,0,0.06), 0 1px 4px rgba(0,0,0,0.04);
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Outfit', sans-serif; background: var(--bg);
            color: var(--text); min-height: 100vh;
            display: flex; flex-direction: column; align-items: center;
            padding: 24px;
        }
        .header {
            text-align: center; margin-bottom: 24px;
        }
        .header h1 { font-size: 24px; font-weight: 700; letter-spacing: -0.3px; }
        .header p { color: var(--text-secondary); font-size: 14px; margin-top: 4px; }
        .progress-bar {
            width: 100%; max-width: 900px; height: 6px;
            background: var(--border); border-radius: 3px;
            margin-bottom: 20px; overflow: hidden;
        }
        .progress-fill {
            height: 100%; background: var(--accent); border-radius: 3px;
            transition: width 0.3s ease;
        }
        .stats-row {
            display: flex; gap: 16px; margin-bottom: 20px;
            font-size: 13px; font-weight: 600;
        }
        .stat { padding: 6px 14px; border-radius: 20px; background: var(--surface); border: 0.5px solid var(--border); }
        .stat.kept { color: var(--green); }
        .stat.deleted { color: var(--red); }
        .stat.progress { color: var(--accent); }
        .image-container {
            background: var(--surface); border-radius: var(--radius);
            box-shadow: var(--shadow); border: 0.5px solid var(--border);
            overflow: hidden; max-width: 900px; width: 100%;
        }
        .image-info {
            padding: 12px 18px; border-bottom: 0.5px solid var(--border);
            display: flex; justify-content: space-between; align-items: center;
            font-size: 13px;
        }
        .image-info .filename {
            font-family: 'Source Code Pro', monospace; color: var(--text-secondary);
        }
        .image-info .det-count { font-weight: 600; }
        .image-wrap {
            width: 100%; display: flex; align-items: center; justify-content: center;
            background: #000; min-height: 400px;
        }
        .image-wrap img {
            width: 100%; height: auto; display: block;
        }
        .buttons {
            display: flex; gap: 12px; padding: 18px;
            justify-content: center; border-top: 0.5px solid var(--border);
        }
        .btn {
            padding: 12px 40px; border-radius: 10px; border: none;
            font-size: 15px; font-weight: 700; cursor: pointer;
            font-family: inherit; transition: all 0.15s;
            display: flex; align-items: center; gap: 8px;
        }
        .btn-keep {
            background: var(--green); color: white;
            box-shadow: 0 2px 8px rgba(52,199,89,0.3);
        }
        .btn-keep:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(52,199,89,0.4); }
        .btn-delete {
            background: var(--red); color: white;
            box-shadow: 0 2px 8px rgba(255,59,48,0.3);
        }
        .btn-delete:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(255,59,48,0.4); }
        .btn-skip {
            background: var(--surface); color: var(--text-secondary);
            border: 0.5px solid var(--border);
        }
        .btn-skip:hover { background: #fafafa; }
        .kbd { 
            display: inline-block; padding: 2px 7px; border-radius: 4px;
            background: rgba(255,255,255,0.2); font-size: 12px;
            font-family: 'Source Code Pro', monospace;
        }
        .btn-skip .kbd { background: var(--bg); }
        .done-screen {
            text-align: center; padding: 60px 20px;
            display: none;
        }
        .done-screen h2 { font-size: 28px; margin-bottom: 12px; }
        .done-screen p { color: var(--text-secondary); font-size: 15px; }
        .split-toggle {
            display: flex; gap: 0; border-radius: 8px; overflow: hidden;
            border: 0.5px solid var(--border); background: var(--surface);
            margin-bottom: 20px;
        }
        .split-toggle button {
            padding: 8px 20px; border: none; font-size: 13px; font-weight: 600;
            cursor: pointer; font-family: inherit; background: transparent;
            color: var(--text-secondary); transition: all 0.2s;
        }
        .split-toggle button.active { background: var(--accent); color: white; }
        .loading { color: var(--text-tertiary); font-size: 14px; padding: 40px; text-align: center; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Label Review</h1>
        <p>Review YOLO labels — keep good ones, delete bad ones</p>
    </div>

    <div class="split-toggle">
        <button class="active" onclick="switchSplit('train')">Train</button>
        <button onclick="switchSplit('val')">Val</button>
    </div>

    <div class="progress-bar"><div class="progress-fill" id="progressFill" style="width:0%"></div></div>

    <div class="stats-row">
        <div class="stat progress" id="statProgress">0 / 0</div>
        <div class="stat kept" id="statKept">0 kept</div>
        <div class="stat deleted" id="statDeleted">0 deleted</div>
    </div>

    <div class="image-container" id="reviewContainer">
        <div class="image-info">
            <span class="filename" id="fileName">Loading...</span>
            <span class="det-count" id="detCount">0 detections</span>
        </div>
        <div class="image-wrap">
            <img id="reviewImage" src="" alt="Loading...">
        </div>
        <div class="buttons">
            <button class="btn btn-delete" onclick="doDelete()">Delete <span class="kbd">N</span></button>
            <button class="btn btn-skip" onclick="doSkip()">Skip <span class="kbd">S</span></button>
            <button class="btn btn-keep" onclick="doKeep()">Keep <span class="kbd">Y</span></button>
        </div>
    </div>

    <div class="done-screen" id="doneScreen">
        <h2>Review Complete</h2>
        <p id="doneSummary"></p>
        <p style="margin-top:12px"><a href="/" style="color:var(--accent)">Back to OtterVision</a></p>
    </div>

    <script>
        let currentIndex = 0, currentSplit = 'train', currentFilename = '';
        let kept = 0, deleted = 0, total = 0;

        async function switchSplit(split) {
            currentSplit = split;
            currentIndex = 0; kept = 0; deleted = 0;
            document.querySelectorAll('.split-toggle button').forEach(b => b.classList.remove('active'));
            event.target.classList.add('active');
            await loadInfo();
            await loadImage();
        }

        async function loadInfo() {
            const r = await fetch('/api/review/info?split=' + currentSplit);
            const d = await r.json();
            total = d.total;
            updateStats();
        }

        async function loadImage() {
            const r = await fetch('/api/review/image?index=' + currentIndex + '&split=' + currentSplit);
            const d = await r.json();

            if (d.done) {
                document.getElementById('reviewContainer').style.display = 'none';
                document.getElementById('doneScreen').style.display = 'block';
                document.getElementById('doneSummary').textContent =
                    'Reviewed ' + (kept + deleted) + ' images. Kept: ' + kept + ', Deleted: ' + deleted;
                return;
            }

            currentFilename = d.filename;
            document.getElementById('reviewImage').src = 'data:image/jpeg;base64,' + d.image;
            document.getElementById('fileName').textContent = d.filename;
            document.getElementById('detCount').textContent = d.detections + ' detection(s)';
            total = d.total;
            updateStats();
        }

        function updateStats() {
            document.getElementById('statProgress').textContent = (currentIndex + 1) + ' / ' + total;
            document.getElementById('statKept').textContent = kept + ' kept';
            document.getElementById('statDeleted').textContent = deleted + ' deleted';
            const pct = total > 0 ? ((currentIndex + 1) / total * 100) : 0;
            document.getElementById('progressFill').style.width = pct + '%';
        }

        async function doKeep() {
            kept++;
            currentIndex++;
            await loadImage();
        }

        async function doDelete() {
            await fetch('/api/review/delete', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filename: currentFilename, split: currentSplit })
            });
            deleted++;
            // Don't increment index since the list shifted after delete
            // But we need to reload info to get updated total
            await loadInfo();
            await loadImage();
        }

        async function doSkip() {
            currentIndex++;
            await loadImage();
        }

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === 'y' || e.key === 'Y') doKeep();
            else if (e.key === 'n' || e.key === 'N') doDelete();
            else if (e.key === 's' || e.key === 'S') doSkip();
        });

        // Init
        loadInfo().then(() => loadImage());
    </script>
</body>
</html>"""