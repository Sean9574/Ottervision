"""
Labeling Tool — annotate otter crops with activity + object classes.
"""

import cv2
import os
import json
from config import ACTIVITY_CLASSES, OBJECT_CLASSES


def run_labeling_tool(crops_dir: str, labels_file: str):
    if os.path.exists(labels_file):
        with open(labels_file) as f:
            labels = json.load(f)
    else:
        labels = {}

    files = sorted([f for f in os.listdir(crops_dir) if f.endswith((".jpg", ".png"))])
    unlabeled = [f for f in files if f not in labels]

    print(f"[Labeler] {len(unlabeled)} unlabeled, {len(labels)} already labeled")
    if not unlabeled:
        print("[Labeler] All done!")
        return

    print("\n" + "=" * 50)
    print("ACTIVITY KEYS:")
    for i, a in enumerate(ACTIVITY_CLASSES):
        print(f"  {i+1} = {a}")
    print("\nOBJECT KEYS (press after activity):")
    for i, o in enumerate(OBJECT_CLASSES):
        print(f"  {chr(ord('a')+i)} = {o}")
    print("\nSpace=skip | u=undo | q=save+quit")
    print("=" * 50)

    current_activity = None
    history = []

    for idx, fname in enumerate(unlabeled):
        frame = cv2.imread(os.path.join(crops_dir, fname))
        if frame is None:
            continue

        display = frame.copy()
        h, w = display.shape[:2]
        # Scale up small crops for easier viewing
        if max(h, w) < 300:
            scale = 300 / max(h, w)
            display = cv2.resize(display, None, fx=scale, fy=scale)

        title = f"{idx+1}/{len(unlabeled)} | {fname}"
        if current_activity:
            title += f" | Activity: {current_activity} → pick object"

        cv2.imshow("OtterVision Labeler", display)
        cv2.setWindowTitle("OtterVision Labeler", title)

        while True:
            key = cv2.waitKey(0) & 0xFF

            if key == ord('q'):
                _save(labels, labels_file)
                cv2.destroyAllWindows()
                return

            if key == ord(' '):
                current_activity = None
                break

            if key == ord('u') and history:
                last = history.pop()
                labels.pop(last, None)
                print(f"  Undid {last}")
                current_activity = None
                break

            if ord('1') <= key <= ord('9'):
                i = key - ord('1')
                if i < len(ACTIVITY_CLASSES):
                    current_activity = ACTIVITY_CLASSES[i]
                    cv2.setWindowTitle("OtterVision Labeler",
                                       f"{title} | {current_activity} → pick object (a-h)")

            if current_activity and ord('a') <= key <= ord('h'):
                i = key - ord('a')
                if i < len(OBJECT_CLASSES):
                    labels[fname] = {"activity": current_activity, "object": OBJECT_CLASSES[i]}
                    history.append(fname)
                    print(f"  {fname}: {current_activity} + {OBJECT_CLASSES[i]}")
                    current_activity = None
                    if len(labels) % 50 == 0:
                        _save(labels, labels_file)
                    break

    _save(labels, labels_file)
    cv2.destroyAllWindows()


def _save(labels, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(labels, f, indent=2)
    print(f"[Labeler] Saved {len(labels)} labels")
