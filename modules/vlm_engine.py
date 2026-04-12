"""
VLM Engine — Qwen2.5-VL-7B on GPU 1
Activity labels (background thread) + detailed Q&A (on demand).
"""

import torch
import numpy as np
import cv2
import time
import threading
from PIL import Image
from typing import Dict, List

from config import QA_MODEL, NUM_GPUS


class VLMEngine:
    def __init__(self):
        self.model = None
        self.processor = None
        self._loaded = False
        self._loading = False
        self.device = "cuda:1" if NUM_GPUS >= 2 else ("cuda:0" if NUM_GPUS == 1 else "cpu")
        self.current_labels = {}
        self._label_lock = threading.Lock()
        self._label_thread = None
        self._last_label_time = 0
        self.label_interval = 4.0

    def load_model(self):
        if self._loaded or self._loading:
            return
        self._loading = True
        print(f"[VLM] Loading {QA_MODEL} on {self.device}...")
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            self.processor = AutoProcessor.from_pretrained(QA_MODEL)
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                QA_MODEL, torch_dtype=torch.float16, device_map={"": self.device},
            )
            self.model.eval()
            print(f"[VLM] Loaded on {self.device}.")
            self._loaded = True
        except Exception as e:
            print(f"[VLM] Error: {e}")
            import traceback
            traceback.print_exc()
        self._loading = False

    @torch.no_grad()
    def _run_inference(self, pil_image, prompt, max_tokens=300):
        if not self._loaded:
            return ""
        messages = [{"role": "user", "content": [{"type": "image", "image": pil_image}, {"type": "text", "text": prompt}]}]
        try:
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=[pil_image], return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
            output_ids = self.model.generate(**inputs, max_new_tokens=max_tokens, temperature=0.3, do_sample=False)
            generated = output_ids[:, inputs["input_ids"].shape[1]:]
            return self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        except Exception as e:
            return ""

    def get_activity_labels(self, frame, num_otters):
        now = time.time()
        if now - self._last_label_time < self.label_interval:
            return self.current_labels
        if not self._loaded:
            if not self._loading:
                threading.Thread(target=self.load_model, daemon=True).start()
            return {}
        if self._label_thread is None or not self._label_thread.is_alive():
            self._last_label_time = now
            self._label_thread = threading.Thread(target=self._auto_label, args=(frame.copy(), num_otters), daemon=True)
            self._label_thread.start()
        return self.current_labels

    def _auto_label(self, frame, num_otters):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        prompt = f"""I can see {num_otters} otter(s) in this zoo camera image. For each otter, describe what it is doing in one or two words.
Use ONLY these labels: floating, diving, eating, grooming, playing, socializing, resting, exploring, active
Reply in this format, one per line:
otter 1: [activity]
otter 2: [activity]
If holding something, add it: otter 1: eating - shellfish
Only output the labels."""
        raw = self._run_inference(pil_image, prompt, max_tokens=100)
        labels = self._parse_labels(raw)
        with self._label_lock:
            self.current_labels = labels

    def _parse_labels(self, raw):
        labels = {}
        valid = {"floating", "diving", "eating", "grooming", "playing", "socializing", "resting", "exploring", "active"}
        for line in raw.strip().split("\n"):
            line = line.strip().lower()
            if not line.startswith("otter"):
                continue
            try:
                parts = line.split(":")
                if len(parts) < 2:
                    continue
                num = int(parts[0].replace("otter", "").strip()) - 1
                ap = parts[1].strip()
                if " - " in ap:
                    activity, obj = ap.split(" - ", 1)
                else:
                    activity, obj = ap, "none"
                activity = activity.strip()
                if activity not in valid:
                    for v in valid:
                        if v in activity:
                            activity = v
                            break
                    else:
                        activity = "active"
                labels[num] = {"activity": activity, "object": obj.strip()}
            except:
                continue
        return labels

    def ask_detailed(self, frame, question, context=""):
        if not self._loaded:
            self.load_model()
            if not self._loaded:
                return "VLM loading, try again."
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        system = "You are an expert marine biologist observing sea otters through a live zoo camera. Give detailed observations."
        if context:
            system += f"\nCurrent detections: {context}"
        return self._run_inference(pil_image, f"{system}\n\nUser question: {question}", max_tokens=400)

    def is_loaded(self):
        return self._loaded