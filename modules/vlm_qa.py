"""
VLM Module — LLaVA
Two modes:
  1. Auto-label: runs every few seconds, returns short activity labels per otter
  2. Detailed Q&A: user asks a question, gets a rich description with temporal context
"""

import torch
import numpy as np
import cv2
import time
import threading
from PIL import Image
from typing import Optional, List, Dict
from collections import deque

from config import (
    DEVICE, LLAVA_MODEL_ID, LLAVA_MAX_NEW_TOKENS,
    LLAVA_TEMPERATURE, LLAVA_SYSTEM_PROMPT, LLAVA_DTYPE,
    ACTIVITY_CLASSES
)

LABEL_PROMPT = f"""You are observing otters through a zoo camera. Look at this image and for each otter visible, output ONLY a short label in this exact format, one per line:

otter 1: [activity]
otter 2: [activity]

Activities must be one of: {', '.join(ACTIVITY_CLASSES)}

If an otter is holding or interacting with something, add it after a dash:
otter 1: eating - shellfish
otter 2: floating - rock

If no otters are visible, just say: no otters visible

Do NOT add any other text, explanation, or formatting. Just the labels."""

DETAIL_PROMPT = """You are an expert marine biologist observing otters through a live zoo camera feed.
The user has asked a question. You are seeing the current frame from the camera.
Give a detailed, informative answer about what the otters are doing.
Describe their behavior, body positions, what they're holding or interacting with,
how many are visible, and any social interactions between them.
Be specific and observational. If you can't clearly see something, say so."""


class OtterVLM:
    def __init__(self, device: str = DEVICE):
        self.device = device
        self.model = None
        self.processor = None
        self._loaded = False
        self._loading = False

        # Auto-label state
        self.current_labels = {}
        self.last_label_time = 0
        self.label_interval = 4.0  # seconds between auto-labels
        self._label_lock = threading.Lock()
        self._label_thread = None

        # Frame history for temporal context
        self.frame_history = deque(maxlen=5)

    def load_models(self):
        if self._loaded or self._loading:
            return
        self._loading = True

        print(f"[VLM] Loading LLaVA ({LLAVA_MODEL_ID.split('/')[-1]})...")
        try:
            from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

            self.processor = LlavaNextProcessor.from_pretrained(LLAVA_MODEL_ID)
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                LLAVA_MODEL_ID,
                torch_dtype=LLAVA_DTYPE,
                low_cpu_mem_usage=True,
                device_map="auto",
            )
            self.model.eval()
            print("[VLM] LLaVA loaded.")
            self._loaded = True
        except Exception as e:
            print(f"[VLM] Error loading LLaVA: {e}")
            self.model = None
        self._loading = False

    @torch.no_grad()
    def _run_inference(self, frame: np.ndarray, prompt: str, max_tokens: int = 200) -> str:
        if self.model is None:
            return ""

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        full_prompt = f"[INST] <image>\n{prompt}\n[/INST]"

        try:
            inputs = self.processor(text=full_prompt, images=pil_image, return_tensors="pt")
            inputs = {k: v.to(self.model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}

            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=LLAVA_TEMPERATURE,
                do_sample=LLAVA_TEMPERATURE > 0,
            )

            generated = output_ids[:, inputs["input_ids"].shape[1]:]
            return self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        except Exception as e:
            return f"Error: {str(e)}"

    def get_activity_labels(self, frame: np.ndarray) -> Dict[int, Dict[str, str]]:
        """
        Quick auto-label: returns dict like {0: {"activity": "swimming", "object": "none"}, ...}
        Runs in background thread so it doesn't block the video stream.
        """
        now = time.time()
        if now - self.last_label_time < self.label_interval:
            return self.current_labels

        if not self._loaded:
            self.load_models()
            if not self._loaded:
                return {}

        # Store frame for history
        self.frame_history.append(frame.copy())

        # Don't block — run in background if not already running
        if self._label_thread is None or not self._label_thread.is_alive():
            self._label_thread = threading.Thread(
                target=self._auto_label_worker,
                args=(frame.copy(),),
                daemon=True
            )
            self._label_thread.start()
            self.last_label_time = now

        return self.current_labels

    def _auto_label_worker(self, frame: np.ndarray):
        """Background worker that runs LLaVA and parses activity labels."""
        raw = self._run_inference(frame, LABEL_PROMPT, max_tokens=100)
        labels = self._parse_labels(raw)
        with self._label_lock:
            self.current_labels = labels

    def _parse_labels(self, raw: str) -> Dict[int, Dict[str, str]]:
        """Parse LLaVA output like 'otter 1: swimming - rock' into structured dict."""
        labels = {}
        for line in raw.strip().split("\n"):
            line = line.strip().lower()
            if not line.startswith("otter"):
                continue
            try:
                # "otter 1: eating - shellfish"
                parts = line.split(":")
                if len(parts) < 2:
                    continue
                otter_num = int(parts[0].replace("otter", "").strip()) - 1
                activity_part = parts[1].strip()

                if " - " in activity_part:
                    activity, obj = activity_part.split(" - ", 1)
                    activity = activity.strip()
                    obj = obj.strip()
                else:
                    activity = activity_part.strip()
                    obj = "none"

                # Validate activity is in our list (fuzzy match)
                matched_activity = "unknown"
                for a in ACTIVITY_CLASSES:
                    if a in activity or activity in a:
                        matched_activity = a
                        break

                labels[otter_num] = {
                    "activity": matched_activity,
                    "object": obj,
                }
            except (ValueError, IndexError):
                continue

        return labels

    def ask_detailed(self, frame: np.ndarray, question: str) -> str:
        """
        Detailed Q&A mode. Uses current frame + context from recent frames.
        """
        if not self._loaded:
            self.load_models()
            if not self._loaded:
                return "VLM not available. Check installation."

        # Store frame
        self.frame_history.append(frame.copy())

        # Build context-aware prompt
        context = ""
        if self.current_labels:
            label_lines = []
            for idx, info in sorted(self.current_labels.items()):
                obj_str = f" (holding: {info['object']})" if info['object'] != 'none' else ""
                label_lines.append(f"Otter {idx+1}: {info['activity']}{obj_str}")
            context = f"\nRecent observations from the last few seconds:\n" + "\n".join(label_lines) + "\n"

        prompt = f"""{DETAIL_PROMPT}
{context}
User question: {question}"""

        return self._run_inference(frame, prompt, max_tokens=LLAVA_MAX_NEW_TOKENS)

    def is_loaded(self) -> bool:
        return self._loaded and self.model is not None