"""
VLM Q&A Module — LLaVA (ON-DEMAND ONLY)
Only loaded when the user asks their first question. Never runs per-frame.
"""

import torch
import numpy as np
import cv2
from PIL import Image
from typing import Optional

from config import (
    DEVICE, LLAVA_MODEL_ID, LLAVA_MAX_NEW_TOKENS,
    LLAVA_TEMPERATURE, LLAVA_SYSTEM_PROMPT, LLAVA_DTYPE, HW_TIER
)


class OtterVLM:
    def __init__(self, device: str = DEVICE):
        self.device = device
        self.model = None
        self.processor = None
        self._loaded = False

    def load_models(self):
        if self._loaded:
            return

        print("[VLM] Loading LLaVA (this may take a minute)...")
        try:
            from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

            self.processor = LlavaNextProcessor.from_pretrained(LLAVA_MODEL_ID)
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                LLAVA_MODEL_ID,
                torch_dtype=LLAVA_DTYPE,
                low_cpu_mem_usage=True,
                device_map="auto" if self.device == "cuda" else None,
            )
            if self.device == "cuda" and self.model.device.type != "cuda":
                self.model = self.model.to(self.device)
            self.model.eval()
            print("[VLM] LLaVA loaded.")
            self._loaded = True
        except Exception as e:
            print(f"[VLM] Error loading LLaVA: {e}")
            self.model = None

    @torch.no_grad()
    def ask(self, frame: np.ndarray, question: str) -> str:
        if not self._loaded:
            self.load_models()
        if self.model is None:
            return "VLM not available. Check installation."

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        prompt = f"[INST] <image>\n{LLAVA_SYSTEM_PROMPT}\n\nUser question: {question}\n[/INST]"

        try:
            inputs = self.processor(text=prompt, images=pil_image, return_tensors="pt")
            inputs = {k: v.to(self.model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}

            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=LLAVA_MAX_NEW_TOKENS,
                temperature=LLAVA_TEMPERATURE,
                do_sample=LLAVA_TEMPERATURE > 0,
            )

            generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
            return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        except Exception as e:
            return f"Error: {str(e)}"

    def is_loaded(self) -> bool:
        return self._loaded and self.model is not None
