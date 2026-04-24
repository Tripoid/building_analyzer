"""
Stable-Diffusion-2 Inpaint provider — optional, higher-quality fallback.

We never autoload this — it consumes 3.5-5GB VRAM with fp16 + sequential CPU
offload, which is fine alongside the detection stack on a T4 but tight. The
pipeline is built on-demand on the first /api/restore?quality=high call, and
falls back to LaMa on any OOM.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import cv2
import numpy as np
from PIL import Image

from backend.core.config import get_settings
from backend.restoration.providers.base import InpaintProvider, _free_vram_mb

logger = logging.getLogger(__name__)

_pipe: Any = None
_pipe_lock = asyncio.Lock()

DEFAULT_PROMPT = (
    "restored clean building facade, same architecture and materials, "
    "no defects, photorealistic, professional photo"
)


async def _get_pipeline():
    global _pipe
    settings = get_settings()
    async with _pipe_lock:
        if _pipe is None:
            import torch
            from diffusers import StableDiffusionInpaintPipeline

            vram = _free_vram_mb()
            if vram is not None and vram < settings.min_free_vram_mb_for_sd:
                raise MemoryError(
                    f"Not enough free VRAM for SD inpaint: {vram}MB < "
                    f"{settings.min_free_vram_mb_for_sd}MB threshold"
                )

            logger.info("Loading SD inpaint pipeline: %s", settings.sd_model_id)
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            pipe = StableDiffusionInpaintPipeline.from_pretrained(
                settings.sd_model_id, torch_dtype=dtype, safety_checker=None
            )
            if torch.cuda.is_available():
                pipe.enable_sequential_cpu_offload()
            else:
                pipe.to("cpu")
            _pipe = pipe
        return _pipe


def _pad_to_multiple(img: np.ndarray, mult: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    h, w = img.shape[:2]
    nh = ((h + mult - 1) // mult) * mult
    nw = ((w + mult - 1) // mult) * mult
    pad = ((0, nh - h), (0, nw - w)) + ((0, 0),) * (img.ndim - 2)
    return np.pad(img, pad, mode="edge"), (0, nh - h, 0, nw - w)


class SDInpaintProvider(InpaintProvider):
    name = "sd"

    async def inpaint(
        self,
        image_rgb: np.ndarray,
        mask_u8: np.ndarray,
        prompt: str | None = None,
    ) -> np.ndarray:
        if image_rgb.shape[:2] != mask_u8.shape[:2]:
            mask_u8 = cv2.resize(
                mask_u8,
                (image_rgb.shape[1], image_rgb.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        h0, w0 = image_rgb.shape[:2]
        # Resize to at most 1024px and pad to 8x multiple.
        max_side = 1024
        scale = min(1.0, max_side / max(h0, w0))
        if scale < 1.0:
            new_size = (int(w0 * scale), int(h0 * scale))
            image_rgb = cv2.resize(image_rgb, new_size, interpolation=cv2.INTER_AREA)
            mask_u8 = cv2.resize(mask_u8, new_size, interpolation=cv2.INTER_NEAREST)

        img_pad, (pt, pb, pl, pr) = _pad_to_multiple(image_rgb, 8)
        mask_pad, _ = _pad_to_multiple(mask_u8, 8)

        pipe = await _get_pipeline()
        prompt = prompt or DEFAULT_PROMPT

        result = await asyncio.to_thread(
            lambda: pipe(
                prompt=prompt,
                image=Image.fromarray(img_pad),
                mask_image=Image.fromarray(mask_pad),
                num_inference_steps=30,
                guidance_scale=7.5,
            ).images[0]
        )
        arr = np.asarray(result.convert("RGB"))
        # Crop pad away
        arr = arr[: arr.shape[0] - pb, : arr.shape[1] - pr]
        # Restore original size if we downscaled
        if scale < 1.0:
            arr = cv2.resize(arr, (w0, h0), interpolation=cv2.INTER_CUBIC)
        return arr
