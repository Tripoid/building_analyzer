"""
LaMa inpainting — the default local provider.

Weighs ~200MB, peaks at 1-3GB VRAM for 1024px inputs. Handles the typical
facade defects (plaster flakes, cracks, paint chips) well and never OOMs on
our T4 budget. Model is loaded lazily — the very first /api/restore call
downloads the checkpoint if needed.

Interface-wise we go through `simple-lama-inpainting`, a minimal wrapper with
no extra dependencies beyond torch.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import cv2
import numpy as np

from backend.restoration.providers.base import InpaintProvider, _free_vram_mb

logger = logging.getLogger(__name__)

_lama_singleton: Any = None
_lama_lock = asyncio.Lock()


async def _get_lama():
    global _lama_singleton
    async with _lama_lock:
        if _lama_singleton is None:
            logger.info("Loading LaMa inpainting model (first call)")
            from simple_lama_inpainting import SimpleLama

            _lama_singleton = SimpleLama()
        return _lama_singleton


class LamaProvider(InpaintProvider):
    name = "lama"

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

        vram = _free_vram_mb()
        if vram is not None and vram < 1000:
            logger.warning("Very low VRAM (%dMB) — LaMa may still OOM", vram)

        lama = await _get_lama()
        # SimpleLama accepts PIL images; convert here for crisp typing.
        from PIL import Image

        result_pil = await asyncio.to_thread(
            lama,
            Image.fromarray(image_rgb),
            Image.fromarray(mask_u8),
        )
        return np.asarray(result_pil.convert("RGB"))
