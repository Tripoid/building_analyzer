"""Abstract inpaint provider interface (local or remote)."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class InpaintProvider(ABC):
    name: str = "base"

    @abstractmethod
    async def inpaint(
        self,
        image_rgb: np.ndarray,
        mask_u8: np.ndarray,
        prompt: str | None = None,
    ) -> np.ndarray:
        """Return the restored RGB image, same HxW as input."""


def _free_vram_mb() -> int | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free, _total = torch.cuda.mem_get_info()
        return int(free / 1024 / 1024)
    except Exception:
        return None
