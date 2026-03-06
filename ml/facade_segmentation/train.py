"""Facade segmentation trainer.

Provides a self-contained training loop with:
- mixed-precision support
- TensorBoard logging
- checkpoint saving / resuming
- configurable loss (cross-entropy + optional Dice)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ml.common.base_model import BaseSegmentationModel
from ml.common.metrics import compute_iou, compute_pixel_accuracy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


class CombinedSegmentationLoss(nn.Module):
    """Cross-entropy + Dice loss combination.

    Args:
        ce_weight: Weight for cross-entropy term.
        dice_weight: Weight for Dice loss term.
        ignore_index: Class index ignored in cross-entropy.
        smooth: Smoothing factor for Dice computation.
    """

    def __init__(
        self,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        ignore_index: int = 255,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.ce_w = ce_weight
        self.dice_w = dice_weight
        self.smooth = smooth
        self.ignore_index = ignore_index

    def _dice_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)  # (B, C, H, W)
        valid = (targets != self.ignore_index).float()  # (B, H, W)
        targets_clamped = targets.clone()
        targets_clamped[targets == self.ignore_index] = 0
        targets_one_hot = F.one_hot(targets_clamped, num_classes).permute(0, 3, 1, 2).float()
        targets_one_hot = targets_one_hot * valid.unsqueeze(1)

        dims = (0, 2, 3)
        inter = (probs * targets_one_hot).sum(dims)
        cardinality = probs.sum(dims) + targets_one_hot.sum(dims)
        dice = (2.0 * inter + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.ce_w * self.ce(logits, targets) + self.dice_w * self._dice_loss(logits, targets)


# ---------------------------------------------------------------------------
# Trainer configuration
# ---------------------------------------------------------------------------


@dataclass
class SegmentationTrainerConfig:
    """Hyper-parameters and paths for the segmentation trainer."""

    output_dir: str = "checkpoints/facade_segmentation"
    num_epochs: int = 50
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    device: str = "auto"
    mixed_precision: bool = True
    log_every_n_steps: int = 10
    val_every_n_epochs: int = 1
    save_every_n_epochs: int = 5
    ce_weight: float = 1.0
    dice_weight: float = 1.0
    ignore_index: int = 255
    extra: dict[str, Any] = field(default_factory=dict)

    def resolve_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class SegmentationTrainer:
    """Full training loop for facade segmentation models.

    Example::

        config = SegmentationTrainerConfig(num_epochs=30)
        model = DeepLabV3PlusSegmentation()
        trainer = SegmentationTrainer(model, config)
        trainer.train(train_loader, val_loader)

    Args:
        model: A :class:`~ml.common.base_model.BaseSegmentationModel` instance.
        config: Training hyper-parameters and paths.
    """

    def __init__(
        self,
        model: BaseSegmentationModel,
        config: SegmentationTrainerConfig | None = None,
    ) -> None:
        self.model = model
        self.cfg = config or SegmentationTrainerConfig()
        self.device = self.cfg.resolve_device()
        self.model.to(self.device)

        self.loss_fn = CombinedSegmentationLoss(
            ce_weight=self.cfg.ce_weight,
            dice_weight=self.cfg.dice_weight,
            ignore_index=self.cfg.ignore_index,
        )
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.mixed_precision)

        self.output_dir = Path(self.cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.output_dir / "logs"))

        self._global_step = 0

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        num_epochs: int | None = None,
    ) -> dict[str, list[float]]:
        """Run the training loop.

        Args:
            train_loader: DataLoader yielding ``(image, mask)`` batches.
            val_loader: Optional validation DataLoader.
            num_epochs: Override ``config.num_epochs`` if provided.

        Returns:
            History dict with keys ``"train_loss"``, ``"val_miou"``, etc.
        """
        epochs = num_epochs or self.cfg.num_epochs
        scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-6)

        history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_miou": [],
            "val_pixel_acc": [],
        }

        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch(train_loader, epoch)
            history["train_loss"].append(train_loss)
            scheduler.step()

            if val_loader and epoch % self.cfg.val_every_n_epochs == 0:
                metrics = self._val_epoch(val_loader, epoch)
                history["val_loss"].append(metrics["loss"])
                history["val_miou"].append(metrics["miou"])
                history["val_pixel_acc"].append(metrics["pixel_acc"])

            if epoch % self.cfg.save_every_n_epochs == 0 or epoch == epochs:
                self._save_checkpoint(epoch)

        self.writer.close()
        return history

    def _train_epoch(self, loader: DataLoader, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        t0 = time.time()

        for step, (images, masks) in enumerate(loader, 1):
            images = images.to(self.device)
            masks = masks.to(self.device)

            with torch.cuda.amp.autocast(enabled=self.cfg.mixed_precision):
                logits = self.model(images)
                loss = self.loss_fn(logits, masks)

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            self._global_step += 1

            if step % self.cfg.log_every_n_steps == 0:
                avg = total_loss / step
                elapsed = time.time() - t0
                logger.info(
                    "Epoch %d/%d  step %d  loss=%.4f  (%.1fs)",
                    epoch,
                    self.cfg.num_epochs,
                    step,
                    avg,
                    elapsed,
                )
                self.writer.add_scalar("train/loss", avg, self._global_step)

        return total_loss / max(len(loader), 1)

    @torch.no_grad()
    def _val_epoch(self, loader: DataLoader, epoch: int) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        all_pred, all_gt = [], []

        for images, masks in loader:
            images = images.to(self.device)
            masks = masks.to(self.device)

            with torch.cuda.amp.autocast(enabled=self.cfg.mixed_precision):
                logits = self.model(images)
                loss = self.loss_fn(logits, masks)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_pred.append(preds.cpu())
            all_gt.append(masks.cpu())

        pred_cat = torch.cat([p.flatten() for p in all_pred])
        gt_cat = torch.cat([g.flatten() for g in all_gt])

        iou_metrics = compute_iou(pred_cat, gt_cat, self.model.num_classes)
        pixel_acc = compute_pixel_accuracy(pred_cat, gt_cat)
        val_loss = total_loss / max(len(loader), 1)

        metrics = {
            "loss": val_loss,
            "miou": iou_metrics["mean_iou"],
            "pixel_acc": pixel_acc,
        }
        logger.info(
            "Val  loss=%.4f  mIoU=%.4f  pixel_acc=%.4f",
            val_loss,
            iou_metrics["mean_iou"],
            pixel_acc,
        )
        self.writer.add_scalar("val/loss", val_loss, epoch)
        self.writer.add_scalar("val/miou", iou_metrics["mean_iou"], epoch)
        self.writer.add_scalar("val/pixel_acc", pixel_acc, epoch)
        return metrics

    def _save_checkpoint(self, epoch: int) -> None:
        ckpt_path = self.output_dir / f"checkpoint_epoch_{epoch:04d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.cfg,
            },
            ckpt_path,
        )
        logger.info("Checkpoint saved → %s", ckpt_path)

    def load_checkpoint(self, path: str | Path) -> int:
        """Load a checkpoint and return the epoch number."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        epoch: int = ckpt["epoch"]
        logger.info("Resumed from checkpoint '%s' (epoch %d)", path, epoch)
        return epoch
