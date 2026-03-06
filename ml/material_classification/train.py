"""Material classification trainer."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ml.common.base_model import BaseClassificationModel
from ml.common.metrics import compute_top_k_accuracy

logger = logging.getLogger(__name__)


@dataclass
class MaterialTrainerConfig:
    """Hyper-parameters and paths for the material classification trainer."""

    output_dir: str = "checkpoints/material_classification"
    num_epochs: int = 30
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    device: str = "auto"
    mixed_precision: bool = True
    log_every_n_steps: int = 10
    val_every_n_epochs: int = 1
    save_every_n_epochs: int = 5
    extra: dict[str, Any] = field(default_factory=dict)

    def resolve_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)


class MaterialTrainer:
    """Training loop for building material classification models.

    Args:
        model: A :class:`~ml.common.base_model.BaseClassificationModel`.
        config: Training configuration.
    """

    def __init__(
        self,
        model: BaseClassificationModel,
        config: MaterialTrainerConfig | None = None,
    ) -> None:
        self.model = model
        self.cfg = config or MaterialTrainerConfig()
        self.device = self.cfg.resolve_device()
        self.model.to(self.device)

        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=self.cfg.label_smoothing)
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
            train_loader: DataLoader yielding ``(image, label)`` batches.
            val_loader: Optional validation DataLoader.
            num_epochs: Override ``config.num_epochs`` if provided.

        Returns:
            History dict with ``"train_loss"``, ``"val_loss"``, ``"val_top1"``,
            and ``"val_top5"`` keys.
        """
        epochs = num_epochs or self.cfg.num_epochs
        scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-6)

        history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_top1": [],
            "val_top5": [],
        }

        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch(train_loader, epoch)
            history["train_loss"].append(train_loss)
            scheduler.step()

            if val_loader and epoch % self.cfg.val_every_n_epochs == 0:
                metrics = self._val_epoch(val_loader, epoch)
                history["val_loss"].append(metrics["loss"])
                history["val_top1"].append(metrics["top1"])
                history["val_top5"].append(metrics["top5"])

            if epoch % self.cfg.save_every_n_epochs == 0 or epoch == epochs:
                self._save_checkpoint(epoch)

        self.writer.close()
        return history

    def _train_epoch(self, loader: DataLoader, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        t0 = time.time()

        for step, (images, labels) in enumerate(loader, 1):
            images = images.to(self.device)
            labels = labels.to(self.device)

            with torch.cuda.amp.autocast(enabled=self.cfg.mixed_precision):
                logits = self.model(images)
                loss = self.loss_fn(logits, labels)

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            self._global_step += 1

            if step % self.cfg.log_every_n_steps == 0:
                avg = total_loss / step
                logger.info(
                    "Epoch %d  step %d  loss=%.4f  (%.1fs)",
                    epoch,
                    step,
                    avg,
                    time.time() - t0,
                )
                self.writer.add_scalar("train/loss", avg, self._global_step)

        return total_loss / max(len(loader), 1)

    @torch.no_grad()
    def _val_epoch(self, loader: DataLoader, epoch: int) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        all_logits: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []

        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            with torch.cuda.amp.autocast(enabled=self.cfg.mixed_precision):
                logits = self.model(images)
                loss = self.loss_fn(logits, labels)

            total_loss += loss.item()
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

        logits_cat = torch.cat(all_logits)
        labels_cat = torch.cat(all_labels)

        val_loss = total_loss / max(len(loader), 1)
        top1 = compute_top_k_accuracy(logits_cat, labels_cat, k=1)
        top5 = compute_top_k_accuracy(logits_cat, labels_cat, k=min(5, self.model.num_classes))

        logger.info("Val  loss=%.4f  top1=%.4f  top5=%.4f", val_loss, top1, top5)
        self.writer.add_scalar("val/loss", val_loss, epoch)
        self.writer.add_scalar("val/top1", top1, epoch)
        self.writer.add_scalar("val/top5", top5, epoch)

        return {"loss": val_loss, "top1": top1, "top5": top5}

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
