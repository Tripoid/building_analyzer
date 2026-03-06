"""Material classification model implementations.

Two concrete classifiers are provided:

* :class:`CNNMaterialClassifier` — a convolutional network with global average
  pooling, suitable for fast inference and small datasets.
* :class:`ViTMaterialClassifier` — a Vision Transformer encoder with a
  linear classification head, offering better long-range feature modelling.

Both are registered with :class:`~ml.common.registry.ModelRegistry`.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ml.common.base_model import BaseClassificationModel
from ml.common.registry import ModelRegistry
from ml.material_classification.dataset import MATERIAL_CLASS_NAMES


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class _ConvBnRelu(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class _ResBlock(nn.Module):
    """Basic residual block with optional downsampling."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = _ConvBnRelu(in_ch, out_ch, stride=stride)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.downsample: nn.Module = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        return self.relu(out + self.downsample(x))


# ---------------------------------------------------------------------------
# CNN classifier
# ---------------------------------------------------------------------------


@ModelRegistry.register("cnn_classifier", namespace="classification")
class CNNMaterialClassifier(BaseClassificationModel):
    """Lightweight ResNet-inspired CNN for material classification.

    Args:
        num_classes: Number of material classes.
        class_names: Ordered list of class names.
        base_channels: Number of channels in the first stage.
        dropout: Dropout probability before the final linear layer.
    """

    def __init__(
        self,
        num_classes: int = len(MATERIAL_CLASS_NAMES),
        class_names: list[str] = MATERIAL_CLASS_NAMES,
        base_channels: int = 32,
        dropout: float = 0.3,
    ) -> None:
        super().__init__(num_classes=num_classes, class_names=class_names)

        ch = base_channels
        self.stem = _ConvBnRelu(3, ch, stride=2)
        self.layer1 = nn.Sequential(_ResBlock(ch, ch * 2, stride=2), _ResBlock(ch * 2, ch * 2))
        self.layer2 = nn.Sequential(_ResBlock(ch * 2, ch * 4, stride=2), _ResBlock(ch * 4, ch * 4))
        self.layer3 = nn.Sequential(_ResBlock(ch * 4, ch * 8, stride=2), _ResBlock(ch * 8, ch * 8))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(ch * 8, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.stem(images)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        return self.fc(self.dropout(x))


# ---------------------------------------------------------------------------
# Vision Transformer classifier
# ---------------------------------------------------------------------------


class _PatchEmbed(nn.Module):
    """Split image into non-overlapping patches and embed them."""

    def __init__(self, img_size: int, patch_size: int, in_ch: int, embed_dim: int) -> None:
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) → (B, N, embed_dim)
        return self.proj(x).flatten(2).transpose(1, 2)


class _MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm(x)
        out, _ = self.attn(normed, normed, normed)
        return x + out


class _FFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class _TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn = _MultiHeadSelfAttention(dim, num_heads, dropout)
        self.ffn = _FFN(dim, int(dim * mlp_ratio), dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(x)
        return self.ffn(x)


@ModelRegistry.register("vit_classifier", namespace="classification")
class ViTMaterialClassifier(BaseClassificationModel):
    """Vision Transformer classifier for building material identification.

    A compact ViT-Tiny style model with configurable depth and width.

    Args:
        num_classes: Number of material classes.
        class_names: Ordered list of class names.
        img_size: Expected input image size (must be divisible by *patch_size*).
        patch_size: Size of each patch (pixels).
        embed_dim: Token embedding dimensionality.
        depth: Number of Transformer blocks.
        num_heads: Number of self-attention heads.
        mlp_ratio: Hidden-to-embed ratio in the FFN.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        num_classes: int = len(MATERIAL_CLASS_NAMES),
        class_names: list[str] = MATERIAL_CLASS_NAMES,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(num_classes=num_classes, class_names=class_names)

        self.patch_embed = _PatchEmbed(img_size, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.Sequential(
            *[_TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        b = images.shape[0]
        x = self.patch_embed(images)  # (B, N, D)
        cls = self.cls_token.expand(b, -1, -1)  # (B, 1, D)
        x = torch.cat([cls, x], dim=1)  # (B, N+1, D)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x[:, 0])  # CLS token
        return self.head(x)


# Default alias
MaterialClassifier = CNNMaterialClassifier
