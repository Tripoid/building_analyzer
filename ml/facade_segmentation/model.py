"""Facade segmentation model implementations.

Two concrete architectures are provided:

* :class:`DeepLabV3PlusSegmentation` — an encoder-decoder with atrous
  convolutions and ASPP pooling, suitable for high-resolution facade images.
* :class:`UNetSegmentation` — a lightweight skip-connection U-Net useful for
  rapid experimentation and fine-tuning on small datasets.

Both are registered with :class:`~ml.common.registry.ModelRegistry` and share
the same :class:`~ml.common.base_model.BaseSegmentationModel` interface so
they can be swapped transparently.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.common.base_model import BaseSegmentationModel
from ml.common.registry import ModelRegistry
from ml.facade_segmentation.dataset import FACADE_CLASS_NAMES


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class _ConvBnRelu(nn.Sequential):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        dilation: int = 1,
    ) -> None:
        padding = dilation * (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class _ASPPModule(nn.Module):
    """Atrous Spatial Pyramid Pooling module."""

    def __init__(self, in_ch: int = 2048, out_ch: int = 256) -> None:
        super().__init__()
        self.conv1x1 = _ConvBnRelu(in_ch, out_ch, kernel_size=1)
        self.atrous6 = _ConvBnRelu(in_ch, out_ch, dilation=6)
        self.atrous12 = _ConvBnRelu(in_ch, out_ch, dilation=12)
        self.atrous18 = _ConvBnRelu(in_ch, out_ch, dilation=18)
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.project = _ConvBnRelu(out_ch * 5, out_ch, kernel_size=1)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2:]
        gp = F.interpolate(self.global_pool(x), size=(h, w), mode="bilinear", align_corners=False)
        feat = torch.cat(
            [self.conv1x1(x), self.atrous6(x), self.atrous12(x), self.atrous18(x), gp],
            dim=1,
        )
        return self.dropout(self.project(feat))


class _SimpleEncoder(nn.Module):
    """Minimal convolutional encoder (no pretrained weights) for testing."""

    OUT_CHANNELS: int = 256
    LOW_LEVEL_CHANNELS: int = 64

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            _ConvBnRelu(3, 64, 3),
            _ConvBnRelu(64, 64, 3),
            nn.MaxPool2d(2, 2),  # /2
        )
        self.layer1 = nn.Sequential(
            _ConvBnRelu(64, 128, 3),
            _ConvBnRelu(128, 128, 3),
            nn.MaxPool2d(2, 2),  # /4
        )
        self.layer2 = nn.Sequential(
            _ConvBnRelu(128, 256, 3),
            _ConvBnRelu(256, 256, 3),
            nn.MaxPool2d(2, 2),  # /8
        )
        self.layer3 = nn.Sequential(
            _ConvBnRelu(256, 256, 3, dilation=2),
            _ConvBnRelu(256, 256, 3, dilation=4),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        low = self.stem(x)  # /2, 64-ch → low-level features for decoder skip
        x = self.layer1(low)
        x = self.layer2(x)
        x = self.layer3(x)
        return x, low  # high-level, low-level


# ---------------------------------------------------------------------------
# DeepLab V3+-style model
# ---------------------------------------------------------------------------


@ModelRegistry.register("deeplabv3plus", namespace="segmentation")
class DeepLabV3PlusSegmentation(BaseSegmentationModel):
    """DeepLab V3+ style encoder-decoder for facade semantic segmentation.

    The encoder is either a lightweight custom backbone (default, no pretrained
    weights) or any ``torchvision`` ResNet/MobileNet backbone passed via
    *encoder_name*.  The ASPP pooling + skip-connection decoder follows the
    original DeepLab V3+ paper.

    Args:
        num_classes: Number of segmentation classes.
        class_names: Ordered list of class names.
        encoder_name: Backbone identifier.  Currently ``"simple"`` uses the
            built-in lightweight encoder; pass a torchvision backbone name
            (e.g. ``"resnet50"``) to use a pretrained feature extractor.
        pretrained_encoder: Load ImageNet-pretrained encoder weights when using
            a torchvision backbone.
    """

    def __init__(
        self,
        num_classes: int = len(FACADE_CLASS_NAMES),
        class_names: list[str] = FACADE_CLASS_NAMES,
        encoder_name: str = "simple",
        pretrained_encoder: bool = False,
    ) -> None:
        super().__init__(num_classes=num_classes, class_names=class_names)

        self.encoder_name = encoder_name
        self._build_encoder(encoder_name, pretrained_encoder)
        self._build_decoder(num_classes)

    def _build_encoder(self, name: str, pretrained: bool) -> None:
        if name == "simple":
            self.encoder = _SimpleEncoder()
            high_ch = _SimpleEncoder.OUT_CHANNELS
            low_ch = _SimpleEncoder.LOW_LEVEL_CHANNELS
        else:
            # torchvision backbone
            import torchvision.models as tvm

            weights = "DEFAULT" if pretrained else None
            backbone = getattr(tvm, name)(weights=weights)
            # Extract feature layers up to layer3 for high-level and layer1 for low-level
            self.encoder = nn.ModuleDict(
                {
                    "low": nn.Sequential(
                        backbone.conv1,
                        backbone.bn1,
                        backbone.relu,
                        backbone.maxpool,
                        backbone.layer1,
                    ),
                    "high": nn.Sequential(backbone.layer2, backbone.layer3),
                }
            )
            high_ch = 1024  # layer3 output for resnet50
            low_ch = 256  # layer1 output for resnet50

        self._high_ch = high_ch
        self._low_ch = low_ch

    def _build_decoder(self, num_classes: int) -> None:
        self.aspp = _ASPPModule(in_ch=self._high_ch, out_ch=256)
        self.low_proj = _ConvBnRelu(self._low_ch, 48, kernel_size=1)
        self.decoder = nn.Sequential(
            _ConvBnRelu(256 + 48, 256, 3),
            _ConvBnRelu(256, 256, 3),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, 1),
        )

    def _encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.encoder_name == "simple":
            return self.encoder(x)  # type: ignore[return-value]
        # torchvision path
        low = self.encoder["low"](x)  # type: ignore[index]
        high = self.encoder["high"](low)  # type: ignore[index]
        return high, low

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        h, w = images.shape[2:]
        high, low = self._encode(images)

        aspp_out = self.aspp(high)
        aspp_up = F.interpolate(aspp_out, size=low.shape[2:], mode="bilinear", align_corners=False)

        low_feat = self.low_proj(low)
        combined = torch.cat([aspp_up, low_feat], dim=1)
        logits = self.decoder(combined)

        # Upsample back to input resolution
        logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        return logits


@ModelRegistry.register("unet", namespace="segmentation")
class UNetSegmentation(BaseSegmentationModel):
    """Lightweight U-Net for facade semantic segmentation.

    Uses a symmetric encoder-decoder with skip connections.  Suitable for
    rapid prototyping or when GPU memory is limited.

    Args:
        num_classes: Number of segmentation classes.
        class_names: Ordered list of class names.
        base_channels: Number of feature channels in the first encoder block.
            Doubles at each downsampling step.
    """

    def __init__(
        self,
        num_classes: int = len(FACADE_CLASS_NAMES),
        class_names: list[str] = FACADE_CLASS_NAMES,
        base_channels: int = 32,
    ) -> None:
        super().__init__(num_classes=num_classes, class_names=class_names)

        ch = base_channels
        # Encoder
        self.enc1 = nn.Sequential(_ConvBnRelu(3, ch), _ConvBnRelu(ch, ch))
        self.enc2 = nn.Sequential(_ConvBnRelu(ch, ch * 2), _ConvBnRelu(ch * 2, ch * 2))
        self.enc3 = nn.Sequential(_ConvBnRelu(ch * 2, ch * 4), _ConvBnRelu(ch * 4, ch * 4))
        self.enc4 = nn.Sequential(_ConvBnRelu(ch * 4, ch * 8), _ConvBnRelu(ch * 8, ch * 8))

        self.pool = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            _ConvBnRelu(ch * 8, ch * 16), _ConvBnRelu(ch * 16, ch * 16)
        )

        # Decoder
        self.up4 = nn.ConvTranspose2d(ch * 16, ch * 8, 2, stride=2)
        self.dec4 = nn.Sequential(_ConvBnRelu(ch * 16, ch * 8), _ConvBnRelu(ch * 8, ch * 8))

        self.up3 = nn.ConvTranspose2d(ch * 8, ch * 4, 2, stride=2)
        self.dec3 = nn.Sequential(_ConvBnRelu(ch * 8, ch * 4), _ConvBnRelu(ch * 4, ch * 4))

        self.up2 = nn.ConvTranspose2d(ch * 4, ch * 2, 2, stride=2)
        self.dec2 = nn.Sequential(_ConvBnRelu(ch * 4, ch * 2), _ConvBnRelu(ch * 2, ch * 2))

        self.up1 = nn.ConvTranspose2d(ch * 2, ch, 2, stride=2)
        self.dec1 = nn.Sequential(_ConvBnRelu(ch * 2, ch), _ConvBnRelu(ch, ch))

        self.head = nn.Conv2d(ch, num_classes, 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(images)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.head(d1)


# Default alias used by the inferencer when no model name is specified
SegmentationModel = DeepLabV3PlusSegmentation
