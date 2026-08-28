"""
unet.py

Blocchi U-Net 3D riutilizzabili per il backbone volumetrico.

Responsabilita':
- definire blocchi convoluzionali 3D;
- definire downsampling e upsampling con skip connection;
- preservare compatibilita' con forme spaziali non potenze di due.

Non esegue:
- training;
- calcolo della loss;
- gestione di variabili 2D di superficie.
"""

import torch
from torch import nn
import torch.nn.functional as F


def _normalization_layer(kind: str, channels: int) -> nn.Module:
    if kind == "none":
        return nn.Identity()
    if kind == "instance":
        return nn.InstanceNorm3d(channels)
    raise ValueError(f"Normalizzazione 3D non supportata: {kind}.")


class ConvBlock3D(nn.Module):
    """Due convoluzioni 3D con normalizzazione e attivazione."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        normalization: str = "instance",
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            _normalization_layer(normalization, out_channels),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            _normalization_layer(normalization, out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock3D(nn.Module):
    """Riduce la risoluzione e aumenta la capacita' rappresentativa."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        normalization: str = "instance",
    ) -> None:
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv = ConvBlock3D(
            in_channels,
            out_channels,
            normalization=normalization,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock3D(nn.Module):
    """
    Aumenta la risoluzione e concatena la skip connection.

    Usiamo interpolate verso la forma esatta della skip connection per gestire
    dimensioni oceanografiche non divisibili perfettamente per due, ad esempio
    46x65x171.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        normalization: str = "instance",
    ) -> None:
        super().__init__()
        self.conv = ConvBlock3D(
            in_channels + skip_channels,
            out_channels,
            normalization=normalization,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            x,
            size=skip.shape[2:],
            mode="trilinear",
            align_corners=False,
        )
        return self.conv(torch.cat([x, skip], dim=1))
