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


class ConvBlock3D(nn.Module):
    """Due convoluzioni 3D con normalizzazione e attivazione."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock3D(nn.Module):
    """Riduce la risoluzione e aumenta la capacita' rappresentativa."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv = ConvBlock3D(in_channels, out_channels)

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
    ) -> None:
        super().__init__()
        self.conv = ConvBlock3D(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            x,
            size=skip.shape[2:],
            mode="trilinear",
            align_corners=False,
        )
        return self.conv(torch.cat([x, skip], dim=1))
