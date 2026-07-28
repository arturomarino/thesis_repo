"""
autoencoder.py

Autoencoder volumetrico basato su backbone U-Net 3D.

Responsabilita':
- comprimere le variabili oceanografiche 3D in un latent space;
- predire media e varianza del volume 3D originale;
- esporre metodi encode/decode utili alla futura latent diffusion.

Non esegue:
- loss masking;
- training loop;
- modellazione della variabile 2D di superficie ``zos_cglo``.
"""

from dataclasses import dataclass

import torch
from torch import nn

from models.unet import ConvBlock3D, DownBlock3D, UpBlock3D


@dataclass(frozen=True)
class VolumeAutoencoderConfig:
    """
    Configurazione dell'autoencoder volumetrico.

    input_channels=4 corrisponde a thetao, so, uo e vo. La variabile zos resta
    fuori da questo primo modello per non trasformare un campo 2D in un volume
    3D artificiale.
    """

    input_channels: int = 4
    output_channels: int = 4
    base_channels: int = 8
    latent_channels: int = 32

    def __post_init__(self) -> None:
        for name, value in (
            ("input_channels", self.input_channels),
            ("output_channels", self.output_channels),
            ("base_channels", self.base_channels),
            ("latent_channels", self.latent_channels),
        ):
            if value <= 0:
                raise ValueError(f"{name} deve essere positivo.")


class VolumeUNetAutoencoder(nn.Module):
    """
    Autoencoder U-Net per tensori ``[batch, channels, depth, height, width]``.

    La parte encoder produce il latent space che, nella fase successiva della
    tesi, diventera' il dominio su cui addestrare il diffusion model.
    """

    def __init__(self, config: VolumeAutoencoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or VolumeAutoencoderConfig()

        base = self.config.base_channels
        latent = self.config.latent_channels

        self.encoder_1 = ConvBlock3D(self.config.input_channels, base)
        self.encoder_2 = DownBlock3D(base, base * 2)
        self.encoder_3 = DownBlock3D(base * 2, latent)

        self.decoder_2 = UpBlock3D(
            in_channels=latent,
            skip_channels=base * 2,
            out_channels=base * 2,
        )
        self.decoder_1 = UpBlock3D(
            in_channels=base * 2,
            skip_channels=base,
            out_channels=base,
        )
        self.mean_projection = nn.Conv3d(
            base,
            self.config.output_channels,
            kernel_size=1,
        )
        self.log_variance_projection = nn.Conv3d(
            base,
            self.config.output_channels,
            kernel_size=1,
        )

        # sigma^2 = 1 e' un punto di partenza neutro e stabile.
        nn.init.zeros_(self.log_variance_projection.weight)
        nn.init.zeros_(self.log_variance_projection.bias)

    def encode(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Restituisce latent space e skip connection."""

        skip_1 = self.encoder_1(x)
        skip_2 = self.encoder_2(skip_1)
        latent = self.encoder_3(skip_2)
        return latent, (skip_1, skip_2)

    def decode(
        self,
        latent: torch.Tensor,
        skips: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predice ``mu`` e ``log(sigma^2)`` a partire dal latent space."""

        skip_1, skip_2 = skips
        x = self.decoder_2(latent, skip_2)
        x = self.decoder_1(x, skip_1)
        mean = self.mean_projection(x)
        log_variance = self.log_variance_projection(x)
        return mean, log_variance

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Restituisce i parametri della distribuzione e il latent space."""

        latent, skips = self.encode(x)
        mean, log_variance = self.decode(latent, skips)
        return {
            "mean": mean,
            "log_variance": log_variance,
            "latent": latent,
        }
