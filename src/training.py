"""
training.py

Primitive di training per l'autoencoder volumetrico.

Responsabilita':
- eseguire un singolo step di ottimizzazione;
- spostare batch e modello sul device scelto;
- restituire metriche minime e leggibili.

Non esegue:
- costruzione di Dataset/DataLoader;
- checkpointing;
- logging persistente;
- loop completo su molte epoche.
"""

from dataclasses import dataclass

import torch
from torch import nn

from dataset import OceanStateSample
from losses import masked_mse_loss


@dataclass(frozen=True)
class AutoencoderTrainMetrics:
    """Metriche prodotte da un singolo training step."""

    loss: float
    valid_points: int
    latent_shape: tuple[int, ...]


def train_autoencoder_step(
    model: nn.Module,
    batch: OceanStateSample,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> AutoencoderTrainMetrics:
    """Esegue forward, masked loss, backward e optimizer step."""

    model.train()

    volume = batch["state"]["volume"].to(device)
    volume_mask = batch["state"]["volume_mask"].to(device)

    optimizer.zero_grad(set_to_none=True)
    output = model(volume)
    reconstruction = output["reconstruction"]
    latent = output["latent"]
    loss = masked_mse_loss(reconstruction, volume, volume_mask)
    loss.backward()
    optimizer.step()

    return AutoencoderTrainMetrics(
        loss=float(loss.detach().cpu()),
        valid_points=int(volume_mask.sum().detach().cpu()),
        latent_shape=tuple(latent.shape),
    )
