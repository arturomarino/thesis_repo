"""
Utility di inferenza per l'autoencoder probabilistico.

Il modello predice ``mean`` e ``log_variance``. Questo modulo converte tali
parametri in statistiche leggibili e permette di campionare volumi dalla
distribuzione Gaussiana appresa.
"""

import torch


def gaussian_statistics(
    mean: torch.Tensor,
    log_variance: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Restituisce media, log-varianza, varianza e deviazione standard."""

    if mean.shape != log_variance.shape:
        raise ValueError(
            "mean e log_variance devono avere la stessa forma: "
            f"{tuple(mean.shape)} != {tuple(log_variance.shape)}"
        )

    variance = torch.exp(log_variance)
    standard_deviation = torch.exp(0.5 * log_variance)
    return {
        "mean": mean,
        "log_variance": log_variance,
        "variance": variance,
        "standard_deviation": standard_deviation,
    }


def sample_gaussian_prediction(
    mean: torch.Tensor,
    log_variance: torch.Tensor,
    num_samples: int = 1,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Campiona volumi con forma ``[num_samples, *mean.shape]``.

    La trasformazione di riparametrizzazione usata e'
    ``sample = mean + exp(0.5 * log_variance) * epsilon`` con
    ``epsilon ~ N(0, 1)``.
    """

    if num_samples <= 0:
        raise ValueError("num_samples deve essere positivo.")

    statistics = gaussian_statistics(mean, log_variance)
    noise = torch.randn(
        (num_samples, *mean.shape),
        device=mean.device,
        dtype=mean.dtype,
        generator=generator,
    )
    return (
        mean.unsqueeze(0)
        + statistics["standard_deviation"].unsqueeze(0) * noise
    )
