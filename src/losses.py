"""
losses.py

Funzioni di loss per l'addestramento dei modelli.

Responsabilita':
- calcolare errori solo sui punti oceanici validi;
- mantenere la loss indipendente dall'architettura del modello.

Non esegue:
- forward pass del modello;
- backward pass;
- aggiornamento dei pesi.
"""

import torch


def masked_mse_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Calcola la mean squared error solo dove ``mask`` e' vera.

    La land-sea mask produce NaN sui punti di terra; nel Dataset quei valori
    sono sostituiti con zero, ma non devono contribuire alla loss.
    """

    if prediction.shape != target.shape:
        raise ValueError(
            "prediction e target devono avere la stessa forma: "
            f"{tuple(prediction.shape)} != {tuple(target.shape)}"
        )

    if mask.shape != target.shape:
        raise ValueError(
            "mask e target devono avere la stessa forma: "
            f"{tuple(mask.shape)} != {tuple(target.shape)}"
        )

    mask = mask.to(device=target.device, dtype=target.dtype)
    squared_error = (prediction - target).pow(2) * mask
    valid_points = mask.sum().clamp_min(eps)
    return squared_error.sum() / valid_points
