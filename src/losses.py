"""
losses.py

Funzioni di loss per l'addestramento dei modelli.

Responsabilita':
- calcolare errori solo sui punti oceanici validi;
- supportare ricostruzioni deterministiche e probabilistiche;
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


def masked_gaussian_nll_loss(
    mean: torch.Tensor,
    log_variance: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    r"""
    Gaussian negative log-likelihood sui soli punti validi.

    Il modello predice due tensori:

    - ``mean``: :math:`\mu`, valore medio della distribuzione;
    - ``log_variance``: :math:`s = \log(\sigma^2)`.

    A meno della costante additiva ``0.5 * log(2*pi)``, che non influenza
    l'ottimizzazione, la loss media e':

    .. math::

        \frac{1}{N}\sum_i
        \left[\frac{(y_i-\mu_i)^2}{2\exp(s_i)}+\frac{s_i}{2}\right].
    """

    for name, tensor in (
        ("mean", mean),
        ("log_variance", log_variance),
        ("mask", mask),
    ):
        if tensor.shape != target.shape:
            raise ValueError(
                f"{name} e target devono avere la stessa forma: "
                f"{tuple(tensor.shape)} != {tuple(target.shape)}"
            )

    valid_mask = mask.to(device=target.device, dtype=torch.bool)
    # Sostituiamo i valori mascherati prima delle operazioni non lineari:
    # moltiplicare semplicemente per zero dopo exp() potrebbe lasciare gradienti
    # NaN quando un punto non valido contiene valori estremi.
    safe_mean = torch.where(valid_mask, mean, torch.zeros_like(mean))
    safe_log_variance = torch.where(
        valid_mask,
        log_variance,
        torch.zeros_like(log_variance),
    )
    safe_target = torch.where(valid_mask, target, torch.zeros_like(target))
    squared_error = (safe_target - safe_mean).pow(2)

    # exp(-s) equivale a 1 / exp(s), ma evita una divisione esplicita.
    pointwise_nll = 0.5 * (
        squared_error * torch.exp(-safe_log_variance) + safe_log_variance
    )
    valid_points = valid_mask.sum().to(dtype=target.dtype).clamp_min(eps)
    return pointwise_nll.sum() / valid_points
