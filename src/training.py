"""
training.py

Training del previsore probabilistico volumetrico.

Responsabilita':
- eseguire un singolo step di ottimizzazione;
- eseguire epoche di training e validation;
- calcolare metriche deterministiche e probabilistiche;
- applicare early stopping e salvare il checkpoint migliore.

Non esegue:
- costruzione di Dataset/DataLoader;
- apertura o preprocessing dei dati.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import nn

from dataset import OceanForecastSample
from losses import masked_gaussian_nll_loss, masked_mse_loss

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional UI dependency
    tqdm = None


@dataclass(frozen=True)
class AutoencoderTrainMetrics:
    """Metriche prodotte da un singolo training step."""

    loss: float
    mean_mse: float
    valid_points: int
    latent_shape: tuple[int, ...]


@dataclass(frozen=True)
class ForecastEpochMetrics:
    """Metriche aggregate, pesate per il numero di punti oceanici validi."""

    gaussian_nll: float
    rmse: float
    mean_standard_deviation: float
    coverage_68: float
    coverage_95: float
    valid_points: int


@dataclass(frozen=True)
class ForecastFitResult:
    """Risultato del fitting e posizione del modello migliore."""

    best_epoch: int
    best_validation_nll: float
    epochs_completed: int
    checkpoint_path: Path
    last_checkpoint_path: Path


def train_autoencoder_step(
    model: nn.Module,
    batch: OceanForecastSample,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> AutoencoderTrainMetrics:
    """Esegue un update con Gaussian NLL eteroschedastica mascherata."""

    model.train()

    input_volume = batch["input"]["volume"].to(device)
    target_volume = batch["target"]["volume"].to(device)
    target_mask = batch["target"]["volume_mask"].to(device)

    optimizer.zero_grad(set_to_none=True)
    output = model(input_volume)
    mean = output["mean"]
    log_variance = output["log_variance"]
    latent = output["latent"]
    loss = masked_gaussian_nll_loss(
        mean=mean,
        log_variance=log_variance,
        target=target_volume,
        mask=target_mask,
    )
    mean_mse = masked_mse_loss(mean, target_volume, target_mask)
    loss.backward()
    optimizer.step()

    return AutoencoderTrainMetrics(
        loss=float(loss.detach().cpu()),
        mean_mse=float(mean_mse.detach().cpu()),
        valid_points=int(target_mask.sum().detach().cpu()),
        latent_shape=tuple(latent.shape),
    )


def run_forecast_epoch(
    model: nn.Module,
    batches: Iterable[OceanForecastSample],
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    progress_label: str | None = None,
) -> ForecastEpochMetrics:
    """
    Esegue un'epoca.

    Se ``optimizer`` e' ``None`` opera in validation senza gradienti.
    """

    is_training = optimizer is not None
    model.train(is_training)

    nll_sum = 0.0
    squared_error_sum = 0.0
    standard_deviation_sum = 0.0
    coverage_68_count = 0
    coverage_95_count = 0
    valid_points_total = 0

    iterable = batches
    progress_bar = None
    if progress_label is not None and tqdm is not None:
        progress_bar = tqdm(
            batches,
            desc=progress_label,
            total=len(batches) if hasattr(batches, "__len__") else None,
            leave=False,
            dynamic_ncols=True,
        )
        iterable = progress_bar

    for batch in iterable:
        input_volume = batch["input"]["volume"].to(
            device,
            non_blocking=True,
        )
        target_volume = batch["target"]["volume"].to(
            device,
            non_blocking=True,
        )
        target_mask = batch["target"]["volume_mask"].to(
            device,
            non_blocking=True,
        )

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            output = model(input_volume)
            mean = output["mean"]
            log_variance = output["log_variance"]
            loss = masked_gaussian_nll_loss(
                mean=mean,
                log_variance=log_variance,
                target=target_volume,
                mask=target_mask,
            )

            if is_training:
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            valid_mask = target_mask.to(dtype=torch.bool)
            valid_points = int(valid_mask.sum().item())
            if valid_points == 0:
                continue

            valid_error = (target_volume - mean)[valid_mask]
            valid_standard_deviation = torch.exp(
                0.5 * log_variance[valid_mask]
            )
            absolute_error = valid_error.abs()

            nll_sum += float(loss.detach().item()) * valid_points
            squared_error_sum += float(valid_error.pow(2).sum().item())
            standard_deviation_sum += float(
                valid_standard_deviation.sum().item()
            )
            coverage_68_count += int(
                (absolute_error <= valid_standard_deviation).sum().item()
            )
            coverage_95_count += int(
                (
                    absolute_error
                    <= 1.96 * valid_standard_deviation
                ).sum().item()
            )
            valid_points_total += valid_points
            if progress_bar is not None:
                progress_bar.set_postfix(
                    nll=f"{nll_sum / valid_points_total:.4f}",
                    rmse=(
                        f"{(squared_error_sum / valid_points_total) ** 0.5:.4f}"
                    ),
                )

    if valid_points_total == 0:
        raise ValueError("L'epoca non contiene punti oceanici validi.")

    return ForecastEpochMetrics(
        gaussian_nll=nll_sum / valid_points_total,
        rmse=(squared_error_sum / valid_points_total) ** 0.5,
        mean_standard_deviation=(
            standard_deviation_sum / valid_points_total
        ),
        coverage_68=coverage_68_count / valid_points_total,
        coverage_95=coverage_95_count / valid_points_total,
        valid_points=valid_points_total,
    )


def fit_forecaster(
    model: nn.Module,
    train_batches: Iterable[OceanForecastSample],
    validation_batches: Iterable[OceanForecastSample],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    patience: int,
    checkpoint_path: Path,
    resume_checkpoint: dict[str, object] | None = None,
    show_progress: bool = True,
) -> ForecastFitResult:
    """Addestra con early stopping e ripresa da un checkpoint opzionale."""

    if epochs <= 0:
        raise ValueError("epochs deve essere positivo.")
    if patience <= 0:
        raise ValueError("patience deve essere positivo.")

    checkpoint_path = Path(checkpoint_path)
    last_checkpoint_path = _last_checkpoint_path(checkpoint_path)
    start_epoch = 1
    best_validation_nll = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, object]] = []

    if resume_checkpoint is not None:
        start_epoch = int(resume_checkpoint["epoch"]) + 1
        best_validation_nll = float(
            resume_checkpoint.get("best_validation_nll", float("inf"))
        )
        best_epoch = int(resume_checkpoint.get("best_epoch", 0))
        epochs_without_improvement = int(
            resume_checkpoint.get("epochs_without_improvement", 0)
        )
        history = list(resume_checkpoint.get("history", []))

    if start_epoch > epochs:
        raise ValueError(
            "Il checkpoint ha gia' raggiunto il numero massimo di epoche."
        )

    for epoch in range(start_epoch, epochs + 1):
        train_metrics = run_forecast_epoch(
            model=model,
            batches=train_batches,
            device=device,
            optimizer=optimizer,
            progress_label=(
                f"Train epoch {epoch}/{epochs}" if show_progress else None
            ),
        )
        validation_metrics = run_forecast_epoch(
            model=model,
            batches=validation_batches,
            device=device,
            progress_label=(
                f"Validation epoch {epoch}/{epochs}"
                if show_progress
                else None
            ),
        )
        history.append(
            {
                "epoch": epoch,
                "train": asdict(train_metrics),
                "validation": asdict(validation_metrics),
            }
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train NLL {train_metrics.gaussian_nll:.6f} | "
            f"val NLL {validation_metrics.gaussian_nll:.6f} | "
            f"val RMSE {validation_metrics.rmse:.6f} | "
            f"coverage 68/95 "
            f"{validation_metrics.coverage_68:.3f}/"
            f"{validation_metrics.coverage_95:.3f}"
        )

        if validation_metrics.gaussian_nll < best_validation_nll:
            best_validation_nll = validation_metrics.gaussian_nll
            best_epoch = epoch
            epochs_without_improvement = 0
            _save_checkpoint(
                path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                validation_metrics=validation_metrics,
                history=history,
                best_validation_nll=best_validation_nll,
                best_epoch=best_epoch,
                epochs_without_improvement=epochs_without_improvement,
                checkpoint_kind="best",
            )
        else:
            epochs_without_improvement += 1

        _save_checkpoint(
            path=last_checkpoint_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            validation_metrics=validation_metrics,
            history=history,
            best_validation_nll=best_validation_nll,
            best_epoch=best_epoch,
            epochs_without_improvement=epochs_without_improvement,
            checkpoint_kind="last",
        )

        if epochs_without_improvement >= patience:
            print(f"Early stopping dopo {epoch} epoche.")
            break

    return ForecastFitResult(
        best_epoch=best_epoch,
        best_validation_nll=best_validation_nll,
        epochs_completed=epoch,
        checkpoint_path=checkpoint_path,
        last_checkpoint_path=last_checkpoint_path,
    )


def load_forecaster_checkpoint(
    path: Path,
    model: nn.Module,
    device: torch.device,
) -> dict[str, object]:
    """Carica nel modello un checkpoint precedentemente salvato."""

    checkpoint = read_forecaster_checkpoint(path, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def read_forecaster_checkpoint(
    path: Path,
    device: torch.device,
) -> dict[str, object]:
    """Legge una sola volta dati, configurazione e pesi del checkpoint."""

    return torch.load(path, map_location=device, weights_only=False)


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    validation_metrics: ForecastEpochMetrics,
    history: list[dict[str, object]],
    best_validation_nll: float,
    best_epoch: int,
    epochs_without_improvement: int,
    checkpoint_kind: str,
) -> None:
    """Scrive atomicamente un checkpoint migliore o di ripresa."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    model_config = getattr(model, "config", None)
    serialized_config = (
        asdict(model_config) if model_config is not None else None
    )
    torch.save(
        {
            "epoch": epoch,
            "model_config": serialized_config,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation_metrics": asdict(validation_metrics),
            "history": history,
            "best_validation_nll": best_validation_nll,
            "best_epoch": best_epoch,
            "epochs_without_improvement": epochs_without_improvement,
            "checkpoint_kind": checkpoint_kind,
        },
        temporary_path,
    )
    temporary_path.replace(path)


def _last_checkpoint_path(checkpoint_path: Path) -> Path:
    """Restituisce il percorso persistente dello stato dell'ultima epoca."""

    return checkpoint_path.with_name(
        f"{checkpoint_path.stem}_last{checkpoint_path.suffix}"
    )
