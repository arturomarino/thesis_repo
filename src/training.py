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
from typing import Iterable, Mapping

import torch
from torch import nn

from dataset import OceanForecastSample
from losses import masked_gaussian_nll_loss, masked_mse_loss
from visualization import (
    plot_learning_curve_snapshot,
    plot_learning_curve_snapshots,
)

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
    objective: float
    rmse: float
    mae: float
    mean_standard_deviation: float
    coverage_68: float
    coverage_95: float
    valid_points: int


@dataclass(frozen=True)
class PersistenceBaselineMetrics:
    """Metriche della previsione banale ``x(t + 1) = x(t)``."""

    rmse: float
    mae: float
    valid_points: int


@dataclass(frozen=True)
class TemperatureComparisonMetrics:
    """Confronto in gradi Celsius fra modello e persistence."""

    model_rmse_c: float
    model_mae_c: float
    model_bias_c: float
    model_mean_standard_deviation_c: float
    model_coverage_68: float
    model_coverage_95: float
    persistence_rmse_c: float
    persistence_mae_c: float
    persistence_bias_c: float
    rmse_skill_score: float
    valid_points: int


@dataclass(frozen=True)
class TemperatureEvaluationResult:
    """Metriche globali e a una profondita' selezionata."""

    all_depths: TemperatureComparisonMetrics
    selected_depth: TemperatureComparisonMetrics
    selected_depth_m: float


@dataclass(frozen=True)
class PhysicalComparisonMetrics:
    """Metriche in unita' fisiche per una variabile volumetrica."""

    model_rmse: float
    model_mae: float
    model_bias: float
    model_mean_standard_deviation: float
    model_coverage_68: float
    model_coverage_95: float
    persistence_rmse: float
    persistence_mae: float
    persistence_bias: float
    rmse_skill_score: float
    valid_points: int


@dataclass(frozen=True)
class VariablePhysicalEvaluation:
    """Metriche globali e superficiali di una singola variabile."""

    all_depths: PhysicalComparisonMetrics
    selected_depth: PhysicalComparisonMetrics
    unit: str


@dataclass(frozen=True)
class PhysicalEvaluationResult:
    """Valutazione fisica multivariata prodotta con un solo passaggio."""

    variables: dict[str, VariablePhysicalEvaluation]
    selected_depth_m: float
    forecast_count: int


@dataclass(frozen=True)
class AnnualErrorMapResult:
    """Statistiche annuali dell'errore superficiale per canale."""

    mean_absolute_error: torch.Tensor
    persistence_mean_absolute_error: torch.Tensor
    mae_difference_model_minus_persistence: torch.Tensor
    error_standard_deviation: torch.Tensor
    valid_counts: torch.Tensor
    selected_depth_m: float
    forecast_count: int


@dataclass(frozen=True)
class ForecastFitResult:
    """Risultato del fitting e posizione del modello migliore."""

    best_epoch: int
    best_validation_nll: float
    best_validation_rmse: float
    train_persistence_rmse: float
    validation_persistence_rmse: float
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
    mean_mse_weight: float = 0.0,
    gradient_clip_norm: float | None = None,
    temperature_mse_weight: float = 1.0,
) -> ForecastEpochMetrics:
    """
    Esegue un'epoca.

    Se ``optimizer`` e' ``None`` opera in validation senza gradienti.
    """

    is_training = optimizer is not None
    if mean_mse_weight < 0:
        raise ValueError("mean_mse_weight non puo' essere negativo.")
    if temperature_mse_weight <= 0:
        raise ValueError("temperature_mse_weight deve essere positivo.")
    model.train(is_training)

    nll_sum = 0.0
    objective_sum = 0.0
    squared_error_sum = 0.0
    absolute_error_sum = 0.0
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
            nll = masked_gaussian_nll_loss(
                mean=mean,
                log_variance=log_variance,
                target=target_volume,
                mask=target_mask,
            )
            mean_mse = _weighted_channel_mse_loss(
                mean,
                target_volume,
                target_mask,
                temperature_weight=temperature_mse_weight,
            )
            objective = nll + mean_mse_weight * mean_mse

            if is_training:
                objective.backward()
                if gradient_clip_norm is not None:
                    if gradient_clip_norm <= 0:
                        raise ValueError(
                            "gradient_clip_norm deve essere positivo."
                        )
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=gradient_clip_norm,
                    )
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

            nll_sum += float(nll.detach().item()) * valid_points
            objective_sum += float(objective.detach().item()) * valid_points
            squared_error_sum += float(valid_error.pow(2).sum().item())
            absolute_error_sum += float(absolute_error.sum().item())
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
        objective=objective_sum / valid_points_total,
        rmse=(squared_error_sum / valid_points_total) ** 0.5,
        mae=absolute_error_sum / valid_points_total,
        mean_standard_deviation=(
            standard_deviation_sum / valid_points_total
        ),
        coverage_68=coverage_68_count / valid_points_total,
        coverage_95=coverage_95_count / valid_points_total,
        valid_points=valid_points_total,
    )


def run_persistence_baseline(
    batches: Iterable[OceanForecastSample],
    device: torch.device,
    progress_label: str | None = None,
) -> PersistenceBaselineMetrics:
    """Valuta la baseline che usa lo stato odierno come previsione di domani.

    Il calcolo usa gli stessi tensori normalizzati e la stessa target mask
    impiegati dal modello, in modo che RMSE e MAE siano confrontabili.
    """

    squared_error_sum = 0.0
    absolute_error_sum = 0.0
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

    with torch.no_grad():
        for batch in iterable:
            input_volume = batch["input"]["volume"].to(
                device,
                non_blocking=True,
            )
            target = batch["target"]["volume"].to(
                device,
                non_blocking=True,
            )
            valid_mask = batch["target"]["volume_mask"].to(
                device,
                non_blocking=True,
                dtype=torch.bool,
            )

            if input_volume.ndim != target.ndim:
                raise ValueError(
                    "Input e target della persistence devono avere lo "
                    "stesso numero di dimensioni."
                )
            if input_volume.shape[1] < target.shape[1]:
                raise ValueError("L'input non contiene l'ultimo stato completo.")
            prediction = input_volume[:, -target.shape[1] :]
            if prediction.shape != target.shape:
                raise ValueError("Forma dell'ultimo stato di input inattesa.")
            if valid_mask.shape != target.shape:
                raise ValueError(
                    "Target mask e target della persistence devono avere "
                    "la stessa forma."
                )

            valid_points = int(valid_mask.sum().item())
            if valid_points == 0:
                continue

            valid_error = (target - prediction)[valid_mask]
            squared_error_sum += float(valid_error.pow(2).sum().item())
            absolute_error_sum += float(valid_error.abs().sum().item())
            valid_points_total += valid_points
            if progress_bar is not None:
                progress_bar.set_postfix(
                    rmse=(
                        f"{(squared_error_sum / valid_points_total) ** 0.5:.4f}"
                    ),
                )

    if valid_points_total == 0:
        raise ValueError("La baseline non contiene punti oceanici validi.")

    return PersistenceBaselineMetrics(
        rmse=(squared_error_sum / valid_points_total) ** 0.5,
        mae=absolute_error_sum / valid_points_total,
        valid_points=valid_points_total,
    )


def _weighted_channel_mse_loss(
    mean: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    temperature_weight: float,
) -> torch.Tensor:
    """MSE mascherato con peso opzionale maggiore sul canale temperatura."""

    if temperature_weight == 1.0:
        return masked_mse_loss(mean, target, mask)
    if mean.ndim < 2:
        raise ValueError("Il tensore previsto deve avere una dimensione canale.")

    weights = torch.ones(
        mean.shape[1],
        dtype=mean.dtype,
        device=mean.device,
    )
    weights[0] = temperature_weight
    broadcast_shape = [1] * mean.ndim
    broadcast_shape[1] = mean.shape[1]
    weights = weights.view(*broadcast_shape)

    valid_mask = mask.to(dtype=mean.dtype)
    weighted_mask = valid_mask * weights
    valid_points = weighted_mask.sum()
    if valid_points <= 0:
        raise ValueError("La loss non contiene punti oceanici validi.")
    return ((mean - target).pow(2) * weighted_mask).sum() / valid_points


def rmse_skill_score(model_rmse: float, persistence_rmse: float) -> float:
    """Restituisce ``1 - MSE_model / MSE_persistence``."""

    if model_rmse < 0:
        raise ValueError("model_rmse non puo' essere negativo.")
    if persistence_rmse <= 0:
        raise ValueError("persistence_rmse deve essere positivo.")
    return 1.0 - (model_rmse / persistence_rmse) ** 2


def run_physical_evaluation(
    model: nn.Module,
    batches: Iterable[OceanForecastSample],
    device: torch.device,
    variable_names: tuple[str, ...],
    standard_deviations: torch.Tensor,
    depth_values_m: tuple[float, ...],
    units: Mapping[str, str],
    requested_depth_m: float = 0.5,
    progress_label: str | None = None,
) -> PhysicalEvaluationResult:
    """Valuta tutti i canali in unita' fisiche con un solo forward per batch."""

    if not variable_names:
        raise ValueError("Specificare almeno una variabile.")
    if standard_deviations.ndim != 2:
        raise ValueError("Gli standard devono avere forma [canale, profondita].")
    if standard_deviations.shape != (len(variable_names), len(depth_values_m)):
        raise ValueError("Forma degli standard non compatibile con variabili e profondita'.")
    if not depth_values_m:
        raise ValueError("Il dataset non contiene profondita'.")
    physical_std = standard_deviations.to(device=device, dtype=torch.float32)
    if not torch.isfinite(physical_std).all() or bool((physical_std <= 0).any()):
        raise ValueError("Gli standard fisici devono essere finiti e positivi.")

    selected_depth_index = min(
        range(len(depth_values_m)),
        key=lambda index: abs(depth_values_m[index] - requested_depth_m),
    )
    accumulators = {
        name: {
            "all_depths": _new_physical_accumulator(),
            "selected_depth": _new_physical_accumulator(),
        }
        for name in variable_names
    }
    model.eval()
    forecast_count = 0

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

    with torch.inference_mode():
        for batch in iterable:
            input_volume = batch["input"]["volume"].to(device, non_blocking=True)
            target = batch["target"]["volume"].to(device, non_blocking=True)
            mask = batch["target"]["volume_mask"].to(
                device,
                non_blocking=True,
                dtype=torch.bool,
            )
            if target.ndim != 5 or target.shape[1] != len(variable_names):
                raise ValueError("Numero di canali target inatteso.")
            if target.shape[2] != len(depth_values_m) or mask.shape != target.shape:
                raise ValueError("Forma di target, mask o profondita' inattesa.")
            if input_volume.shape[1] < target.shape[1]:
                raise ValueError("L'input non contiene l'ultimo stato completo.")

            output = model(input_volume)
            mean = output["mean"]
            log_variance = output["log_variance"]
            if mean.shape != target.shape or log_variance.shape != target.shape:
                raise ValueError("Forma dell'output probabilistico inattesa.")
            latest_input = input_volume[:, -target.shape[1] :]
            forecast_count += int(target.shape[0])

            for channel, name in enumerate(variable_names):
                scale = physical_std[channel].view(1, -1, 1, 1)
                model_error = (mean[:, channel] - target[:, channel]) * scale
                persistence_error = (
                    latest_input[:, channel] - target[:, channel]
                ) * scale
                predicted_std = torch.exp(0.5 * log_variance[:, channel]) * scale
                valid = mask[:, channel]
                _accumulate_physical_metrics(
                    accumulators[name]["all_depths"],
                    model_error,
                    persistence_error,
                    predicted_std,
                    valid,
                )
                _accumulate_physical_metrics(
                    accumulators[name]["selected_depth"],
                    model_error[:, selected_depth_index],
                    persistence_error[:, selected_depth_index],
                    predicted_std[:, selected_depth_index],
                    valid[:, selected_depth_index],
                )

    if forecast_count == 0:
        raise ValueError("La valutazione non contiene previsioni.")
    results = {
        name: VariablePhysicalEvaluation(
            all_depths=_finalize_physical_metrics(
                accumulators[name]["all_depths"]
            ),
            selected_depth=_finalize_physical_metrics(
                accumulators[name]["selected_depth"]
            ),
            unit=str(units.get(name, "unknown")),
        )
        for name in variable_names
    }
    return PhysicalEvaluationResult(
        variables=results,
        selected_depth_m=float(depth_values_m[selected_depth_index]),
        forecast_count=forecast_count,
    )


def run_annual_error_map_evaluation(
    model: nn.Module,
    batches: Iterable[OceanForecastSample],
    device: torch.device,
    standard_deviations: torch.Tensor,
    depth_values_m: tuple[float, ...],
    requested_depth_m: float = 0.5,
    progress_label: str | None = None,
) -> AnnualErrorMapResult:
    """Calcola statistiche annuali dell'errore al livello richiesto."""

    if standard_deviations.ndim != 2:
        raise ValueError("Gli standard devono avere forma [canale, profondita].")
    if standard_deviations.shape[1] != len(depth_values_m):
        raise ValueError("Profondita' degli standard inattese.")
    selected_depth_index = min(
        range(len(depth_values_m)),
        key=lambda index: abs(depth_values_m[index] - requested_depth_m),
    )
    physical_std = standard_deviations.to(device=device, dtype=torch.float32)
    if not torch.isfinite(physical_std).all() or bool((physical_std <= 0).any()):
        raise ValueError("Gli standard fisici devono essere finiti e positivi.")

    absolute_error_sum: torch.Tensor | None = None
    persistence_absolute_error_sum: torch.Tensor | None = None
    error_sum: torch.Tensor | None = None
    squared_error_sum: torch.Tensor | None = None
    valid_counts: torch.Tensor | None = None
    forecast_count = 0
    model.eval()
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

    with torch.inference_mode():
        for batch in iterable:
            input_volume = batch["input"]["volume"].to(device, non_blocking=True)
            target = batch["target"]["volume"].to(device, non_blocking=True)
            mask = batch["target"]["volume_mask"].to(
                device,
                non_blocking=True,
                dtype=torch.bool,
            )
            if target.ndim != 5 or mask.shape != target.shape:
                raise ValueError("Forma del target annuale inattesa.")
            if standard_deviations.shape[0] != target.shape[1]:
                raise ValueError("Numero di canali degli standard inatteso.")
            output = model(input_volume)
            mean = output["mean"]
            if mean.shape != target.shape:
                raise ValueError("Forma della media prevista inattesa.")

            scale = physical_std[:, selected_depth_index].view(1, -1, 1, 1)
            physical_error = (
                mean[:, :, selected_depth_index]
                - target[:, :, selected_depth_index]
            ) * scale
            persistence_prediction = input_volume[:, -target.shape[1] :]
            if persistence_prediction.shape != target.shape:
                raise ValueError("Forma dell'ultimo stato di input inattesa.")
            persistence_physical_error = (
                persistence_prediction[:, :, selected_depth_index]
                - target[:, :, selected_depth_index]
            ) * scale
            absolute_error = physical_error.abs()
            persistence_absolute_error = persistence_physical_error.abs()
            valid = mask[:, :, selected_depth_index]
            batch_absolute_sum = torch.where(
                valid, absolute_error, 0.0
            ).sum(dim=0)
            batch_error_sum = torch.where(
                valid, physical_error, 0.0
            ).sum(dim=0)
            batch_persistence_absolute_sum = torch.where(
                valid, persistence_absolute_error, 0.0
            ).sum(dim=0)
            batch_squared_error_sum = torch.where(
                valid, physical_error.square(), 0.0
            ).sum(dim=0)
            batch_counts = valid.sum(dim=0, dtype=torch.int64)
            if absolute_error_sum is None:
                absolute_error_sum = torch.zeros_like(
                    batch_absolute_sum, dtype=torch.float64
                )
                persistence_absolute_error_sum = torch.zeros_like(
                    batch_persistence_absolute_sum, dtype=torch.float64
                )
                error_sum = torch.zeros_like(batch_error_sum, dtype=torch.float64)
                squared_error_sum = torch.zeros_like(
                    batch_squared_error_sum, dtype=torch.float64
                )
                valid_counts = torch.zeros_like(batch_counts, dtype=torch.int64)
            absolute_error_sum += batch_absolute_sum.to(dtype=torch.float64)
            persistence_absolute_error_sum += batch_persistence_absolute_sum.to(
                dtype=torch.float64
            )
            error_sum += batch_error_sum.to(dtype=torch.float64)
            squared_error_sum += batch_squared_error_sum.to(dtype=torch.float64)
            valid_counts += batch_counts
            forecast_count += int(target.shape[0])

    if (
        absolute_error_sum is None
        or persistence_absolute_error_sum is None
        or error_sum is None
        or squared_error_sum is None
        or valid_counts is None
        or forecast_count == 0
    ):
        raise ValueError("La valutazione annuale non contiene previsioni.")
    mean_absolute_error = torch.full_like(absolute_error_sum, float("nan"))
    persistence_mean_absolute_error = torch.full_like(
        persistence_absolute_error_sum, float("nan")
    )
    mae_difference_model_minus_persistence = torch.full_like(
        absolute_error_sum, float("nan")
    )
    error_standard_deviation = torch.full_like(error_sum, float("nan"))
    valid_cells = valid_counts > 0
    counts = valid_counts[valid_cells].to(dtype=torch.float64)
    mean_absolute_error[valid_cells] = (
        absolute_error_sum[valid_cells] / counts
    )
    persistence_mean_absolute_error[valid_cells] = (
        persistence_absolute_error_sum[valid_cells] / counts
    )
    mae_difference_model_minus_persistence[valid_cells] = (
        mean_absolute_error[valid_cells]
        - persistence_mean_absolute_error[valid_cells]
    )
    mean_error = error_sum[valid_cells] / counts
    error_standard_deviation[valid_cells] = torch.sqrt(
        torch.clamp(
            squared_error_sum[valid_cells] / counts - mean_error.square(),
            min=0.0,
        )
    )
    return AnnualErrorMapResult(
        mean_absolute_error=mean_absolute_error.cpu(),
        persistence_mean_absolute_error=persistence_mean_absolute_error.cpu(),
        mae_difference_model_minus_persistence=(
            mae_difference_model_minus_persistence.cpu()
        ),
        error_standard_deviation=error_standard_deviation.cpu(),
        valid_counts=valid_counts.cpu(),
        selected_depth_m=float(depth_values_m[selected_depth_index]),
        forecast_count=forecast_count,
    )


def _new_physical_accumulator() -> dict[str, float | int]:
    return {
        "model_squared_error_sum": 0.0,
        "model_absolute_error_sum": 0.0,
        "model_error_sum": 0.0,
        "model_standard_deviation_sum": 0.0,
        "model_coverage_68_count": 0,
        "model_coverage_95_count": 0,
        "persistence_squared_error_sum": 0.0,
        "persistence_absolute_error_sum": 0.0,
        "persistence_error_sum": 0.0,
        "valid_points": 0,
    }


def _accumulate_physical_metrics(
    accumulator: dict[str, float | int],
    model_error: torch.Tensor,
    persistence_error: torch.Tensor,
    predicted_std: torch.Tensor,
    valid_mask: torch.Tensor,
) -> None:
    valid_points = int(valid_mask.sum().item())
    if valid_points == 0:
        return
    selected_model_error = model_error[valid_mask]
    selected_persistence_error = persistence_error[valid_mask]
    selected_std = predicted_std[valid_mask]
    absolute_error = selected_model_error.abs()
    accumulator["model_squared_error_sum"] += float(
        selected_model_error.pow(2).sum().item()
    )
    accumulator["model_absolute_error_sum"] += float(absolute_error.sum().item())
    accumulator["model_error_sum"] += float(selected_model_error.sum().item())
    accumulator["model_standard_deviation_sum"] += float(selected_std.sum().item())
    accumulator["model_coverage_68_count"] += int(
        (absolute_error <= selected_std).sum().item()
    )
    accumulator["model_coverage_95_count"] += int(
        (absolute_error <= 1.96 * selected_std).sum().item()
    )
    accumulator["persistence_squared_error_sum"] += float(
        selected_persistence_error.pow(2).sum().item()
    )
    accumulator["persistence_absolute_error_sum"] += float(
        selected_persistence_error.abs().sum().item()
    )
    accumulator["persistence_error_sum"] += float(
        selected_persistence_error.sum().item()
    )
    accumulator["valid_points"] += valid_points


def _finalize_physical_metrics(
    accumulator: Mapping[str, float | int],
) -> PhysicalComparisonMetrics:
    valid_points = int(accumulator["valid_points"])
    if valid_points == 0:
        raise ValueError("La valutazione non contiene punti validi.")
    model_rmse = (
        float(accumulator["model_squared_error_sum"]) / valid_points
    ) ** 0.5
    persistence_rmse = (
        float(accumulator["persistence_squared_error_sum"]) / valid_points
    ) ** 0.5
    return PhysicalComparisonMetrics(
        model_rmse=model_rmse,
        model_mae=float(accumulator["model_absolute_error_sum"]) / valid_points,
        model_bias=float(accumulator["model_error_sum"]) / valid_points,
        model_mean_standard_deviation=(
            float(accumulator["model_standard_deviation_sum"]) / valid_points
        ),
        model_coverage_68=(
            int(accumulator["model_coverage_68_count"]) / valid_points
        ),
        model_coverage_95=(
            int(accumulator["model_coverage_95_count"]) / valid_points
        ),
        persistence_rmse=persistence_rmse,
        persistence_mae=(
            float(accumulator["persistence_absolute_error_sum"]) / valid_points
        ),
        persistence_bias=(
            float(accumulator["persistence_error_sum"]) / valid_points
        ),
        rmse_skill_score=rmse_skill_score(model_rmse, persistence_rmse),
        valid_points=valid_points,
    )


def run_temperature_evaluation(
    model: nn.Module,
    batches: Iterable[OceanForecastSample],
    device: torch.device,
    temperature_std_by_depth: torch.Tensor,
    depth_values_m: tuple[float, ...],
    requested_depth_m: float = 0.5,
    progress_label: str | None = None,
) -> TemperatureEvaluationResult:
    """Valuta la temperatura in unita' fisiche senza riaddestrare il modello."""

    if temperature_std_by_depth.ndim != 1:
        raise ValueError("Lo standard della temperatura deve essere 1D.")
    if len(depth_values_m) != temperature_std_by_depth.numel():
        raise ValueError(
            "Il numero di profondita' non coincide con gli standard "
            "della temperatura."
        )
    if not depth_values_m:
        raise ValueError("Il dataset non contiene profondita'.")

    selected_depth_index = min(
        range(len(depth_values_m)),
        key=lambda index: abs(depth_values_m[index] - requested_depth_m),
    )
    selected_depth_m = float(depth_values_m[selected_depth_index])
    temperature_std = temperature_std_by_depth.to(
        device=device,
        dtype=torch.float32,
    )
    if not torch.isfinite(temperature_std).all() or bool(
        (temperature_std <= 0).any()
    ):
        raise ValueError("Gli standard della temperatura devono essere validi.")
    temperature_std = temperature_std.view(1, -1, 1, 1)

    all_depths = _new_temperature_accumulator()
    selected_depth = _new_temperature_accumulator()
    model.eval()

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

    with torch.inference_mode():
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
                dtype=torch.bool,
            )
            if input_volume.ndim != 5 or input_volume.shape[1] < 1:
                raise ValueError("Forma del volume di input non valida.")
            if input_volume.shape[1] < target_volume.shape[1]:
                raise ValueError("L'input non contiene l'ultimo stato completo.")
            if target_mask.shape != target_volume.shape:
                raise ValueError("Target mask e target devono avere la stessa forma.")
            if target_volume.shape[2] != temperature_std.shape[1]:
                raise ValueError("Profondita' inattesa nel batch.")

            output = model(input_volume)
            mean = output["mean"]
            log_variance = output["log_variance"]
            if mean.shape != target_volume.shape:
                raise ValueError("Forma della media prevista inattesa.")
            if log_variance.shape != target_volume.shape:
                raise ValueError("Forma della log-varianza prevista inattesa.")

            target_temperature = target_volume[:, 0]
            valid_mask = target_mask[:, 0]
            model_error_c = (
                mean[:, 0] - target_temperature
            ) * temperature_std
            latest_input = input_volume[:, -target_volume.shape[1] :]
            persistence_error_c = (
                latest_input[:, 0] - target_temperature
            ) * temperature_std
            model_standard_deviation_c = torch.exp(
                0.5 * log_variance[:, 0]
            ) * temperature_std

            _accumulate_temperature_metrics(
                all_depths,
                model_error_c,
                persistence_error_c,
                model_standard_deviation_c,
                valid_mask,
            )
            _accumulate_temperature_metrics(
                selected_depth,
                model_error_c[:, selected_depth_index],
                persistence_error_c[:, selected_depth_index],
                model_standard_deviation_c[:, selected_depth_index],
                valid_mask[:, selected_depth_index],
            )
            if progress_bar is not None and all_depths["valid_points"]:
                progress_bar.set_postfix(
                    rmse_c=(
                        f"{(all_depths['model_squared_error_sum'] / all_depths['valid_points']) ** 0.5:.3f}"
                    ),
                )

    return TemperatureEvaluationResult(
        all_depths=_finalize_temperature_metrics(all_depths),
        selected_depth=_finalize_temperature_metrics(selected_depth),
        selected_depth_m=selected_depth_m,
    )


def _new_temperature_accumulator() -> dict[str, float | int]:
    return {
        "model_squared_error_sum": 0.0,
        "model_absolute_error_sum": 0.0,
        "model_error_sum": 0.0,
        "model_standard_deviation_sum": 0.0,
        "model_coverage_68_count": 0,
        "model_coverage_95_count": 0,
        "persistence_squared_error_sum": 0.0,
        "persistence_absolute_error_sum": 0.0,
        "persistence_error_sum": 0.0,
        "valid_points": 0,
    }


def _accumulate_temperature_metrics(
    accumulator: dict[str, float | int],
    model_error_c: torch.Tensor,
    persistence_error_c: torch.Tensor,
    model_standard_deviation_c: torch.Tensor,
    valid_mask: torch.Tensor,
) -> None:
    valid_points = int(valid_mask.sum().item())
    if valid_points == 0:
        return

    model_error = model_error_c[valid_mask]
    persistence_error = persistence_error_c[valid_mask]
    standard_deviation = model_standard_deviation_c[valid_mask]
    absolute_model_error = model_error.abs()

    accumulator["model_squared_error_sum"] += float(
        model_error.pow(2).sum().item()
    )
    accumulator["model_absolute_error_sum"] += float(
        absolute_model_error.sum().item()
    )
    accumulator["model_error_sum"] += float(model_error.sum().item())
    accumulator["model_standard_deviation_sum"] += float(
        standard_deviation.sum().item()
    )
    accumulator["model_coverage_68_count"] += int(
        (absolute_model_error <= standard_deviation).sum().item()
    )
    accumulator["model_coverage_95_count"] += int(
        (absolute_model_error <= 1.96 * standard_deviation).sum().item()
    )
    accumulator["persistence_squared_error_sum"] += float(
        persistence_error.pow(2).sum().item()
    )
    accumulator["persistence_absolute_error_sum"] += float(
        persistence_error.abs().sum().item()
    )
    accumulator["persistence_error_sum"] += float(
        persistence_error.sum().item()
    )
    accumulator["valid_points"] += valid_points


def _finalize_temperature_metrics(
    accumulator: dict[str, float | int],
) -> TemperatureComparisonMetrics:
    valid_points = int(accumulator["valid_points"])
    if valid_points == 0:
        raise ValueError("La valutazione non contiene temperature valide.")

    model_rmse_c = (
        float(accumulator["model_squared_error_sum"]) / valid_points
    ) ** 0.5
    persistence_rmse_c = (
        float(accumulator["persistence_squared_error_sum"]) / valid_points
    ) ** 0.5
    return TemperatureComparisonMetrics(
        model_rmse_c=model_rmse_c,
        model_mae_c=(
            float(accumulator["model_absolute_error_sum"]) / valid_points
        ),
        model_bias_c=float(accumulator["model_error_sum"]) / valid_points,
        model_mean_standard_deviation_c=(
            float(accumulator["model_standard_deviation_sum"])
            / valid_points
        ),
        model_coverage_68=(
            int(accumulator["model_coverage_68_count"]) / valid_points
        ),
        model_coverage_95=(
            int(accumulator["model_coverage_95_count"]) / valid_points
        ),
        persistence_rmse_c=persistence_rmse_c,
        persistence_mae_c=(
            float(accumulator["persistence_absolute_error_sum"])
            / valid_points
        ),
        persistence_bias_c=(
            float(accumulator["persistence_error_sum"]) / valid_points
        ),
        rmse_skill_score=rmse_skill_score(
            model_rmse_c,
            persistence_rmse_c,
        ),
        valid_points=valid_points,
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
    learning_curve_directory: Path | None = None,
    mean_mse_weight: float = 1.0,
    training_config: dict[str, object] | None = None,
    gradient_clip_norm: float | None = 1.0,
    temperature_mse_weight: float = 1.0,
) -> ForecastFitResult:
    """Addestra con early stopping e ripresa da un checkpoint opzionale."""

    if epochs <= 0:
        raise ValueError("epochs deve essere positivo.")
    if patience <= 0:
        raise ValueError("patience deve essere positivo.")
    if temperature_mse_weight <= 0:
        raise ValueError("temperature_mse_weight deve essere positivo.")

    checkpoint_path = Path(checkpoint_path)
    last_checkpoint_path = _last_checkpoint_path(checkpoint_path)
    start_epoch = 1
    best_validation_nll = float("inf")
    best_validation_rmse = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, object]] = []

    if resume_checkpoint is not None:
        start_epoch = int(resume_checkpoint["epoch"]) + 1
        best_validation_nll = float(
            resume_checkpoint.get("best_validation_nll", float("inf"))
        )
        best_validation_rmse = float(
            resume_checkpoint.get("best_validation_rmse", float("inf"))
        )
        best_epoch = int(resume_checkpoint.get("best_epoch", 0))
        epochs_without_improvement = int(
            resume_checkpoint.get("epochs_without_improvement", 0)
        )
        history = list(resume_checkpoint.get("history", []))

    if learning_curve_directory is not None and history:
        plot_learning_curve_snapshots(
            history,
            learning_curve_directory,
        )

    if start_epoch > epochs:
        raise ValueError(
            "Il checkpoint ha gia' raggiunto il numero massimo di epoche."
        )

    train_persistence_metrics = run_persistence_baseline(
        batches=train_batches,
        device=device,
        progress_label=(
            "Training persistence baseline" if show_progress else None
        ),
    )
    persistence_metrics = run_persistence_baseline(
        batches=validation_batches,
        device=device,
        progress_label=(
            "Validation persistence baseline" if show_progress else None
        ),
    )

    for epoch in range(start_epoch, epochs + 1):
        optimization_metrics = run_forecast_epoch(
            model=model,
            batches=train_batches,
            device=device,
            optimizer=optimizer,
            progress_label=(
                f"Train epoch {epoch}/{epochs}" if show_progress else None
            ),
            mean_mse_weight=mean_mse_weight,
            gradient_clip_norm=gradient_clip_norm,
            temperature_mse_weight=temperature_mse_weight,
        )
        # Ricalcolo a pesi fissi: rende train e validation confrontabili.
        train_metrics = run_forecast_epoch(
            model=model,
            batches=train_batches,
            device=device,
            progress_label=(
                f"Train eval {epoch}/{epochs}" if show_progress else None
            ),
            mean_mse_weight=mean_mse_weight,
            temperature_mse_weight=temperature_mse_weight,
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
            mean_mse_weight=mean_mse_weight,
            temperature_mse_weight=temperature_mse_weight,
        )
        validation_skill = rmse_skill_score(
            validation_metrics.rmse,
            persistence_metrics.rmse,
        )
        history.append(
            {
                "epoch": epoch,
                "train": asdict(train_metrics),
                "validation": asdict(validation_metrics),
                "train_persistence": asdict(train_persistence_metrics),
                "validation_persistence": asdict(persistence_metrics),
                "validation_rmse_skill": validation_skill,
            }
        )

        print(
            f"Epoch {epoch:03d} | "
            f"optimization objective {optimization_metrics.objective:.6f} | "
            f"train NLL {train_metrics.gaussian_nll:.6f} | "
            f"train RMSE {train_metrics.rmse:.6f} | "
            f"val NLL {validation_metrics.gaussian_nll:.6f} | "
            f"val RMSE {validation_metrics.rmse:.6f} | "
            f"train persistence RMSE {train_persistence_metrics.rmse:.6f} | "
            f"persistence RMSE {persistence_metrics.rmse:.6f} | "
            f"skill {validation_skill:.4f} | "
            f"coverage 68/95 "
            f"{validation_metrics.coverage_68:.3f}/"
            f"{validation_metrics.coverage_95:.3f}"
        )

        best_validation_nll = min(
            best_validation_nll,
            validation_metrics.gaussian_nll,
        )
        if validation_metrics.rmse < best_validation_rmse:
            best_validation_rmse = validation_metrics.rmse
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
                best_validation_rmse=best_validation_rmse,
                best_epoch=best_epoch,
                epochs_without_improvement=epochs_without_improvement,
                checkpoint_kind="best",
                training_config=training_config,
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
            best_validation_rmse=best_validation_rmse,
            best_epoch=best_epoch,
            epochs_without_improvement=epochs_without_improvement,
            checkpoint_kind="last",
            training_config=training_config,
        )

        if learning_curve_directory is not None:
            snapshot_path = plot_learning_curve_snapshot(
                history,
                learning_curve_directory,
            )
            print(f"Curva cumulativa: {snapshot_path}")

        if epochs_without_improvement >= patience:
            print(f"Early stopping dopo {epoch} epoche.")
            break

    return ForecastFitResult(
        best_epoch=best_epoch,
        best_validation_nll=best_validation_nll,
        best_validation_rmse=best_validation_rmse,
        train_persistence_rmse=train_persistence_metrics.rmse,
        validation_persistence_rmse=persistence_metrics.rmse,
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
    best_validation_rmse: float,
    best_epoch: int,
    epochs_without_improvement: int,
    checkpoint_kind: str,
    training_config: dict[str, object] | None,
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
            "best_validation_rmse": best_validation_rmse,
            "best_epoch": best_epoch,
            "epochs_without_improvement": epochs_without_improvement,
            "checkpoint_kind": checkpoint_kind,
            "training_config": dict(training_config or {}),
        },
        temporary_path,
    )
    temporary_path.replace(path)


def _last_checkpoint_path(checkpoint_path: Path) -> Path:
    """Restituisce il percorso persistente dello stato dell'ultima epoca."""

    return checkpoint_path.with_name(
        f"{checkpoint_path.stem}_last{checkpoint_path.suffix}"
    )
