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
