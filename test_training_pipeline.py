import sys
from pathlib import Path

import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from training import (
    fit_forecaster,
    load_forecaster_checkpoint,
    rmse_skill_score,
    run_annual_error_map_evaluation,
    run_forecast_epoch,
    run_physical_evaluation,
    run_persistence_baseline,
    run_temperature_evaluation,
)
from models.autoencoder import VolumeAutoencoderConfig, VolumeUNetAutoencoder
from visualization import plot_learning_curve


class TinyProbabilisticForecaster(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mean_bias = nn.Parameter(torch.tensor(0.0))
        self.log_variance_bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, volume: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "mean": torch.zeros_like(volume) + self.mean_bias,
            "log_variance": (
                torch.zeros_like(volume) + self.log_variance_bias
            ),
            "latent": volume,
        }


def _batch() -> dict[str, object]:
    input_volume = torch.zeros(2, 1, 1, 1, 1)
    target_volume = torch.ones_like(input_volume)
    mask = torch.ones_like(input_volume, dtype=torch.bool)
    return {
        "input": {"volume": input_volume, "volume_mask": mask},
        "target": {"volume": target_volume, "volume_mask": mask},
        "input_time_index": torch.tensor([0, 1]),
        "target_time_index": torch.tensor([1, 2]),
    }


def test_fit_saves_best_checkpoint_and_reports_metrics(
    tmp_path: Path,
) -> None:
    model = TinyProbabilisticForecaster()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    checkpoint_path = tmp_path / "best.pt"

    result = fit_forecaster(
        model=model,
        train_batches=[_batch()],
        validation_batches=[_batch()],
        optimizer=optimizer,
        device=torch.device("cpu"),
        epochs=2,
        patience=2,
        checkpoint_path=checkpoint_path,
    )
    checkpoint = load_forecaster_checkpoint(
        path=checkpoint_path,
        model=TinyProbabilisticForecaster(),
        device=torch.device("cpu"),
    )
    last_checkpoint_path = checkpoint_path.with_name("best_last.pt")
    last_checkpoint = torch.load(
        last_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    resumed_model = TinyProbabilisticForecaster()
    resumed_model.load_state_dict(last_checkpoint["model_state_dict"])
    resumed_optimizer = torch.optim.SGD(resumed_model.parameters(), lr=0.1)
    resumed_optimizer.load_state_dict(last_checkpoint["optimizer_state_dict"])
    resumed_result = fit_forecaster(
        model=resumed_model,
        train_batches=[_batch()],
        validation_batches=[_batch()],
        optimizer=resumed_optimizer,
        device=torch.device("cpu"),
        epochs=3,
        patience=2,
        checkpoint_path=checkpoint_path,
        resume_checkpoint=last_checkpoint,
    )
    metrics = run_forecast_epoch(
        model=model,
        batches=[_batch()],
        device=torch.device("cpu"),
    )

    assert result.best_epoch >= 1
    assert result.epochs_completed == 2
    assert checkpoint_path.exists()
    assert checkpoint_path.with_name("best_last.pt").exists()
    assert checkpoint["epoch"] == result.best_epoch
    assert resumed_result.epochs_completed == 3
    assert metrics.valid_points == 2
    assert 0 <= metrics.coverage_68 <= 1
    assert 0 <= metrics.coverage_95 <= 1


def test_plot_learning_curve_creates_png(tmp_path: Path) -> None:
    history = [
        {
            "epoch": 1,
            "train": {"gaussian_nll": 1.2, "rmse": 1.1},
            "validation": {"gaussian_nll": 1.4, "rmse": 1.2},
            "train_persistence": {"rmse": 1.0},
            "validation_persistence": {"rmse": 0.9},
        },
        {
            "epoch": 2,
            "train": {"gaussian_nll": 0.8, "rmse": 0.7},
            "validation": {"gaussian_nll": 0.9, "rmse": 0.8},
            "train_persistence": {"rmse": 1.0},
            "validation_persistence": {"rmse": 0.9},
        },
    ]
    output_path = tmp_path / "learning_curve.png"

    result = plot_learning_curve(history, output_path)

    assert result == output_path
    assert output_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_persistence_baseline_and_skill_score() -> None:
    batch = _batch()
    # Due giorni di contesto: la persistence deve usare soltanto l'ultimo.
    batch["input"]["volume"] = torch.cat(
        (
            torch.full_like(batch["target"]["volume"], 99.0),
            torch.zeros_like(batch["target"]["volume"]),
        ),
        dim=1,
    )
    batch["input"]["volume_mask"] = torch.ones_like(
        batch["input"]["volume"],
        dtype=torch.bool,
    )
    metrics = run_persistence_baseline(
        batches=[batch],
        device=torch.device("cpu"),
    )

    assert metrics.rmse == 1.0
    assert metrics.mae == 1.0
    assert metrics.valid_points == 2
    assert rmse_skill_score(model_rmse=0.5, persistence_rmse=1.0) == 0.75


def test_temperature_evaluation_returns_physical_metrics() -> None:
    batch = _batch()
    batch["input"]["volume"] = torch.tensor(
        [[[[[1.0]], [[3.0]]]]],
    ).repeat(1, 4, 1, 1, 1)
    batch["target"]["volume"] = torch.tensor(
        [[[[[2.0]], [[5.0]]]]],
    ).repeat(1, 4, 1, 1, 1)
    batch["input"]["volume_mask"] = torch.ones_like(
        batch["input"]["volume"],
        dtype=torch.bool,
    )
    batch["target"]["volume_mask"] = torch.ones_like(
        batch["target"]["volume"],
        dtype=torch.bool,
    )
    result = run_temperature_evaluation(
        model=TinyProbabilisticForecaster(),
        batches=[batch],
        device=torch.device("cpu"),
        temperature_std_by_depth=torch.tensor([10.0, 2.0]),
        depth_values_m=(0.5, 10.0),
        requested_depth_m=0.5,
    )

    assert result.selected_depth_m == 0.5
    assert result.all_depths.model_mae_c == 15.0
    assert result.all_depths.model_bias_c == -15.0
    assert result.all_depths.persistence_mae_c == 7.0
    assert result.selected_depth.model_rmse_c == 20.0
    assert result.selected_depth.persistence_rmse_c == 10.0


def test_multivariable_physical_evaluation_scales_each_channel() -> None:
    input_volume = torch.zeros(1, 4, 2, 1, 1)
    target_volume = torch.ones_like(input_volume)
    mask = torch.ones_like(input_volume, dtype=torch.bool)
    mask[:, 1, 1] = False
    batch = {
        "input": {"volume": input_volume, "volume_mask": mask},
        "target": {"volume": target_volume, "volume_mask": mask},
        "input_time_index": torch.tensor([0]),
        "target_time_index": torch.tensor([1]),
    }
    names = ("temperature", "salinity", "u", "v")
    result = run_physical_evaluation(
        model=TinyProbabilisticForecaster(),
        batches=[batch],
        device=torch.device("cpu"),
        variable_names=names,
        standard_deviations=torch.tensor(
            [[2.0, 4.0], [3.0, 6.0], [0.1, 0.2], [0.5, 1.0]]
        ),
        depth_values_m=(0.506, 10.0),
        units={"temperature": "degC", "salinity": "1e-3", "u": "m/s", "v": "m/s"},
        requested_depth_m=0.5,
    )

    assert result.forecast_count == 1
    assert result.selected_depth_m == 0.506
    assert result.variables["temperature"].selected_depth.model_mae == 2.0
    assert result.variables["temperature"].all_depths.model_mae == 3.0
    assert result.variables["salinity"].all_depths.valid_points == 1
    assert result.variables["salinity"].all_depths.model_rmse == 3.0
    assert result.variables["u"].unit == "m/s"


def test_annual_error_map_contains_model_persistence_and_standard_deviation() -> None:
    input_volume = torch.zeros(2, 4, 1, 1, 2)
    target_volume = torch.ones_like(input_volume)
    target_volume[0, 0, 0, 0] = torch.tensor([1.0, -2.0])
    target_volume[1, 0, 0, 0] = torch.tensor([-3.0, 4.0])
    input_volume[0, 0, 0, 0] = torch.tensor([5.0, 1.0])
    input_volume[1, 0, 0, 0] = torch.tensor([9.0, 2.0])
    mask = torch.ones_like(target_volume, dtype=torch.bool)
    mask[1, 0, 0, 0, 0] = False
    mask[:, 3, 0, 0, 1] = False
    batch = {
        "input": {"volume": input_volume, "volume_mask": torch.ones_like(mask)},
        "target": {"volume": target_volume, "volume_mask": mask},
        "input_time_index": torch.tensor([0, 1]),
        "target_time_index": torch.tensor([1, 2]),
    }

    result = run_annual_error_map_evaluation(
        model=TinyProbabilisticForecaster(),
        batches=[batch],
        device=torch.device("cpu"),
        standard_deviations=torch.ones(4, 1),
        depth_values_m=(0.506,),
    )

    assert result.forecast_count == 2
    assert result.valid_counts[0, 0, 0].item() == 1
    assert result.mean_absolute_error[0, 0, 0].item() == 1.0
    assert result.mean_absolute_error[0, 0, 1].item() == 3.0
    assert result.persistence_mean_absolute_error[0, 0, 0].item() == 4.0
    assert result.persistence_mean_absolute_error[0, 0, 1].item() == 2.5
    assert result.mae_difference_model_minus_persistence[0, 0, 0].item() == -3.0
    assert result.mae_difference_model_minus_persistence[0, 0, 1].item() == 0.5
    assert result.error_standard_deviation[0, 0, 0].item() == 0.0
    assert result.error_standard_deviation[0, 0, 1].item() == 3.0
    assert result.valid_counts[3, 0, 1].item() == 0
    assert torch.isnan(result.mean_absolute_error[3, 0, 1])
    assert torch.isnan(result.persistence_mean_absolute_error[3, 0, 1])
    assert torch.isnan(result.mae_difference_model_minus_persistence[3, 0, 1])
    assert torch.isnan(result.error_standard_deviation[3, 0, 1])


def test_fit_early_stops_after_patience_without_improvement(
    tmp_path: Path,
) -> None:
    model = TinyProbabilisticForecaster()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    learning_curve_directory = tmp_path / "learning_curves"

    result = fit_forecaster(
        model=model,
        train_batches=[_batch()],
        validation_batches=[_batch()],
        optimizer=optimizer,
        device=torch.device("cpu"),
        epochs=50,
        patience=2,
        checkpoint_path=tmp_path / "best.pt",
        show_progress=False,
        learning_curve_directory=learning_curve_directory,
    )

    assert result.best_epoch == 1
    assert result.epochs_completed == 3
    for epoch in range(1, 4):
        snapshot_path = learning_curve_directory / (
            f"learning_curve_epoch_{epoch:03d}.png"
        )
        assert snapshot_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_multiday_context_trains_against_four_channel_target() -> None:
    model = VolumeUNetAutoencoder(
        VolumeAutoencoderConfig(
            input_channels=12,
            output_channels=4,
            base_channels=2,
            latent_channels=4,
            normalization="none",
        )
    )
    input_volume = torch.randn(1, 12, 4, 4, 4)
    target_volume = torch.randn(1, 4, 4, 4, 4)
    batch = {
        "input": {
            "volume": input_volume,
            "volume_mask": torch.ones_like(input_volume, dtype=torch.bool),
        },
        "target": {
            "volume": target_volume,
            "volume_mask": torch.ones_like(target_volume, dtype=torch.bool),
        },
        "input_time_index": torch.tensor([2]),
        "target_time_index": torch.tensor([3]),
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    metrics = run_forecast_epoch(
        model=model,
        batches=[batch],
        device=torch.device("cpu"),
        optimizer=optimizer,
        mean_mse_weight=1.0,
        gradient_clip_norm=1.0,
    )

    assert metrics.valid_points == target_volume.numel()
    assert metrics.objective >= metrics.gaussian_nll
