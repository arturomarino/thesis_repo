import sys
from pathlib import Path

import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from training import (
    fit_forecaster,
    load_forecaster_checkpoint,
    run_forecast_epoch,
)
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
            "train": {"gaussian_nll": 1.2},
            "validation": {"gaussian_nll": 1.4},
        },
        {
            "epoch": 2,
            "train": {"gaussian_nll": 0.8},
            "validation": {"gaussian_nll": 0.9},
        },
    ]
    output_path = tmp_path / "learning_curve.png"

    result = plot_learning_curve(history, output_path)

    assert result == output_path
    assert output_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
