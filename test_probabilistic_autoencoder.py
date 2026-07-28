import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from losses import masked_gaussian_nll_loss
from inference import gaussian_statistics, sample_gaussian_prediction
from models.autoencoder import VolumeAutoencoderConfig, VolumeUNetAutoencoder


def test_gaussian_nll_matches_formula_and_ignores_masked_points() -> None:
    mean = torch.tensor([0.0, 100.0], requires_grad=True)
    log_variance = torch.tensor(
        [torch.log(torch.tensor(4.0)), -100.0],
        requires_grad=True,
    )
    target = torch.tensor([2.0, -100.0])
    mask = torch.tensor([True, False])

    loss = masked_gaussian_nll_loss(mean, log_variance, target, mask)
    expected = 0.5 * (1.0 + torch.log(torch.tensor(4.0)))
    loss.backward()

    torch.testing.assert_close(loss, expected)
    assert mean.grad is not None
    assert log_variance.grad is not None
    assert torch.isfinite(mean.grad).all()
    assert torch.isfinite(log_variance.grad).all()
    assert mean.grad[1] == 0
    assert log_variance.grad[1] == 0


def test_gaussian_nll_has_gradients_for_both_outputs() -> None:
    mean = torch.zeros(2, requires_grad=True)
    log_variance = torch.zeros(2, requires_grad=True)
    target = torch.tensor([1.0, 2.0])
    mask = torch.ones(2, dtype=torch.bool)

    loss = masked_gaussian_nll_loss(mean, log_variance, target, mask)
    loss.backward()

    assert mean.grad is not None
    assert log_variance.grad is not None
    assert torch.isfinite(mean.grad).all()
    assert torch.isfinite(log_variance.grad).all()


def test_autoencoder_predicts_mean_and_log_variance() -> None:
    model = VolumeUNetAutoencoder(
        VolumeAutoencoderConfig(
            input_channels=4,
            output_channels=4,
            base_channels=2,
            latent_channels=4,
        )
    )
    volume = torch.randn(1, 4, 8, 8, 8)

    output = model(volume)

    assert set(output) == {"mean", "log_variance", "latent"}
    assert output["mean"].shape == volume.shape
    assert output["log_variance"].shape == volume.shape
    torch.testing.assert_close(
        output["log_variance"],
        torch.zeros_like(output["log_variance"]),
    )


def test_inference_converts_log_variance_and_samples() -> None:
    mean = torch.zeros(1, 2)
    log_variance = torch.log(torch.full_like(mean, 4.0))

    statistics = gaussian_statistics(mean, log_variance)
    samples = sample_gaussian_prediction(
        mean,
        log_variance,
        num_samples=3,
        generator=torch.Generator().manual_seed(0),
    )

    torch.testing.assert_close(
        statistics["standard_deviation"],
        torch.full_like(mean, 2.0),
    )
    assert samples.shape == (3, 1, 2)
