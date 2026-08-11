import sys
from pathlib import Path

import numpy as np
import torch
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from models.autoencoder import (
    VolumeAutoencoderConfig,
    VolumeUNetAutoencoder,
)
from plot_temperature_forecast import (
    VOLUME_VARIABLES,
    denormalize_temperature_forecast,
    find_input_time_index,
    prepare_normalized_input,
    resolve_forecast_dates,
    run_temperature_forecast,
)


def make_dataset(size: int = 8) -> xr.Dataset:
    coordinates = {
        "time": np.array(["2000-01-01", "2000-01-02"], dtype="datetime64[D]"),
        "depth": np.arange(size, dtype=np.float32),
        "latitude": np.arange(size, dtype=np.float32),
        "longitude": np.arange(size, dtype=np.float32),
    }
    shape = (2, size, size, size)
    return xr.Dataset(
        {
            variable: (
                ("time", "depth", "latitude", "longitude"),
                np.full(shape, channel + 2.0, dtype=np.float32),
            )
            for channel, variable in enumerate(VOLUME_VARIABLES)
        },
        coords=coordinates,
    )


def make_statistics(size: int = 8) -> xr.Dataset:
    coordinates = {"depth": np.arange(size, dtype=np.float32)}
    data_vars: dict[str, tuple[tuple[str], np.ndarray]] = {}
    for variable in VOLUME_VARIABLES:
        data_vars[f"{variable}_mean"] = (
            ("depth",),
            np.ones(size, dtype=np.float32),
        )
        data_vars[f"{variable}_std"] = (
            ("depth",),
            np.full(size, 2.0, dtype=np.float32),
        )
    return xr.Dataset(data_vars, coords=coordinates)


def test_prepares_input_and_denormalizes_temperature() -> None:
    dataset = make_dataset()
    statistics = make_statistics()
    mask = xr.DataArray(
        np.ones((8, 8), dtype=bool),
        dims=("latitude", "longitude"),
        coords={
            "latitude": dataset.latitude,
            "longitude": dataset.longitude,
        },
    )
    mask[0, 0] = False

    time_index = find_input_time_index(dataset, "2000-01-01")
    volume, valid_mask = prepare_normalized_input(
        dataset,
        statistics,
        mask,
        time_index,
    )
    forecast = denormalize_temperature_forecast(
        np.full((8, 8, 8), 3.0, dtype=np.float32),
        statistics,
        valid_mask,
        dataset,
        np.datetime64("2000-01-02"),
    )

    assert volume.shape == (4, 8, 8, 8)
    assert volume[0, 0, 1, 1].item() == 0.5
    assert volume[0, 0, 0, 0].item() == 0.0
    assert np.isnan(forecast.values[:, 0, 0]).all()
    assert forecast.values[0, 1, 1] == 7.0
    assert forecast.attrs["units"] == "degrees_C"


def test_loads_checkpoint_and_runs_forecast(tmp_path: Path) -> None:
    config = VolumeAutoencoderConfig(
        input_channels=4,
        output_channels=4,
        base_channels=2,
        latent_channels=4,
    )
    model = VolumeUNetAutoencoder(config)
    checkpoint_path = tmp_path / "forecast.pt"
    torch.save(
        {
            "epoch": 12,
            "model_config": {
                "input_channels": 4,
                "output_channels": 4,
                "base_channels": 2,
                "latent_channels": 4,
            },
            "model_state_dict": model.state_dict(),
        },
        checkpoint_path,
    )

    forecast, epoch = run_temperature_forecast(
        checkpoint_path,
        torch.zeros(4, 8, 8, 8),
        torch.device("cpu"),
    )

    assert forecast.shape == (8, 8, 8)
    assert epoch == 12


def test_resolves_user_selected_forecast_date() -> None:
    input_date, forecast_day = resolve_forecast_dates(
        input_date=None,
        forecast_date="2000-08-15",
    )

    assert input_date == "2000-08-14"
    assert forecast_day == np.datetime64("2000-08-15")
