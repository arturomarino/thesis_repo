import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dataset import (
    OceanForecastDataset,
    OceanStateDataset,
    build_annual_evaluation_dataset,
)
from dataloader import create_ocean_dataloaders


def test_forecast_dataset_pairs_each_time_with_the_next_one() -> None:
    times = np.arange(
        np.datetime64("2024-01-01"),
        np.datetime64("2024-01-05"),
    )
    volume = np.arange(4, dtype=np.float32).reshape(4, 1, 1, 1)
    surface = np.zeros((4, 1, 1), dtype=np.float32)
    dataset = xr.Dataset(
        {
            variable: (
                ("time", "depth", "latitude", "longitude"),
                volume.copy(),
            )
            for variable in OceanStateDataset.DEFAULT_VOLUME_VARIABLES
        }
        | {
            OceanStateDataset.DEFAULT_SURFACE_VARIABLE: (
                ("time", "latitude", "longitude"),
                surface,
            )
        },
        coords={
            "time": times,
            "depth": [0],
            "latitude": [0],
            "longitude": [0],
        },
    )

    forecasts = OceanForecastDataset(dataset)
    first = forecasts[0]
    last = forecasts[-1]

    assert len(forecasts) == 3
    assert first["input_time_index"] == 0
    assert first["target_time_index"] == 1
    assert first["input"]["volume"][0, 0, 0, 0] == 0
    assert first["target"]["volume"][0, 0, 0, 0] == 1
    assert last["input_time_index"] == 2
    assert last["target_time_index"] == 3


def test_dataloaders_reject_overlapping_temporal_splits() -> None:
    times = np.arange(
        np.datetime64("2024-01-01"),
        np.datetime64("2024-01-09"),
    )
    volume = np.zeros((8, 1, 1, 1), dtype=np.float32)
    surface = np.zeros((8, 1, 1), dtype=np.float32)
    dataset = xr.Dataset(
        {
            variable: (
                ("time", "depth", "latitude", "longitude"),
                volume.copy(),
            )
            for variable in OceanStateDataset.DEFAULT_VOLUME_VARIABLES
        }
        | {
            OceanStateDataset.DEFAULT_SURFACE_VARIABLE: (
                ("time", "latitude", "longitude"),
                surface,
            )
        },
        coords={
            "time": times,
            "depth": [0],
            "latitude": [0],
            "longitude": [0],
        },
    )
    train = OceanForecastDataset(dataset.isel(time=slice(0, 4)))
    validation = OceanForecastDataset(dataset.isel(time=slice(3, 6)))
    test = OceanForecastDataset(dataset.isel(time=slice(6, 8)))

    with pytest.raises(ValueError, match="train e validation condividono"):
        create_ocean_dataloaders(train, validation, test)


def test_forecast_dataset_stacks_multiple_context_days() -> None:
    times = np.arange(
        np.datetime64("2024-01-01"),
        np.datetime64("2024-01-06"),
    )
    volume = np.arange(5, dtype=np.float32).reshape(5, 1, 1, 1)
    surface = np.zeros((5, 1, 1), dtype=np.float32)
    dataset = xr.Dataset(
        {
            variable: (
                ("time", "depth", "latitude", "longitude"),
                volume.copy(),
            )
            for variable in OceanStateDataset.DEFAULT_VOLUME_VARIABLES
        }
        | {
            OceanStateDataset.DEFAULT_SURFACE_VARIABLE: (
                ("time", "latitude", "longitude"),
                surface,
            )
        },
        coords={
            "time": times,
            "depth": [0],
            "latitude": [0],
            "longitude": [0],
        },
    )

    forecasts = OceanForecastDataset(dataset, context_steps=3)
    sample = forecasts[0]

    assert len(forecasts) == 2
    assert sample["input"]["volume"].shape == (12, 1, 1, 1)
    assert sample["input_time_index"] == 2
    assert sample["target_time_index"] == 3
    assert sample["input"]["volume"][0, 0, 0, 0] == 0
    assert sample["input"]["volume"][4, 0, 0, 0] == 1
    assert sample["input"]["volume"][8, 0, 0, 0] == 2


def test_annual_evaluation_uses_previous_year_only_as_context() -> None:
    times = np.arange(
        np.datetime64("1998-01-01"),
        np.datetime64("2000-01-01"),
    )
    shape = (times.size, 1, 1, 1)
    surface_shape = (times.size, 1, 1)
    dataset = xr.Dataset(
        {
            variable: (
                ("time", "depth", "latitude", "longitude"),
                np.zeros(shape, dtype=np.float32),
            )
            for variable in OceanStateDataset.DEFAULT_VOLUME_VARIABLES
        }
        | {
            OceanStateDataset.DEFAULT_SURFACE_VARIABLE: (
                ("time", "latitude", "longitude"),
                np.zeros(surface_shape, dtype=np.float32),
            )
        },
        coords={
            "time": times,
            "depth": [0.506],
            "latitude": [40.0],
            "longitude": [15.0],
        },
    )
    previous = dataset.sel(time=slice("1998-01-01", "1998-12-31"))
    target = dataset.sel(time=slice("1999-01-01", "1999-12-31"))

    forecasts = build_annual_evaluation_dataset(
        previous,
        target,
        context_steps=3,
    )

    target_days = forecasts.target_time_values.astype("datetime64[D]")
    assert len(forecasts) == 364
    assert forecasts.time_values[0].astype("datetime64[D]") == np.datetime64(
        "1998-12-30"
    )
    assert target_days[0] == np.datetime64("1999-01-02")
    assert target_days[-1] == np.datetime64("1999-12-31")
    assert np.all(target_days.astype("datetime64[Y]") == np.datetime64("1999"))
    assert np.all(np.diff(forecasts.time_values.astype("datetime64[D]")) == 1)
