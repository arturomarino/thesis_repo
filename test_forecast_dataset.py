import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dataset import OceanForecastDataset, OceanStateDataset
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
