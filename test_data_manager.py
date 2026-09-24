from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from src.data_manager import DataManager


def test_data_manager_loads_required_variables_lazily(tmp_path: Path) -> None:
    path = tmp_path / "copernicus.nc"
    coordinates = {
        "time": np.array(["1999-01-01"], dtype="datetime64[D]"),
        "depth": [0.506],
        "latitude": [40.0],
        "longitude": [15.0],
    }
    volume = np.ones((1, 1, 1, 1), dtype=np.float32)
    surface = np.ones((1, 1, 1), dtype=np.float32)
    xr.Dataset(
        {
            variable: (
                ("time", "depth", "latitude", "longitude"),
                volume.copy(),
            )
            for variable in ("thetao_cglo", "so_cglo", "uo_cglo", "vo_cglo")
        }
        | {
            "zos_cglo": (
                ("time", "latitude", "longitude"),
                surface,
            )
        },
        coords=coordinates,
    ).to_netcdf(path)

    dataset = DataManager(path, chunks={"time": 1}).load()
    try:
        assert set(dataset.data_vars) == DataManager.REQUIRED_VARIABLES
        assert dataset["thetao_cglo"].chunks is not None
    finally:
        dataset.close()


def test_data_manager_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Dataset non trovato"):
        DataManager(tmp_path / "missing.nc").load()
