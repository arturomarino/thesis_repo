import sys
from pathlib import Path

import numpy as np
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from split import TemporalSplitter


def test_last_year_is_test_and_previous_year_is_validation() -> None:
    time = np.arange(
        np.datetime64("2020-01"),
        np.datetime64("2024-01"),
        dtype="datetime64[M]",
    )
    dataset = xr.Dataset(
        {"value": ("time", np.arange(time.size))},
        coords={"time": time},
    )

    splits = TemporalSplitter().split(dataset)

    assert set(splits.train.time.dt.year.values.tolist()) == {2020, 2021}
    assert set(splits.validation.time.dt.year.values.tolist()) == {2022}
    assert set(splits.test.time.dt.year.values.tolist()) == {2023}
    assert splits.train.sizes["time"] == 24
    assert splits.validation.sizes["time"] == 12
    assert splits.test.sizes["time"] == 12
