from pathlib import Path

import numpy as np
import xarray as xr

from src.plot_temperature_map import (
    apply_land_sea_mask,
    plot_temperature_map,
    select_temperature_slice,
    selected_date,
    selected_depth,
)


def make_temperature() -> xr.DataArray:
    values = np.arange(3 * 2 * 4 * 5, dtype=np.float32).reshape(3, 2, 4, 5)
    return xr.DataArray(
        values,
        dims=("time", "depth", "latitude", "longitude"),
        coords={
            "time": np.arange(
                np.datetime64("2000-01-01"),
                np.datetime64("2000-01-04"),
            ),
            "depth": [0.5, 10.0],
            "latitude": np.linspace(30, 31, 4),
            "longitude": np.linspace(10, 11, 5),
        },
        attrs={"units": "degrees_C"},
        name="thetao_cglo",
    )


def test_selects_exact_date_and_nearest_depth() -> None:
    selected = select_temperature_slice(
        make_temperature(),
        date="2000-01-02",
        depth=1.0,
        seed=42,
    )

    assert selected.dims == ("latitude", "longitude")
    assert selected_date(selected) == "2000-01-02"
    assert selected_depth(selected) == 0.5
    np.testing.assert_array_equal(selected.values, make_temperature()[1, 0])


def test_random_date_is_reproducible() -> None:
    first = select_temperature_slice(
        make_temperature(), date=None, depth=None, seed=7
    )
    second = select_temperature_slice(
        make_temperature(), date=None, depth=None, seed=7
    )

    assert selected_date(first) == selected_date(second)
    np.testing.assert_array_equal(first.values, second.values)


def test_applies_mask_and_writes_plot(tmp_path: Path) -> None:
    selected = select_temperature_slice(
        make_temperature(),
        date="2000-01-01",
        depth=None,
        seed=42,
    )
    mask = xr.DataArray(
        np.ones((4, 5), dtype=np.int8),
        dims=("latitude", "longitude"),
        coords={
            "latitude": selected.latitude,
            "longitude": selected.longitude,
        },
        name="thetao_cglo",
    )
    mask[0, 0] = 0
    mask_path = tmp_path / "mask.nc"
    mask.to_netcdf(mask_path)

    masked = apply_land_sea_mask(selected, mask_path)
    output_path = tmp_path / "temperature.png"
    result = plot_temperature_map(masked, output_path, label_step=2)

    assert np.isnan(masked.values[0, 0])
    assert result == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0
