import sys
from pathlib import Path

import numpy as np
import torch
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from annual_evaluation import (
    build_annual_error_dataset,
    save_annual_error_outputs,
)
from render_thesis_results import render_tables
from training import AnnualErrorMapResult


def test_annual_error_outputs_preserve_counts_units_and_metadata(
    tmp_path: Path,
) -> None:
    result = AnnualErrorMapResult(
        mean_absolute_error=torch.arange(16, dtype=torch.float64).reshape(
            4, 2, 2
        ),
        persistence_mean_absolute_error=torch.arange(
            16, dtype=torch.float64
        ).reshape(4, 2, 2),
        mae_difference_model_minus_persistence=torch.arange(
            16, dtype=torch.float64
        ).reshape(4, 2, 2) - 8.0,
        error_standard_deviation=torch.arange(
            16, dtype=torch.float64
        ).reshape(4, 2, 2),
        valid_counts=torch.full((4, 2, 2), 364, dtype=torch.int64),
        selected_depth_m=0.506,
        forecast_count=364,
    )
    variables = ("thetao_cglo", "so_cglo", "uo_cglo", "vo_cglo")
    dataset = build_annual_error_dataset(
        result,
        variable_names=variables,
        latitudes=np.array([39.0, 40.0]),
        longitudes=np.array([15.0, 16.0]),
        units={
            "thetao_cglo": "degrees_C",
            "so_cglo": "1e-3",
            "uo_cglo": "m s-1",
            "vo_cglo": "m s-1",
        },
        target_year=1999,
    )
    (
        netcdf_path,
        png_path,
        standard_deviation_png_path,
        persistence_png_path,
        difference_png_path,
    ) = save_annual_error_outputs(dataset, tmp_path)

    with xr.open_dataset(netcdf_path) as saved:
        assert saved.attrs["forecast_count"] == 364
        assert saved.attrs["selected_depth_m"] == 0.506
        assert saved["so_cglo_mean_absolute_error"].attrs["units"] == "1e-3"
        assert (
            saved["so_cglo_persistence_mean_absolute_error"].attrs["units"]
            == "1e-3"
        )
        assert saved["so_cglo_error_standard_deviation"].attrs["units"] == "1e-3"
        assert saved["vo_cglo_valid_count"].values.min() == 364
    assert png_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert standard_deviation_png_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert persistence_png_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert difference_png_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_render_thesis_tables_contains_all_variables_and_metrics() -> None:
    records = []
    for variable in ("thetao_cglo", "so_cglo", "uo_cglo", "vo_cglo"):
        for scope in ("all_depths", "selected_depth"):
            records.append(
                {
                    "split": "Validation",
                    "variable": variable,
                    "unit": "m s-1",
                    "scope": scope,
                    "selected_depth_m": 0.506,
                    "model_rmse": 1.0,
                    "model_mae": 0.8,
                    "model_bias": -0.1,
                    "model_mean_standard_deviation": 0.7,
                    "model_coverage_68": 0.68,
                    "model_coverage_95": 0.95,
                    "persistence_rmse": 0.9,
                    "persistence_mae": 0.7,
                    "persistence_bias": 0.0,
                    "rmse_skill_score": -0.2,
                    "valid_points": 10,
                }
            )
    validation = {"records": records}
    test = {
        "records": [{**record, "split": "Test"} for record in records]
    }

    latex = render_tables(validation, test)

    assert "Physical-unit deterministic results" in latex
    assert "Physical-unit probabilistic diagnostics" in latex
    assert "Temperature" in latex
    assert "Salinity" in latex
    assert "Zonal velocity $u$" in latex
    assert "Meridional velocity $v$" in latex
