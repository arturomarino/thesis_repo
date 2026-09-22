"""Esportazione riproducibile delle valutazioni fisiche annuali."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Mapping

import numpy as np
import xarray as xr

from training import AnnualErrorMapResult, PhysicalEvaluationResult


VARIABLE_LABELS = {
    "thetao_cglo": "Potential temperature",
    "so_cglo": "Salinity",
    "uo_cglo": "Zonal velocity u",
    "vo_cglo": "Meridional velocity v",
}


def save_physical_metrics(
    result: PhysicalEvaluationResult,
    output_directory: Path,
    *,
    split_label: str,
) -> tuple[Path, Path]:
    """Salva metriche complete in JSON e CSV, senza arrotondamenti."""

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    stem = f"physical_metrics_{split_label.lower()}"
    json_path = output_directory / f"{stem}.json"
    csv_path = output_directory / f"{stem}.csv"
    records: list[dict[str, object]] = []
    for variable, evaluation in result.variables.items():
        for scope, metrics in (
            ("all_depths", evaluation.all_depths),
            ("selected_depth", evaluation.selected_depth),
        ):
            records.append(
                {
                    "split": split_label,
                    "variable": variable,
                    "unit": evaluation.unit,
                    "scope": scope,
                    "selected_depth_m": result.selected_depth_m,
                    **asdict(metrics),
                }
            )

    payload = {
        "split": split_label,
        "forecast_count": result.forecast_count,
        "selected_depth_m": result.selected_depth_m,
        "records": records,
    }
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return json_path, csv_path


def build_annual_error_dataset(
    result: AnnualErrorMapResult,
    *,
    variable_names: tuple[str, ...],
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    units: Mapping[str, str],
    target_year: int,
) -> xr.Dataset:
    """Converte i tensori aggregati in un dataset NetCDF auto-descrittivo."""

    errors = result.mean_absolute_error.numpy()
    persistence_errors = result.persistence_mean_absolute_error.numpy()
    mae_differences = result.mae_difference_model_minus_persistence.numpy()
    standard_deviations = result.error_standard_deviation.numpy()
    counts = result.valid_counts.numpy()
    expected_shape = (len(variable_names), len(latitudes), len(longitudes))
    if (
        errors.shape != expected_shape
        or persistence_errors.shape != expected_shape
        or mae_differences.shape != expected_shape
        or standard_deviations.shape != expected_shape
        or counts.shape != expected_shape
    ):
        raise ValueError(
            "Forma delle mappe inattesa: "
            f"MAE modello {errors.shape}, MAE persistence "
            f"{persistence_errors.shape}, "
            f"differenza MAE {mae_differences.shape}, deviazione standard "
            f"{standard_deviations.shape}, "
            f"conteggi {counts.shape}; attesa {expected_shape}."
        )
    coordinates = {
        "latitude": np.asarray(latitudes),
        "longitude": np.asarray(longitudes),
    }
    data_vars: dict[str, xr.DataArray] = {}
    for channel, variable in enumerate(variable_names):
        error_name = f"{variable}_mean_absolute_error"
        persistence_error_name = f"{variable}_persistence_mean_absolute_error"
        difference_name = f"{variable}_mae_difference_model_minus_persistence"
        standard_deviation_name = f"{variable}_error_standard_deviation"
        count_name = f"{variable}_valid_count"
        data_vars[error_name] = xr.DataArray(
            errors[channel],
            dims=("latitude", "longitude"),
            coords=coordinates,
            attrs={
                "long_name": f"annual mean absolute error of {variable}",
                "units": str(units.get(variable, "unknown")),
            },
        )
        data_vars[persistence_error_name] = xr.DataArray(
            persistence_errors[channel],
            dims=("latitude", "longitude"),
            coords=coordinates,
            attrs={
                "long_name": (
                    f"annual persistence mean absolute error of {variable}"
                ),
                "units": str(units.get(variable, "unknown")),
            },
        )
        data_vars[difference_name] = xr.DataArray(
            mae_differences[channel],
            dims=("latitude", "longitude"),
            coords=coordinates,
            attrs={
                "long_name": (
                    "annual MAE difference (model minus persistence) for "
                    f"{variable}"
                ),
                "units": str(units.get(variable, "unknown")),
            },
        )
        data_vars[standard_deviation_name] = xr.DataArray(
            standard_deviations[channel],
            dims=("latitude", "longitude"),
            coords=coordinates,
            attrs={
                "long_name": (
                    "annual temporal standard deviation of forecast error for "
                    f"{variable}"
                ),
                "units": str(units.get(variable, "unknown")),
            },
        )
        data_vars[count_name] = xr.DataArray(
            counts[channel],
            dims=("latitude", "longitude"),
            coords=coordinates,
            attrs={"long_name": f"valid forecast count for {variable}"},
        )
    return xr.Dataset(
        data_vars,
        attrs={
            "title": "Annual pointwise forecast error statistics",
            "target_year": int(target_year),
            "forecast_count": int(result.forecast_count),
            "selected_depth_m": float(result.selected_depth_m),
            "aggregation": (
                "model and persistence mean(abs(forecast-observation)), their "
                "difference (model minus persistence), and population standard "
                "deviation of the model signed error over valid dates"
            ),
        },
    )


def save_annual_error_outputs(
    dataset: xr.Dataset,
    output_directory: Path,
) -> tuple[Path, Path, Path, Path, Path]:
    """Salva NetCDF e quattro pannelli 2x2 delle statistiche annuali."""

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    target_year = int(dataset.attrs["target_year"])
    netcdf_path = output_directory / f"annual_mae_maps_{target_year}.nc"
    png_path = output_directory / f"annual_mae_maps_{target_year}.png"
    standard_deviation_png_path = (
        output_directory
        / f"annual_error_standard_deviation_maps_{target_year}.png"
    )
    persistence_png_path = (
        output_directory / f"annual_persistence_mae_maps_{target_year}.png"
    )
    difference_png_path = (
        output_directory
        / f"annual_mae_difference_model_minus_persistence_maps_{target_year}.png"
    )
    dataset.to_netcdf(netcdf_path)
    plot_annual_error_maps(dataset, png_path)
    plot_annual_error_standard_deviation_maps(
        dataset, standard_deviation_png_path
    )
    plot_annual_persistence_error_maps(dataset, persistence_png_path)
    plot_annual_mae_difference_maps(dataset, difference_png_path)
    return (
        netcdf_path,
        png_path,
        standard_deviation_png_path,
        persistence_png_path,
        difference_png_path,
    )


def plot_annual_error_maps(dataset: xr.Dataset, output_path: Path) -> Path:
    """Disegna le quattro mappe MAE del modello al 95° percentile."""

    return _plot_annual_maps(
        dataset,
        output_path,
        metric_suffix="_mean_absolute_error",
        colorbar_label="Mean absolute error",
        title="Annual mean absolute one-day forecast error",
    )


def plot_annual_error_standard_deviation_maps(
    dataset: xr.Dataset,
    output_path: Path,
) -> Path:
    """Disegna le mappe della deviazione standard temporale dell'errore."""

    return _plot_annual_maps(
        dataset,
        output_path,
        metric_suffix="_error_standard_deviation",
        colorbar_label="Error standard deviation",
        title="Annual standard deviation of one-day forecast error",
    )


def plot_annual_persistence_error_maps(
    dataset: xr.Dataset,
    output_path: Path,
) -> Path:
    """Disegna le quattro mappe MAE della baseline di persistence."""

    return _plot_annual_maps(
        dataset,
        output_path,
        metric_suffix="_persistence_mean_absolute_error",
        colorbar_label="Persistence mean absolute error",
        title="Annual mean absolute one-day persistence error",
    )


def plot_annual_mae_difference_maps(
    dataset: xr.Dataset,
    output_path: Path,
) -> Path:
    """Disegna la differenza MAE modello meno persistence, centrata su zero."""

    return _plot_annual_maps(
        dataset,
        output_path,
        metric_suffix="_mae_difference_model_minus_persistence",
        colorbar_label="Model MAE − persistence MAE",
        title="Annual MAE difference: model minus persistence",
        diverging=True,
    )


def _plot_annual_maps(
    dataset: xr.Dataset,
    output_path: Path,
    *,
    metric_suffix: str,
    colorbar_label: str,
    title: str,
    diverging: bool = False,
) -> Path:
    """Disegna un pannello 2x2 usando il 95° percentile come massimo."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    variables = tuple(
        name.removesuffix(metric_suffix)
        for name in dataset.data_vars
        if name.endswith(metric_suffix)
    )
    if len(variables) != 4:
        raise ValueError("La figura annuale richiede esattamente quattro variabili.")
    latitudes = np.asarray(dataset["latitude"].values)
    longitudes = np.asarray(dataset["longitude"].values)
    mean_latitude = float(np.mean(latitudes))
    figure, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    for axis, variable in zip(axes.flat, variables, strict=True):
        field = dataset[f"{variable}{metric_suffix}"]
        values = np.asarray(field.values, dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            raise ValueError(f"La mappa {variable} non contiene valori validi.")
        color_max = max(float(np.nanpercentile(np.abs(finite), 95)), 1e-12)
        color_limits = (
            {"vmin": -color_max, "vmax": color_max}
            if diverging
            else {"vmin": 0.0, "vmax": color_max}
        )
        mesh = axis.pcolormesh(
            longitudes,
            latitudes,
            np.ma.masked_invalid(values),
            cmap=plt.get_cmap("coolwarm" if diverging else "magma").with_extremes(
                bad="#d9d9d9"
            ),
            shading="auto",
            **color_limits,
        )
        color_bar = figure.colorbar(mesh, ax=axis, pad=0.02, extend="max")
        color_bar.set_label(f"{colorbar_label} ({field.attrs['units']})")
        axis.set(
            title=VARIABLE_LABELS.get(variable, variable),
            xlabel="Longitude (°)",
            ylabel="Latitude (°)",
        )
        axis.set_facecolor("#d9d9d9")
        axis.set_aspect(1.0 / np.cos(np.deg2rad(mean_latitude)))
        axis.grid(color="black", alpha=0.15, linewidth=0.5)
    figure.suptitle(
        f"{title} — "
        f"{dataset.attrs['target_year']}\n"
        f"depth {float(dataset.attrs['selected_depth_m']):.3f} m | "
        f"{int(dataset.attrs['forecast_count'])} forecasts"
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path
