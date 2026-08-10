"""Genera una mappa della temperatura marina per un giorno e una profondita'."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr


TEMPERATURE_VARIABLE = "thetao_cglo"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Crea una mappa a colori della temperatura Copernicus con valori "
            "numerici sovrapposti."
        )
    )
    project_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="Percorso del dataset NetCDF Copernicus.",
    )
    parser.add_argument(
        "--mask-path",
        type=Path,
        default=None,
        help="Land-sea mask NetCDF opzionale (1 mare, 0 terra).",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help=(
            "Giorno nel formato YYYY-MM-DD. Se omesso viene scelto un giorno "
            "casuale in modo riproducibile."
        ),
    )
    parser.add_argument(
        "--depth",
        type=float,
        default=None,
        help=(
            "Profondita' in metri. Se omessa viene usato il primo livello, "
            "cioe' quello piu' vicino alla superficie."
        ),
    )
    parser.add_argument(
        "--label-step",
        type=int,
        default=8,
        help=(
            "Mostra un valore ogni N punti della griglia. Default: 8; usare "
            "un numero piu' piccolo per visualizzare piu' valori."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed usato soltanto per la scelta casuale del giorno.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=project_root / "outputs/temperature_maps",
        help="Cartella in cui salvare il PNG.",
    )
    return parser.parse_args()


def select_temperature_slice(
    temperature: xr.DataArray,
    *,
    date: str | None,
    depth: float | None,
    seed: int,
) -> xr.DataArray:
    """Seleziona una matrice latitudine-longitudine senza caricare tutto il file."""

    required_dimensions = {"time", "latitude", "longitude"}
    missing_dimensions = required_dimensions - set(temperature.dims)
    if missing_dimensions:
        raise ValueError(
            "Dimensioni mancanti nella temperatura: "
            f"{sorted(missing_dimensions)}"
        )
    if temperature.sizes["time"] == 0:
        raise ValueError("Il dataset non contiene giorni disponibili.")

    if date is None:
        rng = np.random.default_rng(seed)
        time_index = int(rng.integers(temperature.sizes["time"]))
        selected = temperature.isel(time=time_index)
    else:
        try:
            requested_date = np.datetime64(date, "D")
        except ValueError as error:
            raise ValueError("La data deve avere il formato YYYY-MM-DD.") from error

        available_days = temperature["time"].values.astype("datetime64[D]")
        matching_indices = np.flatnonzero(available_days == requested_date)
        if matching_indices.size == 0:
            first_day = np.datetime_as_string(available_days.min(), unit="D")
            last_day = np.datetime_as_string(available_days.max(), unit="D")
            raise ValueError(
                f"Il giorno {date} non e' presente nel dataset. "
                f"Intervallo disponibile: {first_day} - {last_day}."
            )
        selected = temperature.isel(time=int(matching_indices[0]))

    if "depth" in selected.dims:
        if selected.sizes["depth"] == 0:
            raise ValueError("Il dataset non contiene livelli di profondita'.")
        if depth is None:
            selected = selected.isel(depth=0)
        else:
            selected = selected.sel(depth=depth, method="nearest")
    elif depth is not None:
        raise ValueError("La variabile non possiede la dimensione depth.")

    remaining_dimensions = set(selected.dims)
    if remaining_dimensions != {"latitude", "longitude"}:
        raise ValueError(
            "Dopo la selezione sono rimaste dimensioni inattese: "
            f"{sorted(remaining_dimensions)}"
        )

    return selected.transpose("latitude", "longitude").load()


def apply_land_sea_mask(
    temperature: xr.DataArray,
    mask_path: Path | None,
) -> xr.DataArray:
    """Applica la maschera esterna, quando fornita."""

    if mask_path is None:
        return temperature
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask non trovata: {mask_path}")

    with xr.open_dataarray(mask_path) as opened_mask:
        mask = opened_mask.squeeze(drop=True).load()

    unexpected_dimensions = set(mask.dims) - {"latitude", "longitude"}
    if unexpected_dimensions:
        raise ValueError(
            "Dimensioni inattese nella land-sea mask: "
            f"{sorted(unexpected_dimensions)}"
        )

    try:
        aligned_temperature, aligned_mask = xr.align(
            temperature,
            mask,
            join="exact",
        )
    except ValueError as error:
        raise ValueError(
            "Le coordinate della mask non coincidono con quelle del dataset."
        ) from error

    return aligned_temperature.where(aligned_mask.astype(bool))


def format_temperature_unit(unit: object) -> str:
    normalized_unit = str(unit or "").strip().lower()
    if normalized_unit in {
        "degrees_c",
        "degree_c",
        "degrees celsius",
        "degree celsius",
        "celsius",
        "degc",
    }:
        return "°C"
    return str(unit).strip() if unit else "°C"


def selected_date(temperature: xr.DataArray) -> str:
    return np.datetime_as_string(
        np.asarray(temperature["time"].values).astype("datetime64[D]"),
        unit="D",
    )


def selected_depth(temperature: xr.DataArray) -> float | None:
    if "depth" not in temperature.coords:
        return None
    return float(np.asarray(temperature["depth"].values))


def plot_temperature_map(
    temperature: xr.DataArray,
    output_path: Path,
    *,
    label_step: int,
) -> Path:
    """Salva la mappa a colori e annota un campione regolare di valori."""

    if label_step <= 0:
        raise ValueError("label_step deve essere positivo.")

    values = np.asarray(temperature.values, dtype=float)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        raise ValueError("La mappa non contiene valori di temperatura validi.")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patheffects as path_effects
    import matplotlib.pyplot as plt

    latitudes = np.asarray(temperature["latitude"].values)
    longitudes = np.asarray(temperature["longitude"].values)
    unit = format_temperature_unit(temperature.attrs.get("units"))
    date_label = selected_date(temperature)
    depth_value = selected_depth(temperature)
    depth_label = (
        f"{depth_value:.2f} m" if depth_value is not None else "superficie"
    )

    figure, axis = plt.subplots(figsize=(16, 8))
    color_map = plt.get_cmap("turbo").with_extremes(bad="#d9d9d9")
    mesh = axis.pcolormesh(
        longitudes,
        latitudes,
        np.ma.masked_invalid(values),
        cmap=color_map,
        shading="auto",
        vmin=float(finite_values.min()),
        vmax=float(finite_values.max()),
    )
    axis.set_facecolor("#d9d9d9")

    labels_drawn = 0
    for latitude_index in range(0, len(latitudes), label_step):
        for longitude_index in range(0, len(longitudes), label_step):
            value = values[latitude_index, longitude_index]
            if not np.isfinite(value):
                continue
            label = axis.text(
                longitudes[longitude_index],
                latitudes[latitude_index],
                f"{value:.1f}",
                ha="center",
                va="center",
                color="white",
                fontsize=6.5,
                fontweight="bold",
            )
            label.set_path_effects(
                [path_effects.withStroke(linewidth=1.5, foreground="black")]
            )
            labels_drawn += 1

    color_bar = figure.colorbar(mesh, ax=axis, pad=0.02)
    color_bar.set_label(f"Temperatura ({unit})")
    axis.set(
        title=(
            f"Temperatura marina — {date_label} — profondità {depth_label}\n"
            f"Valori in {unit} mostrati ogni {label_step} punti di griglia"
        ),
        xlabel="Longitudine (°)",
        ylabel="Latitudine (°)",
    )
    mean_latitude = float(np.mean(latitudes))
    axis.set_aspect(1.0 / np.cos(np.deg2rad(mean_latitude)))
    axis.grid(color="black", alpha=0.15, linewidth=0.5)
    figure.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    print(f"Valori numerici disegnati: {labels_drawn}")
    return output_path


def build_output_path(
    output_directory: Path,
    temperature: xr.DataArray,
) -> Path:
    date_label = selected_date(temperature)
    depth_value = selected_depth(temperature)
    depth_token = (
        f"{depth_value:.2f}m".replace(".", "p")
        if depth_value is not None
        else "surface"
    )
    return output_directory / f"temperature_{date_label}_{depth_token}.png"


def main() -> None:
    args = parse_args()
    if args.label_step <= 0:
        raise ValueError("label-step deve essere positivo.")
    if not args.data_path.exists():
        raise FileNotFoundError(f"Dataset non trovato: {args.data_path}")

    with xr.open_dataset(args.data_path, chunks={"time": 1}) as dataset:
        if TEMPERATURE_VARIABLE not in dataset:
            raise ValueError(
                f"Variabile {TEMPERATURE_VARIABLE} assente nel dataset."
            )
        temperature = select_temperature_slice(
            dataset[TEMPERATURE_VARIABLE],
            date=args.date,
            depth=args.depth,
            seed=args.seed,
        )

    temperature = apply_land_sea_mask(temperature, args.mask_path)
    output_path = build_output_path(args.output_directory, temperature)
    plot_temperature_map(
        temperature,
        output_path,
        label_step=args.label_step,
    )

    print(f"Giorno selezionato: {selected_date(temperature)}")
    depth_value = selected_depth(temperature)
    if depth_value is not None:
        print(f"Profondita' selezionata: {depth_value:.3f} m")
    print(f"Mappa salvata in: {output_path}")


if __name__ == "__main__":
    main()
