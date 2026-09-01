"""Genera la mappa della temperatura prevista dal modello per t + 1 giorno."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from models.autoencoder import VolumeAutoencoderConfig, VolumeUNetAutoencoder


VOLUME_VARIABLES = (
    "thetao_cglo",
    "so_cglo",
    "uo_cglo",
    "vo_cglo",
)
TEMPERATURE_CHANNEL = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Usa il checkpoint addestrato per prevedere la temperatura del "
            "giorno successivo e salvarne la mappa in gradi Celsius."
        )
    )
    project_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="Dataset Copernicus NetCDF usato come input della previsione.",
    )
    parser.add_argument(
        "--mask-path",
        type=Path,
        required=True,
        help="Land-sea mask NetCDF (1 mare, 0 terra).",
    )
    parser.add_argument(
        "--stats-path",
        type=Path,
        required=True,
        help="Statistiche di normalizzazione apprese sul training set.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        required=True,
        help="Checkpoint del previsore, normalmente best_forecaster.pt.",
    )
    date_group = parser.add_mutually_exclusive_group(required=True)
    date_group.add_argument(
        "--forecast-date",
        type=str,
        default=None,
        help=(
            "Giorno che vuoi prevedere nel formato YYYY-MM-DD. Il modello "
            "usera' automaticamente il giorno precedente come input."
        ),
    )
    date_group.add_argument(
        "--input-date",
        type=str,
        default=None,
        help=(
            "Giorno fornito al modello nel formato YYYY-MM-DD. La mappa "
            "rappresenta la previsione per il giorno successivo."
        ),
    )
    parser.add_argument(
        "--depth",
        type=float,
        default=None,
        help=(
            "Profondita' in metri; viene scelto il livello disponibile piu' "
            "vicino. Per default usa il primo livello superficiale."
        ),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Device PyTorch; auto seleziona il migliore disponibile.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=project_root / "outputs/temperature_forecasts",
        help="Cartella in cui salvare il PNG della previsione.",
    )
    return parser.parse_args()


def resolve_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA richiesta ma non disponibile.")
    if requested_device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS richiesto ma non disponibile.")
    return torch.device(requested_device)


def find_input_time_index(dataset: xr.Dataset, input_date: str) -> int:
    """Trova esattamente il giorno scelto, senza sostituirlo con uno vicino."""

    if "time" not in dataset.coords or dataset.sizes.get("time", 0) == 0:
        raise ValueError("Il dataset non contiene una coordinata time valida.")
    try:
        requested_date = np.datetime64(input_date, "D")
    except ValueError as error:
        raise ValueError("input-date deve avere il formato YYYY-MM-DD.") from error

    available_days = dataset["time"].values.astype("datetime64[D]")
    matching_indices = np.flatnonzero(available_days == requested_date)
    if matching_indices.size == 0:
        first_day = np.datetime_as_string(available_days.min(), unit="D")
        last_day = np.datetime_as_string(available_days.max(), unit="D")
        raise ValueError(
            f"Il giorno {input_date} non e' presente nel dataset. "
            f"Intervallo disponibile: {first_day} - {last_day}."
        )
    return int(matching_indices[0])


def resolve_forecast_dates(
    *,
    input_date: str | None,
    forecast_date: str | None,
) -> tuple[str, np.datetime64]:
    """Restituisce il giorno di input e il giorno target scelto dall'utente."""

    selected_date = forecast_date if forecast_date is not None else input_date
    if selected_date is None:
        raise ValueError("Specificare forecast-date oppure input-date.")
    try:
        selected_day = np.datetime64(selected_date, "D")
    except ValueError as error:
        raise ValueError("La data deve avere il formato YYYY-MM-DD.") from error

    if forecast_date is not None:
        forecast_day = selected_day
        input_day = forecast_day - np.timedelta64(1, "D")
    else:
        input_day = selected_day
        forecast_day = input_day + np.timedelta64(1, "D")

    return np.datetime_as_string(input_day, unit="D"), forecast_day


def load_land_sea_mask(mask_path: Path) -> xr.DataArray:
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask non trovata: {mask_path}")
    with xr.open_dataarray(mask_path) as opened_mask:
        mask = (
            opened_mask.squeeze(drop=True)
            .reset_coords(drop=True)
            .load()
        )
    if set(mask.dims) != {"latitude", "longitude"}:
        raise ValueError(
            "La mask deve avere soltanto le dimensioni latitude e longitude."
        )
    return mask.transpose("latitude", "longitude").astype(bool)


def prepare_normalized_input(
    dataset: xr.Dataset,
    statistics: xr.Dataset,
    mask: xr.DataArray,
    time_index: int,
    context_steps: int = 1,
) -> tuple[torch.Tensor, np.ndarray]:
    """Materializza gli ultimi stati e concatena i canali temporali."""

    if context_steps <= 0:
        raise ValueError("context_steps deve essere positivo.")
    first_index = time_index - context_steps + 1
    if first_index < 0:
        raise ValueError(
            "Non ci sono abbastanza giorni precedenti per il contesto "
            f"richiesto ({context_steps})."
        )
    context_times = dataset["time"].isel(
        time=slice(first_index, time_index + 1)
    ).values.astype("datetime64[D]")
    if context_times.size > 1 and not np.all(
        np.diff(context_times) == np.timedelta64(1, "D")
    ):
        raise ValueError("I giorni di contesto non sono consecutivi.")

    states: list[torch.Tensor] = []
    valid_mask: np.ndarray | None = None
    for context_index in range(first_index, time_index + 1):
        state, valid_mask = _prepare_normalized_state(
            dataset,
            statistics,
            mask,
            context_index,
        )
        states.append(state)
    if valid_mask is None:
        raise RuntimeError("Contesto della temperatura non costruito.")
    return torch.cat(states, dim=0), valid_mask


def _prepare_normalized_state(
    dataset: xr.Dataset,
    statistics: xr.Dataset,
    mask: xr.DataArray,
    time_index: int,
) -> tuple[torch.Tensor, np.ndarray]:
    """Materializza un singolo stato con forma [4, D, H, W]."""

    missing_variables = set(VOLUME_VARIABLES) - set(dataset.data_vars)
    if missing_variables:
        raise ValueError(
            f"Variabili mancanti nel dataset: {sorted(missing_variables)}"
        )

    normalized_fields: list[xr.DataArray] = []
    temperature_valid_mask: np.ndarray | None = None
    for variable in VOLUME_VARIABLES:
        mean_name = f"{variable}_mean"
        std_name = f"{variable}_std"
        if mean_name not in statistics or std_name not in statistics:
            raise ValueError(
                f"Statistiche mancanti per la variabile {variable}."
            )

        field = dataset[variable].isel(time=time_index)
        try:
            field, aligned_mask = xr.align(field, mask, join="exact")
        except ValueError as error:
            raise ValueError(
                "Le coordinate della mask non coincidono con il dataset."
            ) from error
        field = field.where(aligned_mask)
        if variable == "thetao_cglo":
            temperature_valid_mask = np.isfinite(field.values)

        normalized_fields.append(
            (field - statistics[mean_name]) / statistics[std_name]
        )

    channel = xr.IndexVariable("channel", list(VOLUME_VARIABLES))
    normalized_volume = xr.concat(
        normalized_fields,
        dim=channel,
        coords="minimal",
        compat="override",
        join="exact",
    ).transpose("channel", "depth", "latitude", "longitude")
    values = np.asarray(normalized_volume.load().values, dtype=np.float32)
    values = np.nan_to_num(
        values,
        copy=False,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if temperature_valid_mask is None:
        raise RuntimeError("Maschera della temperatura non costruita.")
    return (
        torch.from_numpy(np.ascontiguousarray(values)),
        temperature_valid_mask,
    )


def read_forecast_context_steps(checkpoint_path: Path) -> int:
    """Ricava dal checkpoint quanti giorni di input richiede il modello."""

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("Configurazione del modello assente nel checkpoint.")
    input_channels = int(model_config.get("input_channels", 0))
    if input_channels <= 0 or input_channels % len(VOLUME_VARIABLES) != 0:
        raise ValueError("Numero di canali di input non compatibile.")
    return input_channels // len(VOLUME_VARIABLES)


def run_temperature_forecast(
    checkpoint_path: Path,
    input_volume: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, int]:
    """Restituisce la media prevista normalizzata del canale temperatura."""

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint non trovato: {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("Configurazione del modello assente nel checkpoint.")

    model = VolumeUNetAutoencoder(
        VolumeAutoencoderConfig(**model_config)
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.inference_mode():
        output = model(input_volume.unsqueeze(0).to(device))
        normalized_temperature = (
            output["mean"][0, TEMPERATURE_CHANNEL]
            .float()
            .cpu()
            .numpy()
        )

    return normalized_temperature, int(checkpoint.get("epoch", -1))


def denormalize_temperature_forecast(
    normalized_temperature: np.ndarray,
    statistics: xr.Dataset,
    valid_mask: np.ndarray,
    dataset: xr.Dataset,
    forecast_date: np.datetime64,
) -> xr.DataArray:
    """Converte la media prevista dalla scala normalizzata ai gradi Celsius."""

    expected_shape = (
        dataset.sizes["depth"],
        dataset.sizes["latitude"],
        dataset.sizes["longitude"],
    )
    if normalized_temperature.shape != expected_shape:
        raise ValueError(
            "Forma della previsione inattesa: "
            f"{normalized_temperature.shape} != {expected_shape}"
        )
    if valid_mask.shape != expected_shape:
        raise ValueError("Forma della maschera valida inattesa.")

    coordinates = {
        "depth": dataset["depth"],
        "latitude": dataset["latitude"],
        "longitude": dataset["longitude"],
    }
    normalized = xr.DataArray(
        normalized_temperature,
        dims=("depth", "latitude", "longitude"),
        coords=coordinates,
    )
    mask = xr.DataArray(
        valid_mask,
        dims=("depth", "latitude", "longitude"),
        coords=coordinates,
    )
    physical = (
        normalized * statistics["thetao_cglo_std"]
        + statistics["thetao_cglo_mean"]
    ).where(mask)
    physical = physical.assign_coords(time=forecast_date)
    physical.name = "predicted_thetao_cglo"
    physical.attrs.update(statistics["thetao_cglo_mean"].attrs)
    physical.attrs["units"] = "degrees_C"
    physical.attrs["prediction"] = "forecast_mean_t_plus_1_day"
    return physical


def select_forecast_depth(
    forecast: xr.DataArray,
    requested_depth: float | None,
) -> xr.DataArray:
    if requested_depth is None:
        return forecast.isel(depth=0).load()
    return forecast.sel(depth=requested_depth, method="nearest").load()


def format_date(value: object) -> str:
    return np.datetime_as_string(
        np.asarray(value).astype("datetime64[D]"),
        unit="D",
    )


def plot_temperature_forecast(
    forecast: xr.DataArray,
    output_path: Path,
    *,
    input_date: str,
    checkpoint_epoch: int,
) -> Path:
    values = np.asarray(forecast.values, dtype=float)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        raise ValueError("La previsione non contiene temperature valide.")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    latitudes = np.asarray(forecast["latitude"].values)
    longitudes = np.asarray(forecast["longitude"].values)
    forecast_date = format_date(forecast["time"].values)
    depth = float(np.asarray(forecast["depth"].values))

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

    color_bar = figure.colorbar(mesh, ax=axis, pad=0.02)
    color_bar.set_label("Temperatura prevista (°C)")
    axis.set(
        title=(
            f"Previsione temperatura marina — {forecast_date}\n"
            f"Input {input_date} → t+1 | profondità {depth:.2f} m | "
            f"checkpoint epoca {checkpoint_epoch} | media μ"
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
    return output_path


def build_output_path(
    output_directory: Path,
    input_date: str,
    forecast: xr.DataArray,
) -> Path:
    forecast_date = format_date(forecast["time"].values)
    depth = float(np.asarray(forecast["depth"].values))
    depth_token = f"{depth:.2f}m".replace(".", "p")
    return output_directory / (
        f"temperature_forecast_input_{input_date}_target_{forecast_date}_"
        f"depth_{depth_token}.png"
    )


def main() -> None:
    args = parse_args()
    for label, path in (
        ("Dataset", args.data_path),
        ("Statistiche", args.stats_path),
        ("Checkpoint", args.checkpoint_path),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} non trovato: {path}")

    device = resolve_device(args.device)
    mask = load_land_sea_mask(args.mask_path)
    with xr.open_dataset(args.stats_path) as opened_statistics:
        statistics = opened_statistics.load()
    input_date, forecast_day = resolve_forecast_dates(
        input_date=args.input_date,
        forecast_date=args.forecast_date,
    )
    context_steps = read_forecast_context_steps(args.checkpoint_path)
    with xr.open_dataset(args.data_path, chunks={"time": 1}) as dataset:
        time_index = find_input_time_index(dataset, input_date)
        input_volume, valid_mask = prepare_normalized_input(
            dataset,
            statistics,
            mask,
            time_index,
            context_steps=context_steps,
        )
        normalized_forecast, checkpoint_epoch = run_temperature_forecast(
            args.checkpoint_path,
            input_volume,
            device,
        )
        forecast = denormalize_temperature_forecast(
            normalized_forecast,
            statistics,
            valid_mask,
            dataset,
            forecast_day,
        )

    forecast_slice = select_forecast_depth(forecast, args.depth)
    output_path = build_output_path(
        args.output_directory,
        input_date,
        forecast_slice,
    )
    plot_temperature_forecast(
        forecast_slice,
        output_path,
        input_date=input_date,
        checkpoint_epoch=checkpoint_epoch,
    )

    print(f"Device: {device}")
    print(f"Input osservato: {input_date}")
    print(f"Giorni di contesto: {context_steps}")
    print(
        "Giorno previsto: "
        f"{format_date(forecast_slice['time'].values)}"
    )
    print(
        "Profondita' selezionata: "
        f"{float(np.asarray(forecast_slice['depth'].values)):.3f} m"
    )
    print(f"Checkpoint epoca: {checkpoint_epoch}")
    print(f"Mappa della previsione salvata in: {output_path}")


if __name__ == "__main__":
    main()
