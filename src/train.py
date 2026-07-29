import argparse
import tempfile
from pathlib import Path

import dask.array as da
import numpy as np
import torch
import xarray as xr
from dask.base import is_dask_collection

from data_manager import DataManager
from dataloader import DataLoaderConfig, create_ocean_dataloaders
from dataset import OceanForecastDataset, OceanStateDataset
from models.autoencoder import VolumeAutoencoderConfig, VolumeUNetAutoencoder
from normalization import Normalizer
from preprocessing import Preprocessor
from split import TemporalSplitter
from training import (
    fit_forecaster,
    read_forecaster_checkpoint,
    run_forecast_epoch,
    train_autoencoder_step,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline della tesi MedFormer.")
    project_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--data-path",
        type=Path,
        default=project_root / "data/raw/copernicus.nc",
        help="Percorso del file NetCDF Copernicus.",
    )
    parser.add_argument(
        "--mask-path",
        type=Path,
        default=project_root / "data/masks/land_sea_mask.nc",
        help="Percorso della land-sea mask NetCDF.",
    )
    parser.add_argument(
        "--stats-path",
        type=Path,
        default=project_root / "data/processed/normalization_stats.nc",
        help="Percorso di salvataggio delle statistiche di normalizzazione.",
    )
    parser.add_argument(
        "--reuse-stats",
        action="store_true",
        help="Carica statistiche gia' salvate invece di ricalcolarle.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size dei DataLoader.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Numero di worker PyTorch per il caricamento dati.",
    )
    parser.add_argument(
        "--time-chunk",
        type=int,
        default=16,
        help="Numero di giorni per chunk Dask durante preprocessing e statistiche.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Numero massimo di epoche.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Epoche senza miglioramento prima dell'early stopping.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate di AdamW.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay di AdamW.",
    )
    parser.add_argument(
        "--base-channels",
        type=int,
        default=8,
        help="Canali base della U-Net 3D.",
    )
    parser.add_argument(
        "--latent-channels",
        type=int,
        default=32,
        help="Canali del latent space.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=project_root / "checkpoints/best_forecaster.pt",
        help="Percorso del checkpoint migliore.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Device PyTorch; auto seleziona il migliore disponibile.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed PyTorch per la riproducibilita'.",
    )
    parser.add_argument(
        "--train-model",
        action="store_true",
        help="Avvia esplicitamente il training completo.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Riprende dal checkpoint dell'ultima epoca salvata.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disattiva le barre di avanzamento batch per batch.",
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Valuta sul test annuale un checkpoint gia' addestrato.",
    )
    parser.add_argument(
        "--smoke-test-dataset",
        action="store_true",
        help="Verifica il PyTorch Dataset usando dati sintetici Dask.",
    )
    parser.add_argument(
        "--smoke-test-normalization",
        action="store_true",
        help="Verifica fit/save/load della normalizzazione su dati sintetici.",
    )
    parser.add_argument(
        "--smoke-test-dataloader",
        action="store_true",
        help="Verifica i DataLoader usando dati sintetici Dask.",
    )
    parser.add_argument(
        "--smoke-test-autoencoder",
        action="store_true",
        help="Verifica forward pass e latent space dell'autoencoder.",
    )
    parser.add_argument(
        "--smoke-test-training",
        action="store_true",
        help="Verifica un training step autoencoder su dati sintetici.",
    )
    parser.add_argument(
        "--first-real-training-step",
        action="store_true",
        help="Esegue un solo training step autoencoder sul primo batch reale.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.smoke_test_dataset:
        smoke_test_dataset()
        return

    if args.smoke_test_normalization:
        smoke_test_normalization()
        return

    if args.smoke_test_dataloader:
        smoke_test_dataloader()
        return

    if args.smoke_test_autoencoder:
        smoke_test_autoencoder()
        return

    if args.smoke_test_training:
        smoke_test_training()
        return

    device = resolve_device(args.device)
    torch.manual_seed(args.seed)

    if args.time_chunk <= 0:
        raise ValueError("time_chunk deve essere positivo.")

    dm = DataManager(
        args.data_path,
        chunks={"time": args.time_chunk},
    )
    ds = dm.load()

    preprocessor = Preprocessor(
        ds,
        args.mask_path,
    )

    ds = preprocessor.process()

    splitter = TemporalSplitter()
    splits = splitter.split(ds)

    print("Temporal split completato.")
    print(
        "Training: anni precedenti; validation: penultimo anno; "
        "test: ultimo anno."
    )
    print(f"Train time steps: {splits.train.sizes['time']}")
    print(f"Validation time steps: {splits.validation.sizes['time']}")
    print(f"Test time steps: {splits.test.sizes['time']}")

    normalizer = Normalizer(scheduler="single-threaded")
    if args.reuse_stats:
        if not args.stats_path.exists():
            raise FileNotFoundError(
                "Statistiche non trovate: eseguire prima la preparazione "
                "senza --reuse-stats."
            )
        normalizer.load(args.stats_path)
        print(f"Statistiche caricate da: {args.stats_path}")
    else:
        print("Calcolo delle statistiche di normalizzazione sul training set...")
        normalizer.fit(splits.train)
        normalizer.save(args.stats_path)
        normalizer.load(args.stats_path)

    normalized_train = normalizer.transform(splits.train)
    normalized_validation = normalizer.transform(splits.validation)
    normalized_test = normalizer.transform(splits.test)

    print("Normalizzazione configurata.")
    print(f"Statistiche calcolate: {len(normalizer.statistics)}")
    print(f"Statistiche salvate in: {args.stats_path}")
    print(f"Train normalizzato: {list(normalized_train.data_vars)}")
    print(f"Validation normalizzato: {list(normalized_validation.data_vars)}")
    print(f"Test normalizzato: {list(normalized_test.data_vars)}")

    train_dataset = OceanForecastDataset(normalized_train)
    validation_dataset = OceanForecastDataset(normalized_validation)
    test_dataset = OceanForecastDataset(normalized_test)
    loaders = create_ocean_dataloaders(
        train_dataset,
        validation_dataset,
        test_dataset,
        DataLoaderConfig(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        ),
    )

    print("Dataset previsionali t -> t+1 configurati in modo lazy.")
    print(f"Coppie train: {len(train_dataset)}")
    print(f"Coppie validation: {len(validation_dataset)}")
    print(f"Coppie test: {len(test_dataset)}")
    print(f"Forma volume: {train_dataset.volume_shape}")
    print("DataLoader configurati.")
    print(f"Batch size train: {loaders.train.batch_size}")
    print(f"Num workers train: {loaders.train.num_workers}")
    print(f"Device selezionato: {device}")

    if args.first_real_training_step:
        run_first_real_training_step(loaders.train)

    if args.train_model:
        run_full_training(args, loaders.train, loaders.validation, device)

    if args.evaluate_test:
        evaluate_test_checkpoint(args, loaders.test, device)


def resolve_device(requested_device: str) -> torch.device:
    """Seleziona il device richiesto verificandone la disponibilita'."""

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


def build_forecaster(args: argparse.Namespace, device: torch.device):
    """Costruisce il previsore probabilistico configurato dalla CLI."""

    return VolumeUNetAutoencoder(
        VolumeAutoencoderConfig(
            input_channels=4,
            output_channels=4,
            base_channels=args.base_channels,
            latent_channels=args.latent_channels,
        )
    ).to(device)


def run_full_training(
    args: argparse.Namespace,
    train_loader,
    validation_loader,
    device: torch.device,
) -> None:
    """Avvia il training soltanto quando e' presente ``--train-model``."""

    model = build_forecaster(args, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    resume_checkpoint = None
    if args.resume:
        last_checkpoint_path = args.checkpoint_path.with_name(
            f"{args.checkpoint_path.stem}_last"
            f"{args.checkpoint_path.suffix}"
        )
        resume_checkpoint = read_forecaster_checkpoint(
            last_checkpoint_path,
            device,
        )
        saved_config = resume_checkpoint.get("model_config")
        if saved_config != {
            "input_channels": 4,
            "output_channels": 4,
            "base_channels": args.base_channels,
            "latent_channels": args.latent_channels,
        }:
            raise ValueError(
                "La configurazione richiesta non coincide con il checkpoint."
            )
        model.load_state_dict(resume_checkpoint["model_state_dict"])
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        print(f"Ripresa dall'epoca {resume_checkpoint['epoch']}.")

    result = fit_forecaster(
        model=model,
        train_batches=train_loader,
        validation_batches=validation_loader,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
        patience=args.patience,
        checkpoint_path=args.checkpoint_path,
        resume_checkpoint=resume_checkpoint,
        show_progress=not args.no_progress,
    )

    print("Training completato.")
    print(f"Epoca migliore: {result.best_epoch}")
    print(f"Validation NLL migliore: {result.best_validation_nll:.6f}")
    print(f"Checkpoint: {result.checkpoint_path}")
    print(f"Checkpoint di ripresa: {result.last_checkpoint_path}")


def evaluate_test_checkpoint(
    args: argparse.Namespace,
    test_loader,
    device: torch.device,
) -> None:
    """Valuta il test set solo su richiesta esplicita."""

    checkpoint = read_forecaster_checkpoint(args.checkpoint_path, device)
    saved_config = checkpoint.get("model_config")
    if not isinstance(saved_config, dict):
        raise ValueError("Configurazione del modello assente nel checkpoint.")

    model = VolumeUNetAutoencoder(
        VolumeAutoencoderConfig(**saved_config)
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics = run_forecast_epoch(
        model=model,
        batches=test_loader,
        device=device,
    )

    print(f"Checkpoint epoca: {checkpoint['epoch']}")
    print(f"Test Gaussian NLL: {metrics.gaussian_nll:.6f}")
    print(f"Test RMSE: {metrics.rmse:.6f}")
    print(
        "Test coverage 68/95: "
        f"{metrics.coverage_68:.3f}/{metrics.coverage_95:.3f}"
    )


def smoke_test_dataset() -> None:
    """Verifica il contratto del Dataset senza leggere il file da 18 GB."""

    n_time, n_depth, n_latitude, n_longitude = 5, 2, 3, 4
    volume_shape = (n_time, n_depth, n_latitude, n_longitude)
    surface_shape = (n_time, n_latitude, n_longitude)

    volume_data = np.arange(
        np.prod(volume_shape), dtype=np.float32
    ).reshape(volume_shape)
    surface_data = np.arange(
        np.prod(surface_shape), dtype=np.float32
    ).reshape(surface_shape)
    volume_data[:, :, 0, 0] = np.nan
    surface_data[:, 0, 0] = np.nan

    coords = {
        "time": np.arange(
            np.datetime64("2024-01-01"),
            np.datetime64("2024-01-01") + np.timedelta64(n_time, "D"),
        ),
        "depth": np.arange(n_depth),
        "latitude": np.arange(n_latitude),
        "longitude": np.arange(n_longitude),
    }
    dataset = xr.Dataset(
        {
            variable: (
                ("time", "depth", "latitude", "longitude"),
                da.from_array(volume_data, chunks=(1, -1, -1, -1)),
            )
            for variable in OceanStateDataset.DEFAULT_VOLUME_VARIABLES
        }
        | {
            OceanStateDataset.DEFAULT_SURFACE_VARIABLE: (
                ("time", "latitude", "longitude"),
                da.from_array(surface_data, chunks=(1, -1, -1)),
            )
        },
        coords=coords,
    )

    ocean_dataset = OceanStateDataset(dataset)
    sample = ocean_dataset[0]
    forecast_dataset = OceanForecastDataset(dataset)
    forecast_sample = forecast_dataset[0]

    assert len(ocean_dataset) == n_time
    assert len(forecast_dataset) == n_time - 1
    assert sample["state"]["volume"].shape == (4, 2, 3, 4)
    assert sample["state"]["surface"].shape == (1, 3, 4)
    assert sample["time_index"] == 0
    assert torch.isfinite(sample["state"]["volume"]).all()
    assert not sample["state"]["volume_mask"][0, 0, 0, 0]
    assert sample["state"]["volume"][0, 0, 0, 0] == 0
    assert forecast_sample["input_time_index"] == 0
    assert forecast_sample["target_time_index"] == 1
    assert torch.equal(
        forecast_sample["input"]["volume"],
        ocean_dataset[0]["state"]["volume"],
    )
    assert torch.equal(
        forecast_sample["target"]["volume"],
        ocean_dataset[1]["state"]["volume"],
    )

    print("Smoke test PyTorch Dataset superato.")
    print(f"Campioni: {len(ocean_dataset)}")
    print(f"Volume: {tuple(sample['state']['volume'].shape)}")
    print(f"Superficie: {tuple(sample['state']['surface'].shape)}")
    print("NaN mascherati e sostituiti con zero: OK")
    print("Coppia previsionale t -> t+1: OK")


def smoke_test_normalization() -> None:
    """Verifica la normalizzazione senza leggere il file Copernicus."""

    coords = {
        "time": np.arange(4),
        "depth": np.arange(2),
        "latitude": np.arange(2),
        "longitude": np.arange(3),
    }
    volume = np.arange(48, dtype=np.float32).reshape(4, 2, 2, 3)
    surface = np.arange(24, dtype=np.float32).reshape(4, 2, 3)
    volume[:, :, 0, 0] = np.nan
    surface[:, 0, 0] = np.nan

    dataset = xr.Dataset(
        {
            "thetao_cglo": (
                ("time", "depth", "latitude", "longitude"),
                da.from_array(volume, chunks=(1, -1, -1, -1)),
            ),
            "zos_cglo": (
                ("time", "latitude", "longitude"),
                da.from_array(surface, chunks=(1, -1, -1)),
            ),
        },
        coords=coords,
    )

    train = dataset.isel(time=slice(0, 3))
    validation = dataset.isel(time=slice(3, None))

    normalizer = Normalizer()
    normalizer.fit(train)

    with tempfile.TemporaryDirectory() as temporary_directory:
        stats_path = Path(temporary_directory) / "normalization_stats.nc"
        normalizer.save(stats_path)

        loaded = Normalizer()
        loaded.load(stats_path)

    normalized_train = loaded.transform(train)
    normalized_validation = loaded.transform(validation)

    assert set(loaded.statistics) == {
        "thetao_cglo_mean",
        "thetao_cglo_std",
        "zos_cglo_mean",
        "zos_cglo_std",
    }
    assert loaded.statistics["thetao_cglo_mean"].shape == (2,)
    assert loaded.statistics["zos_cglo_mean"].shape == ()
    assert not any(
        is_dask_collection(statistic.data)
        for statistic in loaded.statistics.values()
    )
    assert "time" in normalized_train.dims
    assert "time" in normalized_validation.dims

    print("Smoke test normalizzazione superato.")
    print(f"Statistiche salvate e ricaricate: {len(loaded.statistics)}")
    print("Statistiche 3D per profondita': OK")
    print("Statistiche 2D scalari: OK")


def smoke_test_dataloader() -> None:
    """Verifica il batching PyTorch senza leggere il file Copernicus."""

    n_time, n_depth, n_latitude, n_longitude = 8, 2, 3, 4
    volume_shape = (n_time, n_depth, n_latitude, n_longitude)
    surface_shape = (n_time, n_latitude, n_longitude)

    volume_data = np.arange(
        np.prod(volume_shape), dtype=np.float32
    ).reshape(volume_shape)
    surface_data = np.arange(
        np.prod(surface_shape), dtype=np.float32
    ).reshape(surface_shape)
    volume_data[:, :, 0, 0] = np.nan
    surface_data[:, 0, 0] = np.nan

    coords = {
        "time": np.arange(
            np.datetime64("2024-01-01"),
            np.datetime64("2024-01-01") + np.timedelta64(n_time, "D"),
        ),
        "depth": np.arange(n_depth),
        "latitude": np.arange(n_latitude),
        "longitude": np.arange(n_longitude),
    }
    dataset = xr.Dataset(
        {
            variable: (
                ("time", "depth", "latitude", "longitude"),
                da.from_array(volume_data, chunks=(1, -1, -1, -1)),
            )
            for variable in OceanStateDataset.DEFAULT_VOLUME_VARIABLES
        }
        | {
            OceanStateDataset.DEFAULT_SURFACE_VARIABLE: (
                ("time", "latitude", "longitude"),
                da.from_array(surface_data, chunks=(1, -1, -1)),
            )
        },
        coords=coords,
    )

    train_dataset = OceanForecastDataset(dataset.isel(time=slice(0, 4)))
    validation_dataset = OceanForecastDataset(dataset.isel(time=slice(4, 6)))
    test_dataset = OceanForecastDataset(dataset.isel(time=slice(6, 8)))

    loaders = create_ocean_dataloaders(
        train_dataset,
        validation_dataset,
        test_dataset,
        DataLoaderConfig(batch_size=2, num_workers=0),
    )

    train_batch = next(iter(loaders.train))
    validation_batch = next(iter(loaders.validation))
    test_batch = next(iter(loaders.test))

    assert train_batch["input"]["volume"].shape == (2, 4, 2, 3, 4)
    assert train_batch["target"]["volume"].shape == (2, 4, 2, 3, 4)
    assert train_batch["input"]["volume_mask"].shape == (2, 4, 2, 3, 4)
    assert train_batch["target"]["volume_mask"].shape == (2, 4, 2, 3, 4)
    assert validation_batch["input_time_index"].tolist() == [0]
    assert validation_batch["target_time_index"].tolist() == [1]
    assert test_batch["input_time_index"].tolist() == [0]
    assert test_batch["target_time_index"].tolist() == [1]
    assert loaders.train.batch_size == 2
    assert loaders.train.num_workers == 0

    print("Smoke test DataLoader superato.")
    print(f"Input train: {tuple(train_batch['input']['volume'].shape)}")
    print(f"Target t+1: {tuple(train_batch['target']['volume'].shape)}")
    print("Validation/test senza shuffle: OK")


def smoke_test_autoencoder() -> None:
    """Verifica forme dei parametri probabilistici e latent space."""

    batch_size, channels, depth, height, width = 1, 4, 46, 65, 171
    model = VolumeUNetAutoencoder(
        VolumeAutoencoderConfig(
            input_channels=channels,
            output_channels=channels,
            base_channels=4,
            latent_channels=16,
        )
    )
    x = torch.randn(batch_size, channels, depth, height, width)

    with torch.no_grad():
        output = model(x)

    mean = output["mean"]
    log_variance = output["log_variance"]
    latent = output["latent"]

    assert mean.shape == x.shape
    assert log_variance.shape == x.shape
    assert torch.all(log_variance == 0)
    assert latent.ndim == 5
    assert latent.shape[0] == batch_size
    assert latent.shape[1] == 16
    assert latent.shape[2] < depth
    assert latent.shape[3] < height
    assert latent.shape[4] < width

    print("Smoke test Autoencoder superato.")
    print(f"Input volume: {tuple(x.shape)}")
    print(f"Latent space: {tuple(latent.shape)}")
    print(f"Media predetta: {tuple(mean.shape)}")
    print(f"Log-varianza predetta: {tuple(log_variance.shape)}")


def smoke_test_training() -> None:
    """Verifica che loss, backward e optimizer step funzionino."""

    batch_size, channels, depth, height, width = 2, 4, 8, 10, 12
    input_volume = torch.randn(batch_size, channels, depth, height, width)
    target_volume = torch.randn(batch_size, channels, depth, height, width)
    mask = torch.ones_like(target_volume, dtype=torch.bool)
    mask[:, :, :, 0, 0] = False
    target_volume = target_volume.masked_fill(~mask, 0.0)

    batch = {
        "input": {
            "volume": input_volume,
            "volume_mask": torch.ones_like(input_volume, dtype=torch.bool),
        },
        "target": {
            "volume": target_volume,
            "volume_mask": mask,
        },
        "input_time_index": torch.arange(batch_size),
        "target_time_index": torch.arange(1, batch_size + 1),
    }

    model = VolumeUNetAutoencoder(
        VolumeAutoencoderConfig(
            input_channels=channels,
            output_channels=channels,
            base_channels=4,
            latent_channels=8,
        )
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    metrics = train_autoencoder_step(
        model=model,
        batch=batch,
        optimizer=optimizer,
        device=torch.device("cpu"),
    )

    assert np.isfinite(metrics.loss)
    assert metrics.mean_mse > 0
    assert metrics.valid_points == int(mask.sum())
    assert metrics.latent_shape[0] == batch_size
    assert metrics.latent_shape[1] == 8

    print("Smoke test training autoencoder superato.")
    print(f"Gaussian NLL: {metrics.loss:.6f}")
    print(f"MSE della media: {metrics.mean_mse:.6f}")
    print(f"Punti validi: {metrics.valid_points}")
    print(f"Latent space: {metrics.latent_shape}")


def run_first_real_training_step(train_loader) -> None:
    """Esegue un solo update sul primo batch reale."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VolumeUNetAutoencoder(
        VolumeAutoencoderConfig(
            input_channels=4,
            output_channels=4,
            base_channels=4,
            latent_channels=16,
        )
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    batch = next(iter(train_loader))
    metrics = train_autoencoder_step(
        model=model,
        batch=batch,
        optimizer=optimizer,
        device=device,
    )

    print("Primo training step reale completato.")
    print(f"Device: {device}")
    print(f"Gaussian NLL: {metrics.loss:.6f}")
    print(f"MSE della media: {metrics.mean_mse:.6f}")
    print(f"Punti oceanici validi: {metrics.valid_points}")
    print(f"Latent space: {metrics.latent_shape}")


if __name__ == "__main__":
    main()
