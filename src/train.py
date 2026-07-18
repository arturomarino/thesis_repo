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
from dataset import OceanStateDataset
from models.autoencoder import VolumeAutoencoderConfig, VolumeUNetAutoencoder
from normalization import Normalizer
from preprocessing import Preprocessor
from split import TemporalSplitter
from training import train_autoencoder_step


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

    dm = DataManager(
        args.data_path,
        chunks={"time": 1},
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
    print(f"Train time steps: {splits.train.sizes['time']}")
    print(f"Validation time steps: {splits.validation.sizes['time']}")
    print(f"Test time steps: {splits.test.sizes['time']}")

    normalizer = Normalizer()
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

    train_dataset = OceanStateDataset(normalized_train)
    validation_dataset = OceanStateDataset(normalized_validation)
    test_dataset = OceanStateDataset(normalized_test)
    loaders = create_ocean_dataloaders(
        train_dataset,
        validation_dataset,
        test_dataset,
        DataLoaderConfig(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        ),
    )

    print("PyTorch Dataset configurati in modo lazy.")
    print(f"Campioni train: {len(train_dataset)}")
    print(f"Campioni validation: {len(validation_dataset)}")
    print(f"Campioni test: {len(test_dataset)}")
    print(f"Forma volume: {train_dataset.volume_shape}")
    print(f"Forma superficie: {train_dataset.surface_shape}")
    print("DataLoader configurati.")
    print(f"Batch size train: {loaders.train.batch_size}")
    print(f"Num workers train: {loaders.train.num_workers}")

    if args.first_real_training_step:
        run_first_real_training_step(loaders.train)


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
        "time": np.arange(n_time),
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

    assert len(ocean_dataset) == n_time
    assert sample["state"]["volume"].shape == (4, 2, 3, 4)
    assert sample["state"]["surface"].shape == (1, 3, 4)
    assert sample["time_index"] == 0
    assert torch.isfinite(sample["state"]["volume"]).all()
    assert not sample["state"]["volume_mask"][0, 0, 0, 0]
    assert sample["state"]["volume"][0, 0, 0, 0] == 0

    print("Smoke test PyTorch Dataset superato.")
    print(f"Campioni: {len(ocean_dataset)}")
    print(f"Volume: {tuple(sample['state']['volume'].shape)}")
    print(f"Superficie: {tuple(sample['state']['surface'].shape)}")
    print("NaN mascherati e sostituiti con zero: OK")


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

    n_time, n_depth, n_latitude, n_longitude = 6, 2, 3, 4
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
        "time": np.arange(n_time),
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

    train_dataset = OceanStateDataset(dataset.isel(time=slice(0, 4)))
    validation_dataset = OceanStateDataset(dataset.isel(time=slice(4, 5)))
    test_dataset = OceanStateDataset(dataset.isel(time=slice(5, 6)))

    loaders = create_ocean_dataloaders(
        train_dataset,
        validation_dataset,
        test_dataset,
        DataLoaderConfig(batch_size=2, num_workers=0),
    )

    train_batch = next(iter(loaders.train))
    validation_batch = next(iter(loaders.validation))
    test_batch = next(iter(loaders.test))

    assert train_batch["state"]["volume"].shape == (2, 4, 2, 3, 4)
    assert train_batch["state"]["surface"].shape == (2, 1, 3, 4)
    assert train_batch["state"]["volume_mask"].shape == (2, 4, 2, 3, 4)
    assert train_batch["state"]["surface_mask"].shape == (2, 1, 3, 4)
    assert validation_batch["time_index"].tolist() == [0]
    assert test_batch["time_index"].tolist() == [0]
    assert loaders.train.batch_size == 2
    assert loaders.train.num_workers == 0

    print("Smoke test DataLoader superato.")
    print(f"Batch volume train: {tuple(train_batch['state']['volume'].shape)}")
    print(f"Batch superficie train: {tuple(train_batch['state']['surface'].shape)}")
    print("Validation/test senza shuffle: OK")


def smoke_test_autoencoder() -> None:
    """Verifica forme di ricostruzione e latent space."""

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

    reconstruction = output["reconstruction"]
    latent = output["latent"]

    assert reconstruction.shape == x.shape
    assert latent.ndim == 5
    assert latent.shape[0] == batch_size
    assert latent.shape[1] == 16
    assert latent.shape[2] < depth
    assert latent.shape[3] < height
    assert latent.shape[4] < width

    print("Smoke test Autoencoder superato.")
    print(f"Input volume: {tuple(x.shape)}")
    print(f"Latent space: {tuple(latent.shape)}")
    print(f"Ricostruzione: {tuple(reconstruction.shape)}")


def smoke_test_training() -> None:
    """Verifica che loss, backward e optimizer step funzionino."""

    batch_size, channels, depth, height, width = 2, 4, 8, 10, 12
    volume = torch.randn(batch_size, channels, depth, height, width)
    mask = torch.ones_like(volume, dtype=torch.bool)
    mask[:, :, :, 0, 0] = False
    volume = volume.masked_fill(~mask, 0.0)

    batch = {
        "state": {
            "volume": volume,
            "volume_mask": mask,
            "surface": torch.zeros(batch_size, 1, height, width),
            "surface_mask": torch.ones(batch_size, 1, height, width, dtype=torch.bool),
        },
        "time_index": torch.arange(batch_size),
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

    assert metrics.loss > 0
    assert metrics.valid_points == int(mask.sum())
    assert metrics.latent_shape[0] == batch_size
    assert metrics.latent_shape[1] == 8

    print("Smoke test training autoencoder superato.")
    print(f"Loss: {metrics.loss:.6f}")
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
    print(f"Loss: {metrics.loss:.6f}")
    print(f"Punti oceanici validi: {metrics.valid_points}")
    print(f"Latent space: {metrics.latent_shape}")


if __name__ == "__main__":
    main()
