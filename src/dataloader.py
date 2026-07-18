"""
dataloader.py

Costruzione dei DataLoader PyTorch per i dataset oceanografici.

Responsabilita':
- raggruppare campioni PyTorch in batch;
- configurare shuffle, batch size e worker;
- mantenere separati train, validation e test.

Non esegue:
- apertura del file NetCDF;
- preprocessing, split o normalizzazione;
- materializzazione diretta dei dati Xarray.
"""

from dataclasses import dataclass

from torch.utils.data import DataLoader

from dataset import OceanStateDataset, OceanStateSample


@dataclass(frozen=True)
class DataLoaderConfig:
    """
    Configurazione dei DataLoader.

    Il valore iniziale conservativo e' batch_size=1 e num_workers=0. Con
    Xarray, Dask e NetCDF conviene prima garantire correttezza e stabilita';
    l'ottimizzazione parallela va fatta dopo con misure esplicite.
    """

    batch_size: int = 1
    num_workers: int = 0
    pin_memory: bool = False
    drop_last: bool = False

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size deve essere positivo.")

        if self.num_workers < 0:
            raise ValueError("num_workers non puo' essere negativo.")


@dataclass(frozen=True)
class OceanDataLoaders:
    """Contenitore tipizzato per i tre DataLoader della pipeline."""

    train: DataLoader[OceanStateSample]
    validation: DataLoader[OceanStateSample]
    test: DataLoader[OceanStateSample]


def create_ocean_dataloaders(
    train_dataset: OceanStateDataset,
    validation_dataset: OceanStateDataset,
    test_dataset: OceanStateDataset,
    config: DataLoaderConfig | None = None,
) -> OceanDataLoaders:
    """Crea DataLoader coerenti per train, validation e test."""

    config = config or DataLoaderConfig()

    train_loader = _create_loader(
        train_dataset,
        config=config,
        shuffle=True,
    )
    validation_loader = _create_loader(
        validation_dataset,
        config=config,
        shuffle=False,
    )
    test_loader = _create_loader(
        test_dataset,
        config=config,
        shuffle=False,
    )

    return OceanDataLoaders(
        train=train_loader,
        validation=validation_loader,
        test=test_loader,
    )


def _create_loader(
    dataset: OceanStateDataset,
    config: DataLoaderConfig,
    shuffle: bool,
) -> DataLoader[OceanStateSample]:
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=config.drop_last,
    )
