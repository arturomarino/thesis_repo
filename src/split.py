"""
split.py

Train/validation/test split temporale per dataset Xarray.

Responsabilità:
- suddividere il dataset lungo la dimensione temporale;
- preservare l'ordine cronologico dei dati;
- mantenere le operazioni lazy tramite slicing Xarray.

Non esegue:
- normalizzazione;
- generazione di finestre input/target;
- conversione in NumPy o Torch.
"""

from dataclasses import dataclass

import numpy as np
import xarray as xr


@dataclass(frozen=True)
class TemporalSplitConfig:
    """
    Configurazione dello split temporale per anni di calendario.

    L'ultimo anno disponibile viene riservato al test, il penultimo alla
    validation e tutti gli anni precedenti al training.
    """

    time_dim: str = "time"


@dataclass(frozen=True)
class TemporalSplits:
    """Contenitore tipizzato per i tre subset temporali."""

    train: xr.Dataset
    validation: xr.Dataset
    test: xr.Dataset


class TemporalSplitter:
    """
    Crea split temporali non randomici da un dataset Xarray.

    Per serie temporali e problemi di previsione, lo split randomico introduce
    una valutazione troppo ottimistica: dati temporalmente vicini possono finire
    in subset diversi. Lo split temporale simula invece il caso reale in cui il
    modello viene addestrato sul passato e valutato sul futuro.
    """

    def __init__(self, config: TemporalSplitConfig | None = None):
        self.config = config or TemporalSplitConfig()

    def split(self, dataset: xr.Dataset) -> TemporalSplits:
        """Restituisce train, validation e test split in modo lazy."""

        self._validate_dataset(dataset)

        years = self._extract_years(dataset)
        unique_years = np.unique(years)
        self._validate_years(unique_years)

        validation_year = int(unique_years[-2])
        test_year = int(unique_years[-1])

        train = dataset.isel({self.config.time_dim: years < validation_year})
        validation = dataset.isel(
            {self.config.time_dim: years == validation_year}
        )
        test = dataset.isel({self.config.time_dim: years == test_year})

        return TemporalSplits(train=train, validation=validation, test=test)

    def _validate_dataset(self, dataset: xr.Dataset) -> None:
        if self.config.time_dim not in dataset.dims:
            raise ValueError(
                f"Dimensione temporale non trovata: {self.config.time_dim}"
            )

    def _extract_years(self, dataset: xr.Dataset) -> np.ndarray:
        try:
            years = dataset[self.config.time_dim].dt.year.values
        except (AttributeError, TypeError) as error:
            raise ValueError(
                "La coordinata temporale deve contenere date interpretabili."
            ) from error

        return np.asarray(years)

    @staticmethod
    def _validate_years(unique_years: np.ndarray) -> None:
        if unique_years.size < 3:
            raise ValueError(
                "Servono almeno tre anni distinti: training, validation e test."
            )

        if int(unique_years[-1]) - int(unique_years[-2]) != 1:
            raise ValueError(
                "Gli ultimi due anni disponibili devono essere consecutivi."
            )
