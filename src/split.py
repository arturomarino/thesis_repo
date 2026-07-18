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

import xarray as xr


@dataclass(frozen=True)
class TemporalSplitConfig:
    """
    Configurazione dello split temporale.

    Le frazioni devono sommare a 1.0. Lo split e' sequenziale:
    la parte iniziale diventa training set, la successiva validation set,
    la parte finale test set.
    """

    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    time_dim: str = "time"

    def __post_init__(self) -> None:
        fractions = (
            self.train_fraction,
            self.validation_fraction,
            self.test_fraction,
        )

        if any(fraction <= 0 for fraction in fractions):
            raise ValueError("Le frazioni dello split devono essere positive.")

        if not abs(sum(fractions) - 1.0) < 1e-6:
            raise ValueError("Le frazioni dello split devono sommare a 1.0.")


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

        n_time = dataset.sizes[self.config.time_dim]
        train_end = int(n_time * self.config.train_fraction)
        validation_end = train_end + int(
            n_time * self.config.validation_fraction
        )

        self._validate_boundaries(n_time, train_end, validation_end)

        train = dataset.isel({self.config.time_dim: slice(0, train_end)})
        validation = dataset.isel(
            {self.config.time_dim: slice(train_end, validation_end)}
        )
        test = dataset.isel({self.config.time_dim: slice(validation_end, None)})

        return TemporalSplits(train=train, validation=validation, test=test)

    def _validate_dataset(self, dataset: xr.Dataset) -> None:
        if self.config.time_dim not in dataset.dims:
            raise ValueError(
                f"Dimensione temporale non trovata: {self.config.time_dim}"
            )

    def _validate_boundaries(
        self,
        n_time: int,
        train_end: int,
        validation_end: int,
    ) -> None:
        if n_time < 3:
            raise ValueError(
                "Servono almeno 3 time step per creare train/validation/test."
            )

        if train_end == 0:
            raise ValueError("Training set vuoto: aumenta train_fraction.")

        if validation_end <= train_end:
            raise ValueError(
                "Validation set vuoto: aumenta validation_fraction."
            )

        if validation_end >= n_time:
            raise ValueError("Test set vuoto: aumenta test_fraction.")
