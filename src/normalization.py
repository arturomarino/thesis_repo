"""
normalization.py

Calcolo, salvataggio e applicazione della standardizzazione.

Responsabilità:
- stimare media e deviazione standard dal solo training set;
- applicare le stesse statistiche a train, validation e test;
- mantenere le operazioni lazy finché possibile.

Non esegue:
- train/validation/test split;
- conversione in NumPy o Torch.
"""

from pathlib import Path

import xarray as xr
from dask.base import is_dask_collection


class Normalizer:
    """
    Standardizza variabili Xarray usando statistiche apprese dal training set.

    Per ogni variabile, media e deviazione standard sono calcolate sulle
    dimensioni campionarie, preservando eventuali dimensioni verticali. Nel
    dataset Copernicus questo significa:
    - variabili 3D: statistiche per livello di profondità;
    - variabili 2D: statistiche scalari.
    """

    def __init__(
        self,
        sample_dims: tuple[str, ...] = ("time", "latitude", "longitude"),
        eps: float = 1e-8,
    ):
        self.sample_dims = sample_dims
        self.eps = eps
        self.statistics: dict[str, xr.DataArray] = {}

    def fit(self, dataset: xr.Dataset) -> None:
        """
        Calcola media e deviazione standard dal dataset di training.

        Le riduzioni Xarray su array Dask restano lazy: viene costruito il grafo
        di calcolo, ma i valori non vengono materializzati finché non servono.
        """

        self._validate_dataset(dataset)
        self.statistics = {}

        for variable in dataset.data_vars:
            dims = self._reduction_dims(dataset[variable])

            mean = dataset[variable].mean(dim=dims, skipna=True)
            std = dataset[variable].std(dim=dims, skipna=True)
            safe_std = std.where(std > self.eps, 1.0)

            self.statistics[f"{variable}_mean"] = mean
            self.statistics[f"{variable}_std"] = safe_std

    def transform(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        Applica la standardizzazione usando statistiche già calcolate.
        """

        if not self.statistics:
            raise ValueError("Statistiche non disponibili: eseguire fit().")

        self._validate_statistics(dataset)

        normalized = xr.Dataset(coords=dataset.coords, attrs=dataset.attrs)

        for variable in dataset.data_vars:
            mean = self.statistics[f"{variable}_mean"]
            std = self.statistics[f"{variable}_std"]
            normalized[variable] = (dataset[variable] - mean) / std

        return normalized

    def fit_transform(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        Calcola le statistiche sul training set e restituisce il train normalizzato.
        """

        self.fit(dataset)
        return self.transform(dataset)

    def materialize(self) -> None:
        """
        Calcola in memoria le statistiche gia' definite.

        Questo e' il punto intenzionale in cui la pipeline esce dalla pura
        lazy evaluation: le statistiche sono piccole, dipendono solo dal train
        set e non devono essere ricalcolate durante il caricamento dei batch.
        """

        if not self.statistics:
            raise ValueError("Statistiche non disponibili: eseguire fit().")

        stats = xr.Dataset(self.statistics)

        if any(
            is_dask_collection(variable.data)
            for variable in stats.data_vars.values()
        ):
            stats = stats.compute()

        self.statistics = {
            variable: stats[variable]
            for variable in stats.data_vars
        }

    def save(self, path: str | Path) -> None:
        """
        Salva le statistiche in NetCDF.

        Nota: il salvataggio materializza le statistiche su disco. È corretto
        farlo dopo il fit sul training set, perché le statistiche sono piccole e
        servono per rendere riproducibili training e inferenza.
        """

        if not self.statistics:
            raise ValueError("Statistiche non disponibili: eseguire fit().")

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.materialize()
        stats = xr.Dataset(self.statistics)
        stats.attrs.update(
            {
                "normalization": "z_score",
                "sample_dims": ",".join(self.sample_dims),
                "eps": self.eps,
                "fitted_on": "train_split_only",
            }
        )
        stats.to_netcdf(output_path)

    def load(self, path: str | Path) -> None:
        """Carica statistiche salvate in precedenza."""

        with xr.open_dataset(path) as opened:
            stats = opened.load()

        self.statistics = {
            variable: stats[variable]
            for variable in stats.data_vars
        }

    def _validate_dataset(self, dataset: xr.Dataset) -> None:
        missing_dims = [
            dim
            for dim in self.sample_dims
            if dim not in dataset.dims
        ]

        if missing_dims:
            raise ValueError(
                f"Dimensioni mancanti nel dataset: {missing_dims}"
            )

    def _validate_statistics(self, dataset: xr.Dataset) -> None:
        missing_statistics = []

        for variable in dataset.data_vars:
            for suffix in ("mean", "std"):
                statistic_name = f"{variable}_{suffix}"

                if statistic_name not in self.statistics:
                    missing_statistics.append(statistic_name)

        if missing_statistics:
            raise ValueError(
                f"Statistiche mancanti: {missing_statistics}"
            )

    def _reduction_dims(self, variable: xr.DataArray) -> tuple[str, ...]:
        return tuple(
            dim
            for dim in self.sample_dims
            if dim in variable.dims
        )
