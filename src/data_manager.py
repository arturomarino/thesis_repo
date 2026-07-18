
"""
data_manager.py

Gestisce l'apertura del dataset Copernicus utilizzando Xarray e Dask.
Questo modulo è l'unico punto di accesso ai dati NetCDF.
"""

from pathlib import Path

import xarray as xr


class DataManager:
    """
    Classe responsabile dell'apertura del dataset NetCDF.

    Parameters
    ----------
    dataset_path : str | Path
        Percorso del file NetCDF.
    chunks : str | dict, optional
        Configurazione dei chunk per Dask.
        Default: "auto".
    """

    REQUIRED_VARIABLES = {
        "thetao_cglo",
        "so_cglo",
        "uo_cglo",
        "vo_cglo",
        "zos_cglo",
    }

    def __init__(self, dataset_path: str | Path, chunks="auto"):
        self.dataset_path = Path(dataset_path)
        self.chunks = chunks

    def load(self) -> xr.Dataset:
        """
        Apre il dataset in modalità lazy.

        Returns
        -------
        xr.Dataset
            Dataset aperto con Xarray.
        """
        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset non trovato: {self.dataset_path}"
            )

        dataset = xr.open_dataset(
            self.dataset_path,
            chunks=self.chunks,
        )

        self._validate_variables(dataset)

        return dataset

    def _validate_variables(self, dataset: xr.Dataset) -> None:
        """
        Verifica che tutte le variabili necessarie siano presenti.
        """
        missing = self.REQUIRED_VARIABLES - set(dataset.data_vars)

        if missing:
            raise ValueError(
                f"Variabili mancanti nel dataset: {sorted(missing)}"
            )