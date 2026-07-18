"""
preprocessing.py

Preprocessing del dataset Copernicus.

Responsabilità:
- selezionare le variabili necessarie;
- applicare la land-sea mask;
- selezionare i livelli verticali desiderati.

Non esegue:
- normalizzazione;
- conversione in NumPy;
- conversione in Torch.
"""

from pathlib import Path

import xarray as xr


class Preprocessor:

    VARIABLES = [
        "thetao_cglo",
        "so_cglo",
        "uo_cglo",
        "vo_cglo",
        "zos_cglo",
    ]

    def __init__(self, dataset: xr.Dataset, mask_path: str | Path):
        self.dataset = dataset
        self.mask_path = Path(mask_path)

    def process(self) -> xr.Dataset:
        ds = self._select_variables()
        ds = self._apply_land_mask(ds)
        return ds

    def _select_variables(self) -> xr.Dataset:
        return self.dataset[self.VARIABLES]

    def _apply_land_mask(self, dataset: xr.Dataset) -> xr.Dataset:

        if not self.mask_path.exists():
            raise FileNotFoundError(
                f"Mask non trovata: {self.mask_path}"
            )

        mask = xr.open_dataarray(self.mask_path, chunks="auto")

        masked = xr.Dataset()

        for var in dataset.data_vars:
            masked[var] = dataset[var].where(mask)

        return masked
