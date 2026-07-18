"""
dataset.py

Conversione lazy dei dataset Xarray normalizzati in campioni PyTorch.

Responsabilita':
- rappresentare separatamente volume oceanico e superficie;
- materializzare soltanto lo stato temporale richiesto;
- conservare una maschera dei valori oceanici validi.

Non esegue:
- split o normalizzazione;
- batching;
- creazione di coppie temporali per il diffusion model;
- training del modello.
"""

from dataclasses import dataclass
from typing import TypedDict

import numpy as np
import torch
import xarray as xr
from dask.base import is_dask_collection
from torch.utils.data import Dataset


class OceanState(TypedDict):
    """Tensori che descrivono uno stato oceanico."""

    volume: torch.Tensor
    volume_mask: torch.Tensor
    surface: torch.Tensor
    surface_mask: torch.Tensor


class OceanStateSample(TypedDict):
    """Singolo stato restituito dal dataset."""

    state: OceanState
    time_index: int


@dataclass(frozen=True)
class OceanDatasetConfig:
    """Nomi delle dimensioni usate dal dataset Copernicus."""

    time_dim: str = "time"
    depth_dim: str = "depth"
    latitude_dim: str = "latitude"
    longitude_dim: str = "longitude"


class OceanStateDataset(Dataset[OceanStateSample]):
    """
    Espone stati Xarray/Dask indipendenti come campioni PyTorch.

    Le variabili volumetriche sono impilate come canali con forma
    ``[canali, profondita, latitudine, longitudine]``. La variabile
    ``zos_cglo`` resta un campo di superficie con forma
    ``[1, latitudine, longitudine]``: replicarla lungo la profondita'
    introdurrebbe informazione artificiale.

    Questa rappresentazione e' adatta alla prima fase della tesi, nella quale
    l'autoencoder ricostruisce lo stesso stato ricevuto in ingresso. Le future
    coppie temporali del diffusion model saranno responsabilita' di un dataset
    dedicato.

    I NaN prodotti dalla land-sea mask vengono sostituiti con zero soltanto nel
    tensore materializzato. Le corrispondenti maschere booleane permettono al
    modello e alla loss di distinguere terra e oceano.
    """

    DEFAULT_VOLUME_VARIABLES = (
        "thetao_cglo",
        "so_cglo",
        "uo_cglo",
        "vo_cglo",
    )
    DEFAULT_SURFACE_VARIABLE = "zos_cglo"

    def __init__(
        self,
        dataset: xr.Dataset,
        config: OceanDatasetConfig | None = None,
        volume_variables: tuple[str, ...] = DEFAULT_VOLUME_VARIABLES,
        surface_variable: str = DEFAULT_SURFACE_VARIABLE,
    ) -> None:
        self.dataset = dataset
        self.config = config or OceanDatasetConfig()
        self.volume_variables = volume_variables
        self.surface_variable = surface_variable

        self._validate_dataset()
        self._volume = self._build_volume_array()
        self._surface = self._build_surface_array()

    def __len__(self) -> int:
        return self.dataset.sizes[self.config.time_dim]

    def __getitem__(self, index: int) -> OceanStateSample:
        index = self._normalize_index(index)
        state = self._materialize_state(index)
        return {"state": state, "time_index": index}

    @property
    def volume_shape(self) -> tuple[int, int, int, int]:
        """Forma del tensore volumetrico di un campione."""

        return (
            len(self.volume_variables),
            self.dataset.sizes[self.config.depth_dim],
            self.dataset.sizes[self.config.latitude_dim],
            self.dataset.sizes[self.config.longitude_dim],
        )

    @property
    def surface_shape(self) -> tuple[int, int, int]:
        """Forma del tensore superficiale di un campione."""

        return (
            1,
            self.dataset.sizes[self.config.latitude_dim],
            self.dataset.sizes[self.config.longitude_dim],
        )

    def _build_volume_array(self) -> xr.DataArray:
        channel = xr.IndexVariable("channel", list(self.volume_variables))
        volume = xr.concat(
            [self.dataset[name] for name in self.volume_variables],
            dim=channel,
            coords="minimal",
            compat="override",
            join="exact",
        )
        return volume.transpose(
            self.config.time_dim,
            "channel",
            self.config.depth_dim,
            self.config.latitude_dim,
            self.config.longitude_dim,
        )

    def _build_surface_array(self) -> xr.DataArray:
        surface = self.dataset[self.surface_variable].expand_dims(
            channel=[self.surface_variable]
        )
        return surface.transpose(
            self.config.time_dim,
            "channel",
            self.config.latitude_dim,
            self.config.longitude_dim,
        )

    def _materialize_state(self, time_index: int) -> OceanState:
        volume, volume_mask = self._to_tensor_pair(
            self._volume.isel({self.config.time_dim: time_index})
        )
        surface, surface_mask = self._to_tensor_pair(
            self._surface.isel({self.config.time_dim: time_index})
        )

        return {
            "volume": volume,
            "volume_mask": volume_mask,
            "surface": surface,
            "surface_mask": surface_mask,
        }

    @staticmethod
    def _to_tensor_pair(array: xr.DataArray) -> tuple[torch.Tensor, torch.Tensor]:
        data = array.data

        # Confine intenzionale tra pipeline lazy e PyTorch: viene calcolato
        # esclusivamente lo stato richiesto da __getitem__.
        if is_dask_collection(data):
            data = data.compute()

        numpy_data = np.asarray(data, dtype=np.float32)
        valid_mask = np.isfinite(numpy_data)
        clean_data = np.nan_to_num(
            numpy_data,
            copy=False,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        values = torch.from_numpy(np.ascontiguousarray(clean_data))
        mask = torch.from_numpy(np.ascontiguousarray(valid_mask))
        return values, mask

    def _normalize_index(self, index: int) -> int:
        if not isinstance(index, int) or isinstance(index, bool):
            raise TypeError("L'indice del campione deve essere un intero.")

        length = len(self)
        if index < 0:
            index += length

        if index < 0 or index >= length:
            raise IndexError(
                f"Indice {index} fuori intervallo per {length} campioni."
            )

        return index

    def _validate_dataset(self) -> None:
        if not self.volume_variables:
            raise ValueError("Specificare almeno una variabile volumetrica.")

        required_variables = set(self.volume_variables) | {
            self.surface_variable
        }
        missing_variables = required_variables - set(self.dataset.data_vars)

        if missing_variables:
            raise ValueError(
                f"Variabili mancanti: {sorted(missing_variables)}"
            )

        volume_dims = {
            self.config.time_dim,
            self.config.depth_dim,
            self.config.latitude_dim,
            self.config.longitude_dim,
        }
        surface_dims = {
            self.config.time_dim,
            self.config.latitude_dim,
            self.config.longitude_dim,
        }

        for variable in self.volume_variables:
            actual_dims = set(self.dataset[variable].dims)
            if actual_dims != volume_dims:
                raise ValueError(
                    f"{variable} deve avere dimensioni {sorted(volume_dims)}, "
                    f"trovate {list(self.dataset[variable].dims)}."
                )

        actual_surface_dims = set(self.dataset[self.surface_variable].dims)
        if actual_surface_dims != surface_dims:
            raise ValueError(
                f"{self.surface_variable} deve avere dimensioni "
                f"{sorted(surface_dims)}, trovate "
                f"{list(self.dataset[self.surface_variable].dims)}."
            )

        time_index = self.dataset.indexes.get(self.config.time_dim)
        if time_index is None:
            raise ValueError(
                f"Coordinata temporale non trovata: {self.config.time_dim}."
            )

        if not time_index.is_monotonic_increasing or not time_index.is_unique:
            raise ValueError(
                "La coordinata temporale deve essere crescente e senza duplicati."
            )
