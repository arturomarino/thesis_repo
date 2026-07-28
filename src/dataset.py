"""
dataset.py

Conversione lazy dei dataset Xarray normalizzati in campioni PyTorch.

Responsabilita':
- rappresentare separatamente volume oceanico e superficie;
- materializzare soltanto lo stato temporale richiesto;
- creare coppie previsionali ``t -> t+1``;
- conservare una maschera dei valori oceanici validi.

Non esegue:
- split o normalizzazione;
- batching;
- training del modello.
"""

from dataclasses import dataclass
from datetime import timedelta
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


class OceanVolume(TypedDict):
    """Volume 3D multivariato e relativa maschera."""

    volume: torch.Tensor
    volume_mask: torch.Tensor


class OceanForecastSample(TypedDict):
    """Coppia supervisionata: stato odierno e target del giorno successivo."""

    input: OceanVolume
    target: OceanVolume
    input_time_index: int
    target_time_index: int


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

    Questa rappresentazione di base viene riutilizzata da
    ``OceanForecastDataset`` per costruire le coppie temporali supervisionate.

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
        volume_state = self._materialize_volume(time_index)
        surface, surface_mask = self._to_tensor_pair(
            self._surface.isel({self.config.time_dim: time_index})
        )

        return {
            **volume_state,
            "surface": surface,
            "surface_mask": surface_mask,
        }

    def _materialize_volume(self, time_index: int) -> OceanVolume:
        volume, volume_mask = self._to_tensor_pair(
            self._volume.isel({self.config.time_dim: time_index})
        )
        return {
            "volume": volume,
            "volume_mask": volume_mask,
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


class OceanForecastDataset(Dataset[OceanForecastSample]):
    """
    Espone coppie consecutive ``volume(t) -> volume(t+1)``.

    Il modello riceve soltanto le quattro variabili volumetriche. ``zos_cglo``
    resta esclusa sia dall'input sia dal target previsionale.
    """

    DEFAULT_VOLUME_VARIABLES = OceanStateDataset.DEFAULT_VOLUME_VARIABLES

    def __init__(
        self,
        dataset: xr.Dataset,
        config: OceanDatasetConfig | None = None,
        volume_variables: tuple[
            str, ...
        ] = OceanStateDataset.DEFAULT_VOLUME_VARIABLES,
        forecast_horizon: int = 1,
        require_daily_steps: bool = True,
    ) -> None:
        if forecast_horizon <= 0:
            raise ValueError("forecast_horizon deve essere positivo.")

        self.forecast_horizon = forecast_horizon
        self._states = OceanStateDataset(
            dataset=dataset,
            config=config,
            volume_variables=volume_variables,
        )

        if len(self._states) <= self.forecast_horizon:
            raise ValueError(
                "Il dataset deve contenere piu' time step "
                "dell'orizzonte previsionale."
            )

        if require_daily_steps:
            self._validate_daily_steps()

    def __len__(self) -> int:
        return len(self._states) - self.forecast_horizon

    def __getitem__(self, index: int) -> OceanForecastSample:
        index = self._normalize_index(index)
        target_index = index + self.forecast_horizon
        return {
            "input": self._states._materialize_volume(index),
            "target": self._states._materialize_volume(target_index),
            "input_time_index": index,
            "target_time_index": target_index,
        }

    @property
    def volume_shape(self) -> tuple[int, int, int, int]:
        return self._states.volume_shape

    def _normalize_index(self, index: int) -> int:
        if not isinstance(index, int) or isinstance(index, bool):
            raise TypeError("L'indice del campione deve essere un intero.")

        length = len(self)
        if index < 0:
            index += length

        if index < 0 or index >= length:
            raise IndexError(
                f"Indice {index} fuori intervallo per {length} coppie."
            )

        return index

    def _validate_daily_steps(self) -> None:
        time_dim = self._states.config.time_dim
        time_index = self._states.dataset.indexes[time_dim]
        expected_delta = timedelta(days=self.forecast_horizon)

        for input_index in range(len(self)):
            target_index = input_index + self.forecast_horizon
            actual_delta = time_index[target_index] - time_index[input_index]
            if actual_delta != expected_delta:
                raise ValueError(
                    "La coppia previsionale non corrisponde a giorni "
                    "consecutivi: "
                    f"{time_index[input_index]} -> {time_index[target_index]}."
                )
