import argparse
import tempfile
from pathlib import Path

import dask.array as da
import numpy as np
import torch
import xarray as xr
from dask.base import is_dask_collection

from data_manager import DataManager
from annual_evaluation import (
    build_annual_error_dataset,
    save_annual_error_outputs,
    save_physical_metrics,
)
from dataloader import (
    DataLoaderConfig,
    create_evaluation_dataloader,
    create_ocean_dataloaders,
)
from dataset import (
    OceanForecastDataset,
    OceanStateDataset,
    build_annual_evaluation_dataset,
)
from models.autoencoder import VolumeAutoencoderConfig, VolumeUNetAutoencoder
from normalization import Normalizer
from preprocessing import Preprocessor
from split import TemporalSplitter
from training import (
    fit_forecaster,
    read_forecaster_checkpoint,
    rmse_skill_score,
    run_forecast_epoch,
    run_annual_error_map_evaluation,
    run_physical_evaluation,
    run_persistence_baseline,
    run_temperature_evaluation,
    train_autoencoder_step,
)
from visualization import (
    plot_learning_curve,
    plot_learning_curve_snapshots,
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
        default=50,
        help="Numero massimo di epoche (early stopping attivo).",
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
        "--context-steps",
        type=int,
        default=3,
        help="Numero di giorni consecutivi forniti al modello. Default: 3.",
    )
    parser.add_argument(
        "--internal-normalization",
        choices=("none", "instance"),
        default="none",
        help=(
            "Normalizzazione nei blocchi U-Net; none preserva i livelli "
            "assoluti dei campi."
        ),
    )
    parser.add_argument(
        "--mean-mse-weight",
        type=float,
        default=2.0,
        help="Peso MSE aggiunto alla Gaussian NLL. Default: 2.0.",
    )
    parser.add_argument(
        "--temperature-mse-weight",
        type=float,
        default=2.0,
        help=(
            "Peso relativo del canale temperatura dentro la MSE. Default: 2.0."
        ),
    )
    parser.add_argument(
        "--gradient-clip-norm",
        type=float,
        default=1.0,
        help="Norma massima dei gradienti. Default: 1.0.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=project_root / "checkpoints/best_forecaster.pt",
        help="Percorso del checkpoint migliore.",
    )
    parser.add_argument(
        "--learning-curve-path",
        type=Path,
        default=None,
        help=(
            "Percorso PNG della curva di apprendimento; per default viene "
            "salvata accanto al checkpoint."
        ),
    )
    parser.add_argument(
        "--learning-curve-directory",
        type=Path,
        default=project_root / "outputs/learning_curves",
        help=(
            "Cartella degli snapshot cumulativi salvati dopo ogni epoca."
        ),
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
        "--evaluate-validation",
        action="store_true",
        help=(
            "Confronta checkpoint e persistence baseline sulla validation."
        ),
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help=(
            "Confronta checkpoint e persistence baseline sul test annuale."
        ),
    )
    parser.add_argument(
        "--evaluate-temperature-validation",
        action="store_true",
        help=(
            "Alias compatibile di --evaluate-physical-validation; valuta "
            "tutte le variabili."
        ),
    )
    parser.add_argument(
        "--evaluate-temperature-test",
        action="store_true",
        help=(
            "Alias compatibile di --evaluate-physical-test; valuta tutte le "
            "variabili."
        ),
    )
    parser.add_argument(
        "--evaluation-depth",
        "--temperature-depth",
        dest="evaluation_depth",
        type=float,
        default=0.5,
        help="Profondita' per metriche e mappe fisiche. Default: 0.5 m.",
    )
    parser.add_argument(
        "--evaluate-physical-validation",
        action="store_true",
        help="Valuta tutte le variabili in unita' fisiche sulla validation.",
    )
    parser.add_argument(
        "--evaluate-physical-test",
        action="store_true",
        help="Valuta tutte le variabili in unita' fisiche sul test annuale.",
    )
    parser.add_argument(
        "--plot-annual-errors-test",
        action="store_true",
        help=(
            "Salva NetCDF e figure 2x2 di MAE modello/persistence, "
            "differenza e deviazione standard dell'errore sul test."
        ),
    )
    parser.add_argument(
        "--evaluation-output-directory",
        type=Path,
        default=project_root / "outputs/annual_evaluation",
        help="Cartella per metriche, NetCDF e figure della valutazione annuale.",
    )
    parser.add_argument(
        "--plot-learning-curve",
        action="store_true",
        help=(
            "Rigenera la curva dalla history del checkpoint senza caricare "
            "il dataset."
        ),
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

    if args.evaluate_temperature_validation:
        args.evaluate_physical_validation = True
        print(
            "--evaluate-temperature-validation e' mantenuto come alias: "
            "verranno valutate tutte le variabili fisiche."
        )
    if args.evaluate_temperature_test:
        args.evaluate_physical_test = True
        print(
            "--evaluate-temperature-test e' mantenuto come alias: "
            "verranno valutate tutte le variabili fisiche."
        )

    if args.plot_learning_curve:
        create_learning_curve_from_checkpoint(args)
        return

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

    evaluation_requested = any(
        (
            args.evaluate_validation,
            args.evaluate_test,
            args.evaluate_temperature_validation,
            args.evaluate_temperature_test,
            args.evaluate_physical_validation,
            args.evaluate_physical_test,
            args.plot_annual_errors_test,
        )
    )
    if evaluation_requested and not args.train_model:
        checkpoint = read_forecaster_checkpoint(
            args.checkpoint_path,
            torch.device("cpu"),
        )
        model_config = checkpoint.get("model_config")
        if not isinstance(model_config, dict):
            raise ValueError(
                "Configurazione del modello assente nel checkpoint."
            )
        input_channels = int(model_config.get("input_channels", 0))
        if input_channels <= 0 or input_channels % 4 != 0:
            raise ValueError("Canali di input del checkpoint non validi.")
        args.context_steps = input_channels // 4
        args.internal_normalization = str(
            model_config.get("normalization", "instance")
        )
        print(
            "Configurazione di valutazione letta dal checkpoint: "
            f"context_steps={args.context_steps}, "
            f"normalization={args.internal_normalization}."
        )

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

    if args.context_steps <= 0:
        raise ValueError("context-steps deve essere positivo.")
    if args.mean_mse_weight < 0:
        raise ValueError("mean-mse-weight non puo' essere negativo.")
    if args.temperature_mse_weight <= 0:
        raise ValueError("temperature-mse-weight deve essere positivo.")
    if args.gradient_clip_norm <= 0:
        raise ValueError("gradient-clip-norm deve essere positivo.")
    train_dataset = OceanForecastDataset(
        normalized_train,
        context_steps=args.context_steps,
    )
    validation_dataset = OceanForecastDataset(
        normalized_validation,
        context_steps=args.context_steps,
    )
    test_dataset = OceanForecastDataset(
        normalized_test,
        context_steps=args.context_steps,
    )
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
    print(f"Giorni di contesto: {args.context_steps}")
    print("DataLoader configurati.")
    print(f"Batch size train: {loaders.train.batch_size}")
    print(f"Num workers train: {loaders.train.num_workers}")
    print(f"Device selezionato: {device}")

    annual_validation_loader = None
    annual_test_loader = None
    if evaluation_requested:
        annual_loader_config = DataLoaderConfig(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        annual_validation_dataset = build_annual_evaluation_dataset(
            normalized_train,
            normalized_validation,
            context_steps=args.context_steps,
        )
        annual_test_dataset = build_annual_evaluation_dataset(
            normalized_validation,
            normalized_test,
            context_steps=args.context_steps,
        )
        annual_validation_loader = create_evaluation_dataloader(
            annual_validation_dataset,
            annual_loader_config,
        )
        annual_test_loader = create_evaluation_dataloader(
            annual_test_dataset,
            annual_loader_config,
        )
        print(
            "Coppie annuali validation/test: "
            f"{len(annual_validation_dataset)}/{len(annual_test_dataset)}"
        )

    if args.first_real_training_step:
        run_first_real_training_step(loaders.train, args, device)

    if args.train_model:
        run_full_training(args, loaders.train, loaders.validation, device)

    if args.evaluate_validation:
        assert annual_validation_loader is not None
        evaluate_checkpoint_against_persistence(
            args,
            annual_validation_loader,
            device,
            split_label="Validation",
        )

    if args.evaluate_test:
        assert annual_test_loader is not None
        evaluate_checkpoint_against_persistence(
            args,
            annual_test_loader,
            device,
            split_label="Test",
        )

    if args.evaluate_physical_validation:
        assert annual_validation_loader is not None
        evaluate_physical_checkpoint(
            args,
            annual_validation_loader,
            normalizer.statistics,
            splits.validation,
            device,
            split_label="Validation",
        )

    if args.evaluate_physical_test:
        assert annual_test_loader is not None
        evaluate_physical_checkpoint(
            args,
            annual_test_loader,
            normalizer.statistics,
            splits.test,
            device,
            split_label="Test",
        )

    if args.plot_annual_errors_test:
        assert annual_test_loader is not None
        evaluate_annual_error_maps(
            args,
            annual_test_loader,
            normalizer.statistics,
            splits.test,
            device,
        )


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
            input_channels=4 * args.context_steps,
            output_channels=4,
            base_channels=args.base_channels,
            latent_channels=args.latent_channels,
            normalization=args.internal_normalization,
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
        expected_config = {
            "input_channels": 4 * args.context_steps,
            "output_channels": 4,
            "base_channels": args.base_channels,
            "latent_channels": args.latent_channels,
            "normalization": args.internal_normalization,
        }
        if saved_config != expected_config:
            raise ValueError(
                "La configurazione richiesta non coincide con il checkpoint."
            )
        saved_training_config = resume_checkpoint.get("training_config", {})
        if not isinstance(saved_training_config, dict) or (
            saved_training_config.get("context_steps") != args.context_steps
            or saved_training_config.get("mean_mse_weight")
            != args.mean_mse_weight
            or saved_training_config.get("temperature_mse_weight")
            != args.temperature_mse_weight
            or saved_training_config.get("gradient_clip_norm")
            != args.gradient_clip_norm
        ):
            raise ValueError(
                "Contesto o pesi della loss non coincidono con il checkpoint."
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
        learning_curve_directory=args.learning_curve_directory,
        mean_mse_weight=args.mean_mse_weight,
        gradient_clip_norm=args.gradient_clip_norm,
        temperature_mse_weight=args.temperature_mse_weight,
        training_config={
            "context_steps": args.context_steps,
            "mean_mse_weight": args.mean_mse_weight,
            "temperature_mse_weight": args.temperature_mse_weight,
            "gradient_clip_norm": args.gradient_clip_norm,
            "selection_metric": "validation_rmse",
            "post_epoch_train_evaluation": True,
        },
    )

    print("Training completato.")
    print(f"Epoca migliore: {result.best_epoch}")
    print(f"Validation NLL migliore: {result.best_validation_nll:.6f}")
    print(f"Validation RMSE migliore: {result.best_validation_rmse:.6f}")
    print(
        "Training persistence RMSE: "
        f"{result.train_persistence_rmse:.6f}"
    )
    print(
        "Validation persistence RMSE: "
        f"{result.validation_persistence_rmse:.6f}"
    )
    final_skill = rmse_skill_score(
        result.best_validation_rmse,
        result.validation_persistence_rmse,
    )
    print(f"Migliore validation skill vs persistence: {final_skill:.6f}")
    if final_skill <= 0:
        print(
            "ATTENZIONE: il modello non ha ancora superato la persistence "
            "sulla validation. Non eseguire il test finale."
        )
    print(f"Checkpoint: {result.checkpoint_path}")
    print(f"Checkpoint di ripresa: {result.last_checkpoint_path}")
    checkpoint = read_forecaster_checkpoint(
        result.last_checkpoint_path,
        torch.device("cpu"),
    )
    curve_path = plot_learning_curve(
        checkpoint["history"],
        resolve_learning_curve_path(args),
    )
    print(f"Curva di apprendimento: {curve_path}")
    print(
        "Snapshot cumulativi per epoca: "
        f"{args.learning_curve_directory}"
    )


def create_learning_curve_from_checkpoint(args: argparse.Namespace) -> None:
    """Crea il grafico dalla history completa, senza aprire i dati NetCDF."""

    last_checkpoint_path = args.checkpoint_path.with_name(
        f"{args.checkpoint_path.stem}_last{args.checkpoint_path.suffix}"
    )
    history_checkpoint_path = (
        last_checkpoint_path
        if last_checkpoint_path.exists()
        else args.checkpoint_path
    )
    checkpoint = read_forecaster_checkpoint(
        history_checkpoint_path,
        torch.device("cpu"),
    )
    history = checkpoint.get("history")
    if not isinstance(history, list):
        raise ValueError("History del training assente nel checkpoint.")

    curve_path = plot_learning_curve(
        history,
        resolve_learning_curve_path(args),
    )
    snapshot_paths = plot_learning_curve_snapshots(
        history,
        args.learning_curve_directory,
    )
    print(f"Curva di apprendimento: {curve_path}")
    print(
        f"Snapshot cumulativi rigenerati: {len(snapshot_paths)} in "
        f"{args.learning_curve_directory}"
    )


def resolve_learning_curve_path(args: argparse.Namespace) -> Path:
    """Determina il percorso del PNG, accanto al checkpoint per default."""

    if args.learning_curve_path is not None:
        return args.learning_curve_path
    return args.checkpoint_path.with_name(
        f"{args.checkpoint_path.stem}_learning_curve.png"
    )


def evaluate_checkpoint_against_persistence(
    args: argparse.Namespace,
    data_loader,
    device: torch.device,
    *,
    split_label: str,
) -> None:
    """Confronta il checkpoint con la persistence sullo stesso split."""

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
        batches=data_loader,
        device=device,
        progress_label=(
            f"{split_label} modello" if not args.no_progress else None
        ),
    )
    persistence = run_persistence_baseline(
        batches=data_loader,
        device=device,
        progress_label=(
            f"{split_label} persistence" if not args.no_progress else None
        ),
    )
    skill = rmse_skill_score(metrics.rmse, persistence.rmse)

    print(f"Checkpoint epoca: {checkpoint['epoch']}")
    print(f"{split_label} Gaussian NLL modello: {metrics.gaussian_nll:.6f}")
    print(f"{split_label} RMSE modello (normalizzato): {metrics.rmse:.6f}")
    print(f"{split_label} MAE modello (normalizzato): {metrics.mae:.6f}")
    print(
        f"{split_label} sigma media prevista (normalizzata): "
        f"{metrics.mean_standard_deviation:.6f}"
    )
    print(
        f"{split_label} coverage 68/95: "
        f"{metrics.coverage_68:.3f}/{metrics.coverage_95:.3f}"
    )
    print(
        f"{split_label} RMSE persistence (normalizzato): "
        f"{persistence.rmse:.6f}"
    )
    print(
        f"{split_label} MAE persistence (normalizzato): "
        f"{persistence.mae:.6f}"
    )
    print(f"{split_label} RMSE skill vs persistence: {skill:.6f}")


def _physical_evaluation_inputs(
    statistics: dict[str, xr.DataArray],
    split: xr.Dataset,
) -> tuple[tuple[str, ...], torch.Tensor, tuple[float, ...], dict[str, str]]:
    variable_names = OceanStateDataset.DEFAULT_VOLUME_VARIABLES
    if "depth" not in split.coords:
        raise ValueError("Coordinata depth assente dal dataset.")
    depth_values = tuple(float(value) for value in split["depth"].values)
    standard_deviations: list[torch.Tensor] = []
    units: dict[str, str] = {}
    for variable in variable_names:
        statistic_name = f"{variable}_std"
        if statistic_name not in statistics:
            raise ValueError(f"Deviazione standard assente: {statistic_name}.")
        standard = statistics[statistic_name]
        if set(standard.dims) != {"depth"}:
            raise ValueError(f"{statistic_name} deve dipendere solo da depth.")
        standard_deviations.append(
            torch.as_tensor(
                standard.transpose("depth").values,
                dtype=torch.float32,
            )
        )
        units[variable] = str(split[variable].attrs.get("units", "unknown"))
    return (
        variable_names,
        torch.stack(standard_deviations),
        depth_values,
        units,
    )


def _load_evaluation_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[VolumeUNetAutoencoder, dict[str, object]]:
    checkpoint = read_forecaster_checkpoint(checkpoint_path, device)
    saved_config = checkpoint.get("model_config")
    if not isinstance(saved_config, dict):
        raise ValueError("Configurazione del modello assente nel checkpoint.")
    model = VolumeUNetAutoencoder(
        VolumeAutoencoderConfig(**saved_config)
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint


def evaluate_physical_checkpoint(
    args: argparse.Namespace,
    data_loader,
    statistics: dict[str, xr.DataArray],
    split: xr.Dataset,
    device: torch.device,
    *,
    split_label: str,
) -> None:
    """Valuta e salva tutte le variabili nelle rispettive unita' fisiche."""

    variable_names, standards, depths, units = _physical_evaluation_inputs(
        statistics,
        split,
    )
    model, checkpoint = _load_evaluation_model(args.checkpoint_path, device)
    result = run_physical_evaluation(
        model=model,
        batches=data_loader,
        device=device,
        variable_names=variable_names,
        standard_deviations=standards,
        depth_values_m=depths,
        units=units,
        requested_depth_m=args.evaluation_depth,
        progress_label=(
            f"{split_label} variabili fisiche" if not args.no_progress else None
        ),
    )
    print(f"Checkpoint epoca: {checkpoint['epoch']}")
    print(f"{split_label} previsioni annuali: {result.forecast_count}")
    for variable, evaluation in result.variables.items():
        _print_physical_metrics(
            split_label,
            variable,
            "tutte le profondita'",
            evaluation.unit,
            evaluation.all_depths,
        )
        _print_physical_metrics(
            split_label,
            variable,
            f"profondita' {result.selected_depth_m:.3f} m",
            evaluation.unit,
            evaluation.selected_depth,
        )
    json_path, csv_path = save_physical_metrics(
        result,
        args.evaluation_output_directory,
        split_label=split_label,
    )
    print(f"Metriche JSON: {json_path}")
    print(f"Metriche CSV: {csv_path}")


def _print_physical_metrics(
    split_label: str,
    variable: str,
    scope: str,
    unit: str,
    metrics,
) -> None:
    prefix = f"{split_label} {variable} ({scope})"
    print(f"{prefix} RMSE modello: {metrics.model_rmse:.6f} {unit}")
    print(f"{prefix} MAE modello: {metrics.model_mae:.6f} {unit}")
    print(f"{prefix} bias modello: {metrics.model_bias:.6f} {unit}")
    print(
        f"{prefix} sigma media prevista: "
        f"{metrics.model_mean_standard_deviation:.6f} {unit}"
    )
    print(
        f"{prefix} coverage 68/95: "
        f"{metrics.model_coverage_68:.3f}/{metrics.model_coverage_95:.3f}"
    )
    print(f"{prefix} RMSE persistence: {metrics.persistence_rmse:.6f} {unit}")
    print(f"{prefix} MAE persistence: {metrics.persistence_mae:.6f} {unit}")
    print(f"{prefix} bias persistence: {metrics.persistence_bias:.6f} {unit}")
    print(f"{prefix} RMSE skill vs persistence: {metrics.rmse_skill_score:.6f}")


def evaluate_annual_error_maps(
    args: argparse.Namespace,
    data_loader,
    statistics: dict[str, xr.DataArray],
    split: xr.Dataset,
    device: torch.device,
) -> None:
    """Genera NetCDF e pannelli 2x2 di MAE, dispersione e persistence."""

    variable_names, standards, depths, units = _physical_evaluation_inputs(
        statistics,
        split,
    )
    model, _ = _load_evaluation_model(args.checkpoint_path, device)
    result = run_annual_error_map_evaluation(
        model=model,
        batches=data_loader,
        device=device,
        standard_deviations=standards,
        depth_values_m=depths,
        requested_depth_m=args.evaluation_depth,
        progress_label=(
            "Test mappe annuali dell'errore" if not args.no_progress else None
        ),
    )
    years = np.unique(split["time"].dt.year.values)
    if years.size != 1:
        raise ValueError("Lo split della mappa deve contenere un solo anno.")
    dataset = build_annual_error_dataset(
        result,
        variable_names=variable_names,
        latitudes=np.asarray(split["latitude"].values),
        longitudes=np.asarray(split["longitude"].values),
        units=units,
        target_year=int(years[0]),
    )
    (
        netcdf_path,
        png_path,
        standard_deviation_png_path,
        persistence_png_path,
        difference_png_path,
    ) = save_annual_error_outputs(
        dataset,
        args.evaluation_output_directory,
    )
    print(f"Previsioni aggregate nella mappa: {result.forecast_count}")
    print(f"Profondita' della mappa: {result.selected_depth_m:.3f} m")
    print(f"Mappe NetCDF: {netcdf_path}")
    print(f"Figura annuale: {png_path}")
    print(f"Figura deviazione standard annuale: {standard_deviation_png_path}")
    print(f"Figura MAE persistence annuale: {persistence_png_path}")
    print(f"Figura differenza MAE modello-persistence: {difference_png_path}")


def evaluate_temperature_checkpoint(
    args: argparse.Namespace,
    data_loader,
    statistics: dict[str, xr.DataArray],
    split: xr.Dataset,
    device: torch.device,
    *,
    split_label: str,
) -> None:
    """Stampa metriche della temperatura in gradi Celsius."""

    checkpoint = read_forecaster_checkpoint(args.checkpoint_path, device)
    saved_config = checkpoint.get("model_config")
    if not isinstance(saved_config, dict):
        raise ValueError("Configurazione del modello assente nel checkpoint.")
    if "thetao_cglo_std" not in statistics:
        raise ValueError("Deviazione standard della temperatura assente.")
    if "depth" not in split.coords:
        raise ValueError("Coordinata depth assente dal dataset.")

    temperature_std = statistics["thetao_cglo_std"]
    if set(temperature_std.dims) != {"depth"}:
        raise ValueError(
            "thetao_cglo_std deve dipendere soltanto dalla profondita'."
        )
    temperature_std = temperature_std.transpose("depth")
    depth_values = tuple(float(value) for value in split["depth"].values)

    model = VolumeUNetAutoencoder(
        VolumeAutoencoderConfig(**saved_config)
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    result = run_temperature_evaluation(
        model=model,
        batches=data_loader,
        device=device,
        temperature_std_by_depth=torch.as_tensor(
            temperature_std.values,
            dtype=torch.float32,
        ),
        depth_values_m=depth_values,
        requested_depth_m=args.evaluation_depth,
        progress_label=(
            f"{split_label} temperatura" if not args.no_progress else None
        ),
    )

    print(f"Checkpoint epoca: {checkpoint['epoch']}")
    _print_temperature_metrics(
        split_label,
        "tutte le profondita'",
        result.all_depths,
    )
    _print_temperature_metrics(
        split_label,
        f"profondita' {result.selected_depth_m:.3f} m",
        result.selected_depth,
    )


def _print_temperature_metrics(split_label: str, scope: str, metrics) -> None:
    prefix = f"{split_label} temperatura ({scope})"
    print(f"{prefix} RMSE modello: {metrics.model_rmse_c:.6f} °C")
    print(f"{prefix} MAE modello: {metrics.model_mae_c:.6f} °C")
    print(f"{prefix} bias modello: {metrics.model_bias_c:.6f} °C")
    print(
        f"{prefix} sigma media prevista: "
        f"{metrics.model_mean_standard_deviation_c:.6f} °C"
    )
    print(
        f"{prefix} coverage 68/95: "
        f"{metrics.model_coverage_68:.3f}/{metrics.model_coverage_95:.3f}"
    )
    print(
        f"{prefix} RMSE persistence: "
        f"{metrics.persistence_rmse_c:.6f} °C"
    )
    print(
        f"{prefix} MAE persistence: "
        f"{metrics.persistence_mae_c:.6f} °C"
    )
    print(
        f"{prefix} bias persistence: "
        f"{metrics.persistence_bias_c:.6f} °C"
    )
    print(f"{prefix} RMSE skill vs persistence: {metrics.rmse_skill_score:.6f}")


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


def run_first_real_training_step(
    train_loader,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    """Esegue un solo update sul primo batch reale."""

    model = build_forecaster(args, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

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
