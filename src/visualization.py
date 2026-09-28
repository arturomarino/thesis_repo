"""Visualizzazioni delle metriche prodotte durante il training."""

from pathlib import Path
from typing import Iterable


def plot_learning_curve(
    history: Iterable[dict[str, object]],
    output_path: Path,
) -> Path:
    """Salva NLL e RMSE comparabili di training e validation."""

    epochs: list[int] = []
    train_nll: list[float] = []
    validation_nll: list[float] = []
    train_rmse: list[float] = []
    validation_rmse: list[float] = []
    train_persistence_rmse: list[float | None] = []
    persistence_rmse: list[float | None] = []

    for entry in history:
        train_metrics = entry.get("train")
        validation_metrics = entry.get("validation")
        if not isinstance(train_metrics, dict) or not isinstance(
            validation_metrics, dict
        ):
            raise ValueError("Formato della history del training non valido.")

        try:
            epochs.append(int(entry["epoch"]))
            train_nll.append(float(train_metrics["gaussian_nll"]))
            validation_nll.append(
                float(validation_metrics["gaussian_nll"])
            )
            train_rmse.append(float(train_metrics["rmse"]))
            validation_rmse.append(float(validation_metrics["rmse"]))
            train_persistence_metrics = entry.get("train_persistence")
            train_persistence_rmse.append(
                float(train_persistence_metrics["rmse"])
                if isinstance(train_persistence_metrics, dict)
                else None
            )
            persistence_metrics = entry.get("validation_persistence")
            persistence_rmse.append(
                float(persistence_metrics["rmse"])
                if isinstance(persistence_metrics, dict)
                else None
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "La history non contiene epoca, NLL e RMSE valide."
            ) from error

    if not epochs:
        raise ValueError("La history del training e' vuota.")

    # L'import locale evita di caricare Matplotlib durante training e test che
    # non producono grafici. Il backend Agg funziona anche su server e Colab.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, (nll_axis, rmse_axis) = plt.subplots(
        2,
        1,
        figsize=(10, 9),
        sharex=True,
    )
    nll_axis.plot(
        epochs,
        train_nll,
        color="#2563eb",
        linewidth=2,
        marker="o",
        markersize=3,
        label="Training NLL",
    )
    nll_axis.plot(
        epochs,
        validation_nll,
        color="#dc2626",
        linewidth=2,
        marker="o",
        markersize=3,
        label="Validation NLL",
    )

    nll_axis.set(
        title="Gaussian Negative Log-Likelihood",
        ylabel="NLL",
    )
    nll_axis.grid(True, alpha=0.25)
    nll_axis.legend()

    rmse_axis.plot(
        epochs,
        train_rmse,
        color="#2563eb",
        linewidth=2,
        marker="o",
        markersize=3,
        label="Training RMSE (fixed weights)",
    )
    rmse_axis.plot(
        epochs,
        validation_rmse,
        color="#dc2626",
        linewidth=2,
        marker="o",
        markersize=3,
        label="Validation RMSE",
    )
    if train_persistence_rmse and all(
        value is not None for value in train_persistence_rmse
    ):
        rmse_axis.plot(
            epochs,
            [float(value) for value in train_persistence_rmse],
            color="#2563eb",
            linewidth=1.5,
            linestyle=":",
            label="Training persistence RMSE",
        )
    if persistence_rmse and all(
        value is not None for value in persistence_rmse
    ):
        rmse_axis.plot(
            epochs,
            [float(value) for value in persistence_rmse],
            color="#7c3aed",
            linewidth=1.8,
            linestyle="--",
            label="Validation persistence RMSE",
        )

    best_index = min(
        range(len(validation_rmse)), key=validation_rmse.__getitem__
    )
    rmse_axis.scatter(
        epochs[best_index],
        validation_rmse[best_index],
        color="#16a34a",
        edgecolor="white",
        linewidth=1,
        s=70,
        zorder=3,
        label=f"Best validation (epoch {epochs[best_index]})",
    )
    rmse_axis.set(
        xlabel="Epoch",
        ylabel="Normalized RMSE",
    )
    rmse_axis.grid(True, alpha=0.25)
    rmse_axis.legend()
    rmse_axis.set_xlim(left=0, right=max(1, epochs[-1]))
    figure.suptitle("Learning curves and persistence comparison")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)

    return output_path


def plot_learning_curve_snapshot(
    history: Iterable[dict[str, object]],
    output_directory: Path,
) -> Path:
    """Salva la curva cumulativa fino all'ultima epoca della history."""

    history_entries = list(history)
    if not history_entries:
        raise ValueError("La history del training e' vuota.")

    try:
        epoch = int(history_entries[-1]["epoch"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Epoca non valida nella history del training.") from error

    output_path = Path(output_directory) / (
        f"learning_curve_epoch_{epoch:03d}.png"
    )
    return plot_learning_curve(history_entries, output_path)


def plot_learning_curve_snapshots(
    history: Iterable[dict[str, object]],
    output_directory: Path,
) -> list[Path]:
    """Rigenera uno snapshot cumulativo per ogni epoca disponibile."""

    history_entries = list(history)
    if not history_entries:
        raise ValueError("La history del training e' vuota.")

    return [
        plot_learning_curve_snapshot(
            history_entries[:history_end],
            output_directory,
        )
        for history_end in range(1, len(history_entries) + 1)
    ]
