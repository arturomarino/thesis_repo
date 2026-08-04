"""Visualizzazioni delle metriche prodotte durante il training."""

from pathlib import Path
from typing import Iterable


def plot_learning_curve(
    history: Iterable[dict[str, object]],
    output_path: Path,
) -> Path:
    """Salva la Gaussian NLL di training e validation per ogni epoca."""

    epochs: list[int] = []
    train_nll: list[float] = []
    validation_nll: list[float] = []

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
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "La history non contiene epoca e Gaussian NLL valide."
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

    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.plot(
        epochs,
        train_nll,
        color="#2563eb",
        linewidth=2,
        marker="o",
        markersize=3,
        label="Training NLL",
    )
    axis.plot(
        epochs,
        validation_nll,
        color="#dc2626",
        linewidth=2,
        marker="o",
        markersize=3,
        label="Validation NLL",
    )

    best_index = min(
        range(len(validation_nll)), key=validation_nll.__getitem__
    )
    axis.scatter(
        epochs[best_index],
        validation_nll[best_index],
        color="#16a34a",
        edgecolor="white",
        linewidth=1,
        s=70,
        zorder=3,
        label=f"Best validation (epoca {epochs[best_index]})",
    )
    axis.set(
        title="Curva di apprendimento",
        xlabel="Epoca",
        ylabel="Gaussian Negative Log-Likelihood",
    )
    axis.grid(True, alpha=0.25)
    axis.legend()
    axis.margins(x=0.02)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)

    return output_path
