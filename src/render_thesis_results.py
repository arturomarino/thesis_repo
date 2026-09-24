"""Renderizza nei sorgenti LaTeX le metriche annuali gia' calcolate."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


VARIABLE_LABELS = {
    "thetao_cglo": "Temperature",
    "so_cglo": "Salinity",
    "uo_cglo": "Zonal velocity $u$",
    "vo_cglo": "Meridional velocity $v$",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crea le tabelle LaTeX dai JSON della valutazione annuale."
    )
    parser.add_argument("--validation-json", type=Path, required=True)
    parser.add_argument("--test-json", type=Path, required=True)
    parser.add_argument("--output-tex", type=Path, required=True)
    parser.add_argument("--figure-source", type=Path)
    parser.add_argument("--figure-destination", type=Path)
    return parser.parse_args()


def _load_records(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("forecast_count", -1)) != 364:
        raise ValueError(f"{path} non contiene le 364 previsioni richieste.")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != 8:
        raise ValueError(f"{path} deve contenere 8 righe di metriche.")
    return payload


def render_tables(validation: dict[str, object], test: dict[str, object]) -> str:
    """Restituisce le due tabelle LaTeX deterministica e probabilistica."""

    records = [*validation["records"], *test["records"]]
    deterministic_rows: list[str] = []
    probabilistic_rows: list[str] = []
    for record in records:
        variable = str(record["variable"])
        unit = str(record["unit"]).replace("_", r"\_")
        scope = (
            "All"
            if record["scope"] == "all_depths"
            else f"{float(record['selected_depth_m']):.3f} m"
        )
        label = f"{VARIABLE_LABELS.get(variable, variable)} ({unit})"
        prefix = f"{record['split']} & {label} & {scope}"
        deterministic_rows.append(
            prefix
            + " & "
            + " & ".join(
                f"{float(record[key]):.6f}"
                for key in (
                    "model_rmse",
                    "model_mae",
                    "model_bias",
                    "persistence_rmse",
                    "rmse_skill_score",
                )
            )
            + r" \\"
        )
        probabilistic_rows.append(
            prefix
            + " & "
            + " & ".join(
                f"{float(record[key]):.6f}"
                for key in (
                    "model_mean_standard_deviation",
                    "model_coverage_68",
                    "model_coverage_95",
                )
            )
            + r" \\"
        )
    return "\n".join(
        (
            r"\begin{table}[p]",
            r"\centering\scriptsize",
            r"\caption{Physical-unit deterministic results over 364 annual forecasts. Bias is forecast minus target.}",
            r"\label{tab:physical-deterministic-results}",
            r"\begin{tabular}{lllrrrrr}",
            r"\toprule",
            r"Split & Variable & Depth & RMSE & MAE & Bias & Pers. RMSE & Skill \\",
            r"\midrule",
            *deterministic_rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
            r"\begin{table}[p]",
            r"\centering\scriptsize",
            r"\caption{Physical-unit probabilistic diagnostics over the same annual forecasts.}",
            r"\label{tab:physical-probabilistic-results}",
            r"\begin{tabular}{lllrrr}",
            r"\toprule",
            r"Split & Variable & Depth & Mean $\sigma$ & 68\% coverage & 95\% coverage \\",
            r"\midrule",
            *probabilistic_rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        )
    )


def main() -> None:
    args = parse_args()
    validation = _load_records(args.validation_json)
    test = _load_records(args.test_json)
    args.output_tex.parent.mkdir(parents=True, exist_ok=True)
    args.output_tex.write_text(
        render_tables(validation, test),
        encoding="utf-8",
    )
    if (args.figure_source is None) != (args.figure_destination is None):
        raise ValueError(
            "figure-source e figure-destination devono essere forniti insieme."
        )
    if args.figure_source is not None and args.figure_destination is not None:
        args.figure_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.figure_source, args.figure_destination)


if __name__ == "__main__":
    main()
