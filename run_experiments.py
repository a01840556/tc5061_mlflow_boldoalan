"""Lote de experimentos definido de antemano: 10 configuraciones distintas y una repetición del baseline.

Cada configuración ejecuta train.py en un proceso nuevo y todas quedan en el mismo experimento de
MLflow. La configuración se elige con el F1 macro medio de la validación cruzada dentro de train
(desempates: menor desviación, menor C, orden del lote); las métricas de test son descriptivas.

Uso:  python run_experiments.py [--output outputs] [--experiment NOMBRE] [--no-tracking]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path

from train import CLI_FLAGS, DEFAULTS, EXPERIMENT_NAME, ROOT, write_json

# (bloque, cambios respecto al baseline). Valores y orden fijados antes de ver resultados.
CONFIGURATIONS = [
    ("regularizacion_L2", {"C": 0.01}),
    ("regularizacion_L2", {"C": 0.1}),
    ("baseline", {}),
    ("regularizacion_L2", {"C": 10.0}),
    ("regularizacion_L2", {"C": 100.0}),
    ("optimizador", {"solver": "newton-cg"}),
    ("optimizador", {"solver": "saga"}),
    ("penalizacion_L1", {"solver": "saga", "l1_ratio": 1.0}),
    ("penalizacion_L1", {"solver": "saga", "l1_ratio": 1.0, "C": 0.1}),
    ("penalizacion_L1", {"solver": "saga", "l1_ratio": 1.0, "C": 0.01}),
]
REPEATED = 2  # el baseline se vuelve a correr al final con exactamente el mismo comando
COMPARED_KEYS = ["params", "metrics", "predictions_sha256", "split_sha256", "cv_scores_sha256", "environment"]
TABLE_PARAMS = ["C", "l1_ratio", "solver", "max_iter", "tol", "seed"]


def cli_arguments(changes: dict) -> list[str]:
    """Argumentos explícitos de train.py: baseline + cambios (todos los valores a la vista)."""
    config = {**DEFAULTS, **changes}
    arguments = []
    for key, flag in CLI_FLAGS.items():
        value = config[key]
        arguments += [flag, value if isinstance(value, str) else format(value, "g")]
    return arguments


def planned_runs() -> list[tuple[str, str, list[str]]]:
    """(rol, bloque, argumentos) de las 10 configuraciones y la repetición."""
    runs = [("comparacion", block, cli_arguments(changes)) for block, changes in CONFIGURATIONS]
    block, changes = CONFIGURATIONS[REPEATED]
    return runs + [("repeticion", block, cli_arguments(changes))]


def verify_repetition(first: dict, second: dict) -> dict:
    checks = {key: first[key] == second[key] for key in COMPARED_KEYS}
    if not all(checks.values()):
        raise RuntimeError(f"La repetición no reprodujo el resultado: {checks}")
    return checks


def select_best(results: list[dict]) -> dict:
    """Máximo F1 macro de CV; desempates: menor desviación, menor C, orden del lote."""
    ranked = sorted(enumerate(results), key=lambda item: (-item[1]["metrics"]["cv_f1_macro_mean"],
                                                          item[1]["metrics"]["cv_f1_macro_std"],
                                                          item[1]["params"]["C"], item[0]))
    return ranked[0][1]


def run_batch(output: Path, output_arg: str, experiment: str, no_tracking: bool, batch: Path) -> tuple:
    commands, results = [], []
    tracking_flag = ["--no-tracking"] if no_tracking else []
    for index, (role, block, arguments) in enumerate(planned_runs(), start=1):
        shown = ["python", "train.py", *arguments, "--experiment", experiment, "--output", output_arg]
        commands.append({"execution": index, "role": role, "block": block,
                         "command": shlex.join(shown + tracking_flag)})
        # Se pasa --output tal como se escribió: train.py resuelve las rutas relativas desde el repositorio.
        completed = subprocess.run([sys.executable, str(ROOT / "train.py"), *arguments, "--experiment",
                                    experiment, "--output", output_arg, *tracking_flag],
                                   capture_output=True, cwd=ROOT, text=True, encoding="utf-8",
                                   errors="replace", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        (batch / f"execution_{index:02d}.stdout.txt").write_text(completed.stdout, encoding="utf-8")
        (batch / f"execution_{index:02d}.stderr.txt").write_text(completed.stderr, encoding="utf-8")
        write_json(batch / "commands.json", commands)
        if completed.returncode:
            print(completed.stderr, file=sys.stderr)
            raise SystemExit(f"Falló la ejecución {index}; el lote queda incompleto. Registros en {batch}")
        result = json.loads(completed.stdout.strip().splitlines()[-1])
        results.append({**result, "role": role, "block": block, "execution": index})
        write_json(batch / "results.json", results)
        m = result["metrics"]
        print(f"{index:>2} {role:<11} {result['run_name']:<22} CV F1 = {m['cv_f1_macro_mean']:.4f} "
              f"+/- {m['cv_f1_macro_std']:.4f} | test F1 = {m['test_f1_macro']:.4f} | "
              f"coef != 0: {m['n_coef_nonzero']} | iter = {m['n_iter']}", flush=True)
    return commands, results


def write_comparison(batch: Path, results: list[dict]) -> None:
    rows = [{"execution": r["execution"], "role": r["role"], "block": r["block"], "run_name": r["run_name"],
             "run_id": r["run_id"], **{k: r["params"][k] for k in TABLE_PARAMS}, **r["metrics"]}
            for r in results]
    with (batch / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def tag_runs(results: list[dict], selected: dict, batch: Path) -> None:
    """Etiqueta las corridas del lote y adjunta la evidencia del lote al run seleccionado."""
    from mlflow.tracking import MlflowClient
    client = MlflowClient(tracking_uri=results[0]["tracking_uri"])
    for r in results:
        client.set_tag(r["run_id"], "batch_id", batch.name)
        client.set_tag(r["run_id"], "role", r["role"])
        client.set_tag(r["run_id"], "block", r["block"])
    client.set_tag(selected["run_id"], "selected", "true")
    for name in ["commands.json", "comparison.csv", "reproducibility.json", "selection.json"]:
        client.log_artifact(selected["run_id"], str(batch / name), artifact_path="batch")


def main(argv=None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # una consola cp1252 no debe detener el lote
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", default="outputs", help="Carpeta de MLflow y evidencias (relativa al repositorio).")
    parser.add_argument("--experiment", default=EXPERIMENT_NAME, help="Experimento de MLflow del lote.")
    parser.add_argument("--no-tracking", action="store_true", help="Calcula sin MLflow, solo para pruebas.")
    args = parser.parse_args(argv)
    output = Path(args.output) if Path(args.output).is_absolute() else ROOT / args.output
    output = output.resolve()
    if (output / "mlflow.db").exists() and not args.no_tracking:
        print("Aviso: la carpeta ya tiene corridas; para una evidencia limpia usa una carpeta --output nueva.")
    batch = output / "batches" / f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    batch.mkdir(parents=True)

    commands, results = run_batch(output, args.output, args.experiment, args.no_tracking, batch)
    compared = [r for r in results if r["role"] == "comparacion"]
    checks = verify_repetition(compared[REPEATED], results[-1])
    write_json(batch / "reproducibility.json", {
        "identical_arguments": commands[REPEATED]["command"] == commands[-1]["command"],
        "checks": checks, "first_run_id": compared[REPEATED]["run_id"],
        "second_run_id": results[-1]["run_id"], "scope": "mismo entorno, mismos datos y mismos argumentos"})
    write_comparison(batch, results)
    selected = select_best(compared)
    write_json(batch / "selection.json", {
        "criterion": "max cv_f1_macro_mean; desempates: min cv_f1_macro_std, min C, orden del lote",
        "run_name": selected["run_name"], "run_id": selected["run_id"],
        "params": {k: selected["params"][k] for k in TABLE_PARAMS}, "metrics": selected["metrics"],
        "note": "El test ya se observó en la actividad anterior: aquí es descriptivo, no confirmatorio."})
    if not args.no_tracking:
        tag_runs(results, selected, batch)
    print(f"Seleccionada: {selected['run_name']} (CV F1 = {selected['metrics']['cv_f1_macro_mean']:.4f}). "
          f"Repetición idéntica: {all(checks.values())}. Lote: {batch.relative_to(ROOT) if batch.is_relative_to(ROOT) else batch}")


if __name__ == "__main__":
    main()
