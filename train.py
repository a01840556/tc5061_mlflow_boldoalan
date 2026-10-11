"""Entrena el baseline de Wine (StandardScaler + LogisticRegression) y registra la corrida en MLflow.

Es la migración a script del notebook de la actividad anterior: carga y valida los datos,
separa train/test, estima el desempeño con validación cruzada dentro de train, entrena,
evalúa y registra parámetros, métricas, artefactos y el modelo en un experimento de MLflow.

Sin argumentos ejecuta la configuración del baseline. Ejemplo con todos los hiperparámetros:
    python train.py --C 1 --l1-ratio 0 --solver lbfgs --max-iter 1000 --tol 0.0001 --seed 42
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import math
import os
import platform
import re
import sys
import tempfile
import uuid
import warnings
from pathlib import Path

# Un solo hilo en las librerías numéricas (OpenBLAS, MKL, Accelerate, OpenMP) para resultados repetibles;
# se fija antes de importar numpy. Sin telemetría de MLflow; caché de matplotlib en una carpeta temporal.
THREAD_VARS = ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
               "NUMEXPR_NUM_THREADS"]
for _var in THREAD_VARS:
    os.environ[_var] = "1"
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
os.environ.setdefault("DO_NOT_TRACK", "1")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "tc5061-matplotlib"))

import numpy as np
import pandas as pd
import sklearn
from matplotlib.figure import Figure
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "wine.data"
DATA_SHA256 = "6be6b1203f3d51df0b553a70e57b8a723cd405683958204f96d23d7cd6aea659"
EXPERIMENT_NAME = "TC5061_Experimentacion_BoldoAlan_Wine"
AUTHOR = "Alan Boldo"
TARGET = "cultivar"
CLASSES = [1, 2, 3]
FEATURES = ["alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium",
            "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins",
            "color_intensity", "hue", "od280_od315", "proline"]
SOLVERS = ["lbfgs", "newton-cg", "saga"]
# Configuración del baseline del notebook: son también los valores por defecto de la CLI.
DEFAULTS = {"C": 1.0, "l1_ratio": 0.0, "solver": "lbfgs", "max_iter": 1000, "tol": 1e-4,
            "seed": 42, "test_size": 0.2, "cv_folds": 5}
CLI_FLAGS = {"C": "--C", "l1_ratio": "--l1-ratio", "solver": "--solver", "max_iter": "--max-iter",
             "tol": "--tol", "seed": "--seed", "test_size": "--test-size", "cv_folds": "--cv-folds"}
PACKAGES = ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "threadpoolctl", "mlflow",
            "skops", "cloudpickle"]
# Requisitos del modelo registrado (skops/cloudpickle: formatos de serialización de mlflow.sklearn).
MODEL_PACKAGES = ["mlflow", "scikit-learn", "numpy", "scipy", "pandas", "skops", "cloudpickle"]
MIN_SKLEARN = (1, 8)  # desde 1.8, l1_ratio define el tipo de penalización sin usar `penalty`


# ---------------------------------------------------------------- utilidades

def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True,
                               allow_nan=False) + "\n", encoding="utf-8")


def environment_info() -> dict:
    """Versiones de Python y de las librerías que intervienen en el resultado."""
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": versions, "threads": {var: os.environ.get(var) for var in THREAD_VARS}}


# ---------------------------------------------------------------- datos

def validate_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Detiene la ejecución ante datos inesperados; no imputa, recorta ni elimina filas."""
    if raw.shape != (178, 14) or list(raw.columns) != [TARGET] + FEATURES:
        raise ValueError("Wine debe tener 178 filas y las 14 columnas esperadas.")
    if not all(pd.api.types.is_numeric_dtype(raw[c]) for c in raw):
        raise ValueError("Todas las columnas deben ser numéricas.")
    if raw.isna().any().any() or not np.isfinite(raw.to_numpy(dtype=float)).all():
        raise ValueError("Se encontraron valores ausentes o no finitos.")
    if not raw[TARGET].isin(CLASSES).all():
        raise ValueError("Las etiquetas deben pertenecer a {1, 2, 3}.")
    if raw.duplicated().any() or raw[FEATURES].duplicated().any():
        raise ValueError("Se encontraron filas o vectores de predictoras duplicados.")
    return raw.astype({TARGET: "int64", **{f: "float64" for f in FEATURES}})


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Lee wine.data de UCI y comprueba que sean exactamente los bytes analizados en el notebook."""
    if not path.is_file():
        raise FileNotFoundError(f"No existe el archivo de datos: {path}")
    if sha256_file(path) != DATA_SHA256:
        raise ValueError("El SHA-256 de wine.data no coincide con el dataset del notebook "
                         "(si se clonó en Windows, revisar que Git no haya cambiado los saltos de línea).")
    return validate_data(pd.read_csv(path, header=None, names=[TARGET] + FEATURES))


def audit_data(df: pd.DataFrame) -> dict:
    """Resumen medido de la calidad de los datos; se guarda como artefacto de cada corrida."""
    return {"rows": int(len(df)), "features": len(FEATURES),
            "missing_cells": int(df.isna().sum().sum()),
            "non_finite_values": int((~np.isfinite(df.to_numpy(dtype=float))).sum()),
            "duplicated_rows": int(df.duplicated().sum()),
            "duplicated_feature_vectors": int(df[FEATURES].duplicated().sum()),
            "class_counts": {str(k): int(v) for k, v in df[TARGET].value_counts().sort_index().items()},
            "dataset_sha256": DATA_SHA256}


def split_data(df: pd.DataFrame, seed: int, test_size: float) -> tuple:
    """Partición estratificada; verifica que train y test no compartan filas ni vectores."""
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df[TARGET], stratify=df[TARGET], test_size=test_size, random_state=seed)
    if not set(X_train.index).isdisjoint(X_test.index):
        raise RuntimeError("La partición comparte índices entre train y test.")
    if not set(pd.util.hash_pandas_object(X_train, index=False)).isdisjoint(
            pd.util.hash_pandas_object(X_test, index=False)):
        raise RuntimeError("La partición comparte vectores de predictoras entre train y test.")
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------- modelo

def penalty_name(l1_ratio: float) -> str:
    return "l2" if l1_ratio == 0 else "l1" if l1_ratio == 1 else "elasticnet"


def build_pipeline(args: argparse.Namespace) -> Pipeline:
    """Escalador + regresión logística; dentro del Pipeline el escalador solo ve datos de entrenamiento."""
    found = re.match(r"(\d+)\.(\d+)", sklearn.__version__)
    if found is None or tuple(map(int, found.groups())) < MIN_SKLEARN:
        raise RuntimeError(f"Se requiere scikit-learn >= 1.8 (instalado: {sklearn.__version__}).")
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=args.C, l1_ratio=args.l1_ratio, solver=args.solver,
                                     max_iter=args.max_iter, tol=args.tol, random_state=args.seed)),
    ])


def train_model(args: argparse.Namespace, X_train: pd.DataFrame, y_train: pd.Series) -> tuple:
    """F1 macro por validación cruzada estratificada dentro de train y ajuste del modelo final.

    Una advertencia de no convergencia detiene la corrida: no se registran modelos a medio optimizar.
    """
    pipeline = build_pipeline(args)
    folds = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=folds, scoring="f1_macro",
                                    n_jobs=1, error_score="raise")
        pipeline.fit(X_train, y_train)
    scaler = pipeline.named_steps["scale"]
    if int(scaler.n_samples_seen_) != len(X_train) or not np.allclose(scaler.mean_, X_train.mean()):
        raise RuntimeError("El escalador no se ajustó exclusivamente con train.")
    return pipeline, cv_scores


def evaluate_model(pipeline: Pipeline, cv_scores: np.ndarray, split: tuple, seed: int) -> tuple:
    """Métricas de CV (para elegir), de test (descriptivas) y de un control de clase mayoritaria."""
    X_train, X_test, y_train, y_test = split
    pred = pipeline.predict(X_test)
    control = DummyClassifier(strategy="most_frequent", random_state=seed).fit(X_train, y_train)
    pred_control = control.predict(X_test)
    model = pipeline.named_steps["model"]
    metrics = {
        "cv_f1_macro_mean": float(cv_scores.mean()),
        "cv_f1_macro_std": float(cv_scores.std(ddof=1)),
        "test_accuracy": float(accuracy_score(y_test, pred)),
        "test_f1_macro": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "train_accuracy": float(accuracy_score(y_train, pipeline.predict(X_train))),
        "control_test_accuracy": float(accuracy_score(y_test, pred_control)),
        "control_test_f1_macro": float(f1_score(y_test, pred_control, average="macro", zero_division=0)),
        "n_iter": int(np.max(model.n_iter_)),
        "n_coef_nonzero": int(np.count_nonzero(model.coef_)),
    }
    return metrics, pred, pred_control


def run_params(args: argparse.Namespace, split: tuple) -> dict:
    """Hiperparámetros y contexto de la corrida, tal como se registran en MLflow."""
    return {"C": args.C, "l1_ratio": args.l1_ratio, "solver": args.solver,
            "max_iter": args.max_iter, "tol": args.tol, "seed": args.seed,
            "test_size": args.test_size, "cv_folds": args.cv_folds,
            "penalty": penalty_name(args.l1_ratio), "scaler": "StandardScaler",
            "model": "LogisticRegression", "selection_metric": "cv_f1_macro_mean",
            "dataset": "UCI Wine (DOI 10.24432/C5PC7J)", "dataset_sha256": DATA_SHA256,
            "n_train": len(split[0]), "n_test": len(split[1]), "n_features": len(FEATURES),
            "train_script_sha256": sha256_file(Path(__file__)),
            "requirements_sha256": sha256_file(ROOT / "requirements.txt")}


# ---------------------------------------------------------------- artefactos

def save_confusion_matrix(y_true, y_pred, args: argparse.Namespace, folder: Path) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=CLASSES)
    pd.DataFrame(matrix, index=CLASSES, columns=CLASSES).rename_axis("real \\ predicho").to_csv(
        folder / "confusion_matrix.csv")
    fig = Figure(figsize=(5.5, 4.5))
    ax = fig.subplots()
    ConfusionMatrixDisplay(matrix, display_labels=CLASSES).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set(title=f"Test · C={args.C:g}, l1_ratio={args.l1_ratio:g}, {args.solver}",
           xlabel="Cultivar predicho", ylabel="Cultivar real")
    fig.tight_layout()
    fig.savefig(folder / "confusion_matrix.png", dpi=140)


def save_artifacts(folder: Path, args: argparse.Namespace, df: pd.DataFrame, split: tuple,
                   cv_scores: np.ndarray, metrics: dict, pred: np.ndarray,
                   pred_control: np.ndarray, params: dict) -> dict:
    """Escribe los artefactos locales de la corrida en evaluation/, data/ y config/."""
    X_train, X_test, y_train, y_test = split
    evaluation, data, config = folder / "evaluation", folder / "data", folder / "config"
    for sub in (evaluation, data, config):
        sub.mkdir(parents=True, exist_ok=False)
    write_json(evaluation / "metrics.json", metrics)
    write_json(evaluation / "classification_report.json", classification_report(
        y_test, pred, labels=CLASSES, output_dict=True, zero_division=0))
    pd.DataFrame({"fold": range(1, len(cv_scores) + 1), "f1_macro": cv_scores}).to_csv(
        evaluation / "cv_scores.csv", index=False)
    pd.DataFrame({"row_id": X_test.index, "y_true": y_test.to_numpy(), "y_pred": pred,
                  "y_pred_control": pred_control}).to_csv(evaluation / "test_predictions.csv", index=False)
    save_confusion_matrix(y_test, pred, args, evaluation)
    write_json(data / "data_audit.json", audit_data(df))
    pd.DataFrame({"row_id": df.index, "split": np.where(df.index.isin(X_train.index), "train", "test")}
                 ).to_csv(data / "split_membership.csv", index=False)
    write_json(config / "params.json", params)
    write_json(config / "command.json", {"argv": ["python", "train.py"] + list(args.invocation)})
    write_json(config / "environment.json", environment_info())
    return {"predictions_sha256": sha256_file(evaluation / "test_predictions.csv"),
            "split_sha256": sha256_file(data / "split_membership.csv"),
            "cv_scores_sha256": sha256_file(evaluation / "cv_scores.csv")}


# ---------------------------------------------------------------- MLflow

def import_mlflow():
    try:
        import mlflow
        import mlflow.sklearn  # noqa: F401  (registra el sabor sklearn)
    except ImportError as exc:
        raise RuntimeError("MLflow no está instalado: ejecuta 'python -m pip install -r requirements.txt' "
                           "(--no-tracking solo sirve para probar el cálculo sin MLflow).") from exc
    return mlflow


def setup_experiment(mlflow, output: Path, name: str) -> tuple:
    """Base SQLite local en <output>/mlflow.db y experimento con artefactos en <output>/mlartifacts."""
    from mlflow.tracking import MlflowClient
    output.mkdir(parents=True, exist_ok=True)
    tracking_uri = "sqlite:///" + (output / "mlflow.db").as_posix()
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    artifact_root = (output / "mlartifacts").as_uri()
    experiment = client.get_experiment_by_name(name)
    if experiment is None:
        experiment_id = client.create_experiment(name, artifact_location=artifact_root, tags={
            "mlflow.note.content": "TC5061 · Baseline Wine (UCI): StandardScaler + LogisticRegression. "
                                   f"Corridas parametrizadas con train.py · {AUTHOR}."})
    elif experiment.artifact_location.rstrip("/") != artifact_root.rstrip("/"):
        raise RuntimeError("La base de MLflow viene de otra carpeta; usa una carpeta de salida nueva.")
    else:
        experiment_id = experiment.experiment_id
    return client, experiment_id, tracking_uri


def run_name(args: argparse.Namespace) -> str:
    return f"C{args.C:g}_l1r{args.l1_ratio:g}_{args.solver}"


def log_run(mlflow, params: dict, metrics: dict, pipeline: Pipeline, split: tuple, pred: np.ndarray) -> str:
    """Registra parámetros, métricas y el modelo entrenado en el run activo; devuelve el URI del modelo."""
    from mlflow.models import infer_signature
    for name, value in params.items():
        mlflow.log_param(name, value)
    for name, value in metrics.items():
        mlflow.log_metric(name, value)
    X_train, X_test = split[0], split[1]
    versions = environment_info()["packages"]
    model_info = mlflow.sklearn.log_model(
        pipeline, name="model", input_example=X_train.head(3),
        signature=infer_signature(X_train, pipeline.predict(X_train)),
        pip_requirements=[f"{p}=={versions[p]}" for p in MODEL_PACKAGES if versions[p]])
    reloaded = mlflow.sklearn.load_model(model_info.model_uri)
    np.testing.assert_array_equal(reloaded.predict(X_test), pred)  # el modelo registrado predice igual
    return model_info.model_uri


def log_artifact_files(mlflow, folder: Path, data_path: Path) -> None:
    """Sube los artefactos locales conservando sus carpetas, más el dataset y el código usados."""
    for path in sorted(p for p in folder.rglob("*") if p.is_file()):
        subfolder = path.parent.relative_to(folder).as_posix()
        mlflow.log_artifact(str(path), artifact_path=None if subfolder == "." else subfolder)
    mlflow.log_artifact(str(data_path), artifact_path="data")
    mlflow.log_artifact(str(Path(__file__)), artifact_path="source")
    mlflow.log_artifact(str(ROOT / "requirements.txt"), artifact_path="source")


def list_artifacts(client, run_id: str, path: str | None = None) -> set:
    found = set()
    for item in client.list_artifacts(run_id, path):
        found |= list_artifacts(client, run_id, item.path) if item.is_dir else {item.path}
    return found


def verify_run(client, run_id: str, params: dict, metrics: dict, folder: Path, data_path: Path) -> None:
    """Relee el run desde MLflow y comprueba estado, parámetros, métricas y artefactos."""
    saved = client.get_run(run_id)
    if saved.info.status != "FINISHED":
        raise RuntimeError(f"El run {run_id} terminó en {saved.info.status}.")
    bad_params = [k for k, v in params.items() if saved.data.params.get(k) != str(v)]
    bad_metrics = [k for k, v in metrics.items() if k not in saved.data.metrics
                   or not math.isclose(saved.data.metrics[k], v, rel_tol=0, abs_tol=1e-12)]
    expected = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}
    expected |= {f"data/{data_path.name}", "source/train.py", "source/requirements.txt"}
    missing = sorted(expected - list_artifacts(client, run_id))
    if bad_params or bad_metrics or missing:
        raise RuntimeError(f"MLflow no coincide con lo calculado: parámetros={bad_params}, "
                           f"métricas={bad_metrics}, artefactos faltantes={missing}")


# ---------------------------------------------------------------- orquestación

def execute_run(args: argparse.Namespace, output: Path, mlflow=None, active_run=None) -> dict:
    """Datos -> modelo -> evaluación -> artefactos -> registro (si hay un run activo)."""
    run_id = active_run.info.run_id if active_run else None
    relative_folder = Path("runs") / (run_id or "untracked_" + uuid.uuid4().hex)
    folder = output / relative_folder
    df = load_data(args.data)
    split = split_data(df, args.seed, args.test_size)
    pipeline, cv_scores = train_model(args, split[0], split[2])
    metrics, pred, pred_control = evaluate_model(pipeline, cv_scores, split, args.seed)
    params = run_params(args, split)
    hashes = save_artifacts(folder, args, df, split, cv_scores, metrics, pred, pred_control, params)
    model_uri = log_run(mlflow, params, metrics, pipeline, split, pred) if mlflow else None
    summary = {"tracking_enabled": mlflow is not None, "experiment_name": args.experiment,
               "experiment_id": active_run.info.experiment_id if active_run else None,
               "run_id": run_id, "run_name": run_name(args), "model_uri": model_uri,
               "params": params, "metrics": metrics, "environment": environment_info(),
               "artifact_dir": relative_folder.as_posix(), **hashes}
    write_json(folder / "run_summary.json", summary)
    if mlflow:
        log_artifact_files(mlflow, folder, args.data)
    return summary


def run(args: argparse.Namespace) -> dict:
    output = args.output.resolve()
    if args.no_tracking:
        return execute_run(args, output)
    mlflow = import_mlflow()
    client, experiment_id, tracking_uri = setup_experiment(mlflow, output, args.experiment)
    tags = {"mlflow.user": AUTHOR, "mlflow.source.name": "train.py", "course": "TC5061",
            "scope": "individual", "penalty": penalty_name(args.l1_ratio)}
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name(args), tags=tags) as active:
        summary = execute_run(args, output, mlflow, active)
    verify_run(client, summary["run_id"], summary["params"], summary["metrics"],
               output / summary["artifact_dir"], args.data)
    return {**summary, "tracking_uri": tracking_uri}


# ---------------------------------------------------------------- línea de comandos

def positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("debe ser un número positivo y finito")
    return number


def unit_interval(value: str) -> float:
    number = float(value)
    if not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("debe estar entre 0 y 1")
    return number


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("debe ser un entero positivo")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train.py", formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Entrena el baseline de Wine (StandardScaler + LogisticRegression) y registra "
                    "parámetros, métricas, artefactos y el modelo en MLflow. Sin argumentos usa la "
                    "configuración del baseline.",
        epilog="Ejemplo: python train.py --C 0.1 --l1-ratio 1 --solver saga --seed 42")
    model = parser.add_argument_group("hiperparámetros del modelo")
    model.add_argument("--C", type=positive_float, default=DEFAULTS["C"],
                       help="Inverso de la fuerza de regularización: valores menores regularizan más.")
    model.add_argument("--l1-ratio", type=unit_interval, default=DEFAULTS["l1_ratio"],
                       help="Tipo de penalización: 0 = L2, 1 = L1, intermedio = elastic-net. "
                            "Valores mayores que 0 requieren --solver saga.")
    model.add_argument("--solver", choices=SOLVERS, default=DEFAULTS["solver"],
                       help="Algoritmo de optimización de la regresión logística.")
    model.add_argument("--max-iter", type=positive_int, default=DEFAULTS["max_iter"],
                       help="Máximo de iteraciones del optimizador; si no converge, la corrida se detiene.")
    model.add_argument("--tol", type=positive_float, default=DEFAULTS["tol"],
                       help="Tolerancia del criterio de paro del optimizador.")
    evaluation = parser.add_argument_group("semilla y evaluación")
    evaluation.add_argument("--seed", type=int, default=DEFAULTS["seed"],
                            help="Semilla única de la partición train/test, los folds de CV y el modelo.")
    evaluation.add_argument("--test-size", type=float, default=DEFAULTS["test_size"],
                            help="Fracción estratificada reservada para test (entre 0.05 y 0.5).")
    evaluation.add_argument("--cv-folds", type=int, choices=range(2, 11), metavar="{2..10}",
                            default=DEFAULTS["cv_folds"],
                            help="Folds de validación cruzada estratificada dentro de train.")
    tracking = parser.add_argument_group("datos y registro")
    tracking.add_argument("--data", type=Path, default=Path("data/wine.data"),
                          help="Archivo wine.data de UCI (rutas relativas: desde la carpeta del repositorio).")
    tracking.add_argument("--output", type=Path, default=Path("outputs"),
                          help="Carpeta de la base de MLflow (mlflow.db), sus artefactos y las copias locales.")
    tracking.add_argument("--experiment", default=EXPERIMENT_NAME,
                          help="Experimento de MLflow donde se agrupan las corridas.")
    tracking.add_argument("--no-tracking", action="store_true",
                          help="Calcula sin MLflow, solo para pruebas; no crea runs.")
    return parser


def parse_args(argv=None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 0 <= args.seed < 2**32:
        parser.error("--seed debe estar entre 0 y 4294967295.")
    if not 0.05 <= args.test_size <= 0.50:
        parser.error("--test-size debe estar entre 0.05 y 0.50.")
    if args.l1_ratio > 0 and args.solver != "saga":
        parser.error("--l1-ratio mayor que 0 requiere --solver saga (lbfgs y newton-cg solo admiten L2).")
    args.data = args.data if args.data.is_absolute() else ROOT / args.data
    args.output = args.output if args.output.is_absolute() else ROOT / args.output
    args.invocation = list(sys.argv[1:] if argv is None else argv)
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        summary = run(args)
    except (ValueError, OSError, RuntimeError, ConvergenceWarning) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, sort_keys=True))  # ASCII: se lee igual en cualquier consola
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
