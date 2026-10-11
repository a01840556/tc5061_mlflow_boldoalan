"""Pruebas con unittest (no requieren pytest). La de MLflow se omite si MLflow no está instalado."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import train  # noqa: E402
import run_experiments  # noqa: E402


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = train.load_data()

    def test_original_dataset_and_class_counts(self):
        self.assertEqual(self.df.shape, (178, 14))
        self.assertEqual(self.df.cultivar.value_counts().to_dict(), {2: 71, 1: 59, 3: 48})

    def test_audit_is_measured(self):
        audit = train.audit_data(self.df)
        self.assertEqual((audit["missing_cells"], audit["duplicated_rows"], audit["non_finite_values"]), (0, 0, 0))
        self.assertEqual(audit["class_counts"], {"1": 59, "2": 71, "3": 48})

    def test_rejects_changed_dataset_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "wine.data"
            path.write_bytes(train.DATA_PATH.read_bytes().replace(b"\n", b"\r\n"))  # como un checkout CRLF
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                train.load_data(path)

    def test_rejects_invalid_values_and_labels(self):
        for column, value in [("alcohol", np.nan), ("alcohol", np.inf), ("cultivar", 1.5)]:
            with self.subTest(column=column, value=value):
                data = self.df.astype(float).copy()
                data.loc[0, column] = value
                with self.assertRaises(ValueError):
                    train.validate_data(data)

    def test_rejects_duplicate_features(self):
        data = self.df.copy()
        data.loc[1, train.FEATURES] = data.loc[0, train.FEATURES]
        with self.assertRaisesRegex(ValueError, "duplicados"):
            train.validate_data(data)

    def test_seed_changes_split_but_repeats_exactly(self):
        a = train.split_data(self.df, 42, 0.2)
        b = train.split_data(self.df, 42, 0.2)
        c = train.split_data(self.df, 7, 0.2)
        self.assertEqual((len(a[0]), len(a[1])), (142, 36))
        self.assertEqual(a[0].index.tolist(), b[0].index.tolist())
        self.assertNotEqual(a[0].index.tolist(), c[0].index.tolist())
        self.assertFalse(set(a[0].index) & set(a[1].index))


class CliTests(unittest.TestCase):
    def test_defaults_match_notebook_baseline(self):
        args = train.parse_args([])
        self.assertEqual((args.C, args.l1_ratio, args.solver, args.max_iter, args.tol, args.seed,
                          args.test_size, args.cv_folds), (1.0, 0.0, "lbfgs", 1000, 1e-4, 42, 0.2, 5))
        self.assertFalse(args.no_tracking)
        self.assertEqual(args.data, train.DATA_PATH)

    def test_help_documents_every_hyperparameter(self):
        text = train.build_parser().format_help()
        for flag in ["--C", "--l1-ratio", "--solver", "--max-iter", "--tol", "--seed"]:
            self.assertIn(flag, text)
        self.assertIn("(default: 42)", text)

    def test_invalid_cli_arguments(self):
        for argv in [["--C", "0"], ["--C", "nan"], ["--tol", "inf"], ["--max-iter", "0"],
                     ["--seed", "-1"], ["--seed", "4294967296"], ["--solver", "liblinear"],
                     ["--test-size", "nan"], ["--test-size", "0.9"], ["--cv-folds", "1"],
                     ["--l1-ratio", "1.5"], ["--l1-ratio", "1"], ["--l1-ratio", "0.5", "--solver", "newton-cg"]]:
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    train.parse_args(argv)

    def test_cli_hyperparameters_reach_model(self):
        args = train.parse_args(["--C", "0.1", "--l1-ratio", "1", "--solver", "saga", "--max-iter", "700",
                                 "--tol", "0.001", "--seed", "7"])
        model = train.build_pipeline(args).named_steps["model"]
        self.assertEqual((model.C, model.l1_ratio, model.solver, model.max_iter, model.tol, model.random_state),
                         (0.1, 1.0, "saga", 700, 0.001, 7))


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.split = train.split_data(train.load_data(), 42, 0.2)

    def test_pipeline_scaler_sees_train_only(self):
        args = train.parse_args(["--no-tracking"])
        model, scores = train.train_model(args, self.split[0], self.split[2])
        self.assertEqual(model.named_steps["scale"].n_samples_seen_, 142)
        np.testing.assert_allclose(model.named_steps["scale"].mean_, self.split[0].mean())
        self.assertEqual(len(scores), 5)
        metrics, pred, _ = train.evaluate_model(model, scores, self.split, 42)
        self.assertGreater(metrics["test_f1_macro"], metrics["control_test_f1_macro"])
        self.assertEqual(len(pred), 36)

    def test_l1_penalty_produces_sparse_model(self):
        args = train.parse_args(["--no-tracking", "--l1-ratio", "1", "--solver", "saga"])
        model, scores = train.train_model(args, self.split[0], self.split[2])
        metrics, _, _ = train.evaluate_model(model, scores, self.split, 42)
        self.assertLess(metrics["n_coef_nonzero"], 39)
        self.assertGreater(metrics["n_coef_nonzero"], 0)

    def test_nonconvergence_is_failure(self):
        args = train.parse_args(["--no-tracking", "--max-iter", "1"])
        with self.assertRaises(train.ConvergenceWarning):
            train.train_model(args, self.split[0], self.split[2])


class BatchTests(unittest.TestCase):
    def test_batch_has_ten_distinct_valid_configurations_and_one_repetition(self):
        planned = run_experiments.planned_runs()
        compared = [tuple(arguments) for role, _, arguments in planned if role == "comparacion"]
        self.assertEqual(len(compared), 10)
        self.assertEqual(len(set(compared)), 10)
        self.assertEqual(planned[-1][0], "repeticion")
        self.assertEqual(tuple(planned[-1][2]), compared[run_experiments.REPEATED])
        self.assertEqual(train.parse_args(list(compared[run_experiments.REPEATED])).C, 1.0)
        for arguments in compared:
            train.parse_args(list(arguments))  # todas son configuraciones válidas de la CLI

    def test_two_identical_cli_executions(self):
        with tempfile.TemporaryDirectory() as temp:
            command = [sys.executable, str(ROOT / "train.py"), "--no-tracking", "--output", temp]
            runs = [json.loads(subprocess.run(command, text=True, capture_output=True, check=True).stdout)
                    for _ in range(2)]
            self.assertTrue(all(run_experiments.verify_repetition(*runs).values()))
            self.assertIsNone(runs[0]["run_id"])
            folder = Path(temp) / runs[0]["artifact_dir"]
            for name in ["evaluation/confusion_matrix.png", "evaluation/metrics.json",
                         "config/params.json", "data/data_audit.json", "run_summary.json"]:
                self.assertTrue((folder / name).is_file(), name)


@unittest.skipUnless(importlib.util.find_spec("mlflow"), "MLflow no instalado: integración no ejecutada")
class MlflowIntegrationTests(unittest.TestCase):
    def test_real_mlflow_tracking_reload_and_artifacts(self):
        # En Windows MLflow deja abierta la base SQLite: no fallar al limpiar la carpeta temporal.
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp:
            args = train.parse_args(["--output", temp, "--experiment", "tc5061_integration_test"])
            result = train.run(args)  # run() relee el run y falla si algo no coincide
            self.assertTrue(result["tracking_enabled"])
            self.assertIsNotNone(result["run_id"])
            self.assertTrue(result["model_uri"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
