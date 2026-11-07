"""Synthetic C2 regression tests for the TR-01..TR-05 design obligations."""

import csv
import hashlib
import json
import math
import random
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from speedlimit.model_trainer import (
    DEFAULT_SEED,
    TrainingSettings,
    load_model,
    predict_near_accidents,
    train_models,
)


FEATURES = (
    "illuminance_millilux", "water_micrometers", "temperature_celsius", "historical_speed_kmh",
)
FIELDS = (
    "year", "target_hour", "prediction_cutoff", *FEATURES, "near_accident_count",
    "event_coverage", "eligible_for_training", "fixture_private_source_value",
)
PATTERNS = (
    (100, 1500, -2, 40, 4),
    (500, 1000.1, -0.01, 50, 3),
    (499.9, 900, 1, 60, 2),
    (500, 1001, 0, 70, 1),
    (1000, 0, 5, 80, 0),
    (1500, 100, 7, 60, 0),
)
SETTINGS = TrainingSettings(hidden_units=5, epochs=55, learning_rate=0.08)


def _read_predictions(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _fixture_row(year, day, hour, pattern):
    timestamp = f"01-{day:02}T{hour:02}:00:00"
    return {
        "year": year, "target_hour": timestamp, "prediction_cutoff": timestamp,
        **dict(zip(FEATURES, pattern)), "near_accident_count": pattern[-1],
        "eligible_for_training": True, "event_coverage": "assumed_complete",
        "fixture_private_source_value": "PRIVATE-FIXTURE-VALUE",
    }


class ModelTrainerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix=".c2-test-", dir=Path(__file__).resolve().parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.prepared = self.root / "prepared"
        self.prepared.mkdir()
        self.output = self.root / "model"
        self.rows = [
            _fixture_row("N", day, hour, pattern)
            for day in range(1, 32) for hour, pattern in enumerate(PATTERNS)
        ]
        self.rows += [
            _fixture_row("N+1", 26, 0, (100, 1500, -2, 40, 2)),
            _fixture_row("N+1", 26, 1, (500, 1000.1, -0.01, 50, 4)),
            _fixture_row("N+1", 26, 2, (499.9, 900, 1, 60, 0)),
            _fixture_row("N+1", 26, 3, (600, 1001, 0, 70, 0)),
            {**_fixture_row("N+1", 26, 4, PATTERNS[4]), "near_accident_count": "", "eligible_for_training": False},
        ]
        self.write_inputs()

    def write_inputs(self, rows=None, summary_updates=None, fields=FIELDS):
        rows = self.rows if rows is None else rows
        with (self.prepared / "feature_preview.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        summary = {
            "schema_version": 1, "component": "C1 DataPreparation", "training_ready": True,
            "blockers": [], "config": {"prediction_cutoff": "target_hour_start", "fixture": True},
            "sources": {"readings_N": {"filename": "fixture.csv", "sha256": "0" * 64}},
            "feature_preview_sha256": hashlib.sha256((self.prepared / "feature_preview.csv").read_bytes()).hexdigest(),
            "feature_rows_by_year": dict(Counter(row["year"] for row in rows)),
            "eligible_feature_rows_by_year": dict(Counter(row["year"] for row in rows if row["eligible_for_training"])),
        }
        summary.update(summary_updates or {})
        (self.prepared / "preparation_summary.json").write_text(json.dumps(summary, sort_keys=True), encoding="utf-8")

    def train(self, output=None, settings=SETTINGS):
        return train_models(self.prepared, output or self.output, seed=DEFAULT_SEED, settings=settings)

    def test_TR_01_fixed_partitions_and_no_test_or_next_year_leakage(self):
        random.Random(12).shuffle(self.rows)
        self.write_inputs()
        report = self.train()
        model = load_model(self.output / "model.json")
        predictions = _read_predictions(self.output / "predictions.csv")
        self.assertEqual(report["split_sizes"], {"train": 120, "validation": 30, "test": 36, "n_plus_1": 4})
        for row in predictions:
            day = int(row["target_hour"][3:5])
            if row["year"] == "N+1":
                self.assertEqual(row["split"], "n_plus_1")
            else:
                first, last = {"train": (1, 20), "validation": (21, 25), "test": (26, 31)}[row["split"]]
                self.assertLessEqual(first, day)
                self.assertLessEqual(day, last)
        for index, mean in enumerate(model["scaling"]["mean"]):
            self.assertAlmostEqual(mean, sum(pattern[index] for pattern in PATTERNS) / len(PATTERNS))
        self.assertEqual(sum(item["count"] for item in model["training_support_by_speed"].values()), 120)
        self.assertEqual(sum(item["weather_triggered_count"] for item in model["training_support_by_speed"].values()), 60)

        for row in self.rows:
            if row["eligible_for_training"] and (row["year"] == "N+1" or int(row["target_hour"][3:5]) >= 26):
                row["illuminance_millilux"] = 42_000
                row["near_accident_count"] = 100
        self.write_inputs()
        alternate = self.root / "holdout_changed"
        changed = self.train(alternate)
        changed_model = load_model(alternate / "model.json")
        for key in ("scaling", "weights", "fit", "baseline_mean", "training_support_by_speed"):
            self.assertEqual(changed_model[key], model[key])
        self.assertEqual(changed["metrics"]["validation"], report["metrics"]["validation"])
        self.assertNotEqual(changed["metrics"]["test"], report["metrics"]["test"])
        self.assertNotEqual(changed["metrics"]["n_plus_1"], report["metrics"]["n_plus_1"])

    def test_TR_02_same_seed_inputs_and_settings_produce_identical_artifacts(self):
        first = self.train()
        other = self.root / "rerun"
        second = self.train(other)
        self.assertEqual(first, second)
        for name in ("model.json", "metrics.json", "predictions.csv"):
            with self.subTest(artifact=name):
                self.assertEqual((self.output / name).read_bytes(), (other / name).read_bytes())
        self.assertEqual(first, json.loads((self.output / "metrics.json").read_bytes()))
        for name, digest in first["artifact_sha256"].items():
            self.assertEqual(digest, hashlib.sha256((self.output / name).read_bytes()).hexdigest())
        self.assertEqual(first["provenance"]["feature_preview_sha256"],
                         hashlib.sha256((self.prepared / "feature_preview.csv").read_bytes()).hexdigest())

    def test_TR_03_models_use_same_rows_and_metrics_are_recomputable(self):
        report = self.train()
        predictions = _read_predictions(self.output / "predictions.csv")
        expected_baseline = 10 / 6
        self.assertEqual(report["baseline_mean"], expected_baseline)
        for split, metrics in report["metrics"].items():
            samples = [row for row in predictions if row["split"] == split]
            labels = [int(row["near_accident_count"]) for row in samples]
            self.assertEqual(metrics["count"], len(samples))
            for model_name in ("baseline", "neural_network"):
                model = metrics[model_name]
                fitted = [float(row[f"{model_name}_prediction"]) for row in samples]
                residuals = [estimate - actual for estimate, actual in zip(fitted, labels)]
                self.assertEqual(model["count"], len(samples))
                self.assertAlmostEqual(model["mae"], sum(map(abs, residuals)) / len(samples))
                self.assertAlmostEqual(model["mean_signed_error"], sum(residuals) / len(samples))
                self.assertAlmostEqual(model["rmse"], math.sqrt(sum(error ** 2 for error in residuals) / len(samples)))
                self.assertAlmostEqual(model["count_calibration"]["mean_prediction"], sum(fitted) / len(samples))
                self.assertAlmostEqual(model["count_calibration"]["mean_actual"], sum(labels) / len(samples))
            self.assertEqual(metrics["weather_triggered"]["count"], sum(row["weather_triggered"] == "True" for row in samples))
        self.assertLess(report["metrics"]["test"]["neural_network"]["mae"], report["metrics"]["test"]["baseline"]["mae"])
        self.assertEqual(report["next_year"]["matching_N_feature_rows"], 3)
        self.assertEqual(report["next_year"]["excluded_rows"], 1)
        self.assertIn("black_ice", {row["weather_branch"] for row in predictions})
        self.assertNotIn("PRIVATE-FIXTURE-VALUE", (self.output / "predictions.csv").read_text(encoding="utf-8"))

    def test_TR_04_reloaded_model_reproduces_saved_predictions(self):
        report = self.train()
        self.assertLessEqual(1, report["selected_epoch"])
        self.assertLessEqual(report["selected_epoch"], SETTINGS.epochs)
        model = load_model(self.output / "model.json")
        predictions = _read_predictions(self.output / "predictions.csv")
        for hour, pattern in enumerate(PATTERNS):
            row = next(item for item in predictions if item["year"] == "N" and item["target_hour"] == f"01-26T{hour:02}:00:00")
            estimate = predict_near_accidents(model, **dict(zip(FEATURES, pattern)))
            self.assertEqual(estimate, float(row["neural_network_prediction"]))
            self.assertGreater(estimate, 0)
        validation = [row for row in predictions if row["split"] == "validation"]
        losses = [float(row["neural_network_prediction"]) - int(row["near_accident_count"]) *
                  math.log(float(row["neural_network_prediction"])) for row in validation]
        self.assertAlmostEqual(report["fit"]["validation_poisson_loss"], sum(losses) / len(losses))

    def test_TR_05_unready_preparation_fails_without_model_artifacts(self):
        self.write_inputs(summary_updates={"training_ready": False, "blockers": ["event-hour convention is unresolved"]})
        with self.assertRaisesRegex(ValueError, "not training-ready.*event-hour convention"):
            self.train()
        self.assertFalse(self.output.exists())

    def test_TR_05_bad_eligible_row_is_rejected(self):
        cases = (
            ("near_accident_count", "", "must be numeric"),
            ("near_accident_count", "1.5", "nonnegative, representable integer"),
            ("near_accident_count", -1, "nonnegative, representable integer"),
            ("temperature_celsius", "NaN", "must be finite"),
            ("water_micrometers", -1, "must be nonnegative"),
            ("historical_speed_kmh", 0, "speed must be positive"),
            ("event_coverage", "unknown", "configured event coverage"),
            ("year", "N+2", "symbolic year"),
            ("target_hour", "01-01T24:00:00", "valid January hour"),
            ("prediction_cutoff", "01-01T01:00:00", "prediction cutoff"),
        )
        for field, value, message in cases:
            with self.subTest(field=field, value=value):
                self.write_inputs([{**self.rows[0], field: value}, *self.rows[1:]])
                with self.assertRaisesRegex(ValueError, message):
                    self.train()
                self.assertFalse(self.output.exists())

    def test_TR_05_duplicate_missing_or_inconsistent_input_is_rejected(self):
        self.write_inputs([*self.rows, self.rows[0]])
        with self.assertRaisesRegex(ValueError, "duplicate target hour"):
            self.train()
        self.write_inputs(fields=tuple(name for name in FIELDS if name != "temperature_celsius"))
        with self.assertRaisesRegex(ValueError, "missing or duplicate columns"):
            self.train()
        self.write_inputs(summary_updates={"eligible_feature_rows_by_year": {"N": 1, "N+1": 4}})
        with self.assertRaisesRegex(ValueError, "disagrees with eligible_feature_rows_by_year"):
            self.train()
        self.write_inputs([row for row in self.rows if row["year"] == "N+1" or int(row["target_hour"][3:5]) not in range(21, 26)])
        with self.assertRaisesRegex(ValueError, "No eligible Year N rows in validation"):
            self.train()
        self.assertFalse(self.output.exists())

    def test_TR_05_swapped_preview_with_same_row_counts_is_rejected(self):
        path = self.prepared / "feature_preview.csv"
        text = path.read_text(encoding="utf-8")
        changed = text.replace(",100,1500,-2,40,4,", ",101,1500,-2,40,4,", 1)
        self.assertNotEqual(changed, text)
        path.write_text(changed, encoding="utf-8", newline="\n")
        with self.assertRaisesRegex(ValueError, "does not match feature_preview_sha256"):
            self.train()
        self.assertFalse(self.output.exists())

    def test_TR_05_load_rejects_incompatible_order_or_weights(self):
        self.train()
        artifact = load_model(self.output / "model.json")
        path = self.root / "bad_model.json"
        path.write_text(json.dumps({**artifact, "feature_names": list(reversed(FEATURES))}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "feature/target contract"):
            load_model(path)
        artifact["weights"]["hidden_weights"][0].pop()
        path.write_text(json.dumps(artifact), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "hidden weights"):
            load_model(path)

    def test_optional_next_year_and_empty_weather_subsets_are_null_safe(self):
        rows = [
            {**row, "illuminance_millilux": 1000, "water_micrometers": 0}
            for row in self.rows if row["year"] == "N"
        ]
        self.write_inputs(rows)
        report = self.train()
        self.assertEqual(report["next_year"]["status"], "unavailable_no_eligible_rows")
        empty = report["metrics"]["n_plus_1"]
        self.assertEqual(empty["count"], 0)
        self.assertIsNone(empty["neural_network"]["mae"])
        weather = report["metrics"]["test"]["weather_triggered"]
        self.assertEqual(weather["count"], 0)
        self.assertIsNone(weather["baseline"]["count_calibration"]["mean_actual"])


if __name__ == "__main__":
    unittest.main()
