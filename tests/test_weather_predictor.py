"""Synthetic support and inference fixtures for the C4 WP-01..WP-05 cases."""

import csv
import hashlib
import json
import math
import tempfile
import unittest
from collections import Counter
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

from speedlimit.model_trainer import FEATURE_NAMES, TrainingSettings, train_models
from speedlimit.weather_predictor import CANDIDATE_SPEEDS, WeatherPredictor, build_support
from speedlimit.weather_router import evaluate_measurements


_FIELDS = (
    "year", "target_hour", "prediction_cutoff", *FEATURE_NAMES,
    "near_accident_count", "event_coverage", "eligible_for_training", "private_fixture_identifier",
)
_QUERY = (400, 1100, -2)


def _row(year, day, hour, weather, speed):
    timestamp = f"01-{day:02}T{hour:02}:00:00"
    return {
        "year": year, "target_hour": timestamp, "prediction_cutoff": timestamp,
        **dict(zip(FEATURE_NAMES, (*weather, speed))), "near_accident_count": 1,
        "event_coverage": "assumed_complete", "eligible_for_training": True,
        "private_fixture_identifier": "PRIVATE-C4-ROW",
    }


class WeatherPredictorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix=".c4-test-", dir=Path(__file__).resolve().parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.prepared = self.root / "prepared"
        self.model_path = self.root / "fit" / "model.json"
        self.support_path = self.root / "weather_support.json"
        self.active = evaluate_measurements("0.4", 1100, -2)
        self.build_fixture()

    def build_fixture(self, overrides=None):
        self.prepared.mkdir(exist_ok=True)
        overrides = overrides or {}
        rows = [
            _row("N", 1, 3 * index + offset, weather, speed)
            for index, speed in enumerate(CANDIDATE_SPEEDS)
            for offset, weather in enumerate(overrides.get(speed, [_QUERY] * 3))
        ]
        rows += [
            {**_row("N", 2, 0, _QUERY, 80), "eligible_for_training": False, "near_accident_count": ""},
            _row("N", 21, 0, (222222, 100, 4), 80),
            _row("N", 26, 0, (333333, 200, 5), 80),
            _row("N+1", 1, 0, (444444, 300, 6), 80),
        ]
        preview = self.prepared / "feature_preview.csv"
        with preview.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        summary = {
            "schema_version": 1, "component": "C1 DataPreparation", "training_ready": True,
            "blockers": [], "config": {"fixture": True}, "sources": {},
            "feature_preview_sha256": hashlib.sha256(preview.read_bytes()).hexdigest(),
            "feature_rows_by_year": dict(Counter(row["year"] for row in rows)),
            "eligible_feature_rows_by_year": dict(Counter(row["year"] for row in rows if row["eligible_for_training"])),
        }
        (self.prepared / "preparation_summary.json").write_text(json.dumps(summary), encoding="utf-8")
        train_models(self.prepared, self.model_path.parent, settings=TrainingSettings(hidden_units=2, epochs=1))
        return build_support(self.model_path, self.prepared, self.support_path)

    def evaluate_rates(self, rates, *, predictor=None, decision=None):
        predictor = predictor or WeatherPredictor(self.model_path, self.support_path)
        decision = decision or self.active
        with patch("speedlimit.weather_predictor.predict_near_accidents", side_effect=lambda model, **kwargs: rates[kwargs["historical_speed_kmh"]]) as predict:
            answer = predictor.evaluate(decision)
        return answer, [call.kwargs["historical_speed_kmh"] for call in predict.call_args_list]

    def test_WP_01_chooses_highest_supported_speed_under_threshold(self):
        result, calls = self.evaluate_rates({70: 1.3, 60: 0.9, 50: 0.7, 40: 0.6})
        self.assertEqual((result.status, result.limit_kmh), ("recommended", 60))
        self.assertEqual(calls, [70, 60, 50, 40])
        self.assertEqual([item.speed_kmh for item in result.candidates], calls)
        self.assertEqual([item.predicted_count for item in result.candidates], [1.3, 0.9, 0.7, 0.6])
        self.assertEqual([item.neighbors for item in result.candidates], [3, 3, 3, 3])
        self.assertEqual(result.model_id, hashlib.sha256(self.model_path.read_bytes()).hexdigest())

    def test_WP_02_includes_exactly_one_and_rejects_next_float_above(self):
        exact, _ = self.evaluate_rates({70: 1.0, 60: 0.7, 50: 0.6, 40: 0.5})
        above, _ = self.evaluate_rates({70: math.nextafter(1.0, math.inf), 60: 1.0, 50: 0.6, 40: 0.5})
        self.assertEqual(exact.limit_kmh, 70)
        self.assertEqual(above.limit_kmh, 60)
        self.assertEqual(above.candidates[0].status, "over_threshold")

    def test_WP_03_all_supported_speeds_above_threshold_need_review(self):
        result, calls = self.evaluate_rates({70: 3.0, 60: 2.5, 50: 2.0, 40: 1.2})
        self.assertEqual((result.status, result.limit_kmh), ("review_needed", None))
        self.assertEqual(calls, [70, 60, 50, 40])
        self.assertTrue(all(candidate.status == "over_threshold" for candidate in result.candidates))

    def test_WP_04_missing_or_incompatible_model_never_runs_inference(self):
        missing = WeatherPredictor(self.root / "missing.json", self.support_path)
        result, calls = self.evaluate_rates({}, predictor=missing)
        self.assertEqual((result.status, result.limit_kmh, calls), ("review_needed", None, []))
        model = json.loads(self.model_path.read_bytes())
        model["feature_names"].reverse()
        self.model_path.write_text(json.dumps(model), encoding="utf-8")
        result, calls = self.evaluate_rates({})
        self.assertIn("feature/target contract", result.reason)
        self.assertEqual((result.status, result.limit_kmh, calls), ("review_needed", None, []))

    def test_WP_05_invalid_predictions_fail_closed_even_if_other_speeds_qualify(self):
        for invalid in (float("nan"), float("inf"), -0.1, True):
            with self.subTest(invalid=invalid):
                result, _ = self.evaluate_rates({70: invalid, 60: 0.9, 50: 0.8, 40: 0.7})
                self.assertEqual((result.status, result.limit_kmh), ("review_needed", None))
                self.assertEqual(result.candidates[0].status, "invalid_prediction")
                self.assertIsNone(result.candidates[0].predicted_count)

    def test_WP_05_marginal_ranges_do_not_imply_joint_nearby_support(self):
        self.build_fixture({70: [(100, 1100, -2), (400, 1450, -2), (400, 1100, -7)]})
        answer, calls = self.evaluate_rates({60: 0.9, 50: 0.8, 40: 0.7})
        self.assertEqual(calls, [60, 50, 40])
        self.assertEqual(answer.limit_kmh, 60)
        self.assertEqual(answer.candidates[0].status, "unsupported")
        self.assertEqual(answer.candidates[0].neighbors, 0)
        broader = WeatherPredictor(self.model_path, self.support_path, radius=4.0)
        alternate, _ = self.evaluate_rates({70: 0.95, 60: 0.9, 50: 0.8, 40: 0.7}, predictor=broader)
        self.assertEqual(alternate.limit_kmh, 70)

    def test_other_weather_branches_are_excluded_even_with_a_broad_radius(self):
        self.build_fixture({70: [(500, 1100, -2)] * 3})
        predictor = WeatherPredictor(self.model_path, self.support_path, radius=100.0)
        answer, calls = self.evaluate_rates({60: 0.9, 50: 0.8, 40: 0.7}, predictor=predictor)
        self.assertEqual(calls, [60, 50, 40])
        self.assertEqual(answer.limit_kmh, 60)
        self.assertEqual(answer.candidates[0].neighbors, 0)

    def test_group_range_prevents_extrapolation_despite_three_close_points(self):
        outside = evaluate_measurements("0.401", 1100, -2)
        result, calls = self.evaluate_rates({}, decision=outside)
        self.assertEqual((result.status, result.limit_kmh, calls), ("review_needed", None, []))
        self.assertTrue(all(item.neighbors == 3 and item.status == "unsupported" for item in result.candidates))
        self.assertTrue(all("outside the observed range" in item.reason for item in result.candidates))

    def test_support_export_is_reproducible_and_uses_only_training_features(self):
        other = self.root / "copy.json"
        exported = build_support(self.model_path, self.prepared, other)
        self.assertEqual(other.read_bytes(), self.support_path.read_bytes())
        self.assertEqual(exported["training_rows"], 12)
        self.assertEqual(exported["fit_partition"], "N/train")
        self.assertEqual(Counter(record["features"][-1] for record in exported["records"]), {70: 3, 60: 3, 50: 3, 40: 3})
        self.assertTrue(all(set(record) == {"features", "weather_branch"} for record in exported["records"]))
        self.assertNotIn("PRIVATE-C4-ROW", other.read_text(encoding="utf-8"))
        self.assertEqual(exported["model_sha256"], hashlib.sha256(self.model_path.read_bytes()).hexdigest())

    def test_support_export_rejects_swapped_preview_and_model_metadata(self):
        preview = self.prepared / "feature_preview.csv"
        original = preview.read_bytes()
        preview.write_bytes(original.replace(b",400,", b",401,", 1))
        output = self.root / "unexpected.json"
        with self.assertRaisesRegex(ValueError, "SHA-256 does not match"):
            build_support(self.model_path, self.prepared, output)
        self.assertFalse(output.exists())
        preview.write_bytes(original)
        model = json.loads(self.model_path.read_bytes())
        model["training_support_by_speed"]["70"]["weather_branch_counts"] = {"normal": 3}
        self.model_path.write_text(json.dumps(model), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "speed, branch, count or ranges disagree"):
            build_support(self.model_path, self.prepared, output)
        self.assertFalse(output.exists())

    def test_model_and_support_hash_mismatch_or_corrupt_records_never_run_inference(self):
        original = self.model_path.read_bytes()
        self.model_path.write_bytes(original + b"\n")
        result, calls = self.evaluate_rates({})
        self.assertIn("model SHA-256", result.reason)
        self.assertEqual((result.status, result.limit_kmh, calls), ("review_needed", None, []))
        self.model_path.write_bytes(original)
        support = json.loads(self.support_path.read_bytes())
        support["records"][0]["features"][0] += 1
        self.support_path.write_text(json.dumps(support), encoding="utf-8")
        result, calls = self.evaluate_rates({})
        self.assertIn("records SHA-256", result.reason)
        self.assertEqual((result.status, result.limit_kmh, calls), ("review_needed", None, []))

    def test_inactive_or_invalid_router_decision_never_loads_model(self):
        predictor = WeatherPredictor(self.model_path, self.support_path)
        with patch.object(predictor, "_load", side_effect=AssertionError("Unexpected model access")):
            inactive = predictor.evaluate(evaluate_measurements(1, 0, 5))
            nonfinite = predictor.evaluate(replace(self.active, water_um=Decimal("NaN")))
            bad_flags = predictor.evaluate(replace(self.active, dark=False))
        self.assertIn("did not activate", inactive.reason)
        self.assertIn("finite Decimal", nonfinite.reason)
        self.assertIn("flags disagree", bad_flags.reason)
        for result in (inactive, nonfinite, bad_flags):
            self.assertEqual((result.status, result.limit_kmh, result.model_id), ("review_needed", None, None))

    def test_failed_load_can_retry_and_real_model_prediction_remains_finite(self):
        late = self.root / "late_model.json"
        predictor = WeatherPredictor(late, self.support_path)
        self.assertEqual(predictor.evaluate(self.active).status, "review_needed")
        late.write_bytes(self.model_path.read_bytes())
        result = predictor.evaluate(self.active)
        self.assertEqual(len(result.candidates), 4)
        self.assertTrue(all(math.isfinite(item.predicted_count) and item.predicted_count > 0 for item in result.candidates))


if __name__ == "__main__":
    unittest.main()
