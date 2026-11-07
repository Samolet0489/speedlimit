"""DP-01..DP-05 regression evidence for the C1 preparation boundary."""

import csv
import hashlib
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from speedlimit.data_preparation import prepare_data


READINGS = "Month;Day;Hour;Minute;Second;Sensor;Value\n"
EVENTS = "Month;Day;Hour;Second;LicencePlate;Damage;Injured;CloseCarCm;CloseGuardrailCm;SkidAngle\n"
CONFIG = {
    "sensor_type_overrides": {"3": "temperature"},
    "event_hour_convention": "one_based_interval",
    "near_accident_thresholds": {"skid_angle_degrees": 10, "close_car_cm": 10, "close_guardrail_cm": 10},
    "assume_event_log_complete": True,
    "prediction_cutoff": "target_hour_start",
    "max_feature_age_hours": 1,
}


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


class DataPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix=".c1-test-", dir=Path(__file__).resolve().parent)
        self.root = Path(self.temp.name)
        self.resources = self.root / "Resources"
        self.resources.mkdir()
        self.output = self.root / "prepared"
        self.reading_rows = [
            "1;1;0;0;0;-1;80,00", "1;1;0;10;0;1;1001,00",
            "1;1;0;10;0;2;0,499", "1;1;0;10;0;3;-1,00",
            "1;1;1;0;0;2;0,499", "1;1;1;0;1;2;999,00",
            "1;1;1;10;0;1;900,00", "1;1;1;10;0;3;2,00",
            "1;1;1;10;0;-1;70,00", "1;1;2;10;0;1;800,00",
        ]
        self.event_rows = ["1;1;2;17;PRIVATE-A;;;1;2;30"]
        self.write_sources()

    def tearDown(self):
        self.temp.cleanup()

    def write_sources(self, extra_readings=(), event_rows=None, types=None):
        self.resources.joinpath("Sensors.csv").write_text("SensorID;SensorTypeCode\n1;W\n2;L\n3;T\n", encoding="utf-8")
        self.resources.joinpath("SensorTypes.csv").write_text(types or "SensorTypeCode;Type;Unit;Min;Max\nW ;water;micrometer;0;10000\nL;light;lux;0;1000000\nT;temperature;degrees Celsius;-30;70\nT;traffic;cars/second;0;50\n", encoding="utf-8")
        body = "\n".join(self.reading_rows + list(extra_readings)) + "\n"
        self.resources.joinpath("SensorReadingsYearN.csv").write_text(READINGS + body, encoding="utf-8-sig")
        self.resources.joinpath("SensorReadingsYearNPlus1.csv").write_text(body, encoding="utf-8-sig")
        event_body = "\n".join(self.event_rows if event_rows is None else event_rows) + "\n"
        for name in ("AccidentsYearN.csv", "AccidentsYearNPlus1.csv"):
            self.resources.joinpath(name).write_text(EVENTS + event_body, encoding="utf-8-sig")

    def feature(self, target="01-01T01:00:00", year="N", output=None):
        return next(r for r in read_csv((output or self.output) / "feature_preview.csv") if r["year"] == year and r["target_hour"] == target)

    def test_DP01_actual_sources_preserved_headerless_privacy_and_repeatability(self):
        resources = Path(__file__).resolve().parents[3] / "Resources"
        if not resources.is_dir():
            self.skipTest("The six supplied course CSVs are not available")
        hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in resources.glob("*.csv")}
        summary = prepare_data(resources, self.output)
        self.assertEqual(summary["observation_counts_by_year"], {"N": 12648, "N+1": 12648})
        self.assertFalse(summary["sources"]["readings_NPlus1"]["header_present"])
        self.assertEqual(summary["events"]["records"], 1841)
        self.assertFalse(summary["training_ready"])
        self.assertTrue(summary["blockers"])
        feature_hash = hashlib.sha256((self.output / "feature_preview.csv").read_bytes()).hexdigest()
        saved_summary = json.loads((self.output / "preparation_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["feature_preview_sha256"], feature_hash)
        self.assertEqual(saved_summary["feature_preview_sha256"], feature_hash)
        observations = read_csv(self.output / "observations.csv")
        first = next(r for r in observations if r["year"] == "N+1" and r["source_row"] == "1")
        self.assertEqual(first["sensor_id"], "6")
        self.assertEqual(first["timestamp"], "01-01T00:03:35")
        self.assertEqual(len(observations), 25296)
        self.assertEqual(sum(r["sensor_id"] == "19" and r["status"] == "quarantined" for r in observations), 1488)
        self.assertTrue(all(not r["temperature_celsius"] for r in read_csv(self.output / "feature_preview.csv")))
        self.assertTrue(all(not r["near_accident_count"] for r in read_csv(self.output / "feature_preview.csv")))
        before = {p.name: p.read_bytes() for p in self.output.iterdir()}
        prepare_data(resources, self.output)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.output.iterdir()})
        self.assertEqual(hashes, {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in resources.glob("*.csv")})
        all_output = "\n".join(p.read_text(encoding="utf-8") for p in self.output.iterdir())
        for path in resources.glob("*Accidents*.csv"):
            with path.open(encoding="utf-8-sig", newline="") as stream:
                for row in csv.DictReader(stream, delimiter=";"):
                    if row["LicencePlate"]:
                        self.assertNotIn(json.dumps(row["LicencePlate"]), all_output)

    def test_DP02_duplicates_conflicts_and_shuffled_semantic_features(self):
        self.write_sources(extra_readings=[
            "1;1;0;10;0;2;0,499",  # Exact duplicate: preserve, then flag copy.
            "1;1;0;30;0;1;10,00", "1;1;0;30;0;1;20,00",  # Latest water conflicts.
        ])
        summary = prepare_data(self.resources, self.output, CONFIG)
        self.assertEqual(summary["observations"]["exact_duplicate_excess_rows"], 2)
        self.assertEqual(summary["observations"]["conflicting_timestamp_groups"], 2)
        feature = self.feature()
        self.assertEqual(feature["water_micrometers"], "")
        self.assertIn("invalid_latest_road_water", feature["issues"])
        self.assertEqual(float(feature["illuminance_millilux"]), 499)
        self.assertEqual(feature["eligible_for_training"], "False")
        expected = read_csv(self.output / "feature_preview.csv")
        for filename in ("SensorReadingsYearN.csv", "SensorReadingsYearNPlus1.csv"):
            path = self.resources / filename
            lines = path.read_text(encoding="utf-8-sig").splitlines()
            header = lines.pop(0) if filename == "SensorReadingsYearN.csv" else None
            random.Random(7).shuffle(lines)
            path.write_text((header + "\n" if header else "") + "\n".join(lines) + "\n", encoding="utf-8-sig")
        other = self.root / "shuffled"
        prepare_data(self.resources, other, CONFIG)
        actual = read_csv(other / "feature_preview.csv")
        for rows in (expected, actual):
            for row in rows:
                row.pop("feature_source_rows")  # Physical row references necessarily move.
        self.assertEqual(expected, actual)

    def test_DP03_explicit_units_cutoff_boundary_no_future_or_backfill(self):
        summary = prepare_data(self.resources, self.output, CONFIG)
        feature = self.feature()
        self.assertEqual(float(feature["illuminance_millilux"]), 499)
        self.assertEqual(float(feature["water_micrometers"]), 1001)
        self.assertEqual(float(feature["temperature_celsius"]), -1)
        self.assertEqual(float(feature["historical_speed_kmh"]), 80)
        self.assertEqual(feature["near_accident_count"], "1")
        self.assertEqual(feature["eligible_for_training"], "True")
        first = self.feature("01-01T00:00:00")
        self.assertEqual(first["illuminance_millilux"], "")
        self.assertEqual(first["water_micrometers"], "")
        self.assertEqual(first["temperature_celsius"], "")
        source_refs = json.loads(feature["feature_source_rows"])
        selected = source_refs["light"][0].rsplit(":", 1)[1]
        self.assertEqual(selected, "6")  # Header + fifth data row: exactly 01:00:00.
        self.assertTrue(summary["training_ready"])
        self.assertIn("exposure assumption", " ".join(summary["limitations"]))

    def test_DP04_hour24_preserved_until_named_convention_and_invalid_date_blocks(self):
        self.write_sources(event_rows=["1;1;24;59;PRIVATE-HOUR;;;1;2;30"])
        prepare_data(self.resources, self.output)
        event = read_csv(self.output / "events.csv")[0]
        self.assertEqual(event["hour_raw"], "24")
        self.assertEqual(event["second_raw"], "59")
        self.assertEqual(event["target_hour"], "")
        prepare_data(self.resources, self.output, CONFIG)
        self.assertEqual(read_csv(self.output / "events.csv")[0]["target_hour"], "01-01T23:00:00")
        alternate = {**CONFIG, "event_hour_convention": "clock_hour_24_next_day"}
        prepare_data(self.resources, self.output, alternate)
        self.assertEqual(read_csv(self.output / "events.csv")[0]["target_hour"], "01-02T00:00:00")
        self.write_sources(event_rows=["1;32;2;0;PRIVATE-BAD-DATE;;;1;2;30"])
        summary = prepare_data(self.resources, self.output, CONFIG)
        self.assertFalse(summary["training_ready"])
        self.assertEqual(self.feature()["near_accident_count"], "")
        self.assertTrue(any("cannot be assigned" in b for b in summary["blockers"]))

    def test_DP05_single_event_count_privacy_missing_values_and_guardrails(self):
        self.write_sources(
            extra_readings=["1;1;0;20;0;19;-5,00", "1;1;1;0;0;3;NULL"],
            event_rows=[
                "1;1;2;17;PRIVATE-A;;;1;2;-30",
                "1;1;2;17;PRIVATE-B;;;1;2;-30",  # Separate vehicle, identical public fields.
                "1;1;3;2;PRIVATE-CRASH;100;1;;;",
                "1;1;3;3;PRIVATE-NEGATIVE;;;-1;20;0",
            ],
        )
        summary = prepare_data(self.resources, self.output, CONFIG)
        events = read_csv(self.output / "events.csv")
        self.assertEqual([r["qualifying_near_event"] for r in events[:2]], ["True", "True"])
        self.assertEqual(self.feature()["near_accident_count"], "2")
        self.assertEqual(self.feature()["temperature_celsius"], "")
        self.assertEqual(events[2]["event_kind"], "crash")
        self.assertIn("negative_event_field", events[3]["issues"])
        self.assertEqual(summary["events"]["possible_duplicate_source_rows"], 0)
        self.assertEqual(summary["observations"]["quarantine_reason_counts"]["orphan_sensor"], 2)
        for path in self.output.iterdir():
            self.assertNotIn("PRIVATE-", path.read_text(encoding="utf-8"))
        with self.assertRaises(ValueError):
            prepare_data(self.resources, self.resources / "derived")
        source = self.resources / "SensorReadingsYearN.csv"
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        linked = self.output / "observations.csv"
        linked.unlink()
        try:
            try:
                linked.hardlink_to(source)
            except OSError:
                # Some Windows sandboxes prohibit creation of hard links.
                # Exercise the samefile guard without needing that privilege.
                linked.write_text("derived placeholder", encoding="utf-8")
                with patch.object(Path, "samefile", return_value=True), self.assertRaises(ValueError):
                    prepare_data(self.resources, self.output, CONFIG)
            else:
                with self.assertRaises(ValueError):
                    prepare_data(self.resources, self.output, CONFIG)
            self.assertEqual(source_hash, hashlib.sha256(source.read_bytes()).hexdigest())
        finally:
            linked.unlink()
        for bad in ([], {"unknown_key": True}, {"assume_event_log_complete": "false"}):
            with self.subTest(config=type(bad).__name__), self.assertRaises(ValueError):
                prepare_data(self.resources, self.output, bad)
        # A duplicate complete private-source event remains unresolved, not deduplicated.
        self.write_sources(event_rows=[self.event_rows[0], self.event_rows[0]])
        summary = prepare_data(self.resources, self.output, CONFIG)
        self.assertEqual(summary["events"]["possible_duplicate_source_rows"], 4)
        self.assertEqual(self.feature()["near_accident_count"], "")
        # Canonical conversion overflow is quarantined and never emitted as inf.
        self.write_sources(extra_readings=["1;1;1;0;0;2;1e308"], types="SensorTypeCode;Type;Unit;Min;Max\nW;water;micrometer;0;10000\nL;light;lux;0;1e308\nT;temperature;degrees Celsius;-30;70\nT;traffic;cars/second;0;50\n")
        summary = prepare_data(self.resources, self.output, CONFIG)
        self.assertEqual(summary["observations"]["quarantine_reason_counts"]["nonfinite_normalized_value"], 2)
        self.assertEqual(self.feature()["illuminance_millilux"], "")
        self.write_sources(extra_readings=["1;1;1;0;0;2;0.49999999999999999"])
        summary = prepare_data(self.resources, self.output, CONFIG)
        self.assertEqual(summary["observations"]["quarantine_reason_counts"]["missing_or_invalid_value"], 2)
        self.assertEqual(self.feature()["illuminance_millilux"], "")
        self.write_sources()
        path = self.resources / "SensorReadingsYearN.csv"
        lines = path.read_text(encoding="utf-8-sig").splitlines()
        for i in range(1, len(lines)):
            fields = lines[i].split(";")
            if fields[5] == "1":
                fields[6] = "-1"
                lines[i] = ";".join(fields)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
        summary = prepare_data(self.resources, self.output, CONFIG)
        self.assertEqual(summary["eligible_feature_rows_by_year"]["N"], 0)
        self.assertGreater(summary["eligible_feature_rows_by_year"]["N+1"], 0)
        self.assertFalse(summary["training_ready"])

    def test_DP06_recorded_indicator_subtypes_and_explicit_sensor_exclusion(self):
        self.write_sources(
            extra_readings=["1;1;0;20;0;4;8,00"],
            event_rows=[
                "1;1;1;17;PRIVATE-NEAR;;;99;80;-1",
                "1;1;2;2;PRIVATE-CRASH;100;0;;;",
                "1;1;3;2;PRIVATE-EMPTY;;;;;",
                "1;1;3;3;PRIVATE-MIXED;100;;2;;",
                "1;1;3;4;PRIVATE-INVALID;;;invalid;;",
            ],
        )
        with (self.resources / "Sensors.csv").open("a", encoding="utf-8") as stream:
            stream.write("4;T\n")
        recorded = {**CONFIG, "event_label_policy": "recorded_indicator", "near_accident_thresholds": None, "excluded_sensor_ids": [4]}
        summary = prepare_data(self.resources, self.output, recorded)
        self.assertTrue(summary["training_ready"])
        self.assertEqual(summary["observations"]["excluded_observation_counts_by_sensor"], {"4": 2})
        self.assertFalse(summary["observations"]["unresolved_ambiguous_sensor_ids"])
        events = read_csv(self.output / "events.csv")[:5]
        self.assertEqual([r["qualifying_near_event"] for r in events], ["True", "False", "", "", ""])
        self.assertEqual([r["event_kind"] for r in events], ["qualifying_near_event", "crash", "unknown", "mixed_event_subtype", "unknown"])
        self.assertEqual(self.feature("01-01T00:00:00")["near_accident_count"], "1")
        self.assertEqual(self.feature()["near_accident_count"], "0")
        self.assertEqual(self.feature("01-01T02:00:00")["near_accident_count"], "")
        prepare_data(self.resources, self.output, {**recorded, "assume_event_log_complete": False})
        self.assertEqual(self.feature()["near_accident_count"], "")
        prepare_data(self.resources, self.output, {**recorded, "event_hour_convention": "clock_hour_24_next_day"})
        first = self.feature("01-01T00:00:00")
        self.assertEqual((first["near_accident_count"], first["event_coverage"]), ("", "unknown"))


if __name__ == "__main__":
    unittest.main()
