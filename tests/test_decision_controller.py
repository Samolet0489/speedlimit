"""C6 cases from the design and a simulated acknowledgment check."""

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

from speedlimit.decision_controller import DecisionController


@dataclass(frozen=True)
class Branch:
    status: str
    limit_kmh: int | None
    reason: str


class DecisionControllerTests(unittest.TestCase):
    now = datetime(2026, 11, 7, 10, tzinfo=timezone.utc)
    good = {"illuminance_lux": "1", "water_um": "0", "temperature_c": "5", "mode": "replay", "scenario": "good"}
    dark = {**good, "illuminance_lux": "0.1"}

    def controller(self, weather_cap=60, air_cap=80, log_path=None):
        weather = Mock()
        weather.evaluate.return_value = Branch("recommended", weather_cap, "Weather evidence")
        air = Mock()
        air.evaluate.return_value = Branch("recommended", air_cap, "Air evidence")
        return DecisionController(weather, lambda _: air, clock=lambda: self.now, log_path=log_path), weather, air

    def test_DC_01_normal_conditions_skip_the_weather_model(self):
        controller, weather, air = self.controller()
        result = controller.evaluate(self.good)
        self.assertEqual((result["status"], result["limit_kmh"]), ("recommended", 80))
        self.assertEqual(result["mode"], "replay")
        self.assertFalse(result["weather"]["triggered"])
        weather.evaluate.assert_not_called()
        air.evaluate.assert_called_once_with(self.now)

    def test_DC_02_weather_cap_wins_with_good_air(self):
        controller, weather, _ = self.controller()
        result = controller.evaluate(self.dark)
        self.assertEqual(result["limit_kmh"], 60)
        self.assertTrue(weather.evaluate.call_args.args[0].dark)
        self.assertEqual(result["weather"]["reason"], "Weather evidence")

    def test_DC_03_lower_air_cap_wins_when_both_branches_apply(self):
        controller, _, _ = self.controller(air_cap=50)
        result = controller.evaluate(self.dark)
        self.assertEqual(result["limit_kmh"], 50)
        self.assertEqual(result["weather"]["limit_kmh"], 60)
        self.assertEqual(result["air"]["limit_kmh"], 50)
        self.assertEqual(result["reasons"], ["Weather evidence", "Air evidence"])

    def test_DC_04_air_branch_applies_independently(self):
        controller, weather, _ = self.controller(air_cap=60)
        result = controller.evaluate(self.good)
        self.assertEqual(result["limit_kmh"], 60)
        weather.evaluate.assert_not_called()

    def test_DC_05_unavailable_branch_or_invalid_input_requires_review(self):
        for failure in ("weather", "air", "invalid cap", "input"):
            with self.subTest(failure=failure):
                controller, weather, air = self.controller()
                request = self.dark
                if failure == "weather":
                    weather.evaluate.return_value = Branch("review_needed", None, "Missing model evidence")
                elif failure == "air":
                    air.evaluate.return_value = Branch("review_needed", None, "Stale air data")
                elif failure == "invalid cap":
                    weather.evaluate.return_value = Branch("recommended", 80, "Invalid reduced cap")
                else:
                    request = {**request, "temperature_c": ""}
                result = controller.evaluate(request)
                self.assertEqual((result["status"], result["limit_kmh"]), ("review_needed", None))
                if failure == "input":
                    weather.evaluate.assert_not_called()
                    air.evaluate.assert_not_called()
                else:
                    air.evaluate.assert_called_once()

    def test_acknowledgment_correlates_with_the_logged_decision(self):
        with TemporaryDirectory() as folder:
            history = Path(folder) / "decisions.jsonl"
            controller, _, _ = self.controller(log_path=history)
            result = controller.evaluate(self.good)
            controller.acknowledge(result)
            with self.assertRaises(ValueError):
                controller.acknowledge({**result, "status": "review_needed", "limit_kmh": None})
            decision, acknowledgment = [json.loads(line) for line in history.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(decision["event"], "decision")
            self.assertEqual(acknowledgment["event"], "acknowledgment")
            self.assertEqual(decision["decision_id"], acknowledgment["decision_id"])
            self.assertEqual(acknowledgment["limit_kmh"], 80)
