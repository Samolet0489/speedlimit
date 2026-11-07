"""Verify the public command's measurement and preparation status contracts."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from speedlimit.cli import main


class CommandLineTests(unittest.TestCase):
    def invoke(self, args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = main(args)
        return status, stdout.getvalue(), stderr.getvalue()

    def test_route_emits_normalized_measurements_and_branch_decision(self):
        status, out, error = self.invoke([
            "route", "--illuminance-lux", "0.499", "--water-um", "1000", "--temperature-c", "0"])
        self.assertEqual(status, 0)
        self.assertEqual(error, "")
        result = json.loads(out)
        self.assertTrue(result["reduce"])
        self.assertTrue(result["dark"])
        self.assertFalse(result["black_ice"])
        self.assertEqual(result["illuminance_millilux"], "499")
        self.assertNotIn("final_limit", result)

    def test_route_error_is_machine_readable_and_does_not_emit_a_decision(self):
        status, out, error = self.invoke([
            "route", "--illuminance-lux", "NaN", "--water-um", "0", "--temperature-c", "0"])
        self.assertEqual(status, 1)
        self.assertEqual(out, "")
        self.assertEqual(json.loads(error)["status"], "error")

    def test_unrepresentable_lux_conversion_returns_json_error(self):
        status, out, error = self.invoke([
            "route", "--illuminance-lux", "1e999999999999999999", "--water-um", "0", "--temperature-c", "0"])
        self.assertEqual(status, 1)
        self.assertEqual(out, "")
        self.assertEqual(json.loads(error)["status"], "error")

    def test_prepare_strict_status_does_not_call_a_partial_result_training_ready(self):
        with TemporaryDirectory() as temporary:
            base = Path(temporary)
            config = base / "config.json"
            config.write_text("{}", encoding="utf-8")
            with patch("speedlimit.data_preparation.prepare_data", return_value={
                "training_ready": False, "blockers": ["event_hour_convention is unresolved"]}) as prepare:
                status, out, error = self.invoke([
                    "prepare", "--resources", str(base / "resources"), "--output", str(base / "output"),
                    "--config", str(config), "--require-training-ready"])
            self.assertEqual(status, 2)
            self.assertEqual(error, "")
            self.assertFalse(json.loads(out)["training_ready"])
            prepare.assert_called_once()

    def test_prepare_rejects_output_inside_the_resource_directory(self):
        with TemporaryDirectory() as temporary:
            base = Path(temporary)
            status, out, error = self.invoke([
                "prepare", "--resources", str(base), "--output", str(base / "derived")])
            self.assertEqual(status, 1)
            self.assertEqual(out, "")
            self.assertIn("outside", json.loads(error)["message"])

    def test_prepare_rejects_a_non_object_configuration(self):
        with TemporaryDirectory() as temporary:
            base = Path(temporary)
            config = base / "config.json"
            config.write_text("[]", encoding="utf-8")
            status, out, error = self.invoke([
                "prepare", "--resources", str(base / "resources"), "--output", str(base / "output"),
                "--config", str(config)])
            self.assertEqual(status, 1)
            self.assertEqual(out, "")
            self.assertIn("JSON object", json.loads(error)["message"])


if __name__ == "__main__":
    unittest.main()
