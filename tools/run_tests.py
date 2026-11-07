"""Run all component tests and save their real outcomes as JSON evidence."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import unittest

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
IGNORED = {"artifacts", "__pycache__", ".venv", ".git"}


class EvidenceResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = []

    def _record(self, test, outcome, detail=None):
        item = {"test": test.id(), "description": test.shortDescription(), "outcome": outcome}
        if detail:
            item["detail"] = detail
        self.records.append(item)

    def addSuccess(self, test):
        super().addSuccess(test)
        self._record(test, "passed")

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self._record(test, "failed", self._exc_info_to_string(err, test))

    def addError(self, test, err):
        super().addError(test, err)
        self._record(test, "error", self._exc_info_to_string(err, test))

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self._record(test, "skipped", reason)

    def addExpectedFailure(self, test, err):
        super().addExpectedFailure(test, err)
        self._record(test, "expected_failure", self._exc_info_to_string(err, test))

    def addUnexpectedSuccess(self, test):
        super().addUnexpectedSuccess(test)
        self._record(test, "unexpected_success")

    def addSubTest(self, test, subtest, err):
        super().addSubTest(test, subtest, err)
        if err is not None:
            self._record(subtest, "failed_subtest", self._exc_info_to_string(err, test))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=CODE / "artifacts" / "test_results.json")
    args = parser.parse_args(argv)
    suite = unittest.defaultTestLoader.discover(str(CODE / "tests"), top_level_dir=str(CODE))
    start = time.perf_counter()
    result = unittest.TextTestRunner(verbosity=2, resultclass=EvidenceResult).run(suite)
    hashes = {}
    for folder, subdirectories, filenames in os.walk(CODE):
        subdirectories[:] = sorted(set(subdirectories) - IGNORED)
        for filename in sorted(filenames):
            path = Path(folder) / filename
            if path.suffix in {".py", ".json"}:
                hashes[path.relative_to(CODE).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    evidence = {
        "component_scope": ["C1 DataPreparation", "C2 ModelTrainer", "C3 WeatherRouter", "C4 WeatherPredictor",
                            "C5 AirQualityAdvisor", "C6 DecisionController", "C7 OperatorGUI", "command-line integration"],
        "python": platform.python_version(),
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "tests_run": result.testsRun,
        "passed": sum(item["outcome"] == "passed" for item in result.records),
        "failures": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped),
        "expected_failures": len(result.expectedFailures),
        "unexpected_successes": len(result.unexpectedSuccesses),
        "successful": result.wasSuccessful(), "elapsed_seconds": round(time.perf_counter()-start, 4),
        "results": result.records, "source_sha256": hashes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(f"Test evidence saved to {args.output.resolve()}")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
