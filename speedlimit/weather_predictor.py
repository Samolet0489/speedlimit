"""Weather-only recommendations guarded by local training-data overlap.

The default policy needs three Year N training observations for the candidate
speed and exact router branch. Their Euclidean distance must be at most one
after dividing each weather dimension by its fitted training scale. Each query
must also lie within that group's observed per-dimension range. This policy
limits extrapolation in the simulator. It does not validate counterfactual
predictions or establish that a lower limit is safe on a real road.
"""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Literal

from .model_trainer import (
    FEATURE_NAMES,
    SPLITS,
    _check_preparation,
    _partition,
    _read_samples,
    load_model,
    predict_near_accidents,
)
from .weather_router import WeatherDecision, _conditions


CANDIDATE_SPEEDS = (70, 60, 50, 40)
DEFAULT_MIN_NEIGHBORS = 3
DEFAULT_RADIUS = 1.0
_SUPPORT_FORMAT = "speedlimit_weather_support_v1"
_WEATHER_NAMES = FEATURE_NAMES[:-1]
_BRANCHES = {"normal", "dark", "black_ice", "dark_and_black_ice"}
_INPUT_ERRORS = (OSError, ValueError, TypeError, ArithmeticError, RuntimeError)


@dataclass(frozen=True, slots=True)
class Candidate:
    speed_kmh: int
    status: Literal["unsupported", "invalid_prediction", "over_threshold", "qualifies"]
    predicted_count: float | None
    neighbors: int
    reason: str


@dataclass(frozen=True, slots=True)
class WeatherRecommendation:
    status: Literal["recommended", "review_needed"]
    limit_kmh: int | None
    reason: str
    model_id: str | None
    candidates: tuple[Candidate, ...]


@dataclass(frozen=True, slots=True)
class _Bundle:
    model: dict
    model_id: str
    groups: dict[tuple[float, str], list[tuple[float, ...]]]
    scales: tuple[float, ...]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _records_sha256(records: list) -> str:
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return _sha256(canonical.encode("utf-8"))


def _load_bound_model(path: Path) -> tuple[dict, str]:
    # Verify that the exact bytes being hashed decode to the validated model,
    # even if an artifact was replaced between these two reads.
    data = path.read_bytes()
    model = load_model(path)
    if json.loads(data) != model:
        raise ValueError("Model changed while it was being loaded")
    return model, _sha256(data)


def _branch(light: Decimal, water: Decimal, temperature: Decimal) -> str:
    dark, ice = _conditions(light, water, temperature)
    if dark and ice:
        return "dark_and_black_ice"
    if dark:
        return "dark"
    return "black_ice" if ice else "normal"


def _group_support(support: dict, model: dict, model_id: str) -> dict[tuple[float, str], list[tuple[float, ...]]]:
    if (
        not isinstance(support, dict) or type(support.get("schema_version")) is not int
        or support["schema_version"] != 1 or support.get("support_format") != _SUPPORT_FORMAT
        or support.get("feature_names") != list(FEATURE_NAMES) or support.get("fit_partition") != "N/train"
        or support.get("january_days") != SPLITS["train"]
    ):
        raise ValueError("Unsupported weather support contract")
    if support.get("model_sha256") != model_id:
        raise ValueError("Weather support does not match the model SHA-256")
    provenance = model.get("provenance")
    if not isinstance(provenance, dict) or any(
        not isinstance(provenance.get(key), str) or len(provenance[key]) != 64
        or support.get(key) != provenance[key]
        for key in ("feature_preview_sha256", "preparation_summary_sha256")
    ):
        raise ValueError("Weather support does not match the model's prepared-data provenance")

    records = support.get("records")
    split_sizes = model.get("split_sizes")
    train_count = split_sizes.get("train") if isinstance(split_sizes, dict) else None
    if (
        not isinstance(records, list) or type(train_count) is not int or not 1 <= train_count <= 480
        or type(support.get("training_rows")) is not int
        or train_count != support["training_rows"] or train_count != len(records)
    ):
        raise ValueError("Weather support row count disagrees with the fitted training split")
    if support.get("records_sha256") != _records_sha256(records):
        raise ValueError("Weather support records SHA-256 does not match")

    groups: dict[tuple[float, str], list[tuple[float, ...]]] = defaultdict(list)
    by_speed: dict[float, list[tuple[str, tuple[float, ...]]]] = defaultdict(list)
    for record in records:
        if not isinstance(record, dict) or set(record) != {"weather_branch", "features"}:
            raise ValueError("Malformed weather support record")
        branch, values = record["weather_branch"], record["features"]
        if (
            not isinstance(branch, str) or branch not in _BRANCHES or not isinstance(values, list)
            or len(values) != len(FEATURE_NAMES)
            or any(type(value) not in {int, float} or not math.isfinite(value) for value in values)
        ):
            raise ValueError("Malformed weather support features or branch")
        light, water, temperature, speed = values
        if light < 0 or water < 0 or speed <= 0:
            raise ValueError("Invalid physical values in weather support features")
        if _branch(*(Decimal(str(value)) for value in (light, water, temperature))) != branch:
            raise ValueError("Weather support branch disagrees with its recorded measurements")
        weather = tuple(float(value) for value in values[:-1])
        groups[float(speed), branch].append(weather)
        by_speed[float(speed)].append((branch, weather))

    expected = model.get("training_support_by_speed")
    actual = {
        str(int(speed)) if speed.is_integer() else repr(speed): {
            "historical_speed_kmh": speed,
            "count": len(rows),
            "weather_triggered_count": sum(branch != "normal" for branch, _ in rows),
            "weather_branch_counts": dict(Counter(branch for branch, _ in rows)),
            "feature_ranges": {
                name: {"min": min(row[index] for _, row in rows), "max": max(row[index] for _, row in rows)}
                for index, name in enumerate(_WEATHER_NAMES)
            },
        }
        for speed, rows in by_speed.items()
    }
    if actual != expected:
        raise ValueError("Weather support speed, branch, count or ranges disagree with the fitted model")
    return dict(groups)


def build_support(model_path: Path, prepared_dir: Path, output_path: Path) -> dict:
    """Export model-bound feature records from the validated C2 training cohort.

    Reuse C2's prepared-row validation and partitioning so support cannot use a
    different eligibility rule. The support output has no labels, times or raw
    identifiers. Returns the same JSON-safe object that is written to disk.
    """
    model_path = Path(model_path).resolve()
    prepared_dir = Path(prepared_dir).resolve()
    output_path = Path(output_path)
    summary_path = prepared_dir / "preparation_summary.json"
    preview_path = prepared_dir / "feature_preview.csv"
    protected_paths = {model_path, summary_path.resolve(), preview_path.resolve()}
    if output_path.is_symlink() or output_path.resolve() in protected_paths:
        raise ValueError("Weather support output cannot alias an input artifact")

    model, model_id = _load_bound_model(model_path)
    summary_data, preview_data = summary_path.read_bytes(), preview_path.read_bytes()
    summary = json.loads(summary_data)
    if not isinstance(summary, dict):
        raise ValueError("Preparation summary must contain a JSON object")
    _check_preparation(summary)
    preview_hash, summary_hash = _sha256(preview_data), _sha256(summary_data)
    provenance = model.get("provenance")
    if (
        not isinstance(provenance, dict) or preview_hash != summary["feature_preview_sha256"]
        or preview_hash != provenance.get("feature_preview_sha256")
        or summary_hash != provenance.get("preparation_summary_sha256")
    ):
        raise ValueError("Prepared preview or summary SHA-256 does not match the fitted model")
    samples, _ = _read_samples(preview_data, summary)
    training = _partition(samples)["train"]
    records = [{"features": list(row.features), "weather_branch": row.weather_branch} for row in training]
    support = {
        "schema_version": 1,
        "support_format": _SUPPORT_FORMAT,
        "model_sha256": model_id,
        "feature_preview_sha256": preview_hash,
        "preparation_summary_sha256": summary_hash,
        "fit_partition": "N/train",
        "january_days": list(SPLITS["train"]),
        "feature_names": list(FEATURE_NAMES),
        "training_rows": len(training),
        "records_sha256": _records_sha256(records),
        "records": records,
    }
    _group_support(support, model, model_id)
    data = json.dumps(support, sort_keys=True, indent=2, allow_nan=False) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".c4-stage-", dir=output_path.parent) as temporary:
        stage = Path(temporary) / output_path.name
        stage.write_text(data, encoding="utf-8", newline="\n")
        stage.replace(output_path)
    return support


def _measurements(decision: WeatherDecision) -> tuple[tuple[Decimal, ...], tuple[float, ...], str]:
    if not isinstance(decision, WeatherDecision):
        raise ValueError("Weather predictor needs a router decision")
    measurements = (decision.illuminance_millilux, decision.water_um, decision.temperature_c)
    if any(not isinstance(value, Decimal) or not value.is_finite() for value in measurements):
        raise ValueError("Weather predictor needs finite Decimal measurements from the router")
    light, water, temperature = measurements
    if light < 0 or water < 0:
        raise ValueError("Illuminance and road water must be nonnegative")
    values = tuple(float(value) for value in measurements)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("A weather measurement exceeds the model's numeric range")
    branch = _branch(light, water, temperature)
    dark = branch in {"dark", "dark_and_black_ice"}
    ice = branch in {"black_ice", "dark_and_black_ice"}
    if decision.dark is not dark or decision.black_ice is not ice or decision.reduce is not (dark or ice):
        raise ValueError("Weather router flags disagree with the supplied measurements")
    return measurements, values, branch


class WeatherPredictor:
    """Load a validated model/support pair on the first active weather request.

    Each instance retains that artifact snapshot. Construct another predictor
    when replacing the model or support file. An unsuccessful load is retried
    on the next active request.
    """

    def __init__(
        self, model_path: Path, support_path: Path, *, min_neighbors: int = DEFAULT_MIN_NEIGHBORS,
        radius: float = DEFAULT_RADIUS,
    ) -> None:
        if type(min_neighbors) is not int or min_neighbors < 1:
            raise ValueError("min_neighbors must be a positive integer")
        if type(radius) not in {int, float} or not math.isfinite(radius) or radius <= 0:
            raise ValueError("radius must be finite and positive")
        self.model_path, self.support_path = Path(model_path), Path(support_path)
        self.min_neighbors, self.radius = min_neighbors, float(radius)
        self._bundle: _Bundle | None = None

    def _load(self) -> _Bundle:
        if self._bundle is None:
            model, model_id = _load_bound_model(self.model_path)
            support = json.loads(self.support_path.read_bytes())
            groups = _group_support(support, model, model_id)
            scales = tuple(float(value) for value in model["scaling"]["scale"][:-1])
            self._bundle = _Bundle(model, model_id, groups, scales)
        return self._bundle

    def _candidate(
        self, bundle: _Bundle, speed: int, branch: str, measurements: tuple[Decimal, ...],
        values: tuple[float, ...],
    ) -> Candidate:
        points = bundle.groups.get((speed, branch), [])
        neighbors = sum(
            math.hypot(*((query - observed) / scale for query, observed, scale in zip(values, point, bundle.scales)))
            <= self.radius
            for point in points
        )
        if points and any(not min(column) <= query <= max(column) for query, column in zip(values, zip(*points))):
            return Candidate(speed, "unsupported", None, neighbors,
                             "A weather feature is outside the observed range for this speed and branch.")
        if neighbors < self.min_neighbors:
            return Candidate(speed, "unsupported", None, neighbors,
                             f"Found {neighbors} of {self.min_neighbors} required same-speed, same-branch neighbors "
                             f"within normalized radius {self.radius:g}.")
        light, water, temperature = measurements
        try:
            rate = predict_near_accidents(
                bundle.model, illuminance_millilux=light, water_micrometers=water,
                temperature_celsius=temperature, historical_speed_kmh=speed,
            )
            if type(rate) not in {int, float} or not math.isfinite(rate) or rate < 0:
                raise ValueError("Expected a finite nonnegative predicted count")
        except _INPUT_ERRORS as exc:
            return Candidate(speed, "invalid_prediction", None, neighbors, f"Prediction failed. {exc}")
        if rate <= 1:
            return Candidate(speed, "qualifies", float(rate), neighbors, "Predicted count is at most one per hour.")
        return Candidate(speed, "over_threshold", float(rate), neighbors, "Predicted count exceeds one per hour.")

    def evaluate(self, decision: WeatherDecision) -> WeatherRecommendation:
        """Recommend the highest supported reduced speed with count at most one."""
        try:
            measurements, values, branch = _measurements(decision)
            if branch == "normal":
                return WeatherRecommendation("review_needed", None, "The weather router did not activate this branch.", None, ())
            bundle = self._load()
        except _INPUT_ERRORS as exc:
            return WeatherRecommendation("review_needed", None, f"Weather recommendation unavailable. {exc}", None, ())

        candidates = tuple(self._candidate(bundle, speed, branch, measurements, values) for speed in CANDIDATE_SPEEDS)
        if any(item.status == "invalid_prediction" for item in candidates):
            return WeatherRecommendation("review_needed", None, "A supported candidate produced an invalid prediction.",
                                         bundle.model_id, candidates)
        winner = next((item for item in candidates if item.status == "qualifies"), None)
        if winner is None:
            return WeatherRecommendation("review_needed", None,
                                         "No reduced speed had sufficient support and a predicted count of at most one per hour.",
                                         bundle.model_id, candidates)
        return WeatherRecommendation("recommended", winner.speed_kmh,
                                     f"{winner.speed_kmh} km/h is the highest supported candidate with a predicted count of at most one per hour.",
                                     bundle.model_id, candidates)
