"""C2 trains a small count model on the prepared Year N experiment data.

The model is a single tanh hidden layer with a softplus output and a positive
rate floor. All fitting uses the training partition. Only the validation
partition chooses the epoch. Test and N+1 outcomes are read for evaluation.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import math
import platform
import random
import re
import statistics
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path


DEFAULT_SEED = 2026
FEATURE_NAMES = (
    "illuminance_millilux", "water_micrometers", "temperature_celsius",
    "historical_speed_kmh",
)
FEATURE_UNITS = dict(zip(FEATURE_NAMES, ("millilux", "micrometers", "Celsius", "km/h")))
SPLITS = {"train": [1, 20], "validation": [21, 25], "test": [26, 31]}
MODEL_FORMAT = "speedlimit_tanh_softplus_count_v1"
RATE_FLOOR = 1e-9
_YEARS = ("N", "N+1")
_LABEL = "near_accident_count"
_TIMESTAMP = re.compile(r"01-(0[1-9]|[12][0-9]|3[01])T(?:[01][0-9]|2[0-3]):00:00")
_PREPARED_FIELDS = {
    "year", "target_hour", "prediction_cutoff", "eligible_for_training",
    "event_coverage", _LABEL, *FEATURE_NAMES,
}
_PREDICTION_FIELDS = (
    "year", "split", "target_hour", "prediction_cutoff", _LABEL,
    "baseline_prediction", "neural_network_prediction", "weather_triggered", "weather_branch",
)


@dataclass(frozen=True)
class TrainingSettings:
    hidden_units: int = 6
    epochs: int = 220
    learning_rate: float = 0.03
    l2: float = 0.001


@dataclass(frozen=True)
class _Sample:
    year: str
    target_hour: str
    prediction_cutoff: str
    features: tuple[float, ...]
    label: int
    weather_branch: str


@dataclass
class _Network:
    hidden_weights: list[list[float]]
    hidden_bias: list[float]
    output_weights: list[float]
    output_bias: float


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _invalid_json_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant: {value}")


def _json_object(data: bytes, name: str) -> dict:
    try:
        value = json.loads(data.decode("utf-8-sig"), parse_constant=_invalid_json_constant)
    except ValueError as exc:
        raise ValueError(f"Invalid JSON in {name}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return value


def _decimal(value: object, name: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise ValueError(f"{name} must be numeric")
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not number.is_finite():
        raise ValueError(f"{name} must be finite")
    return number


def _float(value: object, name: str) -> float:
    number = float(_decimal(value, name))
    if not math.isfinite(number):
        raise ValueError(f"{name} exceeds the floating-point range of the model")
    return number


def _features(values: tuple[object, ...]) -> tuple[tuple[float, ...], str]:
    light, water, temperature, speed = (
        _decimal(value, name) for value, name in zip(values, FEATURE_NAMES)
    )
    if light < 0 or water < 0 or speed <= 0:
        raise ValueError("Illuminance and water must be nonnegative and historical speed must be positive")
    # Input is already in the router's canonical units. Compare exact decimals
    # before conversion to model floats, including fractional values near zero.
    dark = light < 500
    ice = water > 1000 and temperature < 0
    branch = "dark_and_black_ice" if dark and ice else "dark" if dark else "black_ice" if ice else "normal"
    canonical = (light, water, temperature, speed)
    features = tuple(float(value) for value in canonical)
    if not all(math.isfinite(value) for value in features) or features[3] <= 0:
        raise ValueError("A feature exceeds the floating-point range of the model")
    return features, branch


def _check_preparation(summary: dict) -> None:
    if type(summary.get("schema_version")) is not int or summary["schema_version"] != 1 or summary.get("component") != "C1 DataPreparation":
        raise ValueError("Unsupported preparation summary contract")
    if summary.get("training_ready") is not True:
        blockers = summary.get("blockers")
        details = "; ".join(str(blocker) for blocker in blockers) if isinstance(blockers, list) else ""
        raise ValueError(f"Preparation is not training-ready{': ' + details if details else ''}")
    if summary.get("blockers") != [] or not isinstance(summary.get("config"), dict):
        raise ValueError("The preparation summary lacks a ready configuration")
    if not isinstance(summary.get("sources"), dict):
        raise ValueError("The preparation summary lacks source provenance")
    digest = summary.get("feature_preview_sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("The preparation summary lacks a valid feature_preview_sha256. Run preparation again.")


def _read_samples(csv_data: bytes, summary: dict) -> tuple[list[_Sample], dict]:
    try:
        data = csv_data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("Prepared feature CSV must be UTF-8") from exc
    reader = csv.DictReader(io.StringIO(data, newline=""), strict=True)
    seen: set[tuple[str, str]] = set()
    total: Counter[str] = Counter()
    eligible: Counter[str] = Counter()
    samples: list[_Sample] = []
    try:
        header = reader.fieldnames or []
        if not _PREPARED_FIELDS.issubset(header) or len(header) != len(set(header)):
            raise ValueError("Prepared feature CSV has missing or duplicate columns")
        for row in reader:
            try:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError("incorrect number of columns")
                year, target, cutoff = (row[name] for name in ("year", "target_hour", "prediction_cutoff"))
                if year not in _YEARS or not _TIMESTAMP.fullmatch(target):
                    raise ValueError("expected a symbolic year N or N+1 and a valid January hour")
                if cutoff and cutoff != target:
                    raise ValueError("prediction cutoff does not match the prepared target-hour-start contract")
                key = (year, target)
                if key in seen:
                    raise ValueError(f"duplicate target hour {year} {target}")
                seen.add(key)
                total[year] += 1
                flag = row["eligible_for_training"].strip().lower()
                if flag not in {"true", "false"}:
                    raise ValueError("eligible_for_training must be true or false")
                if flag == "false":
                    continue
                if not cutoff or row["event_coverage"] != "assumed_complete":
                    raise ValueError("eligible row lacks a prediction cutoff or configured event coverage")
                features, branch = _features(tuple(row[name] for name in FEATURE_NAMES))
                label = _decimal(row[_LABEL], _LABEL)
                if label < 0 or label != label.to_integral_value() or not math.isfinite(float(label)):
                    raise ValueError("near_accident_count must be a nonnegative, representable integer")
                eligible[year] += 1
                samples.append(_Sample(year, target, cutoff, features, int(label), branch))
            except ValueError as exc:
                raise ValueError(f"Prepared feature CSV line {reader.line_num}: {exc}") from exc
    except csv.Error as exc:
        raise ValueError(f"Malformed prepared feature CSV near line {reader.line_num}: {exc}") from exc

    for field, observed in (("feature_rows_by_year", total), ("eligible_feature_rows_by_year", eligible)):
        counts = summary.get(field)
        if not isinstance(counts, dict) or any(
            type(counts.get(year, 0)) is not int or counts.get(year, 0) != observed[year]
            for year in _YEARS
        ):
            raise ValueError(f"Prepared feature CSV disagrees with {field} in preparation_summary.json")
    counts = {year: {"prepared": total[year], "eligible": eligible[year], "excluded": total[year] - eligible[year]} for year in _YEARS}
    return sorted(samples, key=lambda sample: (sample.year, sample.target_hour)), counts


def _partition(samples: list[_Sample]) -> dict[str, list[_Sample]]:
    partitions: dict[str, list[_Sample]] = {key: [] for key in (*SPLITS, "n_plus_1")}
    for sample in samples:
        if sample.year == "N+1":
            partitions["n_plus_1"].append(sample)
            continue
        day = int(sample.target_hour[3:5])
        split = "train" if day <= 20 else "validation" if day <= 25 else "test"
        partitions[split].append(sample)
    for split in SPLITS:
        if not partitions[split]:
            first, last = SPLITS[split]
            raise ValueError(f"No eligible Year N rows in {split} partition (January {first}-{last})")
    return partitions


def _scaling(samples: list[_Sample]) -> tuple[list[float], list[float]]:
    columns = list(zip(*(sample.features for sample in samples)))
    try:
        return [statistics.fmean(column) for column in columns], [statistics.pstdev(column) or 1.0 for column in columns]
    except OverflowError as exc:
        raise ValueError("Training features exceed the numeric range for fitting the scaler") from exc


def _scale(features: tuple[float, ...], means: list[float], scales: list[float]) -> tuple[float, ...]:
    scaled = tuple((value - mean) / scale for value, mean, scale in zip(features, means, scales))
    if not all(math.isfinite(value) for value in scaled):
        raise ValueError("Standardized features exceed the numeric range of the model")
    return scaled


def _forward(model: _Network, values: tuple[float, ...]) -> tuple[list[float], float, float]:
    hidden = [
        math.tanh(math.fsum(w * x for w, x in zip(weights, values)) + bias)
        for weights, bias in zip(model.hidden_weights, model.hidden_bias)
    ]
    score = math.fsum(w * x for w, x in zip(model.output_weights, hidden)) + model.output_bias
    if not math.isfinite(score):
        raise ValueError("Model score is non-finite")
    rate = RATE_FLOOR + max(score, 0.0) + math.log1p(math.exp(-abs(score)))
    return hidden, score, rate


def _fit_epoch(model: _Network, data: list[tuple[tuple[float, ...], int]], settings: TrainingSettings) -> None:
    width = settings.hidden_units
    hidden_gradient = [[0.0] * len(FEATURE_NAMES) for _ in range(width)]
    hidden_bias_gradient = [0.0] * width
    output_gradient = [0.0] * width
    output_bias_gradient = 0.0
    for features, label in data:
        hidden, score, rate = _forward(model, features)
        sigmoid = 1.0 / (1.0 + math.exp(-score)) if score >= 0 else math.exp(score) / (1.0 + math.exp(score))
        output_delta = (1.0 - label / rate) * sigmoid
        output_bias_gradient += output_delta
        for unit, activation in enumerate(hidden):
            output_gradient[unit] += output_delta * activation
            hidden_delta = output_delta * model.output_weights[unit] * (1.0 - activation * activation)
            hidden_bias_gradient[unit] += hidden_delta
            for feature, value in enumerate(features):
                hidden_gradient[unit][feature] += hidden_delta * value

    size = len(data)
    model.output_bias -= settings.learning_rate * output_bias_gradient / size
    for unit in range(width):
        output_weight = model.output_weights[unit]
        model.output_weights[unit] -= settings.learning_rate * (output_gradient[unit] / size + settings.l2 * output_weight)
        model.hidden_bias[unit] -= settings.learning_rate * hidden_bias_gradient[unit] / size
        for feature, weight in enumerate(model.hidden_weights[unit]):
            model.hidden_weights[unit][feature] -= settings.learning_rate * (hidden_gradient[unit][feature] / size + settings.l2 * weight)


def _fit_network(
    training: list[tuple[tuple[float, ...], int]], validation: list[tuple[tuple[float, ...], int]],
    baseline: float, seed: int, settings: TrainingSettings,
) -> tuple[_Network, int, float]:
    rng = random.Random(seed)
    width = settings.hidden_units
    initial_rate = max(baseline, 0.05)
    initial_bias = initial_rate if initial_rate > 20 else math.log(math.expm1(initial_rate))
    network = _Network(
        [[rng.uniform(-0.45, 0.45) for _ in FEATURE_NAMES] for _ in range(width)],
        [0.0] * width,
        [rng.uniform(-0.1, 0.1) for _ in range(width)],
        initial_bias,
    )
    best_network = network
    best_epoch = 0
    best_loss = math.inf
    for epoch in range(1, settings.epochs + 1):
        _fit_epoch(network, training, settings)
        values = []
        for features, label in validation:
            rate = _forward(network, features)[2]
            values.append(rate - label * math.log(rate))
        loss = statistics.fmean(values)
        if not math.isfinite(loss):
            raise ValueError("Validation Poisson loss is non-finite")
        if loss < best_loss:
            best_network, best_epoch, best_loss = copy.deepcopy(network), epoch, loss
    return best_network, best_epoch, best_loss


def _metrics(labels: list[int], predictions: list[float]) -> dict:
    if not labels:
        return {"count": 0, "mae": None, "mean_signed_error": None, "rmse": None,
                "count_calibration": {"mean_prediction": None, "mean_actual": None}}
    errors = [prediction - label for prediction, label in zip(predictions, labels)]
    return {
        "count": len(labels),
        "mae": statistics.fmean(abs(error) for error in errors),
        "mean_signed_error": statistics.fmean(errors),
        "rmse": math.sqrt(statistics.fmean(error * error for error in errors)),
        "count_calibration": {
            "mean_prediction": statistics.fmean(predictions),
            "mean_actual": statistics.fmean(labels),
        },
    }


def _group_metrics(labels: list[int], baseline: list[float], network: list[float]) -> dict:
    return {"count": len(labels), "baseline": _metrics(labels, baseline), "neural_network": _metrics(labels, network)}


def _evaluate(
    samples: list[_Sample], split: str, model: _Network, means: list[float], scales: list[float], baseline: float,
) -> tuple[dict, list[dict]]:
    predictions = [_forward(model, _scale(sample.features, means, scales))[2] for sample in samples]
    labels = [sample.label for sample in samples]
    baseline_predictions = [baseline] * len(samples)
    metrics = _group_metrics(labels, baseline_predictions, predictions)
    triggered = [index for index, sample in enumerate(samples) if sample.weather_branch != "normal"]
    metrics["weather_triggered"] = _group_metrics(
        [labels[index] for index in triggered], [baseline] * len(triggered), [predictions[index] for index in triggered],
    )
    diagnostics = [{
        "year": sample.year, "split": split, "target_hour": sample.target_hour,
        "prediction_cutoff": sample.prediction_cutoff, _LABEL: sample.label,
        "baseline_prediction": baseline, "neural_network_prediction": prediction,
        "weather_triggered": sample.weather_branch != "normal", "weather_branch": sample.weather_branch,
    } for sample, prediction in zip(samples, predictions)]
    return metrics, diagnostics


def _support(samples: list[_Sample]) -> dict:
    by_speed: dict[float, list[_Sample]] = {}
    for sample in samples:
        by_speed.setdefault(sample.features[-1], []).append(sample)
    return {
        str(int(speed)) if speed.is_integer() else repr(speed): {
            "historical_speed_kmh": speed,
            "count": len(rows),
            "weather_triggered_count": sum(row.weather_branch != "normal" for row in rows),
            "weather_branch_counts": dict(Counter(row.weather_branch for row in rows)),
            "feature_ranges": {
                name: {
                    "min": min(row.features[index] for row in rows),
                    "max": max(row.features[index] for row in rows),
                }
                for index, name in enumerate(FEATURE_NAMES[:-1])
            },
        }
        for speed, rows in sorted(by_speed.items())
    }


def _next_year_context(partitions: dict[str, list[_Sample]], input_counts: dict) -> dict:
    year_n = {sample.target_hour: sample.features for name in SPLITS for sample in partitions[name]}
    next_rows = partitions["n_plus_1"]
    matching = sum(year_n.get(sample.target_hour) == sample.features for sample in next_rows)
    return {
        "status": "evaluated_eligible_rows" if next_rows else "unavailable_no_eligible_rows",
        "prepared_rows": input_counts["N+1"]["prepared"],
        "eligible_rows": len(next_rows),
        "excluded_rows": input_counts["N+1"]["excluded"],
        "matching_N_feature_rows": matching,
        "eligible_rows_with_N_timestamp": sum(sample.target_hour in year_n for sample in next_rows),
        "interpretation": (
            "N+1 outcomes are evaluated separately on eligible rows using the frozen Year N fit. "
            "The match count compares prepared features at the same symbolic timestamp. "
            "Matched rows repeat Year N conditions and do not measure generalization to new conditions."
        ),
    }


def _source_provenance(summary: dict, summary_data: bytes, feature_data: bytes) -> dict:
    sources = {}
    for role, source in summary["sources"].items():
        if not isinstance(source, dict) or not isinstance(source.get("filename"), str):
            raise ValueError(f"Missing filename for preparation source {role}")
        digest = source.get("sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"Missing or invalid SHA-256 for preparation source {role}")
        sources[role] = {"filename": source["filename"], "sha256": digest}
    return {
        "preparation_summary_sha256": _sha256(summary_data),
        "feature_preview_sha256": _sha256(feature_data),
        "preparation_config_sha256": _sha256(_json_bytes(summary["config"])),
        "preparation_config": summary["config"],
        "raw_sources": sources,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
    }


def _check_settings(seed: int, settings: TrainingSettings) -> None:
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if not isinstance(settings, TrainingSettings):
        raise ValueError("settings must be a TrainingSettings value")
    if type(settings.hidden_units) is not int or settings.hidden_units < 1:
        raise ValueError("hidden_units must be a positive integer")
    if type(settings.epochs) is not int or settings.epochs < 1:
        raise ValueError("epochs must be a positive integer")
    for name in ("learning_rate", "l2"):
        value = getattr(settings, name)
        if type(value) not in {int, float} or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number")
    if settings.learning_rate <= 0 or settings.l2 < 0:
        raise ValueError("learning_rate must be positive and l2 must be nonnegative")


def _prediction_csv(rows: list[dict]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=_PREDICTION_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _write_artifacts(output: Path, files: dict[str, bytes]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name in files:
        target = output / name
        if target.is_symlink() or (target.exists() and not target.is_file()):
            raise ValueError(f"Training artifact target is not a regular file: {target}")
    # Stage the complete run before replacing individual artifacts. The metrics
    # file is written last and contains the hashes for verifying the bundle.
    with tempfile.TemporaryDirectory(prefix=".c2-stage-", dir=output) as directory:
        stage = Path(directory)
        for name, data in files.items():
            (stage / name).write_bytes(data)
        for name in files:
            (stage / name).replace(output / name)


def train_models(
    prepared: Path, output: Path, *, seed: int = DEFAULT_SEED, settings: TrainingSettings = TrainingSettings(),
) -> dict:
    """Train a constant baseline and validation-selected neural network.

    ``prepared`` is the C1 output directory. Returns the JSON-safe metrics
    summary also written to ``output/metrics.json``. The JSON model and the
    diagnostic predictions CSV share this directory. N+1 rows are never fitted.
    """
    _check_settings(seed, settings)
    prepared, output = Path(prepared).resolve(), Path(output).resolve()
    if output == prepared or prepared in output.parents:
        raise ValueError("Training output must be outside the preparation output directory")
    summary_data = (prepared / "preparation_summary.json").read_bytes()
    summary = _json_object(summary_data, "preparation_summary.json")
    _check_preparation(summary)
    feature_data = (prepared / "feature_preview.csv").read_bytes()
    if _sha256(feature_data) != summary["feature_preview_sha256"]:
        raise ValueError("feature_preview.csv does not match feature_preview_sha256 in preparation_summary.json. Run preparation again.")
    samples, input_counts = _read_samples(feature_data, summary)
    partitions = _partition(samples)
    provenance = _source_provenance(summary, summary_data, feature_data)
    training = partitions["train"]
    means, scales = _scaling(training)
    try:
        baseline = statistics.fmean(sample.label for sample in training)
        train_data = [(_scale(sample.features, means, scales), sample.label) for sample in training]
        validation_data = [(_scale(sample.features, means, scales), sample.label) for sample in partitions["validation"]]
        network, selected_epoch, validation_loss = _fit_network(train_data, validation_data, baseline, seed, settings)
        metrics, predictions = {}, []
        for split, rows in partitions.items():
            metrics[split], split_predictions = _evaluate(rows, split, network, means, scales, baseline)
            predictions.extend(split_predictions)
    except OverflowError as exc:
        raise ValueError("Training or evaluation exceeded the model's numeric range") from exc

    split_sizes = {split: len(rows) for split, rows in partitions.items()}
    fit_settings = asdict(settings)
    fit_details = {
        "seed": seed, "settings": fit_settings, "selected_epoch": selected_epoch,
        "validation_poisson_loss": validation_loss,
        "loss": "Mean Poisson rate minus count times log(rate), without the label-only factorial term.",
        "regularization": "L2 on hidden and output weights during fitting. Validation loss excludes the penalty.",
        "selection": "Lowest unpenalized validation loss after a training epoch; ties keep the earliest epoch.",
        "optimizer": "Deterministic full-batch gradient descent",
        "epoch_count": settings.epochs,
        "initialization": "Seeded uniform hidden weights [-0.45,0.45], output weights [-0.1,0.1], zero hidden biases, inverse-softplus bias at the train mean (minimum 0.05).",
    }
    artifact = {
        "schema_version": 1, "model_format": MODEL_FORMAT, "feature_names": list(FEATURE_NAMES),
        "feature_units": FEATURE_UNITS, "target": _LABEL, "rate_floor": RATE_FLOOR,
        "scaling": {"mean": means, "scale": scales, "fit_partition": "N/train",
                    "method": "Train population standard deviation; constant features use scale 1."},
        "weights": asdict(network), "baseline_mean": baseline, "fit": fit_details,
        "split_sizes": split_sizes, "training_support_by_speed": _support(training),
        "support_meaning": "Per-speed training counts and marginal feature ranges describe observations, not joint or causal support.",
        "provenance": provenance,
    }
    # Validate the exported contract before publishing any result.
    _model_parameters(artifact)
    model_data, predictions_data = _json_bytes(artifact), _prediction_csv(predictions)
    report = {
        "schema_version": 1, "component": "C2 ModelTrainer", "status": "trained",
        "split_sizes": split_sizes, "split_days_in_January": SPLITS, "input_rows_by_year": input_counts,
        "metrics": metrics, "metric_conventions": {
            "mean_signed_error": "prediction minus observed count; positive values indicate overprediction",
            "count_calibration": "Mean predicted and observed counts on the same rows",
            "weather_triggered": "Illuminance <500 millilux OR (water >1000 micrometers AND temperature <0 Celsius)",
        },
        "seed": seed, "settings": fit_settings, "fit": fit_details, "selected_epoch": selected_epoch,
        "baseline_mean": baseline, "provenance": provenance, "next_year": _next_year_context(partitions, input_counts),
        "outputs": {"model": "model.json", "metrics": "metrics.json", "predictions": "predictions.csv"},
        "artifact_sha256": {"model.json": _sha256(model_data), "predictions.csv": _sha256(predictions_data)},
    }
    report_data = _json_bytes(report)
    _write_artifacts(output, {"model.json": model_data, "predictions.csv": predictions_data, "metrics.json": report_data})
    return report


def _vector(value: object, size: int, name: str) -> list[float]:
    if not isinstance(value, list) or len(value) != size or any(type(item) not in {int, float} for item in value):
        raise ValueError(f"Invalid model {name}: expected {size} finite numbers")
    return [_float(item, f"model {name}") for item in value]


def _model_parameters(model: dict) -> tuple[list[float], list[float], _Network]:
    if (
        not isinstance(model, dict) or type(model.get("schema_version")) is not int or model["schema_version"] != 1
        or model.get("model_format") != MODEL_FORMAT or model.get("feature_names") != list(FEATURE_NAMES)
        or model.get("feature_units") != FEATURE_UNITS or model.get("target") != _LABEL
        or model.get("rate_floor") != RATE_FLOOR
    ):
        raise ValueError("Unsupported model feature/target contract")
    scaling, weights = model.get("scaling"), model.get("weights")
    if not isinstance(scaling, dict) or not isinstance(weights, dict):
        raise ValueError("Model lacks scaling or weights")
    means = _vector(scaling.get("mean"), len(FEATURE_NAMES), "scaling mean")
    scales = _vector(scaling.get("scale"), len(FEATURE_NAMES), "scaling scale")
    if any(scale <= 0 for scale in scales):
        raise ValueError("Model scaling factors must be positive")
    hidden = weights.get("hidden_weights")
    if not isinstance(hidden, list) or not hidden:
        raise ValueError("Model must contain at least one hidden unit")
    output_bias = weights.get("output_bias")
    if type(output_bias) not in {int, float}:
        raise ValueError("Invalid model output bias")
    network = _Network(
        [_vector(row, len(FEATURE_NAMES), "hidden weights") for row in hidden],
        _vector(weights.get("hidden_bias"), len(hidden), "hidden bias"),
        _vector(weights.get("output_weights"), len(hidden), "output weights"),
        _float(output_bias, "model output bias"),
    )
    return means, scales, network


def load_model(path: Path) -> dict:
    """Read and validate a portable JSON model. No executable deserialization."""
    model = _json_object(Path(path).read_bytes(), "model.json")
    _model_parameters(model)
    return model


def predict_near_accidents(
    model: dict, *, illuminance_millilux: float, water_micrometers: float,
    temperature_celsius: float, historical_speed_kmh: float,
) -> float:
    """Predict the hourly count for canonical measurements and a candidate speed.

    The fitted feature is named after its historical training value. At
    inference, supply the candidate limit in that argument. Support checks and
    policy decisions belong to C4 and are outside this prediction primitive.
    """
    means, scales, network = _model_parameters(model)
    features, _ = _features((illuminance_millilux, water_micrometers, temperature_celsius, historical_speed_kmh))
    return _forward(network, _scale(features, means, scales))[2]
