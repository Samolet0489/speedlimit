"""C1: deterministic, identifier-free preparation of the supplied course CSVs.

Unresolved source facts remain visible. This module never trains a model or
repairs source files. Event distance tests use ``< threshold`` and skid tests
use ``abs(angle) > threshold``. Configure those choices explicitly before
interpreting labels. Missing event indicators are unknown unless another
indicator conclusively qualifies the event; ordinary crashes stay separate.
The opt-in recorded_indicator policy interprets populated event fields using
the case study's stated subtype conventions.
"""

from __future__ import annotations

import bisect
import csv
import hashlib
import io
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path


DEFAULTS = {
    "sensor_type_overrides": {},
    "excluded_sensor_ids": [],
    "event_hour_convention": None,
    "event_label_policy": "threshold",
    "near_accident_thresholds": None,
    "assume_event_log_complete": False,
    "prediction_cutoff": None,
    "max_feature_age_hours": 1,
}
READING_COLUMNS = ["Month", "Day", "Hour", "Minute", "Second", "Sensor", "Value"]
EVENT_COLUMNS = ["Month", "Day", "Hour", "Second", "LicencePlate", "Damage", "Injured", "CloseCarCm", "CloseGuardrailCm", "SkidAngle"]
SCHEMAS = {
    "sensors": ["SensorID", "SensorTypeCode"],
    "sensor_types": ["SensorTypeCode", "Type", "Unit", "Min", "Max"],
    "readings_N": READING_COLUMNS, "readings_NPlus1": READING_COLUMNS,
    "events_N": EVENT_COLUMNS, "events_NPlus1": EVENT_COLUMNS,
}
MISSING = {"", "null", "none", "nan", "na", "n/a"}
OBSERVATION_FIELDS = ["year", "source_file", "source_row", "timestamp", "month", "day", "hour", "minute", "second", "sensor_id", "source_type_code", "measurement_type", "canonical_type_id", "source_unit", "source_value", "canonical_unit", "value", "status", "issues"]
EVENT_FIELDS = ["year", "source_file", "source_row", "month", "day", "hour_raw", "second_raw", "target_hour", "hour_convention", "damage_eur", "injured", "close_car_cm", "close_guardrail_cm", "skid_angle_degrees", "event_kind", "qualifying_near_event", "status", "issues"]
FEATURE_FIELDS = ["year", "target_hour", "prediction_cutoff", "illuminance_millilux", "water_micrometers", "temperature_celsius", "historical_speed_kmh", "near_accident_count", "observed_qualifying_events", "event_coverage", "light_sensor_count", "water_sensor_count", "temperature_sensor_count", "speed_sensor_count", "feature_source_rows", "eligible_for_training", "issues"]
CANONICAL_TYPES = {
    "light": ("illuminance", "millilux"), "water": ("road_water", "micrometers"),
    "temperature": ("temperature", "celsius"), "speed limit": ("speed_limit", "km/h"),
    "traffic": ("traffic", "cars/second"), "humidity": ("humidity", "percent"),
    "noise": ("noise", "decibel"), "wind direction": ("wind_direction", "degrees"),
    "wind strength": ("wind_strength", "hm/h"), "air pressure": ("air_pressure", "mmHG"),
}


def _number(value):
    value = str(value).strip()
    if value.lower() in MISSING:
        return None
    try:
        decimal = Decimal(value.replace(",", "."))
        parsed = float(decimal)
        # Never turn an explicitly more precise value below a threshold into
        # the threshold itself through float rounding. Ordinary source decimal
        # commas remain usable when their decimal text round-trips unchanged.
        return parsed if decimal.is_finite() and math.isfinite(parsed) and Decimal(str(parsed)) == decimal else None
    except (ValueError, InvalidOperation, OverflowError):
        return None


def _integer(value):
    parsed = _number(value)
    return int(parsed) if parsed is not None and parsed.is_integer() else None


def _stamp(value):
    return value.strftime("%m-%dT%H:%M:%S") if value is not None else None


def _date(row, include_clock=True):
    fields = ["Month", "Day", "Hour"] + (["Minute", "Second"] if include_clock else [])
    values = [_integer(row.get(k, "")) for k in fields]
    if any(v is None for v in values):
        return None
    try:
        return datetime(2000, *values)
    except ValueError:
        return None


def _config(config):
    if config is not None and not isinstance(config, dict):
        raise ValueError("Preparation configuration must be an object")
    if set(config or {}) - set(DEFAULTS):
        raise ValueError("Unknown preparation configuration key")
    result = {key: (config or {}).get(key, value) for key, value in DEFAULTS.items()}
    if not isinstance(result["sensor_type_overrides"], dict):
        raise ValueError("sensor_type_overrides must be a sensor-ID-to-metadata-Type object")
    result["sensor_type_overrides"] = {str(k): v for k, v in result["sensor_type_overrides"].items()}
    excluded = result["excluded_sensor_ids"]
    if not isinstance(excluded, list) or any(type(sensor) is not int or sensor <= 0 for sensor in excluded):
        raise ValueError("excluded_sensor_ids must be a list of positive integer sensor IDs")
    result["excluded_sensor_ids"] = sorted(set(excluded))
    if result["event_hour_convention"] not in (None, "one_based_interval", "clock_hour_24_next_day"):
        raise ValueError("Unsupported event_hour_convention")
    if result["event_label_policy"] not in ("threshold", "recorded_indicator"):
        raise ValueError("event_label_policy must be threshold or recorded_indicator")
    if result["prediction_cutoff"] not in {None, "target_hour_start"}:
        raise ValueError("prediction_cutoff must be null or target_hour_start")
    if type(result["assume_event_log_complete"]) is not bool:
        raise ValueError("assume_event_log_complete must be a Boolean")
    age = result["max_feature_age_hours"]
    if isinstance(age, bool) or not isinstance(age, (int, float)) or not math.isfinite(age) or age <= 0:
        raise ValueError("max_feature_age_hours must be positive and finite")
    thresholds = result["near_accident_thresholds"]
    if thresholds is not None:
        if result["event_label_policy"] == "recorded_indicator":
            raise ValueError("recorded_indicator does not use near_accident_thresholds; set them to null")
        required = {"skid_angle_degrees", "close_car_cm", "close_guardrail_cm"}
        if not isinstance(thresholds, dict) or set(thresholds) != required:
            raise ValueError("near_accident_thresholds must contain the three documented threshold keys")
        for value in thresholds.values():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError("Near-accident thresholds must be nonnegative finite numbers")
    return result


def _sources(resources):
    paths = {}
    for path in sorted(resources.glob("*.csv")):
        name = path.name.lower()
        if "sensortypes" in name:
            role = "sensor_types"
        elif "sensorreadings" in name:
            role = "readings_NPlus1" if "nplus1" in name else "readings_N"
        elif "sensors" in name:
            role = "sensors"
        elif "accidents" in name:
            role = "events_NPlus1" if "nplus1" in name else "events_N"
        else:
            continue
        if role in paths:
            raise ValueError("Multiple CSV versions match the same dataset role")
        paths[role] = path
    if set(paths) != set(SCHEMAS):
        raise ValueError("The resource directory must contain all six SpeedLimit CSV roles")
    return paths


def _read(path, role):
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("A resource CSV is not valid UTF-8") from error
    reader = csv.reader(io.StringIO(text), delimiter=";")
    parsed = [(reader.line_num, row) for row in reader]
    schema = SCHEMAS[role]
    has_header = bool(parsed and [v.strip() for v in parsed[0][1]] == schema)
    if not has_header and role != "readings_NPlus1":
        raise ValueError("A required source CSV header is missing or incompatible")
    records = []
    for line, fields in parsed[1:] if has_header else parsed:
        row = {key: fields[i].strip() if i < len(fields) else "" for i, key in enumerate(schema)}
        row["_row"] = line
        row["_width_ok"] = len(fields) == len(schema)
        records.append(row)
    return records, {
        "filename": path.name, "sha256": hashlib.sha256(raw).hexdigest(),
        "encoding": "utf-8-sig" if raw.startswith(b"\xef\xbb\xbf") else "ASCII-compatible / UTF-8",
        "delimiter": ";", "header_present": has_header, "columns": schema,
        "rows": len(records), "width_error_rows": sum(not r["_width_ok"] for r in records),
        "schema_origin": "source header" if has_header else "verified N reading schema; first data row preserved",
    }


def _normal_value(kind, unit, value):
    if value is None:
        return None, None, "missing_or_invalid_value"
    if kind == "light":
        if unit == "lux":
            converted = value * 1000
            return (converted, "millilux", None) if math.isfinite(converted) else (None, "millilux", "nonfinite_normalized_value")
        if unit == "millilux":
            return value, unit, None
        return None, None, "unsupported_light_unit"
    if kind == "water":
        return (value, "micrometers", None) if unit in {"micrometer", "micrometers"} else (None, None, "unsupported_water_unit")
    if kind == "temperature":
        return (value, "celsius", None) if unit in {"degrees Celsius", "Celsius", "celsius"} else (None, None, "unsupported_temperature_unit")
    return value, CANONICAL_TYPES.get(kind, (None, unit))[1], None


def _observations(tables, source_info, cfg):
    by_id = defaultdict(list)
    by_code = defaultdict(list)
    for row in tables["sensors"]:
        by_id[_integer(row["SensorID"])].append(row)
    for row in tables["sensor_types"]:
        by_code[row["SensorTypeCode"]].append(row)
    known_names = {row["Type"] for row in tables["sensor_types"]}
    excluded = set(cfg["excluded_sensor_ids"])
    if excluded - by_id.keys():
        raise ValueError("An excluded sensor must have a source metadata definition")
    for key, value in cfg["sensor_type_overrides"].items():
        if _integer(key) not in by_id or not isinstance(value, str) or value not in known_names:
            raise ValueError("A sensor override must refer to a defined sensor and an exact metadata Type name")
        if _integer(key) in excluded:
            raise ValueError("An excluded sensor cannot also have a type override")
    observations = []
    ambiguous_ids = set()
    for role in ("readings_N", "readings_NPlus1"):
        year = "N" if role == "readings_N" else "N+1"
        for raw in tables[role]:
            issues = []
            sid = _integer(raw["Sensor"])
            when = _date(raw)
            value = _number(raw["Value"])
            code, kind, unit, canonical, canonical_unit = None, None, None, None, None
            if not raw["_width_ok"]:
                issues.append("wrong_column_count")
            if when is None:
                issues.append("invalid_timestamp")
            if sid is None:
                issues.append("invalid_sensor_id")
            elif sid == -1:
                kind, unit = "speed limit", "km/h"
                if value is not None and not 0 <= value <= 80:
                    issues.append("out_of_range")
            elif sid not in by_id:
                issues.append("orphan_sensor")
            elif len(by_id[sid]) != 1:
                issues.append("duplicate_sensor_definition")
            elif sid in excluded:
                code = by_id[sid][0]["SensorTypeCode"]
                issues.append("sensor_excluded_by_config")
            else:
                code = by_id[sid][0]["SensorTypeCode"]
                candidates = by_code.get(code, [])
                override = cfg["sensor_type_overrides"].get(str(sid))
                if override is not None:
                    candidates = [c for c in candidates if c["Type"] == override]
                    if not candidates:
                        raise ValueError("A sensor override does not match that sensor's metadata code")
                if len(candidates) > 1:
                    ambiguous_ids.add(sid)
                    issues.append("ambiguous_sensor_type")
                elif not candidates:
                    issues.append("unknown_sensor_type")
                else:
                    meta = candidates[0]
                    kind, unit = meta["Type"], meta["Unit"]
                    low, high = _number(meta["Min"]), _number(meta["Max"])
                    if low is None or high is None or low > high:
                        issues.append("invalid_metadata_range")
                    elif value is not None and not low <= value <= high:
                        issues.append("out_of_range")
            if value is None:
                issues.append("missing_or_invalid_value")
            if kind is not None:
                canonical, canonical_unit, problem = _normal_value(kind, unit, value)
                if problem and problem not in issues:
                    issues.append(problem)
            item = {
                "year": year, "source_file": source_info[role]["filename"], "source_row": raw["_row"],
                "timestamp": _stamp(when), "month": _integer(raw["Month"]), "day": _integer(raw["Day"]),
                "hour": _integer(raw["Hour"]), "minute": _integer(raw["Minute"]), "second": _integer(raw["Second"]),
                "sensor_id": sid, "source_type_code": code, "measurement_type": kind,
                "canonical_type_id": CANONICAL_TYPES.get(kind, (None, None))[0],
                "source_unit": unit, "source_value": value, "canonical_unit": canonical_unit, "value": canonical,
                "status": "quarantined" if issues else "accepted", "issues": issues,
                "_time": when,
            }
            observations.append(item)
    keys = defaultdict(list)
    for item in observations:
        if item["_time"] is not None and item["sensor_id"] is not None:
            keys[(item["year"], item["sensor_id"], item["_time"])].append(item)
    duplicate_excess, conflicts = 0, 0
    for group in keys.values():
        if len(group) < 2:
            continue
        if len({(r["source_value"], tuple(r["issues"])) for r in group}) == 1:
            duplicate_excess += len(group) - 1
            for item in sorted(group, key=lambda r: r["source_row"])[1:]:
                item["issues"].append("exact_duplicate")
                item["status"] = "quarantined"
        else:
            conflicts += 1
            for item in group:
                item["issues"].append("conflicting_duplicate")
                item["status"] = "quarantined"
    return observations, {
        "exact_duplicate_excess_rows": duplicate_excess, "conflicting_timestamp_groups": conflicts,
        "unresolved_ambiguous_sensor_ids": sorted(ambiguous_ids),
        "excluded_observation_counts_by_sensor": dict(sorted(Counter(str(row["sensor_id"]) for row in observations if "sensor_excluded_by_config" in row["issues"]).items())),
        "quarantine_reason_counts": dict(sorted(Counter(reason for row in observations for reason in row["issues"]).items())),
        "duplicate_policy": "Preserve every source observation. Exact duplicates after numeric/key normalization retain the first row as a candidate; later copies are flagged. All conflicting readings at a sensor/timestamp are unavailable.",
    }


def _event_hour(raw, convention):
    if convention is None:
        return None
    month, day, hour = (_integer(raw[k]) for k in ("Month", "Day", "Hour"))
    if None in (month, day, hour) or not 1 <= hour <= 24:
        return None
    try:
        midnight = datetime(2000, month, day)
    except ValueError:
        return None
    return midnight + timedelta(hours=hour - 1 if convention == "one_based_interval" else hour)


def _events(tables, source_info, cfg):
    output = []
    possible_duplicates = 0
    recorded = cfg["event_label_policy"] == "recorded_indicator"
    for role in ("events_N", "events_NPlus1"):
        year = "N" if role == "events_N" else "N+1"
        # Include the private source identifier in this in-memory comparison so
        # separate vehicles with matching public fields are never deduplicated.
        counts = Counter(tuple(r[k] for k in EVENT_COLUMNS) for r in tables[role])
        for raw in tables[role]:
            issues = []
            when = _event_hour(raw, cfg["event_hour_convention"])
            if cfg["event_hour_convention"] is None:
                issues.append("unresolved_event_hour_convention")
            elif when is None:
                issues.append("invalid_event_date_or_hour")
            if not raw["_width_ok"]:
                issues.append("wrong_column_count")
            second = _integer(raw["Second"])
            if second is None or not 0 <= second <= 59:
                issues.append("invalid_event_second")
            values = {key: _number(raw[key]) for key in EVENT_COLUMNS[5:]}
            for key, value in values.items():
                if raw[key].strip().lower() not in MISSING and value is None:
                    issues.append("invalid_event_numeric_field")
                elif key != "SkidAngle" and value is not None and value < 0:
                    issues.append("negative_event_field")
            if values["Injured"] is not None and not values["Injured"].is_integer():
                issues.append("noninteger_injury_count")
            if counts[tuple(raw[k] for k in EVENT_COLUMNS)] > 1:
                possible_duplicates += 1
                issues.append("possible_duplicate_event_unresolved")
            near_values = [values[k] for k in ("CloseCarCm", "CloseGuardrailCm", "SkidAngle")]
            crash = any(values[k] is not None and (recorded or values[k] > 0) for k in ("Damage", "Injured"))
            recorded_near = any(value is not None for value in near_values)
            near = None
            thresholds = cfg["near_accident_thresholds"]
            invalid = any(i in issues for i in ("invalid_event_numeric_field", "negative_event_field", "noninteger_injury_count", "possible_duplicate_event_unresolved", "wrong_column_count"))
            mixed = recorded and crash and recorded_near
            if mixed:
                issues.append("mixed_event_subtype")
            if recorded:
                invalid_record = invalid or "invalid_event_second" in issues or "invalid_event_date_or_hour" in issues
                if not invalid_record and not mixed and (crash or recorded_near):
                    near = recorded_near
            elif thresholds is None:
                issues.append("unresolved_near_accident_thresholds")
            elif not invalid:
                checks = [
                    v is not None and v < thresholds[key]
                    for v, key in zip(near_values[:2], ("close_car_cm", "close_guardrail_cm"))
                ] + [near_values[2] is not None and abs(near_values[2]) > thresholds["skid_angle_degrees"]]
                if any(checks):
                    near = True
                elif all(v is not None for v in near_values):
                    near = False
                else:
                    issues.append("missing_near_indicator_semantics")
            if all(value is None for value in values.values()):
                issues.append("all_event_fields_missing")
            if mixed:
                kind = "mixed_event_subtype"
            elif recorded and near is None:
                kind = "unknown"
            else:
                kind = "crash_and_qualifying_near_event" if crash and near else "crash" if crash else "qualifying_near_event" if near else "unknown" if near is None else "nonqualifying_event"
            output.append({
                "year": year, "source_file": source_info[role]["filename"], "source_row": raw["_row"],
                "month": _integer(raw["Month"]), "day": _integer(raw["Day"]),
                "hour_raw": _integer(raw["Hour"]), "second_raw": second,
                "target_hour": _stamp(when), "hour_convention": cfg["event_hour_convention"],
                "damage_eur": values["Damage"], "injured": values["Injured"],
                "close_car_cm": values["CloseCarCm"], "close_guardrail_cm": values["CloseGuardrailCm"],
                "skid_angle_degrees": values["SkidAngle"], "event_kind": kind,
                "qualifying_near_event": near, "status": "needs_review" if issues else "classified", "issues": sorted(set(issues)), "_time": when,
            })
    return output, {
        "records": len(output), "possible_duplicate_source_rows": possible_duplicates,
        "unassignable_event_counts_by_year": dict(Counter(e["year"] for e in output if e["_time"] is None)),
        "event_kind_counts": dict(sorted(Counter(e["event_kind"] for e in output).items())),
        "review_reason_counts": dict(sorted(Counter(reason for e in output for reason in e["issues"]).items())),
        "duplicate_policy": "Event source rows stay distinct; no event is deduplicated using privacy-stripped fields. Exact full-source repeats are flagged for review and cannot produce a verified label.",
        "label_policy": cfg["event_label_policy"],
        "label_comparisons": (
            {"recorded_indicator": "Each valid near-only source row with a populated proximity or skid field counts once. A valid crash-only row counts as no near event. Invalid, empty or mixed rows are unresolved."}
            if recorded else
            {"skid": "abs(angle) > skid_angle_degrees", "car": "distance < close_car_cm", "guardrail": "distance < close_guardrail_cm", "combination": "OR; each qualifying source event counted once"}
        ),
        "missing_indicator_policy": (
            "Case study slide 7 distinguishes crash and near-event fields. Blank sibling indicators are allowed when valid populated fields identify a single subtype. Empty, invalid or mixed rows make the hour unknown."
            if recorded else
            "A qualifying observed indicator proves a near event. Missing indicators never prove absence. Nonqualifying records with incomplete indicators keep the hourly near label unknown; crashes are reported separately."
        ),
        "timestamp_policy": "Second is preserved as a raw source component. No Minute is invented and Second is not treated as seconds since the start of an hour.",
    }


def _features(observations, events, cfg):
    streams = defaultdict(lambda: defaultdict(list))
    targets = defaultdict(set)
    for row in observations:
        if row["_time"] is None:
            continue
        targets[row["year"]].add(row["_time"].replace(minute=0, second=0))
        if row["measurement_type"] in {"light", "water", "temperature", "speed limit"}:
            streams[row["year"]][row["sensor_id"]].append(row)
    indexed = {}
    for year, sensors in streams.items():
        for sid, rows in sensors.items():
            rows.sort(key=lambda r: (r["_time"], r["source_row"]))
            indexed[(year, sid)] = ([r["_time"] for r in rows], rows)
    event_hours = defaultdict(list)
    unmapped_event_years = set()
    for event in events:
        if event["_time"] is not None:
            event_hours[(event["year"], event["_time"])].append(event)
        else:
            unmapped_event_years.add(event["year"])
    result = []
    label_definition_ready = cfg["event_label_policy"] == "recorded_indicator" or cfg["near_accident_thresholds"] is not None
    for year in ("N", "N+1"):
        for index_of_hour, target in enumerate(sorted(targets[year])):
            issues, selected, blocked_kinds = [], defaultdict(list), set()
            cutoff = target if cfg["prediction_cutoff"] == "target_hour_start" else None
            if cutoff is None:
                issues.append("unresolved_prediction_cutoff")
            else:
                for sid in sorted(streams[year]):
                    times, rows = indexed[(year, sid)]
                    index = bisect.bisect_right(times, cutoff) - 1
                    kind = rows[0]["measurement_type"]
                    if index < 0:
                        issues.append("no_asof_" + CANONICAL_TYPES[kind][0])
                        continue
                    latest_time = times[index]
                    latest = [r for r in rows[max(0, bisect.bisect_left(times, latest_time)):index + 1] if r["_time"] == latest_time]
                    accepted = [r for r in latest if r["status"] == "accepted"]
                    if any("conflicting_duplicate" in r["issues"] for r in latest) or not accepted:
                        issues.append("invalid_latest_" + CANONICAL_TYPES[kind][0])
                        blocked_kinds.add(kind)
                        continue
                    row = accepted[0]
                    if (cutoff - latest_time).total_seconds() > cfg["max_feature_age_hours"] * 3600:
                        issues.append("stale_" + CANONICAL_TYPES[kind][0])
                        continue
                    selected[kind].append(row)
            feature_values = {}
            for kind, field in (("light", "illuminance_millilux"), ("water", "water_micrometers"), ("temperature", "temperature_celsius"), ("speed limit", "historical_speed_kmh")):
                readings = selected[kind]
                feature_values[field] = sum(r["value"] / len(readings) for r in readings) if readings and kind not in blocked_kinds else None
                if feature_values[field] is None:
                    issues.append("missing_critical_" + CANONICAL_TYPES[kind][0])
            group = event_hours.get((year, target), [])
            observed_count = sum(e["qualifying_near_event"] is True for e in group)
            unknown_group = any(e["qualifying_near_event"] is None or any(flag in e["issues"] for flag in ("invalid_event_second", "invalid_event_date_or_hour", "wrong_column_count", "possible_duplicate_event_unresolved")) for e in group)
            # With a clock-hour convention, midnight on the opening day needs
            # the prior day's hour-24 log, outside the supplied January files.
            clock_boundary = cfg["event_hour_convention"] == "clock_hour_24_next_day" and index_of_hour == 0 and target.hour == 0 and not group
            has_observed_label = cfg["event_hour_convention"] is not None and label_definition_ready and year not in unmapped_event_years and not clock_boundary
            verified = has_observed_label and cfg["assume_event_log_complete"] and not unknown_group
            label = observed_count if verified else None
            if not verified:
                issues.append("unknown_event_label_or_coverage")
            source_refs = {kind: [f"{r['source_file']}:{r['source_row']}" for r in readings] for kind, readings in sorted(selected.items()) if readings}
            result.append({
                "year": year, "target_hour": _stamp(target), "prediction_cutoff": _stamp(cutoff), **feature_values,
                "near_accident_count": label, "observed_qualifying_events": observed_count if has_observed_label else None,
                "event_coverage": "assumed_complete" if cfg["assume_event_log_complete"] and not clock_boundary else "unknown",
                "light_sensor_count": len(selected["light"]), "water_sensor_count": len(selected["water"]),
                "temperature_sensor_count": len(selected["temperature"]), "speed_sensor_count": len(selected["speed limit"]),
                "feature_source_rows": json.dumps(source_refs, sort_keys=True, separators=(",", ":")),
                "eligible_for_training": label is not None and all(v is not None for v in feature_values.values()),
                "issues": sorted(set(issues)),
            })
    return result


def _write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            public = {key: (";".join(row[key]) if key == "issues" else row[key]) for key in fields}
            writer.writerow(public)


def prepare_data(resources: Path, output: Path, config: dict | None = None) -> dict:
    """Prepare CSV previews and return their deterministic quality summary.

    ``output`` must be outside the raw resource directory. Returned readiness
    describes configured experiment inputs, never a factual confirmation of
    metadata, event coverage, safety or suitability for deployment.
    """
    resources, output = Path(resources).resolve(), Path(output).resolve()
    if output == resources or resources in output.parents:
        raise ValueError("The output directory must be outside the source Resources directory")
    cfg = _config(config)
    paths = _sources(resources)
    if any(output == path for path in paths.values()):
        raise ValueError("Output cannot overwrite a source CSV")
    for filename in ("observations.csv", "events.csv", "feature_preview.csv", "preparation_summary.json"):
        target = output / filename
        resolved = target.resolve()
        if resolved == resources or resources in resolved.parents or any(resolved == source.resolve() or (target.exists() and target.samefile(source)) for source in paths.values()):
            raise ValueError("An output target aliases a source resource; preparation will not write it")
    tables, source_info = {}, {}
    for role in SCHEMAS:
        tables[role], source_info[role] = _read(paths[role], role)
    observations, observation_summary = _observations(tables, source_info, cfg)
    events, event_summary = _events(tables, source_info, cfg)
    features = _features(observations, events, cfg)
    blockers = []
    if cfg["event_hour_convention"] is None:
        blockers.append("Q-01: event-hour convention is unresolved")
    if cfg["prediction_cutoff"] is None:
        blockers.append("Q-01: prediction cutoff is unresolved")
    if cfg["event_hour_convention"] is not None and event_summary["unassignable_event_counts_by_year"]:
        blockers.append("Q-01: some event rows cannot be assigned to a target hour; affected-year labels stay unknown")
    if cfg["event_label_policy"] == "threshold" and cfg["near_accident_thresholds"] is None:
        blockers.append("Q-02: near-accident thresholds are unresolved")
    if not cfg["assume_event_log_complete"]:
        blockers.append("Q-02: event-log completeness is not assumed; absence cannot become zero")
    if observation_summary["unresolved_ambiguous_sensor_ids"]:
        blockers.append("Q-03: observed defined sensors still have ambiguous type metadata")
    eligible_N = sum(row["year"] == "N" and row["eligible_for_training"] for row in features)
    if not eligible_N:
        blockers.append("No Year N training feature/label rows are eligible under the configured policies")
    summary = {
        "schema_version": 1, "component": "C1 DataPreparation", "training_ready": not blockers,
        "blockers": blockers, "config": cfg,
        "readiness_basis": "Configured experiment readiness only. Configuration choices are not evidence that source conventions, identities or coverage are factually confirmed.",
        "sources": source_info, "observation_counts_by_year": dict(Counter(o["year"] for o in observations)),
        "accepted_observation_counts_by_year": dict(Counter(o["year"] for o in observations if o["status"] == "accepted")),
        "observations": observation_summary, "events": event_summary,
        "feature_rows_by_year": dict(Counter(f["year"] for f in features)),
        "eligible_feature_rows_by_year": {year: sum(f["year"] == year and f["eligible_for_training"] for f in features) for year in ("N", "N+1")},
        "feature_issue_counts": dict(sorted(Counter(reason for f in features for reason in f["issues"]).items())),
        "feature_contract": {
            "light": "millilux = source lux * 1000; source/canonical values and units retained",
            "aggregation": "Arithmetic mean of available accepted, fresh, confirmed same-type sensors at the cutoff; sensor counts and source row references accompany every quantity. An invalid or conflicting latest reading blocks that quantity instead of silently falling back. Missing streams are flagged and never converted to zero.",
            "asof": "Latest source timestamp <= target-hour-start cutoff; no future value, backfill, or fallback past an invalid/conflicting latest reading. A valid reading exactly at the cutoff is eligible.",
            "target": "Qualifying near-event count in the named target hour only after hour, label-policy and completeness configuration. Unresolved source events can still make the hour unknown.",
            "time_reference": "Exported month-day timestamps have symbolic dataset year N/N+1; 2000 is an internal calendar-validation reference, not an asserted source year.",
        },
        "limitations": [
            "Orphan sensors remain quarantined even when their numeric values look like another measurement type.",
            "The accident Minute field is absent; event Second is retained without inventing a precise timestamp.",
            "Historical speed at target-hour start is an exposure assumption. Changes within the target hour mean that it is not proof of the speed throughout that interval.",
            "Both nominal years cover January only. Supplied N+1 sensor observations repeat N; changed outcomes do not provide new feature conditions.",
            "Confirmed type names do not establish sensor location/pairing. Means of available same-type sensors are a documented experiment choice.",
            "No imputation, model fitting, scaling fitted on all data, or causal safety claims occur in preparation.",
        ],
        "outputs": {"observations": "observations.csv", "events": "events.csv", "feature_preview": "feature_preview.csv", "summary": "preparation_summary.json"},
    }
    # Check source integrity before writing any derived output.
    if any(hashlib.sha256(path.read_bytes()).hexdigest() != source_info[role]["sha256"] for role, path in paths.items()):
        raise RuntimeError("A source CSV changed during preparation; no derived output was written")
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "observations.csv", OBSERVATION_FIELDS, observations)
    _write_csv(output / "events.csv", EVENT_FIELDS, events)
    feature_path = output / "feature_preview.csv"
    _write_csv(feature_path, FEATURE_FIELDS, features)
    summary["feature_preview_sha256"] = hashlib.sha256(feature_path.read_bytes()).hexdigest()
    (output / "preparation_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return summary
