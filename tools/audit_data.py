"""Audit the supplied SpeedLimit CSVs without changing or cleaning them.

Uses only the Python standard library. Run from any working directory:
    python Code/tools/audit_data.py
    python Code/tools/audit_data.py --resources PATH --output PATH

CSV row numbers are physical, one-based source lines. Licence-plate values
are never serialized or printed. Missing event fields are profiled, not
automatically treated as errors or converted into zero.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
SCHEMAS = {
    "sensors": ["SensorID", "SensorTypeCode"],
    "sensor_types": ["SensorTypeCode", "Type", "Unit", "Min", "Max"],
    "readings_N": ["Month", "Day", "Hour", "Minute", "Second", "Sensor", "Value"],
    "readings_NPlus1": ["Month", "Day", "Hour", "Minute", "Second", "Sensor", "Value"],
    "accidents_N": ["Month", "Day", "Hour", "Second", "LicencePlate", "Damage", "Injured", "CloseCarCm", "CloseGuardrailCm", "SkidAngle"],
    "accidents_NPlus1": ["Month", "Day", "Hour", "Second", "LicencePlate", "Damage", "Injured", "CloseCarCm", "CloseGuardrailCm", "SkidAngle"],
}
MISSING = {"", "null", "none", "nan", "n/a", "na"}


def number(value):
    """Interpret decimal commas for profiling; do not modify source data."""
    value = str(value).strip()
    if value.lower() in MISSING:
        return None
    try:
        result = float(value.replace(",", "."))
        return result if math.isfinite(result) else None
    except ValueError:
        return None


def integer(value):
    result = number(value)
    return int(result) if result is not None and result.is_integer() else None


def summary(values):
    values = sorted(x for x in values if x is not None)
    if not values:
        return {"count": 0}
    return {
        "count": len(values), "min": values[0], "max": values[-1],
        "mean": statistics.fmean(values), "median": statistics.median(values),
        "distinct_values": len(set(values)),
    }


def row_refs(rows, predicate, limit=12):
    selected = [r["_row"] for r in rows if predicate(r)]
    return {"count": len(selected), "source_rows_first": selected[:limit]}


def hour_key(row):
    parts = [integer(row.get(k, "")) for k in ("Month", "Day", "Hour")]
    if any(x is None for x in parts):
        return None
    month, day, hour = parts
    try:
        # A leap-safe reference year validates month/day only. No actual year is supplied.
        datetime(2000, month, day, hour)
    except ValueError:
        return None
    return month, day, hour


def hour_label(key):
    return "%02d-%02d %02d:00" % key


def load_table(path, role):
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        encoding, encoding_note = "utf-8-sig", "UTF-8 BOM is present."
    elif all(b < 128 for b in raw):
        encoding, encoding_note = "ascii", "ASCII-only content; the original ASCII-compatible encoding cannot be identified more narrowly."
    else:
        try:
            raw.decode("utf-8")
            encoding, encoding_note = "utf-8", "Valid UTF-8 without a BOM."
        except UnicodeDecodeError:
            encoding, encoding_note = "cp1252", "UTF-8 decoding failed; Windows-1252 fallback used."
    text = raw.decode(encoding)
    try:
        delimiter = csv.Sniffer().sniff(text[:12000], delimiters=";\t|").delimiter
    except csv.Error:
        delimiter = ";"
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    parsed = [(reader.line_num, row) for row in reader]
    schema = SCHEMAS[role]
    first = [v.strip() for v in parsed[0][1]] if parsed else []
    header_present = first == schema
    records = parsed[1:] if header_present else parsed
    rows, malformed = [], []
    for line, values in records:
        if len(values) != len(schema):
            malformed.append({"row": line, "field_count": len(values)})
        row = {key: (values[i] if i < len(values) else "") for i, key in enumerate(schema)}
        row["_row"] = line
        rows.append(row)
    duplicate_groups = defaultdict(list)
    for row in rows:
        duplicate_groups[tuple(row[k] for k in schema)].append(row["_row"])
    duplicate_groups = [v for v in duplicate_groups.values() if len(v) > 1]
    profiles = {}
    for key in schema:
        vals = [r[key] for r in rows]
        missing = Counter(v if v else "<empty>" for v in vals if v.strip().lower() in MISSING)
        profile = {"missing_count": sum(missing.values()), "missing_tokens": dict(missing), "distinct_nonmissing_count": len({v for v in vals if v.strip().lower() not in MISSING})}
        if "licen" in key.lower():
            profile["privacy"] = "Identifier values and samples omitted. Exclude from model features and public examples."
        elif key != "SensorTypeCode" and key not in {"Type", "Unit"}:
            profile["numeric"] = summary(number(v) for v in vals)
            bad = Counter(v for v in vals if v.strip().lower() not in MISSING and number(v) is None)
            profile["nonnumeric_tokens"] = dict(bad)
        else:
            profile["values"] = dict(Counter(vals))
        profiles[key] = profile
    info = {
        "filename": path.name, "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw), "encoding": encoding, "encoding_note": encoding_note,
        "delimiter": delimiter, "decimal_convention": "comma decimal values are accepted for numeric profiling",
        "header_present": header_present, "columns": schema,
        "schema_source": "source header" if header_present else "same-role declared schema; first row retained as data",
        "data_rows": len(rows), "malformed_rows": malformed,
        "columns_profile": profiles,
        "exact_duplicate_groups": len(duplicate_groups),
        "exact_duplicate_excess_rows": sum(len(g) - 1 for g in duplicate_groups),
        "exact_duplicate_source_rows_first": duplicate_groups[:12],
    }
    return rows, info


def time_profile(rows, role):
    valid = [hour_key(r) for r in rows]
    keys = sorted({k for k in valid if k is not None})
    invalid_parts = {}
    bounds = {"Month": (1, 12), "Day": (1, 31), "Hour": (0, 23), "Minute": (0, 59), "Second": (0, 59)}
    for key, (lo, hi) in bounds.items():
        if not rows or key not in rows[0]:
            continue
        invalid_parts[key] = row_refs(rows, lambda r: integer(r[key]) is None or not lo <= integer(r[key]) <= hi)
    raw_hours = Counter(r["Hour"] for r in rows)
    result = {
        "year": "N+1" if "NPlus1" in role else "N",
        "calendar_year_supplied": False, "timezone_supplied": False,
        "date_components": [k for k in ("Month", "Day", "Hour", "Minute", "Second") if rows and k in rows[0]],
        "raw_month_values": dict(Counter(r["Month"] for r in rows)),
        "raw_hour_histogram": dict(sorted(raw_hours.items(), key=lambda x: (integer(x[0]) if integer(x[0]) is not None else 99))),
        "unique_valid_calendar_hours": len(keys),
        "first_valid_hour": hour_label(keys[0]) if keys else None,
        "last_valid_hour": hour_label(keys[-1]) if keys else None,
        "rows_with_invalid_hour_or_date": sum(k is None for k in valid),
        "invalid_time_components": invalid_parts,
        "minute_values": sorted({integer(r["Minute"]) for r in rows if "Minute" in r and integer(r["Minute"]) is not None}),
        "second_values": sorted({integer(r["Second"]) for r in rows if "Second" in r and integer(r["Second"]) is not None}),
    }
    if role.startswith("accidents"):
        result["timestamp_limitation"] = "Minute is absent. Hour=24 is invalid under the 0-23 convention. Do not silently shift hours or invent minutes."
    else:
        precise = []
        for row in rows:
            key = hour_key(row)
            minute, second = integer(row["Minute"]), integer(row["Second"])
            if key is not None and minute is not None and second is not None and 0 <= minute < 60 and 0 <= second < 60:
                precise.append(key + (minute, second))
        def stamp(parts):
            return "%02d-%02d %02d:%02d:%02d" % parts
        result["first_valid_reading_timestamp_without_year"] = stamp(min(precise)) if precise else None
        result["last_valid_reading_timestamp_without_year"] = stamp(max(precise)) if precise else None
        result["adjacent_timestamp_decreases_in_source_order"] = sum(a > b for a, b in zip(precise, precise[1:]))
    return result


def audit_readings(rows, sensors, types):
    sensor_map = {integer(r["SensorID"]): r["SensorTypeCode"] for r in sensors}
    type_map = defaultdict(list)
    for row in types:
        type_map[row["SensorTypeCode"].strip()].append(row)
    by_sensor = defaultdict(list)
    for row in rows:
        by_sensor[row["Sensor"].strip()].append(row)
    expected_january = {(1, d, h) for d in range(1, 32) for h in range(24)}
    per_sensor = {}
    for sid, group in sorted(by_sensor.items(), key=lambda x: integer(x[0]) if integer(x[0]) is not None else 999):
        sensor_id = integer(sid)
        code = sensor_map.get(sensor_id)
        candidates = type_map.get(code.strip(), []) if code else []
        hours = {hour_key(r) for r in group} - {None}
        hour_counts = Counter(hour_key(r) for r in group if hour_key(r) is not None)
        ranges = []
        for typ in candidates:
            lo, hi = number(typ["Min"]), number(typ["Max"])
            outside = [r for r in group if number(r["Value"]) is not None and not lo <= number(r["Value"]) <= hi]
            ranges.append({"type": typ["Type"], "unit_as_supplied": typ["Unit"], "min_as_supplied": lo, "max_as_supplied": hi, "outside_range_count": len(outside), "outside_range_source_rows_first": [r["_row"] for r in outside[:12]]})
        per_sensor[sid or "<missing>"] = {
            "row_count": len(group), "sensor_type_code": code,
            "metadata_status": "special speed-limit record" if sensor_id == -1 else "missing sensor identifier" if sensor_id is None else "missing sensor definition" if code is None else "ambiguous type code" if len(candidates) > 1 else "defined",
            "value_summary": summary(number(r["Value"]) for r in group),
            "value_counts": dict(Counter(r["Value"] for r in group)) if sensor_id == -1 else None,
            "missing_or_nonnumeric_values": row_refs(group, lambda r: number(r["Value"]) is None),
            "unique_valid_hours": len(hours),
            "missing_january_hours_under_raw_dates": len(expected_january - hours),
            "missing_january_hours_first": [hour_label(k) for k in sorted(expected_january - hours)[:12]],
            "multiple_rows_same_hour": {hour_label(k): v for k, v in sorted(hour_counts.items()) if v > 1},
            "candidate_range_checks": ranges,
        }
    observed = {integer(r["Sensor"]) for r in rows}
    exact_times = defaultdict(list)
    for row in rows:
        exact_times[tuple(row[k] for k in ("Month", "Day", "Hour", "Minute", "Second", "Sensor"))].append(row)
    conflicts = [g for g in exact_times.values() if len(g) > 1 and len({r["Value"] for r in g}) > 1]
    return {
        "observed_sensor_ids": sorted(s for s in observed if s is not None),
        "unknown_sensor_ids_excluding_special_minus_one": sorted(s for s in observed if s is not None and s != -1 and s not in sensor_map),
        "defined_sensor_ids_without_readings": sorted(set(sensor_map) - observed),
        "missing_or_invalid_sensor_identifier": row_refs(rows, lambda r: integer(r["Sensor"]) is None),
        "conflicting_values_at_identical_sensor_timestamp": {"groups": len(conflicts), "source_rows_first": [[r["_row"] for r in g] for g in conflicts[:12]]},
        "per_sensor": per_sensor,
    }


def audit_accidents(rows):
    fields = ["Damage", "Injured", "CloseCarCm", "CloseGuardrailCm", "SkidAngle"]
    presence = Counter("+".join(k for k in fields if r[k].strip().lower() not in MISSING) or "none" for r in rows)
    with_proximity_or_skid = sum(any(r[k].strip().lower() not in MISSING for k in fields[2:]) for r in rows)
    return {
        "record_count": len(rows),
        "record_count_is_not_a_verified_near_accident_count": True,
        "field_presence_patterns": dict(presence),
        "records_with_any_proximity_or_skid_field": with_proximity_or_skid,
        "records_with_any_damage_or_injury_field": sum(any(r[k].strip().lower() not in MISSING for k in fields[:2]) for r in rows),
        "records_with_all_event_fields_empty": sum(all(r[k].strip().lower() in MISSING for k in fields) for r in rows),
        "all_event_fields_empty_source_rows": [r["_row"] for r in rows if all(r[k].strip().lower() in MISSING for k in fields)],
        "records_with_skid_field_present": sum(r["SkidAngle"].strip().lower() not in MISSING for r in rows),
        "negative_event_values": {k: row_refs(rows, lambda r, key=k: number(r[key]) is not None and number(r[key]) < 0) for k in fields},
        "noninteger_injury_values": row_refs(rows, lambda r: number(r["Injured"]) is not None and integer(r["Injured"]) is None),
        "label_status": "Near-accident thresholds, missing-field meaning and event/hour conventions remain unconfirmed. Presence counts describe populated fields only.",
        "privacy": "LicencePlate values omitted from this audit and stdout; do not use identifiers as predictive features or presentation samples.",
    }


def slide17_checks(readings, types, sensors):
    """Verify supplied slide examples using raw records; apply no repairs."""
    checks = []
    def add(item, result, status, note=""):
        checks.append({"slide_item": item, "status": status, "evidence": result, "note": note})
    codes = defaultdict(list)
    for r in types:
        codes[r["SensorTypeCode"].strip()].append(r["Type"])
    add("Sensor type T means both traffic and temperature", codes.get("T", []), "confirmed" if len(codes.get("T", [])) > 1 else "not_found")
    for year, rows in readings.items():
        prefix = year + ": "
        id_values = {integer(r["Sensor"]) for r in rows}
        sensor_ids = {integer(r["SensorID"]) for r in sensors}
        add(prefix + "Sensor 19 has readings but no description", row_refs(rows, lambda r: integer(r["Sensor"]) == 19), "confirmed" if 19 in id_values and 19 not in sensor_ids else "not_found", "Its type cannot be assigned from values alone.")
        add(prefix + "Sensors 7, 10, 17 have no readings", {str(s): sum(integer(r["Sensor"]) == s for r in rows) for s in (7, 10, 17)}, "confirmed" if not {7, 10, 17} & id_values else "differs")
        tests = [
            ("Sensor18 missing first five readings Jan26", lambda r: integer(r["Sensor"]) == 18 and integer(r["Month"]) == 1 and integer(r["Day"]) == 26 and integer(r["Hour"]) in range(5) and number(r["Value"]) is None),
            ("Sensor identifier empty Jan24 hours22-23", lambda r: not r["Sensor"].strip() and integer(r["Day"]) == 24 and integer(r["Hour"]) in (22, 23)),
            ("Sensor2 NULL month Jan18 first six hours", lambda r: integer(r["Sensor"]) == 2 and r["Month"].strip().lower() == "null" and integer(r["Day"]) == 18 and integer(r["Hour"]) in range(6)),
            ("Sensor13 wind direction above400 Jan1 hours12-14", lambda r: integer(r["Sensor"]) == 13 and integer(r["Day"]) == 1 and integer(r["Hour"]) in (12,13,14) and number(r["Value"]) is not None and number(r["Value"]) > 400),
            ("Sensor9 month written January", lambda r: integer(r["Sensor"]) == 9 and r["Month"].strip().lower() == "january"),
            ("Sensor11 out-of-range water Jan17 hour9", lambda r: integer(r["Sensor"]) == 11 and integer(r["Day"]) == 17 and integer(r["Hour"]) == 9 and number(r["Value"]) is not None and not 0 <= number(r["Value"]) <= 10000),
        ]
        for title, predicate in tests:
            evidence = row_refs(rows, predicate)
            add(prefix + title, evidence, "confirmed" if evidence["count"] else "not_found")
        count = sum(integer(r["Sensor"]) == 15 and integer(r["Day"]) == 10 for r in rows)
        add(prefix + "Sensor15 missing entire Jan10", {"observed_rows": count, "expected_under_hourly_assumption": 24}, "confirmed" if count == 0 else "not_found")
        group = [r for r in rows if integer(r["Sensor"]) == 4 and integer(r["Day"]) == 12]
        unique = {tuple(r[k] for k in SCHEMAS["readings_N"]) for r in group}
        add(prefix + "Sensor4 Jan12 duplicated", {"rows": len(group), "exact_duplicate_excess": len(group) - len(unique)}, "confirmed" if len(group) > len(unique) else "not_found")
        speed = [r for r in rows if integer(r["Sensor"]) == -1 and integer(r["Day"]) == 6 and integer(r["Hour"]) == 20]
        add(prefix + "Unexplained speed limit Jan6 hour20", {"source_rows": [r["_row"] for r in speed], "values": [r["Value"] for r in speed]}, "requires_domain_review", "A numeric speed change is observable; its justification cannot be determined by a structural audit.")
        zero_speed_times = {tuple(r[k] for k in ("Month", "Day", "Hour", "Minute", "Second")) for r in rows if integer(r["Sensor"]) in (6, 14) and number(r["Value"]) == 0}
        calm = [r for r in rows if integer(r["Sensor"]) in (5,13) and tuple(r[k] for k in ("Month","Day","Hour","Minute","Second")) in zero_speed_times]
        add(prefix + "Wind direction when wind speed is zero", {"direction_rows_sharing_timestamp_with_any_zero_wind_strength": len(calm)}, "requires_semantic_decision", "This diagnostic assumes no pairing between sensors. Sensor locations/pairing and the relevance of direction during calm periods require a feature-engineering decision. No automatic change applied.")
    add("0 and 360 degrees represent equivalent directions", {}, "feature_design_question", "Circular encoding/wraparound is a design issue, not proof that either endpoint is invalid.")
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resources", type=Path, default=PROJECT.parent / "Resources")
    parser.add_argument("--output", type=Path, default=PROJECT / "docs" / "data_audit.json")
    args = parser.parse_args()
    resources = args.resources.resolve()
    candidates = sorted(resources.glob("*.csv"))
    matches = {}
    for path in candidates:
        name = path.name.lower()
        if "sensortypes" in name:
            role = "sensor_types"
        elif "sensorreadings" in name:
            role = "readings_NPlus1" if "nplus1" in name else "readings_N"
        elif "sensors" in name:
            role = "sensors"
        elif "accidents" in name:
            role = "accidents_NPlus1" if "nplus1" in name else "accidents_N"
        else:
            continue
        if role in matches:
            parser.error(f"Multiple CSVs match {role}; choose a resource directory containing one version per role.")
        matches[role] = path
    missing = set(SCHEMAS) - set(matches)
    if missing:
        parser.error("Missing CSV roles: " + ", ".join(sorted(missing)))
    if args.output.suffix.lower() != ".json" or args.output.resolve() in {p.resolve() for p in matches.values()}:
        parser.error("Output must be a .json file separate from the source CSVs.")
    tables, files = {}, {}
    for role in SCHEMAS:
        tables[role], files[role] = load_table(matches[role], role)
    for role in ("readings_N", "readings_NPlus1", "accidents_N", "accidents_NPlus1"):
        files[role]["time_profile"] = time_profile(tables[role], role)
    for role in ("readings_N", "readings_NPlus1"):
        files[role]["sensor_audit"] = audit_readings(tables[role], tables["sensors"], tables["sensor_types"])
    for role in ("accidents_N", "accidents_NPlus1"):
        files[role]["event_audit"] = audit_accidents(tables[role])
    type_codes = defaultdict(list)
    for row in tables["sensor_types"]:
        type_codes[row["SensorTypeCode"].strip()].append(row["Type"])
    raw_type_codes = {r["SensorTypeCode"] for r in tables["sensor_types"]}
    metadata = {
        "sensor_definitions": len(tables["sensors"]),
        "sensor_type_rows": len(tables["sensor_types"]),
        "candidate_range_lookup": "Surrounding whitespace is ignored only to locate candidate metadata rows during this audit. Source codes remain unchanged. Ambiguous T ranges are reported separately, not resolved.",
        "ambiguous_type_codes_after_trimming": {k: v for k, v in type_codes.items() if len(v) > 1},
        "sensor_codes_without_exact_type_match": dict(Counter(r["SensorTypeCode"] for r in tables["sensors"] if r["SensorTypeCode"] not in raw_type_codes)),
        "type_codes_with_surrounding_whitespace": [{"source_row": r["_row"], "raw_code": r["SensorTypeCode"]} for r in tables["sensor_types"] if r["SensorTypeCode"] != r["SensorTypeCode"].strip()],
        "unit_questions": [
            "Light metadata is lux; the router threshold in the slides is 500 millilux (=0.5 lux). Units must be explicit before comparison.",
            "Wind-strength unit is supplied literally as hm/h. Confirm whether this is intended before conversion.",
            "Temperature and traffic share code T, so their units cannot be selected reliably from that code alone.",
        ],
    }
    a, b = tables["readings_N"], tables["readings_NPlus1"]
    cols = SCHEMAS["readings_N"]
    keys_a = Counter(tuple(r[k] for k in cols) for r in a)
    keys_b = Counter(tuple(r[k] for k in cols) for r in b)
    common_rows = sum((keys_a & keys_b).values())
    year_comparison = {
        "reading_rows_N": len(a), "reading_rows_NPlus1": len(b),
        "exact_identical_reading_rows_multiset": common_rows,
        "fraction_N_reading_rows_repeated_in_NPlus1": common_rows / len(a) if a else None,
        "event_records_N": len(tables["accidents_N"]), "event_records_NPlus1": len(tables["accidents_NPlus1"]),
        "event_record_count_change": len(tables["accidents_NPlus1"]) - len(tables["accidents_N"]),
        "reading_value_distributions_by_sensor": {s: {"N": files["readings_N"]["sensor_audit"]["per_sensor"].get(s, {}).get("value_summary"), "NPlus1": files["readings_NPlus1"]["sensor_audit"]["per_sensor"].get(s, {}).get("value_summary")} for s in sorted(set(files["readings_N"]["sensor_audit"]["per_sensor"]) | set(files["readings_NPlus1"]["sensor_audit"]["per_sensor"]))},
        "interpretation": "All supplied N+1 reading observations repeat N exactly, so N+1 supplies no new observed feature conditions. Changed event records can support an outcome comparison after label/time decisions, but cannot establish performance on new feature conditions. Record counts are not verified near-accident counts.",
    }
    result = {
        "audit_version": 1, "resources_directory": str(resources),
        "scope": "Read-only structural and descriptive audit. No data repairs, model fitting, near-accident thresholds or labels are produced.",
        "row_number_convention": "Physical one-based CSV line numbers, including a header when present.",
        "files": files, "metadata_integrity": metadata, "year_comparison": year_comparison,
        "slide17_verification": slide17_checks({"N": a, "NPlus1": b}, tables["sensor_types"], tables["sensors"]),
        "mvp_readiness": {
            "course_prototype": "The six files support a small, local course MVP after documented repairs and label/time decisions. A neural network and full evaluation are not ready to run directly on the raw CSVs.",
            "training_grain": "January offers at most 31*24=744 distinct hourly windows per symbolic year before filtering/repair. Thousands of sensor rows are not thousands of independent labeled hourly samples.",
            "label_blockers": ["Define what constitutes one near-accident, including thresholds, duplicate-event policy and meanings of blank event fields.", "Resolve accident Hour=24/0 conventions and the absent Minute column before hourly joins.", "Verify sensor type identities, units, missing definitions and whether absent event records establish zero events."],
            "evaluation": "Use chronological holdout within N after preparation and compare a simple baseline. N+1 contains both sensors and events, but its sensor observations repeat N exactly. It supports an outcome comparison on repeated conditions after compatible labels/time conventions are established; do not describe it as evaluation on new feature conditions. Do not tune against the final held-out period.",
            "time_alignment": "Sensor readings occur within their hour at different minutes/seconds. Specify the prediction cutoff and whether event bins refer to the following or preceding interval before constructing features; a same-hour join alone does not establish temporal validity.",
            "coverage_limitations": "Both nominal years contain only January. No full annual/seasonal generalization or causal safety guarantee follows from these observational records.",
            "retraining": "A retraining decision requires evaluated prediction performance plus schema/feature checks. Repeated or very similar features alone neither prove nor disprove need; N+1 event outcomes can support the check once compatible labels are available.",
            "air_quality": "No air-quality observations are included in these six CSVs. The LLM/API branch needs a separately specified input source or clearly labeled demo fixtures.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print("Audit saved:", args.output.resolve())
    for role, info in files.items():
        print(f"{role}: {info['data_rows']} rows, header={info['header_present']}, duplicate excess={info['exact_duplicate_excess_rows']}")
    print("Source CSVs were not changed. Licence-plate values are omitted.")


if __name__ == "__main__":
    main()
