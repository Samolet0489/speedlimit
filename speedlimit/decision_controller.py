"""C6. Evaluate the weather and air branches and record simulated decisions."""

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from threading import Lock
from uuid import UUID, uuid4

from .air_quality import REPLAY_SCENARIOS
from .weather_router import evaluate_measurements

BASE_LIMIT = 80
REDUCED_LIMITS = (40, 50, 60, 70)
ALLOWED_LIMITS = (*REDUCED_LIMITS, BASE_LIMIT)


def _utc_now():
    return datetime.now(timezone.utc)


def _timestamp(value):
    if value.utcoffset() is None:
        raise ValueError("The decision clock must provide a timezone-aware timestamp.")
    return value.astimezone(timezone.utc).isoformat()


def _branch(status, reason, limit_kmh=None, **details):
    return {"status": status, "limit_kmh": limit_kmh, "reason": reason, **details}


def _validated_branch(recommendation, limits, component):
    result = asdict(recommendation)
    limit = result.get("limit_kmh")
    if result.get("status") == "recommended" and type(limit) is int and limit in limits:
        return result
    if result.get("status") == "review_needed" and limit is None:
        return result
    return {**result, **_branch("review_needed", f"{component} returned an invalid speed recommendation.")}


class DecisionController:
    """Each request is independent. The GUI owns the current-request token."""

    def __init__(self, weather_predictor, air_for_request, *, clock=_utc_now, log_path=None):
        self._weather = weather_predictor
        self._air_for_request = air_for_request
        self._clock = clock
        self._log_path = Path(log_path) if log_path is not None else None
        self._log_lock = Lock()

    def _record(self, event, details):
        if self._log_path is None:
            return
        line = json.dumps({"event": event, **details}, ensure_ascii=False, allow_nan=False) + "\n"
        with self._log_lock:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(line)

    def evaluate(self, request):
        now = self._clock()
        mode = request.get("mode", "replay")
        result = {
            "decision_id": str(uuid4()), "mode": mode if mode in ("replay", "live") else "invalid",
            "evaluated_at": _timestamp(now), "status": "review_needed", "limit_kmh": None,
            "inputs": {}, "reasons": [],
            "weather": _branch("skipped", "Weather evaluation awaits valid inputs."),
            "air": _branch("skipped", "Air evaluation awaits valid inputs."),
        }
        try:
            if mode not in ("replay", "live"):
                raise ValueError("Choose replay or live mode.")
            route = evaluate_measurements(request.get("illuminance_lux"), request.get("water_um"),
                                          request.get("temperature_c"))
        except (TypeError, ValueError) as error:
            result["reasons"] = [str(error)]
            self._record("decision", result)
            return result

        result["inputs"] = {
            "illuminance_millilux": str(route.illuminance_millilux),
            "water_um": str(route.water_um), "temperature_c": str(route.temperature_c),
        }
        if mode == "replay" and request.get("scenario") in REPLAY_SCENARIOS:
            result["inputs"]["replay_scenario"] = request["scenario"]

        if route.reduce:
            try:
                weather = _validated_branch(self._weather.evaluate(route), REDUCED_LIMITS, "Weather model")
            except (OSError, RuntimeError, TypeError, ValueError):
                weather = _branch("review_needed", "The weather model could not evaluate this request.")
        else:
            weather = _branch("recommended", "Light and road conditions leave the weather cap at 80 km/h.", BASE_LIMIT)
        weather.update(triggered=route.reduce, dark=route.dark, black_ice=route.black_ice)

        try:
            air = _validated_branch(self._air_for_request(request).evaluate(now), ALLOWED_LIMITS, "Air-quality advisor")
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            # The adapter handles service failures internally and reports safe reasons.
            air = _branch("review_needed", str(error) if isinstance(error, ValueError) else
                          "The air-quality branch could not evaluate this request.")

        result.update(weather=weather, air=air, reasons=[weather["reason"], air["reason"]])
        if weather["status"] == air["status"] == "recommended":
            result.update(status="recommended", limit_kmh=min(weather["limit_kmh"], air["limit_kmh"]))
        self._record("decision", result)
        return result

    def acknowledge(self, result):
        """Record the operator's acknowledgment of a valid simulated result."""
        limit = result.get("limit_kmh")
        if result.get("status") != "recommended" or type(limit) is not int or limit not in ALLOWED_LIMITS:
            raise ValueError("Only a current recommended result can be acknowledged.")
        decision_id = result.get("decision_id")
        if not isinstance(decision_id, str):
            raise ValueError("The result has no valid decision identifier.")
        try:
            UUID(decision_id)
        except ValueError as error:
            raise ValueError("The result has no valid decision identifier.") from error
        self._record("acknowledgment", {"decision_id": decision_id, "limit_kmh": limit,
                                        "acknowledged_at": _timestamp(self._clock())})
