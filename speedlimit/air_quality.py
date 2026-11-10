"""C5 air quality advice using a versioned course policy and replaceable clients."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, DecimalException
from http.client import HTTPException
import json
import math
from pathlib import Path
import re
from time import monotonic
from typing import Callable, Literal, Mapping, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


POLLUTANTS = ("pm2_5", "pm10", "nitrogen_dioxide", "ozone", "sulphur_dioxide")
CAPS = (80, 70, 60, 50, 40)
MAX_BODY_BYTES = 128 * 1024
OPEN_METEO_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
OLLAMA_CHAT_URL = "http://127.0.0.1:11434/api/chat"
OLLAMA_MODEL = "qwen3:4b"
AQI_FIELD = "european_aqi"
REPLAY_SCENARIOS = ("good", "poor", "custom", "stale", "malformed", "policy_mismatch")
REPLAY_AQI_MAX = 500
Transport = Callable[[Request, float], dict]


@dataclass(frozen=True, slots=True)
class AirRecommendation:
    status: Literal["recommended", "review_needed"]
    limit_kmh: int | None
    reason: str
    policy_id: str | None
    data_time: str | None
    data_source: str
    response_source: str
    european_aqi: str | None = None
    rounded_european_aqi: int | None = None


@dataclass(frozen=True, slots=True)
class AirObservation:
    valid_time: datetime
    european_aqi: int | float | Decimal
    pollutants_ug_m3: Mapping[str, int | float | Decimal]


class AirDataClient(Protocol):
    source: str

    def fetch(self, now: datetime) -> AirObservation: ...


class AirLLMClient(Protocol):
    source: str

    def recommend(self, policy_input: dict) -> str: ...


class _OllamaUnavailable(ValueError):
    """A local runtime failure with an operator-safe message."""


def _unique_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field")
        result[key] = value
    return result


def _invalid_json_constant(_value):
    raise ValueError("Non-finite JSON constant")


def _json_object(raw):
    try:
        parsed = json.loads(
            raw, object_pairs_hook=_unique_fields, parse_float=Decimal,
            parse_constant=_invalid_json_constant,
        )
    except (DecimalException, RecursionError) as error:
        raise ValueError("Invalid JSON number or structure") from error
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object")
    return parsed


def _utc(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise ValueError("A timezone-aware timestamp is required")
    return value.astimezone(timezone.utc)


def _bounded_number(value, lower: int, upper: int) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise ValueError("Expected a numeric air-quality field")
    try:
        number = Decimal(str(value))
    except DecimalException as error:
        raise ValueError("Invalid numeric air-quality field") from error
    if not number.is_finite() or not lower <= number <= upper:
        raise ValueError("Air-quality field is outside the configured range")
    return number


def parse_replay_aqi(value) -> Decimal:
    """Validate the numeric AQI accepted by both operator entry points."""
    try:
        return _bounded_number(Decimal(value.strip()) if isinstance(value, str) else value, 0, REPLAY_AQI_MAX)
    except (DecimalException, ValueError) as error:
        raise ValueError(f"Enter a European AQI from 0 to {REPLAY_AQI_MAX}, using a decimal point.") from error


@dataclass(frozen=True, slots=True)
class _Band:
    min_aqi: int
    max_aqi: int
    limit_kmh: int
    reason_code: str


@dataclass(frozen=True, slots=True)
class _Policy:
    policy_id: str
    max_data_age_minutes: int
    max_pollutant_ug_m3: int
    bands: tuple[_Band, ...]


def _load_policy(path: Path) -> _Policy:
    data = _json_object(Path(path).read_text(encoding="utf-8-sig"))
    if (
        type(data.get("schema_version")) is not int or data["schema_version"] != 1
        or data.get("aqi_scale") != "European AQI" or data.get("aqi_rounding") != "ceil"
    ):
        raise ValueError("Unsupported air-quality policy schema or AQI scale")
    policy_id = data.get("policy_id")
    if not isinstance(policy_id, str) or not re.fullmatch(r"[a-z0-9-]{1,48}", policy_id):
        raise ValueError("Invalid air-quality policy ID")
    age = data.get("max_data_age_minutes")
    max_pollutant = data.get("max_pollutant_ug_m3")
    if type(age) is not int or not 1 <= age <= 1440:
        raise ValueError("Invalid air-quality freshness bound")
    if type(max_pollutant) is not int or not 1 <= max_pollutant <= 100000:
        raise ValueError("Invalid pollutant concentration bound")
    raw_bands = data.get("bands")
    if not isinstance(raw_bands, list) or len(raw_bands) != len(CAPS):
        raise ValueError("The air-quality policy must contain five bands")
    bands = []
    next_aqi = 0
    codes = set()
    for raw, cap in zip(raw_bands, CAPS):
        if not isinstance(raw, dict) or set(raw) != {"min_aqi", "max_aqi", "limit_kmh", "reason_code"}:
            raise ValueError("Invalid air-quality band schema")
        low, high, limit = (raw[key] for key in ("min_aqi", "max_aqi", "limit_kmh"))
        code = raw["reason_code"]
        if (
            any(type(value) is not int for value in (low, high, limit))
            or low != next_aqi or not low <= high <= 1000 or limit != cap
            or not isinstance(code, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,31}", code)
            or code in codes
        ):
            raise ValueError("Air-quality bands must be closed, contiguous and have distinct reasons")
        bands.append(_Band(low, high, limit, code))
        codes.add(code)
        next_aqi = high + 1
    return _Policy(policy_id, age, max_pollutant, tuple(bands))


class AirQualityAdvisor:
    """Check forecast and model output before returning a course-policy cap."""

    def __init__(self, policy_path: Path, data_client: AirDataClient, llm_client: AirLLMClient):
        self._policy = _load_policy(policy_path)
        self._data = data_client
        self._llm = llm_client

    def evaluate(self, now: datetime) -> AirRecommendation:
        policy = self._policy
        data_source = self._data.source
        raw_aqi = rounded_aqi = None

        def review(reason: str, data_time: str | None = None, response_source: str = "not_requested"):
            return AirRecommendation("review_needed", None, reason, policy.policy_id, data_time,
                                     data_source, response_source, raw_aqi, rounded_aqi)

        try:
            now = _utc(now)
        except (ValueError, OverflowError):
            return review("The evaluation clock requires a valid timezone-aware timestamp.")
        try:
            observation = self._data.fetch(now)
        except TimeoutError:
            return review("The air-data request timed out.")
        except (OSError, ValueError, TypeError, OverflowError):
            return review("The air-data request failed or returned an invalid forecast.")
        if not isinstance(observation, AirObservation):
            return review("The air-data response has an invalid schema.")
        try:
            valid_time = _utc(observation.valid_time)
        except (ValueError, OverflowError):
            return review("The air forecast has an invalid valid-time timestamp.")
        data_time = valid_time.isoformat()
        age = now - valid_time
        if age < timedelta(0):
            return review("The air forecast valid time is in the future.", data_time)
        if age > timedelta(minutes=policy.max_data_age_minutes):
            return review("The air forecast valid time is stale.", data_time)
        try:
            measured_aqi = _bounded_number(observation.european_aqi, 0, policy.bands[-1].max_aqi)
            raw_aqi, rounded_aqi = str(measured_aqi), math.ceil(measured_aqi)
            pollutants = {
                key: float(_bounded_number(observation.pollutants_ug_m3[key], 0, policy.max_pollutant_ug_m3))
                for key in POLLUTANTS
            }
        except (KeyError, ValueError, TypeError):
            return review("European AQI or a required pollutant concentration is missing or invalid.", data_time)
        expected = next(band for band in policy.bands if band.min_aqi <= rounded_aqi <= band.max_aqi)
        # Pass only validated numeric values and the local policy to the model.
        policy_input = {
            "policy_id": policy.policy_id, "european_aqi_ceiling": rounded_aqi,
            "pollutants_ug_m3": pollutants, "closed_bands": [asdict(band) for band in policy.bands],
        }
        response_source = self._llm.source
        try:
            response = self._llm.recommend(policy_input)
        except _OllamaUnavailable as error:
            return review(str(error), data_time, response_source)
        except TimeoutError:
            return review("The air-policy model request timed out.", data_time, response_source)
        except (OSError, ValueError, TypeError, OverflowError):
            return review("The air-policy model request failed or returned an incomplete completion.", data_time, response_source)
        try:
            if not isinstance(response, str) or len(response) > 4096:
                raise ValueError("Invalid completion content")
            result = _json_object(response)
        except (ValueError, TypeError, RecursionError):
            return review("The air-policy model did not return a valid JSON object.", data_time, response_source)
        if (
            set(result) != {"policy_id", "limit_kmh", "reason_code"}
            or type(result["limit_kmh"]) is not int
            or not isinstance(result["policy_id"], str)
            or not isinstance(result["reason_code"], str)
        ):
            return review("The air-policy model output has an invalid schema.", data_time, response_source)
        if (result["policy_id"], result["limit_kmh"], result["reason_code"]) != (policy.policy_id, expected.limit_kmh, expected.reason_code):
            return review("The air-policy model recommendation contradicts the course policy.", data_time, response_source)
        band_label = expected.reason_code.replace("_", " ")
        reason = f"European AQI {raw_aqi} (rounded up to {rounded_aqi}) is in the {band_label} band. Course policy cap is {expected.limit_kmh} km/h."
        return AirRecommendation("recommended", expected.limit_kmh, reason, policy.policy_id, data_time,
                                 data_source, response_source, raw_aqi, rounded_aqi)


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _request_json(request: Request, timeout: float) -> dict:
    # Keep local inference on loopback even on machines with a configured HTTP proxy.
    handlers = [_NoRedirect()]
    if request.full_url == OLLAMA_CHAT_URL:
        handlers.append(ProxyHandler({}))
    try:
        with build_opener(*handlers).open(request, timeout=timeout) as response:
            raw = response.read(MAX_BODY_BYTES + 1)
    except HTTPException as error:
        raise ValueError("The API HTTP response was incomplete or invalid") from error
    if len(raw) > MAX_BODY_BYTES:
        raise ValueError("API response is too large")
    return _json_object(raw.decode("utf-8"))


class OpenMeteoAirDataClient:
    """Select the current UTC valid-hour row of the CAMS Europe forecast."""

    source = "Open-Meteo / CAMS ENSEMBLE (CAMS Europe hourly forecast)"

    def __init__(self, latitude: float, longitude: float, *, transport: Transport = _request_json):
        self._latitude = float(_bounded_number(latitude, -90, 90))
        self._longitude = float(_bounded_number(longitude, -180, 180))
        self._transport = transport

    def fetch(self, now: datetime) -> AirObservation:
        hour = _utc(now).replace(minute=0, second=0, microsecond=0)
        fields = (AQI_FIELD, *POLLUTANTS)
        query = urlencode({
            "latitude": self._latitude, "longitude": self._longitude,
            "hourly": ",".join(fields), "domains": "cams_europe", "timezone": "GMT",
            "timeformat": "unixtime", "forecast_hours": 2,
        })
        request = Request(f"{OPEN_METEO_URL}?{query}", headers={"Accept": "application/json"}, method="GET")
        payload = self._transport(request, 15)
        if not isinstance(payload, dict) or payload.get("error") or type(payload.get("utc_offset_seconds")) is not int or payload["utc_offset_seconds"] != 0:
            raise ValueError("Unexpected Open-Meteo forecast format")
        hourly, units = payload.get("hourly"), payload.get("hourly_units")
        if not isinstance(hourly, dict) or not isinstance(units, dict) or units.get(AQI_FIELD) != "EAQI" or units.get("time") != "unixtime":
            raise ValueError("European AQI is missing from the hourly forecast")
        if any(units.get(key) not in ("µg/m³", "μg/m³", "ug/m3") for key in POLLUTANTS):
            raise ValueError("Missing or unexpected pollutant concentration unit")
        times = hourly.get("time")
        if not isinstance(times, list) or any(type(value) is not int for value in times):
            raise ValueError("Invalid UTC Unix timestamps in hourly forecast")
        matching = [index for index, epoch in enumerate(times) if epoch == int(hour.timestamp())]
        if len(matching) != 1:
            raise ValueError("Expected exactly one row for the current UTC forecast hour")
        values = {}
        for field in fields:
            series = hourly.get(field)
            if not isinstance(series, list) or len(series) != len(times):
                raise ValueError("Forecast arrays have inconsistent lengths")
            values[field] = series[matching[0]]
        return AirObservation(hour, values[AQI_FIELD], {key: values[key] for key in POLLUTANTS})


class OllamaChatLLMClient:
    """Ask the project's local model for a structured policy recommendation."""

    source = f"Local Ollama ({OLLAMA_MODEL})"

    def __init__(self, *, transport: Transport = _request_json):
        self._transport = transport

    def recommend(self, policy_input: dict) -> str:
        bands = policy_input["closed_bands"]
        aqi = policy_input["european_aqi_ceiling"]
        expected = next(band for band in bands if band["min_aqi"] <= aqi <= band["max_aqi"])
        deadline = monotonic() + 120
        content = self._chat(policy_input, bands, 120)
        try:
            answer = _json_object(content)
        except ValueError:
            answer = {}
        if type(answer.get("limit_kmh")) is int and answer == {
            "policy_id": policy_input["policy_id"],
            "limit_kmh": expected["limit_kmh"],
            "reason_code": expected["reason_code"],
        }:
            return content

        # On a policy mismatch, request one new model response with only the
        # applicable values allowed by the schema. The advisor still validates it.
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise TimeoutError("Local Ollama timed out")
        return self._chat(policy_input, [expected], remaining)

    def _chat(self, policy_input: dict, allowed_bands: list[dict], timeout: float) -> str:
        bands = policy_input["closed_bands"]
        cutoffs = [f"if AQI <= {band['max_aqi']}, use {band['reason_code']}" for band in bands]
        caps = [f"{band['reason_code']} = {band['limit_kmh']} km/h" for band in bands]
        body = {
            "model": OLLAMA_MODEL,
            "stream": False,
            "think": False,
            "options": {"temperature": 0, "num_ctx": 2048, "num_predict": 128},
            "format": {
                "type": "object",
                "properties": {
                    "policy_id": {"const": policy_input["policy_id"]},
                    "limit_kmh": {"type": "integer", "enum": [band["limit_kmh"] for band in allowed_bands]},
                    "reason_code": {"enum": [band["reason_code"] for band in allowed_bands]},
                },
                "required": ["policy_id", "limit_kmh", "reason_code"],
                "additionalProperties": False,
            },
            "messages": [
                {"role": "system", "content": (
                    "Choose reason_code by testing the AQI cutoffs in order and stopping at the first true "
                    "condition. Then use that reason_code to look up limit_kmh. AQI and speed are "
                    "different quantities. Return only JSON with policy_id, limit_kmh and reason_code."
                )},
                {"role": "user", "content": "\n".join((
                    f"Policy ID: {policy_input['policy_id']}",
                    "Pollutants (µg/m³, context only): " + json.dumps(policy_input["pollutants_ug_m3"], allow_nan=False),
                    "Test these cutoffs in order:",
                    *cutoffs,
                    "Look up the speed for the selected reason:",
                    *caps,
                    f"Rounded AQI: {policy_input['european_aqi_ceiling']}",
                ))},
            ],
        }
        request = Request(
            OLLAMA_CHAT_URL, data=json.dumps(body, allow_nan=False).encode("utf-8"), method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            payload = self._transport(request, timeout)
        except HTTPError as error:
            if error.code == 404:
                raise _OllamaUnavailable(f"The local Ollama model {OLLAMA_MODEL} is unavailable. Ask the project maintainer to run Live setup.") from error
            raise
        except URLError as error:
            if isinstance(error.reason, TimeoutError):
                raise TimeoutError("Local Ollama timed out") from error
            raise _OllamaUnavailable("Local Ollama could not be reached. Ask the project maintainer to run Live setup.") from error
        message = payload.get("message") if isinstance(payload, dict) else None
        if not isinstance(message, dict) or payload.get("done") is not True or payload.get("done_reason") != "stop":
            raise ValueError("Ollama did not complete its recommendation")
        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError("Completion lacks text content")
        return content


class _ReplayAirDataClient:
    source = "Synthetic replay air data"

    def __init__(self, fixture: dict):
        self._age_minutes = float(_bounded_number(fixture.get("age_minutes"), 0, 10080))
        self._fixture = fixture

    def fetch(self, now: datetime) -> AirObservation:
        return AirObservation(
            now - timedelta(minutes=self._age_minutes), self._fixture.get(AQI_FIELD),
            self._fixture.get("pollutants_ug_m3"),
        )


class _ReplayLLMClient:
    source = "Synthetic replay LLM fixture"

    def __init__(self, fixture: dict):
        self._content = fixture.get("llm_response")

    def recommend(self, policy_input: dict) -> str:
        return self._content if isinstance(self._content, str) else json.dumps(self._content, allow_nan=False)


class _CustomReplayResponse:
    """Simulate a policy-following response for a user-entered AQI."""

    source = "Synthetic replay policy response (custom AQI)"

    def recommend(self, policy_input: dict) -> str:
        aqi = policy_input["european_aqi_ceiling"]
        band = next(band for band in policy_input["closed_bands"] if band["min_aqi"] <= aqi <= band["max_aqi"])
        return json.dumps({"policy_id": policy_input["policy_id"],
                           "limit_kmh": band["limit_kmh"], "reason_code": band["reason_code"]})


def _load_replay_scenarios(replay_path: Path) -> dict:
    replay = _json_object(Path(replay_path).read_text(encoding="utf-8-sig"))
    scenarios = replay.get("scenarios")
    if type(replay.get("schema_version")) is not int or replay["schema_version"] != 1 or not isinstance(scenarios, dict):
        raise ValueError("Unsupported synthetic air replay format")
    return scenarios


def make_air_advisor(
    *, mode: str, policy_path: Path, replay_path: Path, scenario: str = "good",
    replay_aqi: str | int | float | Decimal | None = None,
    latitude: float | None = None, longitude: float | None = None,
) -> AirQualityAdvisor:
    """Create an explicitly selected live advisor or a deterministic synthetic replay."""
    if mode == "live":
        if latitude is None or longitude is None:
            raise ValueError("Live air mode requires latitude and longitude.")
        return AirQualityAdvisor(policy_path, OpenMeteoAirDataClient(latitude, longitude), OllamaChatLLMClient())
    if mode != "replay":
        raise ValueError("Air-quality mode must be replay or live")
    scenarios = _load_replay_scenarios(replay_path)
    if scenario == "custom":
        aqi = parse_replay_aqi(replay_aqi)
        baseline = scenarios.get("good")
        if not isinstance(baseline, dict):
            raise ValueError("Custom replay requires the synthetic good-air sample for placeholder pollutants.")
        data = _ReplayAirDataClient({**baseline, AQI_FIELD: aqi})
        return AirQualityAdvisor(policy_path, data, _CustomReplayResponse())
    fixture = scenarios.get(scenario) if isinstance(scenario, str) else None
    if not isinstance(fixture, dict):
        raise ValueError("Unknown synthetic air replay scenario")
    return AirQualityAdvisor(policy_path, _ReplayAirDataClient(fixture), _ReplayLLMClient(fixture))
