"""Assemble the packaged components for the command line and GUI."""

from pathlib import Path

from .air_quality import make_air_advisor
from .decision_controller import DecisionController
from .weather_predictor import WeatherPredictor

CODE = Path(__file__).resolve().parents[1]
CONFIG = CODE / "config"


def create_controller(log_path=None):
    weather = WeatherPredictor(CONFIG / "runtime_model.json", CONFIG / "weather_support.json")

    def air_for_request(request):
        mode = request.get("mode", "replay")
        inputs = {"mode": mode, "policy_path": CONFIG / "air_policy.json", "replay_path": CONFIG / "air_replay.json"}
        if mode == "replay":
            return make_air_advisor(**inputs, scenario=request.get("scenario", "good"), replay_aqi=request.get("replay_aqi"))

        location = []
        for coordinate in ("latitude", "longitude"):
            try:
                location.append(float(request.get(coordinate)))
            except (TypeError, ValueError) as error:
                raise ValueError(f"Live mode requires a numeric {coordinate}.") from error
        return make_air_advisor(**inputs, latitude=location[0], longitude=location[1])

    return DecisionController(weather, air_for_request, log_path=log_path)
