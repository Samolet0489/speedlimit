"""Command-line entry points for the SpeedLimit simulator and its components."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from decimal import Decimal
import json
from pathlib import Path
import sys

from . import __version__
from .air_quality import REPLAY_SCENARIOS

CODE = Path(__file__).resolve().parents[1]
PROJECT = CODE.parent
EXPERIMENTS = ("primary", "alternate_clock")


def _json_value(value):
    if isinstance(value, Decimal):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description="SpeedLimit local prototype tools")
    cli.add_argument("--version", action="version", version=__version__)
    commands = cli.add_subparsers(dest="command", required=True)

    preparation = commands.add_parser("prepare", help="Prepare the supplied CSVs without guessing unresolved facts")
    preparation.add_argument("--resources", type=Path, default=PROJECT.parent / "Resources")
    preparation.add_argument("--output", type=Path, default=CODE / "artifacts" / "prepared")
    preparation.add_argument("--config", type=Path, default=CODE / "config" / "preparation.json")
    preparation.add_argument("--require-training-ready", action="store_true",
                             help="Return exit code 2 when training is not ready; preserve the preparation report")

    training = commands.add_parser("train", help="Prepare and train a named course experiment")
    training.add_argument("--experiment", choices=EXPERIMENTS, default="primary")
    training.add_argument("--resources", type=Path, default=PROJECT.parent / "Resources")
    training.add_argument("--output", type=Path, help="Run directory for the prepared data and trained model")

    route = commands.add_parser("route", help="Evaluate the exact weather/light rules using decimal measurements")
    route.add_argument("--illuminance-lux", required=True, help="Illuminance in lux, using a dot decimal separator")
    route.add_argument("--water-um", required=True, help="Road-water depth in micrometers")
    route.add_argument("--temperature-c", required=True, help="Temperature in degrees Celsius")

    decide = commands.add_parser("decide", help="Evaluate both branches and record one simulated decision")
    decide.add_argument("--illuminance-lux", required=True)
    decide.add_argument("--water-um", required=True)
    decide.add_argument("--temperature-c", required=True)
    decide.add_argument("--mode", choices=("replay", "live"), default="replay")
    decide.add_argument("--scenario", choices=REPLAY_SCENARIOS, default="good")
    decide.add_argument("--replay-aqi", help="European AQI from 0 to 500 for --scenario custom")
    decide.add_argument("--latitude")
    decide.add_argument("--longitude")
    decide.add_argument("--log", type=Path, default=CODE / "artifacts" / "decisions.jsonl")
    gui = commands.add_parser("gui", help="Open the operator simulator")
    gui.add_argument("--log", type=Path, default=CODE / "artifacts" / "decisions.jsonl")
    return cli


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command in ("decide", "gui"):
            from .runtime import create_controller

            controller = create_controller(args.log)
            if args.command == "gui":
                from .operator_gui import run_gui

                run_gui(controller.evaluate, controller.acknowledge)
                return 0
            request = {key: getattr(args, key) for key in (
                "illuminance_lux", "water_um", "temperature_c", "mode", "scenario", "replay_aqi", "latitude", "longitude")}
            result = controller.evaluate(request)
            print(json.dumps(result, indent=2, allow_nan=False))
            return 0 if result["status"] == "recommended" else 2

        if args.command == "route":
            from .weather_router import evaluate_measurements
            decision = evaluate_measurements(args.illuminance_lux, args.water_um, args.temperature_c)
            result = asdict(decision)
            result["component"] = "C3 WeatherRouter"
            result["meaning"] = "Whether the weather/light branch requires a reduction; this is not the final speed limit."
            print(json.dumps(result, indent=2, default=_json_value, allow_nan=False))
            return 0

        from .data_preparation import prepare_data
        if args.command == "train":
            from .model_trainer import train_models

            resources = args.resources.resolve()
            output = (args.output or CODE / "artifacts" / "experiments" / args.experiment).resolve()
            prepared, trained = (output / "prepared").resolve(), (output / "trained").resolve()
            if any(path == resources or resources in path.parents for path in (output, prepared, trained)):
                raise ValueError("The experiment output directories must be outside the source resource directory.")
            config_path = CODE / "config" / "experiments" / f"{args.experiment}.json"
            with config_path.open(encoding="utf-8-sig") as stream:
                config = json.load(stream)
            preparation = prepare_data(resources, prepared, config)
            if not preparation["training_ready"]:
                print(json.dumps({"status": "preparation_blocked", "experiment": args.experiment,
                                  "blockers": preparation["blockers"]}, indent=2), file=sys.stderr)
                return 2
            result = train_models(prepared, trained)
            print(json.dumps({"experiment": args.experiment, **result}, indent=2, allow_nan=False))
            return 0

        resources, output = args.resources.resolve(), args.output.resolve()
        if output == resources or resources in output.parents:
            raise ValueError("The output directory must be outside the source resource directory.")
        with args.config.open(encoding="utf-8-sig") as stream:
            config = json.load(stream)
        if not isinstance(config, dict):
            raise ValueError("The preparation configuration must be a JSON object.")
        summary = prepare_data(resources, output, config)
        print(json.dumps(summary, indent=2, default=_json_value, allow_nan=False))
        return 2 if args.require_training_ready and not summary["training_ready"] else 0
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, allow_nan=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
