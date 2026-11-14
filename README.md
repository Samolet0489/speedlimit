# SpeedLimit implementation

Version 0.4.2 runs the operator simulator with the fitted weather model, numeric AQI replay and local Ollama for Live mode. The commands and tests use Python 3.10 or later and the standard library. The GUI uses the Tk installation included with standard Windows Python.
  
## Run from the project root

```powershell
python Code/run.py gui
python Code/run.py decide --illuminance-lux 1 --water-um 0 --temperature-c 5 --scenario poor
python Code/run.py prepare
python Code/run.py route --illuminance-lux 0.499 --water-um 1000 --temperature-c 0
python Code/run.py train
python Code/run.py train --experiment alternate_clock
python Code/tools/run_tests.py
```

You can also run `python -m speedlimit ...` from inside Code. All default paths are resolved relative to the source files, so changing the shell's working directory does not change which resource/configuration files are selected.

The router command returns whether weather or darkness calls for the weather-model branch. The `decide` command combines a valid weather cap and a valid air cap using the lower value. Exit code 0 indicates a simulated recommendation, 2 indicates review needed, and 1 indicates a command or runtime error.

## Operator simulator

The ZIP includes the fitted primary model, its training support records, and the replay fixtures. From inside `Code`, use `python .\run.py gui`. Start with the replay mode, choose a scenario, and click **Evaluate**. The default weather inputs produce 80 km/h for `good` air and 50 km/h for `poor` air. The `stale` and `malformed` scenarios exercise review handling. For the weather branch, enter 0.1 lux, 200 micrometers of water and -4 Celsius. This example shows candidate details and review status.

Replay presets fill the European AQI field. Edit the value directly to switch to `custom` and test any AQI from 0 to 500. With clear weather, 20 gives 80 km/h and 20.01 rounds up to 21 and gives 70 km/h. In the CLI use `--scenario custom --replay-aqi 20.01`. Custom AQI uses placeholder pollutants and a simulated policy response. The `policy_mismatch` preset demonstrates rejection of a cap that conflicts with the policy.

The weather branch evaluates 70, 60, 50 and 40 km/h. A candidate needs observations at the same speed and weather branch within the joint training neighborhood and the observed feature ranges. Of the supported candidates with a predicted count at most one, it chooses the highest speed. A required branch that cannot recommend a cap leaves the overall decision for review. The packaged fit required review on all 70 weather-triggered holdout hours at this milestone's support settings.

The GUI shows both caps, reasons, input mode and data provenance. **Acknowledge result** records an operator action for the current simulated recommendation. The app appends decisions and acknowledgments to `Code/artifacts/decisions.jsonl`. Evaluation runs in the background. Editing an input clears the displayed result. The 60-second refresh is optional and resets on mode changes.

### Live air-quality mode

Select `live` and evaluate with the default Vienna coordinates (48.2082, 16.3738), or enter the road's coordinates. Light, road water and temperature still drive the weather branch. The coordinates select the current UTC hour's European AQI forecast from Open-Meteo. SpeedLimit sends the air data to `qwen3:4b` through the local Ollama service and validates the structured answer against `config/air_policy.json`.

For a new Windows machine, the project maintainer installs Ollama and runs `powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\setup_ollama.ps1` from the Code folder. This starts the local service when needed and downloads the project model if missing. The model occupies about 2.5 GB and is provisioned separately from the code ZIP. Open-Meteo still needs internet access. A missing local model, unavailable service or policy mismatch returns **Review needed**.

Provider references. [Open-Meteo Air Quality API](https://open-meteo.com/en/docs/air-quality-api) · [Ollama structured outputs](https://docs.ollama.com/capabilities/structured-outputs) · [Qwen3 4B](https://ollama.com/library/qwen3:4b)

## Data preparation

Original CSVs are read from the sibling `Resources` directory, outside Project. Keep the six supplied files together. They are never edited or included in the code archive.

For a different location:

```powershell
python Code/run.py prepare --resources "PATH/TO/Resources" --output "PATH/TO/prepared"
```

Default derived output is `Code/artifacts/prepared/`:

- `observations.csv`: every normalized source observation, including quarantined readings and their reason flags.
- `events.csv`: event fields and source references, with licence-plate values omitted. Unknown time/label semantics remain explicit.
- `feature_preview.csv`: separate N and N+1 hourly previews, quality flags, contributing sensor counts and source-row references. Missing values stay blank, distinct from zero.
- `preparation_summary.json`: source hashes, configuration, transformation policy, counts and training-readiness blockers.

The checked-in configuration is `config/preparation.json`. The chosen experimental cutoff is the start of the target hour, with at most one-hour-old measurements. Features use only readings already available at that cutoff. Means of available, confirmed same-type sensors are a documented aggregation choice. The first hour has no previous measurements and stays missing.

The following facts are intentionally unset: sensor-specific resolution of ambiguous type T, event-hour interpretation, near-accident thresholds and completeness of the event log. Sensor 19 remains quarantined because its definition is missing. Do not fill these settings by guessing from the numeric values.

Preparation succeeds even when the data is not ready for training: inspect `training_ready` and `blockers`. For automation that requires a fully configured dataset, add `--require-training-ready`; exit code 2 then means preparation completed but training readiness is false. Exit code 1 means an invalid request or preparation error.

Optional policy fields, after the assumptions have been documented:

| Configuration key | Meaning |
| --- | --- |
| `sensor_type_overrides` | Sensor-ID strings mapped to reviewed, exact metadata Type names. Orphan IDs cannot acquire a definition from this setting. |
| `excluded_sensor_ids` | Explicit list of defined sensor IDs to quarantine for an experiment, with the exclusion recorded in the summary. |
| `event_hour_convention` | `one_based_interval` maps source hours 1..24 to 0..23 on that day. `clock_hour_24_next_day` treats 24 as next-day midnight. These are alternative interpretations, not confirmed facts. |
| `event_label_policy` | `threshold` evaluates explicitly configured cutoffs. `recorded_indicator` counts the event category from the source record and leaves its numeric thresholds unset. |
| `near_accident_thresholds` | In threshold mode, supply `skid_angle_degrees`, `close_car_cm` and `close_guardrail_cm`. Qualifying values are absolute skid angle above the cutoff or distance below it. An event counts once. |
| `assume_event_log_complete` | Explicit Boolean coverage assumption. Without it, an absent event cannot establish a zero count. Missing indicators can still make labels unknown. |
| `prediction_cutoff` | `target_hour_start` selects causal as-of measurements. Library calls with no configuration leave the cutoff unresolved. |
| `max_feature_age_hours` | Positive finite maximum age for a contributing reading. |

The default preparation command reports readiness against the unresolved source facts. The named experiments below supply separate, explicit research assumptions. N+1 readings repeat N exactly and the two years have different event outcomes.

## Training experiments

`python Code/run.py train` prepares the `primary` configuration and trains a constant-mean baseline and a small neural network. It saves the prepared inputs and trained outputs under `Code/artifacts/experiments/primary/`. `train --experiment alternate_clock` repeats the experiment under the alternative hour interpretation. Each run has its own output directory. Use `--resources` for a different location or `--output` for a separate run directory.

The primary configuration provisionally treats sensors 15 and 18 as temperature. Their readings fit the temperature range, and each sensor has negative readings outside the declared traffic range. Sensor 16 is excluded because it fits both types. Sensor 19 has no metadata. The labels use the fields designated for near-accidents on case-study slide 7. The experiment assumes log completeness for zero counts in otherwise valid hours. Both configurations specify these choices in `config/experiments/`.

The primary configuration aligns event hour 1 with 00:00 and event hour 24 with 23:00 on the same day. The alternate interprets hour 1 as 01:00 and hour 24 as the following midnight. Both use the same features, split and model settings. Year N days 1–20 train the models, days 21–25 select the network checkpoint, and days 26–31 provide the holdout. N+1 supplies a separate repeated-input comparison.

`trained/model.json` stores the feature order, fitted scaling, network parameters and provenance. `trained/metrics.json` contains the baseline and network results. `trained/predictions.csv` allows the hourly comparisons to be inspected. The saved model estimates a count from historical relationships and the configured event definition. Recorded speeds can change within an hour, and candidate speeds require a later check of data support.

## Tests and evidence

`tools/run_tests.py` executes the actual unittest suite and writes `artifacts/test_results.json`, including individual outcomes, Python version and code/configuration hashes. To choose the evidence file:

```powershell
python Code/tools/run_tests.py --output "Submission/Evidence/step_04_test_results.json"
```

Tests cover the preparation, trainer, weather router and predictor, air-quality advisor, decision controller and GUI session. The milestone report separates controlled provider tests, replay checks and evaluation on the supplied weather data.

The original read-only audit is still available:

```powershell
python Code/tools/audit_data.py --output "docs/data_audit.json"
```

## Code archive

Replay runs directly from the extracted Code folder. Preparation, training and the supplied-data tests use the six course CSVs in the original Resources layout. The `prepare` and `train` commands also accept `--resources`. Generated CSVs, caches, credentials and third-party libraries are excluded from the archive. The milestone PDF and Evidence files describe the execution results.
