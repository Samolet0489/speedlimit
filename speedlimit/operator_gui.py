"""Tkinter operator simulator with an injectable decision and acknowledgment API."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from queue import Empty, SimpleQueue
from threading import Thread

from .air_quality import REPLAY_SCENARIOS, parse_replay_aqi


_DEFAULT_INPUTS = {
    "illuminance_lux": "1000", "water_um": "0", "temperature_c": "10",
    "mode": "replay", "scenario": "good", "replay_aqi": "",
    "latitude": "", "longitude": "",
}
_FIELD_LABELS = {
    "illuminance_lux": "Illuminance (lux)", "water_um": "Road water (µm)",
    "temperature_c": "Temperature (°C)", "latitude": "Latitude (°)", "longitude": "Longitude (°)",
    "replay_aqi": "European AQI (0–500)",
}
_MODES = {"replay": "Replay · simulated air data", "live": "Live · Open-Meteo and local Ollama"}
_REFRESH_MS = 60_000


class _InputError(ValueError):
    def __init__(self, field: str, message: str):
        super().__init__(message)
        self.field = field


def _number(payload: dict, field: str) -> Decimal:
    value, label = payload[field], _FIELD_LABELS[field]
    if not value:
        raise _InputError(field, f"Enter a value for {label}.")
    try:
        number = Decimal(value)
    except (InvalidOperation, ValueError) as exc:
        raise _InputError(field, f"Enter a number for {label}, using a decimal point.") from exc
    if not number.is_finite():
        raise _InputError(field, f"Enter a finite number for {label}.")
    return number


def _validated_inputs(raw: dict) -> dict:
    payload = {key: str(raw.get(key, "")).strip() for key in _DEFAULT_INPUTS}
    if payload["mode"] not in _MODES:
        raise _InputError("mode", "Choose Replay or Live mode.")
    for field in ("illuminance_lux", "water_um", "temperature_c"):
        value = _number(payload, field)
        if field != "temperature_c" and value < 0:
            raise _InputError(field, f"{_FIELD_LABELS[field]} must be zero or greater.")
    if payload["mode"] == "live":
        for field, bound in (("latitude", 90), ("longitude", 180)):
            if not -bound <= _number(payload, field) <= bound:
                raise _InputError(field, f"{_FIELD_LABELS[field]} must be between {-bound} and {bound}.")
    else:
        if payload["scenario"] not in REPLAY_SCENARIOS:
            raise _InputError("scenario", "Choose a replay scenario.")
        if payload["scenario"] == "custom":
            try:
                payload["replay_aqi"] = str(parse_replay_aqi(payload["replay_aqi"]))
            except ValueError as exc:
                raise _InputError("replay_aqi", str(exc)) from exc
        for field in ("latitude", "longitude"):
            payload[field] = ""
    return payload


def _valid_result(value: object, mode: str) -> bool:
    if not isinstance(value, dict) or value.get("mode") != mode:
        return False
    reasons = value.get("reasons")
    if not isinstance(reasons, list) or not all(isinstance(reason, str) for reason in reasons):
        return False
    if not isinstance(value.get("evaluated_at"), str) or not value["evaluated_at"].strip():
        return False
    if not all(isinstance(value.get(branch), dict) for branch in ("weather", "air")):
        return False
    status, limit = value.get("status"), value.get("limit_kmh")
    return (status == "recommended" and type(limit) is int and limit > 0) or (status == "review_needed" and limit is None)


@dataclass(frozen=True)
class _Job:
    generation: int
    kind: str
    payload: dict


@dataclass(frozen=True)
class _Completion:
    job: _Job
    succeeded: bool
    value: object = None


class _OperatorSession:
    """The UI thread owns state. Workers exchange only snapshots and results."""

    def __init__(self, evaluate: Callable[[dict], dict], acknowledge: Callable[[dict], None] | None = None):
        self._evaluate = evaluate
        self._acknowledge = acknowledge
        self._completed: SimpleQueue[_Completion] = SimpleQueue()
        self._working = False
        self._pending: _Job | None = None
        self._generation = 0
        self._closed = False
        self.status = "ready"
        self.message = "Evaluate the default replay scenario or enter your own measurements."
        self.result: dict | None = None
        self.invalid_field: str | None = None

    @property
    def can_acknowledge(self) -> bool:
        return not self._closed and self.status == "recommended" and self.result is not None

    def edit(self) -> None:
        self._generation += 1
        self._pending = None
        self.status = "ready"
        self.message = "Inputs changed. Evaluate to get a current recommendation."
        self.result = None
        self.invalid_field = None

    def submit(self, raw: dict) -> None:
        if self._closed:
            return
        self.edit()
        try:
            payload = _validated_inputs(raw)
        except _InputError as exc:
            self.status, self.message, self.invalid_field = "invalid", str(exc), exc.field
            return
        self.status = "evaluating"
        self.message = "You can continue editing while the request runs."
        job = _Job(self._generation, "evaluate", payload)
        if self._working:
            self._pending = job
        else:
            self._start(job)

    def acknowledge(self) -> None:
        if not self.can_acknowledge or self.result is None:
            return
        if self._acknowledge is None:
            self.status, self.message = "acknowledged", "Acknowledged in this session."
            return
        self.status, self.message = "acknowledging", "Recording the acknowledgment."
        self._start(_Job(self._generation, "acknowledge", deepcopy(self.result)))

    def _start(self, job: _Job) -> None:
        self._working = True
        Thread(target=self._work, args=(job,), daemon=True).start()

    def _work(self, job: _Job) -> None:
        try:
            if job.kind == "evaluate":
                # The callback receives its own dict, so it cannot modify the
                # request mode used to check the reply on the UI thread.
                result = deepcopy(self._evaluate(dict(job.payload)))
            else:
                if self._acknowledge is not None:
                    self._acknowledge(job.payload)
                result = None
        except Exception:
            # Provider exceptions can contain request details or credentials.
            self._completed.put(_Completion(job, False))
        else:
            self._completed.put(_Completion(job, True, result))

    def poll(self) -> bool:
        """Apply a completed job on the UI thread and start any current queued job."""
        if self._closed:
            return False
        try:
            completion = self._completed.get_nowait()
        except Empty:
            return False
        self._working = False
        job = completion.job
        changed = job.generation == self._generation
        if changed:
            self._apply(completion)
        if self._pending is not None:
            queued, self._pending = self._pending, None
            if queued.generation == self._generation:
                self._start(queued)
        return changed

    def _apply(self, completion: _Completion) -> None:
        if completion.job.kind == "acknowledge":
            if completion.succeeded:
                self.status, self.message = "acknowledged", "Acknowledgment recorded for this simulated decision."
            else:
                self.status, self.message = "recommended", "The acknowledgment could not be recorded. Try again."
            return
        if not completion.succeeded:
            self.status, self.message = "review_needed", "Evaluation could not complete. Check the inputs and service configuration, then try again."
            return
        if not _valid_result(completion.value, completion.job.payload["mode"]):
            self.status, self.message = "review_needed", "The decision response was invalid. Evaluate again or check the service configuration."
            return
        self.result = completion.value
        self.status = self.result["status"]
        if self.status == "recommended":
            self.message = "Review the reasons and acknowledge the simulated recommendation."
        else:
            self.message = "Review the reasons before evaluating again."

    def close(self) -> None:
        self._closed = True
        self.edit()


def _display(value: object) -> str:
    return "Unavailable" if value is None or value == "" else str(value)


def _result_details(result: dict | None) -> str:
    if result is None:
        return "Decision reasons, source times, and candidate checks will appear here."
    air, weather = result["air"], result["weather"]
    lines = ["Reasons"]
    lines.extend(f"  • {reason}" for reason in result["reasons"])
    lines.extend(("", "Decision sources", f"  Mode    {_MODES[result['mode']]}",
                  f"  Evaluated    {result['evaluated_at']}"))
    if result.get("decision_id"):
        lines.append(f"  Decision ID    {result['decision_id']}")
    lines.extend(("", "Weather", f"  Status    {_display(weather.get('status')).replace('_', ' ')}",
                  f"  Proposed cap    {_display(weather.get('limit_kmh'))}"
                  + (" km/h" if weather.get("limit_kmh") is not None else "")))
    if weather.get("reason") and weather["reason"] not in result["reasons"]:
        lines.append(f"  {weather['reason']}")
    if weather.get("model_id"):
        lines.append(f"  Model    {weather['model_id']}")
    candidates = weather.get("candidates")
    for candidate in candidates if isinstance(candidates, (list, tuple)) else ():
        if not isinstance(candidate, dict):
            continue
        speed, status = _display(candidate.get("speed_kmh")), _display(candidate.get("status")).replace("_", " ")
        detail = f"  {speed} km/h    {status}"
        if candidate.get("predicted_count") is not None:
            prediction = candidate["predicted_count"]
            detail += f" · {prediction:.3f} predicted events/hour" if type(prediction) in (int, float) else f" · {prediction} predicted events/hour"
        if candidate.get("neighbors") is not None:
            detail += f" · {candidate['neighbors']} supporting observations"
        if candidate.get("reason"):
            detail += f" · {candidate['reason']}"
        lines.append(detail)
    lines.extend(("", "Air quality", f"  Status    {_display(air.get('status')).replace('_', ' ')}",
                  f"  Proposed cap    {_display(air.get('limit_kmh'))}" + (" km/h" if air.get("limit_kmh") is not None else ""),
                  f"  Data source    {_display(air.get('data_source'))}",
                  f"  Response source    {_display(air.get('response_source'))}",
                  f"  Source time    {_display(air.get('data_time'))}",
                  f"  Policy    {_display(air.get('policy_id'))}"))
    if air.get("european_aqi") is not None:
        lines.append(f"  European AQI    {air['european_aqi']} (rounded up to {air['rounded_european_aqi']})")
    if air.get("reason") and air["reason"] not in result["reasons"]:
        lines.append(f"  {air['reason']}")
    return "\n".join(lines)


def run_gui(evaluate: Callable[[dict], dict], acknowledge: Callable[[dict], None] | None = None) -> None:
    """Open the simulator. Callback work runs off Tk's main thread."""
    import tkinter as tk
    from tkinter import ttk

    session = _OperatorSession(evaluate, acknowledge)
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        raise RuntimeError("The operator window needs a graphical desktop with Tk available.") from exc
    root.title("SpeedLimit operator simulator")
    root.geometry("960x720")
    root.minsize(730, 535)
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    page = ttk.Frame(root, padding=20)
    page.grid(sticky="nsew")
    page.columnconfigure(0, weight=1)
    page.rowconfigure(5, weight=1)

    ttk.Label(page, text="SpeedLimit operator simulator", font=("Segoe UI", 22, "bold")).grid(row=0, sticky="w")
    ttk.Label(page, text="Enter road conditions and review a simulated speed recommendation.").grid(row=1, sticky="w", pady=(2, 12))

    inputs = {key: tk.StringVar(root, value=value) for key, value in _DEFAULT_INPUTS.items()}
    fields = {}
    weather = ttk.LabelFrame(page, text="Weather inputs", padding=(12, 8))
    weather.grid(row=2, sticky="ew", pady=(0, 8))
    for column, key in enumerate(("illuminance_lux", "water_um", "temperature_c")):
        weather.columnconfigure(column, weight=1, uniform="weather")
        ttk.Label(weather, text=_FIELD_LABELS[key]).grid(row=0, column=column, sticky="w", padx=(0, 8))
        fields[key] = ttk.Entry(weather, textvariable=inputs[key], width=15)
        fields[key].grid(row=1, column=column, sticky="ew", padx=(0, 8), pady=(3, 2))

    source = ttk.LabelFrame(page, text="Air-quality source", padding=(12, 8))
    source.grid(row=3, sticky="ew", pady=(0, 8))
    source.columnconfigure(0, weight=1)
    mode_row = ttk.Frame(source)
    mode_row.grid(row=0, sticky="w", pady=(0, 6))
    ttk.Label(mode_row, text="Mode").pack(side="left", padx=(0, 8))
    fields["mode"] = ttk.Combobox(mode_row, textvariable=inputs["mode"], values=tuple(_MODES),
                                 state="readonly", width=13)
    fields["mode"].pack(side="left")

    replay_fields = ttk.Frame(source)
    live_fields = ttk.Frame(source)
    for group, keys in ((replay_fields, ("scenario", "replay_aqi")),
                        (live_fields, ("latitude", "longitude"))):
        group.grid(row=1, column=0, sticky="ew")
        for column, key in enumerate(keys):
            group.columnconfigure(column, weight=1, uniform="air")
            ttk.Label(group, text=_FIELD_LABELS.get(key, "Replay scenario")).grid(
                row=0, column=column, sticky="w", padx=(0, 8))
            if key == "scenario":
                fields[key] = ttk.Combobox(group, textvariable=inputs[key], values=REPLAY_SCENARIOS,
                                           state="readonly", width=15)
            else:
                fields[key] = ttk.Entry(group, textvariable=inputs[key], width=15)
            fields[key].grid(row=1, column=column, sticky="ew", padx=(0, 8), pady=(3, 2))
    source_note = ttk.Label(source, wraplength=640)
    source_note.grid(row=2, column=0, sticky="w", pady=(3, 0))

    actions = ttk.Frame(page)
    actions.grid(row=4, sticky="ew", pady=(2, 10))
    refresh = tk.BooleanVar(root, value=False)
    refresh_timer = None
    poll_timer = None

    output = ttk.LabelFrame(page, text="Current decision", padding=(12, 8))
    output.grid(row=5, sticky="nsew")
    output.columnconfigure(0, weight=1)
    output.rowconfigure(2, weight=1)
    heading = ttk.Label(output, font=("Segoe UI", 19, "bold"))
    heading.grid(row=0, column=0, columnspan=2, sticky="w")
    message = ttk.Label(output, wraplength=650)
    message.grid(row=1, column=0, columnspan=2, sticky="w", pady=(1, 7))
    details = tk.Text(output, wrap="word", height=8, width=1, borderwidth=0,
                      font=("Segoe UI", 11), padx=6, pady=5, takefocus=True)
    details.grid(row=2, column=0, sticky="nsew")
    scrollbar = ttk.Scrollbar(output, command=details.yview)
    scrollbar.grid(row=2, column=1, sticky="ns")
    details.configure(yscrollcommand=scrollbar.set)

    def render() -> None:
        state, result = session.status, session.result
        if result and result["status"] == "recommended":
            title = f"{result['limit_kmh']} km/h · simulated recommendation"
        else:
            title = {"evaluating": "Evaluating…", "invalid": "Check the input", "review_needed": "Review needed"}.get(state, "Ready to evaluate")
        heading.configure(text=title)
        message.configure(text=session.message)
        details.configure(state="normal")
        details.delete("1.0", "end")
        details.insert("1.0", _result_details(result))
        details.configure(state="disabled")
        evaluate_button.configure(state="disabled" if state in ("evaluating", "acknowledging") else "normal")
        acknowledge_button.configure(state="normal" if session.can_acknowledge else "disabled")

    def start_evaluation() -> None:
        session.submit({key: variable.get() for key, variable in inputs.items()})
        render()
        if session.invalid_field in fields:
            fields[session.invalid_field].focus_set()

    def record_acknowledgment() -> None:
        session.acknowledge()
        render()

    def schedule_refresh() -> None:
        nonlocal refresh_timer
        if refresh_timer is not None:
            root.after_cancel(refresh_timer)
        refresh_timer = root.after(_REFRESH_MS, refresh_tick) if refresh.get() else None

    def refresh_tick() -> None:
        nonlocal refresh_timer
        refresh_timer = None
        if session.status not in ("evaluating", "acknowledging"):
            start_evaluation()
        schedule_refresh()

    def on_edit(key: str) -> None:
        if key == "mode":
            refresh.set(False)
            schedule_refresh()
        update_source_note()
        session.edit()
        render()

    def update_source_note() -> None:
        if inputs["mode"].get() == "live":
            replay_fields.grid_remove()
            live_fields.grid()
            source_note.configure(text="Enter the road coordinates. Live uses Open-Meteo and the project's local Ollama model.")
        else:
            live_fields.grid_remove()
            replay_fields.grid()
            custom = inputs["scenario"].get() == "custom"
            fields["replay_aqi"].configure(state="normal" if custom else "disabled")
            guidance = "Enter an AQI to test the speed policy." if custom else "Choose custom to enter a specific AQI."
            source_note.configure(text=f"Replay uses simulated data and responses. {guidance}")

    def poll() -> None:
        nonlocal poll_timer
        if session.poll():
            render()
        poll_timer = root.after(60, poll)

    def close() -> None:
        session.close()
        if poll_timer is not None:
            root.after_cancel(poll_timer)
        if refresh_timer is not None:
            root.after_cancel(refresh_timer)
        root.destroy()

    evaluate_button = ttk.Button(actions, text="Evaluate", command=start_evaluation)
    evaluate_button.pack(side="left")
    acknowledge_button = ttk.Button(actions, text="Acknowledge result", command=record_acknowledgment)
    acknowledge_button.pack(side="left", padx=8)
    ttk.Checkbutton(actions, text="Re-evaluate every 60 seconds", variable=refresh, command=schedule_refresh).pack(side="right")

    for key, variable in inputs.items():
        variable.trace_add("write", lambda *_args, key=key: on_edit(key))
    root.protocol("WM_DELETE_WINDOW", close)
    fields["illuminance_lux"].focus_set()
    update_source_note()
    render()
    poll()
    root.mainloop()
