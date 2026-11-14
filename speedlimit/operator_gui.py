"""Tkinter operator simulator with an injectable decision and acknowledgment API."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from queue import Empty, SimpleQueue
from threading import Thread

from .air_quality import AQI_FIELD, REPLAY_SCENARIOS, _load_replay_scenarios, parse_replay_aqi


_DEFAULT_INPUTS = {
    "illuminance_lux": "1000", "water_um": "0", "temperature_c": "10",
    "mode": "replay", "scenario": "good", "replay_aqi": "",
    "latitude": "48.2082", "longitude": "16.3738",
}
_FIELD_LABELS = {
    "illuminance_lux": "Illuminance (lux)", "water_um": "Road water (µm)",
    "temperature_c": "Temperature (°C)", "latitude": "Latitude (°)", "longitude": "Longitude (°)",
    "replay_aqi": "European AQI (0–500)",
}
_MODES = {"replay": "Replay · simulated air data", "live": "Live · Open-Meteo and local Ollama"}
_REFRESH_MS = 60_000
_CANVAS = "#EDF3F1"
_SURFACE = "#FFFFFF"
_INK = "#182824"
_MUTED = "#50625C"
_BORDER = "#CCDAD4"
_FOREST = "#145947"
_WASH = "#E7F2ED"
_HERO_TEXT = "#EDF8EF"
_HERO_PALETTES = {
    "neutral": ("#263C37", "#D7E4DD", "#4E655C", "#97AAA0"),
    "recommended": ("#123B34", "#C3DCD1", "#40665A", "#DC6353"),
    "review": ("#492F25", "#F1D6C7", "#765142", "#E8AB73"),
}


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
        payload.update(scenario="", replay_aqi="")
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
            self.message = "Awaiting acknowledgment"
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


def _style_operator(root, ttk) -> None:
    style = ttk.Style(root)
    style.theme_use("clam")
    root.configure(background=_CANVAS)
    style.configure(".", background=_CANVAS, foreground=_INK, font=("Segoe UI", 10), focuscolor=_FOREST)
    style.configure("Surface.TFrame", background=_SURFACE)
    style.configure("Card.TFrame", background=_SURFACE, bordercolor=_BORDER, borderwidth=1,
                    lightcolor=_BORDER, darkcolor=_BORDER, relief="solid")
    style.configure("Surface.TLabel", background=_SURFACE, foreground=_INK)
    style.configure("Muted.Surface.TLabel", foreground=_MUTED, font=("Segoe UI", 9))
    style.configure("Section.Surface.TLabel", font=("Segoe UI", 11, "bold"))
    style.configure("Title.Surface.TLabel", font=("Segoe UI", 15, "bold"))
    style.configure("Brand.TLabel", background=_FOREST, foreground=_SURFACE, padding=(10, 3),
                    font=("Segoe UI", 17, "bold"))
    style.configure("Desk.TSeparator", background=_BORDER)

    for widget in ("TEntry", "TCombobox"):
        style.configure(widget, fieldbackground=_SURFACE, foreground=_INK, background=_WASH,
                        bordercolor=_BORDER, lightcolor=_SURFACE, darkcolor=_SURFACE,
                        insertcolor=_INK, selectbackground=_FOREST, selectforeground=_SURFACE, padding=(8, 5))
        style.map(widget, bordercolor=[("invalid", "#A34128"), ("focus", _FOREST)])
    style.configure("TCombobox", arrowcolor=_FOREST, arrowsize=12)
    style.map("TCombobox", fieldbackground=[("readonly", _SURFACE)], foreground=[("readonly", _INK)])
    for option, color in (("background", _SURFACE), ("foreground", _INK),
                          ("selectBackground", _FOREST), ("selectForeground", _SURFACE)):
        root.option_add(f"*TCombobox*Listbox.{option}", color)

    style.configure("TButton", background=_WASH, foreground=_FOREST, bordercolor=_BORDER,
                    lightcolor=_WASH, darkcolor=_WASH, borderwidth=1, relief="flat",
                    padding=(12, 7), font=("Segoe UI", 10, "bold"))
    style.map("TButton", background=[("disabled", _CANVAS), ("pressed", "#C6E0D4"), ("active", "#D8EAE1")],
              foreground=[("disabled", _MUTED)])
    style.configure("Primary.TButton", background=_FOREST, foreground=_SURFACE, bordercolor=_FOREST,
                    lightcolor=_FOREST, darkcolor=_FOREST)
    style.map("Primary.TButton", background=[("disabled", _WASH), ("pressed", "#103F34"), ("active", "#1C6A54")],
              foreground=[("disabled", _MUTED)], focuscolor=[("focus", _SURFACE)])
    style.layout("Mode.TRadiobutton", style.layout("TButton"))
    style.configure("Mode.TRadiobutton", background=_CANVAS, foreground=_MUTED, bordercolor=_BORDER,
                    lightcolor=_CANVAS, darkcolor=_CANVAS, padding=(8, 6), font=("Segoe UI", 10, "bold"))
    style.map("Mode.TRadiobutton", background=[("selected", _WASH), ("active", _WASH)],
              foreground=[("selected", _FOREST)], bordercolor=[("selected", _FOREST), ("focus", _FOREST)])
    style.configure("Refresh.TCheckbutton", background=_SURFACE, foreground=_MUTED, indicatorforeground=_FOREST,
                    upperbordercolor=_BORDER, lowerbordercolor=_BORDER)
    style.map("Refresh.TCheckbutton", background=[("active", _SURFACE)],
              indicatorbackground=[("selected", _WASH), ("!selected", _SURFACE)])
    style.configure("Vertical.TScrollbar", background=_WASH, troughcolor=_CANVAS, bordercolor=_BORDER,
                    lightcolor=_WASH, darkcolor=_WASH, arrowcolor=_FOREST, arrowsize=12)
    style.map("Vertical.TScrollbar", background=[("active", "#D8EAE1")])


def run_gui(evaluate: Callable[[dict], dict], acknowledge: Callable[[dict], None] | None = None) -> None:
    """Open the simulator. Callback work runs off Tk's main thread."""
    import tkinter as tk
    from tkinter import ttk

    from .runtime import CONFIG

    try:
        presets = _load_replay_scenarios(CONFIG / "air_replay.json")
    except (OSError, ValueError):
        # A broken replay fixture can still be reported during evaluation.
        presets = {}
    preset_aqis = {name: str(fixture[AQI_FIELD]) for name, fixture in presets.items()
                   if isinstance(fixture, dict) and fixture.get(AQI_FIELD) is not None}

    session = _OperatorSession(evaluate, acknowledge)
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        raise RuntimeError("The operator window needs a graphical desktop with Tk available.") from exc
    _style_operator(root, ttk)
    root.title("SpeedLimit operator simulator")
    root.geometry("1000x640")
    root.minsize(840, 530)
    root.columnconfigure(0, weight=1)
    root.rowconfigure(1, weight=1)
    masthead = ttk.Frame(root, padding=(22, 12), style="Surface.TFrame")
    masthead.grid(row=0, column=0, sticky="ew")
    masthead.columnconfigure(1, weight=1)
    ttk.Label(masthead, text="↗", style="Brand.TLabel").grid(row=0, column=0, rowspan=2, padx=(0, 12))
    ttk.Label(masthead, text="SpeedLimit", style="Title.Surface.TLabel").grid(row=0, column=1, sticky="sw")
    ttk.Label(masthead, text="Operator simulator", style="Muted.Surface.TLabel").grid(row=1, column=1, sticky="nw")
    source_badge = ttk.Label(masthead, style="Muted.Surface.TLabel")
    source_badge.grid(row=0, column=2, rowspan=2, sticky="e")
    page = ttk.Frame(root, padding=20)
    page.grid(row=1, column=0, sticky="nsew")
    page.columnconfigure(0, weight=1, minsize=365)
    page.columnconfigure(1, weight=2, minsize=420)
    page.rowconfigure(0, weight=1)

    input_panel = ttk.Frame(page, padding=18, style="Card.TFrame")
    input_panel.grid(row=0, column=0, sticky="new", padx=(0, 14))
    input_panel.columnconfigure(0, weight=1)
    ttk.Label(input_panel, text="Evaluation inputs", style="Title.Surface.TLabel").grid(
        row=0, sticky="w", pady=(0, 16))

    inputs = {key: tk.StringVar(root, value=value) for key, value in _DEFAULT_INPUTS.items()}
    inputs["replay_aqi"].set(preset_aqis.get(inputs["scenario"].get(), ""))
    syncing_replay = False
    fields = {}
    ttk.Label(input_panel, text="Air-quality source", style="Section.Surface.TLabel").grid(
        row=1, sticky="w", pady=(0, 6))
    mode_row = ttk.Frame(input_panel, style="Surface.TFrame")
    mode_row.grid(row=2, sticky="ew", pady=(0, 10))
    for column, (mode, label) in enumerate((("replay", "Replay"), ("live", "Live"))):
        mode_row.columnconfigure(column, weight=1, uniform="mode")
        ttk.Radiobutton(mode_row, text=label, value=mode, variable=inputs["mode"],
                        style="Mode.TRadiobutton").grid(row=0, column=column, sticky="ew")

    replay_fields = ttk.Frame(input_panel, style="Surface.TFrame")
    live_fields = ttk.Frame(input_panel, style="Surface.TFrame")
    weather = ttk.Frame(input_panel, style="Surface.TFrame")
    for group, keys in ((weather, ("illuminance_lux", "water_um", "temperature_c")),
                        (replay_fields, ("scenario", "replay_aqi")),
                        (live_fields, ("latitude", "longitude"))):
        for column, key in enumerate(keys):
            group.columnconfigure(column, weight=1)
            gap = (0, 8) if column < len(keys) - 1 else 0
            ttk.Label(group, text=_FIELD_LABELS.get(key, "Scenario"), style="Muted.Surface.TLabel").grid(
                row=0, column=column, sticky="w", padx=gap)
            if key == "scenario":
                fields[key] = ttk.Combobox(group, textvariable=inputs[key], values=REPLAY_SCENARIOS,
                                           state="readonly", width=14)
            else:
                fields[key] = ttk.Entry(group, textvariable=inputs[key], width=9)
            fields[key].grid(row=1, column=column, sticky="ew", padx=gap, pady=(3, 0))
    replay_fields.grid(row=3, sticky="ew")
    live_fields.grid(row=3, sticky="ew")
    source_note = ttk.Label(input_panel, wraplength=310, style="Muted.Surface.TLabel")
    source_note.grid(row=4, sticky="w", pady=(8, 16))
    ttk.Label(input_panel, text="Road conditions", style="Section.Surface.TLabel").grid(
        row=5, sticky="w", pady=(0, 6))
    weather.grid(row=6, sticky="ew")
    ttk.Separator(input_panel, style="Desk.TSeparator").grid(row=7, sticky="ew", pady=14)
    refresh = tk.BooleanVar(root, value=False)
    refresh_timer = None
    poll_timer = None

    output = ttk.Frame(page, style="Card.TFrame")
    output.grid(row=0, column=1, sticky="new")
    output.columnconfigure(0, weight=1)
    output.rowconfigure(4, weight=1)
    hero = tk.Frame(output, padx=18, pady=16)
    hero.grid(row=0, sticky="ew")
    hero.columnconfigure(1, weight=1)
    hero_labels = []

    def hero_label(parent, *, text="", size=10, bold=False, secondary=False):
        label = tk.Label(parent, text=text, font=("Segoe UI", size, "bold" if bold else "normal"),
                         justify="left", anchor="w", padx=0, pady=0)
        hero_labels.append((label, secondary))
        return label

    hero_label(hero, text="Current decision", size=12, bold=True).grid(row=0, column=0, sticky="w")
    status_label = hero_label(hero, size=9, bold=True, secondary=True)
    status_label.grid(row=0, column=1, sticky="e", padx=(8, 0))
    sign = tk.Canvas(hero, width=106, height=106, bd=0, highlightthickness=0, name="speed_sign")
    sign.grid(row=1, column=0, sticky="w", pady=16)
    ring = sign.create_oval(4, 4, 102, 102, width=5, fill="#F9FAF5")
    speed_text = sign.create_text(53, 44, font=("Segoe UI", -40, "bold"), fill=_INK, tags="speed")
    speed_unit = sign.create_text(53, 73, text="km/h", font=("Segoe UI", 9, "bold"), fill=_INK)
    explanation = tk.Frame(hero)
    explanation.grid(row=1, column=1, sticky="ew", padx=(14, 0), pady=16)
    heading = hero_label(explanation, size=17, bold=True)
    heading.pack(fill="x")
    message = hero_label(explanation, secondary=True)
    message.pack(fill="x", pady=(5, 0))
    divider = tk.Frame(hero, height=1)
    divider.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 9))
    caps = tk.Frame(hero)
    caps.grid(row=3, column=0, columnspan=2, sticky="ew")
    cap_values = {}
    for column, (branch, label) in enumerate((("weather", "Weather cap"), ("air", "Air-quality cap"))):
        caps.columnconfigure(column, weight=1, uniform="cap")
        hero_label(caps, text=label, size=9, secondary=True).grid(row=0, column=column, sticky="w")
        cap_values[branch] = hero_label(caps, size=14, bold=True)
        cap_values[branch].grid(row=1, column=column, sticky="w")

    def wrap_explanation(event):
        for label in (heading, message):
            label.configure(wraplength=max(1, event.width))

    explanation.bind("<Configure>", wrap_explanation)
    for label in (heading, message):
        label.configure(wraplength=205)
    result_actions = ttk.Frame(output, padding=(16, 10), style="Surface.TFrame")
    result_actions.grid(row=1, sticky="ew")
    result_actions.columnconfigure(0, weight=1)
    acknowledgment_status = ttk.Label(result_actions, style="Muted.Surface.TLabel", wraplength=215)
    acknowledgment_status.grid(row=0, column=0, sticky="w", padx=(0, 8))
    ttk.Separator(output, style="Desk.TSeparator").grid(row=2, sticky="ew")
    details_area = ttk.Frame(output, padding=(16, 0, 16, 12), style="Surface.TFrame")
    details_area.columnconfigure(0, weight=1)
    details_area.rowconfigure(0, weight=1)
    details = tk.Text(details_area, wrap="word", height=1, width=1, borderwidth=0,
                      background=_CANVAS, foreground=_INK, selectbackground=_FOREST, selectforeground=_SURFACE,
                      highlightthickness=1, highlightbackground=_BORDER, highlightcolor=_FOREST,
                      font=("Segoe UI", 10), padx=10, pady=7, takefocus=True)
    details.grid(row=0, column=0, sticky="nsew")
    scrollbar = ttk.Scrollbar(details_area, command=details.yview)
    scrollbar.grid(row=0, column=1, sticky="ns")
    details.configure(yscrollcommand=scrollbar.set)

    def render() -> None:
        state, result = session.status, session.result
        speed = result["limit_kmh"] if result else None
        weather = result["weather"] if result else {}
        air = result["air"] if result else {}
        title = {"evaluating": "Evaluating…", "invalid": "Check the input",
                 "review_needed": "Review needed"}.get(state, "Ready to evaluate")
        reason = session.message
        status_text = "SIMULATED RECOMMENDATION" if speed is not None else "NO RECOMMENDATION"
        acknowledgment_text = "Decision pending review" if state == "review_needed" else ""
        if speed is not None:
            weather_limits = weather.get("limit_kmh") == speed
            air_limits = air.get("limit_kmh") == speed
            if weather_limits and air_limits:
                title, limiting_reason = "Both caps agree", air.get("reason")
            elif weather_limits:
                title, limiting_reason = "Weather sets the limit", weather.get("reason")
            elif air_limits:
                title, limiting_reason = "Air quality sets the limit", air.get("reason")
            else:
                title, limiting_reason = "Speed recommendation", None
            reason = limiting_reason or next(iter(result["reasons"]), "")
            acknowledgment_text = session.message
        elif result:
            review_reasons = [branch["reason"] for branch in (weather, air)
                              if branch.get("status") == "review_needed" and branch.get("reason")]
            if review_reasons:
                reason = " ".join(dict.fromkeys(review_reasons))
            elif result["reasons"]:
                reason = result["reasons"][0]
        palette = "recommended" if speed is not None else "neutral"
        if state in ("review_needed", "invalid"):
            palette = "review"
        background, secondary, line, ring_color = _HERO_PALETTES[palette]
        for frame in (hero, caps, explanation, sign):
            frame.configure(background=background)
        divider.configure(background=line)
        for label, is_secondary in hero_labels:
            label.configure(background=background, foreground=secondary if is_secondary else _HERO_TEXT)
        sign.itemconfigure(ring, outline=ring_color)
        sign.itemconfigure(speed_text, text=str(speed) if speed is not None else "—")
        sign.itemconfigure(speed_unit, state="normal" if speed is not None else "hidden")
        status_label.configure(text=status_text)
        heading.configure(text=title)
        message.configure(text=reason)
        acknowledgment_status.configure(text=acknowledgment_text)
        for branch, data in (("weather", weather), ("air", air)):
            cap = data.get("limit_kmh")
            cap_values[branch].configure(text=f"{cap} km/h" if cap is not None else "—")
        details.configure(state="normal")
        details.delete("1.0", "end")
        details.insert("1.0", _result_details(result))
        details.configure(state="disabled")
        evaluate_button.configure(state="disabled" if state in ("evaluating", "acknowledging") else "normal")
        acknowledge_button.configure(state="normal" if session.can_acknowledge else "disabled")

    def start_evaluation() -> None:
        session.submit({key: variable.get() for key, variable in inputs.items()})
        if session.invalid_field in fields:
            fields[session.invalid_field].state(["invalid"])
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
        nonlocal syncing_replay
        if syncing_replay:
            return
        if session.invalid_field in fields:
            fields[session.invalid_field].state(["!invalid"])
        if key in ("scenario", "replay_aqi"):
            # Only operator edits switch to Custom. Filling a preset also fires a Tk trace.
            syncing_replay = True
            try:
                if key == "replay_aqi":
                    inputs["scenario"].set("custom")
                elif inputs["scenario"].get() != "custom":
                    inputs["replay_aqi"].set(preset_aqis.get(inputs["scenario"].get(), ""))
            finally:
                syncing_replay = False
        if key == "mode":
            refresh.set(False)
            schedule_refresh()
            update_source_note()
        session.edit()
        render()

    def update_source_note() -> None:
        source_badge.configure(text=_MODES.get(inputs["mode"].get(), "Choose an air-quality source"))
        if inputs["mode"].get() == "live":
            replay_fields.grid_remove()
            live_fields.grid()
            source_note.configure(text=(
                "Coordinates default to Vienna. Live uses Open-Meteo and the project's local Ollama model."
            ))
        else:
            live_fields.grid_remove()
            replay_fields.grid()
            source_note.configure(text="Simulated air data and policy response. Edit AQI to switch to custom.")

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

    def toggle_details() -> None:
        if details_area.winfo_manager():
            details_area.grid_remove()
            output.grid_configure(sticky="new")
            details_button.configure(text="Show decision sources")
        else:
            output.grid_configure(sticky="nsew")
            details_area.grid(row=4, sticky="nsew")
            details_button.configure(text="Hide decision sources")

    evaluate_button = ttk.Button(input_panel, text="Evaluate", command=start_evaluation, style="Primary.TButton")
    evaluate_button.grid(row=8, sticky="ew")
    ttk.Checkbutton(input_panel, text="Re-evaluate every 60 seconds", variable=refresh,
                    command=schedule_refresh, style="Refresh.TCheckbutton").grid(row=9, sticky="w", pady=(10, 0))
    acknowledge_button = ttk.Button(result_actions, text="Acknowledge result", command=record_acknowledgment)
    acknowledge_button.grid(row=0, column=1, sticky="e")
    details_button = ttk.Button(output, text="Show decision sources", command=toggle_details)
    details_button.grid(row=3, sticky="w", padx=16, pady=10)

    for key, variable in inputs.items():
        variable.trace_add("write", lambda *_args, key=key: on_edit(key))
    root.protocol("WM_DELETE_WINDOW", close)
    fields["illuminance_lux"].focus_set()
    update_source_note()
    render()
    poll()
    root.mainloop()
