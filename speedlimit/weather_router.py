"""Exact weather/light routing and a validated decimal measurement adapter.

The course exercise accepts integer millilux, micrometers and Celsius values.
The operational adapter accepts Decimal, int, float or numeric strings using a
decimal point (optionally scientific notation). Comma decimal strings are not
accepted here; CSV-specific parsing belongs to data preparation.
"""

from dataclasses import dataclass
from decimal import Decimal, DecimalException, InvalidOperation


NumericInput = Decimal | int | float | str


@dataclass(frozen=True, slots=True)
class WeatherDecision:
    """Validated measurements and the reasons for a weather reduction."""

    reduce: bool
    dark: bool
    black_ice: bool
    illuminance_millilux: Decimal
    water_um: Decimal
    temperature_c: Decimal


def _conditions(
    illuminance_millilux: Decimal | int,
    water_um: Decimal | int,
    temperature_c: Decimal | int,
) -> tuple[bool, bool]:
    dark = illuminance_millilux < 500
    black_ice = water_um > 1000 and temperature_c < 0
    return dark, black_ice


def should_reduce(
    illuminance_millilux: int, water_um: int, temperature_c: int
) -> bool:
    """Evaluate the sourced integer exercise contract without coercion.

    All three arguments must have type ``int``; booleans, floats and integer
    subclasses are rejected. Operational physical-range validation belongs to
    :func:`evaluate_measurements`. Negative Celsius is a valid temperature.
    """
    for name, value in (
        ("illuminance_millilux", illuminance_millilux),
        ("water_um", water_um),
        ("temperature_c", temperature_c),
    ):
        if type(value) is not int:
            raise TypeError(f"{name} must be an int, not {type(value).__name__}")
    dark, black_ice = _conditions(illuminance_millilux, water_um, temperature_c)
    return dark or black_ice


def _measurement(value: NumericInput, name: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (Decimal, int, float, str)):
        raise TypeError(f"{name} must be Decimal, int, float or a numeric string")
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric value using a decimal point") from exc
    if not number.is_finite():
        raise ValueError(f"{name} must be finite")
    return number


def _millilux(lux: Decimal) -> Decimal:
    # Multiplication or scaleb() can round under the caller's Decimal context.
    # Adjust the tuple exponent instead so a value just below 0.5 lux stays below
    # 500 millilux even when it has more digits than that context's precision.
    parts = lux.as_tuple()
    try:
        converted = Decimal((parts.sign, parts.digits, parts.exponent + 3))
    except (DecimalException, ValueError, OverflowError) as exc:
        raise ValueError("illuminance_lux cannot be represented in millilux") from exc
    # With the InvalidOperation trap disabled, an unrepresentable exponent can
    # yield NaN instead of raising. It must not reach the router comparisons.
    if not converted.is_finite():
        raise ValueError("illuminance_lux cannot be represented in millilux")
    return converted


def evaluate_measurements(
    illuminance_lux: NumericInput,
    water_um: NumericInput,
    temperature_c: NumericInput,
) -> WeatherDecision:
    """Validate measurements, convert lux exactly and evaluate the router.

    Values are normalized with ``Decimal(str(value))``. Fractional measurements
    are preserved. Light and road-water values must be nonnegative; temperature
    may be negative. Unsupported types raise TypeError; invalid numeric strings,
    non-finite values, unrepresentable unit conversions and invalid physical
    ranges raise ValueError.
    """
    light = _measurement(illuminance_lux, "illuminance_lux")
    water = _measurement(water_um, "water_um")
    temperature = _measurement(temperature_c, "temperature_c")
    if light < 0:
        raise ValueError("illuminance_lux must be nonnegative")
    if water < 0:
        raise ValueError("water_um must be nonnegative")

    light_millilux = _millilux(light)
    dark, black_ice = _conditions(light_millilux, water, temperature)
    return WeatherDecision(
        reduce=dark or black_ice,
        dark=dark,
        black_ice=black_ice,
        illuminance_millilux=light_millilux,
        water_um=water,
        temperature_c=temperature,
    )
