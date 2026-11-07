"""Sourced WR cases and operational boundary/validation tests for C3."""

from dataclasses import FrozenInstanceError
from decimal import Decimal, InvalidOperation, localcontext
import unittest

from speedlimit.weather_router import (
    WeatherDecision,
    evaluate_measurements,
    should_reduce,
)


class IntegerWeatherRouterTests(unittest.TestCase):
    def test_WR_01_black_ice_requires_reduction(self):
        self.assertIs(should_reduce(500, 1001, -1), True)

    def test_WR_02_water_equal_to_threshold_does_not_trigger(self):
        self.assertIs(should_reduce(500, 1000, -1), False)

    def test_WR_03_zero_temperature_does_not_trigger_ice(self):
        self.assertIs(should_reduce(500, 1001, 0), False)

    def test_WR_04_darkness_requires_reduction_independently(self):
        self.assertIs(should_reduce(499, 1001, 0), True)

    def test_WR_05_ordinary_good_conditions_do_not_trigger(self):
        self.assertIs(should_reduce(1000, 0, 10), False)

    def test_MC_DC_each_atomic_condition_independently_changes_result(self):
        pairs = (
            ("dark", (500, 1001, 0), (499, 1001, 0)),
            ("water", (500, 1000, -1), (500, 1001, -1)),
            ("temperature", (500, 1001, 0), (500, 1001, -1)),
        )
        for condition, false_input, true_input in pairs:
            with self.subTest(condition=condition):
                self.assertIs(should_reduce(*false_input), False)
                self.assertIs(should_reduce(*true_input), True)

    def test_both_weather_reasons_still_produce_one_boolean_decision(self):
        self.assertIs(should_reduce(499, 1001, -1), True)

    def test_integer_contract_rejects_coercion_and_booleans_in_every_position(self):
        class IntSubclass(int):
            pass

        invalid_values = (True, False, 500.0, "500", Decimal("500"), None, IntSubclass(1))
        for position in range(3):
            for invalid in invalid_values:
                with self.subTest(position=position, value=repr(invalid)):
                    args = [500, 1000, 0]
                    args[position] = invalid
                    with self.assertRaises(TypeError):
                        should_reduce(*args)

    def test_monotonic_examples_do_not_reverse_a_weather_condition(self):
        # Brighter conditions alone cannot create darkness; more water while
        # freezing cannot remove black-ice danger; warming cannot create it.
        self.assertEqual(
            [should_reduce(light, 0, 10) for light in (0, 499, 500, 1000)],
            [True, True, False, False],
        )
        self.assertEqual(
            [should_reduce(500, water, -1) for water in (0, 1000, 1001, 2000)],
            [False, False, True, True],
        )
        self.assertEqual(
            [should_reduce(500, 1001, temp) for temp in (-10, -1, 0, 10)],
            [True, True, False, False],
        )


class MeasurementAdapterTests(unittest.TestCase):
    def test_lux_conversion_and_exact_darkness_boundary(self):
        below = evaluate_measurements("0.499", 0, 10)
        equal = evaluate_measurements("0.5", 0, 10)
        above = evaluate_measurements("0.501", 0, 10)
        self.assertEqual(below.illuminance_millilux, Decimal("499"))
        self.assertTrue(below.dark)
        self.assertFalse(equal.dark)
        self.assertFalse(above.dark)

    def test_fractional_negative_temperature_is_not_truncated_to_zero(self):
        decision = evaluate_measurements("0.5", "1000.1", "-0.01")
        self.assertTrue(decision.black_ice)
        self.assertTrue(decision.reduce)
        self.assertFalse(decision.dark)
        self.assertEqual(decision.temperature_c, Decimal("-0.01"))
        self.assertEqual(decision.water_um, Decimal("1000.1"))

    def test_fractional_water_and_zero_temperature_boundaries(self):
        self.assertFalse(evaluate_measurements("0.5", "1000", "-0.01").black_ice)
        self.assertTrue(evaluate_measurements("0.5", "1000.0001", "-0.01").black_ice)
        self.assertFalse(evaluate_measurements("0.5", "1000.0001", "0").black_ice)
        self.assertFalse(evaluate_measurements("0.5", "1000.0001", "-0.0").black_ice)

    def test_preserves_precision_even_under_a_small_decimal_context(self):
        lux = "0.4999999999999999999999999999999999999999"
        expected_millilux = Decimal("499.9999999999999999999999999999999999999")
        with localcontext() as context:
            context.prec = 6
            decision = evaluate_measurements(lux, "1000.0000001", "-0.00000001")
        self.assertEqual(decision.illuminance_millilux, expected_millilux)
        self.assertTrue(decision.dark)
        self.assertTrue(decision.black_ice)

    def test_unrepresentable_millilux_conversion_raises_value_error_with_trap(self):
        with localcontext() as context:
            context.traps[InvalidOperation] = True
            with self.assertRaisesRegex(ValueError, "cannot be represented in millilux"):
                evaluate_measurements("1e999999999999999999", 0, 0)

    def test_unrepresentable_millilux_conversion_raises_value_error_without_trap(self):
        with localcontext() as context:
            context.traps[InvalidOperation] = False
            with self.assertRaisesRegex(ValueError, "cannot be represented in millilux"):
                evaluate_measurements("1e999999999999999999", 0, 0)

    def test_large_representable_light_does_not_have_an_arbitrary_upper_bound(self):
        decision = evaluate_measurements("1e1000", 0, 0)
        self.assertEqual(decision.illuminance_millilux, Decimal("1e1003"))
        self.assertFalse(decision.reduce)

    def test_accepts_decimal_int_float_and_dot_numeric_strings(self):
        for light in (Decimal("0.5"), 0.5, "0.5", " 5e-1 "):
            with self.subTest(light=repr(light)):
                decision = evaluate_measurements(light, 1001, -1)
                self.assertIsInstance(decision, WeatherDecision)
                self.assertEqual(decision.illuminance_millilux, Decimal("500"))
                self.assertEqual(decision.water_um, Decimal("1001"))
                self.assertEqual(decision.temperature_c, Decimal("-1"))
                self.assertTrue(decision.black_ice)
        self.assertFalse(evaluate_measurements(1, 0.0, Decimal("10")).reduce)

    def test_float_normalization_uses_its_string_value(self):
        value = 0.1 + 0.2
        decision = evaluate_measurements(value, 0, 10)
        self.assertEqual(decision.illuminance_millilux, Decimal("300.00000000000004"))

    def test_rejects_unsupported_types_in_every_position(self):
        for position in range(3):
            for invalid in (None, True, False, [], {}, object(), complex(1, 0)):
                with self.subTest(position=position, type=type(invalid).__name__):
                    args = ["0.5", "1000", "0"]
                    args[position] = invalid
                    with self.assertRaises(TypeError):
                        evaluate_measurements(*args)

    def test_rejects_nonnumeric_and_comma_strings_in_every_position(self):
        for position in range(3):
            for invalid in ("", " ", "unknown", "0,5", "1,000"):
                with self.subTest(position=position, value=invalid):
                    args = ["0.5", "1000", "0"]
                    args[position] = invalid
                    with self.assertRaises(ValueError):
                        evaluate_measurements(*args)

    def test_rejects_nonfinite_values_in_every_position(self):
        invalid_values = (
            float("nan"), float("inf"), float("-inf"),
            Decimal("NaN"), Decimal("sNaN"), Decimal("Infinity"),
            "NaN", "Infinity", "-Infinity",
        )
        for position in range(3):
            for invalid in invalid_values:
                with self.subTest(position=position, value=str(invalid)):
                    args = ["0.5", "1000", "0"]
                    args[position] = invalid
                    with self.assertRaises(ValueError):
                        evaluate_measurements(*args)

    def test_rejects_negative_light_or_water_but_accepts_negative_temperature(self):
        for position in (0, 1):
            with self.subTest(position=position):
                args = ["0.5", "1000", "0"]
                args[position] = "-0.0001"
                with self.assertRaises(ValueError):
                    evaluate_measurements(*args)
        self.assertTrue(evaluate_measurements("0.5", "1001", "-30.5").reduce)

    def test_returns_both_reasons_and_immutable_decimal_measurements(self):
        decision = evaluate_measurements("0.499", "1000.1", "-0.01")
        self.assertTrue(decision.dark)
        self.assertTrue(decision.black_ice)
        self.assertIs(decision.reduce, True)
        for value in (decision.illuminance_millilux, decision.water_um, decision.temperature_c):
            self.assertIsInstance(value, Decimal)
        with self.assertRaises(FrozenInstanceError):
            decision.reduce = False


if __name__ == "__main__":
    unittest.main()
