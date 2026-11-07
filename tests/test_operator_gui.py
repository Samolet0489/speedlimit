"""Headless tests for the state machine driving the Tkinter operator window."""

from threading import Event, get_ident
from time import monotonic
import unittest

from speedlimit.operator_gui import _DEFAULT_INPUTS, _OperatorSession, _result_details


def decision(mode="replay", status="recommended", limit=80):
    return {
        "decision_id": "simulated-123", "mode": mode, "status": status, "limit_kmh": limit,
        "reasons": ["Weather and air branches agree on the displayed cap."],
        "evaluated_at": "2026-11-07T12:00:00+00:00",
        "weather": {
            "status": "valid", "limit_kmh": 80, "model_id": "course-model",
            "candidates": ({"speed_kmh": 70, "status": "accepted", "predicted_count": 0.823,
                            "neighbors": 14, "reason": "Candidate has support."},),
        },
        "air": {
            "status": "valid", "limit_kmh": limit, "policy_id": "course-policy-v1",
            "data_time": "2026-11-07T11:00:00+00:00",
            "data_source": "offline fixture" if mode == "replay" else "Open-Meteo",
            "response_source": "scripted response" if mode == "replay" else "live provider",
        },
    }


class OperatorGuiTests(unittest.TestCase):
    def wait_until(self, session, predicate):
        deadline = monotonic() + 3
        tick = Event()
        while monotonic() < deadline:
            session.poll()
            if predicate():
                return
            tick.wait(0.005)
        self.fail(f"Operator session did not reach the expected state. Current status is {session.status}.")

    def test_UI01_default_replay_shows_decision_reasons_times_and_sources(self):
        payloads = []
        def evaluate(payload):
            payloads.append(payload)
            return decision()

        session = _OperatorSession(evaluate)
        session.submit(_DEFAULT_INPUTS)
        self.wait_until(session, lambda: session.status == "recommended")
        self.assertEqual(payloads, [_DEFAULT_INPUTS])
        self.assertEqual(session.result["limit_kmh"], 80)
        self.assertTrue(session.can_acknowledge)
        details = _result_details(session.result)
        for text in ("80 km/h", "Weather and air branches agree", "2026-11-07T12:00:00+00:00",
                     "2026-11-07T11:00:00+00:00", "Replay", "offline fixture", "scripted response",
                     "course-policy-v1", "course-model", "70 km/h", "0.823 predicted events/hour",
                     "14 supporting observations", "Candidate has support"):
            with self.subTest(text=text):
                self.assertIn(text, details)

    def test_UI02_invalid_field_clears_the_decision_and_prevents_evaluation(self):
        calls = []
        def evaluate(payload):
            calls.append(payload)
            return decision()

        session = _OperatorSession(evaluate)
        session.submit(_DEFAULT_INPUTS)
        self.wait_until(session, lambda: session.status == "recommended")
        for key, value, label in (("water_um", "", "Road water"), ("temperature_c", "cold", "Temperature"),
                                  ("illuminance_lux", "NaN", "Illuminance")):
            with self.subTest(field=key):
                session.submit({**_DEFAULT_INPUTS, key: value})
                self.assertEqual(session.status, "invalid")
                self.assertEqual(session.invalid_field, key)
                self.assertIn(label, session.message)
                self.assertIsNone(session.result)
                self.assertFalse(session.can_acknowledge)
        self.assertEqual(len(calls), 1)

    def test_UI03_late_reply_cannot_replace_the_current_request(self):
        started_old, release_old = Event(), Event()
        started_new, release_new = Event(), Event()
        self.addCleanup(release_old.set)
        self.addCleanup(release_new.set)
        gui_thread = get_ident()
        calls = []

        def evaluate(payload):
            calls.append((get_ident(), payload.copy()))
            if payload["scenario"] == "good":
                started_old.set()
                release_old.wait(5)
                return decision(limit=80)
            started_new.set()
            release_new.wait(5)
            return decision(limit=60)

        session = _OperatorSession(evaluate)
        session.submit(_DEFAULT_INPUTS)
        self.assertTrue(started_old.wait(2))
        session.edit()
        self.assertIsNone(session.result)
        session.submit({**_DEFAULT_INPUTS, "scenario": "stale"})
        # Replacing a queued job keeps only the newest request.
        latest = {**_DEFAULT_INPUTS, "scenario": "poor"}
        session.submit(latest)
        latest["scenario"] = "malformed"
        release_old.set()
        self.wait_until(session, started_new.is_set)
        self.assertEqual(session.status, "evaluating")
        self.assertIsNone(session.result)
        self.assertFalse(session.can_acknowledge)
        self.assertEqual([payload["scenario"] for _, payload in calls], ["good", "poor"])
        self.assertTrue(all(thread != gui_thread for thread, _ in calls))
        release_new.set()
        self.wait_until(session, lambda: session.status == "recommended")
        self.assertEqual(session.result["limit_kmh"], 60)

    def test_UI04_live_mode_requires_coordinates_and_keeps_sources_distinct(self):
        calls = []
        def evaluate(payload):
            calls.append(payload)
            return decision(mode=payload["mode"])

        session = _OperatorSession(evaluate)
        session.submit(_DEFAULT_INPUTS)
        self.wait_until(session, lambda: session.status == "recommended")
        session.edit()
        self.assertIsNone(session.result)
        session.submit({**_DEFAULT_INPUTS, "mode": "live"})
        self.assertEqual((session.status, session.invalid_field), ("invalid", "latitude"))
        session.submit({**_DEFAULT_INPUTS, "mode": "live", "latitude": "48.2082", "longitude": "16.3738"})
        self.wait_until(session, lambda: session.status == "recommended")
        self.assertEqual([payload["mode"] for payload in calls], ["replay", "live"])
        details = _result_details(session.result)
        self.assertIn("Live", details)
        self.assertIn("Open-Meteo", details)
        self.assertIn("live provider", details)
        self.assertNotIn("offline fixture", details)

    def test_UI05_acknowledgment_is_for_the_current_valid_decision_only(self):
        recorded = []
        def evaluate(payload):
            if payload["scenario"] == "stale":
                return decision(status="review_needed", limit=None)
            return decision()

        session = _OperatorSession(evaluate, recorded.append)
        session.submit({**_DEFAULT_INPUTS, "scenario": "stale"})
        self.wait_until(session, lambda: session.status == "review_needed")
        self.assertIsNone(session.result["limit_kmh"])
        self.assertFalse(session.can_acknowledge)
        session.acknowledge()
        self.assertEqual(recorded, [])
        session.submit(_DEFAULT_INPUTS)
        self.wait_until(session, lambda: session.status == "recommended")
        session.acknowledge()
        session.acknowledge()
        self.wait_until(session, lambda: session.status == "acknowledged")
        self.assertEqual([value["decision_id"] for value in recorded], ["simulated-123"])
        self.assertIn("recorded", session.message)
        session.edit()
        session.acknowledge()
        self.assertEqual(len(recorded), 1)

    def test_UI05_acknowledgment_failure_leaves_the_decision_available_to_retry(self):
        def fail(_result):
            raise OSError("private log path")

        session = _OperatorSession(lambda _payload: decision(), fail)
        session.submit(_DEFAULT_INPUTS)
        self.wait_until(session, lambda: session.status == "recommended")
        session.acknowledge()
        self.wait_until(session, lambda: session.status == "recommended")
        self.assertTrue(session.can_acknowledge)
        self.assertIn("could not be recorded", session.message)
        self.assertNotIn("private log path", session.message)

    def test_worker_and_mislabeled_results_return_review_without_a_limit(self):
        def fail(_payload):
            raise RuntimeError("provider secret")

        for callback in (fail, lambda _payload: decision(mode="live"), lambda _payload: decision(limit=True)):
            with self.subTest(callback=callback):
                session = _OperatorSession(callback)
                session.submit(_DEFAULT_INPUTS)
                self.wait_until(session, lambda: session.status == "review_needed")
                self.assertIsNone(session.result)
                self.assertFalse(session.can_acknowledge)
                self.assertNotIn("provider secret", session.message)

    def test_close_returns_without_waiting_for_a_pending_worker(self):
        started, release, finished = Event(), Event(), Event()
        self.addCleanup(release.set)

        def evaluate(_payload):
            started.set()
            release.wait(5)
            finished.set()
            return decision()

        session = _OperatorSession(evaluate)
        session.submit(_DEFAULT_INPUTS)
        self.assertTrue(started.wait(2))
        session.close()
        self.assertFalse(finished.is_set())
        release.set()
        self.assertTrue(finished.wait(2))
        self.assertFalse(session.poll())
        self.assertIsNone(session.result)
        self.assertFalse(session.can_acknowledge)


if __name__ == "__main__":
    unittest.main()
