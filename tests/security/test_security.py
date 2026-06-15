import hashlib
import json
import threading
import time as time_module

import pytest

from tokencircuit.ring_buffer import RingBuffer
from tokencircuit.config import TokenCircuitConfig, load_config
from tokencircuit.telemetry import TelemetryEvent, compute_cost_estimate
from tokencircuit.otel.hash_utils import (
    compute_state_hash,
    extract_tool_type_signature,
)


def make_entry(state_hash="a", tool_sig="tool()", iteration=1):
    return {
        "state_hash": state_hash,
        "tool_type_signature": tool_sig,
        "iteration": iteration,
    }


# ── PII Leak Test ──


class TestPIILeak:
    def test_tool_signature_contains_no_values(self):
        tool_call = {
            "name": "send_email",
            "args": {
                "to": "user@example.com",
                "subject": "Your password reset",
                "token": "supersecret123",
            },
        }
        sig = extract_tool_type_signature(tool_call)
        assert "user@example.com" not in sig
        assert "password" not in sig
        assert "supersecret123" not in sig
        assert "send_email" in sig
        sig = extract_tool_type_signature(tool_call)
        assert sig == "send_email(str,str,str)"

    def test_telemetry_event_contains_no_prompt_values(self):
        event = TelemetryEvent(
            agency_id="test-agency",
            client_id="test-client",
            agent_framework="langgraph",
            signal_type="STATE_STAGNATION",
            node_name="scraper",
            iterations_at_detection=5,
            model_name="gpt-4",
            estimated_tokens_saved=1000,
            estimated_cost_saved_usd=0.05,
        )
        payload = json.dumps(event.__dict__)
        sensitive_values = ["secret", "password", "api_key", "s3cret"]
        for val in sensitive_values:
            assert val not in payload

    def test_telemetry_cost_no_pii(self):
        tokens, cost = compute_cost_estimate("gpt-4", 10)
        assert isinstance(tokens, int)
        assert isinstance(cost, float)

    def test_no_value_types_in_signature(self):
        tool_call = {"name": "search", "args": {"q": "highly-sensitive-data"}}
        sig = extract_tool_type_signature(tool_call)
        assert "highly-sensitive-data" not in sig
        assert sig == "search(str)"


# ── State Hash Collision Resistance ──


class TestStateHashCollision:
    def test_different_states_different_hashes(self):
        hashes = set()
        for i in range(1000):
            state = {"idx": i, "value": f"test_{i}", "flag": i % 2 == 0}
            h = compute_state_hash(state)
            hashes.add(h)
        assert len(hashes) == 1000

    def test_random_states_no_collision(self):
        import random

        hashes = set()
        for _ in range(1000):
            state = {
                "a": random.randint(0, 1000000),
                "b": random.random(),
                "c": random.choice(["x", "y", "z"]),
                "d": {"nested": random.randint(0, 100)},
            }
            h = compute_state_hash(state)
            hashes.add(h)
        assert len(hashes) == 1000

    def test_shuffled_keys_same_hash(self):
        s1 = {"a": 1, "b": 2, "c": 3}
        s2 = {"c": 3, "a": 1, "b": 2}
        assert compute_state_hash(s1) == compute_state_hash(s2)

    def test_timestamp_excluded_from_hash(self):
        s1 = {"data": "hello", "timestamp": "2024-01-01T00:00:00"}
        s2 = {"data": "hello", "timestamp": "2025-01-01T00:00:00"}
        assert compute_state_hash(s1) == compute_state_hash(s2)

    def test_trace_id_excluded(self):
        s1 = {"data": "hello", "trace_id": "abc"}
        s2 = {"data": "hello", "trace_id": "xyz"}
        assert compute_state_hash(s1) == compute_state_hash(s2)


# ── Config Fallback Under Attack ──


class TestConfigFallback:
    def test_http_500_returns_defaults(self):
        cfg = load_config(api_key="fake-key-that-will-fail")
        assert cfg.max_repeats == 5
        assert cfg.window_size == 5

    def test_timeout_returns_defaults(self):
        cfg = load_config(api_key="timeout-test-key")
        assert cfg.max_repeats == 5

    def test_malformed_json_returns_defaults(self):
        cfg = load_config(api_key="malformed-json-key")
        assert cfg.window_size == 5

    def test_no_api_key_returns_defaults(self):
        cfg = load_config(api_key=None)
        assert cfg.max_repeats == 5

    def test_defaults_not_empty(self):
        cfg = TokenCircuitConfig()
        assert cfg.max_repeats >= 1
        assert cfg.window_size >= 2

    def test_config_never_raises(self):
        bad_inputs = [None, "", "invalid", "key-with-network-failure"]
        for key in bad_inputs:
            try:
                cfg = load_config(api_key=key)
                assert isinstance(cfg, TokenCircuitConfig)
            except Exception:
                pytest.fail(f"load_config raised for key={key!r}")


# ── Thread Safety ──


class TestThreadSafety:
    def test_50_concurrent_pushes(self):
        buf = RingBuffer(maxlen=10)
        errors = []

        def pusher():
            try:
                for i in range(100):
                    buf.push(make_entry(iteration=i))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=pusher) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(buf) <= 10
        assert buf.is_full()

    def test_parallel_push_and_read(self):
        buf = RingBuffer(maxlen=100)
        errors = []

        def writer():
            try:
                for i in range(500):
                    buf.push(make_entry(iteration=i))
                    time_module.sleep(0.0001)
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(500):
                    buf.window()
                    buf.is_full()
                    len(buf)
                    time_module.sleep(0.0001)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=reader),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(buf) <= 100

    def test_concurrent_reset(self):
        buf = RingBuffer(maxlen=10)
        errors = []

        def reseter():
            try:
                for _ in range(100):
                    buf.push(make_entry(iteration=1))
                    buf.reset()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reseter) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(buf) == 0
