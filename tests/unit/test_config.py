import logging

import pytest

from tokencircuit.config import TokenCircuitConfig, load_config


class TestTokenCircuitConfig:
    def test_default_values(self):
        cfg = TokenCircuitConfig()
        assert cfg.max_repeats == 5
        assert cfg.window_size == 5
        assert cfg.agency_id is None

    def test_custom_values(self):
        cfg = TokenCircuitConfig(max_repeats=10, window_size=3, agency_id="abc")
        assert cfg.max_repeats == 10
        assert cfg.window_size == 3
        assert cfg.agency_id == "abc"

    def test_invalid_max_repeats(self):
        with pytest.raises(ValueError, match="max_repeats must be >= 1"):
            TokenCircuitConfig(max_repeats=0)

    def test_invalid_window_size(self):
        with pytest.raises(ValueError, match="window_size must be >= 2"):
            TokenCircuitConfig(window_size=1)

    def test_default_model_name(self):
        cfg = TokenCircuitConfig()
        assert cfg.model_name == "unknown"

    def test_telemetry_default_enabled(self):
        cfg = TokenCircuitConfig()
        assert cfg.telemetry_enabled is True


class TestLoadConfig:
    def test_no_api_key_returns_defaults(self):
        cfg = load_config(api_key=None)
        assert cfg.max_repeats == 5
        assert cfg.window_size == 5

    def test_invalid_api_key_returns_defaults(self, caplog):
        caplog.set_level(logging.WARNING)
        cfg = load_config(api_key="invalid_key_xyz")
        assert cfg.max_repeats == 5
        assert cfg.window_size == 5
        assert len(caplog.records) >= 1
        assert "config fetch failed" in caplog.text

    def test_empty_string_api_key_returns_defaults(self):
        cfg = load_config(api_key="")
        assert cfg.max_repeats == 5
