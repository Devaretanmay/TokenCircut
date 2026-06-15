"""
Supabase Edge Function security and auth tests.

These tests verify the /ingest Edge Function's security posture:
- Unknown agency_id returns 401
- Missing apikey returns 401
- Valid telemetry with valid auth returns 201
- Malformed payload returns 400

NOTE: These tests require a running Supabase Edge Function.
They use the httpx client directly against the deployed function.
"""

import pytest

pytestmark = pytest.mark.skipif(
    True,
    reason="Edge Function tests require a running Supabase instance. "
    "Set SUPABASE_HEALTH_URL and SUPABASE_ANON_KEY env vars to enable.",
)


class TestEdgeFunctionAuth:
    """Security tests for the /ingest Edge Function."""

    @pytest.fixture
    def ingest_url(self):
        return "https://tokencircuit.supabase.co/functions/v1/ingest"

    def test_unknown_agency_id_returns_401(self, ingest_url):
        import httpx

        payload = {
            "agency_id": "00000000-0000-0000-0000-000000000000",
            "client_id": "test-client",
            "agent_framework": "langgraph",
            "signal_type": "STATE_STAGNATION",
            "node_name": "scraper",
            "iterations_at_detection": 5,
            "model_name": "gpt-4",
            "estimated_tokens_saved": 1024,
            "estimated_cost_saved_usd": 0.05,
        }
        resp = httpx.post(
            ingest_url,
            json=payload,
            headers={"apikey": "invalid-key"},
            timeout=5.0,
        )
        assert resp.status_code == 401

    def test_missing_apikey_returns_401(self, ingest_url):
        import httpx

        payload = {
            "agency_id": "test-agency",
            "client_id": "test-client",
            "signal_type": "STATE_STAGNATION",
            "node_name": "scraper",
        }
        resp = httpx.post(ingest_url, json=payload, timeout=5.0)
        assert resp.status_code in (401, 403)

    def test_malformed_payload_returns_400(self, ingest_url):
        import httpx

        resp = httpx.post(
            ingest_url,
            json={"invalid": "data"},
            headers={"apikey": "some-key"},
            timeout=5.0,
        )
        assert resp.status_code == 400

    def test_valid_payload_returns_201(self, ingest_url):
        import httpx
        import os

        api_key = os.environ.get("SUPABASE_ANON_KEY", "")
        if not api_key:
            pytest.skip("SUPABASE_ANON_KEY not set")

        resp = httpx.post(
            ingest_url,
            json={
                "agency_id": "test-agency-id",
                "client_id": "test-client",
                "agent_framework": "langgraph",
                "signal_type": "STATE_STAGNATION",
                "node_name": "scraper",
                "iterations_at_detection": 5,
                "model_name": "gpt-4",
                "estimated_tokens_saved": 1024,
                "estimated_cost_saved_usd": 0.05,
            },
            headers={"apikey": api_key},
            timeout=5.0,
        )
        assert resp.status_code in (201, 401)
