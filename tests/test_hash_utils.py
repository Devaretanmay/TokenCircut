
from tokencircuit.otel.hash_utils import compute_state_hash, extract_tool_type_signature


class TestComputeStateHash:
    def test_same_state_produces_same_hash(self):
        s1 = {"a": 1, "b": "hello"}
        s2 = {"b": "hello", "a": 1}
        assert compute_state_hash(s1) == compute_state_hash(s2)

    def test_different_state_produces_different_hash(self):
        s1 = {"a": 1}
        s2 = {"a": 2}
        assert compute_state_hash(s1) != compute_state_hash(s2)

    def test_timestamp_key_excluded(self):
        s1 = {"a": 1, "timestamp": "2024-01-01"}
        s2 = {"a": 1, "timestamp": "2024-01-02"}
        assert compute_state_hash(s1) == compute_state_hash(s2)

    def test_trace_id_excluded(self):
        s1 = {"a": 1, "trace_id": "abc"}
        s2 = {"a": 1, "trace_id": "def"}
        assert compute_state_hash(s1) == compute_state_hash(s2)

    def test_meta_key_excluded(self):
        s1 = {"a": 1, "_meta": {"x": 1}}
        s2 = {"a": 1, "_meta": {"x": 2}}
        assert compute_state_hash(s1) == compute_state_hash(s2)

    def test_tc_prefix_excluded(self):
        s1 = {"a": 1, "_tc_internal": "secret"}
        s2 = {"a": 1, "_tc_internal": "different"}
        assert compute_state_hash(s1) == compute_state_hash(s2)

    def test_empty_state(self):
        assert compute_state_hash({}) is not None
        assert len(compute_state_hash({})) == 64

    def test_nested_state(self):
        s = {"outer": {"inner": [1, 2, 3]}}
        h = compute_state_hash(s)
        assert isinstance(h, str)
        assert len(h) == 64


class TestExtractToolTypeSignature:
    def test_none_tool_call(self):
        assert extract_tool_type_signature(None) == "NO_TOOL_CALL"

    def test_empty_tool_call(self):
        assert extract_tool_type_signature({}) == "unknown()"

    def test_tool_without_args(self):
        tc = {"name": "search"}
        assert extract_tool_type_signature(tc) == "search()"

    def test_tool_with_args(self):
        tc = {"name": "search", "args": {"query": "hello", "limit": 10}}
        sig = extract_tool_type_signature(tc)
        assert sig.startswith("search(")
        assert "str" in sig
        assert "int" in sig

    def test_tool_with_multiple_arg_types(self):
        tc = {
            "name": "query_db",
            "args": {"table": "users", "limit": 100, "active": True},
        }
        sig = extract_tool_type_signature(tc)
        assert "str" in sig
        assert "int" in sig
        assert "bool" in sig

    def test_no_values_in_signature(self):
        tc = {"name": "search", "args": {"query": "secret-api-key"}}
        sig = extract_tool_type_signature(tc)
        assert "secret" not in sig
        assert "str" in sig
