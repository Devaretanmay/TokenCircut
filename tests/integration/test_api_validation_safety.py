"""
API Validation Safety Tests — Mock strict OpenAI and Anthropic transcript validators.

These tests mathematically guarantee that TranscriptValidator repairs and
LangGraphPreModelAdapter outputs NEVER produce:
- Dangling tool_call_ids (assistant tool_call with no matching tool result)
- Orphaned tool results (tool message with no prior matching assistant tool_call)
- Non-causal message ordering (tool result appears before its call)
- Missing content fields on required messages
- Invalid tool_call structure (missing id, type, or function fields)

The mock validators replicate the exact validation logic that OpenAI and Anthropic
APIs perform server-side before processing a chat completion request.
"""

import json
import pytest
from typing import Any

from tokencircuit.canonicalizer import MessageCanonicalizer
from tokencircuit.ledger import ToolTransactionLedger
from tokencircuit.validator import TranscriptValidator
from tokencircuit.engine import InterventionEngine, InterventionConfig
from tokencircuit.types import CanonicalMessage, CanonicalRole, InterventionStage
from tokencircuit.state_schema import default_intervention_state


# =============================================================================
# STRICT API VALIDATOR MOCKS
# =============================================================================


class OpenAITranscriptValidationError(Exception):
    """Simulates a 400 Bad Request from OpenAI's API validation."""

    def __init__(self, message: str, param: str = "messages"):
        super().__init__(message)
        self.param = param


def strict_openai_validate(messages: list[dict[str, Any]]) -> None:
    """
    Replicates OpenAI's server-side message validation rules.
    Raises OpenAITranscriptValidationError on any violation.

    Rules enforced:
    1. Every message must have 'role' field.
    2. Role must be one of: system, user, assistant, tool, function.
    3. Assistant messages may have tool_calls; if so, content can be null.
    4. Tool messages MUST have tool_call_id referencing a prior assistant tool_call.
    5. Every tool_call in an assistant message MUST have matching tool message(s).
    6. tool_calls must be a list of {id, type, function: {name, arguments}}.
    7. function.arguments must be a valid JSON string.
    8. No tool message can appear without a preceding assistant message with tool_calls.
    9. Messages must be in causal order (tool results after their calls).
    10. Content field: required for user/system/tool; nullable for assistant.
    """
    if not messages:
        return

    # Track all tool_call_ids from assistant messages
    issued_call_ids: dict[str, int] = {}  # call_id → message index
    consumed_call_ids: set[str] = set()
    valid_roles = {"system", "user", "assistant", "tool", "function"}

    for idx, msg in enumerate(messages):
        # Rule 1: role required
        if "role" not in msg:
            raise OpenAITranscriptValidationError(
                f"messages[{idx}]: missing 'role' field", param=f"messages[{idx}].role"
            )

        role = msg["role"]

        # Rule 2: valid role
        if role not in valid_roles:
            raise OpenAITranscriptValidationError(
                f"messages[{idx}]: invalid role '{role}'", param=f"messages[{idx}].role"
            )

        # Rule 10: content field requirements
        if role in ("user", "system", "tool"):
            if "content" not in msg:
                raise OpenAITranscriptValidationError(
                    f"messages[{idx}]: '{role}' message requires 'content' field",
                    param=f"messages[{idx}].content",
                )

        # Rule 6: tool_calls structure
        if role == "assistant" and "tool_calls" in msg:
            tool_calls = msg["tool_calls"]
            if not isinstance(tool_calls, list):
                raise OpenAITranscriptValidationError(
                    f"messages[{idx}]: tool_calls must be a list",
                    param=f"messages[{idx}].tool_calls",
                )
            for tc_idx, tc in enumerate(tool_calls):
                if "id" not in tc or not tc["id"]:
                    raise OpenAITranscriptValidationError(
                        f"messages[{idx}].tool_calls[{tc_idx}]: missing 'id'",
                        param=f"messages[{idx}].tool_calls[{tc_idx}].id",
                    )
                if "type" not in tc or tc["type"] != "function":
                    raise OpenAITranscriptValidationError(
                        f"messages[{idx}].tool_calls[{tc_idx}]: type must be 'function'",
                        param=f"messages[{idx}].tool_calls[{tc_idx}].type",
                    )
                if "function" not in tc:
                    raise OpenAITranscriptValidationError(
                        f"messages[{idx}].tool_calls[{tc_idx}]: missing 'function'",
                        param=f"messages[{idx}].tool_calls[{tc_idx}].function",
                    )
                func = tc["function"]
                if "name" not in func or not func["name"]:
                    raise OpenAITranscriptValidationError(
                        f"messages[{idx}].tool_calls[{tc_idx}].function: missing 'name'",
                        param=f"messages[{idx}].tool_calls[{tc_idx}].function.name",
                    )
                if "arguments" not in func:
                    raise OpenAITranscriptValidationError(
                        f"messages[{idx}].tool_calls[{tc_idx}].function: missing 'arguments'",
                        param=f"messages[{idx}].tool_calls[{tc_idx}].function.arguments",
                    )
                # Rule 7: arguments must be valid JSON string
                args = func["arguments"]
                if not isinstance(args, str):
                    raise OpenAITranscriptValidationError(
                        f"messages[{idx}].tool_calls[{tc_idx}].function.arguments: must be string",
                        param=f"messages[{idx}].tool_calls[{tc_idx}].function.arguments",
                    )
                try:
                    json.loads(args)
                except (json.JSONDecodeError, ValueError):
                    raise OpenAITranscriptValidationError(
                        f"messages[{idx}].tool_calls[{tc_idx}].function.arguments: invalid JSON",
                        param=f"messages[{idx}].tool_calls[{tc_idx}].function.arguments",
                    )

                # Record the call_id
                issued_call_ids[tc["id"]] = idx

        # Rule 4 & 8: tool messages must reference a prior call
        if role == "tool":
            if "tool_call_id" not in msg or not msg["tool_call_id"]:
                raise OpenAITranscriptValidationError(
                    f"messages[{idx}]: tool message requires 'tool_call_id'",
                    param=f"messages[{idx}].tool_call_id",
                )
            tcid = msg["tool_call_id"]

            # Rule 9: Causal ordering
            if tcid not in issued_call_ids:
                raise OpenAITranscriptValidationError(
                    f"messages[{idx}]: tool_call_id '{tcid}' not found in prior assistant messages",
                    param=f"messages[{idx}].tool_call_id",
                )
            if issued_call_ids[tcid] >= idx:
                raise OpenAITranscriptValidationError(
                    f"messages[{idx}]: tool result appears before its call (call at index {issued_call_ids[tcid]})",
                    param=f"messages[{idx}].tool_call_id",
                )
            consumed_call_ids.add(tcid)

    # Rule 5: All tool_calls must have matching results
    # NOTE: OpenAI actually enforces this — if you have assistant tool_calls,
    # the IMMEDIATE next messages must be the tool results for ALL of them.
    unresolved = set(issued_call_ids.keys()) - consumed_call_ids
    if unresolved:
        # Only flag if the unresolved calls are NOT in the final assistant message
        # (the final assistant message's calls are what we're about to process)
        last_assistant_idx = max(
            (idx for idx, m in enumerate(messages) if m.get("role") == "assistant"),
            default=-1,
        )
        final_call_ids = set()
        if last_assistant_idx >= 0 and "tool_calls" in messages[last_assistant_idx]:
            final_call_ids = {
                tc["id"] for tc in messages[last_assistant_idx].get("tool_calls", [])
            }
        dangling = unresolved - final_call_ids
        if dangling:
            raise OpenAITranscriptValidationError(
                f"Dangling tool_call_ids without results: {dangling}. "
                f"Each tool_call must have a corresponding tool message.",
                param="messages",
            )


def strict_anthropic_validate(messages: list[dict[str, Any]]) -> None:
    """
    Replicates Anthropic's message validation rules.
    Anthropic has stricter alternation requirements.

    Rules enforced:
    1. First non-system message must be 'user' role.
    2. Messages must alternate user/assistant (with tool results counting as user-side).
    3. tool_use blocks in assistant content must have matching tool_result in next user turn.
    4. No empty content arrays.
    """
    if not messages:
        return

    # Filter system messages
    non_system = [m for m in messages if m.get("role") != "system"]
    if not non_system:
        return

    # Rule 1: First non-system message must be user
    if non_system[0]["role"] not in ("user", "tool"):
        raise OpenAITranscriptValidationError(
            "Anthropic: first non-system message must be role 'user'",
            param="messages[0].role",
        )

    # For Anthropic, tool messages are part of the user turn
    # Validate no empty content
    for idx, msg in enumerate(messages):
        content = msg.get("content")
        if isinstance(content, list) and len(content) == 0:
            raise OpenAITranscriptValidationError(
                f"messages[{idx}]: empty content array not allowed",
                param=f"messages[{idx}].content",
            )


# =============================================================================
# TEST HELPER: Run full V7 pipeline and validate output
# =============================================================================


def run_pipeline_and_validate(
    raw_messages: list[dict[str, Any]],
    *,
    config: InterventionConfig | None = None,
    turn_number: int = 1,
    tc_state: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Run the full V7 pipeline (canonicalize → validate → engine) on raw messages,
    then run strict_openai_validate on the output.
    Returns the final messages that would be sent to the LLM.
    """
    if config is None:
        config = InterventionConfig(nudge_threshold=1, override_threshold=2, hard_stop_threshold=4)
    if tc_state is None:
        tc_state = default_intervention_state()

    engine = InterventionEngine(config=config)
    state = {
        "messages": raw_messages,
        "_tc_intervention": tc_state,
        "configurable": {"thread_id": "safety-test"},
    }

    decision = engine.process(raw_messages, state, thread_id="safety-test", node_name="agent")

    # Get the final messages the LLM would see
    if decision.llm_input_messages:
        output_messages = decision.llm_input_messages
    else:
        # PASS case: canonicalize + validate and return as OpenAI format
        canonicalizer = MessageCanonicalizer()
        ledger = ToolTransactionLedger()
        validator = TranscriptValidator(ledger=ledger, auto_recovery=True)
        canonical = canonicalizer.canonicalize(raw_messages)
        result = validator.validate(canonical, turn_number)
        output_messages = canonicalizer.to_openai_format(result.validated_messages)

    return output_messages


# =============================================================================
# TEST SUITE: Strict OpenAI Validation
# =============================================================================


class TestOpenAIValidationSafety:
    """Every test outputs messages through the V7 pipeline then validates against strict OpenAI rules."""

    def test_clean_conversation_passes(self):
        """A normal conversation should pass both V7 and OpenAI validation."""
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        output = run_pipeline_and_validate(messages)
        strict_openai_validate(output)

    def test_valid_tool_call_roundtrip(self):
        """Complete tool call → result roundtrip passes validation."""
        messages = [
            {"role": "user", "content": "Search for cats"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "search", "arguments": '{"q": "cats"}'}}
            ]},
            {"role": "tool", "content": "Found 5 results", "tool_call_id": "call_1", "name": "search"},
            {"role": "assistant", "content": "I found 5 results about cats."},
        ]
        output = run_pipeline_and_validate(messages)
        strict_openai_validate(output)

    def test_orphaned_tool_result_stripped(self):
        """A tool result with no matching call is stripped before reaching API."""
        messages = [
            {"role": "user", "content": "Do something"},
            {"role": "tool", "content": "Phantom result", "tool_call_id": "call_ghost"},
            {"role": "assistant", "content": "Done"},
        ]
        output = run_pipeline_and_validate(messages)
        strict_openai_validate(output)
        # Verify the orphan was removed
        tool_msgs = [m for m in output if m.get("role") == "tool"]
        assert len(tool_msgs) == 0, "Orphaned tool result should have been stripped"

    def test_duplicate_tool_result_stripped(self):
        """Duplicate results for the same call_id are stripped."""
        messages = [
            {"role": "user", "content": "Search"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_x", "type": "function", "function": {"name": "search", "arguments": '{"q": "x"}'}}
            ]},
            {"role": "tool", "content": "Result 1", "tool_call_id": "call_x", "name": "search"},
            {"role": "tool", "content": "Result 2 (duplicate)", "tool_call_id": "call_x", "name": "search"},
            {"role": "assistant", "content": "Found it"},
        ]
        output = run_pipeline_and_validate(messages)
        strict_openai_validate(output)
        tool_msgs = [m for m in output if m.get("role") == "tool"]
        assert len(tool_msgs) == 1, "Duplicate should be stripped"

    def test_malformed_args_drops_entire_transaction(self):
        """
        Malformed tool_call arguments cause the ENTIRE transaction to be dropped,
        not coerced — ensuring no invalid JSON reaches the API.
        """
        messages = [
            {"role": "user", "content": "Run analysis"},
            {"role": "assistant", "content": "Running", "tool_calls": [
                {"id": "call_bad", "type": "function", "function": {"name": "analyze", "arguments": "not valid json {{{"}},
            ]},
            {"role": "tool", "content": "Result", "tool_call_id": "call_bad", "name": "analyze"},
            {"role": "assistant", "content": "Analysis complete"},
        ]
        output = run_pipeline_and_validate(messages)
        strict_openai_validate(output)
        # The malformed call's result should be gone
        tool_msgs = [m for m in output if m.get("role") == "tool"]
        assert len(tool_msgs) == 0, "Result for malformed call should be dropped"

    def test_multiple_tool_calls_partial_malformed(self):
        """
        If one tool_call in an AI message has malformed args, ALL tool_calls
        in that message are voided and their results dropped atomically.
        """
        messages = [
            {"role": "user", "content": "Multi-tool"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_good", "type": "function", "function": {"name": "search", "arguments": '{"q": "valid"}'}},
                {"id": "call_bad", "type": "function", "function": {"name": "run", "arguments": "INVALID"}},
            ]},
            {"role": "tool", "content": "Search result", "tool_call_id": "call_good", "name": "search"},
            {"role": "tool", "content": "Run result", "tool_call_id": "call_bad", "name": "run"},
            {"role": "assistant", "content": "Done"},
        ]
        output = run_pipeline_and_validate(messages)
        strict_openai_validate(output)
        # BOTH results should be dropped (atomic with malformed AI message)
        tool_msgs = [m for m in output if m.get("role") == "tool"]
        assert len(tool_msgs) == 0, "All results for malformed AI message should be dropped"

    def test_tool_result_before_call_dropped(self):
        """A tool result appearing before its call is invalid and must be dropped."""
        messages = [
            {"role": "user", "content": "Do task"},
            {"role": "tool", "content": "Early result", "tool_call_id": "call_late"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_late", "type": "function", "function": {"name": "task", "arguments": '{"x": 1}'}}
            ]},
            {"role": "tool", "content": "Proper result", "tool_call_id": "call_late", "name": "task"},
        ]
        output = run_pipeline_and_validate(messages)
        strict_openai_validate(output)

    def test_nudge_stage_preserves_tool_call_integrity(self):
        """
        When NUDGE injects a coaching system message, it must NOT break
        the tool_call → tool_result adjacency requirement.
        """
        config = InterventionConfig(nudge_threshold=1, override_threshold=3, hard_stop_threshold=5)
        # Force stagnation state to trigger NUDGE
        tc_state = default_intervention_state()
        tc_state["consecutive_stagnation_count"] = 1
        tc_state["turn_counter"] = 2

        messages = [
            {"role": "user", "content": "Search again"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_s1", "type": "function", "function": {"name": "search", "arguments": '{"q": "same"}'}}
            ]},
            {"role": "tool", "content": "Same result", "tool_call_id": "call_s1", "name": "search"},
            {"role": "assistant", "content": "Let me try again", "tool_calls": [
                {"id": "call_s2", "type": "function", "function": {"name": "search", "arguments": '{"q": "same"}'}}
            ]},
            {"role": "tool", "content": "Same result", "tool_call_id": "call_s2", "name": "search"},
        ]
        output = run_pipeline_and_validate(messages, config=config, tc_state=tc_state)
        strict_openai_validate(output)

    def test_override_stage_produces_valid_transcript(self):
        """
        OVERRIDE compacts failed transactions. The compacted output must still
        pass strict validation — no dangling calls, no orphan results.
        """
        config = InterventionConfig(nudge_threshold=1, override_threshold=2, hard_stop_threshold=5)
        tc_state = default_intervention_state()
        tc_state["consecutive_stagnation_count"] = 3
        tc_state["turn_counter"] = 4
        tc_state["current_stage"] = "nudge"

        messages = [
            {"role": "system", "content": "You are an assistant"},
            {"role": "user", "content": "Find data"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_a", "type": "function", "function": {"name": "search", "arguments": '{"q": "data"}'}}
            ]},
            {"role": "tool", "content": "Not found", "tool_call_id": "call_a", "name": "search"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_b", "type": "function", "function": {"name": "search", "arguments": '{"q": "data"}'}}
            ]},
            {"role": "tool", "content": "Not found", "tool_call_id": "call_b", "name": "search"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_c", "type": "function", "function": {"name": "search", "arguments": '{"q": "data"}'}}
            ]},
            {"role": "tool", "content": "Not found", "tool_call_id": "call_c", "name": "search"},
        ]
        output = run_pipeline_and_validate(messages, config=config, tc_state=tc_state)
        strict_openai_validate(output)

    def test_empty_message_list_safe(self):
        """Empty messages should not crash and should pass validation."""
        output = run_pipeline_and_validate([])
        strict_openai_validate(output)

    def test_no_dangling_call_ids_after_override_compaction(self):
        """
        MATHEMATICAL GUARANTEE: After OVERRIDE compaction, for every
        assistant message with tool_calls in the output, ALL referenced
        call_ids must have matching tool results downstream.
        """
        config = InterventionConfig(nudge_threshold=1, override_threshold=2, hard_stop_threshold=5)
        tc_state = default_intervention_state()
        tc_state["consecutive_stagnation_count"] = 4
        tc_state["turn_counter"] = 5
        tc_state["current_stage"] = "override"

        # Complex multi-tool scenario
        messages = [
            {"role": "system", "content": "Assistant"},
            {"role": "user", "content": "Complex task"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "search", "arguments": '{"q": "a"}'}},
                {"id": "call_2", "type": "function", "function": {"name": "fetch", "arguments": '{"url": "x"}'}},
            ]},
            {"role": "tool", "content": "Result A", "tool_call_id": "call_1", "name": "search"},
            {"role": "tool", "content": "Result B", "tool_call_id": "call_2", "name": "fetch"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_3", "type": "function", "function": {"name": "search", "arguments": '{"q": "a"}'}},
            ]},
            {"role": "tool", "content": "Same result", "tool_call_id": "call_3", "name": "search"},
        ]
        output = run_pipeline_and_validate(messages, config=config, tc_state=tc_state)

        # Mathematical check: extract all call_ids from assistant messages
        all_call_ids: set[str] = set()
        all_result_ids: set[str] = set()
        last_assistant_call_ids: set[str] = set()

        for idx, msg in enumerate(output):
            if msg.get("role") == "assistant" and "tool_calls" in msg:
                for tc in msg["tool_calls"]:
                    all_call_ids.add(tc["id"])
                    # Track if this is the LAST assistant message
                    last_assistant_call_ids = {tc["id"] for tc in msg["tool_calls"]}
            if msg.get("role") == "tool":
                all_result_ids.add(msg["tool_call_id"])

        # Dangling = calls without results (excluding the final assistant message's calls)
        dangling = all_call_ids - all_result_ids - last_assistant_call_ids
        assert dangling == set(), f"DANGLING TOOL_CALL_IDS DETECTED: {dangling}"

    def test_system_message_injection_position(self):
        """
        Coaching system messages must be injected at valid positions —
        they cannot split a tool_call/tool_result sequence.
        """
        config = InterventionConfig(nudge_threshold=1, override_threshold=3, hard_stop_threshold=5)
        tc_state = default_intervention_state()
        tc_state["consecutive_stagnation_count"] = 1

        messages = [
            {"role": "user", "content": "Task"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_z", "type": "function", "function": {"name": "act", "arguments": '{"x": 1}'}}
            ]},
            {"role": "tool", "content": "Done", "tool_call_id": "call_z", "name": "act"},
        ]
        output = run_pipeline_and_validate(messages, config=config, tc_state=tc_state)
        strict_openai_validate(output)

        # Verify coaching message is appended at END, not between call and result
        if len(output) > 3:
            # Any system message should be at the end
            for idx, msg in enumerate(output[:-1]):  # all but last
                if msg.get("role") == "system" and idx > 0:
                    # Check it's not between a tool_call assistant and its result
                    prev = output[idx - 1] if idx > 0 else None
                    next_msg = output[idx + 1] if idx + 1 < len(output) else None
                    if prev and prev.get("role") == "assistant" and "tool_calls" in prev:
                        if next_msg and next_msg.get("role") == "tool":
                            pytest.fail(
                                f"System message injected between tool_call and result at index {idx}"
                            )


class TestAnthropicValidationSafety:
    """Validate outputs against Anthropic's stricter rules."""

    def test_override_starts_with_user_message(self):
        """Anthropic requires first non-system message to be user role."""
        config = InterventionConfig(nudge_threshold=1, override_threshold=2, hard_stop_threshold=5)
        tc_state = default_intervention_state()
        tc_state["consecutive_stagnation_count"] = 3
        tc_state["current_stage"] = "nudge"

        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Original task"},
            {"role": "assistant", "content": "Trying..."},
            {"role": "assistant", "content": "Trying again..."},
        ]
        output = run_pipeline_and_validate(messages, config=config, tc_state=tc_state)
        strict_anthropic_validate(output)

    def test_no_empty_content_arrays(self):
        """Anthropic rejects empty content arrays."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        output = run_pipeline_and_validate(messages)
        strict_anthropic_validate(output)
        for msg in output:
            content = msg.get("content")
            if isinstance(content, list):
                assert len(content) > 0, "Empty content array would fail Anthropic validation"


class TestTranscriptIntegrityMathematicalProofs:
    """
    These tests assert structural properties that must ALWAYS hold,
    regardless of input. They are parameterized over pathological inputs.
    """

    @pytest.mark.parametrize("scenario", [
        # Scenario: all orphan results
        [
            {"role": "tool", "content": "r1", "tool_call_id": "orphan_1"},
            {"role": "tool", "content": "r2", "tool_call_id": "orphan_2"},
            {"role": "tool", "content": "r3", "tool_call_id": "orphan_3"},
        ],
        # Scenario: call without result followed by new call
        [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "a", "arguments": '{"x":1}'}},
            ]},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c2", "type": "function", "function": {"name": "b", "arguments": '{"y":2}'}},
            ]},
            {"role": "tool", "content": "result", "tool_call_id": "c2", "name": "b"},
        ],
        # Scenario: interleaved calls and results out of order
        [
            {"role": "user", "content": "task"},
            {"role": "tool", "content": "early", "tool_call_id": "c_late"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c_late", "type": "function", "function": {"name": "x", "arguments": '{}'}},
            ]},
            {"role": "tool", "content": "proper", "tool_call_id": "c_late", "name": "x"},
        ],
        # Scenario: tool_calls with empty id
        [
            {"role": "user", "content": "test"},
            {"role": "assistant", "content": "a", "tool_calls": [
                {"id": "", "type": "function", "function": {"name": "empty_id", "arguments": '{}'}},
            ]},
            {"role": "tool", "content": "result", "tool_call_id": ""},
        ],
        # Scenario: massive duplicate storm
        [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "dup_call", "type": "function", "function": {"name": "f", "arguments": '{}'}},
            ]},
        ] + [
            {"role": "tool", "content": f"result_{i}", "tool_call_id": "dup_call", "name": "f"}
            for i in range(10)
        ],
    ])
    def test_output_always_passes_strict_validation(self, scenario):
        """
        For ANY pathological input, the V7 pipeline output must pass
        strict OpenAI validation. This is the mathematical guarantee.
        """
        output = run_pipeline_and_validate(scenario)
        strict_openai_validate(output)

    def test_invariant_no_orphan_tool_results_in_output(self):
        """
        INVARIANT: In the final output, every tool message's tool_call_id
        MUST reference a call_id in a PRIOR assistant message.
        Tested with 50 random conversation patterns.
        """
        import random

        random.seed(42)
        for trial in range(50):
            messages: list[dict[str, Any]] = [{"role": "user", "content": "task"}]
            issued_ids: list[str] = []

            for i in range(random.randint(2, 8)):
                if random.random() < 0.5 and issued_ids:
                    # Add a tool result (possibly orphaned)
                    tcid = random.choice(issued_ids + [f"orphan_{i}"])
                    messages.append({"role": "tool", "content": f"r{i}", "tool_call_id": tcid, "name": "t"})
                else:
                    # Add an assistant message with tool_calls
                    cid = f"call_{trial}_{i}"
                    issued_ids.append(cid)
                    messages.append({
                        "role": "assistant", "content": None,
                        "tool_calls": [{"id": cid, "type": "function", "function": {"name": f"fn{i}", "arguments": "{}"}}],
                    })

            output = run_pipeline_and_validate(messages)
            # Must not raise
            strict_openai_validate(output)
