"""TokenCircuit V7.0 — Real-World Stress Tests

Three production-realistic agentic failure scenarios using the V7 InterventionEngine:

  1. Cloudflare Wall  — Agent hitting 403 Forbidden (FUTILE_ACTION)
  2. Delegation Deadlock — CrewAI-style delegation loop (FUTILE_ACTION)
  3. Silent State Rot — Agent with broken tool returning null (STATE_STAGNATION)

No real API keys required. LLM behavior is simulated deterministically,
isolating TokenCircuit's detection from network flakiness and cost.
"""

import asyncio
import json
import logging
import sys
import time

from tokencircuit.engine import InterventionConfig, InterventionEngine
from tokencircuit.types import (
    CanonicalMessage,
    CanonicalRole,
    InterventionStage,
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

# ── helpers ──────────────────────────────────────────────────────────────────

_RECURSION_LIMIT = 25
_MODEL = "gpt-4o"
_TOKENS_PER_CALL = 1024
_COST_PER_1K_INPUT = 0.0025
_COST_PER_1K_OUTPUT = 0.01


def _real_cost(tokens: int, input_frac: float = 0.6) -> float:
    """Estimate real USD cost for token count."""
    input_t = int(tokens * input_frac)
    output_t = tokens - input_t
    input_cost = input_t / 1000 * _COST_PER_1K_INPUT
    output_cost = output_t / 1000 * _COST_PER_1K_OUTPUT
    return input_cost + output_cost


def _projected_cost(iterations: int) -> float:
    return _real_cost(iterations * _TOKENS_PER_CALL)


def _make_tool_call_messages(
    tool_name: str,
    args: dict,
    call_id: str,
    ai_content: str,
    tool_result: str,
) -> list[CanonicalMessage]:
    """Create a pair of AI+Tool messages simulating one tool call cycle."""
    return [
        CanonicalMessage(
            role=CanonicalRole.AI,
            content=ai_content,
            tool_calls=[{"name": tool_name, "args": args, "id": call_id}],
            source_index=0,
        ),
        CanonicalMessage(
            role=CanonicalRole.TOOL,
            content=tool_result,
            tool_call_id=call_id,
            source_index=1,
        ),
    ]


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO 1 — The Cloudflare Wall (FUTILE_ACTION)
# ══════════════════════════════════════════════════════════════════════════════

def run_scenario_1() -> dict:
    """Simulates an agent that repeatedly calls fetch_url and gets 403 Forbidden.

    The agent's AI output is semantically identical each turn.
    Expected: TokenCircuit detects FUTILE_ACTION/SEMANTIC_STAGNATION within 5-8 turns.
    """
    print("=" * 72)
    print("SCENARIO 1: Cloudflare Wall (FUTILE_ACTION)")
    print("=" * 72)

    config = InterventionConfig(
        nudge_threshold=3,
        override_threshold=5,
        hard_stop_threshold=8,
        window_size=5,
        similarity_threshold=0.92,
    )
    engine = InterventionEngine(config=config)

    thread_id = "scenario_1"
    node_name = "agent"
    state: dict = {}

    start = time.perf_counter()
    interrupted_at = 0
    signal_type = "NONE"
    final_stage = InterventionStage.PASS

    for step in range(1, _RECURSION_LIMIT + 1):
        # Build the growing transcript (same tool call every time)
        messages = []
        for i in range(step):
            messages.extend(
                _make_tool_call_messages(
                    tool_name="fetch_url",
                    args={"url": "https://pricing.example.com/v2/data"},
                    call_id=f"call_fetch_{i}",
                    ai_content="I need to fetch the pricing data.",
                    tool_result=json.dumps({"error": "403 Forbidden", "body": ""}),
                )
            )

        decision = engine.process(
            messages=messages,
            state=state,
            thread_id=thread_id,
            node_name=node_name,
        )

        # Apply state patch
        if decision.state_patch:
            state["_tc_intervention"] = decision.state_patch

        final_stage = decision.stage

        if decision.should_terminate:
            interrupted_at = step
            signal_type = ", ".join(s.value for s in decision.signals) if decision.signals else "UNKNOWN"  # noqa: E501
            break

        if decision.stage > InterventionStage.PASS:
            if interrupted_at == 0:
                signal_type = ", ".join(s.value for s in decision.signals) if decision.signals else "UNKNOWN"  # noqa: E501

    if interrupted_at == 0:
        interrupted_at = step  # noqa: F821

    elapsed = time.perf_counter() - start

    total_tokens = interrupted_at * _TOKENS_PER_CALL
    projected_tokens = (_RECURSION_LIMIT - 1) * _TOKENS_PER_CALL
    interrupted_cost = _projected_cost(interrupted_at)
    projected_cost = _projected_cost(_RECURSION_LIMIT - 1)
    margin = projected_cost - interrupted_cost

    result = {
        "scenario": "Cloudflare Wall",
        "framework": "V7 InterventionEngine",
        "model": _MODEL,
        "interrupted_at": interrupted_at,
        "final_stage": final_stage.name,
        "recursion_limit": _RECURSION_LIMIT,
        "signal_type": signal_type,
        "tokens_consumed": total_tokens,
        "tokens_saved": projected_tokens - total_tokens,
        "cost_interrupted_usd": round(interrupted_cost, 4),
        "cost_projected_usd": round(projected_cost, 4),
        "margin_saved_usd": round(margin, 4),
        "elapsed_seconds": round(elapsed, 3),
        "pass": final_stage >= InterventionStage.HARD_STOP and interrupted_at < 15,
    }
    return result


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO 2 — Delegation Deadlock (CrewAI-style)
# ══════════════════════════════════════════════════════════════════════════════

def run_scenario_2() -> dict:
    """Simulates a delegation deadlock: Researcher → rate_limited → escalates to
    Manager → re-delegates to Researcher → rate_limited → repeats.

    Uses the V7 InterventionEngine directly with simulated canonical messages.
    """
    print("=" * 72)
    print("SCENARIO 2: Delegation Deadlock (CrewAI-style — FUTILE_ACTION)")
    print("=" * 72)

    config = InterventionConfig(
        nudge_threshold=3,
        override_threshold=5,
        hard_stop_threshold=8,
        window_size=5,
    )
    engine = InterventionEngine(config=config)

    thread_id = "scenario_2"
    node_name = "ResearchAgent"
    state: dict = {}

    start = time.perf_counter()
    interrupted_at = 0
    signal_type = "NONE"
    final_stage = InterventionStage.PASS

    for step in range(1, _RECURSION_LIMIT + 1):
        messages = []
        for i in range(step):
            messages.extend(
                _make_tool_call_messages(
                    tool_name="fetch_pricing",
                    args={"product_id": "X-200"},
                    call_id=f"call_pricing_{i}",
                    ai_content="Let me fetch the pricing for product X-200.",
                    tool_result=json.dumps({"status": "rate_limited", "retry_after": 60}),  # noqa: E501
                )
            )

        decision = engine.process(
            messages=messages,
            state=state,
            thread_id=thread_id,
            node_name=node_name,
        )

        if decision.state_patch:
            state["_tc_intervention"] = decision.state_patch

        final_stage = decision.stage

        if decision.should_terminate:
            interrupted_at = step
            signal_type = ", ".join(s.value for s in decision.signals)
            break

        if decision.stage > InterventionStage.PASS and interrupted_at == 0:
            signal_type = ", ".join(s.value for s in decision.signals) if decision.signals else "UNKNOWN"  # noqa: E501

    if interrupted_at == 0:
        interrupted_at = step  # noqa: F821

    elapsed = time.perf_counter() - start

    total_tokens = interrupted_at * _TOKENS_PER_CALL
    projected_tokens = (_RECURSION_LIMIT - 1) * _TOKENS_PER_CALL
    interrupted_cost = _projected_cost(interrupted_at)
    projected_cost = _projected_cost(_RECURSION_LIMIT - 1)
    margin = projected_cost - interrupted_cost

    result = {
        "scenario": "Delegation Deadlock",
        "framework": "V7 InterventionEngine (simulated CrewAI)",
        "model": _MODEL,
        "interrupted_at": interrupted_at,
        "final_stage": final_stage.name,
        "signal_type": signal_type,
        "tokens_consumed": total_tokens,
        "tokens_saved": projected_tokens - total_tokens,
        "cost_interrupted_usd": round(interrupted_cost, 4),
        "cost_projected_usd": round(projected_cost, 4),
        "margin_saved_usd": round(margin, 4),
        "elapsed_seconds": round(elapsed, 3),
        "pass": final_stage >= InterventionStage.HARD_STOP and interrupted_at < 15,
    }
    return result


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO 3 — Silent State Rot (STATE_STAGNATION)
# ══════════════════════════════════════════════════════════════════════════════

def run_scenario_3() -> dict:
    """Simulates an agent whose tool always returns
    {"summary": null, "status": "processing"}.
    The AI content is near-identical each turn.


    Expected: STATE_STAGNATION or SEMANTIC_STAGNATION detected within 5-8 turns.
    """
    print("=" * 72)
    print("SCENARIO 3: Silent State Rot (STATE_STAGNATION)")
    print("=" * 72)

    config = InterventionConfig(
        nudge_threshold=3,
        override_threshold=5,
        hard_stop_threshold=8,
        window_size=5,
        similarity_threshold=0.92,
    )
    engine = InterventionEngine(config=config)

    thread_id = "scenario_3"
    node_name = "agent"
    state: dict = {}

    start = time.perf_counter()
    interrupted_at = 0
    signal_type = "NONE"
    final_stage = InterventionStage.PASS

    for step in range(1, _RECURSION_LIMIT + 1):
        messages = []
        for i in range(step):
            messages.extend(
                _make_tool_call_messages(
                    tool_name="chunk_and_summarize",
                    args={"text": "Long document content here..."},
                    call_id=f"call_summary_{i}",
                    ai_content="The summary is null, let me retry chunk_and_summarize.",
                    tool_result=json.dumps({"summary": None, "status": "processing"}),
                )
            )

        decision = engine.process(
            messages=messages,
            state=state,
            thread_id=thread_id,
            node_name=node_name,
        )

        if decision.state_patch:
            state["_tc_intervention"] = decision.state_patch

        final_stage = decision.stage

        if decision.should_terminate:
            interrupted_at = step
            signal_type = ", ".join(s.value for s in decision.signals)
            break

        if decision.stage > InterventionStage.PASS and interrupted_at == 0:
            signal_type = ", ".join(s.value for s in decision.signals) if decision.signals else "UNKNOWN"  # noqa: E501

    if interrupted_at == 0:
        interrupted_at = step  # noqa: F821

    elapsed = time.perf_counter() - start

    total_tokens = interrupted_at * _TOKENS_PER_CALL
    projected_tokens = (_RECURSION_LIMIT - 1) * _TOKENS_PER_CALL
    interrupted_cost = _projected_cost(interrupted_at)
    projected_cost = _projected_cost(_RECURSION_LIMIT - 1)
    margin = projected_cost - interrupted_cost

    result = {
        "scenario": "Silent State Rot",
        "framework": "V7 InterventionEngine",
        "model": _MODEL,
        "interrupted_at": interrupted_at,
        "final_stage": final_stage.name,
        "signal_type": signal_type,
        "tokens_consumed": total_tokens,
        "tokens_saved": projected_tokens - total_tokens,
        "cost_interrupted_usd": round(interrupted_cost, 4),
        "cost_projected_usd": round(projected_cost, 4),
        "margin_saved_usd": round(margin, 4),
        "elapsed_seconds": round(elapsed, 3),
        "pass": final_stage >= InterventionStage.HARD_STOP and interrupted_at < 15,
    }
    return result


# ══════════════════════════════════════════════════════════════════════════════
# CONCURRENT STRESS TEST — 50 agents in parallel
# ══════════════════════════════════════════════════════════════════════════════

async def run_concurrent_stress() -> dict:
    """Run 50 independent agents concurrently to verify thread safety and
    that interventions fire correctly under contention."""
    print("=" * 72)
    print("CONCURRENT: 50 agents in parallel")
    print("=" * 72)

    config = InterventionConfig(
        nudge_threshold=3,
        override_threshold=5,
        hard_stop_threshold=8,
        window_size=5,
    )
    engine = InterventionEngine(config=config)

    async def run_agent(agent_id: int) -> tuple[int, bool, str]:
        """Simulate one looping agent."""
        thread_id = f"concurrent_{agent_id}"
        node_name = "agent"
        state: dict = {}

        for step in range(1, 20):
            messages = []
            for i in range(step):
                messages.extend(
                    _make_tool_call_messages(
                        tool_name="search",
                        args={"query": f"query_{agent_id}"},
                        call_id=f"call_{agent_id}_{i}",
                        ai_content=f"Agent {agent_id}: searching for data.",
                        tool_result="No results found.",
                    )
                )

            decision = engine.process(
                messages=messages,
                state=state,
                thread_id=thread_id,
                node_name=node_name,
            )

            if decision.state_patch:
                state["_tc_intervention"] = decision.state_patch

            if decision.should_terminate:
                return agent_id, True, decision.stage.name

            # Small yield to allow concurrency
            await asyncio.sleep(0)

        return agent_id, False, "NO_TERMINATION"

    start = time.perf_counter()
    tasks = [run_agent(i) for i in range(50)]
    agent_results = await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - start

    terminated = sum(1 for _, term, _ in agent_results if term)
    stages = [stage for _, term, stage in agent_results if term]

    result = {
        "scenario": "Concurrent 50 Agents",
        "total_agents": 50,
        "terminated": terminated,
        "termination_stages": dict(zip(*__import__("numpy", fromlist=["unique"]).unique(stages, return_counts=True))) if False else {s: stages.count(s) for s in set(stages)},  # noqa: E501
        "elapsed_seconds": round(elapsed, 3),
        "pass": terminated == 50,
    }
    return result


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

async def main():
    results = []

    s1 = run_scenario_1()
    results.append(s1)
    print(f"  -> {'PASS' if s1['pass'] else 'FAIL'}: interrupted at iter "
          f"{s1['interrupted_at']} ({s1['signal_type']}), "
          f"stage={s1['final_stage']}, margin ${s1['margin_saved_usd']:.4f}")
    print()

    s2 = run_scenario_2()
    results.append(s2)
    print(f"  -> {'PASS' if s2['pass'] else 'FAIL'}: interrupted at iter "
          f"{s2['interrupted_at']} ({s2['signal_type']}), "
          f"stage={s2['final_stage']}, margin ${s2['margin_saved_usd']:.4f}")
    print()

    s3 = run_scenario_3()
    results.append(s3)
    print(f"  -> {'PASS' if s3['pass'] else 'FAIL'}: interrupted at iter "
          f"{s3['interrupted_at']} ({s3['signal_type']}), "
          f"stage={s3['final_stage']}, margin ${s3['margin_saved_usd']:.4f}")
    print()

    s4 = await run_concurrent_stress()
    results.append(s4)
    print(f"  -> {'PASS' if s4['pass'] else 'FAIL'}: {s4['terminated']}/50 agents "
          f"terminated in {s4['elapsed_seconds']}s")
    print()

    # ── Report ──────────────────────────────────────────────────────────────
    all_pass = all(r["pass"] for r in results)

    print("=" * 72)
    print(f"OVERALL: {'ALL PASS ✓' if all_pass else 'SOME FAILURES ✗'}")
    print("=" * 72)

    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
