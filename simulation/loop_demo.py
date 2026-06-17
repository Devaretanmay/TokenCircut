#!/usr/bin/env python3
"""TokenCircuit V7 — Progressive Intervention demo with rich terminal output."""

from __future__ import annotations

import time
import sys
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.align import Align
from rich import box

from tokencircuit.adapters.langgraph import tc_pre_model_hook, tc_wrap_tool_call
from tokencircuit.engine import InterventionConfig, InterventionEngine
from tokencircuit.state_schema import InterventionStateSchema, default_intervention_state, tc_state_reducer
from tokencircuit.types import InterventionStage

console = Console()

# ── Step timing ──────────────────────────────────────────────────────
STEP_DELAY = 1.5

# ── Mock tool ────────────────────────────────────────────────────────

@tool
def fetch_secure_data(query: str) -> str:
    """Fetches secure data — always fails."""
    return "Error: 403 Forbidden. Invalid headers."

# ── Mock LLM ─────────────────────────────────────────────────────────

class MockModel:
    call_count = 0
    _tool_name = "fetch_secure_data"
    _text_response = "I cannot bypass the 403. I will return a summary of the failure instead."
    _bind_tools_called = False

    def bind_tools(self, tools, **kwargs):
        self._bind_tools_called = True
        return self

    def invoke(self, messages, config=None, **kwargs):
        self.call_count += 1
        for m in messages:
            if hasattr(m, "content"):
                c = str(m.content) if m.content else ""
            elif isinstance(m, dict):
                c = str(m.get("content", ""))
            else:
                c = ""
            if "SYSTEM DIRECTIVE" in c:
                return AIMessage(content=self._text_response)
        return AIMessage(
            content="",
            tool_calls=[{"name": self._tool_name, "args": {"query": "x"}, "id": f"call_{self.call_count}", "type": "tool_call"}],
        )

# ── State ────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    _tc_intervention: Annotated[InterventionStateSchema, tc_state_reducer]

# ── TokenCircuit Engine ──────────────────────────────────────────────

engine = InterventionEngine(config=InterventionConfig(
    nudge_threshold=2,
    override_threshold=3,
    hard_stop_threshold=5,
    enable_semantic_detection=True,
    window_size=5,
))

# ── Build Graph ──────────────────────────────────────────────────────

model = MockModel()
tool_node = ToolNode([fetch_secure_data], wrap_tool_call=tc_wrap_tool_call(engine.get_thread_ledger))

def call_model(state):
    hook_out = tc_pre_model_hook(state, engine=engine, node_name="agent")
    msgs = hook_out.get("llm_input_messages") or state["messages"]
    response = model.invoke(msgs)
    decision = engine.pop_last_decision("default_thread")
    patch = decision.state_patch if decision else {}
    return {"messages": [response], "_tc_intervention": patch}

builder = StateGraph(AgentState)
builder.add_node("agent", call_model)
builder.add_node("tools", tool_node)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
builder.add_edge("tools", "agent")
graph = builder.compile()

# ── Header ───────────────────────────────────────────────────────────

def show_header():
    console.clear()
    title = Text()
    title.append("  TokenCircuit V7  ", style="bold yellow")
    title.append("Progressive Intervention", style="bold white")
    panel = Panel(title, box=box.DOUBLE_EDGE, border_style="cyan")
    console.print(Align.center(panel))
    console.print()
    console.print(Align.center("[dim]Simulating a LangGraph agent stuck in a 403 loop...[/dim]"))
    console.print()

def show_agent_turn(turn):
    text = Text()
    text.append(f"  iteration #{turn}  ", style="bold white on blue")
    console.print(text)

def show_tool_call():
    panel = Panel(
        "[bold red]fetch_secure_data[/bold red] → [red]Error: 403 Forbidden. Invalid headers.[/red]",
        border_style="red",
    )
    console.print(panel)

def show_nudge():
    panel = Panel(
        "[bold yellow]TokenCircuit NUDGE[/bold yellow]\n"
        "[yellow]Coaching message sent — encouraging strategy shift[/yellow]",
        border_style="yellow",
    )
    console.print(panel)

def show_override():
    panel = Panel(
        "[bold red]TokenCircuit OVERRIDE[/bold red]\n"
        "[red]SYSTEM DIRECTIVE injected — stripping 3 failed transactions[/red]",
        border_style="bright_red",
    )
    console.print(panel)

def show_pivot():
    panel = Panel(
        "[bold green]Agent pivoted[/bold green]\n"
        "[green]\"I cannot bypass the 403. I will return a summary of the failure instead.\"[/green]",
        border_style="green",
    )
    console.print(panel)

def show_summary(tc_state):
    console.print()
    final_stage = tc_state.get("current_stage", "pass")
    stage_colors = {"pass": "green", "nudge": "yellow", "override": "red", "hard_stop": "red"}
    color = stage_colors.get(final_stage, "white")
    summary = Panel(
        f"[bold]Intervention Stage:[/bold] [{color}]{final_stage.upper()}[/{color}]\n"
        f"[bold]Turn Counter:[/bold] {tc_state.get('turn_counter', '?')}\n"
        f"[bold]Total Interventions:[/bold] {tc_state.get('total_interventions', 0)}\n"
        f"[bold]Consecutive Stagnation:[/bold] {tc_state.get('consecutive_stagnation_count', 0)}",
        title="Summary", border_style=color, box=box.ROUNDED,
    )
    console.print(summary)

# ── Run Demo ─────────────────────────────────────────────────────────

def main():
    show_header()
    time.sleep(1)

    result = graph.invoke(
        {
            "messages": [
                SystemMessage(content="You are a helpful assistant."),
                HumanMessage(content="Fetch the secure data."),
            ],
            "_tc_intervention": default_intervention_state(),
        },
        {"recursion_limit": 10},
    )

    msgs = result["messages"]
    tc_state = result.get("_tc_intervention", default_intervention_state())

    turn = 0
    nudge_shown = False
    override_shown = False
    pivoted = False

    for msg in msgs:
        if isinstance(msg, AIMessage):
            if msg.tool_calls:
                turn += 1
                # Show intervention BEFORE the agent action on that turn
                if turn == 3 and not nudge_shown:
                    nudge_shown = True
                    console.print()
                    show_nudge()
                    time.sleep(STEP_DELAY)
                if turn == 4 and not override_shown:
                    override_shown = True
                    show_override()
                    time.sleep(STEP_DELAY)
                show_agent_turn(turn)
                time.sleep(0.5)
                console.print(f"  [bold cyan]Agent:[/bold cyan] Calling fetch_secure_data...")
                time.sleep(0.3)
            else:
                turn += 1
                if not override_shown:
                    override_shown = True
                    show_override()
                    time.sleep(STEP_DELAY)
                if not nudge_shown:
                    nudge_shown = True
                    show_nudge()
                    time.sleep(STEP_DELAY)
                pivoted = True
                show_agent_turn(turn)
                time.sleep(0.5)
                console.print(f"  [bold cyan]Agent:[/bold cyan] {msg.content}")
                time.sleep(0.3)
        elif isinstance(msg, ToolMessage):
            show_tool_call()
            time.sleep(0.5)

    if pivoted:
        show_pivot()
        time.sleep(STEP_DELAY)

    console.print()
    console.print(Align.center("[bold green]Agent loop terminated — TokenCircuit active.[/bold green]"))
    console.print()
    show_summary(tc_state)

    final_msg = msgs[-1]
    assert isinstance(final_msg, AIMessage), f"Expected AIMessage, got {type(final_msg).__name__}"
    assert not getattr(final_msg, "tool_calls", None), "Model pivoted"
    sys.exit(0)

if __name__ == "__main__":
    main()
