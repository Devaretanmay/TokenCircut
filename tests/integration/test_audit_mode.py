import pytest

pytest.importorskip("langgraph")

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from tokencircuit import InterventionConfig
from tokencircuit.state_schema import InterventionStateSchema, tc_state_reducer


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    _tc_intervention: Annotated[InterventionStateSchema, tc_state_reducer]

async def stub_llm(state: AgentState):
    msg = AIMessage(
        content="",
        tool_calls=[{"name": "fetch_weather", "args": {"location": "San Francisco"}, "id": "call_1", "type": "tool_call"}]  # noqa: E501
    )
    return {"messages": [msg]}

async def fetch_weather_tool(state: AgentState):
    last_message = state["messages"][-1]
    responses = [ToolMessage(content="Error: Timeout.", name=tc["name"], tool_call_id=tc["id"]) for tc in last_message.tool_calls]  # noqa: E501
    return {"messages": responses}

def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and not last_message.tool_calls:
        return END
    if isinstance(last_message, ToolMessage):
        return "llm"
    return "tool"

@pytest.mark.asyncio
async def test_audit_mode_prevents_intervention():
    from tokencircuit.adapters.langgraph import LangGraphPreModelAdapter

    config = InterventionConfig(
        nudge_threshold=1,
        override_threshold=2,
        hard_stop_threshold=3,
        window_size=5,
        audit_mode=True,
    )
    adapter = LangGraphPreModelAdapter(config=config)

    builder = StateGraph(AgentState)
    builder.add_node("llm", stub_llm, pre_model_hook=adapter.create_hook(node_name="llm"))  # noqa: E501
    builder.add_node("tool", fetch_weather_tool)
    builder.add_edge(START, "llm")
    builder.add_conditional_edges("llm", should_continue)
    builder.add_edge("tool", "llm")

    graph = builder.compile()

    state = {"messages": [HumanMessage(content="What is the weather?")], "_tc_intervention": {}}  # noqa: E501
    run_config = {"configurable": {"thread_id": "audit_1"}}

    turns = 0
    try:
        async for chunk in graph.astream(state, config=run_config):
            if "llm" in chunk:
                turns += 1
            if turns >= 10:
                break
    except Exception as e:
        pytest.fail(f"Audit mode failed to prevent exception: {e}")

    assert turns >= 10, "Agent did not loop as expected, suggesting intervention mutated the flow"  # noqa: E501
