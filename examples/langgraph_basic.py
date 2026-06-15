"""Basic LangGraph loop detection example."""

from tokencircuit import TokenCircuitConfig, instrument_langgraph

# Build your LangGraph graph
# graph = ...

config = TokenCircuitConfig(max_repeats=5, window_size=5)
safe = instrument_langgraph(graph, config=config)

input = {"messages": [{"role": "user", "content": "Hello"}]}
async for step in safe.astream(input, {"recursion_limit": 50}):
    print(step)
