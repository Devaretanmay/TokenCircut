# TokenCircuit V6.0 — Real World Test Report

## Scenario 1: Cloudflare Wall
- Framework: LangGraph
- Model simulated: gpt-4o
- Interrupted at iteration: 7 / 25
- Signal type: STATE_STAGNATION
- Tokens consumed: 7168 (interrupted) vs ~24576 (projected to recursion limit)
- Real cost: $0.0394 interrupted vs $0.1352 uninterrupted
- Margin saved: $0.0957
- State preserved post-interrupt: YES
- Messages in state: 7
- Elapsed: 0.013s
- Error message: `TokenCircuit [STATE_STAGNATION]: node='fetch_url_tool' at iteration 6 (est. 1024 tokens saved, ~$0.0015)`
- Result: PASS

## Scenario 2: Delegation Deadlock
- Framework: CrewAI (simulated — crewai not installable on Python 3.14)
- Model simulated: gpt-4o
- Interrupted at iteration: 5 / 25
- Signal type: STATE_STAGNATION
- Agent role identified: ResearchAgent
- Repeated tool signature: `fetch_pricing(str)`
- Tokens consumed: 5120 (interrupted) vs ~24576 (projected to recursion limit)
- Real cost: $0.0282 interrupted vs $0.1352 uninterrupted
- Margin saved: $0.107
- Elapsed: 0.0s
- Error message: `TokenCircuit [STATE_STAGNATION]: agent='ResearchAgent' at iteration 5`
- Result: PASS

## Scenario 3: Silent State Rot
- Framework: LangGraph
- Model simulated: gpt-4o
- Interrupted at iteration: 9 / 25
- Signal type: STATE_STAGNATION
- Tokens consumed: 9216 (interrupted) vs ~24576 (projected to recursion limit)
- Real cost: $0.0507 interrupted vs $0.1352 uninterrupted
- Margin saved: $0.0845
- State preserved post-interrupt: YES
- Messages in state: 9
- Elapsed: 0.007s
- Double alert (duplicate signal): NO
- Error message: `TokenCircuit [STATE_STAGNATION]: node='llm' at iteration 5 (est. 1024 tokens saved, ~$0.0015)`
- Result: PASS

## Aggregate
- Total loops intercepted: 3
- Total estimated margin saved: $0.2872
- Any false positives observed: NO
- Any missed detections: NO
- Overall result: ALL PASS
- Recommended threshold adjustments: none
