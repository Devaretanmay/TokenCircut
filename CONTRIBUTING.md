# Contributing

## Setup

```bash
git clone https://github.com/Devaretanmay/TokenCircut
cd TokenCircut
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,langgraph,openai,otel]"
```

## Development

```bash
# Run all checks
make check

# Format code
make format

# Run tests
make test

# Type check
make typecheck

# Lint
make lint
```

## Project Structure

```
src/tokencircuit/           # Library source
├── __init__.py             # Public API & instrument_langgraph/crewai
├── engine.py               # InterventionEngine — central orchestrator
├── types.py                # Core types (enums, Pydantic models, CanonicalMessage)
├── config.py               # Remote configuration loading
├── canonicalizer.py        # MessageCanonicalizer — normalizes message formats
├── ledger.py               # ToolTransactionLedger — tracks tool call lifecycle
├── validator.py            # TranscriptValidator — enforces 10 invariants
├── semantic_detector.py    # SemanticStagnationDetector — n-gram Jaccard
├── state_schema.py         # _tc_intervention state channel & reducer
├── telemetry.py            # OpenTelemetry integration & cost estimation
├── exceptions.py           # TokenCircuitError hierarchy
├── adapters/
│   ├── langgraph.py        # LangGraphPreModelAdapter (pre_model_hook)
│   ├── crewai.py           # CrewAIInterventionAdapter (step_callback)
│   └── wrapper.py          # ModelNodeWrapper (fallback for custom graphs)
├── otel/
│   └── hash_utils.py       # State & action fingerprinting utilities
└── clients/                # OpenAI client wrapper (future)

tests/                      # Test suite
├── unit/                   # Unit tests for each module
├── integration/            # End-to-end integration tests
├── performance/            # Benchmarks & stress tests
└── security/               # Security edge function tests
```

## Code Style

- Ruff with default rules
- Line length: 88
- Python 3.11+ type annotations
- No comments on obvious code
- `from __future__ import annotations` in all modules

## Pull Request Process

1. Open an issue first (bug or feature)
2. Fork and create a feature branch
3. Add tests for new behavior
4. Run `make check` before committing
5. Update CHANGELOG.md
6. Submit PR

## Release Process

Maintainers cut releases via GitHub Releases. Publishing to PyPI is automated.
