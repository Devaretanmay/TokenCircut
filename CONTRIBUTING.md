# Contributing

## Setup

```bash
git clone https://github.com/your-org/tokencircuit
cd tokencircuit
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,langgraph,openai]"
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

## Project structure

```
src/tokencircuit/       # Library source
├── config.py           # Configuration
├── detectors/          # Loop detection logic
├── interceptors/       # Framework wrappers
├── clients/            # OpenAI client wrapper
├── otel/               # Hashing utilities
├── ring_buffer.py      # Sliding window
├── telemetry.py        # Cost tracking
└── exceptions.py       # Error types

tests/                  # Test suite
├── unit/               # Unit tests
├── integration/        # Integration tests
├── performance/        # Benchmarks
└── security/           # Security tests

control-plane/          # Next.js dashboard (optional)
```

## Code style

- Ruff with default rules
- Line length: 88
- Python 3.11+ type annotations
- No comments on obvious code

## Pull request process

1. Open an issue first (bug or feature)
2. Fork and create a feature branch
3. Add tests for new behavior
4. Run `make check` before committing
5. Update CHANGELOG.md
6. Submit PR

## Release process

Maintainers cut releases via GitHub Releases. Publishing to PyPI is automated.
