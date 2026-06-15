.PHONY: install lint typecheck test check format clean

install:
	pip install -e ".[dev,langgraph,openai,crewai]"

lint:
	ruff check src/ tests/

typecheck:
	pyright

test:
	python -m pytest tests/ -v --tb=short --cov=src --cov-config=.coveragerc

test-quick:
	python -m pytest tests/unit/ tests/test_*.py -v --tb=short

format:
	ruff format src/ tests/

check: lint typecheck test

clean:
	rm -rf .mypy_cache/ .pytest_cache/ .ruff_cache/
	rm -rf *.egg-info dist build
	rm -rf .coverage coverage.xml htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
