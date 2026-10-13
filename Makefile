.PHONY: install run debug clean lint lint-strict fclean test test-silent
		test-verbose test-errors

export HF_HOME := $(CURDIR)/.hf-cache
export TORCH_HOME := $(CURDIR)/.torch-cache

all: install

install:
	uv sync
	@echo "=== Installation complete! ==="

run:
	uv run python -m src \
		--functions_definition	data/input/functions_definition.json \
		--input	data/input/function_calling_tests.json	\
		--output data/output/function_calling_results.json

debug:
	uv run python -m pdb -m src -- \
		--functions_definition	data/input/functions_definition.json \
		--input	data/input/function_calling_tests.json	\
		--output data/output/function_calling_results.json

clean:
	@rm -rf .mypy_cache .pytest_cache/ ; \
	find . -type d -name "__pycache__" -exec rm -rf {} + ; \
	echo "Cleanup complete!"; \

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores \
		--ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 .
	uv run mypy . --strict

fclean: clean
	@rm -rf .venv .hf-cache/ .uv-cache/
	@rm -rf data/output
	@echo "Removed virtual environment and output files - deep clean complete!"

# Run all tests normally
test:
	uv run pytest

# Run tests quietly without tracebacks
test-silent:
	uv run pytest -q --tb=no

# Run tests with verbose test names and output
test-verbose:
	uv run pytest -v

# Run tests while showing custom diagnostic output
test-errors:
	uv run pytest -s -q