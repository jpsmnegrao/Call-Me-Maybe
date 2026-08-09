.PHONY: install run debug clean lint lint-strict fclean test test-silent,
		test-verbose, test-errors

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
	@rm -rf .venv
	@rm -rf data/output
	@echo "Removed virtual environment and output files - deep clean complete!"

test:
	uv run pytest

test-silent:
	uv run pytest -q --tb=no

test-verbose:
	uv run pytest -v

test-errors:
	uv run pytest -s -q