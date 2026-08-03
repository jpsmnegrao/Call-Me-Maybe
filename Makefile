.PHONY: install run debug clean lint lint-strict test

all: install

install:
	uv sync
	@echo "=== Installation complete! ==="

run:
	uv run python -m src --functions_definition	\
	data/input/functions_definition.json --input	\
	data/input/function_calling_tests.json	\
	--output data/output/function_calls.json

debug:
	uv run python -m pdb -m src -- --functions_definition	\
	data/input/functions_definition.json --input	\
	data/input/function_calling_tests.json	\
	--output data/output/function_calls.json

clean:
	@rm -rf .mypy_cache .pytest_cache/ ; \
	find . -type d -name "__pycache__" -exec rm -rf {} + ; \
	echo "Cleanup complete!"; \

fclean: clean
	@rm -rf .venv
	@echo "Removed virtual environment - deep clean complete!"

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores \
		--ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 .
	uv run mypy . --strict

test:
	uv run pytest -v

test-silent:
	uv run pytest -q --tb=no -s