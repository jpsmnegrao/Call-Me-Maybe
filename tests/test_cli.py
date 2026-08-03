"""Tests for CLI argument parsing.

These tests verify that command-line arguments are parsed correctly and that
invalid command-line inputs are handled cleanly.

CLI validation is tested here. JSON parsing behaviour is tested separately in
json_parser tests. Application exit behaviour is tested separately in main
tests.
"""


import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.cli import CLI, CLIError, parse_cli


DATA = Path("tests/data")


def run_cli_test(
        monkeypatch, functions_file: Path, input_file: Path, output_file: Path
        ) -> CLI:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "call-me-maybe",
            "--functions_definition",
            str(functions_file),
            "--input",
            str(input_file),
            "--output",
            str(output_file),
        ],
    )

    try:
        cli = parse_cli()
        print("\n" + "=" * 70)
        print(f"FUNCTIONS: {functions_file.name}")
        print(f"INPUT:     {input_file.name}")
        print(f"OUTPUT:    {output_file.name}")
        print("MESSAGE: No output produced (expected)")
        print("=" * 70)
        return cli
    except Exception as error:
        print("\n" + "=" * 70)
        print(f"FUNCTIONS: {functions_file.name}")
        print(f"INPUT:     {input_file.name}")
        print(f"OUTPUT:    {output_file.name}")
        print(f"ERROR TYPE: {type(error).__name__}")
        print("MESSAGE:")
        print(error)
        print("=" * 70)
        raise


def test_parse_cli_valid_arguments(monkeypatch) -> None:
    """Valid arguments should return a CLI object."""
    cli = run_cli_test(
        monkeypatch,
        [
            "--functions_definition",
            str(DATA / "functions_definition.json"),
            "--input",
            str(DATA / "function_calling_tests.json"),
            "--output",
            str(DATA / "output.json"),
        ],
    )
    assert isinstance(cli, CLI)
    assert cli.functions == DATA / "functions_definition.json"
    assert cli.input == DATA / "function_calling_tests.json"
    assert cli.output == DATA / "output.json"


def test_parse_cli_missing_functions_definition(monkeypatch) -> None:
    """Missing functions definition argument should raise CLIError."""
    with pytest.raises(CLIError):
        run_cli_test(
            monkeypatch,
            [
                "--functions_definition",
                str(DATA / "functions_definition.json"),
                "--input",
                str(DATA / "function_calling_tests.json"),
                "--output",
                str(DATA / "output.json"),
            ],
        )


def test_parse_cli_missing_input(monkeypatch) -> None:
    """Missing input argument should raise CLIError."""
    with pytest.raises(CLIError):
        run_cli_test(
            monkeypatch,
            [
                "--functions_definition",
                str(DATA / "functions_definition.json"),
                "--output",
                str(DATA / "output.json"),
            ],
        )


def test_parse_cli_missing_output(monkeypatch) -> None:
    """Missing output argument should raise CLIError."""
    with pytest.raises(CLIError):
        run_cli_test(
            monkeypatch,
            [
                "--functions_definition",
                str(DATA / "functions_definition.json"),
                "--input",
                str(DATA / "function_calling_tests.json"),
            ],
        )


def test_parse_cli_functions_file_not_found(monkeypatch) -> None:
    """Missing functions file should fail CLI validation."""
    with pytest.raises(ValidationError):
        run_cli_test(
            monkeypatch,
            [
                "--functions_definition",
                str(DATA / "functions_definition_no_file.json"),
                "--input",
                str(DATA / "function_calling_tests.json"),
                "--output",
                str(DATA / "output.json"),
            ],
        )


def test_parse_cli_input_file_not_found(monkeypatch) -> None:
    """Missing input file should fail CLI validation."""
    with pytest.raises(ValidationError):
        run_cli_test(
            monkeypatch,
            [
                "--functions_definition",
                str(DATA / "functions_definition.json"),
                "--input",
                str(DATA / "function_calling_tests_no_file.json"),
                "--output",
                str(DATA / "output.json"),
            ],
        )


def test_parse_cli_output_file_can_be_missing(monkeypatch) -> None:
    """Output path can be missing because it uses Path, not FilePath."""
    output = DATA / "new_output.json"
    cli = run_cli_test(
        monkeypatch,
        [
            "--functions_definition",
            str(DATA / "functions_definition.json"),
            "--input",
            str(DATA / "function_calling_tests.json"),
            "--output",
            str(output),
        ],
    )
    assert cli.output == output
