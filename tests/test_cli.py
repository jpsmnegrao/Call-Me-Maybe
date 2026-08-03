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
        monkeypatch: pytest.MonkeyPatch, args: list[str], label: str) -> CLI:
    monkeypatch.setattr(sys, "argv", ["call-me-maybe"] + args)
    try:
        cli = parse_cli()
        print("\n" + "=" * 70)
        print(f"TEST: {label}")
        print(f"ARGS: {args}")
        print("MESSAGE: No output produced (expected)")
        print("=" * 70)
        return cli
    except Exception as error:
        print("\n" + "=" * 70)
        print(f"TEST: {label}")
        print(f"ARGS: {args}")
        print(f"ERROR TYPE: {type(error).__name__}")
        print("MESSAGE:")
        print(error)
        print("=" * 70)
        raise


def test_parse_cli_valid_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    """Valid arguments should return a CLI object."""
    output = DATA / "output.json"
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
        "valid_arguments",
    )
    assert isinstance(cli, CLI)
    assert cli.functions == DATA / "functions_definition.json"
    assert cli.input == DATA / "function_calling_tests.json"
    assert cli.output == DATA / "output.json"


def test_parse_cli_missing_functions_definition_uses_default(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Omitting --functions_definition should fall back to its default."""
    cli = run_cli_test(
        monkeypatch,
        [
            "--input",
            str(DATA / "function_calling_tests.json"),
            "--output",
            str(DATA / "output.json"),
        ],
        "missing_functions_definition_uses_default",
    )
    assert str(cli.functions) == "data/input/functions_definition.json"


def test_parse_cli_missing_input_uses_default(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Omitting --input should fall back to its default."""
    cli = run_cli_test(
        monkeypatch,
        [
            "--functions_definition",
            str(DATA / "functions_definition.json"),
            "--output",
            str(DATA / "output.json"),
        ],
        "missing_input_uses_default",
    )
    assert str(cli.input) == "data/input/function_calling_tests.json"


def test_parse_cli_missing_output_uses_default(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Omitting --output should fall back to its default."""
    cli = run_cli_test(
        monkeypatch,
        [
            "--functions_definition",
            str(DATA / "functions_definition.json"),
            "--input",
            str(DATA / "function_calling_tests.json"),
        ],
        "missing_output_uses_default",
    )
    assert str(cli.output) == "data/output/function_calling_results.json"


def test_parse_cli_functions_file_not_found(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Nonexistent functions file should fail CLI validation."""
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
            "functions_file_not_found",
        )


def test_parse_cli_input_file_not_found(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Nonexistent input file should fail CLI validation."""
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
            "input_file_not_found",
        )


def test_parse_cli_output_file_can_be_missing(
        monkeypatch: pytest.MonkeyPatch) -> None:
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
        "output_file_can_be_missing",
    )
    assert cli.output == output


def test_parse_cli_unrecognized_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unrecognized flag should raise CLIError, not exit the process."""
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
                "--nonexistent_flag",
                "foo",
            ],
            "unrecognized_flag",
        )
