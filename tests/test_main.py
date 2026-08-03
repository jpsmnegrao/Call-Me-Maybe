"""Tests for the main application entry point.

These tests verify that main() handles expected runtime errors from the input
parsers cleanly. Parser behavior itself (such as invalid JSON, schema
validation failures, empty input, and invalid data structures) is covered
separately in parser tests.

The purpose of these tests is to ensure that when those errors occur, main()
catches them and exits cleanly using SystemExit instead of allowing uncaught
exceptions to propagate and produce a traceback.

Test cases use the real test fixture files to trigger the parser failures
through the normal application flow."""


import sys
from pathlib import Path
import pytest
from src.main import main


DATA = Path("tests/data")


def run_main_with_files(
        monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
        input_file: Path, functions_file: Path, label: str) -> None:
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
            str(DATA / "output.json"),
        ],
    )
    try:
        main()
    except SystemExit:
        output = capsys.readouterr().out.strip()
        print("\n" + "=" * 70)
        print(f"TEST: MAIN {label}")
        print(f"INPUT: {input_file.name}")
        print(f"FUNCTIONS: {functions_file.name}")
        if output:
            print("MESSAGE:")
            print(output)
        else:
            print("MESSAGE: No output produced (expected)")
        print("=" * 70)
        raise


def test_main_clean_exit_on_prompt_file_not_found(
        monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
        ) -> None:
    """mambo"""
    with pytest.raises(SystemExit):
        run_main_with_files(
            monkeypatch,
            capsys,
            DATA / "function_calling_tests_no_file.json",
            DATA / "functions_definition.json",
            "prompt_file_not_found",
        )


def test_main_clean_exit_on_prompt_invalid_syntax(
        monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
        ) -> None:
    """mambo"""
    with pytest.raises(SystemExit):
        run_main_with_files(
            monkeypatch,
            capsys,
            DATA / "function_calling_tests_invalid_syntax.json",
            DATA / "functions_definition.json",
            "prompt_invalid_syntax",
        )


def test_main_clean_exit_on_prompt_validation_error(
        monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
        ) -> None:
    """mambo"""
    with pytest.raises(SystemExit):
        run_main_with_files(
            monkeypatch,
            capsys,
            DATA / "function_calling_tests_wrong_key.json",
            DATA / "functions_definition.json",
            "prompt_validation_error",
        )


def test_main_clean_exit_on_prompt_empty_file(
        monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
        ) -> None:
    """mambo"""
    with pytest.raises(SystemExit):
        run_main_with_files(
            monkeypatch,
            capsys,
            DATA / "function_calling_tests_empty_array.json",
            DATA / "functions_definition.json",
            "prompt_empty_file",
        )


def test_main_clean_exit_on_prompt_not_array(
        monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
        ) -> None:
    """mambo"""
    with pytest.raises(SystemExit):
        run_main_with_files(
            monkeypatch,
            capsys,
            DATA / "function_calling_tests_not_array.json",
            DATA / "functions_definition.json",
            "prompt_not_array",
        )


def test_main_clean_exit_on_function_file_not_found(
        monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
        ) -> None:
    """mambo"""
    with pytest.raises(SystemExit):
        run_main_with_files(
            monkeypatch,
            capsys,
            DATA / "function_calling_tests.json",
            DATA / "functions_definition_no_file.json",
            "function_file_not_found",
        )


def test_main_clean_exit_on_function_invalid_syntax(
        monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
        ) -> None:
    """mambo"""
    with pytest.raises(SystemExit):
        run_main_with_files(
            monkeypatch,
            capsys,
            DATA / "function_calling_tests.json",
            DATA / "functions_definition_invalid_syntax.json",
            "function_invalid_syntax",
        )


def test_main_clean_exit_on_function_validation_error(
        monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
        ) -> None:
    """mambo"""
    with pytest.raises(SystemExit):
        run_main_with_files(
            monkeypatch,
            capsys,
            DATA / "function_calling_tests.json",
            DATA / "functions_definition_wrong_key.json",
            "function_validation_error",
        )


def test_main_clean_exit_on_function_empty_file(
        monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
        ) -> None:
    """mambo"""
    with pytest.raises(SystemExit):
        run_main_with_files(
            monkeypatch,
            capsys,
            DATA / "function_calling_tests.json",
            DATA / "functions_definition_empty_array.json",
            "function_empty_file",
        )


def test_main_clean_exit_on_function_not_array(
        monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
        ) -> None:
    """mambo"""
    with pytest.raises(SystemExit):
        run_main_with_files(
            monkeypatch,
            capsys,
            DATA / "function_calling_tests.json",
            DATA / "functions_definition_not_array.json",
            "function_not_array",
        )
