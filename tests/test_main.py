"""mambo jambo"""


import json
import pytest
from pydantic import ValidationError
from schemas import Prompt, FunctionDefinition
from main import main


def test_main_clean_exit_on_prompt_file_not_found(monkeypatch) -> None:
    """mambo"""
    def fake_prompt_parser(path):
        raise FileNotFoundError("simulated missing file")

    monkeypatch.setattr("main.prompt_parser", fake_prompt_parser)

    with pytest.raises(SystemExit):
        main()


def test_main_clean_exit_on_prompt_invalid_syntax(monkeypatch) -> None:
    """mambo"""
    def fake_prompt_parser(path):
        raise json.JSONDecodeError("simulated bad syntax", "doc", 0)

    monkeypatch.setattr("main.prompt_parser", fake_prompt_parser)

    with pytest.raises(SystemExit):
        main()


def test_main_clean_exit_on_prompt_validation_error(monkeypatch) -> None:
    """mambo"""
    def fake_prompt_parser(path):
        try:
            Prompt()
        except ValidationError as e:
            raise e

    monkeypatch.setattr("main.prompt_parser", fake_prompt_parser)

    with pytest.raises(SystemExit):
        main()


def test_main_clean_exit_on_prompt_empty_file(monkeypatch) -> None:
    """mambo"""
    def fake_prompt_parser(path):
        raise ValueError("simulated empty file")

    monkeypatch.setattr("main.prompt_parser", fake_prompt_parser)

    with pytest.raises(SystemExit):
        main()


def test_main_clean_exit_on_prompt_not_array(monkeypatch) -> None:
    """mambo"""
    def fake_prompt_parser(path):
        raise ValueError("simulated not-array data")

    monkeypatch.setattr("main.prompt_parser", fake_prompt_parser)

    with pytest.raises(SystemExit):
        main()


def test_main_clean_exit_on_function_file_not_found(monkeypatch) -> None:
    """mambo"""
    def fake_function_parser(path):
        raise FileNotFoundError("simulated missing file")

    monkeypatch.setattr("main.function_parser", fake_function_parser)

    with pytest.raises(SystemExit):
        main()


def test_main_clean_exit_on_function_invalid_syntax(monkeypatch) -> None:
    """mambo"""
    def fake_function_parser(path):
        raise json.JSONDecodeError("simulated bad syntax", "doc", 0)

    monkeypatch.setattr("main.function_parser", fake_function_parser)

    with pytest.raises(SystemExit):
        main()


def test_main_clean_exit_on_function_validation_error(monkeypatch) -> None:
    """mambo"""
    def fake_function_parser(path):
        try:
            FunctionDefinition()
        except ValidationError as e:
            raise e

    monkeypatch.setattr("main.function_parser", fake_function_parser)

    with pytest.raises(SystemExit):
        main()


def test_main_clean_exit_on_function_empty_file(monkeypatch) -> None:
    """mambo"""
    def fake_function_parser(path):
        raise ValueError("simulated empty file")

    monkeypatch.setattr("main.function_parser", fake_function_parser)

    with pytest.raises(SystemExit):
        main()


def test_main_clean_exit_on_function_not_array(monkeypatch) -> None:
    """mambo"""
    def fake_function_parser(path):
        raise ValueError("simulated not-array data")

    monkeypatch.setattr("main.function_parser", fake_function_parser)

    with pytest.raises(SystemExit):
        main()
