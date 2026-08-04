"""mambo jambo"""

import pytest
import json

from pathlib import Path
from collections.abc import Callable
from pydantic import ValidationError
from typing import cast, Any

from src.schemas import Prompt, FunctionDefinition
from src.json_parser import prompt_parser, function_parser


DATA = Path("tests/data")


def run_parser_test(
        parser: Callable[[Path], list[Prompt] | list[FunctionDefinition]],
        file_path: Path, label: str
        ) -> list[Prompt] | list[FunctionDefinition]:
    try:
        result = parser(file_path)
        print("\n" + "=" * 70)
        print(f"TEST: PARSER {label}")
        print(f"FILE: {file_path.name}")
        print("MESSAGE: No output produced (expected)")
        print("=" * 70)
        return result
    except Exception as error:
        print("\n" + "=" * 70)
        print(f"TEST: PARSER {label}")
        print(f"FILE: {file_path.name}")
        print(f"ERROR TYPE: {type(error).__name__}")
        print("MESSAGE:")
        print(error)
        print("=" * 70)
        raise


# PROMPT TESTING


def test_prompt_parser_valid() -> None:
    """amambo jambo"""
    with open("tests/data/function_calling_tests.json", 'r') as file:
        raw_prompts: list[dict[str, Any]] = json.load(file)
    prompts = cast(
        list[Prompt],
        run_parser_test(
            prompt_parser,
            DATA / "function_calling_tests.json",
            "valid",
        )
    )
    assert len(raw_prompts) == len(prompts)
    for prompt in prompts:
        assert isinstance(prompt, Prompt)
    assert all(
        parsed.prompt == raw["prompt"]
        for parsed, raw in zip(prompts, raw_prompts)
    )


def test_prompt_parser_file_not_found() -> None:
    """jambo jambo"""
    with pytest.raises(FileNotFoundError):
        run_parser_test(
            prompt_parser,
            DATA / "function_calling_tests_no_file.json",
            "file_not_found",
        )


def test_prompt_parser_invalid_syntax() -> None:
    """jambo jambo"""
    with pytest.raises(json.JSONDecodeError):
        run_parser_test(
            prompt_parser,
            DATA / "function_calling_tests_invalid_syntax.json",
            "invalid_syntax",
        )


def test_prompt_parser_empty_array() -> None:
    """jambo jambo"""
    with pytest.raises(ValueError):
        run_parser_test(
            prompt_parser,
            DATA / "function_calling_tests_empty_array.json",
            "empty_array",
        )


def test_prompt_parser_not_array() -> None:
    """jambo jambo"""
    with pytest.raises(ValueError):
        run_parser_test(
            prompt_parser,
            DATA / "function_calling_tests_not_array.json",
            "not_array",
        )


def test_prompt_parser_empty_key_value() -> None:
    """jambo jambo"""
    with pytest.raises(ValidationError):
        run_parser_test(
            prompt_parser,
            DATA / "function_calling_tests_empty_key_value.json",
            "empty_key_value",
        )


def test_prompt_parser_wrong_key() -> None:
    """jambo jambo"""
    with pytest.raises(ValidationError):
        run_parser_test(
            prompt_parser,
            DATA / "function_calling_tests_wrong_key.json",
            "wrong_key",
        )


def test_prompt_parser_invalid_data_type() -> None:
    """jambo jambo"""
    with pytest.raises(ValidationError):
        run_parser_test(
            prompt_parser,
            DATA / "function_calling_tests_invalid_prompt_data_type.json",
            "invalid_data_type",
        )


def test_prompt_parser_extra_key() -> None:
    """jambo jambo"""
    with pytest.raises(ValidationError):
        run_parser_test(
            prompt_parser,
            DATA / "function_calling_tests_extra_key.json",
            "extra_key",
        )


# FUNCTION DEFINITION TESTING


def test_function_parser_valid() -> None:
    """amambo jambo"""
    with open("tests/data/functions_definition.json", 'r') as file:
        raw_functions: list[dict[str, Any]] = json.load(file)
    functions = cast(
        list[FunctionDefinition],
        run_parser_test(
            function_parser,
            DATA / "functions_definition.json",
            "valid",
        )
    )
    assert len(raw_functions) == len(functions)
    for function in functions:
        assert isinstance(function, FunctionDefinition)
    assert all(
        parsed.name == raw["name"]
        for parsed, raw in zip(functions, raw_functions)
    )


def test_function_parser_file_not_found() -> None:
    """jambo jambo"""
    with pytest.raises(FileNotFoundError):
        run_parser_test(
            function_parser,
            DATA / "functions_definition_no_file.json",
            "file_not_found",
        )


def test_function_parser_invalid_syntax() -> None:
    """jambo jambo"""
    with pytest.raises(json.JSONDecodeError):
        run_parser_test(
            function_parser,
            DATA / "functions_definition_invalid_syntax.json",
            "invalid_syntax",
        )


def test_function_parser_empty_array() -> None:
    """jambo jambo"""
    with pytest.raises(ValueError):
        run_parser_test(
            function_parser,
            DATA / "functions_definition_empty_array.json",
            "empty_array",
        )


def test_function_parser_not_array() -> None:
    """jambo jambo"""
    with pytest.raises(ValueError):
        run_parser_test(
            function_parser,
            DATA / "functions_definition_not_array.json",
            "not_array",
        )


def test_function_parser_wrong_key() -> None:
    """jambo jambo"""
    with pytest.raises(ValidationError):
        run_parser_test(
            function_parser,
            DATA / "functions_definition_wrong_key.json",
            "wrong_key",
        )


def test_function_parser_invalid_data_type() -> None:
    """jambo jambo"""
    with pytest.raises(ValidationError):
        run_parser_test(
            function_parser,
            DATA / "functions_definition_invalid_function_data_type.json",
            "invalid_data_type",
        )


def test_function_parser_extra_key() -> None:
    """jambo jambo"""
    with pytest.raises(ValidationError):
        run_parser_test(
            function_parser,
            DATA / "functions_definition_extra_key.json",
            "extra_key",
        )


def test_function_parser_extra_nested_key() -> None:
    """jambo jambo"""
    with pytest.raises(ValidationError):
        run_parser_test(
            function_parser,
            DATA / "functions_definition_extra_nested_key.json",
            "extra_nested_key",
        )


def test_function_parser_empty_parameters() -> None:
    """jambo jambo"""
    functions = cast(
        list[FunctionDefinition],
        run_parser_test(
            function_parser,
            DATA / "functions_definition_empty_parameters.json",
            "empty_parameters",
        )
    )
    assert isinstance(functions, list)
    assert len(functions) > 0
    assert any(function.parameters == {} for function in functions)


VALID_FUNCTION_DEFINITION: dict[str, Any] = {
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": {
        "a": {"type": "number"},
        "b": {"type": "number"},
    },
    "returns": {"type": "number"},
}


@pytest.mark.parametrize(
    "missing_field", list(VALID_FUNCTION_DEFINITION.keys()))
def test_function_definition_missing_required_field(
        missing_field: str) -> None:
    """jambo jambo"""
    incomplete = VALID_FUNCTION_DEFINITION.copy()
    del incomplete[missing_field]
    with pytest.raises(ValidationError):
        FunctionDefinition(**incomplete)


def test_function_definition_empty_name() -> None:
    """jambo jambo"""
    invalid = VALID_FUNCTION_DEFINITION.copy()
    invalid["name"] = ""
    with pytest.raises(ValidationError):
        FunctionDefinition(**invalid)


def test_function_definition_empty_parameter_key() -> None:
    """jambo jambo"""
    invalid = VALID_FUNCTION_DEFINITION.copy()
    invalid["parameters"] = {
        "": {"type": "number"},
        "b": {"type": "number"},
    }
    with pytest.raises(ValidationError):
        FunctionDefinition(**invalid)
