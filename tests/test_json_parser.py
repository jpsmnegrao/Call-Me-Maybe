"""Tests for JSON parsing and validation utilities.

These tests verify that the prompt and function definition parsers correctly
load valid JSON input, reject invalid files and structures, and enforce the
corresponding Pydantic schemas.

The tests use real JSON fixture files to exercise parser behaviour, including
missing files, invalid JSON syntax, invalid structures, validation errors, and
valid input.
"""

import pytest
import json
from pathlib import Path
from collections.abc import Callable
from pydantic import ValidationError
from typing import cast, Any
from src.schemas import Prompt, FunctionDefinition, Output
from src.json_parser import prompt_parser, function_parser


DATA = Path("tests/data")


def run_parser_test(
        parser: Callable[[Path], list[Prompt] | list[FunctionDefinition]],
        file_path: Path, label: str
        ) -> list[Prompt] | list[FunctionDefinition]:
    """Run a parser test and print diagnostic information.

    Args:
        parser: Parser function to execute.
        file_path: Path to the JSON fixture being tested.
        label: Descriptive name for the test case.

    Returns:
        The parsed and validated objects returned by the parser.

    Raises:
        Exception: Re-raises any exception produced by the parser after
            printing its type and message.
    """
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
    """Verify that a valid prompt file is parsed correctly."""
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
    """Verify that a missing prompt file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        run_parser_test(
            prompt_parser,
            DATA / "function_calling_tests_no_file.json",
            "file_not_found",
        )


def test_prompt_parser_invalid_syntax() -> None:
    """Verify that invalid prompt JSON raises JSONDecodeError."""
    with pytest.raises(json.JSONDecodeError):
        run_parser_test(
            prompt_parser,
            DATA / "function_calling_tests_invalid_syntax.json",
            "invalid_syntax",
        )


def test_prompt_parser_empty_array() -> None:
    """Verify that an empty prompt array raises ValueError."""
    with pytest.raises(ValueError):
        run_parser_test(
            prompt_parser,
            DATA / "function_calling_tests_empty_array.json",
            "empty_array",
        )


def test_prompt_parser_not_array() -> None:
    """Verify that non-array prompt JSON raises ValueError."""
    with pytest.raises(ValueError):
        run_parser_test(
            prompt_parser,
            DATA / "function_calling_tests_not_array.json",
            "not_array",
        )


def test_prompt_parser_non_object_item() -> None:
    """Verify that a prompt array containing a non-object item is rejected."""
    with pytest.raises((TypeError, ValidationError)):
        run_parser_test(
            prompt_parser,
            DATA / "function_calling_tests_non_object_item.json",
            "non_object_item",
        )


def test_prompt_parser_empty_key_value() -> None:
    """Verify that an empty prompt value fails Pydantic validation."""
    with pytest.raises(ValidationError):
        run_parser_test(
            prompt_parser,
            DATA / "function_calling_tests_empty_key_value.json",
            "empty_key_value",
        )


def test_prompt_parser_wrong_key() -> None:
    """Verify that a prompt with an incorrect field fails validation."""
    with pytest.raises(ValidationError):
        run_parser_test(
            prompt_parser,
            DATA / "function_calling_tests_wrong_key.json",
            "wrong_key",
        )


def test_prompt_parser_invalid_data_type() -> None:
    """Verify that an invalid prompt field type fails validation."""
    with pytest.raises(ValidationError):
        run_parser_test(
            prompt_parser,
            DATA / "function_calling_tests_invalid_prompt_data_type.json",
            "invalid_data_type",
        )


def test_prompt_parser_extra_key() -> None:
    """Verify that an unexpected prompt field fails validation."""
    with pytest.raises(ValidationError):
        run_parser_test(
            prompt_parser,
            DATA / "function_calling_tests_extra_key.json",
            "extra_key",
        )


# FUNCTION DEFINITION TESTING


def test_function_parser_valid() -> None:
    """Verify that valid function definitions are parsed correctly."""
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
    """Verify that a missing func definition file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        run_parser_test(
            function_parser,
            DATA / "functions_definition_no_file.json",
            "file_not_found",
        )


def test_function_parser_invalid_syntax() -> None:
    """Verify that invalid function definition JSON raises JSONDecodeError."""
    with pytest.raises(json.JSONDecodeError):
        run_parser_test(
            function_parser,
            DATA / "functions_definition_invalid_syntax.json",
            "invalid_syntax",
        )


def test_function_parser_empty_array() -> None:
    """Verify that an empty function definition array raises ValueError."""
    with pytest.raises(ValueError):
        run_parser_test(
            function_parser,
            DATA / "functions_definition_empty_array.json",
            "empty_array",
        )


def test_function_parser_not_array() -> None:
    """Verify that non-array function definition JSON raises ValueError."""
    with pytest.raises(ValueError):
        run_parser_test(
            function_parser,
            DATA / "functions_definition_not_array.json",
            "not_array",
        )


def test_function_parser_non_object_item() -> None:
    """Verify that a func array containing a non-object item is rejected."""
    with pytest.raises((TypeError, ValidationError)):
        run_parser_test(
            function_parser,
            DATA / "functions_definition_non_object_item.json",
            "non_object_item",
        )


def test_function_parser_wrong_key() -> None:
    """Verify that an incorrect function definition field fails validation."""
    with pytest.raises(ValidationError):
        run_parser_test(
            function_parser,
            DATA / "functions_definition_wrong_key.json",
            "wrong_key",
        )


def test_function_parser_invalid_data_type() -> None:
    """Verify that an invalid func definition field type fails validation."""
    with pytest.raises(ValidationError):
        run_parser_test(
            function_parser,
            DATA / "functions_definition_invalid_function_data_type.json",
            "invalid_data_type",
        )


def test_function_parser_extra_key() -> None:
    """Verify that an unexpected function definition field fails validation."""
    with pytest.raises(ValidationError):
        run_parser_test(
            function_parser,
            DATA / "functions_definition_extra_key.json",
            "extra_key",
        )


def test_function_parser_extra_nested_key() -> None:
    """Verify that an unexpected nested parameter field fails validation."""
    with pytest.raises(ValidationError):
        run_parser_test(
            function_parser,
            DATA / "functions_definition_extra_nested_key.json",
            "extra_nested_key",
        )


def test_function_parser_empty_parameters() -> None:
    """Verify that an empty parameter mapping is accepted."""
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


# Base valid definition used by the tests below to verify individual
# FunctionDefinition validation rules without going through the JSON parser.
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
    """Verify that each required function field is validated."""
    incomplete = VALID_FUNCTION_DEFINITION.copy()
    del incomplete[missing_field]
    with pytest.raises(ValidationError):
        FunctionDefinition(**incomplete)


def test_function_definition_empty_name() -> None:
    """Verify that an empty function name fails validation."""
    invalid = VALID_FUNCTION_DEFINITION.copy()
    invalid["name"] = ""
    with pytest.raises(ValidationError):
        FunctionDefinition(**invalid)


def test_function_definition_empty_parameter_key() -> None:
    """Verify that an empty parameter name fails validation."""
    invalid = VALID_FUNCTION_DEFINITION.copy()
    invalid["parameters"] = {
        "": {"type": "number"},
        "b": {"type": "number"},
    }
    with pytest.raises(ValidationError):
        FunctionDefinition(**invalid)


def test_function_definition_accepts_integer_type() -> None:
    """Verify that a parameter with type 'integer' is accepted."""
    valid = VALID_FUNCTION_DEFINITION.copy()
    valid["parameters"] = {"n": {"type": "integer"}}
    function = FunctionDefinition(**valid)
    assert function.parameters["n"].type == "integer"


@pytest.mark.parametrize("rejected_type", ["array", "object"])
def test_function_definition_rejects_array_and_object_types(
        rejected_type: str) -> None:
    """Verify that 'array' and 'object' parameter types are rejected."""
    invalid = VALID_FUNCTION_DEFINITION.copy()
    invalid["parameters"] = {"x": {"type": rejected_type}}
    with pytest.raises(ValidationError):
        FunctionDefinition(**invalid)


def test_output_accepts_mixed_parameter_types() -> None:
    """Verify that Output accepts a mix of float, int, str, and bool
    parameter values."""
    output = Output(
        prompt="test prompt",
        name="fn_test",
        parameters={"a": 2.0, "b": 3, "c": "text", "d": True},
    )
    assert output.parameters["a"] == 2.0
    assert output.parameters["b"] == 3
    assert output.parameters["c"] == "text"
    assert output.parameters["d"] is True
