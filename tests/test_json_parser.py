"""mambo jambo"""

import pytest
import json
from pydantic import ValidationError
from schemas import Prompt, FunctionDefinition
from json_parser import prompt_parser, function_parser

# PROMPT TESTING


def test_prompt_parser_valid() -> None:
    """amambo jambo"""
    with open("tests/data/function_calling_tests.json", 'r') as file:
        raw_prompts: list[dict] = json.load(file)
    prompts: list[Prompt] = prompt_parser(
        "tests/data/function_calling_tests.json")
    assert len(raw_prompts) == len(prompts)
    for prompt in prompts:
        assert isinstance(prompt, Prompt)
    assert all(
        parsed.prompt == raw["prompt"]
        for parsed, raw in zip(prompts, raw_prompts)
    )


def test_prompt_parser_file_not_found() -> None:
    """jambo jambo"""
    with pytest.raises(FileNotFoundError) as error_info:
        prompt_parser("tests/data/function_calling_tests_no_file.json")
    print(str(error_info.value))


def test_prompt_parser_invalid_syntax() -> None:
    """jambo jambo"""
    with pytest.raises(json.JSONDecodeError):
        prompt_parser("tests/data/function_calling_tests_invalid_syntax.json")


def test_prompt_parser_empty_array() -> None:
    """jambo jambo"""
    with pytest.raises(ValueError):
        prompt_parser("tests/data/function_calling_tests_empty_array.json")


def test_prompt_parser_not_array() -> None:
    """jambo jambo"""
    with pytest.raises(ValueError):
        prompt_parser("tests/data/function_calling_tests_not_array.json")


def test_prompt_parser_empty_key_value() -> None:
    """jambo jambo"""
    with pytest.raises(ValidationError):
        prompt_parser("tests/data/function_calling_tests_empty_key_value.json")


def test_prompt_parser_wrong_key() -> None:
    """jambo jambo"""
    with pytest.raises(ValidationError):
        prompt_parser("tests/data/function_calling_tests_wrong_key.json")


def test_prompt_parser_invalid_data_type() -> None:
    """jambo jambo"""
    with pytest.raises(ValidationError):
        prompt_parser(
            "tests/data/function_calling_tests_invalid_prompt_data_type.json")


def test_prompt_parser_extra_key() -> None:
    """jambo jambo"""
    with pytest.raises(ValidationError):
        prompt_parser("tests/data/function_calling_tests_extra_key.json")

# FUNCTION DEFINITION TESTING


def test_function_parser_valid() -> None:
    """amambo jambo"""
    with open("tests/data/functions_definition.json", 'r') as file:
        raw_functions: list[dict] = json.load(file)
    functions: list[FunctionDefinition] = function_parser(
        "tests/data/functions_definition.json")
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
        function_parser("tests/data/functions_definition_no_file.json")


def test_function_parser_invalid_syntax() -> None:
    """jambo jambo"""
    with pytest.raises(json.JSONDecodeError):
        function_parser(
            "tests/data/functions_definition_invalid_syntax.json")


def test_function_parser_empty_array() -> None:
    """jambo jambo"""
    with pytest.raises(ValueError):
        function_parser(
            "tests/data/functions_definition_empty_array.json")


def test_function_parser_not_array() -> None:
    """jambo jambo"""
    with pytest.raises(ValueError):
        function_parser("tests/data/functions_definition_not_array.json")


def test_function_parser_wrong_key() -> None:
    """jambo jambo"""
    with pytest.raises(ValidationError):
        function_parser("tests/data/functions_definition_wrong_key.json")


def test_function_parser_invalid_data_type() -> None:
    """jambo jambo"""
    with pytest.raises(ValidationError):
        function_parser(
            "tests/data/functions_definition_invalid_function_data"
            "_type.json"
        )


def test_function_parser_extra_key() -> None:
    """jambo jambo"""
    with pytest.raises(ValidationError):
        function_parser("tests/data/functions_definition_extra_key.json")


def test_function_parser_extra_nested_key() -> None:
    """jambo jambo"""
    with pytest.raises(ValidationError):
        function_parser(
            "tests/data/functions_definition_extra_nested_key.json")


VALID_FUNCTION_DEFINITION: dict = {
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
