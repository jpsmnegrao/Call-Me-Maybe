"""mambo jambo"""

import pytest
import json
from schemas import Prompt
from json_parser import prompt_parser


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
    with pytest.raises(ValueError):
        prompt_parser("tests/data/function_calling_no_file.json")


def test_prompt_parser_invalid_syntax() -> None:
    """jambo jambo"""
    with pytest.raises(ValueError):
        prompt_parser("tests/data/function_calling_tests_invalid_syntax.json")


def test_prompt_parser_empty_array() -> None:
    """jambo jambo"""
    with pytest.raises(ValueError):
        prompt_parser("tests/data/function_calling_tests_empty_array.json")


def test_prompt_parser_not_array() -> None:
    """jambo jambo"""
    with pytest.raises(ValueError):
        prompt_parser("tests/data/function_calling_tests_not_array.json")


def test_prompt_parser_missing_key() -> None:
    """jambo jambo"""
    with pytest.raises(ValueError):
        prompt_parser("tests/data/function_calling_tests_missing_key.json")


def test_prompt_parser_wrong_key() -> None:
    """jambo jambo"""
    with pytest.raises(ValueError):
        prompt_parser("tests/data/function_calling_tests_wrong_key.json")


def test_prompt_parser_invalid_data_type() -> None:
    """jambo jambo"""
    with pytest.raises(ValueError):
        prompt_parser(
            "tests/data/function_calling_tests_invalid_prompt_data_type.json")


def test_prompt_parser_extra_key() -> None:
    """jambo jambo"""
    with pytest.raises(ValueError):
        prompt_parser("tests/data/function_calling_tests_extra_key.json")
