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
