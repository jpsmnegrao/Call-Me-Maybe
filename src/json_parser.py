"""JSON parsing and validation utilities.

This module reads the two mandatory input files (function calling test prompts
and function definitions) from disk, parses their JSON content, and validates
each entry against the corresponding pydantic schema.
"""

import json
from schemas import Prompt, FunctionDefinition


def prompt_parser(filename: str) -> list[Prompt]:
    """Load and validate a list of prompts from a JSON file.

    Reads the given file, parses its content as JSON, and validates each entry
    against the Prompt schema.

    Args:
        filename: Path to the JSON file containing an array of prompt objects.

    Returns:
        A list of validated Prompt instances.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file does not contain valid JSON.
        ValidationError: If an entry does not match the expected Prompt schema.
        ValueError: If the file contains an empty array.
    """
    prompts: list[Prompt] = []
    with open(filename, 'r') as file:
        raw_prompts: list[dict] = json.load(file)
        if not isinstance(raw_prompts, list):
            raise ValueError("Expected a JSON array!")
        if not raw_prompts:
            raise ValueError("Empty file provided!")
        for prompt in raw_prompts:
            prompts.append(Prompt(**prompt))
    return prompts


def function_parser(filename: str) -> list[FunctionDefinition]:
    """Load and validate a list of function definitions from a JSON file.

    Reads the given file, parses its content as JSON, and validates each entry
    against the FunctionDefinition schema.

    Args:
        filename: Path to the JSON file containing an array of function
            definition objects.

    Returns:
        A list of validated FunctionDefinition instances.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file does not contain valid JSON.
        ValidationError: If an entry does not match the expected
            FunctionDefinition schema.
        ValueError: If the file contains an empty array.
    """
    functions: list[FunctionDefinition] = []
    with open(filename, 'r') as file:
        raw_funcs: list[dict] = json.load(file)
        if not isinstance(raw_funcs, list):
            raise ValueError("Expected a JSON array!")
        if not raw_funcs:
            raise ValueError("Empty file provided!")
        for function in raw_funcs:
            functions.append(FunctionDefinition(**function))
    return functions
