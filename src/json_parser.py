"""JSON parsing and validation utilities.

This module loads the two mandatory JSON input files (prompts and function
definitions), parses their contents, and validates each entry against the
corresponding Pydantic schema.
"""

import json
from typing import Any
from pydantic import FilePath
from .schemas import Prompt, FunctionDefinition


def prompt_parser(filename: FilePath) -> list[Prompt]:
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
        ValueError: If the file does not contain a JSON array, or the array is
            empty.
    """
    prompts: list[Prompt] = []
    with open(filename, 'r') as file:
        raw_prompts: list[dict[str, Any]] = json.load(file)
        if not isinstance(raw_prompts, list):
            raise ValueError("Expected a JSON array!")
        if not raw_prompts:
            raise ValueError("Empty file provided!")
        for prompt in raw_prompts:
            prompts.append(Prompt(**prompt))
    return prompts


def function_parser(filename: FilePath) -> list[FunctionDefinition]:
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
        ValueError: If the file does not contain a JSON array, or the array is
            empty.
    """
    functions: list[FunctionDefinition] = []
    with open(filename, 'r') as file:
        raw_funcs: list[dict[str, Any]] = json.load(file)
        if not isinstance(raw_funcs, list):
            raise ValueError("Expected a JSON array!")
        if not raw_funcs:
            raise ValueError("Empty file provided!")
        for function in raw_funcs:
            functions.append(FunctionDefinition(**function))
    return functions
