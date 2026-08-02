"""JSON parsing and validation utilities.

This module reads the two mandatory input files (function calling test prompts
and function definitions) from disk, parses their JSON content, and validates
each entry against the corresponding pydantic schema.
"""

import json
from pydantic import ValidationError
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
        ValueError: If file does not exist, is not valid JSON, is empty, or
            contains an entry that does not match the expected Prompt schema.
    """
    try:
        prompts: list[Prompt] = []
        with open(filename, 'r') as file:
            raw_prompts: list[dict] = json.load(file)
            if not raw_prompts:
                raise ValueError("Empty file provided!")
            for prompt in raw_prompts:
                prompts.append(Prompt(**prompt))
    except FileNotFoundError:
        raise ValueError("File not found!")
    except json.JSONDecodeError:
        raise ValueError("Invalid json format!")
    except ValidationError:
        raise ValueError("Prompt entry does not match the expected format!")
    except ValueError:
        raise
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
        ValueError: If file does not exist, is not valid JSON, is empty, or
            contains an entry that does not match the expected
            FunctionDefinition schema.
    """
    try:
        functions: list[FunctionDefinition] = []
        with open(filename, 'r') as file:
            raw_funcs: list[dict] = json.load(file)
            if not raw_funcs:
                raise ValueError("Empty file provided!")
            for function in raw_funcs:
                functions.append(FunctionDefinition(**function))
    except FileNotFoundError:
        raise ValueError("File not found!")
    except json.JSONDecodeError:
        raise ValueError("Invalid json format!")
    except ValidationError:
        raise ValueError("Function entry does not match the expected format!")
    except ValueError:
        raise
    return functions
