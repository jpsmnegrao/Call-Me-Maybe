"""mambo jambo"""


import json
import sys
from pydantic import ValidationError
from schemas import Prompt, FunctionDefinition
from json_parser import prompt_parser, function_parser


def main() -> None:
    """mambo"""
    try:
        prompts: list[Prompt] = prompt_parser(
            "data/input/function_calling_tests.json")
    except FileNotFoundError:
        print("File not found!")
        sys.exit(1)
    except json.JSONDecodeError:
        print("Invalid json format!")
        sys.exit(1)
    except ValidationError:
        print("Prompt entry does not match the expected format!")
        sys.exit(1)
    except ValueError as error:
        print(error)
        sys.exit(1)
    try:
        functions: list[FunctionDefinition] = function_parser(
            "data/input/functions_definition.json")
    except FileNotFoundError:
        print("File not found!")
        sys.exit(1)
    except json.JSONDecodeError:
        print("Invalid json format!")
        sys.exit(1)
    except ValidationError:
        print("Function entry does not match the expected format!")
        sys.exit(1)
    except ValueError as error:
        print(error)
        sys.exit(1)
