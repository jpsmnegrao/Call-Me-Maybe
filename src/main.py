"""mambo jambo"""


import json
import sys
from pydantic import ValidationError
from .schemas import Prompt, FunctionDefinition
from .json_parser import prompt_parser, function_parser
from .cli import CLIError, parse_cli


def main() -> None:
    """mambo"""
    try:
        cli = parse_cli()
    except CLIError as error:
        print(f"Command-line error: {error}")
        sys.exit(1)
    except ValidationError as error:
        print(f"Invalid command-line arguments: {error}")
        sys.exit(1)
    try:
        prompts: list[Prompt] = prompt_parser(cli.input)
    except FileNotFoundError:
        print("Error reading prompts file: file not found!")
        sys.exit(1)
    except json.JSONDecodeError:
        print("Error reading prompts file: invalid JSON format!")
        sys.exit(1)
    except ValidationError as error:
        print(f"Error reading prompts file: {error}")
        sys.exit(1)
    except ValueError as error:
        print(f"Error reading prompts file: {error}")
        sys.exit(1)
    try:
        functions: list[FunctionDefinition] = function_parser(cli.functions)
    except FileNotFoundError:
        print("Error reading functions definition file: file not found!")
        sys.exit(1)
    except json.JSONDecodeError:
        print("Error reading functions definition file: invalid JSON format!")
        sys.exit(1)
    except ValidationError as error:
        print(f"Error reading functions definition file: {error}")
        sys.exit(1)
    except ValueError as error:
        print(f"Error reading functions definition file: {error}")
        sys.exit(1)
