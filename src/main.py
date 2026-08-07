"""mambo jambo"""


import json
import sys
from pydantic import ValidationError
from .json_parser import prompt_parser, function_parser
from .cli import CLIError, parse_cli
from .pipeline import run_pipeline


def main() -> None:
    """mambo"""
    try:
        cli = parse_cli()
    except (CLIError, ValidationError) as error:
        print(f"Command-line error: {error}")
        sys.exit(1)

    try:
        prompts = prompt_parser(cli.input)
    except json.JSONDecodeError:
        print("Error reading prompts file: invalid JSON format!")
        sys.exit(1)
    except (ValidationError, ValueError) as error:
        print(f"Error reading prompts file: {error}")
        sys.exit(1)

    try:
        functions = function_parser(cli.functions)
    except json.JSONDecodeError:
        print("Error reading functions definition file: invalid JSON format!")
        sys.exit(1)
    except (ValidationError, ValueError) as error:
        print(f"Error reading functions definition file: {error}")
        sys.exit(1)
    run_pipeline(prompts, functions)
