"""Main application entry point.

This module coordinates command-line argument parsing, input file parsing,
pipeline execution, result serialization, and application output.
"""

import json
import sys
from pydantic import ValidationError
from .cli import CLIError, parse_cli
from .json_parser import prompt_parser, function_parser
from .pipeline import run_pipeline
from .json_builder import json_builder


def main() -> None:
    """Run the main application workflow.

    Parses command-line arguments, loads and validates prompt and function
    definition files, runs the function-calling pipeline, writes the results
    to the configured output file, and prints the generated results.

    Expected errors are reported to the user and terminate the application
    cleanly.
    """
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
    except (ValidationError, ValueError, TypeError, OSError) as error:
        print(f"Error reading prompts file: {error}")
        sys.exit(1)
    try:
        functions = function_parser(cli.functions)
    except json.JSONDecodeError:
        print("Error reading functions definition file: invalid JSON format!")
        sys.exit(1)
    except (ValidationError, ValueError, TypeError, OSError) as error:
        print(f"Error reading functions definition file: {error}")
        sys.exit(1)
    try:
        results = run_pipeline(prompts, functions)
    except ValueError as error:
        print(f"Error generating function call: {error}")
        sys.exit(1)
    try:
        json_builder(results, cli.output)
    except OSError as error:
        print(f"Error writing output file: {error}")
        sys.exit(1)
    for result in results:
        print(result)
