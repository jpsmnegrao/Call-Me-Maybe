"""Command-line interface for the application."""


import argparse
from typing_extensions import Never
from pathlib import Path
from pydantic import BaseModel, FilePath


class CLIError(Exception):
    """Raised when command-line arguments are invalid."""


class CLIArgParser(argparse.ArgumentParser):
    """Argument parser that raises CLIError instead of exiting.

    The default ArgumentParser behavior terminates the program when an
    argument error occurs. This parser raises CLIError instead so that
    command-line errors can be handled by the application.
    """

    def error(self, message: str) -> Never:
        """Raise CLIError with the parser error message.

        Args:
            message: Description of the command-line parsing error.

        Raises:
            CLIError: Always raised with the provided error message.
        """
        raise CLIError(message)


class CLI(BaseModel):
    """Validated command-line arguments used by the application.

    Attributes:
        functions: Path to the JSON file containing function definitions.
        input: Path to the JSON file containing function-calling test prompts.
        output: Path where the generated function-calling results are saved.
    """
    functions: FilePath
    input: FilePath
    output: Path


def parse_cli() -> CLI:
    """Parse command-line arguments and validate their file paths.

    Returns:
        A CLI object containing the functions definition, input, and
        output paths.

    Raises:
        CLIError: If the command-line arguments cannot be parsed.
    """
    parser = CLIArgParser()
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json")
    parser.add_argument(
        "--input", default="data/input/function_calling_tests.json")
    parser.add_argument(
        "--output", default="data/output/function_calling_results.json")
    args = parser.parse_args()
    return CLI(
        functions=args.functions_definition,
        input=args.input,
        output=args.output
    )
