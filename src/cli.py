"""mambo jambo"""


import argparse
from typing_extensions import Never
from pathlib import Path
from pydantic import BaseModel, FilePath


class CLIError(Exception):
    """Raised when command-line arguments are invalid."""


class CLIArgParser(argparse.ArgumentParser):
    """Argument parser that raises exceptions instead of exiting."""

    def error(self, message: str) -> Never:
        raise CLIError(message)


class CLI(BaseModel):
    """mambo jambo"""
    functions: FilePath
    input: FilePath
    output: Path


def parse_cli() -> CLI:
    """mambo jambo"""
    parser = CLIArgParser()
    parser.add_argument("--functions_definition", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    return CLI(
        functions=args.functions_definition,
        input=args.input,
        output=args.output
    )
