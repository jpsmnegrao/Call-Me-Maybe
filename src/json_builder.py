"""Output writing utilities.

This module serializes the pipeline's results to the output JSON file,
matching the format required by the project specifications.
"""


import json
from pathlib import Path
from .schemas import Output


def json_builder(results: list[Output], path: Path) -> None:
    """Write a list of Output records to a JSON file.

    Creates the parent directory if it does not already exist.

    Args:
        results: The list of validated Output instances to write.
        path: Destination file path.

    Raises:
        OSError: If the file cannot be written (e.g. permissions).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [result.model_dump() for result in results]
    with open(path, "w") as file:
        json.dump(data, file, indent=2)
