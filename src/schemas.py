"""Pydantic schemas for input definitions and generated function calls.

Defines the validated data models used for prompts, function definitions,
parameter types, and generated function-call results.
"""


# TODO: Revisit support for "array" and "object" parameter types.
# ParameterInfo currently accepts them, but the generation pipeline does not
# handle them. BOOM CRASH
# Decide whether to implement generation support or restrict the
# accepted types to those the pipeline can actually generate.


from pydantic import BaseModel, Field, field_validator
from typing import Literal


class Prompt(BaseModel):
    """Represent a natural-language prompt to be processed by the pipeline."""
    model_config = {"extra": "forbid"}
    prompt: str


class ParameterInfo(BaseModel):
    """Define the type of a function parameter or return value."""
    model_config = {"extra": "forbid"}
    # TODO: Revisit support for "array" and "object" parameter types.
    type: Literal["string", "number", "boolean", "array", "object"]


class FunctionDefinition(BaseModel):
    """Define a function available for the LLM to select and call."""
    model_config = {"extra": "forbid"}
    name: str = Field(min_length=1)
    description: str
    parameters: dict[str, ParameterInfo]
    returns: ParameterInfo

    @field_validator("parameters")
    @classmethod
    def reject_empty_parameter_names(
            cls, value: dict[str, ParameterInfo]
    ) -> dict[str, ParameterInfo]:
        """Reject parameters whose names are empty strings.

        Args:
            value: Mapping of parameter names to their definitions.

        Returns:
            The validated parameter mapping.

        Raises:
            ValueError: If any parameter name is an empty string.
        """
        if any(key == "" for key in value):
            raise ValueError("Parameter names must not be empty!")
        return value


class Output(BaseModel):
    """Represent a generated function call for a processed prompt.

    The output contains the original prompt, the selected function name,
    and the generated parameter values.
    """
    prompt: str
    name: str
    parameters: dict[str, float | str | bool]
