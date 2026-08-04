"""mambo jambo"""


from pydantic import BaseModel, Field, field_validator
from typing import Literal


class Prompt(BaseModel):
    """mambo"""
    model_config = {"extra": "forbid"}
    prompt: str


class ParameterInfo(BaseModel):
    """mambo"""
    model_config = {"extra": "forbid"}
    type: Literal["string", "number", "boolean", "array", "object"]


class FunctionDefinition(BaseModel):
    """mambo"""
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
        """Reject any parameter whose key is an empty string."""
        if any(key == "" for key in value):
            raise ValueError("Parameter names must not be empty!")
        return value


class Output(BaseModel):
    """mambo"""
    prompt: str
    name: str
    parameters: dict[str, float | str | bool]
