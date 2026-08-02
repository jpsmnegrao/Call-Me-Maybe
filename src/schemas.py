"""mambo jambo"""


from pydantic import BaseModel
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
    name: str
    description: str
    parameters: dict[str, ParameterInfo]
    returns: ParameterInfo


class Output(BaseModel):
    """mambo"""
    prompt: str
    name: str
    parameters: dict[str, float | str | bool]
