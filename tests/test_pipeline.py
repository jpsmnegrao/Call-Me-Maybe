"""Tests for the end-to-end function-calling pipeline.

These tests verify that the pipeline selects the appropriate function and
extracts the expected parameter values for representative number and string
function-calling prompts.
"""


import pytest
from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
from src.pipeline import run_pipeline
from src.schemas import Prompt, FunctionDefinition, ParameterInfo


@pytest.fixture(scope="module")
def model() -> Small_LLM_Model:
    """Load the language model once for all tests in this module.

    Returns:
        A loaded Small_LLM_Model instance shared by the module's tests.
    """
    return Small_LLM_Model()


def test_pipeline_produces_correct_output_for_simple_addition(
        model: Small_LLM_Model) -> None:
    """Verify that the pipeline extracts numbers for an addition function."""
    functions = [
        FunctionDefinition(
            name="fn_add_numbers",
            description="Add two numbers together and return their sum.",
            parameters={
                "a": ParameterInfo(type="number"),
                "b": ParameterInfo(type="number"),
            },
            returns=ParameterInfo(type="number"),
        )
    ]
    prompts = [Prompt(prompt="What is the sum of 2 and 3?")]
    results = run_pipeline(prompts, functions)
    print("RESULT:", results[0])
    assert len(results) == 1
    assert results[0].name == "fn_add_numbers"
    assert results[0].parameters["a"] == 2.0
    assert results[0].parameters["b"] == 3.0


def test_pipeline_produces_correct_output_for_greet(
        model: Small_LLM_Model) -> None:
    """Verify that the pipeline extracts a person's name for a greeting."""
    functions = [
        FunctionDefinition(
            name="fn_greet",
            description="Generate a greeting message for a person by name.",
            parameters={"name": ParameterInfo(type="string")},
            returns=ParameterInfo(type="string"),
        )
    ]
    prompts = [Prompt(prompt="Greet shrek")]
    results = run_pipeline(prompts, functions)
    print("RESULT:", results[0])
    assert len(results) == 1
    assert results[0].name == "fn_greet"
    assert results[0].parameters["name"] == "shrek", (
        f"Expected 'shrek', got {results[0].parameters['name']!r}")
