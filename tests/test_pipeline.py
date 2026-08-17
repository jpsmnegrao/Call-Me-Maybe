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


def test_pipeline_uses_regex_keyword_fallback(
        model: Small_LLM_Model) -> None:
    """Verify that a regex-type parameter implied by a known keyword is
    filled via the keyword fallback rather than extraction."""
    functions = [
        FunctionDefinition(
            name="fn_substitute_string_with_regex",
            description="Replace all occurrences matching a regex pattern.",
            parameters={
                "source_string": ParameterInfo(type="string"),
                "regex": ParameterInfo(type="string"),
                "replacement": ParameterInfo(type="string"),
            },
            returns=ParameterInfo(type="string"),
        )
    ]
    prompts = [Prompt(
        prompt="Replace all vowels in 'Programming is fun' with asterisks")]
    results = run_pipeline(prompts, functions)
    assert results[0].parameters["regex"] == "[aeiouAEIOU]"


def test_pipeline_disambiguates_multiple_string_parameters(
        model: Small_LLM_Model) -> None:
    """Verify that distinct string parameters receive distinct, correctly
    assigned values when multiple candidates exist in the prompt."""
    functions = [
        FunctionDefinition(
            name="fn_substitute_string_with_regex",
            description="Replace all occurrences of a word with another.",
            parameters={
                "source_string": ParameterInfo(type="string"),
                "regex": ParameterInfo(type="string"),
                "replacement": ParameterInfo(type="string"),
            },
            returns=ParameterInfo(type="string"),
        )
    ]
    prompts = [Prompt(
        prompt="Substitute the word 'cat' with 'dog' in "
               "'The cat sat on the mat with another cat'")]
    results = run_pipeline(prompts, functions)
    params = results[0].parameters
    assert params["source_string"] == (
        "The cat sat on the mat with another cat")
    assert params["regex"] == "cat"
    assert params["replacement"] == "dog"


def test_pipeline_handles_integer_parameter(
        model: Small_LLM_Model) -> None:
    """Verify that an integer-typed parameter is generated as an int."""
    functions = [
        FunctionDefinition(
            name="fn_is_even",
            description="Check if an integer is even.",
            parameters={"n": ParameterInfo(type="integer")},
            returns=ParameterInfo(type="boolean"),
        )
    ]
    prompts = [Prompt(prompt="Is 42 an even number?")]
    results = run_pipeline(prompts, functions)
    assert isinstance(results[0].parameters["n"], int)
    assert results[0].parameters["n"] == 42


def test_pipeline_handles_zero_parameter_function(
        model: Small_LLM_Model) -> None:
    """Verify that a function with no parameters produces an empty parameters
    dict."""
    functions = [
        FunctionDefinition(
            name="fn_get_current_time",
            description="Return the current time.",
            parameters={},
            returns=ParameterInfo(type="string"),
        )
    ]
    prompts = [Prompt(prompt="What time is it right now?")]
    results = run_pipeline(prompts, functions)
    assert results[0].name == "fn_get_current_time"
    assert results[0].parameters == {}
