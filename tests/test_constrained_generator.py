"""Tests for constrained value and function-name generation.

These tests verify that constrained generation produces values of the expected
types and, where deterministic prompts are used, extracts the expected
function names, numbers, and strings.
"""


import pytest
from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
from src.token_setup import (
    encode_numbers,
    encode_terminators,
    build_string_trie
)
from src.constrained_generator import (
    emit_number,
    emit_string,
    emit_boolean,
    emit_function_name,
)
from src.trie import build_trie


@pytest.fixture(scope="module")
def model() -> Small_LLM_Model:
    """Load the model once for the whole test module (expensive)."""
    return Small_LLM_Model()


# EMIT FUNCTION TESTS (model-dependent, minimal/isolated contexts)


def test_emit_boolean_returns_bool(model: Small_LLM_Model) -> None:
    """Verify that boolean emission returns a boolean value."""
    bool_tokens = {
        "true": model.encode("true").tolist()[0],
        "false": model.encode("false").tolist()[0],
    }
    output_tokens = model.encode("Is the sky blue? ").tolist()[0]
    result = emit_boolean(model, output_tokens, bool_tokens)
    assert isinstance(result, bool)


def test_emit_number_returns_float_and_terminates(
        model: Small_LLM_Model) -> None:
    """Verify that number emission returns a float and terminates."""
    digit_tokens = encode_numbers(model)
    terminators = encode_terminators(model, [",", "}", "."])
    output_tokens = model.encode("The sum of 2 and 3 is ").tolist()[0]
    result = emit_number(
        model, output_tokens, digit_tokens, terminators, "float",
        max_tokens=20)
    assert isinstance(result, float)


def test_emit_string_returns_str_and_terminates(
        model: Small_LLM_Model) -> None:
    """Verify that string emission returns a non-empty string."""
    candidates = ["shrek"]
    string_trie = build_string_trie(candidates, model)
    terminators = encode_terminators(model, ['"'])
    output_tokens = model.encode(
        'The name mentioned in "Greet shrek" is "').tolist()[0]
    result = emit_string(model, output_tokens, string_trie, terminators)
    assert isinstance(result, str)
    assert len(result) > 0


def test_emit_function_name_selects_a_valid_name(
        model: Small_LLM_Model) -> None:
    """Verify that function-name emission selects a valid function name."""
    function_names = ["fn_add_numbers", "fn_greet"]
    trie = build_trie(function_names, model)
    output_tokens = model.encode(
        "The function to add two numbers is called ").tolist()[0]
    selected = emit_function_name(trie, model, output_tokens)
    assert selected in function_names


def test_emit_number_extracts_the_correct_value(
        model: Small_LLM_Model) -> None:
    """Verify that number emission extracts the expected value."""
    digit_tokens = encode_numbers(model)
    terminators = encode_terminators(model, [",", "}", "."])
    output_tokens = model.encode(
        "The first number mentioned in 'the sum of 2 and 3' is ").tolist()[0]
    result = emit_number(
        model, output_tokens, digit_tokens, terminators, "float",
        max_tokens=20)
    assert result == 2.0, f"Expected 2.0, got {result}"


def test_emit_string_extracts_the_correct_name(
        model: Small_LLM_Model) -> None:
    """Verify that string emission extracts the expected candidate."""
    candidates = ["shrek"]
    string_trie = build_string_trie(candidates, model)
    terminators = encode_terminators(model, ['"'])
    output_tokens = model.encode(
        'The name mentioned in "Greet shrek" is "').tolist()[0]
    result = emit_string(model, output_tokens, string_trie, terminators)
    assert result == "shrek", f"Expected 'shrek', got {result!r}"
