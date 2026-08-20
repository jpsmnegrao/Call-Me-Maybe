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
    emit_parameter_type,
    match_regex_keyword
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


def test_emit_number_returns_int_for_integer_type(
        model: Small_LLM_Model) -> None:
    """Verify that number emission returns an int when param_type is
    'integer', and excludes the decimal point token."""
    digit_tokens = encode_numbers(model)
    terminators = encode_terminators(model, [",", "}"])
    output_tokens = model.encode(
        "The first number mentioned in 'the sum of 2 and 3' is "
    ).tolist()[0]
    result = emit_number(
        model, output_tokens, digit_tokens, terminators, "integer",
        max_tokens=20)
    assert isinstance(result, int)


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


def test_emit_parameter_type_dispatches_number(
        model: Small_LLM_Model) -> None:
    """Verify that emit_parameter_type routes 'number' to emit_number."""
    digit_tokens = encode_numbers(model)
    bool_tokens = {
        "true": model.encode("true").tolist()[0],
        "false": model.encode("false").tolist()[0],
    }
    output_tokens = model.encode("The sum of 2 and 3 is ").tolist()[0]
    result = emit_parameter_type(
        "number", model, output_tokens, digit_tokens, bool_tokens, None)
    assert isinstance(result, float)


def test_emit_parameter_type_dispatches_integer(
        model: Small_LLM_Model) -> None:
    """Verify that emit_parameter_type routes 'integer' to emit_number
    and returns an int."""
    digit_tokens = encode_numbers(model)
    bool_tokens = {
        "true": model.encode("true").tolist()[0],
        "false": model.encode("false").tolist()[0],
    }
    output_tokens = model.encode("The sum of 2 and 3 is ").tolist()[0]
    result = emit_parameter_type(
        "integer", model, output_tokens, digit_tokens, bool_tokens, None)
    assert isinstance(result, int)


def test_emit_parameter_type_dispatches_boolean(
        model: Small_LLM_Model) -> None:
    """Verify that emit_parameter_type routes 'boolean' to emit_boolean."""
    digit_tokens = encode_numbers(model)
    bool_tokens = {
        "true": model.encode("true").tolist()[0],
        "false": model.encode("false").tolist()[0],
    }
    output_tokens = model.encode("Is the sky blue? ").tolist()[0]
    result = emit_parameter_type(
        "boolean", model, output_tokens, digit_tokens, bool_tokens, None)
    assert isinstance(result, bool)


def test_emit_parameter_type_raises_for_string_without_trie(
        model: Small_LLM_Model) -> None:
    """Verify that a string parameter without a trie raises ValueError."""
    digit_tokens = encode_numbers(model)
    bool_tokens = {
        "true": model.encode("true").tolist()[0],
        "false": model.encode("false").tolist()[0],
    }
    output_tokens = model.encode("Greet shrek").tolist()[0]
    with pytest.raises(ValueError):
        emit_parameter_type(
            "string", model, output_tokens, digit_tokens, bool_tokens, None)


def test_emit_parameter_type_raises_for_unsupported_type(
        model: Small_LLM_Model) -> None:
    """Verify that an unsupported parameter type raises ValueError."""
    digit_tokens = encode_numbers(model)
    bool_tokens = {
        "true": model.encode("true").tolist()[0],
        "false": model.encode("false").tolist()[0],
    }
    output_tokens = model.encode("Test").tolist()[0]
    with pytest.raises(ValueError):
        emit_parameter_type(
            "array", model, output_tokens, digit_tokens, bool_tokens, None)


def test_match_regex_keyword_finds_known_keywords() -> None:
    """Verify that recognized keywords map to their regex pattern."""
    assert match_regex_keyword("Replace all vowels") == "[aeiouAEIOU]"
    assert match_regex_keyword("Replace all NUMBERS") == r"\d+"


def test_match_regex_keyword_returns_none_for_unknown_prompt() -> None:
    """Verify that a prompt with no known keyword returns None."""
    assert match_regex_keyword("Substitute cat with dog") is None
