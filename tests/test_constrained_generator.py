"""mambo hambo"""

import pytest
from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
from src.trie import build_trie, TrieNode
from src.token_setup import (
    literals_encoder,
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


@pytest.fixture(scope="module")
def model() -> Small_LLM_Model:
    """Load the model once for the whole test module (expensive)."""
    return Small_LLM_Model()


def next_tokens(self, node: TrieNode) -> set[int]:
    """Return the token IDs that can legally follow this node."""
    return set(node.children.keys())


# ENCODER TESTS (deterministic, no model judgment involved)


def test_literals_encoder_has_expected_keys(model: Small_LLM_Model) -> None:
    """jambo"""
    encoded = literals_encoder(model)
    expected_keys = {
        "lit_open", "lit_prompt", "lit_name", "lit_name_close",
        "lit_parameters", "lit_comma", "lit_close_entry", "lit_close"}
    assert expected_keys.issubset(encoded.keys())
    for value in encoded.values():
        assert isinstance(value, list)
        assert len(value) > 0


def test_encode_numbers_covers_all_digits_and_symbols(
        model: Small_LLM_Model) -> None:
    """jambo"""
    digits = encode_numbers(model)
    expected = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ".", "-"}
    assert set(digits.keys()) == expected
    for token_id in digits.values():
        assert isinstance(token_id, int)


def test_encode_terminators_matches_known_token_ids(
        model: Small_LLM_Model) -> None:
    """jambo"""
    terminators = encode_terminators(model, [",", "}", "]"])
    comma = model.encode(",").tolist()[0][0]
    close_brace = model.encode("}").tolist()[0][0]
    close_bracket = model.encode("]").tolist()[0][0]
    assert terminators == {comma, close_brace, close_bracket}


# EMIT FUNCTION TESTS (model-dependent, minimal/isolated contexts)


def test_emit_boolean_returns_bool(model: Small_LLM_Model) -> None:
    """jambo"""
    bool_tokens = {
        "true": model.encode("true").tolist()[0],
        "false": model.encode("false").tolist()[0],
    }
    output_tokens = model.encode("Is the sky blue? ").tolist()[0]
    result = emit_boolean(model, output_tokens, bool_tokens)
    assert isinstance(result, bool)


def test_emit_number_returns_float_and_terminates(
        model: Small_LLM_Model) -> None:
    """jambo"""
    digit_tokens = encode_numbers(model)
    terminators = encode_terminators(model, [",", "}", "."])
    output_tokens = model.encode("The sum of 2 and 3 is ").tolist()[0]
    result = emit_number(
        model, output_tokens, digit_tokens, terminators, max_tokens=20)
    assert isinstance(result, float)


def test_emit_string_returns_str_and_terminates(
        model: Small_LLM_Model) -> None:
    """jambo"""
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
    """jambo"""
    function_names = ["fn_add_numbers", "fn_greet"]
    trie = build_trie(function_names, model)
    output_tokens = model.encode(
        "The function to add two numbers is called ").tolist()[0]
    selected = emit_function_name(trie, model, output_tokens)
    assert selected in function_names


def test_emit_number_extracts_the_correct_value(
        model: Small_LLM_Model) -> None:
    """jambo"""
    digit_tokens = encode_numbers(model)
    terminators = encode_terminators(model, [",", "}", "."])
    output_tokens = model.encode(
        "The first number mentioned in 'the sum of 2 and 3' is ").tolist()[0]
    result = emit_number(
        model, output_tokens, digit_tokens, terminators, max_tokens=20)
    assert result == 2.0, f"Expected 2.0, got {result}"


def test_emit_string_extracts_the_correct_name(
        model: Small_LLM_Model) -> None:
    """jambo"""
    candidates = ["shrek"]
    string_trie = build_string_trie(candidates, model)
    terminators = encode_terminators(model, ['"'])
    output_tokens = model.encode(
        'The name mentioned in "Greet shrek" is "').tolist()[0]
    result = emit_string(model, output_tokens, string_trie, terminators)
    assert result == "shrek", f"Expected 'shrek', got {result!r}"
