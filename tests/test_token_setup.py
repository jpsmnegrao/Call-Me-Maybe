"""Tests for token setup and prompt-preparation utilities.

Model-independent functions (extract_string_candidates, is_main_content_param,
build_prompt_text) are tested directly. The encoder functions
(literals_encoder, encode_numbers, encode_terminators, build_string_trie)
require a real model and are tested using the shared `model` fixture.
"""


import pytest
from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
from src.token_setup import (extract_string_candidates, is_main_content_param,
                             build_prompt_text, literals_encoder,
                             encode_numbers, encode_terminators,
                             build_string_trie)
from src.schemas import FunctionDefinition, ParameterInfo, Prompt


@pytest.fixture(scope="module")
def model() -> Small_LLM_Model:
    """Load the model once for the whole test module (expensive)."""
    return Small_LLM_Model()


# EXTRACT_STRING_CANDIDATES


def test_extract_candidates_finds_double_quoted_substring() -> None:
    """jambo"""
    result = extract_string_candidates('Greet "shrek" now')
    assert "shrek" in result


def test_extract_candidates_finds_single_quoted_substring() -> None:
    """jambo"""
    result = extract_string_candidates("Greet 'shrek' now")
    assert "shrek" in result


def test_extract_candidates_handles_apostrophe_inside_double_quotes() -> None:
    """Regression test for quote-nesting."""
    result = extract_string_candidates(
        'Replace numbers in "Hello 34 I\'m 233 years old" with NUMBERS')
    assert "Hello 34 I'm 233 years old" in result


def test_extract_candidates_finds_capitalized_proper_noun() -> None:
    """jambo"""
    result = extract_string_candidates(
        "Book a meeting called 'Budget Review' with Alice")
    assert "Budget Review" in result
    assert "Alice" in result


def test_extract_candidates_does_not_capture_sentence_start() -> None:
    """The first word (even if capitalized) shouldn't count as a proper-noun
    candidate just because it starts the sentence."""
    result = extract_string_candidates("Greet shrek")
    assert "Greet" not in result


def test_extract_candidates_falls_back_to_last_word() -> None:
    """jambo"""
    result = extract_string_candidates("what time is it")
    assert result == ["it"]


def test_extract_candidates_deduplicates() -> None:
    """jambo"""
    result = extract_string_candidates("'cat' and 'cat' again")
    assert result.count("cat") == 1


# IS_MAIN_CONTENT_PARAM


def test_is_main_content_param_recognizes_known_names() -> None:
    """jambo"""
    assert is_main_content_param("source_string") is True
    assert is_main_content_param("text") is True
    assert is_main_content_param("TEXT") is True


def test_is_main_content_param_rejects_other_names() -> None:
    """jambo"""
    assert is_main_content_param("regex") is False
    assert is_main_content_param("replacement") is False


# BUILD_PROMPT_TEXT


def test_build_prompt_text_includes_all_functions_and_request() -> None:
    """jambo"""
    functions = [
        FunctionDefinition(
            name="fn_add_numbers",
            description="Add two numbers.",
            parameters={"a": ParameterInfo(type="number")},
            returns=ParameterInfo(type="number"),
        )
    ]
    prompt = Prompt(prompt="What is 2 plus 3?")
    text = build_prompt_text(prompt, functions)
    assert "fn_add_numbers" in text
    assert "What is 2 plus 3?" in text
    assert "a: number" in text


# MODEL-DEPENDENT ENCODERS


def test_literals_encoder_has_expected_keys(model: Small_LLM_Model) -> None:
    """jambo"""
    encoded = literals_encoder(model)
    expected_keys = {
        "lit_open", "lit_prompt", "lit_name", "lit_name_close",
        "lit_parameters", "lit_comma", "lit_close_entry", "lit_close",
    }
    assert set(encoded.keys()) == expected_keys
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


def test_build_string_trie_contains_all_candidates(
        model: Small_LLM_Model) -> None:
    """jambo"""
    trie = build_string_trie(["shrek", "john"], model)
    node = trie.root
    shrek_tokens = model.encode("shrek").tolist()[0]
    for token_id in shrek_tokens:
        assert token_id in node.children
        node = node.children[token_id]
    assert node.finished
    assert node.name == "shrek"
