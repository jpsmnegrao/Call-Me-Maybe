"""Tests for token setup and prompt-preparation utilities.

Model-independent functions (extract_string_candidates, is_main_content_param,
build_prompt_text) are tested directly with plain Python values. The encoder
functions (literals_encoder, encode_numbers, encode_terminators,
build_string_trie) require a real model instance and are tested using the
shared `model` fixture.
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
    """Verify that a double-quoted substring is captured as a candidate."""
    result = extract_string_candidates('Greet "shrek" now')
    assert "shrek" in result


def test_extract_candidates_finds_single_quoted_substring() -> None:
    """Verify that a single-quoted substring is captured as a candidate."""
    result = extract_string_candidates("Greet 'shrek' now")
    assert "shrek" in result


def test_extract_candidates_handles_apostrophe_inside_double_quotes() -> None:
    """Verify that an apostrophe inside a double-quoted span does not
    prematurely end the match."""
    result = extract_string_candidates(
        'Replace numbers in "Hello 34 I\'m 233 years old" with NUMBERS')
    assert "Hello 34 I'm 233 years old" in result


def test_extract_candidates_finds_capitalized_proper_noun() -> None:
    """Verify that quoted and capitalized-proper-noun candidates are both
    captured together from a mixed prompt."""
    result = extract_string_candidates(
        "Book a meeting called 'Budget Review' with Alice")
    assert "Budget Review" in result
    assert "Alice" in result


def test_extract_candidates_does_not_capture_sentence_start() -> None:
    """Verify that a capitalized word at the very start of the prompt is not
    treated as a proper-noun candidate."""
    result = extract_string_candidates("Greet shrek")
    assert "Greet" not in result


def test_extract_candidates_falls_back_to_last_word() -> None:
    """Verify that the last word is used as a fallback when no quoted or
    capitalized candidates are found."""
    result = extract_string_candidates("what time is it")
    assert result == ["it"]


def test_extract_candidates_deduplicates() -> None:
    """Verify that duplicate candidates are only returned once."""
    result = extract_string_candidates("'cat' and 'cat' again")
    assert result.count("cat") == 1


# IS_MAIN_CONTENT_PARAM


def test_is_main_content_param_recognizes_known_names() -> None:
    """Verify that known main-content parameter names are recognized,
    case-insensitively."""
    assert is_main_content_param("source_string") is True
    assert is_main_content_param("text") is True
    assert is_main_content_param("TEXT") is True


def test_is_main_content_param_rejects_other_names() -> None:
    """Verify that parameter names outside the known set are rejected."""
    assert is_main_content_param("regex") is False
    assert is_main_content_param("replacement") is False


# BUILD_PROMPT_TEXT


def test_build_prompt_text_includes_all_functions_and_request() -> None:
    """Verify that the built prompt text includes every function's
    signature and the original request."""
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
    """Verify that all expected literal keys are encoded."""
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
    """Verify that all supported numeric characters are encoded."""
    digits = encode_numbers(model)
    expected = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ".", "-"}
    assert set(digits.keys()) == expected
    for token_id in digits.values():
        assert isinstance(token_id, int)


def test_encode_terminators_matches_known_token_ids(
        model: Small_LLM_Model) -> None:
    """Verify that terminator token IDs match the model's own encodings."""
    terminators = encode_terminators(model, [",", "}", "]"])
    comma = model.encode(",").tolist()[0][0]
    close_brace = model.encode("}").tolist()[0][0]
    close_bracket = model.encode("]").tolist()[0][0]
    assert terminators == {comma, close_brace, close_bracket}


def test_build_string_trie_contains_all_candidates(
        model: Small_LLM_Model) -> None:
    """Verify that the built trie contains a complete, correctly-named path
    for each candidate string."""
    trie = build_string_trie(["shrek", "john"], model)
    node = trie.root
    shrek_tokens = model.encode("shrek").tolist()[0]
    for token_id in shrek_tokens:
        assert token_id in node.children
        node = node.children[token_id]
    assert node.finished
    assert node.name == "shrek"
