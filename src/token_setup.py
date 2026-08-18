"""Utilities for preparing tokens and prompts for constrained generation.

This module provides functions for encoding fixed JSON literals and numeric
tokens, extracting potential string values from prompts, building tries for
string candidates, and constructing the text prompt supplied to the LLM.
"""


import re
from llm_sdk import Small_LLM_Model  # type: ignore
from .schemas import FunctionDefinition, Prompt
from .trie import Trie


def literals_encoder(model: Small_LLM_Model) -> dict[str, list[int]]:
    """Encode the JSON literals used during constrained output generation.

    Args:
        model: LLM model used to encode the literals into token IDs.

    Returns:
        A mapping from literal names to their encoded token IDs.
    """
    literals: dict[str, str] = {
        "lit_open": '[',
        "lit_prompt": '{"prompt": "',
        "lit_name": '", "name": "',
        "lit_name_close": '"',
        "lit_parameters": ', "parameters": {',
        "lit_comma": ', ',
        "lit_close_entry": '}}',
        "lit_close": ']',
    }
    encoded_lits: dict[str, list[int]] = {}
    for key, literal in literals.items():
        encoded_lits[key] = model.encode(literal).tolist()[0]
    return encoded_lits


def encode_numbers(model: Small_LLM_Model) -> dict[str, int]:
    """Encode numeric characters used during constrained number generation.

    Args:
        model: LLM model used to encode the numeric characters.

    Returns:
        A mapping from numeric characters to their corresponding token IDs.
    """
    digit_strings = [
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ".", "-"]
    encoded_digits: dict[str, int] = {}
    for digit in digit_strings:
        encoded_digits[digit] = model.encode(digit).tolist()[0][0]
    return encoded_digits


def encode_terminators(
        model: Small_LLM_Model, characters: list[str]) -> set[int]:
    """Encode characters used to terminate constrained token generation.

    Args:
        model: LLM model used to encode the terminator characters.
        characters: Characters that may terminate the current generated value.

    Returns:
        A set containing the token IDs corresponding to the terminators.
    """
    terminators: set[int] = set()
    for char in characters:
        encoded = model.encode(char)
        token_id = encoded.tolist()[0][0]
        terminators.add(token_id)
    return terminators


def extract_string_candidates(prompt: str) -> list[str]:
    """Extract plausible string values from a natural-language prompt.

    Quoted substrings and likely proper nouns are collected as candidates.
    If neither is found, the final word in the prompt is used as a fallback.

    Args:
        prompt: Natural-language prompt from which to extract candidates.

    Returns:
        A list of unique candidate string values.
    """
    # Capture the content between any pair of quote characters
    quoted = (
        re.findall(r'"([^"]*)"', prompt) + re.findall(r"'([^']*)'", prompt))
    # Blank out quoted spans so they aren't re-matched below
    text_without_quotes = re.sub(r"""["'][^"']*["']""", " ", prompt)
    # Capture capitalized words not at the start of the sentence
    capitalized_words = re.findall(
        r"(?<!^)(?<!\. )\b[A-Z][a-zA-Z0-9_]*\b", text_without_quotes)
    candidates = quoted + capitalized_words
    if candidates:
        return list(dict.fromkeys(candidates))
    # Fallback: no quoted or capitalized candidates found at all,
    # so just grab the last word in the prompt as a last resort
    words = re.findall(r"\b[A-Za-z][A-Za-z0-9_]*\b", prompt)
    return list(dict.fromkeys(words[-1:]))


def build_string_trie(candidates: list[str], model: Small_LLM_Model) -> Trie:
    """Build a trie containing the supplied string candidates.

    Args:
        candidates: String values that may be selected during generation.
        model: LLM model used to encode the candidate strings.

    Returns:
        A trie containing the encoded candidate strings.
    """
    trie = Trie()
    for candidate in candidates:
        token_ids = model.encode(candidate).tolist()[0]
        trie.insert(token_ids, candidate)
    return trie


def build_prompt_text(
        prompt: Prompt, functions: list[FunctionDefinition]) -> str:
    """Build the instruction text supplied to the LLM for function selection.

    Args:
        prompt: Natural-language request to process.
        functions: Available function definitions and their parameters.

    Returns:
        The formatted prompt text containing the available functions and
        the request.
    """
    lines = [
        "Choose a function for each prompt given.",
        "Extract the correct arguments and exact values from the prompt.",
        "",
        "Available functions:",
    ]
    for function in functions:
        param_list = ", ".join(
            f"{name}: {info.type}"
            for name, info in function.parameters.items()
        )
        lines.append(
            f"- {function.name}({param_list}): {function.description}"
        )
    lines.append("")
    lines.append(f"Request: {prompt.prompt}")
    return "\n".join(lines)


# Parameter names commonly used for the main string being operated on.
# These parameters should prefer the longest candidate extracted from the
# prompt, since they typically contain the primary input text.

MAIN_CONTENT_PARAM_NAMES = {
    "source_string", "text", "input", "content", "string", "s",
}


def is_main_content_param(param_name: str) -> bool:
    """Check whether a parameter conventionally represents main content.

    Args:
        param_name: The parameter name to check.

    Returns:
        True if the name matches a known "main content" convention.
    """
    return param_name.lower() in MAIN_CONTENT_PARAM_NAMES
