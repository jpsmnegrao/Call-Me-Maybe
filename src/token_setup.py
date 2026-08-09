"""mambo jambo"""


import re
from llm_sdk import Small_LLM_Model  # type: ignore
from .schemas import FunctionDefinition, Prompt
from .trie import Trie


def literals_encoder(model: Small_LLM_Model) -> dict[str, list[int]]:
    """mambo"""
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
    """mambo"""
    digit_strings = [
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ".", "-"]
    encoded_digits: dict[str, int] = {}
    for digit in digit_strings:
        encoded_digits[digit] = model.encode(digit).tolist()[0][0]
    return encoded_digits


def encode_terminators(
        model: Small_LLM_Model, characters: list[str]) -> set[int]:
    """mambo"""
    terminators: set[int] = set()
    for char in characters:
        encoded = model.encode(char)
        token_id = encoded.tolist()[0][0]
        terminators.add(token_id)
    return terminators


def extract_string_candidates(prompt: str) -> list[str]:
    """Extract plausible string values from a prompt.

    Returns quoted substrings and unquoted capitalized words (likely
    proper nouns) as candidates. Falls back to the last word in the
    prompt if nothing else matches.
    """
    # Capture the content between any pair of quote characters
    quoted = re.findall(r"""["']([^"']*)["']""", prompt)
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
    """Build a trie containing complete string candidates."""
    trie = Trie()
    for candidate in candidates:
        token_ids = model.encode(candidate).tolist()[0]
        trie.insert(token_ids, candidate)
    return trie


def build_prompt_text(
        prompt: Prompt, functions: list[FunctionDefinition]) -> str:
    """mambo"""
    lines = [
        "Choose a function for each prompt given.",
        "Extract the correct arguments and their values from the prompt.",
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
