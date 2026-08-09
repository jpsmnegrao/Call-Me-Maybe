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


def emit_function_name(
        trie: Trie, model: Small_LLM_Model, output_tokens: list[int]) -> str:
    """hambo"""
    node = trie.root
    while not node.finished:
        logits = model.get_logits_from_input_ids(output_tokens)
        allowed = node.children.keys()
        masked = [-1e30] * len(logits)
        for token_id in allowed:
            masked[token_id] = logits[token_id]
        max_token = max(allowed, key=lambda i: masked[i])
        output_tokens.append(max_token)
        node = node.children[max_token]
    assert node.name is not None
    return node.name


def emit_number(
        model: Small_LLM_Model, output_tokens: list[int],
        digit_tokens: dict[str, int], terminators: set[int],
        max_tokens: int = 20) -> float:
    """mambo"""
    start = len(output_tokens)
    digits = set(digit_tokens.values())
    for _ in range(max_tokens):
        logits = model.get_logits_from_input_ids(output_tokens)
        unconstrained_choice = max(range(len(logits)), key=lambda i: logits[i])
        if unconstrained_choice in digits:
            masked = [-1e30] * len(logits)
            for token_id in digits:
                masked[token_id] = logits[token_id]
            token = max(digits, key=lambda i: masked[i])
            output_tokens.append(token)
        else:
            masked = [-1e30] * len(logits)
            for token_id in terminators:
                masked[token_id] = logits[token_id]
            token = max(terminators, key=lambda i: masked[i])
            output_tokens.append(token)
            break
    generated_tokens = output_tokens[start:-1]
    number_str = model.decode(generated_tokens)
    return float(number_str)


def emit_string(
        model: Small_LLM_Model, output_tokens: list[int], trie: Trie,
        terminators: set[int], max_tokens: int = 20) -> str:
    """Generate one complete string candidate from a trie."""
    node = trie.root
    while not node.finished:
        logits = model.get_logits_from_input_ids(output_tokens)
        allowed = node.children.keys()
        masked = [-1e30] * len(logits)
        for token_id in allowed:
            masked[token_id] = logits[token_id]
        token = max(allowed, key=lambda i: masked[i])
        output_tokens.append(token)
        node = node.children[token]
    assert node.name is not None
    return node.name


def emit_boolean(
        model: Small_LLM_Model, output_tokens: list[int],
        bool_tokens: dict[str, list[int]]) -> bool:
    """mambo"""
    true_tokens = bool_tokens["true"]
    false_tokens = bool_tokens["false"]
    logits = model.get_logits_from_input_ids(output_tokens)
    true_score = logits[true_tokens[0]]
    false_score = logits[false_tokens[0]]
    is_true = bool(true_score > false_score)
    chosen = true_tokens if is_true else false_tokens
    output_tokens.extend(chosen)
    return is_true


def emit_parameter_type(
        param_type: str, model: Small_LLM_Model, output_tokens: list[int],
        digit_tokens: dict[str, int], bool_tokens: dict[str, list[int]],
        string_trie: Trie | None) -> float | str | bool:
    """hambo"""
    if param_type == "number":
        terminators = encode_terminators(model, [",", "}"])
        return emit_number(model, output_tokens, digit_tokens, terminators)
    elif param_type == "string":
        if string_trie is None:
            raise ValueError("String parameter requires a string_trie!")
        return emit_string(
            model, output_tokens, string_trie,
            encode_terminators(model, ['"']))
    elif param_type == "boolean":
        return emit_boolean(model, output_tokens, bool_tokens)
    raise ValueError(f"Unknown parameter type: {param_type}")


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
