"""Utilities for constrained generation of function-call components.

This module uses the LLM's logits together with tries and token constraints
to generate function names and parameter values that conform to the expected
function-call structure.
"""


from llm_sdk import Small_LLM_Model  # type: ignore
from .trie import Trie
from .token_setup import encode_terminators


def emit_function_name(
        trie: Trie, model: Small_LLM_Model, output_tokens: list[int]) -> str:
    """Generate a function name constrained by a function-name trie.

    Args:
        trie: Trie containing the available function names as token sequences.
        model: LLM model used to score the possible next tokens.
        output_tokens: Token IDs generated so far, used as the model input
            during generation.

    Returns:
        The name of the selected function.
    """
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
        param_type: str, max_tokens: int = 20) -> int | float:
    """Generate a number while restricting tokens to valid numeric tokens.

    Args:
        model: LLM model used to score the possible next tokens.
        output_tokens: Token IDs generated so far.
        digit_tokens: Mapping of numeric characters to their token IDs.
        terminators: Token IDs that may terminate the generated number.
        param_type: Declared parameter type to generate.
        max_tokens: Maximum number of numeric tokens to generate.

    Returns:
        The generated value converted to a float or an integer.
    """
    start = len(output_tokens)
    if param_type == "integer":
        digits = set(digit_tokens.values()) - {digit_tokens["."]}
    else:
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
    if param_type == "integer":
        return int(number_str)
    return float(number_str)


def emit_string(
        model: Small_LLM_Model, output_tokens: list[int], trie: Trie,
        terminators: set[int], max_tokens: int = 20) -> str:
    """Generate a string constrained to candidates stored in a trie.

    Args:
        model: LLM model used to score the possible next tokens.
        output_tokens: Token IDs generated so far.
        trie: Trie containing the allowed string candidates.
        terminators: Token IDs that may terminate the generated string.
        max_tokens: Maximum number of tokens to generate.

    Returns:
        The selected string candidate.
    """
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
    """Generate a boolean value by comparing true and false token scores.

    Args:
        model: LLM model used to score the boolean tokens.
        output_tokens: Token IDs generated so far.
        bool_tokens: Mapping of boolean values to their token IDs.

    Returns:
        True if the true token receives the higher score, otherwise False.
    """
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
        string_trie: Trie | None) -> float | int | str | bool:
    """Generate a parameter value according to its declared type.

    Args:
        param_type: Declared parameter type to generate.
        model: LLM model used for constrained generation.
        output_tokens: Token IDs generated so far.
        digit_tokens: Mapping of numeric characters to their token IDs.
        bool_tokens: Mapping of boolean values to their token IDs.
        string_trie: Trie containing allowed string candidates, when the
            parameter type is string.

    Returns:
        The generated parameter value.

    Raises:
        ValueError: If a string parameter has no candidate trie or if the
            parameter type is unsupported.
    """
    if (param_type == "number" or param_type == "float"
            or param_type == "integer"):
        terminators = encode_terminators(model, [",", "}"])
        return emit_number(
            model, output_tokens, digit_tokens, terminators, param_type)
    elif param_type == "string":
        if string_trie is None:
            raise ValueError("String parameter requires a string_trie!")
        return emit_string(
            model, output_tokens, string_trie,
            encode_terminators(model, ['"']))
    elif param_type == "boolean":
        return emit_boolean(model, output_tokens, bool_tokens)
    raise ValueError(f"Unknown parameter type: {param_type}")


# Extraction-based generation can only produce values that literally appear
# in the prompt. This fails for regex parameters implied by keywords (e.g.
# "vowels") when the regex itself (e.g. "[aeiouAEIOU]") is absent. This map
# is a narrow fallback for that specific case, not a general LLM substitute.
REGEX_KEYWORD_MAP: dict[str, str] = {
    "vowels": "[aeiouAEIOU]",
    "consonants": r"[^aeiouAEIOU\s]",
    "numbers": r"\d+",
    "digits": r"\d",
    "letters": "[a-zA-Z]",
    "uppercase": "[A-Z]",
    "lowercase": "[a-z]",
    "whitespace": r"\s+",
    "spaces": r"\s+",
    "punctuation": r"[.,!?;:]",
}


def match_regex_keyword(prompt_text: str) -> str | None:
    """Check if the prompt mentions a known character-class keyword, and
    return its regex equivalent if so.

    This is a deliberate, narrow fallback for cases where a regex pattern is
    implied by a keyword (e.g. "vowels") rather than literally present in the
    prompt text, which extraction-based generation cannot otherwise produce.
    See README for discussion.
    """
    prompt_lower = prompt_text.lower()
    for keyword, pattern in REGEX_KEYWORD_MAP.items():
        if keyword in prompt_lower:
            return pattern
    return None
