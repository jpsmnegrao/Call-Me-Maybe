"""mambo jambo"""


from llm_sdk import Small_LLM_Model  # type: ignore
from .trie import Trie
from .token_setup import encode_terminators


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
