"""mambo jambo"""


import json
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
        "lit_key_open": '"',
        "lit_key_close": '": ',
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


def prompt_encoder(
        prompts: list[Prompt], model: Small_LLM_Model
        ) -> dict[str, list[int]]:
    """mambo"""
    encoded_prompts: dict[str, list[int]] = {}
    for item in prompts:
        prompt = item.prompt
        encoded_prompts["prompt"] = model.encode(prompt).tolist()[0]
    return encoded_prompts


def func_names_encoder(
        functions: list[FunctionDefinition], model: Small_LLM_Model
        ) -> dict[str, list[int]]:
    """mambo"""
    encoded_func_names: dict[str, list[int]] = {}
    for function in functions:
        name = function.name
        encoded_func_names[name] = model.encode(name).tolist()[0]
    return encoded_func_names


def parameters_encoder(
        functions: list[FunctionDefinition], model: Small_LLM_Model
        ) -> dict[str, list[int]]:
    """mambo"""
    encoded_paramaters: dict[str, list[int]] = {}
    for function in functions:
        for key in function.parameters:
            if key not in encoded_paramaters:
                encoded_paramaters[key] = model.encode(key).tolist()[0]
    return encoded_paramaters


def build_string_tokens(model: Small_LLM_Model) -> set[int]:
    """mambo"""
    vocab_path = model.get_path_to_vocab_file()
    with open(vocab_path, "r") as file:
        vocab: dict[str, int] = json.load(file)
    legal: set[int] = set()
    for token_str, token_id in vocab.items():
        if '"' not in token_str:
            legal.add(token_id)
    return legal


def emit_json_skeleton(encoded_lits: dict[str, list[int]]) -> list[int]:
    """jambo"""
    json_output: list[int] = []
    json_output.extend(encoded_lits["lit_open"])
    json_output.extend(encoded_lits["lit_prompt"])
    json_output.extend(encoded_lits["lit_name"])
    json_output.extend(encoded_lits["lit_parameters"])
    json_output.extend(encoded_lits["lit_close_entry"])
    json_output.extend(encoded_lits["lit_close"])
    return json_output


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
        masked = [-1e30] * len(logits)
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
        model: Small_LLM_Model, output_tokens: list[int],
        string_tokens: set[int], terminators: set[int], max_tokens: int = 100
        ) -> str:
    """mambo"""
    start = len(output_tokens)
    for _ in range(max_tokens):
        logits = model.get_logits_from_input_ids(output_tokens)
        unconstrained_choice = max(range(len(logits)), key=lambda i: logits[i])
        if unconstrained_choice in string_tokens:
            masked = [-1e30] * len(logits)
            for token_id in string_tokens:
                masked[token_id] = logits[token_id]
            token = max(string_tokens, key=lambda i: masked[i])
            output_tokens.append(token)
        else:
            masked = [-1e30] * len(logits)
            for token_id in terminators:
                masked[token_id] = logits[token_id]
            token = max(terminators, key=lambda i: masked[i])
            output_tokens.append(token)
            print("chose terminator:", token, repr(model.decode([token])))
            break
    generated_tokens = output_tokens[start:-1]
    return model.decode(generated_tokens)


def emit_boolean(
        model: Small_LLM_Model, output_tokens: list[int],
        bool_tokens: dict[str, list[int]]
        ) -> bool:
    """mambo"""
    true_tokens = bool_tokens["true"]
    false_tokens = bool_tokens["false"]
    logits = model.get_logits_from_input_ids(output_tokens)
    true_score = logits[true_tokens[0]]
    false_score = logits[false_tokens[0]]
    is_true = true_score > false_score
    chosen = true_tokens if true_score > false_score else false_tokens
    output_tokens.extend(chosen)
    return is_true


def emit_parameter_type(
        param_type: str, model: Small_LLM_Model, output_tokens: list[int],
        digit_tokens: dict[str, int], bool_tokens: dict[str, list[int]],
        string_tokens: set[int]
        ) -> float | str | bool:
    """hambo"""
    print(repr(param_type))
    if param_type == "number":
        terminators = encode_terminators(model, [",", "}", "]"])
        return emit_number(model, output_tokens, digit_tokens, terminators)
    elif param_type == "string":
        terminators = encode_terminators(model, ['"'])
        terminators = encode_terminators(model, ['"'])
        return emit_string(model, output_tokens, string_tokens, terminators)
    elif param_type == "boolean":
        return emit_boolean(model, output_tokens, bool_tokens)
    raise ValueError(f"Unknown parameter type: {param_type}")


def build_prompt_text(
        prompt: Prompt, functions: list[FunctionDefinition]) -> str:
    """mambo"""
    lines = [
        "Chooose a function for each prompt given.",
        "Identify the correct arguments and their values from the prompt.",
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
