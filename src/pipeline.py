"""mambo jambo"""


from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
from .schemas import FunctionDefinition, Prompt, Output
from .token_setup import (literals_encoder, encode_numbers, build_prompt_text,
                          extract_string_candidates, build_string_trie)
from .constrained_generator import emit_function_name, emit_parameter_type
from .trie import Trie, build_trie


def setup_generation_context(
        functions: list[FunctionDefinition], model: Small_LLM_Model
        ) -> tuple[dict[str, list[int]], dict[str, int],
                   dict[str, list[int]], Trie]:
    """mambo"""
    encoded_lits = literals_encoder(model)
    digit_tokens = encode_numbers(model)
    bool_tokens = {
        "true": model.encode("true").tolist()[0],
        "false": model.encode("false").tolist()[0]
    }
    function_names = [function.name for function in functions]
    trie = build_trie(function_names, model)
    return encoded_lits, digit_tokens, bool_tokens, trie


def process_prompt(
        prompt: Prompt, functions: list[FunctionDefinition],
        trie: Trie, encoded_lits: dict[str, list[int]],
        digit_tokens: dict[str, int], bool_tokens: dict[str, list[int]],
        model: Small_LLM_Model) -> Output:
    """mambo"""
    prompt_text = build_prompt_text(prompt, functions)
    string_candidates = extract_string_candidates(prompt.prompt)
    output_tokens: list[int] = model.encode(prompt_text).tolist()[0]
    output_tokens.extend(encoded_lits["lit_open"])
    output_tokens.extend(encoded_lits["lit_prompt"])
    prompt_tokens = model.encode(prompt.prompt).tolist()[0]
    output_tokens.extend(prompt_tokens)
    output_tokens.extend(encoded_lits["lit_name"])
    selected_name = emit_function_name(trie, model, output_tokens)
    output_tokens.extend(encoded_lits["lit_name_close"])
    selected_function = next(
        f for f in functions if f.name == selected_name)
    output_tokens.extend(encoded_lits["lit_parameters"])
    parameter_values: dict[str, float | str | bool] = {}
    param_items = list(selected_function.parameters.items())
    for i, (param_name, param_info) in enumerate(param_items):
        key_segment = f'"{param_name}": '
        output_tokens.extend(model.encode(key_segment).tolist()[0])
        if param_info.type == "string":
            remaining_candidates = [
                c for c in string_candidates
                if c not in parameter_values.values()
            ]
            if not remaining_candidates:
                remaining_candidates = string_candidates
            selection_hint = (
                f"(Choosing the value for '{param_name}' from candidates "
                f"{remaining_candidates}: "
            )
            hint_tokens = model.encode(selection_hint).tolist()[0]
            output_tokens.extend(hint_tokens)
            current_string_trie = build_string_trie(
                remaining_candidates, model)
            value = emit_parameter_type(
                param_info.type, model, output_tokens, digit_tokens,
                bool_tokens, current_string_trie)
        else:
            value = emit_parameter_type(
                param_info.type, model, output_tokens, digit_tokens,
                bool_tokens, None)
        parameter_values[param_name] = value
        if i < len(param_items) - 1:
            output_tokens.extend(encoded_lits["lit_comma"])
    output_tokens.extend(encoded_lits["lit_close_entry"])
    output_tokens.extend(encoded_lits["lit_close"])
    return Output(
        prompt=prompt.prompt,
        name=selected_name,
        parameters=parameter_values,
    )


def run_pipeline(
        prompts: list[Prompt], functions: list[FunctionDefinition]
        ) -> list[Output]:
    """mambo"""
    model = Small_LLM_Model()
    encoded_lits, digit_tokens, bool_tokens, trie = (
        setup_generation_context(functions, model)
    )
    results: list[Output] = []
    for prompt in prompts:
        result = process_prompt(
            prompt,
            functions,
            trie,
            encoded_lits,
            digit_tokens,
            bool_tokens,
            model,
        )
        results.append(result)
    return results
