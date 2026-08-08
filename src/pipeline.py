"""mambo jambo"""


from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
from .schemas import FunctionDefinition, Prompt
from .constrained_generator import (
    literals_encoder,
    encode_numbers,
    build_string_tokens,
    emit_function_name,
    emit_parameter_type,
)
from .trie import build_trie


def run_pipeline(
        prompts: list[Prompt], functions: list[FunctionDefinition]) -> None:
    """mambo"""
    model = Small_LLM_Model()
    encoded_lits = literals_encoder(model)
    digit_tokens = encode_numbers(model)
    string_tokens = build_string_tokens(model)
    bool_tokens = {
        "true": model.encode("true").tolist()[0],
        "false": model.encode("false").tolist()[0]
    }
    function_names = [function.name for function in functions]
    trie = build_trie(function_names, model)
    for prompt in prompts:
        output_tokens: list[int] = []
        output_tokens.extend(encoded_lits["lit_open"])
        output_tokens.extend(encoded_lits["lit_prompt"])
        prompt_tokens = model.encode(prompt.prompt).tolist()[0]
        output_tokens.extend(prompt_tokens)
        output_tokens.extend(encoded_lits["lit_name"])
        selected_name = emit_function_name(trie, model, output_tokens)
        selected_function = next(
            f for f in functions if f.name == selected_name)
        output_tokens.extend(encoded_lits["lit_parameters"])
        for param_type in selected_function.parameters.values():
            emit_parameter_type(param_type.type, model, output_tokens,
                                digit_tokens, bool_tokens, string_tokens)
        output_tokens.extend(encoded_lits["lit_close_entry"])
        output_tokens.extend(encoded_lits["lit_close"])
        decoded_output = model.decode(output_tokens)
        print(decoded_output)
