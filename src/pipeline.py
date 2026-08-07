"""mambo jambo"""


from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
from .decoder import (
    literals_encoder,
    encode_number,
    emit_function_name,
    emit_parameter_type,
)
from .trie import Trie


def run_pipeline(prompts, functions) -> None:
    """mambo"""
    model = Small_LLM_Model()
    encoded_lits = literals_encoder(model)
    digit_tokens = encode_number(model, [])
    bool_tokens = {
        "true": model.encode("true").tolist()[0],
        "false": model.encode("false").tolist()[0]
    }
    trie = Trie()
    for function in functions:
        token_ids = model.encode(function.name).tolist()[0]
        trie.insert(token_ids)
    print("Trie root children token IDs:", list(trie.root.children.keys()))
    for prompt in prompts:
        output_tokens = []
        output_tokens.extend(encoded_lits["lit_open"])
        output_tokens.extend(encoded_lits["lit_prompt"])
        prompt_tokens = model.encode(prompt.prompt).tolist()[0]
        output_tokens.extend(prompt_tokens)
        output_tokens.extend(encoded_lits["lit_name"])
        emit_function_name(trie, model, output_tokens)
        output_tokens.extend(encoded_lits["lit_parameters"])
        for function in functions:
            for param_type in function.parameters.values():
                emit_parameter_type(param_type, model, output_tokens,
                                    digit_tokens, bool_tokens)
        output_tokens.extend(encoded_lits["lit_close_entry"])
        output_tokens.extend(encoded_lits["lit_close"])
        decoded_output = model.decode(output_tokens)
        print(decoded_output)
