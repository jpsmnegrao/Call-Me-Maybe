"""mambo jambo"""


from __future__ import annotations
from pydantic import BaseModel, Field

from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]


class TrieNode(BaseModel):
    """Represent a node in a token-based trie.

    Each node stores its child tokens, whether it marks the end of a function
    name, and the function name associated with that completed path.
    """
    children: dict[int, TrieNode] = Field(default_factory=dict)
    finished: bool = False
    name: str | None = None


class Trie(BaseModel):
    """Represent a trie used to store tokenized function names.

    The trie allows function names to be matched token by token during
    constrained generation.
    """
    root: TrieNode = Field(default_factory=TrieNode)

    def insert(self, token_ids: list[int], name: str) -> None:
        """Insert a tokenized function name into the trie.

        Args:
            token_ids: Token IDs representing the function name.
            name: Function name associated with the token sequence.
        """
        current = self.root
        for token_id in token_ids:
            if token_id not in current.children:
                current.children[token_id] = TrieNode()
            current = current.children[token_id]
        current.finished = True
        current.name = name


def build_trie(functions: list[str], model: Small_LLM_Model) -> Trie:
    """Build a trie containing the tokenized function names.

    Args:
        functions: Function names to insert into the trie.
        model: LLM model used to tokenize the function names.

    Returns:
        A trie containing all provided function names.
    """
    trie = Trie()
    for function in functions:
        token_ids: list[int] = model.encode(function).tolist()[0]
        trie.insert(token_ids, function)
    return trie
