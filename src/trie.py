"""mambo jambo"""


from __future__ import annotations
from pydantic import BaseModel, Field

from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]


class TrieNode(BaseModel):
    """mambo"""
    children: dict[int, TrieNode] = Field(default_factory=dict)
    finished: bool = False
    name: str | None = None


class Trie(BaseModel):
    """mambo jambo"""
    root: TrieNode = Field(default_factory=TrieNode)

    def insert(self, token_ids: list[int], name: str) -> None:
        """mambo"""
        current = self.root
        for token_id in token_ids:
            if token_id not in current.children:
                current.children[token_id] = TrieNode()
            current = current.children[token_id]
        current.finished = True
        current.name = name


def build_trie(functions: list[str], model: Small_LLM_Model) -> Trie:
    """mambo"""
    trie = Trie()
    for function in functions:
        token_ids: list[int] = model.encode(function).tolist()[0]
        trie.insert(token_ids, function)
    return trie
