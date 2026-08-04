"""mambo jambo"""


from __future__ import annotations
from pydantic import BaseModel, Field

from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]


class TrieNode(BaseModel):
    """mambo"""
    children: dict[int, TrieNode] = Field(default_factory=dict)
    finished: bool = False


class Trie(BaseModel):
    """mambo jambo"""
    root: TrieNode = Field(default_factory=TrieNode)

    def insert(self, token_ids: list[int]) -> None:
        """mambo"""
        current = self.root
        for id in token_ids:
            if id not in current.children:
                current.children[id] = TrieNode()
            current = current.children[id]
        current.finished = True

    def next_tokens(self, current: TrieNode) -> set[int]:
        """jambo"""
        return set(current.children.keys())


def build_trie(functions: list[str], model: Small_LLM_Model) -> Trie:
    """mambo"""
    trie = Trie()
    for function in functions:
        token_ids: list[int] = model.encode(function).tolist()[0]
        trie.insert(token_ids)
    return trie
