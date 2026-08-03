"""mambo jambo"""


from pydantic import BaseModel, Field


class TrieNode(BaseModel):
    """mambo"""

    def __init__(self) -> None:
        self.children: dict[int, TrieNode] = Field(default_factory=dict)
        self.finished: bool = False


class Trie:
    """mambo jambo"""

    def __init__(self) -> None:
        """mammbo"""
        self.root: TrieNode = TrieNode()
