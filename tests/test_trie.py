"""Tests for the Trie data structure and trie construction."""


import pytest
from llm_sdk import Small_LLM_Model  # type: ignore
from src.trie import Trie, TrieNode


@pytest.fixture(scope="module")
def model() -> Small_LLM_Model:
    """Load the model once for the whole test module (expensive)."""
    return Small_LLM_Model()


def test_insert_marks_correct_node_finished() -> None:
    """Verify that insert marks only the final node as finished."""
    trie = Trie()
    trie.insert([1, 2, 3], "fn_test")
    after_first = trie.root.children[1]
    after_second = after_first.children[2]
    after_third = after_second.children[3]
    assert after_first.finished is False
    assert after_second.finished is False
    assert after_third.finished is True


def test_insert_stores_name_at_finished_node() -> None:
    """Verify that the function name is stored at the finished node."""
    trie = Trie()
    trie.insert([1, 2, 3], "fn_test")
    leaf_node = trie.root.children[1].children[2].children[3]
    assert leaf_node.finished is True
    assert leaf_node.name == "fn_test"


def test_insert_shares_common_prefix() -> None:
    """Verify that two token sequences with a shared prefix merge onto
    the same path in the trie."""
    trie = Trie()
    trie.insert([1, 2, 3], "fn_a")
    trie.insert([1, 4, 5], "fn_b")
    assert set(trie.root.children.keys()) == {1}
    shared_node = trie.root.children[1]
    assert set(shared_node.children.keys()) == {2, 4}


def test_children_reflect_legal_next_tokens_at_branch_point() -> None:
    """Verify that a shared node's children reflect all legal next
    tokens for the diverging candidates."""
    trie = Trie()
    trie.insert([1, 2, 3], "fn_a")
    trie.insert([1, 4, 5], "fn_b")
    shared_node = trie.root.children[1]
    assert set(shared_node.children.keys()) == {2, 4}


def test_leaf_node_has_no_children() -> None:
    """Verify that a finished node has no children."""
    trie = Trie()
    trie.insert([1, 2, 3], "fn_test")
    leaf_node = trie.root.children[1].children[2].children[3]
    assert leaf_node.children == {}


def test_build_trie_contains_all_function_names(
        model: Small_LLM_Model) -> None:
    """Verify that build_trie inserts every provided function name."""
    from src.trie import build_trie
    function_names = ["fn_add_numbers", "fn_greet", "fn_reverse_string"]
    trie = build_trie(function_names, model)
    found_names = set()

    def collect_names(node: TrieNode) -> None:
        if node.finished:
            found_names.add(node.name)
        for child in node.children.values():
            collect_names(child)

    collect_names(trie.root)
    assert found_names == set(function_names)


def test_build_trie_handles_prefix_relationship(
        model: Small_LLM_Model) -> None:
    """Verify that build_trie correctly stores both a function name and
    a longer name that has it as a prefix."""
    from src.trie import build_trie
    function_names = ["fn_get", "fn_get_current_time"]
    trie = build_trie(function_names, model)
    found_names = set()

    def collect_names(node: TrieNode) -> None:
        if node.finished:
            found_names.add(node.name)
        for child in node.children.values():
            collect_names(child)

    collect_names(trie.root)
    assert found_names == set(function_names)
