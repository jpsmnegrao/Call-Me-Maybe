"""Tests for the Trie data structure and trie construction."""


from src.trie import Trie


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


# TODO: Add tests for build_trie(). Currently the tests cover Trie.insert()
# directly, but do not verify build_trie() itself, including the case where
# one function name is a prefix of another. Decide later how best to test this
# without making the test dependent on the real model/tokenizer.
