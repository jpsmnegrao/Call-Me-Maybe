"""mambo jambose"""

from src.trie import Trie


def test_insert_marks_correct_node_finished() -> None:
    """jambo"""
    trie = Trie()
    trie.insert([1, 2, 3], "fn_test")

    after_first = trie.root.children[1]
    after_second = after_first.children[2]
    after_third = after_second.children[3]

    assert after_first.finished is False
    assert after_second.finished is False
    assert after_third.finished is True


def test_insert_shares_common_prefix() -> None:
    """jambo"""
    trie = Trie()
    trie.insert([1, 2, 3], "fn_test")
    trie.insert([1, 4, 5], "fn_test")

    assert set(trie.root.children.keys()) == {1}
    shared_node = trie.root.children[1]
    assert set(shared_node.children.keys()) == {2, 4}


def test_legal_next_tokens_at_branch_point() -> None:
    """jambo"""
    trie = Trie()
    trie.insert([1, 2, 3], "fn_test")
    trie.insert([1, 4, 5], "fn_test")

    shared_node = trie.root.children[1]
    assert trie.next_tokens(shared_node) == {2, 4}


def test_legal_next_tokens_returns_empty_set_at_leaf() -> None:
    """jambo"""
    trie = Trie()
    trie.insert([1, 2, 3], "fn_test")

    leaf_node = trie.root.children[1].children[2].children[3]
    assert trie.next_tokens(leaf_node) == set()


def test_insert_stores_name_at_finished_node() -> None:
    """jambo"""
    trie = Trie()
    trie.insert([1, 2, 3], "fn_test")

    leaf_node = trie.root.children[1].children[2].children[3]
    assert leaf_node.finished is True
    assert leaf_node.name == "fn_test"
