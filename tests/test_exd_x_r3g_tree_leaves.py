"""Tests for the multi-part exam source `exd_x_r3g_tree_leaves` (tree: leaves, vertices, edges, edges to delete)."""

from __future__ import annotations

import re

from tests import exd_x_r3g_support as sup

TOPIC = "exd_x_r3g_tree_leaves"
SKILL = "ucv1-tree-spanning-edge-count"


def _derive(q):
    """Leaves, vertices, edges and extra edges from the printed structure alone."""
    total = int(re.search(r"has (\d+) edges\. How many", q.text).group(1))
    degrees = re.findall(r"(\d+) vert(?:ex|ices) of degree (\d)", q.text)
    if degrees:
        # The degrees sum to 2 (n - 1) with n = high + L:  weight + L = 2 (high + L - 1).
        high = sum(int(k) for k, _ in degrees)
        weight = sum(int(k) * int(d) for k, d in degrees)
        leaves = weight - 2 * high + 2
    else:
        # A full m-ary tree with i internal vertices has m i + 1 vertices in all.
        m = int(re.search(r"exactly (\d+) children", q.text).group(1))
        high = int(re.search(r"(\d+) internal vertices", q.text).group(1))
        leaves = m * high + 1 - high
    vertices = high + leaves
    return leaves, vertices, vertices - 1, total - (vertices - 1)


def test_wired():
    sup.check_wired(SKILL, TOPIC)


def test_variety_and_new_shapes():
    sup.check_variety(TOPIC, "p14b_tree_edge_count", {})


def test_clean_and_deterministic():
    sup.check_clean_and_deterministic(TOPIC)


def test_shape():
    for q in sup.questions(TOPIC, 40):
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True, True]


def test_keys_match_independent_derivation_over_300_seeds():
    bad = []
    for q in sup.questions(TOPIC):
        leaves, vertices, edges, extra = _derive(q)
        want = {"a": str(leaves), "b": str(vertices), "c": str(edges), "d": str(extra)}
        if sup.canonical(q) != want or leaves < 3 or extra < 2:
            bad.append((q.text, sup.canonical(q), want))
    assert bad == [], bad[:2]


def test_both_stems_occur():
    texts = [q.text for q in sup.questions(TOPIC, 60)]
    assert any("of degree" in t for t in texts) and any("children" in t for t in texts)


def test_answer_space():
    sup.check_answer_space(TOPIC)


def test_grading_and_format():
    sup.check_grading(SKILL, TOPIC)


def test_tags_are_catalogue_ids():
    sup.check_tagged(TOPIC)


def test_follow_through_from_a_wrong_leaf_count():
    def replies(q):
        leaves, vertices, edges, extra = _derive(q)
        wrong = int(q.parts[0].distractors[0])
        v = wrong + (vertices - leaves)
        total = extra + edges
        if total - (v - 1) == v - 1:  # a reply that repeats an earlier reply earns no follow-through
            return None
        return {"a": str(wrong), "b": str(v), "c": str(v - 1), "d": str(total - (v - 1))}, {"b", "c", "d"}, {"a"}

    sup.check_follow_through(SKILL, TOPIC, replies)


def test_follow_through_from_a_wrong_edge_count():
    def replies(q):
        leaves, vertices, edges, extra = _derive(q)
        wrong = int(q.parts[2].distractors[0])
        return {**sup.canonical(q), "c": str(wrong), "d": str(edges + extra - wrong)}, {"d"}, {"c"}

    sup.check_follow_through(SKILL, TOPIC, replies)
