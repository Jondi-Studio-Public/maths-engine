"""Tests the exam chain for ``f10-probability-tree-diagrams``: two missing branches, then two path probabilities."""

from __future__ import annotations

from fractions import Fraction

import pytest

from mathgen import figures, make
from mathgen.core import rational_text
from mathgen.generators import _space_kit as kit
from mathgen.generators.fig_tree_build import _complete
from tests.exd_x_r2c_support import (
    canonical,
    check_answer_space,
    check_clean_and_deterministic,
    check_follow_through,
    check_grading,
    check_shapes,
    check_wired,
    exam,
)

TOPIC = "exd_x_fc_probability_tree"
SKILL_ID = "f10-probability-tree-diagrams"


@pytest.fixture(scope="module")
def questions():
    return [exam(TOPIC, s) for s in range(200)]


def oracle(q):
    """Every key, re-derived from the drawn tree alone (a blank branch is one minus its siblings)."""
    first, second = _complete(q.figure)
    both_first = first[0] * second[0][0]
    return first[0], second[1][1], both_first, both_first + first[1] * second[1][1]


def test_clean_and_deterministic():
    check_clean_and_deterministic(TOPIC)


def test_shape_and_figure(questions):
    for q in questions[:60]:
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.answer_kind for p in q.parts] == ["fraction"] * 4
        assert [p.marks for p in q.parts] == [1, 1, 2, 2]
        assert [p.follow_through is not None for p in q.parts] == [False, False, True, True]
        assert q.figure["kind"] == "tree_diagram"
        assert figures.validate(q.figure) == []


def test_two_blank_branches_in_different_sets(questions):
    for q in questions[:60]:
        root = q.figure["root"]["children"]
        blanks = [(i,) for i, n in enumerate(root) if n["branch"] == "?"]
        blanks += [(i, j) for i, n in enumerate(root) for j, k in enumerate(n["children"]) if k["branch"] == "?"]
        assert blanks == [(0,), (1, 1)], q.text


def test_oracle_agrees_over_200_seeds(questions):
    bad = []
    for q in questions:
        keys = oracle(q)
        got = tuple(Fraction(p.answer_text) for p in q.parts)
        if got != keys:
            bad.append((q.text, got, keys))
    assert bad == [], bad[:2]


def test_stem_names_the_colours_on_the_tree(questions):
    for q in questions[:60]:
        labels = [n["label"].lower() for n in q.figure["root"]["children"]]
        assert all(c in q.text for c in labels), (labels, q.text)
        assert f"{labels[0]}, {labels[0]}" in q.parts[2].working[0] or labels[0] in q.parts[2].working[0]


def test_no_answer_appears_on_the_figure(questions):
    for q in questions:
        for p in q.parts:
            assert not kit.leaks(q.figure, p.answer_text), (q.text, p.label, p.answer_text)


def test_grading_and_distractors():
    check_grading(SKILL_ID, TOPIC, 120)


def test_follow_through_from_wrong_first_branch():
    """Using the other colour's probability for (a) carries through (c), and (d) when (b) is right."""

    def carried(q):
        first, second = _complete(q.figure)
        wrong_a = 1 - first[0]
        wrong_c = wrong_a * second[0][0]
        wrong_d = wrong_c + first[1] * second[1][1]
        return "a", rational_text(wrong_a), {"c": rational_text(wrong_c), "d": rational_text(wrong_d)}

    check_follow_through(SKILL_ID, TOPIC, carried, 60)


def test_follow_through_from_wrong_second_branch():
    def carried(q):
        first, second = _complete(q.figure)
        wrong_b = second[1][0]  # the sibling branch
        wrong_d = first[0] * second[0][0] + first[1] * wrong_b
        return "b", rational_text(wrong_b), {"d": rational_text(wrong_d)}

    check_follow_through(SKILL_ID, TOPIC, carried, 60)
