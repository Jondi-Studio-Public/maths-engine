"""Tests the exam chain for ``f10-venn-diagram-counts``: a missing region, the circle total, two probabilities."""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen import figures, make
from mathgen.core import rational_text
from mathgen.generators import _space_kit as kit
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

TOPIC = "exd_x_fc_venn_counts"
SKILL_ID = "f10-venn-diagram-counts"


@pytest.fixture(scope="module")
def questions():
    return [exam(TOPIC, s) for s in range(200)]


def oracle(q):
    """Every key, re-derived from the drawn regions and the group total in the stem."""
    regions = q.figure["regions"]
    total = int(re.search(r"(\d+) (?:students|children|people)", q.stem).group(1))
    shown = {k: int(v) for k, v in regions.items() if v != "?"}
    missing = total - sum(shown.values())
    in_second = missing + shown["AB"]
    exactly_one = Fraction(shown["A"] + missing, total)
    return missing, in_second, exactly_one, Fraction(shown["AB"], in_second)


def test_source_wired():
    check_wired(SKILL_ID, TOPIC)


def test_prompt_variety_and_new_shapes_over_hard_tier():
    check_shapes(TOPIC, "fig_venn_counts")


def test_clean_and_deterministic():
    check_clean_and_deterministic(TOPIC)


def test_shape_and_figure(questions):
    for q in questions[:60]:
        assert [p.answer_kind for p in q.parts] == ["number", "number", "fraction", "fraction"]
        assert [p.marks for p in q.parts] == [1, 1, 2, 2]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True, True]
        assert q.figure["kind"] == "venn"
        assert figures.validate(q.figure) == []
        assert [k for k, v in q.figure["regions"].items() if v == "?"] == ["B"]


def test_oracle_agrees_over_200_seeds(questions):
    bad = []
    for q in questions:
        keys = oracle(q)
        got = (
            int(q.parts[0].answer_text),
            int(q.parts[1].answer_text),
            Fraction(q.parts[2].answer_text),
            Fraction(q.parts[3].answer_text),
        )
        if got != keys:
            bad.append((q.text, got, keys))
    assert bad == [], bad[:2]


def test_no_answer_appears_on_the_figure(questions):
    for q in questions:
        for p in q.parts:
            assert not kit.leaks(q.figure, p.answer_text), (q.text, p.label, p.answer_text)


def test_grading_and_distractors():
    check_grading(SKILL_ID, TOPIC, 120)


def test_answer_space():
    check_answer_space(TOPIC)


def test_follow_through_from_wrong_missing_region():
    """A missing region that is too big carries through the circle total and both probabilities."""

    def carried(q):
        regions = q.figure["regions"]
        total = int(re.search(r"(\d+) (?:students|children|people)", q.stem).group(1))
        only_a, both = int(regions["A"]), int(regions["AB"])
        wrong_a = int(q.parts[0].answer_text) + 2
        wrong_b = wrong_a + both
        return (
            "a",
            str(wrong_a),
            {
                "b": str(wrong_b),
                "c": rational_text(Fraction(only_a + wrong_a, total)),
                "d": rational_text(Fraction(both, wrong_b)),
            },
        )

    check_follow_through(SKILL_ID, TOPIC, carried, 60)


def test_follow_through_from_wrong_circle_total():
    def carried(q):
        both = int(q.figure["regions"]["AB"])
        wrong_b = int(q.parts[1].answer_text) + 3
        return "b", str(wrong_b), {"d": rational_text(Fraction(both, wrong_b))}

    check_follow_through(SKILL_ID, TOPIC, carried, 60)


def test_unknown_tier_rejected():
    with pytest.raises(ValueError):
        make(TOPIC, seed=0, tier="nonsense")
