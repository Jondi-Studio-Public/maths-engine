"""Tests `f10-mixed-numerals-number-line` (topic `fr2b_mixed_numerals_number_line`, answer kind `fraction`).

The oracle reads the marks and the tick step out of the figure spec (never the key), rebuilds each marked value as
an exact Fraction and compares it with the key.
"""

from __future__ import annotations

from collections import Counter
from fractions import Fraction

import pytest

import mathgen.generators.fr2b_mixed_numerals_number_line_build  # noqa: F401  (registers the topic)
from mathgen import figures, make
from mathgen.engine import grade, load_skills
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects

TOPIC = "fr2b_mixed_numerals_number_line"
SKILL_ID = "f10-mixed-numerals-number-line"
TIERS = ("easy", "medium", "hard")
SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


def _points(q):
    """{label: exact value} of the lettered marks, from the figure alone."""
    return {m["label"]: Fraction(m["at"]).limit_denominator(12) for m in q.figure["marks"] if m["label"].isalpha()}


def _steps(q):
    return round(1 / q.figure["tick"])


def _mixed(value):
    whole, rest = divmod(value.numerator, value.denominator)
    return (
        f"{whole} {rest}/{value.denominator}"
        if whole and rest
        else (f"{rest}/{value.denominator}" if rest else str(whole))
    )


def test_wiring():
    assert SKILL.answer == "fraction"
    assert SKILL.needs_figure and "figure_stimulus" in SKILL.capability_gates
    assert [(s.topic, s.difficulty) for s in SKILL.sources] == [(TOPIC, t) for t in TIERS]


@pytest.mark.parametrize("tier", ("easy", "medium"))
def test_single_point_key_is_read_from_the_figure(tier):
    denominators = Counter()
    for seed in range(150):
        q = _q(tier, seed)
        n = _steps(q)
        value = _points(q)["P"]
        assert value * n == int(value * n) and value > 1, (seed, value)
        assert q.answer_text == _mixed(value), (seed, q.answer_text, value)
        assert grade(SKILL, q, q.answer_text).correct
        denominators[n] += 1
        spec = q.figure
        assert spec["kind"] == "number_line" and spec["tick_labels"] is False
        labelled = [m["label"] for m in spec["marks"] if not m["label"].isalpha()]
        assert labelled == (
            [str(spec["range"][0]), str(spec["range"][1])]
            if tier == "medium"
            else [str(w) for w in range(spec["range"][0], spec["range"][1] + 1)]
        )
    assert set(denominators) == ({2, 4} if tier == "easy" else {3, 5}), denominators


def test_hard_key_is_the_distance_between_the_two_points():
    denominators = Counter()
    for seed in range(150):
        q = _q("hard", seed)
        points = _points(q)
        diff = points["Q"] - points["P"]
        assert diff > 1 and diff.denominator != 1
        assert q.answer_text == _mixed(diff), (seed, q.answer_text, diff)
        assert points["Q"] % 1 < points["P"] % 1, "the subtraction must regroup"
        denominators[_steps(q)] += 1
    assert len(denominators) >= 3, denominators


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_and_every_distractor_rejected(tier):
    for seed in range(120):
        q = _q(tier, seed)
        assert grade(SKILL, q, q.answer_text).correct, (seed, q.answer_text)
        assert len(q.distractors) >= 3
        for text in q.distractors:
            assert not grade(SKILL, q, text).correct, (seed, text, q.answer_text)
            assert q.option_notes[text] in CATALOGUE


@pytest.mark.parametrize("tier", TIERS)
def test_figure_is_valid_and_does_not_leak_the_answer(tier):
    reader = figures.get_reader(TOPIC)
    for seed in range(100):
        q = _q(tier, seed)
        assert figures.validate(q.figure) == []
        figures.assert_no_leak(q.figure, [q.answer_text])
        assert grade(SKILL, q, reader(q)).correct


def test_a_mixed_numeral_may_be_typed_as_an_improper_fraction():
    for seed in range(40):
        q = _q("easy", seed)
        value = _points(q)["P"]
        assert grade(SKILL, q, f"{value.numerator}/{value.denominator}").correct


@pytest.mark.parametrize("tier", TIERS)
def test_text_quality_and_determinism(tier):
    for seed in range(60):
        q = _q(tier, seed)
        assert find_text_defects(q, include_working=True) == [], (seed, find_text_defects(q, include_working=True))
        assert q.to_json() == _q(tier, seed).to_json()


def test_variety_counts_the_figure():
    for tier in TIERS:
        assert len({(_q(tier, s).text, str(_q(tier, s).figure)) for s in range(60)}) >= 40
