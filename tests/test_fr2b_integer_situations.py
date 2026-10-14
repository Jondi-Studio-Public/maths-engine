"""Tests `f10-integer-situations-number-line` (topic `fr2b_integer_situations`).

The oracle reads the marks out of the figure spec and the amounts out of the wording, never the key.
"""

from __future__ import annotations

import re
from collections import Counter

import pytest

import mathgen.generators.fr2b_integer_situations_build  # noqa: F401  (registers the topic)
from mathgen import figures, make
from mathgen.engine import grade, load_skills
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects

TOPIC = "fr2b_integer_situations"
SKILL_ID = "f10-integer-situations-number-line"
TIERS = ("easy", "medium", "hard")
SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)

_UP = ("rises", "swims up", "goes up")


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


def _marks(q):
    return {m["label"]: m["at"] for m in q.figure["marks"]}


def test_wiring():
    assert SKILL.answer == "number"
    assert SKILL.needs_figure and "figure_stimulus" in SKILL.capability_gates
    assert [(s.topic, s.difficulty) for s in SKILL.sources] == [(TOPIC, t) for t in TIERS]


def test_easy_key_is_the_marked_value():
    signs = Counter()
    for seed in range(200):
        q = _q("easy", seed)
        marks = _marks(q)
        assert set(marks) == {"A"} and q.answer == marks["A"] != 0, (seed, q.text)
        signs[marks["A"] < 0] += 1
        assert q.figure["tick_labels"] is True
    assert min(signs.values()) >= 60, signs


def test_medium_key_is_the_distance_from_a_to_b():
    kinds = Counter()
    for seed in range(200):
        q = _q("medium", seed)
        a, b = (_marks(q)[k] for k in "AB")
        assert a < b and q.answer == b - a > 0, (seed, q.text)
        kinds["across zero" if a < 0 < b else "one side"] += 1
        if a < 0 < b:
            assert q.answer == abs(a) + abs(b)
        else:
            assert b < 0 and q.answer == abs(a) - abs(b)
    assert min(kinds.values()) >= 40, kinds


def test_hard_key_follows_the_moves_or_the_midpoint():
    kinds = Counter()
    for seed in range(250):
        q = _q("hard", seed)
        marks = _marks(q)
        if "halfway" in q.text:
            a, b = marks["A"], marks["B"]
            assert q.answer * 2 == a + b and a < q.answer < b, (seed, q.text)
            kinds["midpoint"] += 1
        else:
            moves = re.findall(r"(rises|swims up|goes up|falls|dives down|goes down) by (\d+)", q.text)
            assert len(moves) == 2 and (moves[0][0] in _UP) != (moves[1][0] in _UP)
            path = [marks["A"]]
            for verb, n in moves:
                path.append(path[-1] + (int(n) if verb in _UP else -int(n)))
            assert q.answer == path[-1], (seed, q.text)
            assert min(path) < 0 < max(path), "zero must be crossed on the way"
            kinds["moves"] += 1
    assert min(kinds.values()) >= 60, kinds


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_and_every_distractor_rejected(tier):
    tags = set()
    for seed in range(120):
        q = _q(tier, seed)
        assert grade(SKILL, q, q.answer_text).correct, (seed, q.answer_text)
        assert len(q.distractors) >= 3
        for text in q.distractors:
            assert not grade(SKILL, q, text).correct, (seed, text, q.answer_text)
            assert "." not in text
            assert q.option_notes[text] in CATALOGUE
            tags.add(q.option_notes[text])
    assert len(tags) >= 3, tags


def test_a_negative_answer_may_use_a_unicode_minus_and_a_unit():
    for seed in range(80):
        q = _q("easy", seed)
        if q.answer < 0:
            assert grade(SKILL, q, f"−{abs(q.answer)}").correct
            assert not grade(SKILL, q, str(abs(q.answer))).correct


@pytest.mark.parametrize("tier", TIERS)
def test_figure_is_valid_does_not_leak_and_the_reader_agrees(tier):
    reader = figures.get_reader(TOPIC)
    for seed in range(120):
        q = _q(tier, seed)
        assert figures.validate(q.figure) == []
        figures.assert_no_leak(q.figure, [q.answer_text])
        assert grade(SKILL, q, reader(q)).correct
        low, high = q.figure["range"]
        assert all(low <= m["at"] <= high for m in q.figure["marks"])


@pytest.mark.parametrize("tier", TIERS)
def test_text_quality_and_determinism(tier):
    for seed in range(60):
        q = _q(tier, seed)
        assert find_text_defects(q, include_working=True) == [], (seed, find_text_defects(q, include_working=True))
        assert q.to_json() == _q(tier, seed).to_json()


def test_variety_counts_the_figure():
    for tier in TIERS:
        keys = {(_q(tier, s).text, str(_q(tier, s).figure)) for s in range(80)}
        assert len(keys) >= 40, (tier, len(keys))
