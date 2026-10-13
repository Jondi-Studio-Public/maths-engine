"""Tests `f10-mental-compensation-partition` (topic `fr2b_mental_compensation`), a two-part question: the value the
strategy gives, then the answer that follows from it.

The oracle reads the calculation back out of the stem and rebuilds both keys by direct arithmetic.
"""

from __future__ import annotations

import re
from collections import Counter

import pytest

import mathgen.generators.fr2b_mental_compensation_build  # noqa: F401  (registers the topic)
from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects

TOPIC = "fr2b_mental_compensation"
SKILL_ID = "f10-mental-compensation-partition"
TIERS = ("easy", "medium", "hard")
SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


def _calc(q):
    a, op, b = re.search(r"work out (\d+) ([+-]) (\d+)", q.stem).groups()
    return int(a), op, int(b)


def _strategy(q):
    if "round both" in q.stem:
        return "both"
    return "round-one" if "round" in q.stem else "partition"


def test_wiring():
    assert SKILL.answer == "number"
    assert [(s.topic, s.difficulty) for s in SKILL.sources] == [(TOPIC, t) for t in TIERS]


@pytest.mark.parametrize("tier", TIERS)
def test_both_parts_match_direct_arithmetic(tier):
    strategies = Counter()
    for seed in range(200):
        q = _q(tier, seed)
        a, op, b = _calc(q)
        exact = a + b if op == "+" else a - b
        assert int(q.parts[1].answer) == exact and exact > 0, (seed, q.stem)
        strategy = _strategy(q)
        strategies[(strategy, op)] += 1
        part_a = int(q.parts[0].answer)
        if strategy == "round-one":
            r = int(re.search(r"up to (\d+)", q.stem).group(1))
            assert part_a == (a + r if op == "+" else a - r) and 1 <= r - b <= 3
            assert q.parts[1].follow_through == (f"part_a - {r - b}" if op == "+" else f"part_a + {r - b}")
        elif strategy == "partition":
            tens = (b // 10) * 10
            assert part_a == (a + tens if op == "+" else a - tens)
            ones = b % 10
            assert q.parts[1].follow_through == (f"part_a + {ones}" if op == "+" else f"part_a - {ones}")
            if op == "-":
                assert ones > a % 10, "the ones must bridge a ten"
        else:
            ra, rb = round(a, -2), round(b, -2)
            assert part_a == (ra + rb if op == "+" else ra - rb)
            adjust = abs(ra - a) + abs(rb - b)
            assert q.parts[1].follow_through == (f"part_a - {adjust}" if op == "+" else f"part_a + {adjust}")
    if tier == "easy":
        assert set(strategies) == {("round-one", "+"), ("partition", "+")}, strategies
    if tier == "medium":
        assert set(strategies) == {("round-one", "-"), ("partition", "-")}, strategies
    if tier == "hard":
        assert {s for s, _ in strategies} == {"round-one", "both"}, strategies
        assert ("both", "-") in strategies and ("both", "+") in strategies


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_and_every_distractor_rejected(tier):
    tags = set()
    for seed in range(120):
        q = _q(tier, seed)
        assert [p.answer_kind for p in q.parts] == ["number", "number"]
        full = grade(SKILL, q, q.answer_text)
        assert full.correct and full.marks == full.max_marks == 2
        canon = {p.label: p.answer_text for p in q.parts}
        for p in q.parts:
            assert len(p.distractors) >= 3
            for text in p.distractors:
                result = grade(SKILL, q, {**canon, p.label: text})
                assert result.parts[p.label].marks == 0, (seed, p.label, text)
                assert p.option_notes[text] in CATALOGUE
                tags.add(p.option_notes[text])
    assert len(tags) >= 5, tags


@pytest.mark.parametrize("tier", TIERS)
def test_follow_through_from_a_wrong_first_part(tier):
    for seed in range(60):
        q = _q(tier, seed)
        canon = {p.label: p.answer_text for p in q.parts}
        right_a = int(q.parts[0].answer)
        shift = int(q.parts[1].answer) - right_a
        wrong_a = right_a + 10
        result = grade(SKILL, q, {**canon, "a": str(wrong_a), "b": str(wrong_a + shift)})
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == 1 and result.parts["b"].follow_through is True, (seed, result)
        # Forgetting the adjustment altogether is not follow-through.
        assert grade(SKILL, q, {**canon, "b": str(right_a)}).parts["b"].marks == 0


@pytest.mark.parametrize("tier", TIERS)
def test_text_quality_and_determinism(tier):
    for seed in range(60):
        q = _q(tier, seed)
        assert find_text_defects(q, include_working=True) == [], (seed, find_text_defects(q, include_working=True))
        assert q.to_json() == _q(tier, seed).to_json()


def test_variety():
    for tier in TIERS:
        assert len({_q(tier, s).text for s in range(80)}) >= 60
