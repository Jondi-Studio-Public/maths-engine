"""Tests `senior-define-the-reciprocal-trigonometric-functions` (topic `salg_reciprocal_trig_values`) and its exam chain
`salg_exam_reciprocal_trig_values`. The oracle reads the angle back out of the question text and evaluates the
function with SymPy and with the `math` module."""

from __future__ import annotations

import math
import re
from collections import Counter

import pytest
import sympy as sp

import mathgen.generators.salg_reciprocal_trig_values_build  # noqa: F401  (registers the topics)
from mathgen import make
from salg_support import check_exam, check_follow_through, check_single, parse, skill

TOPIC = "salg_reciprocal_trig_values"
EXAM = "salg_exam_reciprocal_trig_values"
SKILL = skill("senior-define-the-reciprocal-trigonometric-functions")
TIERS = ("easy", "medium", "hard")
_FLOAT = {
    "sec": lambda r: 1 / math.cos(r),
    "cosec": lambda r: 1 / math.sin(r),
    "cot": lambda r: math.cos(r) / math.sin(r),
    "sin": math.sin,
    "cos": math.cos,
}


def _angle(text):
    """(function, angle in radians as a float and as an exact multiple of pi) from a stem."""
    fn, arg = re.search(r"\b(sec|cosec|cot) (\(?[-\dπ/]+°?\)?)", text).groups()
    arg = arg.strip("()")
    if arg.endswith("°"):
        frac = sp.Rational(int(arg[:-1]), 180)
    else:
        m = re.fullmatch(r"(-?)(\d*)π(?:/(\d+))?", arg)
        sign, num, den = m.groups()
        frac = sp.Rational(int(num or 1), int(den or 1)) * (-1 if sign else 1)
    return fn, float(frac) * math.pi, frac


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


def test_wiring():
    assert SKILL.answer == "expression"
    assert [(s.topic, s.difficulty) for s in SKILL.sources] == [
        (TOPIC, "easy"),
        (TOPIC, "medium"),
        (TOPIC, "hard"),
        (EXAM, "exam"),
    ]
    assert SKILL.sources[-1].stages == ["exam"]
    assert SKILL.domain == "trigonometry"


@pytest.mark.parametrize("tier", TIERS)
def test_key_matches_float_evaluation(tier):
    for seed in range(150):
        q = _q(tier, seed)
        fn, rad, _frac = _angle(q.text)
        got = float(parse(q.answer_text))
        assert abs(got - _FLOAT[fn](rad)) < 1e-9, (tier, seed, q.text, q.answer_text)


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_and_every_distractor_rejected(tier):
    check_single(SKILL, TOPIC, tier, seeds=100)


def test_tiers_differ_in_shape():
    easy = [_angle(_q("easy", s).text)[2] for s in range(80)]
    assert {a * 180 for a in easy} == {30, 45, 60}
    medium = [_angle(_q("medium", s).text)[2] for s in range(80)]
    assert all(0.5 < a < 2 and a not in (1, sp.Rational(3, 2)) for a in medium)
    hard = [_q("hard", s).text for s in range(80)]
    assert all("π" in t and "°" not in t for t in hard)
    fracs = [_angle(t)[2] for t in hard]
    assert any(f < 0 for f in fracs) and any(f > 2 for f in fracs)
    assert {_angle(_q("easy", s).text)[0] for s in range(80)} == {"sec", "cosec", "cot"}


def test_signs_follow_the_quadrant():
    for seed in range(150):
        q = _q("medium", seed)
        fn, rad, frac = _angle(q.text)
        value = float(parse(q.answer_text))
        deg = float(frac) * 180
        sin_pos, cos_pos = 0 < deg < 180, deg < 90 or deg > 270
        want = {"sec": cos_pos, "cosec": sin_pos, "cot": sin_pos == cos_pos}[fn]
        assert (value > 0) == want, (seed, q.text, q.answer_text)


@pytest.mark.parametrize("tier", TIERS)
def test_answer_space(tier):
    answers = [_q(tier, s).answer_text for s in range(120)]
    assert len(set(answers)) >= (5 if tier == "easy" else 7), Counter(answers)
    texts = {_q(tier, s).text for s in range(120)}
    assert len(texts) >= (20 if tier == "easy" else 40)


# -- exam chain ---------------------------------------------------------------------------------------------------


def test_exam_shape_and_marks():
    for seed in range(40):
        q = make(EXAM, seed=seed, difficulty="exam")
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.marks for p in q.parts] == [1, 1, 2, 2]
        assert all(p.answer_kind == "expression" for p in q.parts)
        assert [p.follow_through for p in q.parts] == [None, None, "1/part_a", "part_b/part_a"]


def test_exam_keys_match_independent_derivation():
    seen = set()
    for seed in range(150):
        q = make(EXAM, seed=seed, difficulty="exam")
        m = re.fullmatch(r"Let θ = (-?)(\d*)π(?:/(\d+))?\.", q.text.split(" (a)")[0])
        sign, num, den = m.groups()
        rad = float(sp.Rational(int(num or 1), int(den or 1))) * math.pi * (-1 if sign else 1)
        seen.add(q.text.split(" (a)")[0])
        want = [math.sin(rad), math.cos(rad), 1 / math.sin(rad), math.cos(rad) / math.sin(rad)]
        for part, w in zip(q.parts, want):
            assert abs(float(parse(part.answer_text)) - w) < 1e-9, (seed, part.label)
        # the angle is not in the first quadrant
        assert not (0 < (rad / math.pi) % 2 < 0.5)
    assert len(seen) >= 15


def test_exam_key_accepted_and_every_distractor_rejected():
    check_exam(SKILL, EXAM, seeds=60)


def test_exam_follow_through_from_a_wrong_sine():
    def wrong(q):
        s = parse(q.parts[0].answer_text)
        c = parse(q.parts[1].answer_text)
        bad = -s  # sign slip in (a)
        return str(bad).replace("**", "^"), {
            "c": str(1 / bad).replace("**", "^"),
            "d": str(sp.radsimp(c / bad)).replace("**", "^"),
        }

    check_follow_through(SKILL, EXAM, "a", wrong, ["c", "d"])
