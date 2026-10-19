"""Tests the exam source of `annulus-cross-section-area`: metal area, volume and mass of a hollow tube."""

from __future__ import annotations

import math
import re
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_m1_pipe_mass"
SKILL_ID = "annulus-cross-section-area"
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)
_PI = Fraction(314159, 100000)


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _round(x: Fraction, step: int = 1) -> Fraction:
    return Fraction(math.floor(x / step + Fraction(1, 2))) * step


def _read(text: str):
    outer = re.search(r"outer diameter of ([\d.]+) cm and a wall thickness of ([\d.]+) cm", text)
    hole = re.search(r"hole with a diameter of ([\d.]+) cm and a wall thickness of ([\d.]+) cm", text)
    if outer:
        d_out, wall = Fraction(outer.group(1)), Fraction(outer.group(2))
    else:
        d_out, wall = Fraction(hole.group(1)) + 2 * Fraction(hole.group(2)), Fraction(hole.group(2))
    length = int(re.search(r"is (\d+) cm long", text).group(1))
    density = Fraction(re.search(r"density of ([\d.]+) g/cm", text).group(1))
    return d_out, wall, length, density


def _oracle(q) -> dict[str, str]:
    """Re-derive every part's answer from the rendered prompt alone, carrying each rounded answer forward."""
    d_out, wall, length, density = _read(q.text)
    big, small = d_out / 2, d_out / 2 - wall
    area = _round(_PI * (big**2 - small**2) * 10) / 10
    vol = _round(area * length, 10)
    mass = _round(vol * density / 1000)
    return {"a": f"{float(area):.1f}", "b": str(int(vol)), "c": str(int(mass))}


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def test_shape():
    q = _exam(1)
    assert [p.label for p in q.parts] == ["a", "b", "c"]
    assert [p.follow_through is not None for p in q.parts] == [False, True, True]


def test_key_rederived_from_prompt():
    for seed in range(300):
        q = _exam(seed)
        assert _oracle(q) == _canonical(q), (seed, q.text)


def test_exact_route_agrees_with_chained_route():
    """A learner who keeps full precision reaches the same answers as one who carries the rounded figures."""
    for seed in range(300):
        q = _exam(seed)
        d_out, wall, length, density = _read(q.text)
        big, small = d_out / 2, d_out / 2 - wall
        vol = _PI * (big**2 - small**2) * length
        assert str(int(_round(vol, 10))) == q.parts[1].answer_text, (seed, q.text)
        assert str(int(_round(vol * density / 1000))) == q.parts[2].answer_text, (seed, q.text)


def test_prompt_does_not_contain_a_key():
    for seed in range(300):
        q = _exam(seed)
        numbers = set(re.findall(r"\d+(?:\.\d+)?", q.stem + " ".join(p.prompt for p in q.parts)))
        assert not numbers & {p.answer_text for p in q.parts}, (seed, q.text)


@pytest.fixture(scope="module")
def questions():
    return [_exam(s) for s in range(120)]


def test_composed_answer_full_marks(questions):
    for q in questions:
        result = grade(_SKILL, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks == 5


def test_distractors_rejected(questions):
    shown = 0
    for q in questions:
        canon = _canonical(q)
        for p in q.parts:
            assert p.distractors, (q.text, p.label)
            for d in p.distractors:
                shown += 1
                assert grade(_SKILL, q, {**canon, p.label: d}).parts[p.label].marks == 0, (q.text, p.label, d)
    assert shown >= 300


def test_follow_through_from_wrong_area(questions):
    for q in questions:
        _, _, length, density = _read(q.text)
        a_wrong = q.parts[0].distractors[0]
        vol = _round(Fraction(a_wrong) * length, 10)
        mass = _round(vol * density / 1000)
        result = grade(_SKILL, q, {"a": a_wrong, "b": str(int(vol)), "c": str(int(mass))})
        assert result.parts["a"].marks == 0
        for label in "bc":
            assert result.parts[label].marks == result.parts[label].max_marks, (q.text, label, a_wrong, vol, mass)
            if {"b": str(int(vol)), "c": str(int(mass))}[label] != q.parts["abc".index(label)].answer_text:
                assert result.parts[label].follow_through is True
        assert not result.correct
