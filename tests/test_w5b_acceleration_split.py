"""Split-instant hard tier of p10a_acceleration_to_position_line."""

from __future__ import annotations

import random
import re

import pytest
from sympy import Rational, diff, sympify, symbols

from mathgen.grading import GRADERS
from mathgen import make

TOPIC = "p10a_acceleration_to_position_line"
SKILL_ANSWER_KIND = "expression"
t = symbols("t")

# Captured BEFORE the edit (default path must be byte-identical).
OLD_TEXTS = {
    (
        "easy",
        0,
    ): "A particle moves along a straight line with acceleration a(t) = -7 m/s² at time t seconds (t ≥ 0). At time t = 0, its velocity is 3 m/s and its position is 1 m. Find an expression for its position s(t) at time t.",
    (
        "medium",
        1,
    ): "A particle moves along a straight line with acceleration a(t) = 5·t + 7 m/s² at time t seconds (t ≥ 0). At time t = 0, its velocity is -6 m/s and its position is 15 m. Find an expression for its position s(t) at time t.",
    (
        "hard",
        0,
    ): "A particle moves along a straight line with acceleration a(t) = 5·t - 4 m/s² at time t seconds (t ≥ 0). At time t = 1, its velocity is 2 m/s and its position is -1 m. Find an expression for its position s(t) at time t.",
    (
        "hard",
        2,
    ): "A particle moves along a straight line with acceleration a(t) = 3·t - 4 m/s² at time t seconds (t ≥ 0). At time t = 2, its velocity is 6 m/s and its position is -11 m. Find an expression for its position s(t) at time t.",
}
OLD_HARD_0 = (
    "5*t^3/6 - 2*t^2 + 7*t/2 - 10/3",
    ["5*t^3/6 - 2*t^2 + 7*t/2", "5*t^3/6 - 2*t^2 + 1/6", "5*t^3/6 - 2*t^2 + 7*t/2 + 4/3"],
)

_TEXT_RE = re.compile(
    r"acceleration a\(t\) = (?P<a>.+?) m/s² at time t seconds \(t ≥ 0\)\. "
    r"At time t = 0, its velocity is (?P<v0>-?\d+) m/s\. "
    r"At time t = (?P<b>\d+), its position is (?P<s0>-?\d+) m\."
)


def _expr(text: str):
    return sympify(text.replace("^", "**").replace("·", "*"))


def _split(seed):
    return make(TOPIC, seed=seed, difficulty="hard", split=True)


@pytest.mark.parametrize("key", sorted(OLD_TEXTS))
def test_default_path_unchanged(key):
    tier, seed = key
    assert make(TOPIC, seed=seed, difficulty=tier).text == OLD_TEXTS[key]


def test_default_hard_answer_and_distractors_unchanged():
    q = make(TOPIC, seed=0, difficulty="hard")
    assert q.answer_text == OLD_HARD_0[0]
    assert list(q.distractors) == OLD_HARD_0[1]


@pytest.mark.parametrize("tier", ["easy", "medium"])
def test_split_requires_hard(tier):
    with pytest.raises(ValueError):
        make(TOPIC, seed=0, difficulty=tier, split=True)


def test_split_re_derivation_300_seeds():
    for seed in range(300):
        q = _split(seed)
        m = _TEXT_RE.search(q.text)
        assert m, q.text
        a = _expr(m["a"])
        v0, b, s0 = int(m["v0"]), int(m["b"]), int(m["s0"])
        assert b in (1, 2, 3)
        V = sympify(0)
        # Integrate twice independently from the prompt only.
        from sympy import integrate

        V = integrate(a, t) + v0 - integrate(a, t).subs(t, 0)
        S0 = integrate(V, t)
        S = S0 + (s0 - S0.subs(t, b))
        assert (_expr(q.answer_text) - S).expand() == 0, (seed, q.text, q.answer_text)
        assert diff(_expr(q.answer_text), t).subs(t, 0) == v0
        assert _expr(q.answer_text).subs(t, b) == s0
        # genuinely split: position at t = 0 is not the stated s0
        assert S.subs(t, 0) != s0


def test_split_distractors_distinct_300_seeds():
    for seed in range(300):
        q = _split(seed)
        ans = _expr(q.answer_text)
        assert q.answer_text not in q.distractors
        assert len(q.distractors) == len(set(q.distractors)) >= 3
        for d in q.distractors:
            assert (_expr(d) - ans).expand() != 0, (seed, d)


def test_split_grades_through_api_grading_150_seeds():
    grader = GRADERS[SKILL_ANSWER_KIND]
    for seed in range(150):
        q = _split(seed)
        assert grader(q, q.answer_text).correct, (seed, q.answer_text)
        for d in q.distractors:
            assert not grader(q, d).correct, (seed, d)
        assert not grader(q, f"{q.answer_text}+1").correct
