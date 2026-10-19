"""Tests `uc2-wave-equation` (topic `s2c3_wave_equation`) and its exam chain `exd_x_s2c3_wave_equation`.

The oracle reads the problem back out of the wording. A line-string problem is re-solved with SymPy (exact integral of
the initial velocity in d'Alembert's formula); a bounded string from the modal sum, and for a sample of seeds by an
explicit leapfrog finite-difference scheme that shares nothing with the modal formula.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
import pytest
import sympy

import mathgen.generators.s2c3_wave_equation_build  # noqa: F401  (registers the topics)
from _s2c3_helpers import (
    assert_exam_grading,
    assert_follow_through,
    assert_key_accepted_and_distractors_rejected,
    assert_text_quality_and_determinism,
    skill_by_id,
    spread,
    tier_question,
)
from _s2c3_oracle import X, expr
from mathgen import make

TOPIC = "s2c3_wave_equation"
EXAM = "exd_x_s2c3_wave_equation"
SKILL = skill_by_id("uc2-wave-equation")
TIERS = ("easy", "medium", "hard")
_S = sympy.symbols("s")
_FIND = re.compile(r"Find u\(([-\d.]+), ([-\d.]+)\)")
_TERM = re.compile(r"([+-]?)\s*(\d*)\s*sin\((\d*)πx/([\d.]+)\)")


def _line_string(text):
    """(f, g, c, x0, t0) of a d'Alembert question."""
    K = int(re.search(r"= (\d+) ∂²u/∂x²", text).group(1))
    f = expr(re.search(r"u\(x, 0\) = (\d+/\(\d+ \+ x²\))", text).group(1).replace("²", "**2"))
    g = expr(re.search(r"∂u/∂t \(x, 0\) = ([^.]*?)\. Find", text).group(1).replace("²", "**2"))
    x0, t0 = (sympy.Rational(v) for v in _FIND.search(text).groups())
    return f, g, sympy.sqrt(K), x0, t0


def _dalembert(f, g, c, x0, t0):
    integral = sympy.integrate(g.subs(X, _S), (_S, x0 - c * t0, x0 + c * t0))
    return float(sympy.N((f.subs(X, x0 - c * t0) + f.subs(X, x0 + c * t0)) / 2 + integral / (2 * c), 30))


def _modes(body):
    out = {}
    for sign, amp, n, _ in _TERM.findall(body):
        out[int(n or 1)] = (-1 if sign == "-" else 1) * int(amp or 1)
    return out


class Bounded:
    def __init__(self, text):
        self.L = float(re.search(r"string of length ([\d.]+) m", text).group(1))
        m = re.search(r"c = (\d+) m/s", text)
        if m:
            self.c = float(m.group(1))
        else:
            T = float(re.search(r"T = ([\d.]+) N", text).group(1))
            rho = float(re.search(r"ρ = ([\d.]+) kg/m", text).group(1))
            self.c = math.sqrt(T / rho)
        self.A = _modes(re.search(r"u\(x, 0\) = (.*?) mm", text).group(1))
        v = re.search(r"∂u/∂t \(x, 0\) = (.*?) mm/s", text)
        self.B = _modes(v.group(1)) if v else {}

    def u(self, x, t):
        total = 0.0
        for n in set(self.A) | set(self.B):
            w = n * math.pi * self.c / self.L
            total += (self.A.get(n, 0) * math.cos(w * t) + self.B.get(n, 0) / w * math.sin(w * t)) * math.sin(
                n * math.pi * x / self.L
            )
        return total

    def leapfrog(self, x0, t0, cells=600):
        x = np.linspace(0.0, self.L, cells + 1)
        dx = x[1] - x[0]
        dt = 0.5 * dx / self.c
        steps = int(round(t0 / dt))
        dt = t0 / steps
        r2 = (self.c * dt / dx) ** 2
        u0 = sum(a * np.sin(n * np.pi * x / self.L) for n, a in self.A.items())
        v0 = sum(b * np.sin(n * np.pi * x / self.L) for n, b in self.B.items()) if self.B else np.zeros_like(x)
        lap = np.zeros_like(x)
        lap[1:-1] = u0[2:] - 2 * u0[1:-1] + u0[:-2]
        u1 = u0 + dt * v0 + 0.5 * r2 * lap
        u1[0] = u1[-1] = 0.0
        prev, cur = u0, u1
        for _ in range(steps - 1):
            nxt = np.zeros_like(x)
            nxt[1:-1] = 2 * cur[1:-1] - prev[1:-1] + r2 * (cur[2:] - 2 * cur[1:-1] + cur[:-2])
            prev, cur = cur, nxt
        return float(np.interp(x0, x, cur))


def test_skill_wiring():
    assert SKILL.answer == "number"
    assert [(s.topic, s.difficulty) for s in SKILL.sources] == [
        (TOPIC, "easy"),
        (TOPIC, "medium"),
        (TOPIC, "hard"),
        (EXAM, "exam"),
    ]
    assert SKILL.sources[-1].stages == ["exam"]
    assert "ucv1-wave-equation-separation" in SKILL.prerequisites
    assert SKILL.year_level == ("university",)


def test_easy_is_dalembert_from_rest():
    for seed in range(60):
        q = tier_question(TOPIC, "easy", seed)
        f, g, c, x0, t0 = _line_string(q.text)
        assert g == 0
        assert abs(_dalembert(f, g, c, x0, t0) - float(q.answer)) < 6e-4, (seed, q.text)


def test_medium_covers_dalembert_with_velocity_and_the_bounded_string():
    kinds = Counter()
    for seed in range(100):
        q = tier_question(TOPIC, "medium", seed)
        if "long string" in q.text:
            f, g, c, x0, t0 = _line_string(q.text)
            assert g != 0
            assert abs(_dalembert(f, g, c, x0, t0) - float(q.answer)) < 6e-4, (seed, q.text)
            kinds["line"] += 1
        else:
            s = Bounded(q.text)
            assert not s.B
            x0, t0 = (float(v) for v in _FIND.search(q.text).groups())
            assert abs(s.u(x0, t0) - float(q.answer)) < 6e-4, (seed, q.text)
            kinds["bounded"] += 1
    assert kinds["line"] >= 20 and kinds["bounded"] >= 20


def test_hard_bounded_string_with_velocity_and_tension():
    for seed in range(60):
        q = tier_question(TOPIC, "hard", seed)
        s = Bounded(q.text)
        assert s.B and "T = " in q.text
        x0, t0 = (float(v) for v in _FIND.search(q.text).groups())
        assert abs(s.u(x0, t0) - float(q.answer)) < 6e-4, (seed, q.text)


@pytest.mark.parametrize("tier", ("medium", "hard"))
def test_leapfrog_agrees_on_a_sample(tier):
    checked = 0
    for seed in range(0, 40, 4):
        q = tier_question(TOPIC, tier, seed)
        if "long string" in q.text:
            continue
        s = Bounded(q.text)
        x0, t0 = (float(v) for v in _FIND.search(q.text).groups())
        assert abs(s.leapfrog(x0, t0) - float(q.answer)) < 0.05 * (1 + abs(float(q.answer))), (seed, q.text)
        checked += 1
    assert checked >= 3


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_and_every_distractor_rejected(tier):
    assert_key_accepted_and_distractors_rejected(SKILL, TOPIC, tier, minimum=3)


@pytest.mark.parametrize("tier", TIERS)
def test_text_quality_and_determinism(tier):
    assert_text_quality_and_determinism(TOPIC, tier)


@pytest.mark.parametrize("tier", TIERS)
def test_answer_space(tier):
    spread(TOPIC, tier, minimum_answers=30 if tier != "easy" else 20, minimum_texts=60)


def test_exam_shape_and_keys():
    assert_exam_grading(SKILL, EXAM, [2, 2, 2, 3], ["number"] * 4)
    q = make(EXAM, seed=2, difficulty="exam")
    assert [p.follow_through is None for p in q.parts] == [True, False, False, True]
    for seed in range(40):
        q = make(EXAM, seed=seed, difficulty="exam")
        s = Bounded(q.stem)
        assert abs(float(q.parts[0].answer) - s.c) < 1e-9
        assert abs(float(q.parts[1].answer) - s.c / (2 * s.L)) < 6e-4
        assert abs(float(q.parts[2].answer) - 2 * math.pi * s.c / s.L) < 6e-4
        x0, t0 = (float(v) for v in _FIND.search(q.parts[3].prompt).groups())
        assert abs(s.u(x0, t0) - float(q.parts[3].answer)) < 6e-4, seed


def _wrong_speed(q):
    return "60.0" if float(q.parts[0].answer) != 60 else "70.0"


def _length(q):
    return float(re.search(r"string of length ([\d.]+) m", q.stem).group(1))


def test_exam_follow_through_from_a_wrong_speed():
    assert_follow_through(
        SKILL,
        EXAM,
        "b",
        {"a": _wrong_speed},
        lambda q, replies: f"{float(replies['a']) / (2 * _length(q)):.3f}",
    )
    assert_follow_through(
        SKILL,
        EXAM,
        "c",
        {"a": _wrong_speed},
        lambda q, replies: f"{2 * math.pi * float(replies['a']) / _length(q):.3f}",
    )
