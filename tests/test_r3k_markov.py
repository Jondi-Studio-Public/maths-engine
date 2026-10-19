"""Tests `uc2-markov-n-step-three-state` (topic `r3k_markov`) and its exam chain `exd_x_r3k_markov`.

The oracle reads the transition matrix P (and the question's states, steps and initial distribution) back out of the
question TEXT and re-derives each key with SymPy matrix algebra: matrix powers, the fundamental matrix, the null space of
P^T - I.
"""

from __future__ import annotations

import re
from collections import Counter
from fractions import Fraction

import pytest
import sympy

import mathgen.generators.r3k_markov_build  # noqa: F401  (registers the topics)
from mathgen import make
from mathgen.engine import grade
from tests.r3k_support import check_no_follow_through_collision, second_route_on  # noqa: F401  (autouse fixture)
from tests.r3k_support import (
    as_fraction,
    check_exam_grading,
    check_follow_through,
    check_single_tier,
    check_wiring,
    skill,
    stem,
)

TOPIC = "r3k_markov"
EXAM = "exd_x_r3k_markov"
SK = skill("uc2-markov-n-step-three-state")
TIERS = ("easy", "medium", "hard")


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


def _exam(seed):
    return make(EXAM, seed=seed, difficulty="exam")


def _matrix(text):
    body = re.search(r"P = (\[\[.*?\]\])", text).group(1)
    rows = re.findall(r"\[([^\[\]]+)\]", body)
    return sympy.Matrix([[sympy.Rational(v.strip()) for v in row.split(",")] for row in rows])


def _frac(value):
    value = sympy.nsimplify(value)
    return Fraction(int(value.p), int(value.q))


def _key(q):
    return as_fraction(q.answer)


def test_skill_wiring():
    check_wiring(SK, TOPIC, EXAM, ["ucv1-markov-hitting-probability", "ucv1-poisson-count-probability"], "fraction")


def test_easy_key_is_an_entry_of_p_squared():
    for seed in range(150):
        q = _q("easy", seed)
        p = _matrix(q.text)
        assert all(sum(p.row(i)) == 1 for i in range(3)), seed
        m = re.search(r"in state (\d) now, find P\(X_2 = (\d)\)", q.text)
        if m:
            i, j = int(m.group(1)), int(m.group(2))
        else:
            j, i = (
                int(v)
                for v in re.search(
                    r"in state (\d) two steps from now, given that it is in state (\d) now", q.text
                ).groups()
            )
        assert _key(q) == _frac((p**2)[i - 1, j - 1]), (seed, q.text)


def test_medium_keys():
    seen = Counter()
    for seed in range(160):
        q = _q("medium", seed)
        p = _matrix(q.text)
        if "eventually absorbed" in q.text:
            assert list(p.row(0)) == [1, 0, 0] and list(p.row(2)) == [0, 0, 1]
            h = sympy.Symbol("h")
            (sol,) = sympy.solve(sympy.Eq(h, p[1, 0] + p[1, 1] * h), h) if p[1, 1] != 1 else (None,)
            assert _key(q) == _frac(sol), (seed, q.text)
            seen["absorption"] += 1
        else:
            init = [sympy.Rational(int(a), int(b or 1)) for a, b in re.findall(r"probability (\d+)(?:/(\d+))?", q.text)]
            dist = sympy.Matrix([init])
            assert sum(init) == 1
            n, j = (int(v) for v in re.search(r"P\(X_(\d) = (\d)\)", q.text).groups())
            assert _key(q) == _frac((dist * p**n)[j - 1]), (seed, q.text)
            seen[f"forward-{n}"] += 1
    assert len(seen) == 3 and min(seen.values()) >= 10, seen


def test_hard_keys():
    seen = Counter()
    for seed in range(160):
        q = _q("hard", seed)
        p = _matrix(q.text)
        if "expected number of steps" in q.text:
            start = int(re.search(r"starts in state (\d)", q.text).group(1))
            assert list(p.row(2)) == [0, 0, 1]
            qm = p[:2, :2]
            times = (sympy.eye(2) - qm).inv() * sympy.Matrix([1, 1])
            assert _key(q) == _frac(times[start - 1]), (seed, q.text)
            seen["time"] += 1
        else:
            j = int(re.search(r"time that the chain spends in state (\d)", q.text).group(1))
            (vec,) = (p.T - sympy.eye(3)).nullspace()
            pi = vec / sum(vec)
            assert all(v > 0 for v in pi)
            assert len(set(pi)) > 1, seed
            assert _key(q) == _frac(pi[j - 1]), (seed, q.text)
            seen["stationary"] += 1
    assert len(seen) == 2 and min(seen.values()) >= 30, seen


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_distractors_rejected_text_and_determinism(tier):
    check_single_tier(SK, lambda seed: _q(tier, seed), seeds=60)


def test_answers_vary():
    for tier in TIERS:
        answers = [_q(tier, seed).answer_text for seed in range(120)]
        assert len(set(answers)) >= 15, (tier, Counter(answers).most_common(3))


# -- exam chain -----------------------------------------------------------------------------------------------


def test_exam_shape_and_marks():
    for seed in range(40):
        q = _exam(seed)
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.marks for p in q.parts] == [3, 2, 3, 2]
        assert "part_c" in q.parts[3].follow_through


def test_exam_keys_match_sympy():
    for seed in range(100):
        q = _exam(seed)
        p = _matrix(stem(q))
        assert list(p.row(2)) == [0, 0, 1]
        assert as_fraction(q.parts[1].answer) == _frac((p**3)[0, 2]), (seed, stem(q))
        qm = p[:2, :2]
        times = (sympy.eye(2) - qm).inv() * sympy.Matrix([1, 1])
        assert as_fraction(q.parts[2].answer) == _frac(times[1])
        assert as_fraction(q.parts[3].answer) == _frac(times[0])


def test_exam_matrix_part_text_matches_p_squared():
    for seed in range(60):
        q = _exam(seed)
        p = _matrix(stem(q))
        text = q.parts[0].answer_text
        shown = sympy.Matrix(
            [[sympy.Rational(v.strip()) for v in row.split(",")] for row in re.findall(r"\[([^\[\]]+)\]", text)]
        )
        assert shown == p**2, (seed, text)


def test_exam_grading():
    check_exam_grading(SK, _exam, seeds=30)


def test_exam_follow_through_never_lands_on_a_distractor():
    check_no_follow_through_collision(_exam)


def test_exam_follow_through_carries_a_wrong_time():
    carried = 0
    for seed in range(60):
        q = _exam(seed)
        c, d = q.parts[2], q.parts[3]
        canon = {p.label: p.answer_text for p in q.parts}
        expr = sympy.sympify(d.follow_through)
        for wrong in c.distractors:
            value = expr.subs(sympy.Symbol("part_c"), sympy.nsimplify(wrong))
            reply = {**canon, "c": wrong, "d": str(value)}
            if value == sympy.nsimplify(wrong):
                continue  # writing the same number again is not following through (the grader withholds it by design)
            result = grade(SK, q, reply)
            assert result.parts["c"].marks == 0
            assert result.parts["d"].marks == d.marks, (seed, reply)
            carried += 1
    assert carried > 10
