"""`ucv1-alternating-series-error`: the alternating series estimation theorem.

The oracle reads b_n out of the question's own text, then re-derives the bound, the partial sum, the interval and the
least N from exact rational arithmetic; it never imports the generator module.
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest
import sympy

from mathgen import make
from tests.fy1so_support import SEEDS, check_chain, check_follow_through, check_single, check_tiers_differ, parse_text

TOPIC = "fy1so_alternating_series"
EXAM = "fy1so_exd_alternating_series"
n = sympy.Symbol("n")


def _b_from(body: str):
    body = re.sub(r"\(?(\w+|\d+)!\)?", lambda m: f"factorial({m.group(1)})", body.strip())
    return sympy.lambdify(n, parse_text(body, {"n": n, "factorial": sympy.factorial}), "sympy")


def _term_of(text: str):
    found = re.search(r"\(-1\)\^\(n\+1\) (.+?)(?: satisfies|\. | is | has |$)", text) or re.search(
        r"b_n = (.+?)\. ", text
    )
    return _b_from(found.group(1))


def _fraction(token: str) -> Fraction:
    return Fraction(token)


def _tolerance(text: str) -> Fraction:
    token = re.search(r"(?:within|less than) (\d+(?:/\d+|\.\d+)?)", text).group(1)
    return Fraction(token)


def _as_fraction(value) -> Fraction:
    value = sympy.nsimplify(value, rational=True)
    return Fraction(int(value.p), int(value.q))


def _b(f, k: int) -> Fraction:
    return _as_fraction(f(sympy.Integer(k)))


def _partial(f, upto: int) -> Fraction:
    return sum((((-1) ** (k + 1)) * _b(f, k) for k in range(1, upto + 1)), Fraction(0))


def _least_n(f, tol: Fraction) -> int:
    k = 1
    while _b(f, k + 1) >= tol:
        k += 1
    return k


def _terms_count(text: str) -> int:
    return int(re.search(r"S_(\d+)|first (\d+) terms", text).group(1) or re.search(r"first (\d+) terms", text).group(1))


def test_easy_bound_is_the_first_dropped_term():
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="easy")
        f = _term_of(q.text)
        count = _terms_count(q.text)
        assert Fraction(q.answer_text) == _b(f, count + 1), (seed, q.text, q.answer_text)
        # the bound really bounds the error: the next partial sums straddle the limit within it
        assert abs(_partial(f, count + 1) - _partial(f, count)) == _b(f, count + 1)


@pytest.mark.parametrize("difficulty", ["medium", "hard"])
def test_least_terms_oracle(difficulty):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        f = _term_of(q.text)
        tol = _tolerance(q.text)
        least = _least_n(f, tol)
        assert _b(f, least + 1) < tol <= _b(f, least)
        if difficulty == "medium":
            assert int(q.answer_text) == least, (seed, q.text, q.answer_text, least)
        else:
            assert Fraction(q.answer_text) == _partial(f, least), (seed, q.text, q.answer_text)


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_tier_answers_and_distractors(difficulty):
    check_single(TOPIC, difficulty, "fraction")


def test_b_n_is_positive_and_decreasing():
    for seed in SEEDS:
        for tier in ("easy", "medium", "hard"):
            f = _term_of(make(TOPIC, seed=seed, difficulty=tier).text)
            values = [_b(f, k) for k in range(1, 40)]
            assert all(v > 0 for v in values)
            assert all(a > b for a, b in zip(values, values[1:], strict=False))
            assert values[-1] < values[0] / 5  # tends to 0 (a coarse numeric check)


def test_tiers_differ_in_shape():
    pools = check_tiers_differ(TOPIC)
    assert all("first" in t or "S_" in t for t in pools["easy"])
    assert all("terms" in t for t in pools["medium"])
    assert all("partial sum" in t for t in pools["hard"])


def test_exam_chain_shape_and_marks():
    check_chain(EXAM, ["fraction"] * 4, [1, 2, 1, 2])
    check_follow_through(EXAM)


def test_exam_oracle():
    for seed in SEEDS:
        q = make(EXAM, seed=seed, difficulty="exam")
        f = _b_from(re.search(r"b_n = (.+?)\. Let", q.stem).group(1))
        a, b, c, d = q.parts
        count = int(re.search(r"S_(\d+)\|", a.prompt).group(1))
        tol = Fraction(re.search(r"< (\d+(?:/\d+|\.\d+)?)", d.prompt).group(1))
        bound = _b(f, count + 1)
        s = _partial(f, count)
        assert Fraction(a.answer_text) == bound
        assert Fraction(b.answer_text) == s
        assert Fraction(c.answer_text) == s - bound
        assert int(d.answer_text) == _least_n(f, tol)
        assert c.follow_through == "part_b - part_a"
        # the interval brackets the limit, which a long partial sum approximates
        limit = sum(((-1) ** (k + 1)) * float(_b(f, k)) for k in range(1, 400))
        if count < 300:
            assert float(s - bound) - 1e-9 <= limit <= float(s + bound) + 1e-9


def test_skill_loads_and_has_an_exam_tier():
    from mathgen.skills import load_skills

    _, skills = load_skills()
    skill = next(s for s in skills if s.id == "ucv1-alternating-series-error")
    assert skill.status == "active" and skill.answer == "fraction"
    assert [s.difficulty for s in skill.sources] == ["easy", "medium", "hard", "exam"]
    assert skill.sources[-1].stages == ["exam"]


def test_medium_tolerance_is_never_a_boundary_case_and_reads_strictly():
    """ "Less than" the tolerance, and N - 1 terms never already reach it (b_N is strictly above the tolerance)."""
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="medium")
        f = _term_of(q.text)
        tol = _tolerance(q.text)
        least = _least_n(f, tol)
        assert _b(f, least) > tol, (seed, q.text)
        assert "within" not in q.text, q.text
