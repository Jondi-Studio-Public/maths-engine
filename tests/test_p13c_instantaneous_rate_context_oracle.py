"""Independent oracle for senior-find-instantaneous-rates-of-change
(ACMMM092, topic ``p13c_instantaneous_rate_context``).

Parses ONLY ``question.text``/``question.answer_text`` -- never imports
anything from ``mathgen.generators.p13c_instantaneous_rate_context_build``
(not a helper, not a constant, not a table) -- and re-derives the answer from
scratch with ``sympy.diff``/``sympy.solve``, a route independent of however
the generator's own construction built the coefficient arithmetic.
"""

import re

import pytest
from sympy import Symbol, diff, solve, sympify

from mathgen import make

SEEDS = range(60)
TIERS = ("easy", "medium", "hard", "exam")
TOPIC = "p13c_instantaneous_rate_context"

t = Symbol("t")

_POLY_RE = re.compile(r"\(t\) = (?P<poly>[^,]+), where t is the time")
_EVAL_AT_RE = re.compile(r"at t = (-?\d+) [a-zA-Z ]+, in ")
_HARD_TARGET_RE = re.compile(r"equals (-?\d+) [a-zA-Z]+/[a-zA-Z]+\.")
_EXAM_RE = re.compile(
    r"from t = (-?\d+) to t = (-?\d+) [a-zA-Z]+, find the INSTANTANEOUS rate of change "
    r"of [a-zA-Z]+ at t = (-?\d+) [a-zA-Z]+, in"
)


def _parse_poly(text: str):
    m = _POLY_RE.search(text)
    assert m is not None, f"could not find a Q(t) = ... formula in {text!r}"
    poly_str = m.group("poly")
    # "2t^2 - 37t + 377" -> "2*t**2 - 37*t + 377" (insert the multiplication
    # sign a learner reads implicitly, translate "^" to sympy's "**").
    poly_str = re.sub(r"(\d)(t)", r"\1*\2", poly_str)
    poly_str = poly_str.replace("^", "**")
    return sympify(poly_str, locals={"t": t})


@pytest.mark.parametrize("tier", ("easy", "medium"))
def test_oracle_matches_evaluate_derivative(tier):
    checked = 0
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        poly = _parse_poly(q.text)
        m = _EVAL_AT_RE.search(q.text)
        assert m is not None, f"{tier} seed={seed}: could not find 'at t = N' in {q.text!r}"
        t0 = int(m.group(1))
        expected = diff(poly, t).subs(t, t0)
        assert str(expected) == q.answer_text, (
            f"{tier} seed={seed}: oracle expected {expected}, generator said {q.answer_text}"
        )
        checked += 1
    assert checked == len(SEEDS)


def test_oracle_matches_solved_time_hard():
    checked = 0
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="hard")
        poly = _parse_poly(q.text)
        m = _HARD_TARGET_RE.search(q.text)
        assert m is not None, f"seed={seed}: could not find 'equals K unit/unit.' in {q.text!r}"
        k = int(m.group(1))
        roots = solve(diff(poly, t) - k, t)
        assert len(roots) == 1, f"seed={seed}: expected exactly one root, got {roots}"
        assert str(roots[0]) == q.answer_text, (
            f"seed={seed}: oracle expected t={roots[0]}, generator said {q.answer_text}"
        )
        checked += 1
    assert checked == len(SEEDS)


def test_oracle_matches_instantaneous_rate_at_stated_instant_exam():
    checked = 0
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="exam")
        poly = _parse_poly(q.text)
        m = _EXAM_RE.search(q.text)
        assert m is not None, f"seed={seed}: could not parse the interval/target in {q.text!r}"
        t1, t2, target = (int(m.group(i)) for i in (1, 2, 3))
        assert target in (t1, t2, (t1 + t2) // 2), (
            f"seed={seed}: stated target {target} is not t1, t2 or their midpoint ({t1}, {t2})"
        )
        expected = diff(poly, t).subs(t, target)
        assert str(expected) == q.answer_text, (
            f"seed={seed}: oracle expected {expected} at t={target}, generator said {q.answer_text}"
        )
        # The average rate over [t1, t2] must be a DIFFERENT number from the
        # instantaneous rate at `target` whenever the generator actually
        # offered it as a distractor -- confirms the exam tier's own
        # boundary claim (the average-rate misconception is a genuinely
        # different value here, not a collision hidden by dedup) on the
        # seeds where it was offered at all.
        avg_rate = (poly.subs(t, t2) - poly.subs(t, t1)) / (t2 - t1)
        if str(avg_rate) in q.distractors:
            assert str(avg_rate) != q.answer_text, f"seed={seed}: average-rate distractor equals the canonical answer"
        checked += 1
    assert checked == len(SEEDS)
