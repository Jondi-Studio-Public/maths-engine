"""Tests for the multi-part exam source of senior-examine-the-position-of-two-particles-each-described: closest
approach of two linear trajectories and the time within range. Keys are re-derived from the rendered stem with
sympy (the squared separation is minimised and the range equation solved)."""

from __future__ import annotations

import re
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction

import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

import mathgen.generators.exd_x_r2i_particles_approach  # noqa: F401  (registers the topic)
from tests.exd_x_r2i_support import exam, follow_through_check, standard_checks

TOPIC = "exd_x_r2i_particles_approach"
SKILL = "senior-examine-the-position-of-two-particles-each-described"
LABELS = ["a", "b", "c", "d"]
MARKS = [2, 2, 2, 1]
T = sympy.Symbol("t")
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)


def _coords(text: str, name: str):
    inner = re.search(rf"{name}\(t\) = \(([^()]*)\)", text).group(1)
    return [parse_expr(c.strip(), local_dict={"t": T}, transformations=_TRANSFORMS) for c in inner.split(", ")]


def _text(v) -> str:
    v = sympy.nsimplify(v)
    return str(int(v)) if v == int(v) else f"{float(v):.1f}"


def _two(v) -> str:
    exact = Fraction(str(sympy.N(sympy.nsimplify(v), 30)))
    return str(
        (Decimal(exact.numerator) / Decimal(exact.denominator)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    )


def _solve(q):
    ra, rb = _coords(q.stem, "r_A"), _coords(q.stem, "r_B")
    radius = int(re.search(r"(?:no more than|at most) (\d+) m", q.stem).group(1))
    d = [b - a for a, b in zip(ra, rb)]
    sq = sympy.expand(d[0] ** 2 + d[1] ** 2)
    t_star = sympy.solve(sympy.diff(sq, T), T)[0]
    least = sympy.sqrt(sq.subs(T, t_star))
    times = sorted(sympy.solve(sympy.Eq(sq, radius**2), T))
    return t_star, least, times[1] - times[0], times[0]


def oracle(q):
    t_star, least, duration, first = _solve(q)
    return {"a": _text(t_star), "b": _two(least), "c": _two(duration), "d": _two(first)}


CHECKS = standard_checks(TOPIC, SKILL, oracle, LABELS, MARKS, {"b", "c", "d"})


def test_shape_and_text():
    CHECKS["shape"]()


def test_prompt_shapes_beyond_hard_tier():
    CHECKS["shapes"]()


def test_oracle_agrees_with_every_part():
    CHECKS["oracle_agrees"]()


def test_composed_reply_earns_full_marks():
    CHECKS["composed_full_marks"]()


def test_distractors_are_tagged_and_rejected():
    CHECKS["distractors_rejected"]()


def test_particles_never_meet_and_window_opens_after_the_start():
    for s in range(120):
        q = exam(TOPIC, s)
        t_star, least, duration, first = _solve(q)
        assert least > 0 and first > 0 and duration > 0


def test_wrong_time_carries_through_distance_range_and_start():
    def chain(q):
        ra, rb = _coords(q.stem, "r_A"), _coords(q.stem, "r_B")
        radius = int(re.search(r"(?:no more than|at most) (\d+) m", q.stem).group(1))
        d = [b - a for a, b in zip(ra, rb)]
        t_star, least, duration, first = _solve(q)
        wrong_a = t_star + 1
        dist = sympy.sqrt(sum(c.subs(T, wrong_a) ** 2 for c in d))
        if dist >= radius:
            return None
        shown_dist = Fraction(_two(dist))  # the learner carries the distance as rounded in (b)
        wrong_c = (
            2
            * sympy.sqrt(radius**2 - sympy.Rational(shown_dist) ** 2)
            / sympy.sqrt(sum(sympy.diff(c, T) ** 2 for c in d))
        )
        replies = {
            "a": _text(wrong_a),
            "b": _two(dist),
            "c": _two(wrong_c),
            "d": _two(wrong_a - Fraction(_two(wrong_c)) / 2),
        }
        return "a", replies, sum(MARKS) - MARKS[0]

    follow_through_check(TOPIC, SKILL, chain)


def test_wrong_distance_carries_into_range_and_start():
    def chain(q):
        radius = int(re.search(r"(?:no more than|at most) (\d+) m", q.stem).group(1))
        ra, rb = _coords(q.stem, "r_A"), _coords(q.stem, "r_B")
        w = [sympy.diff(b - a, T) for a, b in zip(ra, rb)]
        wlen = sympy.sqrt(sum(c**2 for c in w))
        t_star, least, duration, first = _solve(q)
        wrong_b = least + 2
        if wrong_b >= radius:
            return None
        wrong_c = 2 * sympy.sqrt(radius**2 - wrong_b**2) / wlen
        replies = {
            "a": _text(t_star),
            "b": _two(wrong_b),
            "c": _two(wrong_c),
            "d": _two(t_star - Fraction(_two(wrong_c)) / 2),
        }
        return "b", replies, sum(MARKS) - MARKS[1]

    follow_through_check(TOPIC, SKILL, chain)
