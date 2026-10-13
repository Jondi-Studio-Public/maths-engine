"""Exam-tier checks for `classify-constant-coefficient-pde`: parabolic value of k, largest elliptic whole k, discriminant."""

import math
import re
from fractions import Fraction

import sympy

from mathgen import make
from tests.exd_x_sc2_support import (
    check_all_wrong_earns_nothing,
    check_answer_space,
    check_deterministic,
    check_follow_through,
    check_key_accepted_distractors_rejected,
    check_no_text_defects,
    check_oracle,
    check_variety,
    check_wired,
)

TOPIC = "exd_x_sc2_pde_parameter"
SKILL = "classify-constant-coefficient-pde"
K = sympy.Symbol("k", positive=True)


def _coefficients(q):
    stem = q.stem
    a = re.search(r"equation (\d*)u_xx", stem).group(1)
    sign, m = re.search(r"([+-]) (\d*)k u_xy", stem).groups()
    c = re.search(r"(\d*)u_yy", stem).group(1)
    return int(a or 1), int(f"{sign}{m or 1}"), int(c or 1)


def _decimal(value: sympy.Rational) -> str:
    text = format(Fraction(int(value.p), int(value.q)).limit_denominator(10**6), "")
    frac = Fraction(int(value.p), int(value.q))
    for places in range(0, 5):
        if (frac * 10**places).denominator == 1:
            return f"{float(frac):.{places}f}"
    raise AssertionError(text)


def oracle(q):
    a, b, c = _coefficients(q)
    disc = b * b * K**2 - 4 * a * c
    k0 = sympy.solve(sympy.Eq(disc, 0), K)
    assert len(k0) == 1
    k0 = sympy.nsimplify(k0[0])
    biggest = int(sympy.ceiling(k0)) - 1
    return {
        "a": _decimal(sympy.Rational(k0)),
        "b": str(biggest),
        "c": str(int(disc.subs(K, biggest))),
    }


def wrong_chain(q):
    a, b, c = _coefficients(q)
    k0 = Fraction(2 * math.isqrt(a * c), abs(b))
    wrong_a = k0 + Fraction(23, 10)
    wrong_b = math.ceil(wrong_a) - 1
    return {
        "a": format(float(wrong_a), ".1f"),
        "b": str(wrong_b),
        "c": str(b * b * wrong_b * wrong_b - 4 * a * c),
    }


def test_no_text_defects():
    check_no_text_defects(TOPIC)


def test_variety_and_new_shapes():
    check_variety(
        TOPIC, lambda s: make("constant_coefficient_pde_classification", seed=s, difficulty="hard", kind="rearranged")
    )


def test_wired():
    check_wired(SKILL, TOPIC)


def test_oracle_agrees():
    check_oracle(TOPIC, oracle)


def test_answer_space():
    check_answer_space(TOPIC)


def test_deterministic():
    check_deterministic(TOPIC)


def test_key_accepted_distractors_rejected():
    check_key_accepted_distractors_rejected(SKILL, TOPIC)


def test_follow_through_from_wrong_parabolic_value():
    check_follow_through(SKILL, TOPIC, wrong_chain)


def test_part_c_is_negative_and_confirms_elliptic():
    for seed in range(100):
        q = make(TOPIC, seed=seed, difficulty="exam")
        assert int(q.parts[2].answer_text) < 0


def test_all_wrong_earns_nothing():
    check_all_wrong_earns_nothing(SKILL, TOPIC, {"a": "99.5", "b": "905", "c": "907"})
