"""Independent oracle for the `pv0b_log_equation` questions: it re-derives each answer from the rendered question
text.

Parses ONLY the rendered artefact (`question.text`, `question.answer_text`)
and re-derives the exact value of x from scratch, algebraically, using
`fractions.Fraction` throughout. Imports NOTHING from
`mathgen.generators.pv0b_log_equation_build` -- not a helper, not a constant,
not the internal a/b/c/k construction.

Proven red: each parser/solver below was first run against a scratch copy of
the generator with a single deliberately-flipped sign (`answer_text` built
from `(value + c) / a` instead of `(value - c) / a`) -- every oracle test
below failed on that copy (mismatched x), then passed again once the flip
was reverted, confirming the oracle actually recomputes rather than
replaying the generator's own arithmetic (done by hand while writing this
file; the flipped copy was deleted afterwards).
"""

from __future__ import annotations

import math
import re
from fractions import Fraction

from mathgen import make

TOPIC = "pv0b_log_equation"
SEEDS = range(250)


def _check_log_identity(b: int, exponent: Fraction, target: Fraction) -> None:
    """Independent numeric confirmation that b**exponent == target, via
    floating change-of-base -- never the exact Fraction algebra used to
    derive x itself."""
    assert abs(math.log(float(target), b) - float(exponent)) < 1e-6, (
        f"log_{b}({target}) does not recompute to {exponent} via change-of-base"
    )


# --- easy: log_b(ax + c) = k -------------------------------------------------

_EASY_RE = re.compile(r"log_(\d+)\((-?\d+)x ([+-]) (\d+)\) = (\d+)\b")


def test_easy_solved_from_scratch():
    checked = 0
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="easy")
        m = _EASY_RE.search(q.text)
        assert m, f"seed {seed}: unparseable {q.text!r}"
        b, a, sign, c_abs, k = m.groups()
        b, a, c_abs, k = int(b), int(a), int(c_abs), int(k)
        c = c_abs if sign == "+" else -c_abs

        value = Fraction(b) ** k
        _check_log_identity(b, Fraction(k), value)
        expected_x = (value - c) / Fraction(a)

        assert Fraction(q.answer_text) == expected_x, (
            f"easy seed {seed}: parsed answer {q.answer_text} != independently solved {expected_x}"
        )
        checked += 1
    assert checked >= 250


# --- medium: m log_b(ax + c) [+/- n] = p ------------------------------------

_MEDIUM_RE = re.compile(r"(?:(\d+) )?log_(\d+)\((-?\d+)?x ([+-]) (\d+)\)(?: ([+-]) (\d+))? = (-?\d+)\b")


def test_medium_solved_from_scratch_isolating_the_log_first():
    """When the isolated log value b**k is a proper fraction the generator forces the argument's coefficient a =
    1 (to keep x's own denominator from compounding with a) -- printed as bare "x", not "1x"
    (`join_terms`/`fmt_signed`'s convention). The coefficient group is therefore optional here, defaulting to
    1 when absent."""
    checked = 0
    saw_non_positive_k = False
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="medium")
        m = _MEDIUM_RE.search(q.text)
        assert m, f"seed {seed}: unparseable {q.text!r}"
        m_raw, b, a_raw, sign, c_abs, n_sign, n_abs, p = m.groups()
        m_coef = int(m_raw) if m_raw is not None else 1
        a = int(a_raw) if a_raw is not None else 1
        b, c_abs, p = int(b), int(c_abs), int(p)
        c = c_abs if sign == "+" else -c_abs
        n = 0 if n_sign is None else (int(n_abs) if n_sign == "+" else -int(n_abs))

        # isolate: m*k + n = p  =>  k = (p - n) / m, and it must be a whole
        # number for the equation to have an exact answer at all.
        k_frac = Fraction(p - n, m_coef)
        assert k_frac.denominator == 1, f"seed {seed}: isolated k {k_frac} is not an integer"
        k = int(k_frac)
        if k <= 0:
            saw_non_positive_k = True

        value = Fraction(b) ** k  # may be a proper fraction when k < 0, or 1 when k == 0
        _check_log_identity(b, Fraction(k), value)
        expected_x = (value - c) / Fraction(a)

        assert Fraction(q.answer_text) == expected_x, (
            f"medium seed {seed}: parsed answer {q.answer_text} != independently solved {expected_x}"
        )
        checked += 1
    assert checked >= 250
    assert saw_non_positive_k, "medium tier never isolated to k <= 0 over 250 seeds"


# --- hard: log_b(ax + c) +/- log_b(d) = k -----------------------------------

_HARD_RE = re.compile(r"log_(\d+)\((-?\d+)x ([+-]) (\d+)\) ([+-]) log_(\d+)\((\d+)\) = (\d+)\b")


def test_hard_solved_from_scratch_via_product_or_quotient_law():
    checked = 0
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="hard")
        m = _HARD_RE.search(q.text)
        assert m, f"seed {seed}: unparseable {q.text!r}"
        b, a, sign, c_abs, op, b2, d, k = m.groups()
        assert b == b2, f"seed {seed}: the two logs do not share a base"
        b, a, c_abs, d, k = int(b), int(a), int(c_abs), int(d), int(k)
        c = c_abs if sign == "+" else -c_abs

        base_value = Fraction(b) ** k
        _check_log_identity(b, Fraction(k), base_value)
        # "+": log_b((ax+c)*d) = k  => ax+c = b^k / d
        # "-": log_b((ax+c)/d) = k  => ax+c = b^k * d
        value = base_value / d if op == "+" else base_value * d
        expected_x = (value - c) / Fraction(a)

        assert Fraction(q.answer_text) == expected_x, (
            f"hard seed {seed}: parsed answer {q.answer_text} != independently solved {expected_x}"
        )
        checked += 1
    assert checked >= 250


# --- exam: two unknowns-in-logs, both forms ---------------------------------

_EXAM_DIFF_RE = re.compile(r"log_(\d+)\((-?\d+)x ([+-]) (\d+)\) - log_(\d+)\((-?\d+)x ([+-]) (\d+)\) = (\d+)\b")
_EXAM_EQUATED_RE = re.compile(r"log_(\d+)\((-?\d+)x ([+-]) (\d+)\) = log_(\d+)\((-?\d+)x ([+-]) (\d+)\) \+ (\d+)\b")


def _solve_exam(b, a, c, d, e, k) -> Fraction:
    big_b = Fraction(b) ** k
    denom = Fraction(a) - big_b * d
    return (big_b * e - c) / denom


def test_exam_solved_from_scratch_both_arguments_confirmed_positive():
    checked = 0
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="exam")
        m = _EXAM_DIFF_RE.search(q.text)
        if m:
            b, a, sa, c_abs, b2, d, sd, e_abs, k = m.groups()
        else:
            m = _EXAM_EQUATED_RE.search(q.text)
            assert m, f"seed {seed}: unparseable {q.text!r}"
            b, a, sa, c_abs, b2, d, sd, e_abs, k = m.groups()
        assert b == b2, f"seed {seed}: the two logs do not share a base"
        b, a, c_abs, d, e_abs, k = int(b), int(a), int(c_abs), int(d), int(e_abs), int(k)
        c = c_abs if sa == "+" else -c_abs
        e = e_abs if sd == "+" else -e_abs

        expected_x = _solve_exam(b, a, c, d, e, k)
        assert Fraction(q.answer_text) == expected_x, (
            f"exam seed {seed}: parsed answer {q.answer_text} != independently solved {expected_x}"
        )

        arg1 = a * expected_x + c
        arg2 = d * expected_x + e
        assert arg1 > 0, f"exam seed {seed}: first argument {arg1} is not positive at the solution"
        assert arg2 > 0, f"exam seed {seed}: second argument {arg2} is not positive at the solution"
        _check_log_identity(b, Fraction(k), arg1 / arg2)
        checked += 1
    assert checked >= 250
