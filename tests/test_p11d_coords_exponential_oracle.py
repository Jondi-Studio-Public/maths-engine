"""Independent oracle for the `p11d_coords_exponential` questions: it re-derives each answer from the rendered
question text.

Parses ONLY the rendered artefact (``question.text``, ``question.answer_text``,
``question.distractors``) and re-derives the answer from scratch. Imports
nothing from ``mathgen.generators.p11d_coords_exponential_build`` -- not a
helper, not a constant, not the internal p/m/n construction.

Skill 1: parses the stated coordinate triples and recomputes the displacement
by subtraction directly from the parsed points (the only mathematical route
to a coordinate delta, so this is the same operation the generator performs,
written independently against the rendered text rather than the generator's
own locals) -- and independently reconstructs the three misconception values
(swapped, wrong-direction, and, hard tier, ignores-first-leg) to confirm the
rendered distractor set matches.

Skill 2: parses the stated equation's numbers and resolves x via a from-scratch
prime-power factorisation and grid search over Fraction values -- never via
sympy.solve (which can hand back log forms or complex branches for a
fractional-exponent equation) and never via the generator's own p/m/n
internals.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction

from mathgen import make

COORD_TOPIC = "p11d_three_dimensional_coordinates"
EXP_TOPIC = "p11d_exponential_common_base_solve"
TIERS = ["easy", "medium", "hard"]
SEEDS = range(60)

# --- skill 1 oracle ---------------------------------------------------------

_SIMPLE_MOVE_RE = re.compile(r"from \((-?\d+), (-?\d+), (-?\d+)\) to \((-?\d+), (-?\d+), (-?\d+)\)")
_HARD_MOVE_RE = re.compile(
    r"A\((-?\d+), (-?\d+), (-?\d+)\), moves to B\((-?\d+), (-?\d+), (-?\d+)\), "
    r"then moves again to C\((-?\d+), (-?\d+), (-?\d+)\)"
)
_ANSWER_RE = re.compile(r"^dx = (-?\d+), dy = (-?\d+), dz = (-?\d+)$")


def _parse_answer(text: str) -> tuple[int, int, int]:
    m = _ANSWER_RE.match(text)
    assert m, f"unparseable pair answer: {text!r}"
    return tuple(int(v) for v in m.groups())


def test_coord_easy_medium_displacement_recomputed_from_parsed_points():
    checked = 0
    for tier in ("easy", "medium"):
        for seed in SEEDS:
            q = make(COORD_TOPIC, seed=seed, difficulty=tier)
            m = _SIMPLE_MOVE_RE.search(q.text)
            assert m, f"{tier} seed {seed}: unparseable question {q.text!r}"
            x1, y1, z1, x2, y2, z2 = (int(v) for v in m.groups())
            expected = (x2 - x1, y2 - y1, z2 - z1)
            assert _parse_answer(q.answer_text) == expected, (
                f"{tier} seed {seed}: parsed answer {_parse_answer(q.answer_text)} "
                f"!= independently recomputed delta {expected}"
            )
            checked += 1
    assert checked == 2 * len(SEEDS)


def test_coord_hard_total_displacement_and_distractor_set_recomputed():
    checked = 0
    for seed in SEEDS:
        q = make(COORD_TOPIC, seed=seed, difficulty="hard")
        m = _HARD_MOVE_RE.search(q.text)
        assert m, f"seed {seed}: unparseable question {q.text!r}"
        ax, ay, az, bx, by, bz, cx, cy, cz = (int(v) for v in m.groups())
        expected = (cx - ax, cy - ay, cz - az)
        assert _parse_answer(q.answer_text) == expected, (
            f"seed {seed}: parsed answer {_parse_answer(q.answer_text)} "
            f"!= independently recomputed A-to-C delta {expected}"
        )

        # Independently reconstruct the three misconception values and
        # confirm the rendered distractor set matches exactly.
        dx, dy, dz = expected
        swapped = (dz, dx, dy)
        wrong_direction = (-dx, -dy, -dz)
        ignores_first_leg = (cx - bx, cy - by, cz - bz)
        expected_distractor_triples = {swapped, wrong_direction, ignores_first_leg}
        actual_distractor_triples = {_parse_answer(d) for d in q.distractors}
        assert actual_distractor_triples == expected_distractor_triples, (
            f"seed {seed}: distractor triples {actual_distractor_triples} != "
            f"independently reconstructed {expected_distractor_triples}"
        )
        checked += 1
    assert checked == len(SEEDS)


# --- skill 2 oracle ---------------------------------------------------------

_EASY_RE = re.compile(r"(\d+)\^x = (\d+)")
_MEDIUM_RE = re.compile(r"(\d+)\^x = (-?[\d/]+)")
_HARD_RE = re.compile(r"(\d+) × (\d+)\^x = (\d+)")


def _sole_prime_factor(n: int) -> tuple[int, int]:
    """(p, m) such that n == p**m, for a positive integer n with exactly one
    distinct prime factor -- a from-scratch trial-division factorisation,
    never assuming the generator's own {2, 3, 5} base pool."""
    assert n >= 2, f"{n} has no prime factor"
    remaining = n
    factors: dict[int, int] = {}
    d = 2
    while d * d <= remaining:
        while remaining % d == 0:
            factors[d] = factors.get(d, 0) + 1
            remaining //= d
        d += 1
    if remaining > 1:
        factors[remaining] = factors.get(remaining, 0) + 1
    assert len(factors) == 1, f"{n} is not a prime power: factors {factors}"
    ((p, m),) = factors.items()
    return p, m


def _integer_exponent_of(value: Fraction, base: int) -> int:
    """The integer e with ``Fraction(base) ** e == value``, found by a local
    grid search around a float-log estimate -- a from-scratch numeric search,
    never sympy.solve and never the generator's own n/m construction."""
    guess = round(math.log(float(value), base))
    for e in range(guess - 4, guess + 5):
        if Fraction(base) ** e == value:
            return e
    raise AssertionError(f"{value} is not an exact power of {base} near exponent {guess}")


def test_exp_easy_solved_by_from_scratch_prime_power_search():
    checked = 0
    for seed in SEEDS:
        q = make(EXP_TOPIC, seed=seed, difficulty="easy")
        m = _EASY_RE.search(q.text)
        assert m, f"seed {seed}: unparseable question {q.text!r}"
        a_base, b_value = int(m.group(1)), int(m.group(2))
        p, m_exp = _sole_prime_factor(a_base)
        n_exp = _integer_exponent_of(Fraction(b_value), p)
        expected_x = Fraction(n_exp, m_exp)
        assert Fraction(q.answer_text) == expected_x, (
            f"seed {seed}: parsed answer {q.answer_text} != independently solved x {expected_x}"
        )
        checked += 1
    assert checked == len(SEEDS)


def test_exp_medium_solved_by_from_scratch_prime_power_search():
    checked = 0
    for seed in SEEDS:
        q = make(EXP_TOPIC, seed=seed, difficulty="medium")
        m = _MEDIUM_RE.search(q.text)
        assert m, f"seed {seed}: unparseable question {q.text!r}"
        a_base, b_text = int(m.group(1)), m.group(2)
        b_value = Fraction(b_text)
        p, m_exp = _sole_prime_factor(a_base)
        n_exp = _integer_exponent_of(b_value, p)
        expected_x = Fraction(n_exp, m_exp)
        assert Fraction(q.answer_text) == expected_x, (
            f"seed {seed}: parsed answer {q.answer_text} != independently solved x {expected_x}"
        )
        checked += 1
    assert checked == len(SEEDS)


def test_exp_hard_solved_by_isolating_coefficient_then_from_scratch_search():
    checked = 0
    for seed in SEEDS:
        q = make(EXP_TOPIC, seed=seed, difficulty="hard")
        m = _HARD_RE.search(q.text)
        assert m, f"seed {seed}: unparseable question {q.text!r}"
        a_coef, p_base, c_value = int(m.group(1)), int(m.group(2)), int(m.group(3))
        # Isolate the power term by dividing through by the coefficient --
        # the correct first algebraic step, independently performed here.
        isolated = Fraction(c_value, a_coef)
        expected_x = Fraction(_integer_exponent_of(isolated, p_base), 1)
        assert Fraction(q.answer_text) == expected_x, (
            f"seed {seed}: parsed answer {q.answer_text} != independently solved x {expected_x}"
        )
        checked += 1
    assert checked == len(SEEDS)
