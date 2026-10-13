"""Tests for the multi-part exam source of senior-recognise-and-use-the-additivity-and-linearity-of: combine stated
definite integrals (one given with reversed limits), integrate a constant, solve for k and find a mean value. Keys
are re-derived from the rendered stem."""

from __future__ import annotations

import re
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction

import mathgen.generators.exd_x_r2i_integral_chain  # noqa: F401  (registers the topic)
from tests.exd_x_r2i_support import exam, follow_through_check, standard_checks

TOPIC = "exd_x_r2i_integral_chain"
SKILL = "senior-recognise-and-use-the-additivity-and-linearity-of"
LABELS = ["a", "b", "c", "d"]
MARKS = [1, 2, 2, 1]
FT_LABELS = {"b", "c", "d"}


def _two(x: Fraction) -> str:
    return str((Decimal(x.numerator) / Decimal(x.denominator)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _frac(x: Fraction) -> str:
    return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"


def _facts(q):
    pat = r"integral of ([fg])\(x\) from x = (-?\d+) to x = (-?\d+) is (-?\d+)"
    given = [(fn, int(lo), int(hi), Fraction(int(val))) for fn, lo, hi, val in re.findall(pat, q.stem)]
    assert [g[0] for g in given] == ["f", "f", "g"]
    (_, a, b, i1), (_, lo2, hi2, v2), (_, ga, gc, j) = given
    assert ga == a
    c = gc
    # A forward integral is added; one stated with its limits reversed has the opposite value.
    forward2 = v2 if (lo2, hi2) == (b, c) else -v2
    assert (lo2, hi2) in ((b, c), (c, b))
    total_f = i1 + forward2
    combo = re.search(r"Find the integral of (.*) from x = -?\d+ to x = -?\d+\.", q.parts[1].prompt).group(1)
    return a, b, c, total_f, j, combo, q


def _coefficients(combo: str):
    m = re.fullmatch(r"(-?\d*)f\(x\) ([+-]) (\d*)g\(x\) ([+-]) (\d+)", combo)
    p = int(m.group(1)) if m.group(1) not in ("", "-") else (1 if m.group(1) == "" else -1)
    qq = int(m.group(3) or 1) * (1 if m.group(2) == "+" else -1)
    r = int(m.group(5)) * (1 if m.group(4) == "+" else -1)
    return p, qq, r


def oracle(q):
    a, b, c, total_f, j, combo, _ = _facts(q)
    p, qq, r = _coefficients(combo)
    target = int(re.search(r"is (-?\d+)\. Give k", q.parts[2].prompt).group(1))
    return {
        "a": _frac(total_f),
        "b": _frac(p * total_f + qq * j + r * (c - a)),
        "c": _frac((Fraction(target) - total_f) / j),
        "d": _two(total_f / (c - a)),
    }


CHECKS = standard_checks(TOPIC, SKILL, oracle, LABELS, MARKS, FT_LABELS)


def test_source_wired():
    CHECKS["wired"]()


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


def test_answer_space():
    CHECKS["answer_space"]()


def test_deterministic():
    CHECKS["deterministic"]()


def test_half_the_draws_state_the_second_integral_with_reversed_limits():
    reversed_seen = forward_seen = 0
    for s in range(120):
        q = exam(TOPIC, s)
        _, b, c, *_ = _facts(q)
        if f"from x = {c} to x = {b} is" in q.stem:
            reversed_seen += 1
        else:
            forward_seen += 1
    assert reversed_seen >= 20 and forward_seen >= 20


def test_wrong_total_carries_into_every_later_part():
    def chain(q):
        a, b, c, total_f, j, combo, _ = _facts(q)
        p, qq, r = _coefficients(combo)
        target = int(re.search(r"is (-?\d+)\. Give k", q.parts[2].prompt).group(1))
        wrong_a = total_f + 3
        replies = {
            "a": _frac(wrong_a),
            "b": _frac(p * wrong_a + qq * j + r * (c - a)),
            "c": _frac((Fraction(target) - wrong_a) / j),
            "d": _two(wrong_a / (c - a)),
        }
        return "a", replies, sum(MARKS) - MARKS[0]

    follow_through_check(TOPIC, SKILL, chain)


def test_k_is_graded_as_an_exact_fraction_when_carried_from_a_wrong_total():
    def chain(q):
        a, b, c, total_f, j, combo, _ = _facts(q)
        target = int(re.search(r"is (-?\d+)\. Give k", q.parts[2].prompt).group(1))
        wrong_a = total_f + 1
        k = (Fraction(target) - wrong_a) / j
        if k.denominator == 1:
            return None
        p, qq, r = _coefficients(combo)
        replies = {
            "a": _frac(wrong_a),
            "b": _frac(p * wrong_a + qq * j + r * (c - a)),
            "c": _frac(k),
            "d": _two(wrong_a / (c - a)),
        }
        return "a", replies, sum(MARKS) - MARKS[0]

    follow_through_check(TOPIC, SKILL, chain)
