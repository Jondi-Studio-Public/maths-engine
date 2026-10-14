"""Exam tier of `scientific-notation`: a product of two scientific-notation quantities, its exponent, a unit conversion."""

from __future__ import annotations

import re
from fractions import Fraction

from mathgen.engine import grade
from tests.exd_x_sc6_support import (
    assert_answer_space,
    assert_follow_through,
    assert_grading,
    assert_new_shapes,
    assert_no_text_defects,
    assert_source_wired,
    canonical,
    exam,
    frac,
    skill_of,
)

SKILL = skill_of("scientific-notation")
TOPIC = "exd_x_sc6_sci_notation"
_SUP = {c: str(i) for i, c in enumerate("⁰¹²³⁴⁵⁶⁷⁸⁹")} | {"⁻": "-"}
_SCI = re.compile(r"(\d+\.\d) × 10([⁰¹²³⁴⁵⁶⁷⁸⁹⁻]+)")
_UNITS = {
    "megalitres": Fraction(1, 10**6),
    "megawatt-hours": Fraction(1, 1000),
    "millions of dollars": Fraction(1, 10**6),
    "millimetres": Fraction(1000),
    "centimetres": Fraction(100),
    "millilitres": Fraction(1000),
}


def _value(text: str, start: int = 0):
    m = _SCI.search(text, start)
    return Fraction(m.group(1)) * Fraction(10) ** int("".join(_SUP[c] for c in m.group(2))), m.end()


def _plain(value: Fraction) -> str:
    from decimal import Decimal

    text = format(Decimal(value.numerator) / Decimal(value.denominator), "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def _exponent(value: Fraction) -> int:
    n = 0
    while value >= 10:
        value /= 10
        n += 1
    while value < 1:
        value *= 10
        n -= 1
    return n


def oracle(q) -> dict[str, str]:
    first, end = _value(q.stem)
    second, _ = _value(q.stem, end)
    total = first * second
    unit = next(u for u in _UNITS if u in q.parts[2].prompt)
    return {"a": _plain(total), "b": str(_exponent(total)), "c": _plain(total * _UNITS[unit])}


def test_chain_shape():
    for seed in range(60):
        q = exam(SKILL, seed)
        assert [p.follow_through is not None for p in q.parts] == [False, True, True]
        assert all(p.answer_kind == "number" for p in q.parts)


def test_both_signs_and_renormalising_draws_occur():
    exps = [int(exam(SKILL, s).parts[1].answer_text) for s in range(120)]
    assert min(exps) < 0 < max(exps)
    renorm = 0
    for s in range(120):
        q = exam(SKILL, s)
        first, end = _value(q.stem)
        second, _ = _value(q.stem, end)
        renorm += _exponent(first * second) != _exponent(first) + _exponent(second)
    assert 40 <= renorm <= 80


def test_key_rederived_from_prompt():
    for seed in range(300):
        q = exam(SKILL, seed)
        got = oracle(q)
        want = canonical(q)
        assert {k: frac(v) for k, v in got.items()} == {k: frac(v) for k, v in want.items()}, (seed, q.stem)


def test_grader_accepts_key_rejects_distractors():
    assert_grading(SKILL)


def test_follow_through_from_wrong_total():
    checked = 0
    for seed in range(120):
        q = exam(SKILL, seed)
        wrong_a = frac(q.parts[0].distractors[0])
        unit = next(u for u in _UNITS if u in q.parts[2].prompt)
        b = _exponent(wrong_a)
        c = wrong_a * _UNITS[unit]
        assert_follow_through(SKILL, q, {"a": q.parts[0].distractors[0], "b": str(b), "c": _plain(c)}, "a")
        checked += 1
    assert checked == 120


def test_all_wrong_earns_nothing():
    for seed in range(30):
        result = grade(SKILL, exam(SKILL, seed), {"a": "1", "b": "99", "c": "7"})
        assert result.marks == 0


def test_deterministic():
    assert exam(SKILL, 7).to_dict() == exam(SKILL, 7).to_dict()
