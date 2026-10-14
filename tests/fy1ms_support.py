"""Shared checks for the first-year multivariable, 3D geometry and statistics skills (`tests/test_fy1ms_*.py`).

Each skill's test file supplies its own independent oracle (a re-derivation of every key from the question TEXT);
the grading, diagnosis, follow-through and wiring checks that every one of these skills needs live here.
"""

from __future__ import annotations

import re
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction

import sympy
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects

SEEDS = range(60)

_TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)


def load(skill_id: str):
    return next(s for s in load_skills()[1] if s.id == skill_id)


def rnd(value, dp: int) -> str:
    """`value` rounded half up to `dp` places, written with exactly `dp` places (an integer string when dp is 0)."""
    if isinstance(value, Fraction):
        value = Decimal(value.numerator) / Decimal(value.denominator)
    elif isinstance(value, sympy.Basic):
        value = Decimal(str(sympy.N(value, 40)))
    else:
        value = Decimal(repr(float(value)))
    quantum = Decimal(1).scaleb(-dp)
    text = str(value.quantize(quantum, rounding=ROUND_HALF_UP))
    return "0" + text[1:] if text.startswith("-0") and float(text) == 0 else text


def dp_of(text: str) -> int:
    return len(text.split(".")[1]) if "." in text else 0


def parse_plain(text: str, **symbols):
    """The SymPy expression for a plain-text formula such as ``3x²y - 2y`` or ``2√(x² + y²)``."""
    s = text.replace("²", "**2").replace("³", "**3").replace("−", "-").replace("√", "sqrt").replace("^", "**")
    return parse_expr(s, local_dict=symbols, transformations=_TRANSFORMS)


def exact_decimal(text: str) -> sympy.Rational:
    return sympy.Rational(text)


# --- wiring ------------------------------------------------------------------------------------------------


def assert_wiring(sk, topic: str, exam_topics: list[str], answer: str = "number") -> None:
    got = [(s.topic, s.difficulty, s.stages) for s in sk.sources]
    want = [(topic, "easy", [0, 1, 2]), (topic, "medium", [3, 4]), (topic, "hard", [5, 6, "maintenance"])]
    want += [(t, "exam", ["exam"]) for t in exam_topics]
    assert got == want
    assert sk.status == "active" and sk.answer == answer


# --- single-answer tiers -----------------------------------------------------------------------------------


def single(topic: str, tier: str, seed: int):
    return make(topic, seed=seed, difficulty=tier)


def assert_single_grading(sk, topic: str, tier: str, seeds=SEEDS) -> None:
    """The key is accepted, every distractor is rejected and diagnosed as its own error, a unit-off value is rejected."""
    for seed in seeds:
        q = single(topic, tier, seed)
        assert grade(sk, q, q.answer_text).correct, (tier, seed)
        assert len(q.distractors) >= 2, (tier, seed)
        for d in q.distractors:
            assert q.option_notes.get(d), (tier, seed, d)
            result = grade(sk, q, d)
            assert not result.correct, (tier, seed, d)
            assert result.misconception == q.option_notes[d], (tier, seed, d)
            assert ("." in d) == ("." in q.answer_text) and dp_of(d) == dp_of(q.answer_text), (tier, seed, d)
        dp = dp_of(q.answer_text)
        off = Fraction(q.answer_text) + (Fraction(2, 10**dp) if dp else 2)
        assert not grade(sk, q, rnd(off, dp)).correct, (tier, seed)


def assert_clean_variety_and_determinism(
    topic: str, tier: str, seeds=SEEDS, *, distinct_prompts=40, distinct_answers=12
):
    prompts, answers = set(), []
    for seed in seeds:
        q = single(topic, tier, seed)
        assert q.to_json() == single(topic, tier, seed).to_json()
        assert find_text_defects(q, include_working=True) == [], (tier, seed)
        prompts.add(q.text)
        answers.append(q.answer_text)
    assert len(prompts) >= distinct_prompts, (tier, len(prompts))
    assert len(set(answers)) >= distinct_answers, (tier, len(set(answers)))
    assert max(answers.count(a) for a in set(answers)) <= 8, tier


# --- exam chains -------------------------------------------------------------------------------------------


def exam(topic: str, seed: int):
    return make(topic, seed=seed, difficulty="exam")


def canonical_reply(q) -> dict:
    return {p.label: p.answer_text for p in q.parts}


def assert_exam_shape(topic: str, labels: list[str], marks: list[int], ft: list[bool], seeds=SEEDS) -> None:
    for seed in seeds:
        q = exam(topic, seed)
        assert [p.label for p in q.parts] == labels
        assert [p.marks for p in q.parts] == marks
        assert [p.follow_through is not None for p in q.parts] == ft
        assert find_text_defects(q, include_working=True) == [], seed
        assert q.to_json() == exam(topic, seed).to_json()


def assert_exam_grading(sk, topic: str, seeds=range(80)) -> None:
    """Full marks for the canonical reply; every distractor of every part earns that part nothing, and has a note."""
    for seed in seeds:
        q = exam(topic, seed)
        full = grade(sk, q, canonical_reply(q))
        total = sum(p.marks for p in q.parts)
        assert full.correct and full.marks == full.max_marks == total, (seed, full)
        for p in q.parts:
            assert len(p.distractors) >= 2, (seed, p.label)
            for d in p.distractors:
                assert p.option_notes.get(d), (seed, p.label, d)
                result = grade(sk, q, {**canonical_reply(q), p.label: d})
                assert result.parts[p.label].marks == 0, (seed, p.label, d)


_PART_REF = re.compile(r"part_([a-z])")


def _follow_value(part, replies: dict):
    """The reply a learner who used `replies` for the parts `part` follows would get, or None when undefined."""
    expr = sympy.sympify(part.follow_through, locals={"x": sympy.Symbol("x"), "y": sympy.Symbol("y")})
    subs = {}
    for label in set(_PART_REF.findall(part.follow_through)):
        try:
            subs[sympy.Symbol(f"part_{label}")] = sympy.Rational(replies[label]) if replies[label] else sympy.Integer(0)
        except (TypeError, ValueError):
            return None
    value = expr.subs(subs)
    if part.answer_kind == "expression":
        return str(sympy.expand(value))
    if not value.is_number or value.has(sympy.zoo, sympy.nan, sympy.oo):
        return None
    dp = dp_of(part.answer_text)
    return rnd(value, dp)


def follow_through_reply(q, wrong_label: str, wrong_reply: str) -> dict:
    """The replies of a learner who answered `wrong_label` with `wrong_reply` and used it correctly afterwards."""
    replies = canonical_reply(q)
    replies[wrong_label] = wrong_reply
    wrong_labels = {wrong_label}
    for p in q.parts:
        if p.follow_through and set(_PART_REF.findall(p.follow_through)) & wrong_labels:
            value = _follow_value(p, replies)
            if value is None:
                continue
            replies[p.label] = value
            wrong_labels.add(p.label)
    return replies


def assert_follow_through_credited(sk, topic: str, seeds=range(80), *, minimum: int = 40) -> None:
    """A wrong earlier answer used correctly in later parts earns those parts their marks, flagged as follow-through."""
    credited = 0
    for seed in seeds:
        q = exam(topic, seed)
        for first in q.parts:
            if not first.distractors or not any(
                first.label in set(_PART_REF.findall(p.follow_through or "")) for p in q.parts
            ):
                continue
            wrong = first.distractors[0]
            replies = follow_through_reply(q, first.label, wrong)
            result = grade(sk, q, replies)
            assert result.parts[first.label].marks == 0, (seed, first.label)
            for p in q.parts:
                if p.label in replies and replies[p.label] != p.answer_text and p.label != first.label:
                    got = result.parts[p.label]
                    if got.marks == p.marks and got.follow_through:
                        credited += 1
    assert credited >= minimum, credited


def assert_exam_variety(topic: str, seeds=range(200), *, distinct: int = 100, repeat: int = 8) -> None:
    answers = [exam(topic, s).answer_text for s in seeds]
    assert len(set(answers)) >= distinct, len(set(answers))
    assert max(answers.count(a) for a in set(answers)) <= repeat
