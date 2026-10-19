"""Shared assertions for the `exd_x_r2b_*` multi-part exam-tier test files (not a test module itself)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import yaml

from mathgen import make
from mathgen.engine import grade
from mathgen.misconceptions import CATALOGUE
from mathgen.skills import load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

ROOT = Path(__file__).resolve().parent.parent


def skill_yaml(skill_id: str) -> dict:
    (path,) = list((ROOT / "skills").rglob(f"{skill_id}.yaml"))
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def skill_object(skill_id: str):
    _, skills = load_skills()
    return next(s for s in skills if s.id == skill_id)


def exam_question(topic: str, seed: int):
    return make(topic, seed=seed, difficulty="exam", tier="exam")


def hard_question(skill_id: str, seed: int):
    hard = [s for s in skill_yaml(skill_id)["sources"] if s["difficulty"] == "hard"][0]
    params = {k: (tuple(v) if isinstance(v, list) else v) for k, v in (hard.get("params") or {}).items()}
    return make(hard["topic"], seed=seed, difficulty="hard", **params)


def assert_registered(skill_id: str, topic: str) -> None:
    sources = skill_yaml(skill_id)["sources"]
    # A second exam source keyed exam-single is the maintainer's re-added single-part question (bf2fca7).
    exam = [s for s in sources if s["difficulty"] == "exam" and s.get("key") != "exam-single"]
    assert len(exam) == 1
    assert exam[0]["topic"] == topic
    assert exam[0]["stages"] == ["exam"]
    assert exam[0]["params"] == {"tier": "exam"}


def assert_clean_and_deterministic(topic: str, seeds=range(60)) -> None:
    for seed in seeds:
        q = exam_question(topic, seed)
        assert find_text_defects(q) == [], (seed, q.text)
        assert q.to_json() == exam_question(topic, seed).to_json()


def assert_shapes(skill_id: str, topic: str) -> None:
    exam = {prompt_shape(exam_question(topic, s)) for s in range(60)}
    hard = {prompt_shape(hard_question(skill_id, s)) for s in range(60)}
    assert len(exam) >= 3, exam
    assert exam - hard, "the exam tier adds no prompt shape beyond the hard tier"


def answer_tuple(q) -> tuple:
    return tuple(p.answer_text for p in q.parts)


def assert_answer_space(topic: str, seeds=range(300), distinct=100, top=0.05) -> None:
    counts = Counter(answer_tuple(exam_question(topic, s)) for s in seeds)
    assert len(counts) >= distinct, len(counts)
    value, n = counts.most_common(1)[0]
    assert n / len(seeds) <= top, (value, n)


def assert_chain_shape(topic: str, kinds, marks, follow_through) -> None:
    for seed in range(40):
        q = exam_question(topic, seed)
        assert [p.answer_kind for p in q.parts] == list(kinds)
        assert [p.marks for p in q.parts] == list(marks)
        assert [bool(p.follow_through) for p in q.parts] == list(follow_through)


def assert_tagged(topic: str, seeds=range(60)) -> None:
    """Every distractor of every part is tagged with a catalogue id and at least two are offered."""
    for seed in seeds:
        q = exam_question(topic, seed)
        for p in q.parts:
            assert len(p.distractors) >= 2, (seed, p.label)
            for d in p.distractors:
                tag = p.option_notes.get(d)
                assert tag in CATALOGUE, (seed, p.label, d, tag)


def assert_grading(skill_id: str, topic: str, seeds=range(100)) -> None:
    """The composed key earns every mark and every distractor of every part earns nothing for its part."""
    sk = skill_object(skill_id)
    for seed in seeds:
        q = exam_question(topic, seed)
        canon = {p.label: p.answer_text for p in q.parts}
        full = grade(sk, q, canon)
        assert full.correct and full.marks == full.max_marks, (seed, q.answer_text)
        assert grade(sk, q, q.answer_text).correct
        for p in q.parts:
            for d in p.distractors:
                result = grade(sk, q, {**canon, p.label: d})
                assert result.parts[p.label].marks == 0, (seed, p.label, d, p.answer_text)
        junk = grade(sk, q, {p.label: f"{7919 * (i + 1)}.5" for i, p in enumerate(q.parts)})
        assert junk.marks == 0, (seed, junk.marks)


# -- reading a stem back with independent code --------------------------------------------------------------

import re  # noqa: E402

import sympy  # noqa: E402
from sympy.parsing.sympy_parser import (  # noqa: E402
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

_TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)


def parse_poly(text: str, **symbols):
    """A polynomial written the way the stems write it ("kx^3 - 9x^2 + 24x + 7") as a SymPy expression."""
    return parse_expr(text, local_dict=symbols, transformations=_TRANSFORMS)


def to_rational(value) -> sympy.Rational:
    from fractions import Fraction

    f = Fraction(str(value))
    return sympy.Rational(f.numerator, f.denominator)


def dp(value, places: int) -> str:
    """`value` (anything SymPy can evaluate) rounded half up to `places` decimal places, as the text an answer shows."""
    from decimal import ROUND_HALF_UP, Decimal

    exact = Decimal(str(sympy.N(value, 40)))
    return str(exact.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))


def rounded(value, places: int) -> sympy.Rational:
    """`value` rounded half up to `places` decimal places, as an exact rational."""
    return sympy.Rational(dp(value, places))
