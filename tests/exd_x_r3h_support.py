"""Shared checks for the `test_exd_x_r3h_*` files: answer space, grading every option, and follow-through.

Nothing here imports a generator module; each test file derives its keys from the rendered stem itself.
"""

from __future__ import annotations

from collections import Counter
from fractions import Fraction

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.grading import parts as parts_mod
from mathgen.grading import typed
from mathgen.text_quality import find_text_defects, prompt_shape


def skill(skill_id: str):
    return next(s for s in load_skills()[1] if s.id == skill_id)


def exam_source(skill_id: str):
    sources = [s for s in skill(skill_id).sources if s.difficulty == "exam"]
    assert len(sources) == 1, sources
    return sources[0]


def draw(topic: str, seed: int):
    return make(topic, seed=seed, difficulty="exam")


def shape_check(skill_id: str, topic: str, n: int = 60) -> None:
    exam = {prompt_shape(draw(topic, s)) for s in range(n)}
    hard_src = next(s for s in skill(skill_id).sources if s.difficulty == "hard")
    params = dict(hard_src.params or {})
    hard = {prompt_shape(make(hard_src.topic, seed=s, **params)) for s in range(n)}
    assert len(exam) >= 3, exam
    assert exam - hard, (exam, hard)


def answer_space(topic: str, n: int = 300) -> Counter:
    tuples = Counter(draw(topic, s).answer_text for s in range(n))
    assert len(tuples) >= 100, len(tuples)
    assert max(tuples.values()) <= 0.05 * n, tuples.most_common(3)
    return tuples


def canonical(q) -> dict[str, str]:
    return {p.label: p.answer_text for p in q.parts}


def full_marks_and_distractors(skill_id: str, topic: str, n: int = 120) -> None:
    sk = skill(skill_id)
    for seed in range(n):
        q = draw(topic, seed)
        result = grade(sk, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks, (seed, q.answer_text)
        canon = canonical(q)
        for p in q.parts:
            assert len(p.distractors) >= 2, (seed, p.label, p.distractors)
            for d in p.distractors:
                one = grade(sk, q, {**canon, p.label: d})
                assert one.parts[p.label].marks == 0, (seed, p.label, d, canon[p.label])
        assert find_text_defects(q) == [], (seed, q.text)
        assert q.to_json() == draw(topic, seed).to_json()


def follow_through_check(skill_id: str, topic: str, n: int = 80) -> int:
    """A wrong answer to a part that later parts build on, carried correctly through them, earns their marks."""
    sk = skill(skill_id)
    checked = 0
    for seed in range(n):
        q = draw(topic, seed)
        used = {name for p in q.parts if p.follow_through for name in parts_mod.named_parts(p)}
        for source in [p for p in q.parts if p.label in used]:
            wrong = next((d for d in source.distractors if parts_mod.reply_value(source.answer_kind, d) != 0), None)
            if wrong is None:
                continue
            canon = canonical(q)
            replies = dict(canon)
            replies[source.label] = wrong
            values = {p.label: parts_mod.reply_value(p.answer_kind, canon[p.label]) for p in q.parts}
            values[source.label] = parts_mod.reply_value(source.answer_kind, wrong)
            changed = {source.label}
            dependent = []
            ok = True
            after = q.parts[q.parts.index(source) + 1 :]
            for p in after:
                if p.follow_through and changed & set(parts_mod.named_parts(p)):
                    target = parts_mod.follow_through_target(p, values)
                    if target is None:
                        ok = False
                        break
                    named = set(parts_mod.named_parts(p))
                    own = parts_mod.reply_value(p.answer_kind, target.answer_text)
                    if (own == 0 and values[p.label] != 0) or own in {values[n] for n in named}:
                        ok = False  # the engine withholds follow-through into a zero or a repeated reply
                        break
                    replies[p.label] = target.answer_text
                    values[p.label] = parts_mod.reply_value(p.answer_kind, target.answer_text)
                    if target.answer_text != canon[p.label]:
                        changed.add(p.label)
                    dependent.append(p)
            if not ok or not dependent:
                continue
            result = grade(sk, q, replies)
            assert result.parts[source.label].marks == 0
            for p in dependent:
                assert result.parts[p.label].marks == p.marks, (seed, p.label, replies, result.parts[p.label])
                if not typed.grade(p.answer_kind, p, replies[p.label]).correct:  # not already right outright
                    assert result.parts[p.label].follow_through is True, (seed, p.label, replies)
            checked += 1
    assert checked >= n // 2, checked
    return checked


def as_fraction(text: str) -> Fraction:
    return Fraction(text)


# -- independent parsing of rendered text -----------------------------------------------------------------

import re  # noqa: E402
from decimal import ROUND_HALF_UP, Decimal  # noqa: E402

import sympy  # noqa: E402
from sympy.parsing.sympy_parser import (  # noqa: E402
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

_TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)


def parse_math(text: str, **symbols):
    """Learner-style algebra such as ``3x^2 - 2xy + (k + 3)z`` as a SymPy expression."""
    local = {name: sympy.Symbol(name) for name in symbols} or {}
    return parse_expr(text, transformations=_TRANSFORMS, local_dict=local)


def rnd(value, places: int) -> str:
    """`value` (an int, Fraction, float or SymPy number) rounded half up and written at `places` places."""
    exact = Decimal(str(sympy.N(value, 30))) if isinstance(value, sympy.Basic) else Decimal(str(float(value)))
    if isinstance(value, (int, Fraction)) and not isinstance(value, bool):
        exact = Decimal(value.numerator) / Decimal(value.denominator) if isinstance(value, Fraction) else Decimal(value)
    quantum = Decimal(1).scaleb(-places)
    return str(exact.quantize(quantum, rounding=ROUND_HALF_UP))


def first_int(pattern: str, text: str) -> int:
    match = re.search(pattern, text)
    assert match, (pattern, text)
    return int(match.group(1))


def run_oracle(topic: str, oracle, n: int = 300) -> None:
    bad = []
    for seed in range(n):
        q = draw(topic, seed)
        expected = oracle(q)
        got = canonical(q)
        if expected != got:
            bad.append((seed, expected, got))
    assert bad == [], bad[:3]
