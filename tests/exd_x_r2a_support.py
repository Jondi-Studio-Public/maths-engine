"""Shared helpers for the `test_exd_x_r2a_*` files: independent parsing and the checks every chain runs.

Nothing here imports a generator module. Each test file re-derives the answers of its chain from the RENDERED stem
with SymPy and exact fractions, and these helpers read a rendered function, an answer and a reply back into values.
"""

from __future__ import annotations

import re
from collections import Counter
from fractions import Fraction
from functools import cache

import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from mathgen import make
from mathgen.engine import _tuplify, grade, load_skills
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects, prompt_shape

x = sympy.Symbol("x")
t = sympy.Symbol("t")
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)
_PROBES = [sympy.Rational(7, 3), sympy.Rational(11, 5), sympy.Rational(-5, 7), sympy.Rational(13, 4), 3, 5, -2, 9]


@cache
def skill(skill_id: str):
    return next(s for s in load_skills()[1] if s.id == skill_id)


def exam_source(skill_id: str):
    return [s for s in skill(skill_id).sources if s.difficulty == "exam" and s.effective_key == "exam"]


def parse_math(text: str, variable: sympy.Symbol = x):
    """Parse learner-style maths (``3sin^2(4x)``, ``(2x + 1)^3(x^2 - 5)``, ``√(x)``, ``2/√(x + 1)``)."""
    s = text.replace("√(", "sqrt(").replace("√", "sqrt").replace("·", "*").replace("π", "pi")
    s = re.sub(r"\b(sin|cos|tan)\^(\d)\(([^()]*)\)", r"(\1(\3))**\2", s)
    s = s.replace("^", "**")
    local = {variable.name: variable, "pi": sympy.pi, "e": sympy.E}
    return parse_expr(s, transformations=_TRANSFORMS, local_dict=local)


def value(text: str, variable: sympy.Symbol = x):
    """The exact value or expression an answer text states (a leading ``y =`` is dropped)."""
    return parse_math(re.sub(r"^\s*y\s*=\s*", "", text), variable)


def same_function(left, right, variable: sympy.Symbol = x) -> bool:
    """True when two expressions agree numerically at several probe points (at least three must be finite)."""
    agreed = 0
    for point in _PROBES:
        a = complex(sympy.N(sympy.sympify(left).subs(variable, point)))
        b = complex(sympy.N(sympy.sympify(right).subs(variable, point)))
        if any(v != v or abs(v) == float("inf") for v in (a, b)):
            continue
        if abs(a - b) > 1e-9 * (1 + abs(a)):
            return False
        agreed += 1
    assert agreed >= 3, (left, right)
    return True


@cache
def _exam(topic: str, seed: int):
    return make(topic, seed=seed, difficulty="exam")


def questions(topic: str, n: int, start: int = 0):
    """The first `n` exam questions of `topic` from `start`; each seed is generated once per test process."""
    return [_exam(topic, s) for s in range(start, start + n)]


def hard_shapes(skill_id: str, n: int = 60) -> set[str]:
    hard = next(s for s in skill(skill_id).sources if s.difficulty == "hard")
    return {prompt_shape(make(hard.topic, seed=s, difficulty="hard", **_tuplify(hard.params))) for s in range(n)}


def check_wired(skill_id: str, topic: str) -> None:
    exam = exam_source(skill_id)
    assert [s.topic for s in exam] == [topic], exam
    assert exam[0].stages == ["exam"], exam[0].stages


def check_shapes(skill_id: str, topic: str, n: int = 60) -> None:
    exam = {prompt_shape(q) for q in questions(topic, n)}
    assert len(exam) >= 3, exam
    assert exam - hard_shapes(skill_id, n), "the exam stem adds no shape the hard tier lacks"


def check_structure(topic: str, kinds: list[str], marks: list[int], follow_through: list[bool], n: int = 60) -> None:
    for q in questions(topic, n):
        assert [p.label for p in q.parts] == [chr(97 + i) for i in range(len(kinds))]
        assert [p.answer_kind for p in q.parts] == kinds
        assert [p.marks for p in q.parts] == marks
        assert [p.follow_through is not None for p in q.parts] == follow_through
        for p in q.parts:
            assert 2 <= len(p.distractors) <= 3, (q.text, p.label, p.distractors)
            for d in p.distractors:
                note = p.option_notes.get(d)
                assert note in CATALOGUE, (p.label, d, note)


def check_graded(skill_id: str, topic: str, n: int = 120) -> None:
    """The composed key earns full marks; each distractor, dropped into its own part, earns that part nothing."""
    sk = skill(skill_id)
    for q in questions(topic, n):
        full = grade(sk, q, q.answer_text)
        assert full.correct and full.marks == full.max_marks, (q.text, q.answer_text)
        canon = {p.label: p.answer_text for p in q.parts}
        for p in q.parts:
            for d in p.distractors:
                result = grade(sk, q, {**canon, p.label: d})
                assert result.parts[p.label].marks == 0, (q.text, p.label, d)
                assert result.parts[p.label].follow_through is not True


def check_answer_space(topic: str, n: int = 300, minimum: int = 100, ceiling: float = 0.05) -> None:
    tuples = Counter(tuple(p.answer_text for p in q.parts) for q in questions(topic, n))
    assert len(tuples) >= minimum, len(tuples)
    assert max(tuples.values()) / n <= ceiling, tuples.most_common(3)


def check_clean_and_deterministic(topic: str, n: int = 60) -> None:
    for seed in range(n):
        q = make(topic, seed=seed, difficulty="exam")
        assert find_text_defects(q) == [], (seed, q.text, find_text_defects(q))
        for p in q.parts:
            assert "**" not in p.answer_text and "**" not in p.prompt
        assert q.to_json() == make(topic, seed=seed, difficulty="exam").to_json()


def frac(text: str) -> Fraction:
    return Fraction(str(sympy.Rational(sympy.sympify(text.replace("^", "**")))))


def rational_text(r) -> str:
    r = sympy.Rational(r)
    return str(r.p) if r.q == 1 else f"{r.p}/{r.q}"


def stem_function(q, variable: sympy.Symbol = x):
    """The function defined in the rendered stem (``f(x) = ...`` up to the first full stop)."""
    return parse_math(re.search(r"f\(x\) = (.+?)(?: for | on |, |\. )", q.stem).group(1), variable)


def stem_point(q) -> int:
    """The last ``x = <integer>`` in the stem."""
    return int(re.findall(r"x = (-?\d+)", q.stem)[-1])


def tangent_replies(f, x0, a_text: str, variant: str, gradient_shift=0):
    """A learner's replies to a tangent-style chain who answered (a) with `a_text` (and, optionally, shifted their
    gradient by `gradient_shift`), carrying the error forward with correct working. `variant` is "x" (x-intercept of
    the tangent), "area" (triangle with the axes) or "normal" (y-intercept of the normal)."""
    g = value(a_text)
    f0 = f.subs(x, x0)
    m = sympy.nsimplify(g.subs(x, x0)) + gradient_shift
    if m == 0 or not m.is_Rational:
        return None
    slope = -1 / m if variant == "normal" else m
    line = slope * (x - x0) + f0
    c0 = line.subs(x, 0)
    if variant == "x":
        last = -c0 / slope
    elif variant == "area":
        last = c0**2 / (2 * abs(slope))
    else:
        last = c0
    if c0 == 0 or not sympy.Rational(last).is_Rational:
        return None
    return {"a": a_text, "b": rational_text(m), "c": "y = " + str(sympy.expand(line)), "d": rational_text(last)}


def check_follow_through_chain(skill_id: str, topic: str, variant: str, n: int = 100, minimum: int = 40) -> None:
    """Replies built from each question's first distractor of (a) earn every later part (with `follow_through`)."""
    sk = skill(skill_id)
    used = 0
    for q in questions(topic, n):
        f, x0 = stem_function(q), stem_point(q)
        canon_b = frac(q.parts[1].answer_text)
        replies = None
        for wrong in q.parts[0].distractors:
            candidate = tangent_replies(f, x0, wrong, variant)
            if (
                candidate
                and sympy.Rational(frac(candidate["b"])) != canon_b
                and frac(candidate["d"]) != frac(q.parts[3].answer_text)
            ):
                replies = candidate
                break
        if replies is None:
            continue
        result = grade(sk, q, replies)
        assert result.parts["a"].marks == 0
        for label in "bcd":
            part = result.parts[label]
            assert part.marks == part.max_marks and part.follow_through is True, (q.text, label, replies, part)
        assert not result.correct and result.marks == result.max_marks - 2
        used += 1
    assert used >= minimum, used


def check_all_wrong_earns_nothing(skill_id: str, topic: str, replies: dict, n: int = 20) -> None:
    sk = skill(skill_id)
    for q in questions(topic, n):
        result = grade(sk, q, replies)
        assert result.marks == 0 and not result.correct, (q.text, replies)
