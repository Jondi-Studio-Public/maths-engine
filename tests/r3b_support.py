"""Shared assertions for the `test_r3b_*` skill test files (not a test module itself).

The helpers re-derive keys from the printed stem with a small reader of the stems' plain-text maths and numerical
differentiation or quadrature, so a test does not lean on the generator's own SymPy construction.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

import sympy
import yaml
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from mathgen import make
from mathgen.engine import grade
from mathgen.misconceptions import CATALOGUE
from mathgen.skills import load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

ROOT = Path(__file__).resolve().parent.parent
TIERS = ("easy", "medium", "hard")
X = sympy.Symbol("x")
A = sympy.Symbol("a")
K = sympy.Symbol("k")
_TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)
_BARS = re.compile(r"ln\|([^|]*)\|")
_POWER_OF_FUNCTION = re.compile(r"(sin|cos|tan)\^(\d+)\(([^()]*)\)")


def skill_yaml(skill_id: str) -> dict:
    (path,) = list((ROOT / "skills").rglob(f"{skill_id}.yaml"))
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def skill_object(skill_id: str):
    _, skills = load_skills()
    return next(s for s in skills if s.id == skill_id)


def tier_question(topic: str, tier: str, seed: int):
    return make(topic, seed=seed, difficulty=tier)


def exam_question(topic: str, seed: int):
    return make(topic, seed=seed, difficulty="exam")


# -- reading a stem -----------------------------------------------------------------------------------------------


def read_math(text: str):
    """The expression in `x` that a stem's plain-text maths (`3e^(2x)`, `sin^3(2x) cos^2(x)`, `ln(x)`) stands for."""
    text = _BARS.sub(r"ln(Abs(\1))", text)
    text = _POWER_OF_FUNCTION.sub(r"((\1(\3))**\2)", text.strip().replace("×", "*").replace("·", "*"))
    names = {
        "x": X,
        "e": sympy.E,
        "ln": sympy.log,
        "pi": sympy.pi,
        "a": A,
        "k": K,
        "Abs": sympy.Abs,
        "arctan": sympy.atan,
    }
    names.update({name: sympy.Symbol(name) for name in "ABCDFGPQR"})
    return parse_expr(text, local_dict=names, transformations=_TRANSFORMS)


def integrand(stem: str):
    """The integrand of the first indefinite integral in a stem."""
    return read_math(re.search(r"∫ (.*?) dx", stem).group(1))


def definite_parts(stem: str):
    """(integrand, lower limit, upper limit) of the first definite integral in a stem."""
    found = re.search(r"∫ from (.*?) to (.*?) of (.*?) dx", stem)
    return read_math(found.group(3)), read_math(found.group(1)), read_math(found.group(2))


def _real(f, point):
    try:
        value = complex(f(point))
    except (ValueError, ZeroDivisionError, OverflowError, TypeError):
        return None
    return value.real if abs(value.imag) < 1e-9 else None


def numeric_function(expr):
    return sympy.lambdify(X, expr, modules=["math"])


def assert_derivative_is(key, f, points=(0.37, 0.91, 1.41, 2.63, 3.17), minimum=2) -> None:
    """The key, differentiated by central differences at points where it is defined, equals the integrand f."""
    g, target = numeric_function(key), numeric_function(f)
    h = 1e-5
    checked = 0
    for p in points:
        lo, hi, centre = _real(g, p - h), _real(g, p + h), _real(target, p)
        if lo is None or hi is None or centre is None:
            continue
        slope = (hi - lo) / (2 * h)
        assert math.isclose(slope, centre, rel_tol=1e-4, abs_tol=1e-5), (key, f, p, slope, centre)
        checked += 1
    assert checked >= minimum, (key, f, checked)


def simpson(f, lo: float, hi: float, strips: int = 4000) -> float:
    g = numeric_function(f)
    h = (hi - lo) / strips
    total = g(lo) + g(hi) + sum((4 if i % 2 else 2) * g(lo + i * h) for i in range(1, strips))
    return total * h / 3


def key_value(q) -> float:
    return float(sympy.N(sympy.sympify(q.answer, locals={"Abs": sympy.Abs}), 25))


def key_expression(q):
    return sympy.sympify(q.answer, locals={"Abs": sympy.Abs})


def assert_indefinite_key(q) -> None:
    assert_derivative_is(key_expression(q), integrand(q.text))


def assert_definite_key(q, rel=1e-6) -> None:
    f, lo, hi = definite_parts(q.text)
    exact = simpson(f, float(sympy.N(lo)), float(sympy.N(hi)))
    assert math.isclose(exact, key_value(q), rel_tol=rel, abs_tol=1e-8), (q.text, exact, q.answer_text)


# -- wiring, cleanliness, grading ---------------------------------------------------------------------------------


def assert_wired(
    skill_id: str, topic: str, exam_topic: str, answer: str, prerequisites: list[str], exam_params: dict | None = None
) -> None:
    data = skill_yaml(skill_id)
    assert data["answer"] == answer
    assert data["prerequisites"] == prerequisites
    by_difficulty = {s["difficulty"]: s for s in data["sources"]}
    for tier in TIERS:
        assert by_difficulty[tier]["topic"] == topic
        assert not by_difficulty[tier]["params"]
    assert by_difficulty["exam"]["topic"] == exam_topic
    assert by_difficulty["exam"]["stages"] == ["exam"]
    assert by_difficulty["exam"]["params"] == (exam_params or {})
    assert skill_object(skill_id).status == "active"


def assert_clean_and_deterministic(topic: str, tier: str, seeds=range(40)) -> None:
    for seed in seeds:
        q = tier_question(topic, tier, seed)
        assert find_text_defects(q) == [], (seed, q.text)
        assert q.to_json() == tier_question(topic, tier, seed).to_json()


def assert_exam_clean_and_deterministic(topic: str, seeds=range(40)) -> None:
    for seed in seeds:
        q = exam_question(topic, seed)
        assert find_text_defects(q) == [], (seed, q.text)
        assert q.to_json() == exam_question(topic, seed).to_json()


def assert_tier_shapes(topic: str, exam_topic: str, minimum_hard=2) -> None:
    """The tiers differ in the kind of prompt they set and the exam adds a shape beyond the hard tier."""
    shapes = {t: {prompt_shape(tier_question(topic, t, s)) for s in range(60)} for t in TIERS}
    assert len(shapes["hard"]) >= minimum_hard, shapes["hard"]
    assert shapes["hard"] - shapes["easy"], "hard sets no prompt shape that easy does not"
    exam = {prompt_shape(exam_question(exam_topic, s)) for s in range(60)}
    assert exam - shapes["hard"], "the exam tier adds no prompt shape beyond the hard tier"


def assert_answer_space(topic: str, tier: str, seeds=range(200), distinct=12, top=0.25) -> None:
    counts = Counter(tier_question(topic, tier, s).answer_text for s in seeds)
    assert len(counts) >= distinct, (tier, len(counts))
    value, n = counts.most_common(1)[0]
    assert n / len(seeds) <= top, (tier, value, n)


def assert_exam_answer_space(topic: str, seeds=range(200), distinct=60, top=0.05) -> None:
    counts = Counter(tuple(p.answer_text for p in exam_question(topic, s).parts) for s in seeds)
    assert len(counts) >= distinct, len(counts)
    value, n = counts.most_common(1)[0]
    assert n / len(seeds) <= top, (value, n)


def assert_single_graded(skill_id: str, topic: str, tier: str, seeds=range(60)) -> None:
    """The key is graded correct, every distractor is graded wrong, and each carries a catalogue tag."""
    sk = skill_object(skill_id)
    for seed in seeds:
        q = tier_question(topic, tier, seed)
        assert grade(sk, q, q.answer_text).correct, (seed, q.answer_text)
        assert len(q.distractors) >= 2, (seed, q.distractors)
        assert len(set(q.distractors)) == len(q.distractors)
        for d in q.distractors:
            assert d != q.answer_text
            assert not grade(sk, q, d).correct, (seed, q.text, q.answer_text, d)
            assert q.option_notes.get(d) in CATALOGUE, (seed, d, q.option_notes.get(d))


def assert_chain_shape(topic: str, kinds, marks, follow_through) -> None:
    for seed in range(40):
        q = exam_question(topic, seed)
        assert [p.label for p in q.parts] == list("abcd"[: len(kinds)])
        assert [p.answer_kind for p in q.parts] == list(kinds)
        assert [p.marks for p in q.parts] == list(marks)
        assert [bool(p.follow_through) for p in q.parts] == list(follow_through)
        for p in q.parts:
            if p.answer_kind == "number":
                assert p.sensible_range is not None, (seed, p.label)


def assert_exam_graded(skill_id: str, topic: str, seeds=range(60)) -> None:
    """The composed key earns every mark, every distractor of every part earns nothing for its part, and every
    distractor is tagged."""
    sk = skill_object(skill_id)
    for seed in seeds:
        q = exam_question(topic, seed)
        canon = {p.label: p.answer_text for p in q.parts}
        full = grade(sk, q, canon)
        assert full.correct and full.marks == full.max_marks, (seed, q.answer_text)
        for p in q.parts:
            assert len(p.distractors) >= 2, (seed, p.label)
            for d in p.distractors:
                assert p.option_notes.get(d) in CATALOGUE, (seed, p.label, d)
                result = grade(sk, q, {**canon, p.label: d})
                assert result.parts[p.label].marks == 0, (seed, p.label, d, p.answer_text)
        junk = grade(sk, q, {p.label: f"{7919 * (i + 1)}.5" for i, p in enumerate(q.parts)})
        assert junk.marks == 0, (seed, junk.marks)


def assert_follow_through_holds(topic: str, seeds=range(60)) -> None:
    """Each part's follow-through formula, given the key of the earlier parts, gives the part's own key."""
    for seed in seeds:
        q = exam_question(topic, seed)
        values = {}
        for p in q.parts:
            value = part_expr(p)
            if p.follow_through:
                carried = sympy.sympify(p.follow_through, locals={"Abs": sympy.Abs, **values})
                assert (
                    sympy.simplify(carried - value) == 0
                    or math.isclose(float(sympy.N(carried - value, 20)), 0.0, abs_tol=1e-9)
                    or _same_function(carried, value)
                ), (seed, p.label, p.follow_through, p.answer_text)
            values[f"part_{p.label}"] = value


def _same_function(a, b) -> bool:
    fa, fb = numeric_function(a), numeric_function(b)
    pairs = [(_real(fa, p), _real(fb, p)) for p in (0.37, 1.41, 2.63, 4.2)]
    pairs = [(u, v) for u, v in pairs if u is not None and v is not None]
    return len(pairs) >= 2 and all(math.isclose(u, v, rel_tol=1e-9, abs_tol=1e-9) for u, v in pairs)


def part_marks(skill_id: str, q, replies: dict[str, str], label: str) -> int:
    """The marks the reply `replies[label]` earns for its part when the other replies are as given."""
    return grade(skill_object(skill_id), q, replies).parts[label].marks


def gradient_chain(stem: str):
    """(gradient function with the unknown constant, x0, gradient at x0, point (px, py)) read from a curve-through-a-point
    exam stem such as "f'(x) = a e^(3x) + 4 ... gradient ... at x = 0 is 12 ... passes through the point (0, -5)"."""
    fn = read_math(re.search(r"f'\(x\) = (.*?)(?:, where| \.|\. )", stem).group(1).rstrip("."))
    x0, g = re.search(r"gradient of the curve at x = (-?\d+) is (-?[\d/]+)", stem).groups()
    px, py = re.search(r"passes through the point \((-?\d+), (-?\d+)\)", stem).groups()
    return fn, sympy.Rational(x0), sympy.Rational(g), (sympy.Rational(px), sympy.Rational(py))


def solve_unknown(fn, x0, g):
    """The value of the constant (a or k) for which the gradient function takes the value g at x0."""
    unknown = next(s for s in fn.free_symbols if s != X)
    (value,) = sympy.solve(sympy.Eq(fn.subs(X, x0), g), unknown)
    return unknown, value


def part_expr(part):
    """The key of an expression or fraction part as a SymPy expression."""
    if isinstance(part.answer, str):
        return sympy.sympify(part.answer, locals={"Abs": sympy.Abs})
    return sympy.Rational(part.answer_text)  # a fraction part stores its key as a tagged value


def part_value(part) -> float:
    return float(sympy.N(part_expr(part), 25))
