"""Shared checks for the `test_fy1so_*` files: the first-year series and ODE skills.

Each skill's test file calls these with its own topics, then adds an oracle: a re-derivation of every answer from the
question's own text, never importing the generator module.
"""

from __future__ import annotations

import re
from fractions import Fraction

import sympy
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from mathgen import make
from mathgen.core import round_half_up, stated_precision_string
from mathgen.grading import grade
from mathgen.grading.parts import named_parts
from mathgen.misconceptions import CATALOGUE

SEEDS = range(60)
TIERS = ("easy", "medium", "hard")

_SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
_TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)


def parse_text(text: str, symbols: dict | None = None):
    """A learner-facing formula (``e^(-2x) sin(3x)``, ``√(1 + 2x)``, ``x² e^(-x)``) as a SymPy expression."""
    s = text.strip()
    s = re.sub(r"([²³⁴⁵⁶⁷⁸⁹])", lambda m: "^" + m.group(1).translate(_SUPERSCRIPTS), s)
    s = re.sub(r"√(\w+)", r" sqrt(\1)", s)
    s = s.replace("√", " sqrt").replace("·", "*").replace("−", "-")
    s = re.sub(r"\bln\b", "log", s)
    s = re.sub(r"(?<![A-Za-z])e\^", "E^", s)
    names = {"E": sympy.E, "pi": sympy.pi, "x": sympy.Symbol("x"), "y": sympy.Symbol("y")}
    names.update(symbols or {})
    return parse_expr(s, local_dict=names, transformations=_TRANSFORMS)


def reply_for(part, value) -> str:
    """`value` (a SymPy number or expression) as a reply to `part`, written the way the part states its own answer."""
    if part.answer_kind == "fraction":
        value = value if value.is_Rational else sympy.nsimplify(value, rational=True)
        return f"{value.p}/{value.q}" if value.q != 1 else str(value.p)
    if part.answer_kind == "number":
        literals = re.findall(r"-?\d+(?:\.\d+)?", part.answer_text)
        places = len(literals[0].split(".")[1]) if "." in literals[0] else 0
        exact = Fraction(str(sympy.N(value, 40)))
        rounded = round_half_up(exact, places) if exact >= 0 else -round_half_up(-exact, places)
        return stated_precision_string(rounded, places) if places else str(int(rounded))
    return sympy.sstr(value)


def check_single(topic: str, difficulty: str, kind: str, seeds=SEEDS, floor: int = 20) -> set[str]:
    """The canonical answer grades correct, every distractor grades wrong and is tagged; returns the distinct prompts."""
    prompts = set()
    for seed in seeds:
        q = make(topic, seed=seed, difficulty=difficulty)
        assert q.answer_kind == kind, (topic, difficulty, seed, q.answer_kind)
        assert grade(kind, q, q.answer_text).correct, (topic, difficulty, seed, q.answer_text)
        assert len(q.distractors) >= 2, (topic, difficulty, seed, q.distractors)
        assert len(set(q.distractors)) == len(q.distractors) and q.answer_text not in q.distractors
        for d in q.distractors:
            assert not grade(kind, q, d).correct, (topic, difficulty, seed, "distractor graded right", d)
            assert q.option_notes.get(d) in CATALOGUE, (topic, difficulty, seed, d, q.option_notes.get(d))
            value = q.distractor_values.get(d)
            if kind == "expression" and value is not None:
                assert not grade(kind, q, str(value)).correct, (topic, difficulty, seed, "value graded right", d)
        if kind == "number":
            places = {len(o.split(".")[1]) if "." in o else 0 for o in [q.answer_text, *q.distractors]}
            assert len(places) == 1, (topic, difficulty, seed, "decimal places differ", q.answer_text, q.distractors)
        prompts.add(q.text)
    assert len(prompts) >= floor, (topic, difficulty, len(prompts))
    return prompts


def check_tiers_differ(topic: str, tiers=TIERS, seeds=SEEDS) -> dict[str, set[str]]:
    """Each tier draws its own questions: no prompt appears in two tiers."""
    pools = {t: {make(topic, seed=s, difficulty=t).text for s in seeds} for t in tiers}
    names = list(pools)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            assert not pools[a] & pools[b], (topic, a, b, sorted(pools[a] & pools[b])[:2])
    return pools


def canonical_replies(q) -> dict[str, str]:
    return {p.label: p.answer_text for p in q.parts}


def check_chain(topic: str, kinds: list[str], marks: list[int], seeds=SEEDS, floor: int = 20) -> set[str]:
    """A multi-part exam chain: full marks for the canonical reply, no marks for any distractor of a part (and the other
    parts, answered canonically, keep theirs), a sensible shape, and follow-through that the engine awards."""
    prompts = set()
    skill_kind = "mcq"
    for seed in seeds:
        q = make(topic, seed=seed, difficulty="exam")
        assert [p.answer_kind for p in q.parts] == kinds, (seed, [p.answer_kind for p in q.parts])
        assert [p.marks for p in q.parts] == marks, (seed, [p.marks for p in q.parts])
        assert 3 <= len(q.parts) <= 4
        assert q.parts[0].follow_through is None
        assert any(p.follow_through for p in q.parts)
        full = grade(skill_kind, q, q.answer_text)
        assert full.correct and full.marks == full.max_marks == sum(marks), (seed, q.answer_text)
        assert grade(skill_kind, q, canonical_replies(q)).correct
        for part in q.parts:
            assert len(part.distractors) >= 2, (seed, part.label)
            for d in part.distractors:
                assert d != part.answer_text
                assert part.option_notes.get(d) in CATALOGUE, (seed, part.label, d)
                result = grade(skill_kind, q, {**canonical_replies(q), part.label: d})
                assert result.parts[part.label].marks == 0, (seed, part.label, d)
                others = [p for p in q.parts if p.label != part.label]
                assert all(result.parts[p.label].marks == p.marks for p in others), (seed, part.label, d)
        prompts.add(q.text)
    assert len(prompts) >= floor, (topic, len(prompts))
    return prompts


def follow_through_value(part, replies: dict[str, str], q):
    """Part `part`'s follow-through answer, worked out with SymPy from the replies to the earlier parts it names; None
    when a reply does not read as a single value."""
    values = {}
    for label in named_parts(part):
        source = next(p for p in q.parts if p.label == label)
        text = replies[label]
        try:
            if source.answer_kind == "expression":
                values[f"part_{label}"] = parse_text(text)
            else:
                values[f"part_{label}"] = sympy.Rational(
                    Fraction(text.replace(" ", "")).numerator, Fraction(text.replace(" ", "")).denominator
                )
        except (ValueError, ZeroDivisionError, sympy.SympifyError, SyntaxError, TypeError):
            return None
    expr = sympy.sympify(part.follow_through, locals={**values, "x": sympy.Symbol("x"), "y": sympy.Symbol("y")})
    return sympy.N(expr.doit(), 40) if expr.free_symbols == set() else expr.doit()


def check_follow_through(topic: str, seeds=range(30), minimum: int = 15) -> int:
    """For every part that follows through, a wrong earlier answer (one of its distractors, carried forward with an
    independent SymPy evaluation of the stated expression) earns the later part its marks, with `follow_through` set."""
    checked = tried = 0
    for seed in seeds:
        q = make(topic, seed=seed, difficulty="exam")
        for part in q.parts:
            if not part.follow_through:
                continue
            for label in named_parts(part):
                source = next(p for p in q.parts if p.label == label)
                if source.answer_kind not in ("fraction", "number", "expression") or not source.distractors:
                    continue
                wrong = source.distractors[0]
                replies = {**canonical_replies(q), label: wrong}
                target = follow_through_value(part, replies, q)
                if target is None or isinstance(target, sympy.Basic) and not target.is_number:
                    continue
                if not target.is_real:
                    continue
                if reply_for(part, target) == part.answer_text:
                    continue
                replies[part.label] = reply_for(part, target)
                tried += 1
                result = grade("mcq", q, replies)
                if result.parts[part.label].marks == part.marks:
                    assert result.parts[part.label].follow_through is True, (seed, part.label, label)
                    checked += 1
    assert checked >= minimum and checked >= 0.7 * tried, f"{topic}: {checked} of {tried} follow-through cases awarded"
    return checked
