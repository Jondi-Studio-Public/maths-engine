"""Shared checks for the `uc2-*` linear algebra skills built by the `s2b2_` generators.

Each skill's test file adds an oracle that reads the question TEXT back with `parse_matrix` and re-derives the key with
SymPy; the checks here are the ones every skill shares: wiring, key accepted, every distractor rejected and tagged,
text quality, determinism, and the shape, marks and follow-through of the exam chain.
"""

from __future__ import annotations

import ast
import re

import sympy
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    parse_expr,
    split_symbols,
    standard_transformations,
)

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects

TIERS = ("easy", "medium", "hard")
_TRANSFORMS = standard_transformations + (split_symbols, implicit_multiplication_application)
_MATRIX_RE = re.compile(r"\[\[[^\]]*\](?:, \[[^\]]*\])*\]")
_SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")


def load_skill(skill_id):
    return next(s for s in load_skills()[1] if s.id == skill_id)


def parse(text, **local):
    """A SymPy expression from text such as `-4x**2 + 2xy` (implicit multiplication allowed)."""
    return parse_expr(text.replace("^", "**"), local_dict=local, transformations=_TRANSFORMS)


def matrices(text, symbols=None):
    """Every `[[...], ...]` literal in `text`, as SymPy matrices (entries may contain the symbols named in `symbols`)."""
    local = {name: sympy.Symbol(name) for name in (symbols or "")}
    return [
        sympy.Matrix(
            [[sympy.nsimplify(parse(c, **local), rational=True) for c in row] for row in ast.literal_eval(_quote(lit))]
        )
        for lit in _MATRIX_RE.findall(text)
    ]


def _quote(literal):
    """Quote each entry of a matrix literal so `ast` can read entries such as `k` or `-k` or `2k`."""
    return re.sub(r"(?<![\w'])(-?[\w.]+)(?=[,\]])", lambda m: repr(m.group(1)), literal)


def value(x):
    """A SymPy number from a stored answer element: an int, a decimal, a `{p, q}` fraction or an expression string."""
    if isinstance(x, dict):
        return sympy.Rational(x["p"], x["q"])
    if isinstance(x, str):
        return parse(x)
    return sympy.Rational(str(x))


def answer_matrix(answer):
    return sympy.Matrix([[value(v) for v in row] for row in answer])


def numerically_equal(a, b, tol=1e-9):
    return abs(complex(sympy.N(a - b))) < tol


# -- checks every skill shares ------------------------------------------------------------------------------------


def check_wiring(skill, topic, exam, answer, prerequisites):
    assert skill.answer == answer
    assert [(s.topic, s.difficulty) for s in skill.sources] == [
        (topic, "easy"),
        (topic, "medium"),
        (topic, "hard"),
        (exam, "exam"),
    ]
    assert skill.sources[-1].stages == ["exam"]
    assert list(skill.prerequisites) == prerequisites
    assert list(skill.year_level) == ["university"]
    assert "university-year-2" in skill.tags


def check_single_tier(skill, topic, tier, seeds=60):
    """Key accepted, every distractor rejected and tagged, no text defects, deterministic."""
    for seed in range(seeds):
        q = make(topic, seed=seed, difficulty=tier)
        assert grade(skill, q, q.answer_text).correct, (seed, q.answer_text)
        assert len(q.distractors) >= 2, seed
        for text in q.distractors:
            assert not grade(skill, q, text).correct, (seed, text, q.answer_text)
            assert q.option_notes[text] in CATALOGUE, (seed, text)
            assert text in q.distractor_values
        assert find_text_defects(q, include_working=True) == [], (seed, find_text_defects(q, include_working=True))
        assert q.to_json() == make(topic, seed=seed, difficulty=tier).to_json()


def distinct_questions(topic, tier, seeds=60):
    return len({make(topic, seed=s, difficulty=tier).text for s in range(seeds)})


def check_exam_shape(skill, exam, kinds, marks, follow_through_count, seeds=60):
    """Part kinds and marks as declared; the key scores full marks; each part's distractors score zero."""
    for seed in range(seeds):
        q = make(exam, seed=seed, difficulty="exam")
        assert [p.label for p in q.parts] == list("abcd"[: len(kinds)])
        assert [p.answer_kind for p in q.parts] == list(kinds)
        assert [p.marks for p in q.parts] == list(marks)
        assert sum(p.follow_through is not None for p in q.parts) == follow_through_count
        assert find_text_defects(q, include_working=True) == [], (seed, find_text_defects(q, include_working=True))
        assert q.to_json() == make(exam, seed=seed, difficulty="exam").to_json()
        canon = {p.label: p.answer_text for p in q.parts}
        full = grade(skill, q, canon)
        assert full.correct and full.marks == full.max_marks == sum(marks), (seed, canon)
        for p in q.parts:
            assert len(p.distractors) >= 2, (seed, p.label)
            for text in p.distractors:
                result = grade(skill, q, {**canon, p.label: text})
                assert result.parts[p.label].marks == 0, (seed, p.label, text, p.answer_text)
                assert p.option_notes[text] in CATALOGUE, (seed, p.label, text)


def check_follow_through_strings(exam, seeds=60):
    """Each part's `follow_through` string, evaluated on the earlier canonical answers, equals the part's own answer."""
    for seed in range(seeds):
        q = make(exam, seed=seed, difficulty="exam")
        known = {}
        for p in q.parts:
            if p.follow_through is not None:
                expr = parse(p.follow_through, **{k: v for k, v in known.items()})
                assert numerically_equal(expr, part_value(p), 1e-6), (seed, p.label, p.follow_through, p.answer_text)
            known["part_" + p.label] = part_value(p)


def part_value(p):
    if p.answer_kind == "matrix":
        return answer_matrix(p.answer)
    if p.answer_kind == "fraction":
        return value(p.answer)
    if p.answer_kind in ("number", "expression"):
        return value(p.answer)
    return sympy.Symbol("x")


def check_follow_through_marks(skill, exam, seeds=40, minimum_share=0.7):
    """A wrong earlier answer still earns the dependent part its marks when that part follows through from it.

    For each dependent part, a catalogued wrong answer is put in an earlier part it names and the part is answered
    with the value its `follow_through` gives. The marker may decline a follow-through that is degenerate (an answer
    of 0, or one far from the canonical size), so the test asks that most parts follow through, not every one.
    """
    attempted = succeeded = 0
    for seed in range(seeds):
        q = make(exam, seed=seed, difficulty="exam")
        canon = {p.label: p.answer_text for p in q.parts}
        known = {"part_" + p.label: part_value(p) for p in q.parts}
        for p in q.parts:
            if p.follow_through is None:
                continue
            attempted += 1
            done = False
            for name in [n for n in known if re.search(rf"\b{n}\b", p.follow_through)]:
                source = next(s for s in q.parts if "part_" + s.label == name)
                if source.answer_kind not in ("number", "fraction"):
                    continue
                for bad in source.distractors:
                    try:
                        bad_value = sympy.nsimplify(parse(bad), rational=True)
                        follow = sympy.nsimplify(parse(p.follow_through, **{**known, name: bad_value}), rational=True)
                    except Exception:  # noqa: BLE001  (a distractor that is not a plain number)
                        continue
                    if p.answer_kind == "number" and not follow.is_Integer:
                        continue
                    reply = {**canon, source.label: bad, p.label: _text(follow, p.answer_kind)}
                    result = grade(skill, q, reply)
                    assert result.parts[source.label].marks == 0
                    if result.parts[p.label].marks == p.marks:
                        assert result.parts[p.label].follow_through is True, (seed, p.label, reply)
                        done = True
                        break
                if done:
                    break
            succeeded += done
    assert attempted >= seeds // 2, attempted
    assert succeeded >= minimum_share * attempted, (succeeded, attempted)


def _text(expr, kind):
    expr = sympy.nsimplify(expr)
    if kind == "expression":
        return str(expr).replace("**", "^")
    return str(expr)
