"""Tests `uc2-surface-normal-parametrised`: the topic `s2a2_surface_normal` (easy, medium, hard) and the exam chain
`exd_x_s2a2_surface_normal`.

The oracle reads only the question text. It parses r(u, v) out of the stem and re-derives every key with SymPy:
`diff` for the tangent vectors, `Matrix.cross` for the normal, substitution for the point values.
"""

from __future__ import annotations

import re

import pytest
import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

import mathgen.generators.s2a2_surface_normal_build  # noqa: F401  (registers the topics)
from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects

SKILL_ID = "uc2-surface-normal-parametrised"
TOPIC = "s2a2_surface_normal"
EXAM_TOPIC = "exd_x_s2a2_surface_normal"
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)
u, v = sympy.symbols("u v")
TIERS = ("easy", "medium", "hard")


def read(text: str):
    s = text.replace("²", "**2").replace("³", "**3").replace("^", "**")
    return parse_expr(s, transformations=_TRANSFORMS, local_dict={"u": u, "v": v})


def split_top(s: str):
    parts, depth, cur = [], 0, ""
    for ch in s:
        depth += ch == "("
        depth -= ch == ")"
        if ch == "," and depth == 0:
            parts.append(cur.strip())
            cur = ""
        else:
            cur += ch
    parts.append(cur.strip())
    return parts


def surface_in(text: str):
    match = re.search(r"r\(u, v\) = ⟨(.*?)⟩", text)
    return sympy.Matrix([read(c) for c in split_top(match.group(1))])


def normal(r):
    return sympy.Matrix(r.diff(u)).cross(sympy.Matrix(r.diff(v))).applyfunc(sympy.expand)


def at_point(text):
    match = re.search(r"u = (-?\d+) and v = (-?\d+)", text)
    return {u: int(match.group(1)), v: int(match.group(2))}


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


def _exam(seed):
    return make(EXAM_TOPIC, seed=seed, difficulty="exam")


def oracle(q):
    r = surface_in(q.text)
    N = normal(r)
    if "unit normal" in q.text:
        n = N.subs(at_point(q.text))
        return tuple(c / sympy.sqrt(sum(d * d for d in n)) for c in n)
    if "Find d" in q.text:
        point = at_point(q.text)
        return N.subs(point).dot(r.subs(point))
    return tuple(N)


def _same(answer_text: str, expected) -> bool:
    if isinstance(expected, tuple):
        parts = split_top(answer_text.strip()[1:-1])
        return len(parts) == len(expected) and all(sympy.simplify(read(p) - e) == 0 for p, e in zip(parts, expected))
    return sympy.simplify(read(answer_text) - expected) == 0


@pytest.mark.parametrize("tier", TIERS)
def test_key_matches_independent_sympy_derivation(tier):
    for seed in range(40):
        q = _q(tier, seed)
        assert _same(q.answer_text, oracle(q)), (seed, q.text, q.answer_text, oracle(q))


def test_source_wired_last():
    last = _SKILL.sources[-1]
    assert (last.topic, last.difficulty, last.stages) == (EXAM_TOPIC, "exam", ["exam"])
    assert [s.difficulty for s in _SKILL.sources] == ["easy", "medium", "hard", "exam"]


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_and_every_distractor_rejected(tier):
    for seed in range(40):
        q = _q(tier, seed)
        assert grade(_SKILL, q, q.answer_text).correct, (seed, q.text, q.answer_text)
        assert len(q.distractors) >= 3
        for text in q.distractors:
            assert not grade(_SKILL, q, text).correct, (seed, q.text, text)


@pytest.mark.parametrize("tier", TIERS)
def test_distractors_are_tagged_and_distinct(tier):
    for seed in range(40):
        q = _q(tier, seed)
        assert len(q.option_notes) == len(q.distractors) >= 3
        assert len(set(q.distractors)) == len(q.distractors)
        assert q.answer_text not in q.distractors


def test_a_reply_in_another_spelling_is_accepted():
    for seed in range(30):
        q = _q("easy", seed)
        parts = split_top(q.answer_text.strip()[1:-1])
        expanded = "(" + ", ".join(str(sympy.expand(read(p))).replace("**", "^") for p in parts) + ")"
        assert grade(_SKILL, q, expanded).correct, (seed, expanded)


@pytest.mark.parametrize("tier", TIERS)
def test_variety(tier):
    answers = [_q(tier, seed).answer_text for seed in range(60)]
    assert len(set(answers)) >= 25
    assert max(answers.count(a) for a in set(answers)) <= 8


def test_tiers_escalate_in_reasoning_not_range():
    easy = [_q("easy", s) for s in range(30)]
    assert all(not any(f in q.text for f in ("sin", "cos")) for q in easy)
    assert all("point" not in q.text and "plane" not in q.text for q in easy)
    medium = [_q("medium", s) for s in range(30)]
    assert all(any(f in q.text for f in ("sin", "cos")) for q in medium)
    hard = [_q("hard", s).text for s in range(90)]
    kinds = {
        "unit": sum("unit normal" in t for t in hard),
        "plane": sum("Find d" in t for t in hard),
        "revolution": sum(("sin(v)" in t or "cos(v)" in t) for t in hard),
    }
    assert all(n >= 10 for n in kinds.values()), kinds
    assert sum(kinds.values()) == 90


def test_unit_normals_have_unit_length_and_whole_number_lengths():
    seen = 0
    for seed in range(90):
        q = _q("hard", seed)
        if "unit normal" not in q.text:
            continue
        r = surface_in(q.text)
        n = normal(r).subs(at_point(q.text))
        assert sympy.sqrt(sum(c * c for c in n)).is_Integer
        parts = [read(p) for p in split_top(q.answer_text.strip()[1:-1])]
        assert sympy.simplify(sum(p * p for p in parts) - 1) == 0
        seen += 1
    assert seen >= 15


@pytest.mark.parametrize("tier", TIERS)
def test_text_quality_and_determinism(tier):
    for seed in range(60):
        q = _q(tier, seed)
        assert find_text_defects(q) == [], (seed, find_text_defects(q))
        assert q.to_json() == _q(tier, seed).to_json()


# ---------------------------------------------------------------------------------------------------------------
# the exam chain


def _facts(q):
    r = surface_in(q.stem)
    N = normal(r)
    point = at_point(q.parts[1].prompt)
    n0 = N.subs(point)
    r0 = r.subs(point)
    d0 = n0.dot(r0)
    return dict(r=r, N=N, point=point, n0=n0, r0=r0, d0=d0, intercept=d0 / n0[2])


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def test_exam_shape_and_marks():
    for seed in range(40):
        q = _exam(seed)
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.answer_kind for p in q.parts] == ["expression", "fraction", "expression", "fraction"]
        assert [p.marks for p in q.parts] == [2, 1, 2, 3]
        assert [p.follow_through is not None for p in q.parts] == [False, True, False, True]
        assert all(len(p.distractors) >= 2 for p in q.parts)
        assert find_text_defects(q) == []


def test_exam_keys_match_independent_derivation():
    for seed in range(40):
        q = _exam(seed)
        f = _facts(q)
        parts = _canonical(q)
        assert sympy.simplify(read(parts["a"]) - f["N"][2]) == 0, (seed, q.stem)
        assert sympy.Rational(parts["b"]) == f["n0"][2]
        assert _same(parts["c"], tuple(f["n0"]))
        assert sympy.Rational(parts["d"]) == f["intercept"]
        # the tangent plane really passes through the point of the surface and the axis point
        z = f["intercept"]
        assert f["n0"].dot(sympy.Matrix([0, 0, z]) - f["r0"]) == 0


def test_exam_key_accepted_every_distractor_rejected():
    for seed in range(30):
        q = _exam(seed)
        full = grade(_SKILL, q, q.answer_text)
        assert full.correct and full.marks == full.max_marks == sum(p.marks for p in q.parts)
        canon = _canonical(q)
        for p in q.parts:
            for text in p.distractors:
                assert grade(_SKILL, q, {**canon, p.label: text}).parts[p.label].marks == 0, (seed, p.label, text)


def test_follow_through_from_a_wrong_z_component_into_its_value():
    checked = 0
    for seed in range(40):
        q = _exam(seed)
        f = _facts(q)
        canon = _canonical(q)
        for wrong in q.parts[0].distractors:
            wb = read(wrong).subs(f["point"])
            if wb in (0, f["n0"][2]) or not wb.is_Integer or not read(wrong).free_symbols:
                continue  # writing a constant again at the point is not following through
            result = grade(_SKILL, q, {**canon, "a": wrong, "b": str(wb)})
            assert result.parts["a"].marks == 0
            assert result.parts["b"].marks == 1 and result.parts["b"].follow_through is True, (seed, wrong, wb)
            checked += 1
    assert checked >= 30


def test_follow_through_from_a_wrong_value_into_the_intercept():
    checked = 0
    for seed in range(40):
        q = _exam(seed)
        f = _facts(q)
        canon = _canonical(q)
        for wrong in q.parts[1].distractors:
            wb = sympy.Rational(wrong)
            if wb in (0, f["n0"][2]) or f["d0"] / wb == f["intercept"]:
                continue
            reply = {**canon, "b": wrong, "d": str(f["d0"] / wb)}
            result = grade(_SKILL, q, reply)
            assert result.parts["b"].marks == 0
            assert result.parts["d"].marks == 3 and result.parts["d"].follow_through is True, (seed, wrong)
            checked += 1
    assert checked >= 60


def test_exam_answer_space_and_determinism():
    answers = [_exam(seed).answer_text for seed in range(200)]
    assert len(set(answers)) >= 150
    assert _exam(7).to_json() == _exam(7).to_json()
