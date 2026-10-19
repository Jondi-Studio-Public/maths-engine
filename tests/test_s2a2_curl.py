"""Tests `uc2-curl-of-field`: the topic `s2a2_curl` (easy, medium, hard) and the exam chain `exd_x_s2a2_curl`.

The oracle reads only the question text. It parses the field out of the stem and re-derives every key with SymPy:
`diff` for the curl, `solve` for the unknown constants, direct substitution for the point values.
"""

from __future__ import annotations

import re

import pytest
import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

import mathgen.generators.s2a2_curl_build  # noqa: F401  (registers the topics)
from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects

SKILL_ID = "uc2-curl-of-field"
TOPIC = "s2a2_curl"
EXAM_TOPIC = "exd_x_s2a2_curl"
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)
x, y, z, a, b = sympy.symbols("x y z a b")
TIERS = ("easy", "medium", "hard")


def read(text: str):
    s = re.sub(r"e\^\(", "exp(", text)
    s = s.replace("²", "**2").replace("³", "**3").replace("^", "**").replace("ln(", "log(")
    return parse_expr(s, transformations=_TRANSFORMS, local_dict={"x": x, "y": y, "z": z, "a": a, "b": b})


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


def vector_after(text: str, pattern: str):
    match = re.search(pattern + r" ?= ?⟨(.*?)⟩", text)
    return tuple(read(c) for c in split_top(match.group(1)))


def field_in(text: str):
    return vector_after(text, r"[A-Za-z]\(x, y, z\)")


def point_in(text: str):
    match = re.search(r"point (?:P)?\((-?\d+), (-?\d+), (-?\d+)\)", text)
    return tuple(int(c) for c in match.groups()) if match else None


def curl(F):
    P, Q, R = F
    return (
        sympy.diff(R, y) - sympy.diff(Q, z),
        sympy.diff(P, z) - sympy.diff(R, x),
        sympy.diff(Q, x) - sympy.diff(P, y),
    )


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


def _exam(seed):
    return make(EXAM_TOPIC, seed=seed, difficulty="exam")


def oracle(q):
    text = q.text
    F = field_in(text)
    pt = point_in(text)
    at = dict(zip((x, y, z), pt)) if pt else None
    c = curl(F)
    if "Find a and b" in text:
        target = vector_after(text, r"∇ × [A-Za-z]")  # the curl value quoted after "At the point"
        eqs = [e.subs(at) - t for e, t in zip(c, target)]
        sol = sympy.solve(eqs, [a, b], dict=True)
        assert len(sol) == 1
        return (sol[0][a], sol[0][b])
    return tuple(e.subs(at) for e in c) if at else c


def _same(answer_text: str, expected) -> bool:
    parts = split_top(answer_text.strip()[1:-1])
    return len(parts) == len(expected) and all(sympy.simplify(read(p) - e) == 0 for p, e in zip(parts, expected))


@pytest.mark.parametrize("tier", TIERS)
def test_key_matches_independent_sympy_derivation(tier):
    for seed in range(30):
        q = _q(tier, seed)
        assert _same(q.answer_text, oracle(q)), (seed, q.text, q.answer_text, oracle(q))


def test_source_wired_last():
    last = _SKILL.sources[-1]
    assert (last.topic, last.difficulty, last.stages) == (EXAM_TOPIC, "exam", ["exam"])
    assert [s.difficulty for s in _SKILL.sources] == ["easy", "medium", "hard", "exam"]


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_and_every_distractor_rejected(tier):
    for seed in range(30):
        q = _q(tier, seed)
        assert grade(_SKILL, q, q.answer_text).correct, (seed, q.text, q.answer_text)
        assert len(q.distractors) >= 3
        for text in q.distractors:
            assert not grade(_SKILL, q, text).correct, (seed, q.text, text)


@pytest.mark.parametrize("tier", TIERS)
def test_distractors_are_tagged_and_distinct(tier):
    for seed in range(30):
        q = _q(tier, seed)
        assert len(q.option_notes) == len(q.distractors) >= 3
        assert len(set(q.distractors)) == len(q.distractors)
        assert q.answer_text not in q.distractors


def test_a_reply_in_another_order_is_not_accepted():
    for seed in range(20):
        q = _q("easy", seed)
        i, j, k = split_top(q.answer_text.strip()[1:-1])
        if len({i, j, k}) == 3:
            assert not grade(_SKILL, q, f"({j}, {i}, {k})").correct


@pytest.mark.parametrize("tier", TIERS)
def test_variety(tier):
    answers = [_q(tier, seed).answer_text for seed in range(60)]
    assert len(set(answers)) >= 25
    assert max(answers.count(a_) for a_ in set(answers)) <= 8


def test_tiers_escalate_in_reasoning_not_range():
    easy = [_q("easy", s) for s in range(30)]
    assert all("point" in q.text and not any(f in q.text for f in ("sin", "cos", "e^")) for q in easy)
    medium = [_q("medium", s) for s in range(30)]
    assert all(any(f in q.text for f in ("sin", "cos", "e^")) and "point" not in q.text for q in medium)
    hard = [_q("hard", s).text for s in range(90)]
    kinds = {
        "unknowns": sum("Find a and b" in t for t in hard),
        "exact": sum("exact value" in t for t in hard),
        "rotation": sum("Find the exact" not in t and "Find a and b" not in t for t in hard),
    }
    assert all(v >= 10 for v in kinds.values()), kinds


def test_rotation_questions_have_a_constant_curl_twice_the_axis():
    seen = 0
    for seed in range(90):
        q = _q("hard", seed)
        if "Find a and b" in q.text or "exact value" in q.text:
            continue
        c = [sympy.simplify(e) for e in curl(field_in(q.text))]
        assert all(e.is_number and e % 2 == 0 for e in c), (seed, c)
        seen += 1
    assert seen >= 10


@pytest.mark.parametrize("tier", TIERS)
def test_text_quality_and_determinism(tier):
    for seed in range(40):
        q = _q(tier, seed)
        assert find_text_defects(q) == [], (seed, find_text_defects(q))
        assert q.to_json() == _q(tier, seed).to_json()


# ---------------------------------------------------------------------------------------------------------------
# the exam chain


def _facts(q):
    F = field_in(q.stem)
    c = [sympy.expand(e) for e in curl(F)]
    pt = point_in(q.parts[3].prompt)
    at = dict(zip((x, y, z), pt))
    return dict(F=F, curl=c, pt=pt, vals=[e.subs(at) for e in c], mag=sympy.sqrt(sum(e.subs(at) ** 2 for e in c)))


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def test_exam_shape_and_marks():
    for seed in range(40):
        q = _exam(seed)
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.answer_kind for p in q.parts] == ["expression"] * 4
        assert [p.marks for p in q.parts] == [2, 2, 2, 3]
        assert [p.follow_through is not None for p in q.parts] == [False, False, False, True]
        assert all(len(p.distractors) >= 2 for p in q.parts)
        assert find_text_defects(q) == []


def test_exam_keys_match_independent_derivation():
    for seed in range(40):
        q = _exam(seed)
        f = _facts(q)
        parts = _canonical(q)
        for label, expected in zip("abc", f["curl"]):
            assert sympy.simplify(read(parts[label]) - expected) == 0, (seed, label)
        assert sympy.simplify(read(parts["d"]) - f["mag"]) == 0
        assert not f["mag"].is_Integer


def test_exam_key_accepted_every_distractor_rejected():
    for seed in range(25):
        q = _exam(seed)
        full = grade(_SKILL, q, q.answer_text)
        assert full.correct and full.marks == full.max_marks == sum(p.marks for p in q.parts)
        canon = _canonical(q)
        for p in q.parts:
            for text in p.distractors:
                assert grade(_SKILL, q, {**canon, p.label: text}).parts[p.label].marks == 0, (seed, p.label, text)


def test_follow_through_into_the_magnitude_from_wrong_components():
    checked = 0
    for seed in range(40):
        q = _exam(seed)
        f = _facts(q)
        canon = _canonical(q)
        at = dict(zip((x, y, z), f["pt"]))
        for index, label in enumerate("abc"):
            for wrong in q.parts[index].distractors:
                w = read(wrong).subs(at)
                if w == 0 or w == f["vals"][index]:
                    continue
                vals = list(f["vals"])
                vals[index] = w
                mag = sympy.sqrt(sum(v * v for v in vals))
                if abs(float(mag) - float(f["mag"])) < 1e-9:
                    continue  # a sign slip leaves the magnitude unchanged
                reply = {**canon, label: wrong, "d": str(mag).replace("**", "^")}
                result = grade(_SKILL, q, reply)
                assert result.parts[label].marks == 0
                assert result.parts["d"].marks == 3 and result.parts["d"].follow_through is True, (seed, label, wrong)
                checked += 1
    assert checked >= 60


def test_exam_answer_space_and_determinism():
    answers = [_exam(seed).answer_text for seed in range(150)]
    assert len(set(answers)) >= 120
    assert _exam(7).to_json() == _exam(7).to_json()
