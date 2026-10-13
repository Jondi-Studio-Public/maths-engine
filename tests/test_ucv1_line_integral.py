"""Tests `ucv1-line-integral`: the topic `line_integral` (easy, medium, hard) and the exam chain
`exd_x_uc_line_integral`.

The oracle reads only the question text. It parses the function, the field, the segment or the curve back out of
the stem and re-derives every key with SymPy: `integrate` along the parametrised curve for the work (never the
generator's potential), `solve` for k, and `integrate` plus a matching step for a potential.
"""

from __future__ import annotations

import re

import pytest
import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

import mathgen.generators.ucv1_line_integral_build  # noqa: F401  (registers the topics)
from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects

SKILL_ID = "ucv1-line-integral"
TOPIC = "line_integral"
EXAM_TOPIC = "exd_x_uc_line_integral"
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)
x, y, t, k = sympy.symbols("x y t k")
TIERS = ("easy", "medium", "hard")
_VEC = r"(\([^()]*\)|[^\s()]+) i \+ (\([^()]*\)|[^\s()]+) j"


def _read(text: str):
    text = text.replace("²", "**2").replace("³", "**3").replace("^", "**")
    text = re.sub(r"√(\d+)", r"sqrt(\1)", text)
    return parse_expr(text, transformations=_TRANSFORMS, local_dict={"x": x, "y": y, "t": t, "k": k})


def _unbracket(s: str) -> str:
    return s[1:-1] if s.startswith("(") and s.endswith(")") else s


def _vector_after(text: str, marker: str):
    start = text.index(marker) + len(marker)
    match = re.match(_VEC, text[start:])
    assert match, text[start : start + 80]
    return _read(_unbracket(match.group(1))), _read(_unbracket(match.group(2)))


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


def _exam(seed):
    return make(EXAM_TOPIC, seed=seed, difficulty="exam")


def _points(text: str):
    return [(int(a), int(b)) for a, b in re.findall(r"\((-?\d+), (-?\d+)\)", text)]


def _work_along(P, Q, xt, yt, lo, hi):
    integrand = P.subs({x: xt, y: yt}, simultaneous=True) * sympy.diff(xt, t) + Q.subs(
        {x: xt, y: yt}, simultaneous=True
    ) * sympy.diff(yt, t)
    return sympy.integrate(sympy.expand(integrand), (t, lo, hi))


def _solve_k(P, Q):
    return sympy.solve(sympy.expand(sympy.diff(P, y) - sympy.diff(Q, x)), k)[0] if P.has(k) else None


def _potential(P, Q):
    phi = sympy.integrate(P, x)
    phi += sympy.integrate(sympy.expand(Q - sympy.diff(phi, y)), y)
    return sympy.expand(phi)


def oracle_easy(q):
    f_text = re.search(r"f\(x, y\) = (.+?)(?: and C is| over C)", q.text) or re.search(r"∫C \((.+?)\) ds", q.text)
    f = _read(f_text.group(1))
    (x1, y1), (x2, y2) = _points(q.text)[-2:]
    xt, yt = x1 + (x2 - x1) * t, y1 + (y2 - y1) * t
    speed = sympy.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return sympy.integrate(sympy.expand(f.subs({x: xt, y: yt}, simultaneous=True)) * speed, (t, 0, 1))


def _curve_range(text: str):
    lo, hi = re.search(r"(-?\d+) ≤ t ≤ (-?\d+)", text).groups()
    return int(lo), int(hi)


def oracle_medium(q):
    P, Q = _vector_after(q.text, "F(x, y) = ")
    xt, yt = _vector_after(q.text, "r(t) = ")
    lo, hi = _curve_range(q.text)
    return _work_along(P, Q, xt, yt, lo, hi)


def oracle_hard(q):
    P, Q = _vector_after(q.text, "F(x, y) = ")
    if "r(t) = " in q.text:
        xt, yt = _vector_after(q.text, "r(t) = ")
        return _work_along(P, Q, xt, yt, 0, 1)
    kv = _solve_k(P, Q)
    assert kv is not None
    P, Q = P.subs(k, kv), Q.subs(k, kv)
    (x1, y1), (x2, y2) = _points(q.text)[-2:]
    xt, yt = x1 + (x2 - x1) * t, y1 + (y2 - y1) * t
    return _work_along(P, Q, xt, yt, 0, 1)  # any path will do for a conservative field


ORACLES = {"easy": oracle_easy, "medium": oracle_medium, "hard": oracle_hard}


def _same(text: str, value) -> bool:
    return sympy.simplify(_read(text) - sympy.nsimplify(value)) == 0


@pytest.mark.parametrize("tier", TIERS)
def test_key_matches_independent_sympy_derivation(tier):
    for seed in range(40):
        q = _q(tier, seed)
        expected = ORACLES[tier](q)
        assert _same(q.answer_text, expected), (seed, q.text, q.answer_text, expected)


def test_source_wired_last():
    last = _SKILL.sources[-1]
    assert (last.topic, last.difficulty, last.stages, last.params) == (EXAM_TOPIC, "exam", ["exam"], {"tier": "exam"})
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
        assert len(q.distractor_values) >= 3
        assert len(set(q.distractors)) == len(q.distractors)
        assert q.answer_text not in q.distractors


def test_easy_surd_answers_are_accepted_in_other_spellings():
    checked = 0
    for seed in range(60):
        q = _q("easy", seed)
        if "√" not in q.answer_text:
            continue
        assert grade(_SKILL, q, q.answer_text).correct
        machine = str(_read(q.answer_text)).replace("**", "^")
        assert grade(_SKILL, q, machine).correct, (seed, machine, q.answer_text)
        checked += 1
    assert checked >= 30


@pytest.mark.parametrize("tier", TIERS)
def test_variety(tier):
    answers = [_q(tier, seed).answer_text for seed in range(60)]
    assert len(set(answers)) >= 30
    assert max(answers.count(a) for a in set(answers)) <= 6


def test_tiers_escalate_in_reasoning_not_range():
    easy = [_q("easy", s).text for s in range(30)]
    assert all("ds" in text or "scalar line integral" in text for text in easy)
    medium = [_q("medium", s).text for s in range(30)]
    assert all("r(t) = " in text for text in medium)
    hard = [_q("hard", s) for s in range(60)]
    with_k = [q for q in hard if " k" in q.text or "ky" in q.text]
    curved = [q for q in hard if "r(t) = " in q.text]
    assert len(with_k) >= 15 and len(curved) >= 15 and len(with_k) + len(curved) == 60


def test_hard_fields_are_conservative_once_k_is_found():
    for seed in range(40):
        q = _q("hard", seed)
        P, Q = _vector_after(q.text, "F(x, y) = ")
        kv = _solve_k(P, Q)
        if kv is not None:
            P, Q = P.subs(k, kv), Q.subs(k, kv)
        assert sympy.expand(sympy.diff(P, y) - sympy.diff(Q, x)) == 0, (seed, q.text)


def test_medium_fields_are_not_conservative_so_the_path_matters():
    for seed in range(40):
        P, Q = _vector_after(_q("medium", seed).text, "F(x, y) = ")
        assert sympy.expand(sympy.diff(P, y) - sympy.diff(Q, x)) != 0


@pytest.mark.parametrize("tier", TIERS)
def test_text_quality_and_determinism(tier):
    for seed in range(60):
        q = _q(tier, seed)
        assert find_text_defects(q) == [], (seed, find_text_defects(q))
        assert q.to_json() == _q(tier, seed).to_json()


# ---------------------------------------------------------------------------------------------------------------
# the exam chain


def _facts(q) -> dict:
    """What the stem and prompts give: the field, k (or None), the potential, and the work along both routes."""
    P, Q = _vector_after(q.stem, "F(x, y) = ")
    kv = _solve_k(P, Q)
    P1, Q1 = (P.subs(k, kv), Q.subs(k, kv)) if kv is not None else (P, Q)
    phi = _potential(P1, Q1)
    offset = 1 if kv is not None else 0
    b_prompt, c_prompt = q.parts[1 + offset].prompt, q.parts[2 + offset].prompt
    (ax, ay), (bx, by) = _points(b_prompt)
    xt, yt = _vector_after(c_prompt, "r(t) = ")
    return dict(
        P=P1,
        Q=Q1,
        k=kv,
        phi=phi,
        work_ab=_work_along(P1, Q1, ax + (bx - ax) * t, ay + (by - ay) * t, 0, 1),
        work_c=_work_along(P1, Q1, xt, yt, 0, 1),
        offset=offset,
        A=(ax, ay),
        B=(bx, by),
        curve=(xt, yt),
    )


def test_exam_shape_and_marks():
    shapes = set()
    for seed in range(60):
        q = _exam(seed)
        labels = [p.label for p in q.parts]
        assert labels in (["a", "b", "c"], ["a", "b", "c", "d"])
        with_k = len(labels) == 4
        shapes.add(with_k)
        if with_k:
            assert [p.answer_kind for p in q.parts] == ["fraction", "expression", "fraction", "fraction"]
            assert [p.marks for p in q.parts] == [2, 3, 2, 2]
            assert [p.follow_through is not None for p in q.parts] == [False, True, True, True]
        else:
            assert [p.answer_kind for p in q.parts] == ["expression", "fraction", "fraction"]
            assert [p.marks for p in q.parts] == [3, 2, 2]
            assert [p.follow_through is not None for p in q.parts] == [False, True, True]
        assert all(len(p.distractors) >= 2 for p in q.parts)
        assert find_text_defects(q) == []
    assert shapes == {True, False}


def test_exam_keys_match_independent_derivation():
    for seed in range(40):
        q = _exam(seed)
        f = _facts(q)
        parts = {p.label: p.answer_text for p in q.parts}
        labels = list(parts)
        if f["k"] is not None:
            assert sympy.Rational(parts["a"]) == f["k"], (seed, q.stem)
        phi_label = labels[f["offset"]]
        assert sympy.expand(_read(parts[phi_label]) - f["phi"]) == 0, (seed, parts[phi_label], f["phi"])
        assert sympy.Rational(parts[labels[1 + f["offset"]]]) == f["work_ab"], seed
        assert sympy.Rational(parts[labels[2 + f["offset"]]]) == f["work_c"], seed


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def test_exam_key_accepted_every_distractor_rejected():
    for seed in range(40):
        q = _exam(seed)
        full = grade(_SKILL, q, q.answer_text)
        assert full.correct and full.marks == full.max_marks == sum(p.marks for p in q.parts)
        canon = _canonical(q)
        for p in q.parts:
            for text in p.distractors:
                assert grade(_SKILL, q, {**canon, p.label: text}).parts[p.label].marks == 0, (seed, p.label, text)


def test_follow_through_from_a_wrong_potential():
    checked = 0
    for seed in range(45):
        q = _exam(seed)
        f = _facts(q)
        canon = _canonical(q)
        labels = list(canon)
        pl = labels[f["offset"]]
        if f["offset"]:
            continue  # the parameter chain follows through from k below
        for wrong in q.parts[0].distractors:
            wphi = _read(wrong)
            wab = wphi.subs({x: f["B"][0], y: f["B"][1]}) - wphi.subs({x: f["A"][0], y: f["A"][1]})
            if wab in (0, f["work_ab"]) or wab.q > 12:
                continue
            reply = {**canon, pl: wrong, "b": str(wab)}
            result = grade(_SKILL, q, reply)
            assert result.parts[pl].marks == 0
            assert result.parts["b"].marks == 2 and result.parts["b"].follow_through is True, (seed, wrong, result)
            checked += 1
    assert checked >= 30


def test_follow_through_from_a_wrong_k_into_the_potential():
    checked = 0
    for seed in range(45):
        q = _exam(seed)
        f = _facts(q)
        if not f["offset"]:
            continue
        canon = _canonical(q)
        for wrong in q.parts[0].distractors:
            wk = sympy.Rational(wrong)
            if wk == f["k"] or wk == 0:
                continue
            P, Q = _vector_after(q.stem, "F(x, y) = ")
            # the potential of the k-field with the learner's k (what the follow-through formula says)
            wphi = _potential(P.subs(k, f["k"]), Q) + (wk - f["k"]) * x * y**2
            reply = {**canon, "a": wrong, "b": str(sympy.expand(wphi)).replace("**", "^")}
            result = grade(_SKILL, q, reply)
            assert result.parts["a"].marks == 0
            assert result.parts["b"].marks == 3 and result.parts["b"].follow_through is True, (seed, wrong, result)
            checked += 1
    assert checked >= 20


def test_follow_through_from_a_wrong_potential_into_the_second_path():
    checked = 0
    for seed in range(60):
        q = _exam(seed)
        f = _facts(q)
        canon = _canonical(q)
        labels = list(canon)
        pl = labels[f["offset"]]
        c_label = labels[2 + f["offset"]]
        xt, yt = f["curve"]
        end, start = (xt.subs(t, 1), yt.subs(t, 1)), (xt.subs(t, 0), yt.subs(t, 0))
        for wrong in q.parts[f["offset"]].distractors:
            wphi = _read(wrong)
            wc = wphi.subs({x: end[0], y: end[1]}) - wphi.subs({x: start[0], y: start[1]})
            if wc in (0, f["work_c"]) or wc.q > 12:
                continue
            reply = {**canon, pl: wrong, c_label: str(wc)}
            result = grade(_SKILL, q, reply)
            assert result.parts[c_label].marks == 2 and result.parts[c_label].follow_through is True, (seed, wrong)
            checked += 1
    assert checked >= 30


def test_exam_answer_space():
    answers = [_exam(seed).answer_text for seed in range(300)]
    assert len(set(answers)) >= 100
    assert max(answers.count(a) for a in set(answers)) <= 15


def test_exam_deterministic():
    assert _exam(7).to_json() == _exam(7).to_json()
