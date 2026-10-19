"""Tests the skill `gradient-vector` (topic `w5c_gradient_vector`) and its exam chain `exd_x_w5c_gradient_vector`.

The oracle reads the function and the point back out of the question TEXT, parses the function with SymPy and
re-derives every key (gradient, unit vector, tangent, solution of the gradient equations, rates of change)."""

from __future__ import annotations

import math
import re
from decimal import ROUND_HALF_UP, Decimal

import pytest
import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects

TOPIC = "w5c_gradient_vector"
EXAM_TOPIC = "exd_x_w5c_gradient_vector"
SKILL_ID = "gradient-vector"
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)
TIERS = ("easy", "medium", "hard")
SEEDS = range(60)
X, Y = sympy.symbols("x y")
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


def _exam(seed):
    return make(EXAM_TOPIC, seed=seed, difficulty="exam")


def _r(x: float, dp: int) -> str:
    return str(Decimal(repr(round(x, 12))).quantize(Decimal(1).scaleb(-dp), rounding=ROUND_HALF_UP))


def parse_function(text: str):
    m = re.search(r"(?:\(x, y\)|z) = (.+?)(?:\. | has | equals |, calculate| have )", text)
    raw = m.group(1)
    raw = raw.replace("^", "**").replace("√(", "sqrt(").replace("ln(", "log(").replace("e**(", "exp(")
    raw = re.sub(r"\)(?=[a-z])", ")*", raw)
    return parse_expr(raw, local_dict={"x": X, "y": Y}, transformations=_TRANSFORMS)


def pairs(text: str):
    return [(int(a), int(b)) for a, b in re.findall(r"\((-?\d+), (-?\d+)\)", text)]


def grad(f, pt):
    sub = {X: pt[0], Y: pt[1]}
    return sympy.nsimplify(sympy.diff(f, X).subs(sub)), sympy.nsimplify(sympy.diff(f, Y).subs(sub))


def _fmt(v) -> str:
    v = sympy.Rational(v)
    if v.q == 1:
        return str(v.p)
    for dp in range(1, 5):
        if (v * 10**dp).q == 1:
            return f"{float(v):.{dp}f}"
    raise AssertionError(v)


def pair_text(a, b) -> str:
    return f"({_fmt(a)}, {_fmt(b)})"


def unit_text(a, b) -> str:
    return f"({_r(float(a), 2)}, {_r(float(b), 2)})"


# --- oracle ----------------------------------------------------------------------------------------------------


def oracle_gradient(q):
    f = parse_function(q.text)
    gx, gy = grad(f, pairs(q.text)[-1])
    return pair_text(gx, gy)


def _hard_kind(text):
    if "D = …" in text:
        return "directional"
    if "Find the point" in text or "At which point" in text or "coordinates of that point" in text:
        return "solve"
    if "level curve" in text or "contour" in text or "stays constant" in text:
        return "tangent"
    return "steepest"


def oracle_hard(q):
    t = q.text
    kind = _hard_kind(t)
    if kind == "solve":
        f = parse_function(t)
        g = pairs(t)[-1]
        sol = sympy.solve([sympy.diff(f, X) - g[0], sympy.diff(f, Y) - g[1]], [X, Y], dict=True)
        assert len(sol) == 1
        return pair_text(sol[0][X], sol[0][Y])
    f = parse_function(t)
    if kind == "directional":
        pt, v = pairs(t)[0], pairs(t)[1]
        gx, gy = grad(f, pt)
        vl = math.hypot(*v)
        d = (gx * v[0] + gy * v[1]) / sympy.nsimplify(vl)
        m = sympy.sqrt(gx**2 + gy**2)
        return f"D = {_fmt(d)}, M = {_fmt(m)}"
    pt = pairs(t)[-1]
    gx, gy = grad(f, pt)
    length = float(sympy.sqrt(gx**2 + gy**2))
    ux, uy = float(gx) / length, float(gy) / length
    if kind == "steepest":
        if "decrease" in t or "descent" in t or "falls" in t:
            ux, uy = -ux, -uy
        return unit_text(ux, uy)
    tx, ty = (uy, -ux) if uy > 0 else (-uy, ux)
    return unit_text(tx, ty)


@pytest.mark.parametrize("tier", ["easy", "medium"])
def test_gradient_key_matches_the_independent_derivation(tier):
    bad = [s for s in range(200) if oracle_gradient(_q(tier, s)) != _q(tier, s).answer_text]
    assert bad == []


def test_hard_key_matches_the_independent_derivation():
    bad = [(s, oracle_hard(_q("hard", s)), _q("hard", s).answer_text) for s in range(200)]
    assert [b for b in bad if b[1] != b[2]] == []


def test_hard_tier_uses_all_four_shapes():
    assert {_hard_kind(_q("hard", s).text) for s in range(80)} == {"directional", "solve", "tangent", "steepest"}


def test_steepest_covers_ascent_and_descent():
    texts = [_q("hard", s).text for s in range(120) if _hard_kind(_q("hard", s).text) == "steepest"]
    assert any("decrease" in t or "descent" in t or "falls" in t for t in texts) and any(
        "increase" in t or "ascent" in t for t in texts
    )


def test_medium_covers_log_exp_root_and_quotient():
    seen = set()
    for s in range(120):
        t = _q("medium", s).text
        seen.add("log" if "ln(" in t else "exp" if "e^(" in t else "root" if "√" in t else "quotient")
    assert seen == {"log", "exp", "root", "quotient"}


def test_easy_has_a_cross_term_and_no_special_function():
    for s in range(60):
        t = _q("easy", s).text
        assert not any(k in t for k in ("ln(", "√", "e^("))
        f = parse_function(t)
        assert sympy.diff(f, X, Y) != 0


# --- wiring ----------------------------------------------------------------------------------------------------


def test_skill_is_active_and_pair_graded():
    assert _SKILL.status == "active" and _SKILL.answer == "pair"
    assert not getattr(_SKILL, "deferred_reason", None)
    assert [(s.topic, s.difficulty) for s in _SKILL.sources] == [
        (TOPIC, "easy"),
        (TOPIC, "medium"),
        (TOPIC, "hard"),
        (EXAM_TOPIC, "exam"),
    ]
    assert [s.stages for s in _SKILL.sources] == [[0, 1, 2], [3, 4], [5, 6, "maintenance"], ["exam"]]


# --- grading ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_and_every_distractor_rejected(tier):
    for seed in SEEDS:
        q = _q(tier, seed)
        assert q.answer_kind == "pair"
        assert grade(_SKILL, q, q.answer_text).correct, (tier, seed, q.answer_text)
        assert len(q.distractors) >= 2
        assert all(q.option_notes.get(d) for d in q.distractors)
        for d in q.distractors:
            assert not grade(_SKILL, q, d).correct, (tier, seed, d)


@pytest.mark.parametrize("tier", TIERS)
def test_each_distractor_is_diagnosed_as_its_own_error(tier):
    for seed in range(30):
        q = _q(tier, seed)
        for d in q.distractors:
            assert grade(_SKILL, q, d).misconception == q.option_notes[d], (tier, seed, d)


def test_a_swapped_gradient_is_wrong_and_component_form_is_read():
    for seed in range(40):
        q = _q("easy", seed)
        a, b = (v.strip() for v in q.answer_text.strip("()").split(","))
        assert grade(_SKILL, q, f"({a}, {b})").correct
        assert not grade(_SKILL, q, f"({b}, {a})").correct
        assert grade(_SKILL, q, f"{a}i + {b}j".replace("+ -", "- ")).correct or b.startswith("-")


def test_labelled_pair_accepts_either_order_and_rejects_swapped_values():
    for seed in range(120):
        q = _q("hard", seed)
        if _hard_kind(q.text) != "directional":
            continue
        m = re.fullmatch(r"D = (.+), M = (.+)", q.answer_text)
        d, mm = m.groups()
        assert grade(_SKILL, q, f"M = {mm}, D = {d}").correct
        if d != mm:
            assert not grade(_SKILL, q, f"D = {mm}, M = {d}").correct


@pytest.mark.parametrize("tier", TIERS)
def test_variety_and_no_dominant_answer(tier):
    answers = [_q(tier, s).answer_text for s in SEEDS]
    assert len(set(answers)) >= 20
    assert max(answers.count(a) for a in set(answers)) <= 5
    assert len({_q(tier, s).text for s in SEEDS}) >= 50


@pytest.mark.parametrize("tier", TIERS)
def test_deterministic_and_clean_text(tier):
    for seed in SEEDS:
        q = _q(tier, seed)
        assert q.to_json() == _q(tier, seed).to_json()
        assert find_text_defects(q) == []


def test_unit_vectors_really_are_unit_and_perpendicular():
    for seed in range(200):
        q = _q("hard", seed)
        kind = _hard_kind(q.text)
        if kind not in ("steepest", "tangent"):
            continue
        u = [float(v) for v in q.answer_text.strip("()").split(",")]
        assert abs(math.hypot(*u) - 1) < 0.01
        if kind == "tangent":
            assert u[0] > 0


# --- the exam chain --------------------------------------------------------------------------------------------


def _facts(q):
    stem = q.text.split(" (a)")[0]
    f = parse_function(stem)
    pt = re.search(r"P\((-?\d+), (-?\d+)\)", stem)
    qt = re.search(r"Q\((-?\d+), (-?\d+)\)", stem)
    return f, (int(pt.group(1)), int(pt.group(2))), (int(qt.group(1)), int(qt.group(2)))


def exam_oracle(q):
    f, p, qq = _facts(q)
    gx, gy = grad(f, p)
    dx, dy = qq[0] - p[0], qq[1] - p[1]
    length = math.hypot(dx, dy)
    return (
        float(gx),
        float(gy),
        (float(gx) * dx + float(gy) * dy) / length,
        math.hypot(float(gx), float(gy)),
    )


def test_source_wired_last():
    src = _SKILL.sources[-1]
    assert (src.topic, src.difficulty, src.stages, src.params) == (EXAM_TOPIC, "exam", ["exam"], {"tier": "exam"})


def test_exam_shape_marks_and_follow_through():
    for seed in range(60):
        q = _exam(seed)
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.marks for p in q.parts] == [2, 2, 3, 2]
        assert [p.follow_through is not None for p in q.parts] == [False, False, True, True]
        assert find_text_defects(q) == []


def test_exam_keys_match_the_independent_derivation():
    for seed in range(200):
        q = _exam(seed)
        got = [float(p.answer_text) for p in q.parts]
        want = exam_oracle(q)
        assert all(abs(g - w) < 0.0051 for g, w in zip(got, want)), (seed, got, want)


def test_exam_covers_all_three_function_families():
    seen = set()
    for s in range(60):
        t = _exam(s).text
        seen.add("log" if "ln(" in t else "root" if "√" in t else "quotient")
    assert seen == {"log", "root", "quotient"}


def test_key_accepted_and_every_distractor_rejected_in_the_exam():
    for seed in range(100):
        q = _exam(seed)
        full = grade(_SKILL, q, q.answer_text)
        assert full.correct and full.marks == full.max_marks == 9
        canon = {p.label: p.answer_text for p in q.parts}
        for p in q.parts:
            assert len(p.distractors) >= 2
            for text in p.distractors:
                assert grade(_SKILL, q, {**canon, p.label: text}).parts[p.label].marks == 0, (seed, p.label, text)


def test_follow_through_from_a_wrong_partial_derivative():
    checked = 0
    for seed in range(120):
        q = _exam(seed)
        f, p, qq = _facts(q)
        dx, dy = qq[0] - p[0], qq[1] - p[1]
        length = math.hypot(dx, dy)
        wrong_a = float(q.parts[0].distractors[0])
        gy = float(q.parts[1].answer_text)
        if wrong_a == 0:
            continue
        reply = {
            "a": f"{wrong_a:.2f}",
            "b": q.parts[1].answer_text,
            "c": f"{(wrong_a * dx + gy * dy) / length:.2f}",
            "d": f"{math.hypot(wrong_a, gy):.2f}",
        }
        result = grade(_SKILL, q, reply)
        assert result.parts["a"].marks == 0
        assert result.parts["c"].marks == 3 and result.parts["c"].follow_through is True, (seed, result.parts["c"])
        assert result.parts["d"].marks == 2 and result.parts["d"].follow_through is True, (seed, result.parts["d"])
        checked += 1
    assert checked >= 60


def test_rounded_partials_can_be_used_downstream():
    for seed in range(80):
        q = _exam(seed)
        f, p, qq = _facts(q)
        dx, dy = qq[0] - p[0], qq[1] - p[1]
        length = math.hypot(dx, dy)
        ga, gb = float(q.parts[0].answer_text), float(q.parts[1].answer_text)
        reply = {
            "a": q.parts[0].answer_text,
            "b": q.parts[1].answer_text,
            "c": f"{(ga * dx + gb * dy) / length:.2f}",
            "d": f"{math.hypot(ga, gb):.2f}",
        }
        assert grade(_SKILL, q, reply).marks == 9, seed
