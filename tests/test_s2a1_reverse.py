"""Tests `uc2-reverse-order-of-integration`: the topic `s2a1_reverse_order` (easy, medium, hard) and the exam chain
`exd_x_s2a1_reverse_order`.

The oracle reads only the question text. For an iterated integral it integrates in the order GIVEN (which never uses the
reversal the generator relies on); for a region it builds the region from the stated vertices or curves and sums a
fine grid. Separate tests check that every inner integrand really has no elementary antiderivative.
"""

from __future__ import annotations

import re

import numpy as np
import pytest
import sympy

import mathgen.generators.s2a1_reverse_build as build
from mathgen.engine import grade
from mathgen.text_quality import find_text_defects
from tests import s2a1_oracle as oracle

SKILL_ID = "uc2-reverse-order-of-integration"
TOPIC = "s2a1_reverse_order"
EXAM_TOPIC = "exd_x_s2a1_reverse_order"
SK = oracle.skill(SKILL_ID)
TIERS = ("easy", "medium", "hard")
SYMS = {"x": sympy.Symbol("x"), "y": sympy.Symbol("y")}


def _read(text: str):
    text = text.replace("π", "pi").replace("²", "**2").replace("³", "**3").replace("⁴", "**4").replace("^", "**")
    text = re.sub(r"∛(\w)", r"cbrt(\1)", text)
    text = re.sub(r"√(\w)", r"sqrt(\1)", text).replace("√(", "sqrt(")
    text = re.sub(r"e\*\*\(([^()]*)\)", r"exp(\1)", text)
    text = re.sub(r"(\d)(?=[a-z(])", r"\1*", text)
    text = re.sub(r"(\))(?=[a-z])", r"\1*", text)
    text = re.sub(r"(?<=\d) (?=sin)", "*", text)
    text = text.replace(" sin", "*sin")
    return sympy.sympify(text, locals=SYMS)


ITER = re.compile(
    r"^Evaluate the iterated integral ∫ from (\w) = 0 to (.+?) of ∫ from (\w) = (.+?) to (.+?) of (.+?) d\3 d\1\.$"
)


def iterated_value(text: str) -> float:
    outer, top, inner, lower, upper, f = ITER.match(text).groups()
    top, upper = float(_read(top)), float(_read(upper))
    lower_fn = sympy.lambdify(SYMS[outer], _read(lower), "numpy")
    f_fn = sympy.lambdify(SYMS[inner], _read(f), "numpy")
    n = 2400
    o = (np.arange(n) + 0.5) * top / n
    t, w = np.polynomial.legendre.leggauss(80)
    lo = lower_fn(o) + 0 * o
    pts = lo[:, None] + (upper - lo)[:, None] * (t[None, :] + 1) / 2
    inner_values = ((upper - lo)[:, None] / 2 * w[None, :] * f_fn(pts)).sum(axis=1)
    return float(inner_values.sum() * top / n)


def region_value(text: str) -> float:
    found = re.match(r"^Evaluate ∬ (.+?) dA, where R is (.+)\.$", text)
    f_text, where = found.groups()
    integrand = _read(f_text)
    cells = 1600
    tri = re.search(r"triangle with vertices \((\d+), (\d+)\), \((\d+), (\d+)\) and \((\d+), (\d+)\)", where)
    if tri:
        ax, ay, bx, by, cx, cy = map(int, tri.groups())
        x0, x1, y0, y1 = min(ax, bx, cx), max(ax, bx, cx), min(ay, by, cy), max(ay, by, cy)

        def mask(gx, gy):
            s1 = (bx - ax) * (gy - ay) - (by - ay) * (gx - ax)
            s2 = (cx - bx) * (gy - by) - (cy - by) * (gx - bx)
            s3 = (ax - cx) * (gy - cy) - (ay - cy) * (gx - cx)
            return ((s1 >= 0) & (s2 >= 0) & (s3 >= 0)) | ((s1 <= 0) & (s2 <= 0) & (s3 <= 0))
    else:
        curve = re.search(r"curve (\w) = (\w)([²³])", where)
        line = re.search(r"line (\w) = (\d+)", where)
        v, w, power = curve.groups()
        k = {"²": 2, "³": 3}[power]
        top = float(line.group(2))
        assert line.group(1) == w and "first quadrant" in where
        if v == "x":
            x0, x1, y0, y1 = 0, top**k, 0, top

            def mask(gx, gy):
                return (gx >= 0) & (gx <= gy**k) & (gy <= top) & (gy >= 0)
        else:
            x0, x1, y0, y1 = 0, top, 0, top**k

            def mask(gx, gy):
                return (gy >= 0) & (gy <= gx**k) & (gx <= top) & (gx >= 0)

    hx, hy = (x1 - x0) / cells, (y1 - y0) / cells
    gx, gy = np.meshgrid(x0 + hx * (np.arange(cells) + 0.5), y0 + hy * (np.arange(cells) + 0.5), indexing="ij")
    fx = sympy.lambdify((SYMS["x"], SYMS["y"]), integrand, "numpy")
    return float(np.sum(np.where(mask(gx, gy), fx(gx, gy) + 0 * gx, 0.0)) * hx * hy)


def expected(text: str) -> float:
    return iterated_value(text) if text.startswith("Evaluate the iterated") else region_value(text)


def _num(answer_text: str) -> float:
    s = answer_text.replace("π", "pi").replace("^", "**")
    s = re.sub(r"√(\d+)", r"sqrt(\1)", s)
    s = re.sub(r"(?<=\d)(?=[a-z(])", "*", s)
    s = re.sub(r"\be\b", "E", s)
    s = re.sub(r"(?<=\))(?=E|pi)", "*", s)
    s = re.sub(r"(?<=[\dE])(?=\()", "*", s)
    return float(sympy.sympify(s))


def close(a: float, b: float, rel: float = 1.5e-3) -> bool:
    return abs(a - b) <= rel * max(abs(b), 1e-9)


def test_source_wired_last():
    last = SK.sources[-1]
    assert (last.topic, last.difficulty, last.stages, last.params) == (EXAM_TOPIC, "exam", ["exam"], {"tier": "exam"})
    assert [s.difficulty for s in SK.sources] == ["easy", "medium", "hard", "exam"]
    assert list(SK.year_level) == ["university"] and "university-year-2" in SK.tags


@pytest.mark.parametrize("key_", sorted(build.FAMILIES))
def test_every_inner_integrand_has_no_elementary_antiderivative(key_):
    result = sympy.integrate(build.FAMILIES[key_].h, build.u)
    special = {"erfi", "lowergamma", "uppergamma", "hyper", "Si", "Ei", "meijerg", "Integral"}
    assert {type(a).__name__ for a in result.atoms(sympy.Function, sympy.Integral)} & special, result


@pytest.mark.parametrize("tier", TIERS)
def test_key_matches_integral_in_the_order_given(tier):
    for seed in range(24):
        q = oracle.question(TOPIC, tier, seed)
        assert close(expected(q.text), _num(q.answer_text)), (seed, q.text, q.answer_text, expected(q.text))


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_and_every_distractor_rejected(tier):
    oracle.check_key_and_distractors(SK, TOPIC, tier, range(60))


@pytest.mark.parametrize("tier", (*TIERS, "exam"))
def test_distractors_are_tagged_with_declared_errors(tier):
    oracle.check_tags(SK, EXAM_TOPIC if tier == "exam" else TOPIC, tier, range(30))


def test_each_tier_has_the_shapes_the_design_asks_for():
    easy = [oracle.question(TOPIC, "easy", s).text for s in range(40)]
    assert all("e^(" in t and "²)" in t for t in easy)
    medium = [oracle.question(TOPIC, "medium", s).text for s in range(60)]
    assert (
        sum("sin(" in t for t in medium) >= 8
        and sum("√(1+" in t for t in medium) >= 8
        and sum("e^(" in t for t in medium) >= 5
    )
    hard = [oracle.question(TOPIC, "hard", s).text for s in range(60)]
    assert sum("∛" in t for t in hard) >= 8 and sum("where R is" in t for t in hard) >= 12
    assert sum("triangle" in t for t in hard) >= 5 and sum("quadrant" in t for t in hard) >= 5


def test_the_given_order_is_never_the_easy_one():
    # the inner variable always appears in the inner integrand and in no limit that could be integrated directly
    for tier in TIERS:
        for seed in range(40):
            q = oracle.question(TOPIC, tier, seed)
            if q.text.startswith("Evaluate the iterated"):
                outer, inner = ITER.match(q.text).group(1), ITER.match(q.text).group(3)
                assert outer != inner and f"d{inner} d{outer}" in q.text


@pytest.mark.parametrize("tier", TIERS)
def test_variety(tier):
    answers = [oracle.question(TOPIC, tier, seed).answer_text for seed in range(100)]
    assert len(set(answers)) >= 8
    assert max(answers.count(a) for a in set(answers)) <= 30


@pytest.mark.parametrize("tier", TIERS)
def test_text_quality_and_determinism(tier):
    for seed in range(40):
        q = oracle.question(TOPIC, tier, seed)
        assert find_text_defects(q) == [], (seed, find_text_defects(q))
        assert q.to_json() == oracle.question(TOPIC, tier, seed).to_json()


def test_closed_forms():
    # e^(y^2) over 0 <= x <= m y, y <= U: the value is c m (e^(U^2) - 1) / 2
    seen = 0
    for seed in range(120):
        q = oracle.question(TOPIC, "easy", seed)
        got = re.match(
            r"^Evaluate the iterated integral ∫ from \w = 0 to (\d+) of ∫ from \w = \w(?:/(\d+))? to (\d+) of (\d*)e\^\(\w²\) d\w d\w\.$",
            q.text,
        )
        assert got, q.text
        top, m, upper, c = got.groups()
        m, c, upper = int(m or 1), int(c or 1), int(upper)
        assert int(top) == m * upper
        assert abs(_num(q.answer_text) - float(c * m * (sympy.exp(upper**2) - 1) / 2)) < 1e-9 * float(
            sympy.exp(upper**2)
        )
        seen += 1
    assert seen == 120


# ---------------------------------------------------------------------------------------------------------------
# the exam chain


def test_exam_shape_and_marks():
    oracle.check_exam_shape(SK, EXAM_TOPIC, range(40), [1, 2, 3, 1], [{"fraction", "expression"}] * 4)
    for seed in range(20):
        assert find_text_defects(oracle.question(EXAM_TOPIC, "exam", seed)) == []


def _iterated_text(stem: str) -> str:
    return (
        stem.replace("Let I be the iterated integral", "Evaluate the iterated integral", 1).split(", and let R")[0]
        + "."
    )


def test_exam_keys_match_independent_integrals():
    for seed in range(20):
        q = oracle.question(EXAM_TOPIC, "exam", seed)
        text = _iterated_text(q.stem)
        total = iterated_value(text)
        outer, top, inner, lower, upper, f = ITER.match(text).groups()
        # area: the iterated integral of 1
        area = iterated_value(text.replace(f" of {f} d", " of 1 d"))
        keys = [_num(p.answer_text) for p in q.parts]
        assert close(area, keys[0]) and close(total, keys[2]) and close(total / area, keys[3]), (
            seed,
            q.stem,
            keys,
            area,
            total,
        )
        # (b): the reversed inner integral at 1 is h(1) times phi(1), where v = phi(u) inverts the lower limit
        p = sympy.Symbol("p", positive=True)
        phi_one = sympy.solve(sympy.Eq(_read(lower).subs(SYMS[outer], p), 1), p)
        assert len(phi_one) == 1, text
        g = float(_read(f).subs(SYMS[inner], 1)) * float(phi_one[0])
        assert close(g, keys[1]), (seed, text, g, keys[1])


def test_exam_key_accepted_every_distractor_rejected():
    oracle.check_exam_distractors(SK, EXAM_TOPIC, range(40))


def test_follow_through_carries_a_wrong_answer_forward():
    oracle.check_follow_through(SK, EXAM_TOPIC, range(40), minimum=10)


def test_exam_answer_space_and_determinism():
    answers = [oracle.question(EXAM_TOPIC, "exam", seed).answer_text for seed in range(150)]
    assert len(set(answers)) >= 12
    assert oracle.question(EXAM_TOPIC, "exam", 7).to_json() == oracle.question(EXAM_TOPIC, "exam", 7).to_json()


def test_a_wrong_reply_is_not_accepted_for_the_whole_chain():
    q = oracle.question(EXAM_TOPIC, "exam", 3)
    assert not grade(SK, q, {p.label: "0" for p in q.parts}).correct
