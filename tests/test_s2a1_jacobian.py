"""Tests `uc2-jacobian-change-of-variables`: the topic `s2a1_jacobian_cov` (easy, medium, hard) and the exam chain
`exd_x_s2a1_jacobian_cov`.

The oracle reads only the question text, rebuilds the region from the vertices, lines or curves it states, and sums a
fine grid of the integrand there. It never changes variables and never forms a Jacobian, so a wrong determinant, a
missing absolute value or a wrongly transformed limit in the generator shows up as a mismatch.
"""

from __future__ import annotations

import re

import numpy as np
import pytest
import sympy

import mathgen.generators.s2a1_jacobian_build  # noqa: F401  (registers the topics)
from mathgen.engine import grade
from mathgen.text_quality import find_text_defects
from tests import s2a1_oracle as oracle

SKILL_ID = "uc2-jacobian-change-of-variables"
TOPIC = "s2a1_jacobian_cov"
EXAM_TOPIC = "exd_x_s2a1_jacobian_cov"
SK = oracle.skill(SKILL_ID)
TIERS = ("easy", "medium", "hard")
X, Y = oracle.x, oracle.y


def _expr(text: str):
    text = text.strip().replace("²", "**2").replace("³", "**3")
    text = re.sub(r"(\d|\))(?=[xy(])", r"\1*", text)
    text = re.sub(r"(?<=\*\*2)(?=[xy])", "*", text)
    return oracle.read(text)


def _polygon_mask(vertices):
    pts = list(vertices)

    def mask(gx, gy):
        signs = []
        for (ax, ay), (bx, by) in zip(pts, pts[1:] + pts[:1]):
            signs.append((bx - ax) * (gy - ay) - (by - ay) * (gx - ax))
        return np.all([s >= -1e-12 for s in signs], axis=0) | np.all([s <= 1e-12 for s in signs], axis=0)

    return mask, (min(p[0] for p in pts), max(p[0] for p in pts), min(p[1] for p in pts), max(p[1] for p in pts))


def _bounded(mask):
    n = 1500
    g = np.linspace(-40, 40, n)
    gx, gy = np.meshgrid(g, g, indexing="ij")
    hit = mask(gx, gy)
    assert hit.sum() > 30 and np.abs(gx[hit]).max() < 39 and np.abs(gy[hit]).max() < 39
    return gx[hit].min() - 0.05, gx[hit].max() + 0.05, gy[hit].min() - 0.05, gy[hit].max() + 0.05


def region(text: str):
    """(mask, box) of the region the text describes."""
    vertices = re.search(r"vertices (.+?)\.", text)
    if "vertices" in text:
        pts = [tuple(int(n) for n in found) for found in re.findall(r"\((-?\d+), (-?\d+)\)", vertices.group(1))]
        return _polygon_mask(pts)
    hyper = re.search(r"xy = (\d+), xy = (\d+) and the lines y = (\d*)x(?:,| and) y = (\d*)x", text)
    if hyper:
        a, b, c, d = (int(g or 1) for g in hyper.groups())

        def mask(gx, gy):
            return (gx > 0) & (gx * gy >= a) & (gx * gy <= b) & (gy >= c * gx) & (gy <= d * gx)

        return mask, (0.0, (b / c) ** 0.5 + 0.02, 0.0, (b * d) ** 0.5 + 0.02)
    seg = text.split("bounded by the lines ", 1)[1]
    seg = re.split(r"\. |, with", seg, maxsplit=1)[0]
    limits: dict[str, list[int]] = {}
    for lhs, rhs in re.findall(r"([xy0-9+\- ]*?[xy]) = (-?\d+)", seg):
        limits.setdefault(lhs.strip(), []).append(int(rhs))
    assert len(limits) == 2 and all(len(v) == 2 for v in limits.values()), (text, limits)
    funcs = [(sympy.lambdify((X, Y), _expr(lhs), "numpy"), min(v), max(v)) for lhs, v in limits.items()]

    def mask(gx, gy):
        ok = np.ones(gx.shape, dtype=bool)
        for fn, lo, hi in funcs:
            val = fn(gx, gy) + 0 * gx
            ok &= (val >= lo - 1e-12) & (val <= hi + 1e-12)
        return ok

    return mask, _bounded(mask)


def grid(f, text: str, n: int = 1500) -> float:
    mask, (x0, x1, y0, y1) = region(text)
    hx, hy = (x1 - x0) / n, (y1 - y0) / n
    gx, gy = np.meshgrid(x0 + hx * (np.arange(n) + 0.5), y0 + hy * (np.arange(n) + 0.5), indexing="ij")
    fx = sympy.lambdify((X, Y), f, "numpy")
    return float(np.sum(np.where(mask(gx, gy), fx(gx, gy) + 0 * gx, 0.0)) * hx * hy)


def integrand(text: str):
    if "area of R" in text:
        return sympy.Integer(1)
    return _expr(re.search(r"∬ (.+?) dA", text).group(1))


def _num(answer_text: str) -> float:
    s = answer_text.replace("^", "**").replace("ln", "log")
    s = re.sub(r"(?<=\d)(?=[a-z(])", "*", s)
    s = re.sub(r"(?<=\))(?=log|\d)", "*", s)
    return float(sympy.sympify(s))


def close(a: float, b: float, rel: float = 1e-2) -> bool:
    return abs(a - b) <= rel * max(abs(b), 1e-9) + 1e-6


def test_source_wired_last():
    last = SK.sources[-1]
    assert (last.topic, last.difficulty, last.stages, last.params) == (EXAM_TOPIC, "exam", ["exam"], {"tier": "exam"})
    assert [s.difficulty for s in SK.sources] == ["easy", "medium", "hard", "exam"]
    assert list(SK.year_level) == ["university"] and "university-year-2" in SK.tags
    assert "jacobian-two-variable" in SK.prerequisites


@pytest.mark.parametrize("tier", TIERS)
def test_key_matches_the_region_summed_directly(tier):
    for seed in range(24):
        q = oracle.question(TOPIC, tier, seed)
        expected = grid(integrand(q.text), q.text)
        assert close(expected, _num(q.answer_text)), (seed, q.text, q.answer_text, expected)


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_and_every_distractor_rejected(tier):
    oracle.check_key_and_distractors(SK, TOPIC, tier, range(60))


@pytest.mark.parametrize("tier", (*TIERS, "exam"))
def test_distractors_are_tagged_with_declared_errors(tier):
    oracle.check_tags(SK, EXAM_TOPIC if tier == "exam" else TOPIC, tier, range(30))


def test_each_tier_has_the_shapes_the_design_asks_for():
    easy = [oracle.question(TOPIC, "easy", s).text for s in range(60)]
    assert all("parallelogram" in t and "maps the rectangle" in t for t in easy)
    assert sum("area of R" in t for t in easy) >= 3
    medium = [oracle.question(TOPIC, "medium", s).text for s in range(40)]
    assert all("bounded by the lines" in t and "Use the substitution" in t for t in medium)
    hard = [oracle.question(TOPIC, "hard", s).text for s in range(90)]
    assert sum("first quadrant" in t for t in hard) >= 15
    assert sum("triangle" in t for t in hard) >= 15
    assert sum("bounded by the lines" in t and "substitution" not in t for t in hard) >= 15


def test_negative_determinants_appear_and_are_handled():
    negative = 0
    for seed in range(80):
        q = oracle.question(TOPIC, "easy", seed)
        found = re.search(r"x = (.+?), y = (.+?) maps", q.text)
        u_, v_ = sympy.symbols("U V")
        xe = sympy.sympify(re.sub(r"(\d)(?=[uv])", r"\1*", found.group(1)), locals={"u": u_, "v": v_})
        ye = sympy.sympify(re.sub(r"(\d)(?=[uv])", r"\1*", found.group(2)), locals={"u": u_, "v": v_})
        det = sympy.Matrix([[xe.coeff(u_), xe.coeff(v_)], [ye.coeff(u_), ye.coeff(v_)]]).det()
        if det < 0:
            negative += 1
            assert not q.answer_text.startswith("-"), (q.text, q.answer_text)
    assert negative >= 10


@pytest.mark.parametrize("tier", TIERS)
def test_variety(tier):
    answers = [oracle.question(TOPIC, tier, seed).answer_text for seed in range(100)]
    assert len(set(answers)) >= 25
    assert max(answers.count(a) for a in set(answers)) <= 8


@pytest.mark.parametrize("tier", TIERS)
def test_text_quality_and_determinism(tier):
    for seed in range(40):
        q = oracle.question(TOPIC, tier, seed)
        assert find_text_defects(q) == [], (seed, find_text_defects(q))
        assert q.to_json() == oracle.question(TOPIC, tier, seed).to_json()


def test_hyperbolic_area_closed_form():
    # the area between xy = a, xy = b, y = c x and y = d x is (b - a) ln(d/c) / 2
    seen = 0
    for seed in range(120):
        q = oracle.question(TOPIC, "hard", seed)
        found = re.search(
            r"curves xy = (\d+), xy = (\d+) and the lines y = (\d*)x, y = (\d*)x\. .* find the area", q.text
        ) or re.search(
            r"curves xy = (\d+), xy = (\d+) and the lines y = (\d*)x, y = (\d*)x\. Use the substitution u = xy, v = y/x to find the area",
            q.text,
        )
        if found:
            a, b, c, d = (int(g or 1) for g in found.groups())
            assert abs(_num(q.answer_text) - float((b - a) * sympy.log(sympy.Rational(d, c)) / 2)) < 1e-9
            seen += 1
    assert seen >= 2


# ---------------------------------------------------------------------------------------------------------------
# the exam chain


def test_exam_shape_and_marks():
    oracle.check_exam_shape(SK, EXAM_TOPIC, range(40), [2, 2, 3, 1], [{"fraction"}] * 4)
    for seed in range(20):
        assert find_text_defects(oracle.question(EXAM_TOPIC, "exam", seed)) == []


def test_exam_keys_match_the_plate_summed_directly():
    for seed in range(20):
        q = oracle.question(EXAM_TOPIC, "exam", seed)
        stem = q.stem
        k, m, which = re.search(r"ρ = (\d+) \+ (\d*)([xy])", stem).groups()
        rho = int(k) + int(m or 1) * (X if which == "x" else Y)
        area, mass = grid(sympy.Integer(1), stem), grid(rho, stem)
        keys = [sympy.Rational(p.answer_text) for p in q.parts]
        assert close(area, float(keys[1])) and close(mass, float(keys[2])) and close(mass / area, float(keys[3])), (
            seed,
            stem,
            keys,
        )
        # the Jacobian: the area of the plate is |J| times the area of the rectangle in the (u, v)-plane
        bounds = {}
        for lhs, rhs in re.findall(
            r"«?([xy0-9+\- ]*?[xy]) = (-?\d+)", stem.split("bounded by the lines ")[1].split(", with")[0]
        ):
            bounds.setdefault(lhs.strip(), []).append(int(rhs))
        rect = 1
        for pair in bounds.values():
            rect *= max(pair) - min(pair)
        assert abs(keys[0]) == keys[1] / rect, (seed, stem, keys)
        assert (keys[0] < 0) == (keys[0] != abs(keys[0]))


def test_exam_key_accepted_every_distractor_rejected():
    oracle.check_exam_distractors(SK, EXAM_TOPIC, range(40))


def test_follow_through_carries_a_wrong_answer_forward():
    oracle.check_follow_through(SK, EXAM_TOPIC, range(40), minimum=10)


def test_exam_has_negative_jacobians():
    negative = sum(
        float(sympy.Rational(oracle.question(EXAM_TOPIC, "exam", s).parts[0].answer_text)) < 0 for s in range(60)
    )
    assert negative >= 10


def test_exam_answer_space_and_determinism():
    answers = [oracle.question(EXAM_TOPIC, "exam", seed).answer_text for seed in range(150)]
    assert len(set(answers)) >= 40
    assert oracle.question(EXAM_TOPIC, "exam", 7).to_json() == oracle.question(EXAM_TOPIC, "exam", 7).to_json()


def test_a_wrong_reply_is_not_accepted_for_the_whole_chain():
    q = oracle.question(EXAM_TOPIC, "exam", 3)
    assert not grade(SK, q, {p.label: "0" for p in q.parts}).correct
