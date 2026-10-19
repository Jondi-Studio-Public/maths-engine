"""Tests `uc2-double-integral-general-region`: the topic `s2a1_region_integral` (easy, medium, hard) and the exam chain
`exd_x_s2a1_region_integral`.

The oracle reads only the question text: it pulls the boundary equations (or the triangle's vertices) out of the stem,
rebuilds the region as the bounded set those boundaries enclose, and integrates over it with a fine midpoint grid.
That route knows nothing about the generator's limits, slices or order of integration.
"""

from __future__ import annotations

import re

import numpy as np
import pytest
import sympy

import mathgen.generators.s2a1_region_build  # noqa: F401  (registers the topics)
from mathgen.engine import grade
from mathgen.text_quality import find_text_defects
from tests import s2a1_oracle as oracle

SKILL_ID = "uc2-double-integral-general-region"
TOPIC = "s2a1_region_integral"
EXAM_TOPIC = "exd_x_s2a1_region_integral"
SK = oracle.skill(SKILL_ID)
TIERS = ("easy", "medium", "hard")
X, Y = oracle.x, oracle.y


def _expr(text: str):
    text = re.sub(r"(\d|\))(?=[xy])", r"\1*", text.replace("²", "**2").replace("³", "**3"))
    text = re.sub(r"(?<=\*\*2)(?=[xy])", "*", text)
    return oracle.read(text)


def _segment(text: str) -> str:
    seg = text.split(" bounded by ", 1)[-1] if " bounded by " in text else text.split(" boundary is ", 1)[-1]
    seg = seg.split(" and below ", 1)[0]
    return re.split(r", with|\.", seg, maxsplit=1)[0]


def region_mask(text: str):
    """(mask function over a grid, bounding box) of the region the stem describes."""
    seg = _segment(text)
    tri = re.search(r"triangle with vertices \((\d+), (\d+)\), \((\d+), (\d+)\) and \((\d+), (\d+)\)", text)
    if tri:
        ax, ay, bx, by, cx, cy = map(int, tri.groups())

        def mask(gx, gy):
            s1 = (bx - ax) * (gy - ay) - (by - ay) * (gx - ax)
            s2 = (cx - bx) * (gy - by) - (cy - by) * (gx - bx)
            s3 = (ax - cx) * (gy - cy) - (ay - cy) * (gx - cx)
            return ((s1 >= 0) & (s2 >= 0) & (s3 >= 0)) | ((s1 <= 0) & (s2 <= 0) & (s3 <= 0))

        return mask, (min(ax, bx, cx), max(ax, bx, cx), min(ay, by, cy), max(ay, by, cy))
    funcs = []
    for found in re.finditer(r"(?:(?<=\s)|^)(x|y|\d*x \+ \d*y) = (.+?)(?= and |$)", seg):
        lhs, rhs = found.groups()
        funcs.append(sympy.lambdify((X, Y), _expr(lhs) - _expr(rhs), "numpy"))
    axes = "coordinate axes" in seg
    assert len(funcs) == 2 or (axes and len(funcs) == 1), (text, seg)
    n = 1500
    g = np.linspace(-30, 30, n)
    gx, gy = np.meshgrid(g, g, indexing="ij")
    for signs in ((1, 1), (1, -1), (-1, 1), (-1, -1)):

        def mask(px, py, signs=signs):
            ok = np.ones(px.shape, dtype=bool)
            if axes:
                ok &= (px >= 0) & (py >= 0)
            for fn, s in zip(funcs, signs):
                ok &= s * (fn(px, py) + 0 * px) >= -1e-12
            return ok

        hit = mask(gx, gy)
        if hit.sum() < 30 or np.abs(gx[hit]).max() > 29 or np.abs(gy[hit]).max() > 29:
            continue
        box = (gx[hit].min() - 0.05, gx[hit].max() + 0.05, gy[hit].min() - 0.05, gy[hit].max() + 0.05)
        return mask, box
    raise AssertionError(("no bounded region", text))


def grid(text: str, f, cells: int = 1400) -> float:
    mask, (x0, x1, y0, y1) = region_mask(text)
    hx, hy = (x1 - x0) / cells, (y1 - y0) / cells
    gx, gy = np.meshgrid(x0 + hx * (np.arange(cells) + 0.5), y0 + hy * (np.arange(cells) + 0.5), indexing="ij")
    fx = sympy.lambdify((X, Y), f, "numpy")
    return float(np.sum(np.where(mask(gx, gy), fx(gx, gy) + 0 * gx, 0.0)) * hx * hy)


def integrand(text: str):
    for pattern in (r"∬ (.+?) dA", r"ρ = (.+?) \(in kilograms", r"z = (.+?)\.$", r"f\(x, y\) = (.+?) over"):
        found = re.search(pattern, text)
        if found:
            return _expr(found.group(1))
    raise AssertionError(text)


def expected(text: str) -> float:
    f = integrand(text)
    value = grid(text, f)
    return value / grid(text, sympy.Integer(1)) if text.startswith("Find the average value") else value


def close(a: float, b: float, rel: float = 0.01) -> bool:
    return abs(a - b) <= rel * max(abs(b), 1e-9) + 2e-4


def test_source_wired_last():
    last = SK.sources[-1]
    assert (last.topic, last.difficulty, last.stages, last.params) == (EXAM_TOPIC, "exam", ["exam"], {"tier": "exam"})
    assert [s.difficulty for s in SK.sources] == ["easy", "medium", "hard", "exam"]
    assert list(SK.year_level) == ["university"] and "university-year-2" in SK.tags


@pytest.mark.parametrize("tier", TIERS)
def test_key_matches_independent_grid_integral(tier):
    for seed in range(24):
        q = oracle.question(TOPIC, tier, seed)
        key = float(sympy.Rational(q.answer_text))
        assert close(expected(q.text), key), (seed, q.text, q.answer_text, expected(q.text))


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_and_every_distractor_rejected(tier):
    oracle.check_key_and_distractors(SK, TOPIC, tier, range(60))


@pytest.mark.parametrize("tier", (*TIERS, "exam"))
def test_distractors_are_tagged_with_declared_errors(tier):
    oracle.check_tags(SK, EXAM_TOPIC if tier == "exam" else TOPIC, tier, range(30))


def test_no_giveaway_negatives_for_physical_quantities():
    for tier in TIERS:
        for seed in range(60):
            q = oracle.question(TOPIC, tier, seed)
            if not q.text.startswith("Evaluate"):
                assert all(not d.startswith("-") for d in q.distractors), (seed, q.text, q.distractors)


def test_each_tier_has_the_shapes_the_design_asks_for():
    easy = [oracle.question(TOPIC, "easy", s).text for s in range(60)]
    assert sum("coordinate axes" in t for t in easy) >= 3 and sum("curves" in t for t in easy) >= 15
    medium = [oracle.question(TOPIC, "medium", s).text for s in range(60)]
    assert sum(re.search(r"parabola x = ", t) is not None for t in medium) >= 10
    assert sum("thin plate" in t for t in medium) >= 8 and sum("volume" in t for t in medium) >= 8
    hard = [oracle.question(TOPIC, "hard", s).text for s in range(80)]
    assert sum("triangle" in t for t in hard) >= 15 and sum("x = y²" in t for t in hard) >= 3


def test_no_question_is_a_rectangle():
    for tier in TIERS:
        for seed in range(40):
            q = oracle.question(TOPIC, tier, seed)
            assert "rectangle" not in q.text


@pytest.mark.parametrize("tier", TIERS)
def test_variety(tier):
    answers = [oracle.question(TOPIC, tier, seed).answer_text for seed in range(80)]
    assert len(set(answers)) >= 12
    assert max(answers.count(a) for a in set(answers)) <= 10


@pytest.mark.parametrize("tier", TIERS)
def test_text_quality_and_determinism(tier):
    for seed in range(40):
        q = oracle.question(TOPIC, tier, seed)
        assert find_text_defects(q) == [], (seed, find_text_defects(q))
        assert q.to_json() == oracle.question(TOPIC, tier, seed).to_json()


def test_closed_forms():
    # the area between y = k x^2 and y = k a x is k a^3 / 6, and the lens y = x^2, x = y^2 has area 1/3
    seen = 0
    for seed in range(80):
        q = oracle.question(TOPIC, "easy", seed)
        found = re.match(
            r"^Evaluate ∬ (\w) dA, where R is the region bounded by the curves y = (\d*)x² and y = (\d*)x\.$", q.text
        )
        if found and found.group(1) == "x":
            k = int(found.group(2) or 1)
            ka = int(found.group(3) or 1)
            a = sympy.Rational(ka, k)
            # integral of x over the region = k * a^4 / 12
            assert sympy.Rational(q.answer_text) == k * a**4 / 12
            seen += 1
    assert seen >= 1


# ---------------------------------------------------------------------------------------------------------------
# the exam chain


def test_exam_shape_and_marks():
    oracle.check_exam_shape(SK, EXAM_TOPIC, range(40), [3, 3, 3, 1], [{"fraction"}] * 4)
    for seed in range(20):
        assert find_text_defects(oracle.question(EXAM_TOPIC, "exam", seed)) == []


def test_exam_keys_match_independent_grid_integral():
    for seed in range(20):
        q = oracle.question(EXAM_TOPIC, "exam", seed)
        stem = q.stem
        k, m, w = re.search(r"ρ = (\d+) \+ (\d*)([xy])", stem).groups()
        rho = int(k) + int(m or 1) * (X if w == "x" else Y)
        u = X if "«x»" in q.parts[2].prompt or "x-coordinate" in q.parts[2].prompt else Y
        area, mass, moment = (grid(stem, f) for f in (sympy.Integer(1), rho, rho * u))
        keys = [float(sympy.Rational(p.answer_text)) for p in q.parts]
        for estimate, key in zip((area, mass, moment / mass, mass / area), keys):
            assert close(estimate, key), (seed, stem, estimate, key)


def test_exam_key_accepted_every_distractor_rejected():
    oracle.check_exam_distractors(SK, EXAM_TOPIC, range(40))


def test_follow_through_carries_a_wrong_answer_forward():
    oracle.check_follow_through(SK, EXAM_TOPIC, range(40), minimum=20)


def test_exam_answer_space_and_determinism():
    answers = [oracle.question(EXAM_TOPIC, "exam", seed).answer_text for seed in range(150)]
    assert len(set(answers)) >= 40
    assert oracle.question(EXAM_TOPIC, "exam", 7).to_json() == oracle.question(EXAM_TOPIC, "exam", 7).to_json()


def test_a_wrong_reply_is_not_accepted_for_the_whole_chain():
    q = oracle.question(EXAM_TOPIC, "exam", 3)
    assert not grade(SK, q, {p.label: "0" for p in q.parts}).correct
