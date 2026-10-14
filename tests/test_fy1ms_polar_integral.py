"""Tests the skill `ucv1-polar-double-integral`: an independent oracle that re-derives every key from the question TEXT
(SymPy integration in polar coordinates, and nested numerical integration of the Cartesian integrand), the typed
grader on every key and distractor, and the four-part exam chain with follow-through."""

from __future__ import annotations

import re

import mpmath
import pytest
import sympy

from tests import fy1ms_support as T

SKILL_ID = "ucv1-polar-double-integral"
TOPIC = "fy1ms_polar_integral"
EXAM = "exd_x_fy1ms_polar_integral"
SK = T.load(SKILL_ID)
TIERS = ("easy", "medium", "hard")
x, y, r, th = sympy.symbols("x y r theta", positive=True)
pi = sympy.pi


def _angles(constraints: str | None):
    c = constraints or ""
    if "x ≥ 0" in c and "y ≥ 0" in c:
        return 0, pi / 2
    if "y ≥ 0" in c:
        return 0, pi
    if "x ≥ 0" in c:
        return -pi / 2, pi / 2
    return 0, 2 * pi


def _polar_integral(body, r_lo, r_hi, angles):
    integrand = body.subs({x: r * sympy.cos(th), y: r * sympy.sin(th)}) * r
    return sympy.integrate(sympy.integrate(sympy.simplify(integrand), (r, r_lo, r_hi)), (th, *angles))


_REGION = re.compile(r"(?:(\d+) ≤ )?x² \+ y² ≤ (\d+)(?: with (.*))?")


def oracle(q) -> str:
    text = q.text
    m = re.fullmatch(
        r"Evaluate ∬_D (\d*)(.*?) dA, where D is (?:the disc|the region) (.*)\. Give your answer correct to 2 decimal places\.",
        text,
    )
    if m:
        body = T.parse_plain((m.group(1) or "1") + "*" + m.group(2), x=x, y=y)
        reg = _REGION.fullmatch(m.group(3))
        lo, hi = (int(reg.group(1)) if reg.group(1) else 0), int(reg.group(2))
        value = _polar_integral(body, sympy.sqrt(lo), sympy.sqrt(hi), _angles(reg.group(3)))
        return T.rnd(value, 2)
    m = re.fullmatch(
        r"Evaluate the iterated integral ∫ from x = (-?\d+) to (\d+) of ∫ from y = 0 to √\((\d+) - x²\) of (.*?) dy dx\. "
        r"Give your answer correct to 2 decimal places\.",
        text,
    )
    if m:
        a, b, n = int(m.group(1)), int(m.group(2)), int(m.group(3))
        body = T.parse_plain(m.group(4), x=x, y=y)
        f = sympy.lambdify((x, y), body, "mpmath")
        with mpmath.workdps(20):
            value = mpmath.quad(lambda xv: mpmath.quad(lambda yv: f(xv, yv), [0, mpmath.sqrt(n - xv * xv)]), [a, b])
        return T.rnd(sympy.Float(str(value), 20), 2)
    m = re.fullmatch(
        r"Evaluate ∬_D e\^\(-\(x² \+ y²\)\) dA, where D is (?:the disc|the half disc|the quarter disc) x² \+ y² ≤ (\d+)(?:, (.*))?\. "
        r"Give your answer correct to 3 decimal places\.",
        text,
    )
    if m:
        n = int(m.group(1))
        value = sympy.integrate(sympy.exp(-(r**2)) * r, (r, 0, sympy.sqrt(n))) * (
            _angles(m.group(2))[1] - _angles(m.group(2))[0]
        )
        return T.rnd(value, 3)
    m = re.fullmatch(
        r"Find the volume of the solid that lies above the xy-plane and below the surface z = (\d+) - (x² - y²|√\(x² \+ y²\))\. "
        r"Give your answer correct to 2 decimal places\.",
        text,
    )
    assert m, text
    k = int(m.group(1))
    height = k - (r**2 if "√" not in m.group(2) else r)
    radius = sympy.solve(height, r)[0]
    return T.rnd(sympy.integrate(sympy.integrate(height * r, (r, 0, radius)), (th, 0, 2 * pi)), 2)


def test_wiring():
    T.assert_wiring(SK, TOPIC, [EXAM])


@pytest.mark.parametrize("tier", TIERS)
def test_key_matches_the_independent_derivation(tier):
    bad = [s for s in range(80) if oracle(T.single(TOPIC, tier, s)) != T.single(TOPIC, tier, s).answer_text]
    assert bad == []


def test_tiers_use_their_shapes():
    def shape(tier, s):
        text = T.single(TOPIC, tier, s).text
        if "iterated" in text:
            return "iterated"
        if "e^(" in text:
            return "gaussian"
        if "volume" in text:
            return "volume"
        return "ring" if re.search(r"\d+ ≤ x²", text) else "angular" if re.search(r"∬_D \d?[xy]", text) else "radial"

    assert {shape("easy", s) for s in range(80)} == {"radial"}
    assert {"radial", "angular", "ring"} <= {shape("medium", s) for s in range(120)}
    assert {shape("hard", s) for s in range(80)} == {"iterated", "gaussian", "volume"}


@pytest.mark.parametrize("tier", TIERS)
def test_grading_and_diagnosis(tier):
    T.assert_single_grading(SK, TOPIC, tier)


@pytest.mark.parametrize("tier", TIERS)
def test_clean_text_variety_and_determinism(tier):
    T.assert_clean_variety_and_determinism(TOPIC, tier, distinct_prompts=20, distinct_answers=15)


# --- the exam chain --------------------------------------------------------------------------------------------


def exam_oracle(q):
    m = re.search(
        r"the part of the (?:ring|disc) (?:(\d+) ≤ )?x² \+ y² ≤ (\d+) that lies in the first quadrant", q.stem
    )
    lo, hi = sympy.sqrt(int(m.group(1) or 0)), sympy.sqrt(int(m.group(2)))
    quarter = (0, pi / 2)
    area = _polar_integral(sympy.Integer(1), lo, hi, quarter)
    moment = _polar_integral(x, lo, hi, quarter)
    xbar = moment / area
    ybar = _polar_integral(y, lo, hi, quarter) / area
    return dict(a=T.rnd(area, 2), b=T.rnd(moment, 2), c=T.rnd(xbar, 2), d=T.rnd(sympy.sqrt(xbar**2 + ybar**2), 2))


def test_exam_shape():
    T.assert_exam_shape(EXAM, ["a", "b", "c", "d"], [1, 2, 2, 1], [False, False, True, True])


def test_exam_keys_match_the_independent_derivation():
    for seed in range(80):
        q = T.exam(EXAM, seed)
        assert {p.label: p.answer_text for p in q.parts} == exam_oracle(q), seed


def test_exam_grading_distractors_and_notes():
    T.assert_exam_grading(SK, EXAM)


def test_follow_through_from_a_wrong_area_and_a_wrong_centroid():
    T.assert_follow_through_credited(SK, EXAM, minimum=100)


def test_exam_answer_space():
    T.assert_exam_variety(EXAM, seeds=range(200), distinct=15, repeat=30)
    assert len({T.exam(EXAM, s).stem for s in range(60)}) >= 30
