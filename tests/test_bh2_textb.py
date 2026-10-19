"""Regression tests for the 2026-10-19 bug hunt, text and key defects in the algebra, geometry, calculus,
vector and finance skills (worker textb)."""

import math
import re

import pytest

from mathgen import engine
from mathgen.core import display_question

_ERRS, _SKILLS = engine.load_skills()
SKILLS = {s.id: s for s in _SKILLS}


def _make(skill_id, seed, source):
    return engine.make_for_skill(SKILLS[skill_id], seed, source=source)


def _shown(skill_id, seed, source):
    q = _make(skill_id, seed, source)
    d = display_question(q)
    return q, d


def _all_text(d):
    out = [d["text"], d["answer_text"], *d["options"], *d["working"]]
    for p in d.get("parts", ()):
        out += [p["text"], p["answer_text"], *p["options"], *p["working"]]
    return out


# --- A1: the cosine/sine ratio angle question asked for one angle and keyed another ------------------------

COS_SIN = "senior-apply-the-cosine-and-sine-ratios-to-find"


@pytest.mark.parametrize("seed", range(80))
def test_angle_from_sides_key_matches_the_angle_asked(seed):
    q = _make(COS_SIN, seed, 1)
    m = re.search(r"an (opposite|adjacent) side of (\d+) units and a hypotenuse of (\d+) units", q.text)
    direction, leg, hyp = m.group(1), int(m.group(2)), int(m.group(3))
    other = math.sqrt(hyp**2 - leg**2)  # Pythagoras, no trig function shared with the generator
    if direction == "opposite":
        # "the acute angle that the opposite side is opposite to"
        assert "that the opposite side is opposite to" in q.text
        expected = math.degrees(math.atan2(leg, other))
    else:
        # the vertex where the adjacent side meets the hypotenuse: tan(angle) = other / leg
        assert "where the adjacent side meets the hypotenuse" in q.text
        expected = math.degrees(math.atan2(other, leg))
    assert q.answer_text == str(round(expected))
    assert engine.grade(SKILLS[COS_SIN], q, q.answer_text).correct
    for d in q.distractors:
        assert d != q.answer_text


# --- complex number stem dropped the sign of a negative imaginary part (wrong key) ---------------------------

COMPLEX_POINT = "senior-consider-complex-numbers-as-points-in-a-plane"


@pytest.mark.parametrize("seed", range(150))
def test_complex_point_stem_matches_key(seed):
    q = _make(COMPLEX_POINT, seed, 0)
    m = re.search(r"complex number (\S.*?) is represented", q.text)
    z = m.group(1).replace(" ", "").replace("-", "+-")
    parts = [p for p in z.split("+") if p]
    re_part, im_part = 0, 0
    for p in parts:
        if p.endswith("i"):
            body = p[:-1]
            im_part = int(body) if body not in ("", "-") else (-1 if body == "-" else 1)
        else:
            re_part = int(p)
    assert q.answer_text == f"x = {re_part}, y = {im_part}"


# --- display text -----------------------------------------------------------------------------------------

DISPLAY_CASES = [
    ("Find ∫ 1 cos(4x - 2) dx", "Find ∫ cos(4x - 2) dx"),
    ("h(x) = 2f(x) + 1g(x)", "h(x) = 2f(x) + g(x)"),
    ("y = -1x^2 - 1x - 4", "y = -x^2 - x - 4"),
    ("(1 − 1i)", "(1 − i)"),
    ("f(t) = 1e^(-1t)", "f(t) = e^(-t)"),
    ("x² + 0x + 1", "x² + 1"),
    ("shortened by 1 days", "shortened by 1 day"),
    ("n - 1 edges", "n - 1 edges"),
    ("(x - -23) / (x - -5)", "(x - (-23)) / (x - (-5))"),
    ("2 × -5", "2 × (-5)"),
    ("x = +-2", "x = ±2"),
    ("Valid -- each", "Valid – each"),
    ("x = 9/-1", "x = 9/(-1)"),
    ("20 x 12 = 240", "20 × 12 = 240"),
    ("the 1h layover, 21x, 1.1x", "the 1h layover, 21x, 1.1x"),
]


@pytest.mark.parametrize(("raw", "shown"), DISPLAY_CASES)
def test_display_text_tidies_notation(raw, shown):
    from mathgen.core import _display_text

    assert _display_text(raw) == shown


# --- generator text ---------------------------------------------------------------------------------------

STEMS = [
    ("definite-integral", 0),
    ("definite-integral", 1),
    ("definite-integral", 2),
    ("integration-by-substitution", 0),
    ("solve-quadratics-formula", 2),
    ("common-difference", 3),
    ("nth-term-rule", 3),
    ("inequality-solutions-number-line", 2),
    ("senior-graph-dilation-read-factor", 2),
    ("senior-solve-equations-involving-exponential-functions-using-technology-and", 2),
    ("ucv1-sequence-limits-squeeze-monotone", 1),
    ("variables-on-both-sides", 0),
    ("one-step-equations", 1),
    ("function-notation", 0),
    ("hybrid-functions", 3),
    ("f10-polygon-angle-sums", 2),
    ("f10-describe-transformation-from-points", 2),
    ("power-functions-rational-index", 0),
    ("senior-develop-a-linear-formula-from-a-word-description", 0),
    ("senior-graph-dilation-read-factor", 1),
    ("senior-use-partial-fractions-where-necessary-for-integration-in", 3),
    ("senior-integrate-by-parts", 3),
    ("senior-with-the-aid-of-a-financial-calculator-or", 3),
    ("senior-use-factorial-notation", 2),
    ("convex-quadratic-minimum", 0),
    ("compare-unit-prices", 0),
    ("compare-unit-prices", 1),
    ("riemann-sum-to-definite-integral", 1),
    ("senior-use-forward-and-backward-scanning-to-determine-the", 3),
    ("senior-use-the-critical-path-to-determine-the-minimum", 3),
    ("senior-define-and-use-unit-vectors-and-the-perpendicular", 0),
    ("ucv1-trig-substitution-integrals", 3),
    ("differentiate-quotient-rule", 3),
    ("senior-estimate-the-limit-of-frac-a-h-1", 3),
    ("vector-proof-geometry", 1),
    ("locus-argument-ray-wedge", 1),
    ("curve-features-from-derivative-signs", 0),
    ("senior-integrate-using-the-trigonometric-identities-mathrm-s-mathrm", 3),
    ("senior-solve-simple-polynomial-equations", 1),
    ("add-complex-numbers", 0),
    (COMPLEX_POINT, 0),
    (COMPLEX_POINT, 2),
]

BAD = [
    (r"\{'[pq]'", "dict printed"),
    (r"\(…\)|\(boundary", "placeholder"),
    (r"(?<![\w.^/])(?<![-+−] )1[xyztnuid](?![A-Za-z0-9_])", "coefficient 1"),
    (r"(?<![\w.^/])-1[xyztnu]", "coefficient -1"),
    (r"\b1 (days|hours|parts|times|points)\b", "singular count with plural noun"),
    (r"\d-th root", "ordinal root"),
    (r"\$[A-Za-z]\$", "latex dollars"),
    (r"\+-|\+/-", "plus-minus spelled out"),
    (r"\bA (octagon|isosceles)", "article"),
    (r"then the (reflected|rotated)", "adjective for noun"),
    (r"'s's", "double possessive"),
    (r"\d/-\d", "unbracketed negative divisor"),
    (r"\d/\d+/\d", "nested fraction"),
    (r"day\(s\)", "day(s)"),
    (r"\$\d+/\d+", "money as fraction"),
    (r" -- ", "double hyphen"),
]


def _strings(d):
    out = [d["text"], d["answer_text"], *d["options"], *d["working"]]
    for p in d.get("parts", ()):
        out += [p["text"], p["answer_text"], *p["options"], *p["working"]]
    return out


@pytest.mark.parametrize(("skill_id", "source"), STEMS)
def test_learner_text_has_no_notation_leaks(skill_id, source):
    for seed in range(12):
        q = _make(skill_id, seed, source)
        for text in _strings(display_question(q)):
            for pattern, what in BAD:
                assert not re.search(pattern, text), f"{what} in {skill_id}/{source}/{seed}: {text!r}"
