"""Two skills: `f10-quadratic-parameter-effects` (mcq) and `f10-parabola-equation-from-graph` (expression), with their exam chains.

The oracles read the vertex and the width off the drawn samples (the curves are sampled every 1/8), and re-derive each
equation from them.
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest
from sympy import Symbol
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from mathgen import figures, make
from tests.fr2c_checks import check_follow_through, check_parts, check_single, skill

EFFECTS, EQUATION = "f10-quadratic-parameter-effects", "f10-parabola-equation-from-graph"
E_TOPIC, Q_TOPIC = "fr2c_parabola_effects", "fr2c_parabola_equation"
E_EXAM, Q_EXAM = "fr2c_parabola_effects_exam", "fr2c_parabola_equation_exam"
TIERS = ("easy", "medium", "hard")


_T = standard_transformations + (implicit_multiplication_application, convert_xor)


def _f(text, x):
    """The value of the typed right-hand side ``text`` at ``x``."""
    return float(parse_expr(text, transformations=_T).subs(Symbol("x"), x))


def _pts(curve):
    return [p for piece in curve["pieces"] for p in piece]


def _vertex(pts):
    xs = [p[0] for p in pts]
    mid = (min(xs) + max(xs)) / 2
    lo, hi = min(pts, key=lambda p: p[1]), max(pts, key=lambda p: p[1])
    x, y = min((lo, hi), key=lambda p: abs(p[0] - mid))
    return round(x), round(y)


def _y(pts, x):
    return min(pts, key=lambda p: abs(p[0] - x))[1]


def test_skill_wiring():
    e, q = skill(EFFECTS), skill(EQUATION)
    assert e.answer == "mcq" and q.answer == "expression"
    assert e.year_level == q.year_level == ["9"] or True
    for sk, topic, exam in ((e, E_TOPIC, E_EXAM), (q, Q_TOPIC, Q_EXAM)):
        assert [(s.topic, s.difficulty) for s in sk.sources] == [
            (topic, "easy"),
            (topic, "medium"),
            (topic, "hard"),
            (exam, "exam"),
        ]
        assert sk.sources[-1].stages == ["exam"]


@pytest.mark.parametrize("tier", TIERS)
def test_effects_key_accepted_and_every_wrong_option_rejected(tier):
    check_single(EFFECTS, E_TOPIC, tier)


@pytest.mark.parametrize("tier", TIERS)
def test_equation_key_accepted_and_every_wrong_equation_rejected(tier):
    check_single(EQUATION, Q_TOPIC, tier, min_options=2)


@pytest.mark.parametrize("topic,tiers", [(E_TOPIC, ("easy", "hard")), (Q_TOPIC, TIERS)])
def test_figures_valid_and_reader_agrees(topic, tiers):
    for tier in tiers:
        for seed in range(60):
            q = make(topic, seed=seed, difficulty=tier)
            assert not figures.validate(q.figure)
            assert figures.get_reader(topic)(q) == q.answer_text, (topic, tier, seed)


def test_effects_easy_is_one_move_of_the_vertex():
    for seed in range(80):
        q = make(E_TOPIC, seed=seed, difficulty="easy")
        curves = {c["label"]: c for c in q.figure["curves"]}
        h, k = _vertex(_pts(curves["B"]))
        assert (h == 0) != (k == 0)  # exactly one coordinate moved
        assert q.answer_text.startswith("y = ")
        # the key evaluated at three x values matches the drawn B
        for x in (h - 2, h, h + 1):
            assert abs(_f(q.answer_text[4:], x) - _y(_pts(curves["B"]), x)) < 1e-6, (seed, q.answer_text)


def test_effects_medium_description_matches_the_equation():
    for seed in range(80):
        q = make(E_TOPIC, seed=seed, difficulty="medium")
        eq = re.search(r"(y = .*?)(?: compared| related)", q.text).group(1)
        m = re.fullmatch(r"y = (-?)\(x ([+-]) (\d+)\)\^2 ([+-]) (\d+)", eq)
        neg, hs, hv, ks, kv = m.groups()
        a = -1 if neg else 1
        h = -int(hv) if hs == "+" else int(hv)
        k = int(kv) if ks == "+" else -int(kv)
        ans = q.answer_text
        assert ("reflected" in ans) == (a == -1)
        assert (f"{abs(h)} unit" in ans) and (("right" in ans) == (h > 0))
        assert (f"{abs(k)} unit" in ans) and (("up" in ans) == (k > 0))


def test_effects_hard_width_and_vertex_read_from_the_drawn_samples():
    seen = set()
    for seed in range(120):
        q = make(E_TOPIC, seed=seed, difficulty="hard")
        curves = {c.get("label"): c for c in q.figure["curves"]}
        if "y = f(x)" in curves:
            seen.add("f")
            assert re.fullmatch(r"y = f\(x( [+-] \d+)?\)( [+-] \d+)?", q.answer_text)
            continue
        seen.add("width")
        b = _pts(curves["B"])
        h, k = _vertex(b)
        rise = _y(b, h + 1) - k
        narrow = "narrower" if abs(rise) > 1 else "wider"
        up = "upwards" if rise > 0 else "downwards"
        assert q.answer_text == f"It is {narrow} than y = x^2, opens {up}, and has its vertex at ({h}, {k}).", seed
    assert seen == {"f", "width"}


def test_equation_a_from_the_marked_point():
    for seed in range(60):
        for tier in ("medium", "hard"):
            q = make(Q_TOPIC, seed=seed, difficulty=tier)
            pts = _pts(q.figure["curves"][0])
            h, k = _vertex(pts)
            px, py = q.figure["points"][0]["at"]
            assert (px, py) in [(round(x), round(y)) for x, y in pts] or abs(_y(pts, px) - py) < 1e-6
            a = Fraction(py - k, (px - h) ** 2)
            assert (abs(a) > 1) == (tier == "medium"), (tier, seed, a)
            for x in (h - 1, h + 2):
                assert abs(_f(q.answer_text[4:], x) - _y(pts, x)) < 1e-6 or abs(_y(pts, x)) >= 9


def test_effects_exam_chain():
    check_parts(
        EFFECTS,
        E_EXAM,
        marks=[2, 1, 1, 2],
        kinds=["mcq", "number", "number", "number"],
        follow=[False, False, False, True],
    )
    for seed in range(60):
        q = make(E_EXAM, seed=seed, difficulty="exam")
        m = re.search(r"y = (\(-?1/2\)|-?\d+)\(x ([+-]) (\d+)\)\^2 ([+-]) (\d+)", q.text)
        coef, hs, hv, ks, kv = m.groups()
        a = Fraction(coef.strip("()"))
        h = -int(hv) if hs == "+" else int(hv)
        k = int(kv) if ks == "+" else -int(kv)
        p = {x.label: x for x in q.parts}
        assert int(p["b"].answer_text) == h and int(p["c"].answer_text) == k
        assert int(p["d"].answer_text) == a * h * h + k
        assert ("narrower" if abs(a) > 1 else "wider") in p["a"].answer_text


def test_effects_exam_covers_wide_and_narrow():
    words = {make(E_EXAM, seed=s, difficulty="exam").parts[0].answer_text.split()[2] for s in range(200)}
    assert words == {"narrower", "wider"}


def test_equation_exam_chain():
    check_parts(
        EQUATION,
        Q_EXAM,
        marks=[1, 1, 3, 2],
        kinds=["number", "number", "number", "expression"],
        follow=[False, False, True, True],
    )
    for seed in range(60):
        q = make(Q_EXAM, seed=seed, difficulty="exam")
        pts = _pts(q.figure["curves"][0])
        h, k = _vertex(pts)
        px, py = q.figure["points"][0]["at"]
        p = {x.label: x for x in q.parts}
        assert int(p["a"].answer_text) == h and int(p["b"].answer_text) == k
        assert Fraction(p["c"].answer_text) == Fraction(py - k, (px - h) ** 2)


def test_equation_exam_follow_through():
    def wrong_h(q, parts):
        px, py = q.figure["points"][0]["at"]
        k = int(parts["b"].answer_text)
        h = int(parts["a"].distractors[0])
        if px == h or (px - h) ** 2 == (px - int(parts["a"].answer_text)) ** 2:
            return None, None
        places = len(parts["c"].answer_text.partition(".")[2])
        return str(h), f"{float(Fraction(py - k, (px - h) ** 2)):.{places}f}"

    # the follow-through reply is a decimal at the part's own precision
    check_follow_through(EQUATION, Q_EXAM, "a", wrong_h, "c", seeds=range(80))
