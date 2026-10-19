"""Tests `uc2-quadric-surfaces-level-sets` (topic `w7c_quadric`) and its exam chain `exd_x_w7c_quadric`.

The oracle reads each equation back out of the question TEXT and re-derives the key a different way from the generator:

* a surface with a centre is named by counting the real points where the lines through the centre, along the principal
  axes (eigenvectors of the Hessian), meet it: 6 for an ellipsoid, 4 for a hyperboloid of one sheet, 2 for two sheets,
  0 for no points, 3 for a cone or a single point; without a centre it is a paraboloid or a parabolic cylinder, told
  apart by the eigenvalue signs;
* a conic cross-section is named from its centre and the eigenvalue signs of its quadratic part;
* a semi-axis is the largest value of one coordinate on the cross-section; a radius is read from the completed square;
  an intercept is the root of f(t·eᵢ) = f(P).
"""

from __future__ import annotations

import re
from collections import Counter
from fractions import Fraction

import pytest
import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

import mathgen.generators.w7c_quadric_build  # noqa: F401  (registers the topics)
from mathgen import make
from mathgen.core import mcq_options
from tests import w6c_support as support

TOPIC = "w7c_quadric"
EXAM = "exd_x_w7c_quadric"
SKILL = support.skill_named("uc2-quadric-surfaces-level-sets")
X, Y, Z = sympy.symbols("x y z")
VARS = (X, Y, Z)
BY_NAME = {"x": X, "y": Y, "z": Z}
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)
_TERM = re.compile(r"^[−-]?[0-9xyz²/]+$")
_OPS = ("+", "−")


# -- reading the text back --------------------------------------------------------------------------------------


def _expr(text):
    return parse_expr(text.replace("−", "-").replace("²", "**2"), local_dict=BY_NAME, transformations=_TRANSFORMS)


def equations(text):
    """Every ``left = right`` in the text, as (left, right) strings of alternating terms and + / − signs. A side that is
    prose or a function call (``f(x, y, z)``) comes out empty."""
    tokens = text.split()
    found = []
    for i, token in enumerate(tokens):
        if token != "=":
            continue
        left, j, want_term = [], i - 1, True
        while j >= 0:
            t = tokens[j]
            if want_term and _TERM.match(t) and t not in ("-", "−"):
                left.append(t)
                want_term = False
            elif not want_term and t in _OPS:
                left.append(t)
                want_term = True
            else:
                break
            j -= 1
        if left and left[0] in _OPS:
            left = left[1:]
        right, j, want_term, ended = [], i + 1, True, False
        while j < len(tokens) and not ended:
            t = tokens[j]
            core = t.rstrip(".,?")
            ended = core != t
            if want_term and _TERM.match(core) and core not in ("-", "−"):
                right.append(core)
                want_term = False
            elif not want_term and t in _OPS:
                right.append(t)
                want_term = True
            else:
                break
            j += 1
        if right and right[-1] in _OPS:
            right = right[:-1]
        found.append((" ".join(reversed(left)), " ".join(right)))
    return found


def polys(text):
    """The polynomials left − right of every equation in the text that has both sides."""
    return [_expr(left) - _expr(right) for left, right in equations(text) if left and right]


def degree(p):
    return sympy.Poly(sympy.expand(p), *VARS).total_degree()


def the_surface(text):
    surfaces = [p for p in polys(text) if degree(p) == 2]
    assert len(surfaces) == 1, (text, surfaces)
    return sympy.expand(surfaces[0])


def the_plane(text):
    planes = [p for p in polys(text) if degree(p) == 1]
    assert len(planes) == 1, (text, planes)
    return planes[0]


def coordinate_plane(text):
    """(variable, value) of a plane ``z = 3`` in the text."""
    match = re.search(r"\b([xyz]) = (−?\d+)(?![\w²/])", text)
    return BY_NAME[match.group(1)], int(match.group(2).replace("−", "-"))


# -- independent oracles ----------------------------------------------------------------------------------------


def _numbers(matrix):
    return [complex(sympy.N(v)).real for v in matrix.eigenvals(multiple=True)]


def oracle_name(f):
    h = sympy.hessian(f, VARS)
    g = sympy.Matrix([sympy.diff(f, v).subs({X: 0, Y: 0, Z: 0}) for v in VARS])
    eig = _numbers(h)
    pos, neg = sum(v > 1e-9 for v in eig), sum(v < -1e-9 for v in eig)
    rank = pos + neg
    solution = sympy.linsolve((h, -g), *VARS)
    if not solution:  # no centre
        return (
            "parabolic cylinder" if rank == 1 else "elliptic paraboloid" if pos * neg == 0 else "hyperbolic paraboloid"
        )
    centre = [s.subs({free: 0 for free in s.free_symbols}) for s in list(solution)[0]]
    if rank == 2:
        return "hyperbolic cylinder" if pos * neg else "elliptic cylinder"
    total = 0
    t = sympy.Symbol("t")
    for _eigenvalue, _mult, vectors in h.eigenvects():
        for vector in vectors:
            along = sympy.expand(f.subs({v: c + t * w for v, c, w in zip(VARS, centre, vector)}, simultaneous=True))
            a, b, d = (complex(sympy.N(along.coeff(t, k))).real for k in (2, 1, 0))
            assert abs(b) < 1e-9
            if abs(d) < 1e-9:
                total += 1
            elif -d / a > 0:
                total += 2
    if total == 3:
        return "cone" if pos * neg else "single point"
    return {6: "ellipsoid", 4: "hyperboloid of one sheet", 2: "hyperboloid of two sheets", 0: "empty set"}[total]


def oracle_conic(expr, s, t):
    poly = sympy.Poly(sympy.expand(expr), s, t)
    a, b, c = (poly.coeff_monomial(m) for m in (s * s, s * t, t * t))
    d, e, f0 = poly.coeff_monomial(s), poly.coeff_monomial(t), poly.coeff_monomial(1)
    m = sympy.Matrix([[a, b / 2], [b / 2, c]])
    lin = sympy.Matrix([d, e])
    if m.det() != 0:
        centre = -m.inv() * lin / 2
        value = -(f0 + (lin.T * centre)[0] / 2)  # (x − centre)ᵀ M (x − centre) = value
        eig = _numbers(m)
        if eig[0] * eig[1] > 0:
            if value == 0:
                return "single point"
            if value * eig[0] < 0:
                return "empty set"
            return "circle" if (b == 0 and a == c) else "ellipse"
        return "pair of intersecting lines" if value == 0 else "hyperbola"
    null = m.nullspace()[0]
    return "parabola" if (lin.T * null)[0] != 0 else "parallel lines"


def level_name(f, c):
    """The level set f = c of a diagonal quadratic form, from the coefficients' signs against the sign of c."""
    coeffs = [sympy.Poly(f, *VARS).coeff_monomial(v * v) for v in VARS]
    if c == 0:
        return "cone" if {co > 0 for co in coeffs} == {True, False} else "single point"
    return {3: "ellipsoid", 2: "hyperboloid of one sheet", 1: "hyperboloid of two sheets", 0: "empty set"}[
        sum(bool(co * c > 0) for co in coeffs)
    ]


def kind_of(q):
    t = q.text
    if "level set" in t or "set of points where" in t:
        return "level"
    if "coordinate" in t:
        return "centre"
    if "semi-axis" in t:
        return "axis"
    if "level surface" in t:
        return "intercept"
    if "radius" in t:
        return "radius"
    if "plane" in t:
        plane = the_plane(t)
        return "trace" if len(plane.free_symbols) == 1 else "oblique"
    return "name"


def check_question(q, seed=None):
    t = q.text
    kind = kind_of(q)
    if kind == "name":
        assert oracle_name(the_surface(t)) == q.answer_text, (seed, t)
    elif kind == "level":
        sides = [_expr(r) for left, r in equations(t) if not left and r]
        f = next(e for e in sides if e.free_symbols)
        c = next(e for e in sides if not e.free_symbols)
        assert level_name(f, c) == q.answer_text, (seed, t)
    elif kind == "trace":
        var, k = coordinate_plane(t)
        s, u = [v for v in VARS if v != var]
        assert oracle_conic(the_surface(t).subs(var, k), s, u) == q.answer_text, (seed, t)
    elif kind == "centre":
        f = the_surface(t)
        var = BY_NAME[
            re.search(r"the ([xyz])-coordinate|its ([xyz])-coordinate", t).group(1)
            or re.search(r"its ([xyz])-coordinate", t).group(1)
        ]
        h = sympy.hessian(f, VARS)
        g = sympy.Matrix([sympy.diff(f, v).subs({X: 0, Y: 0, Z: 0}) for v in VARS])
        centre = -h.inv() * g
        assert centre[VARS.index(var)] == sympy.Rational(q.answer_text), (seed, t)
    elif kind == "axis":
        var = BY_NAME[re.search(r"the ([xyz])(?: direction|-axis)", t).group(1)]
        plane_var, k = coordinate_plane(t)
        curve = sympy.expand(the_surface(t).subs(plane_var, k))
        other = next(v for v in VARS if v not in (plane_var, var))
        assert oracle_conic(curve, var, other) in ("ellipse", "circle")
        assert max(abs(r) for r in sympy.solve(curve.subs(other, 0), var)) == sympy.Rational(q.answer_text), (seed, t)
    elif kind == "oblique":
        plane, surface = the_plane(t), the_surface(t)
        var = sorted(plane.free_symbols, key=str)[0]
        curve = sympy.expand(surface.subs(var, sympy.solve(plane, var)[0]))
        assert oracle_conic(curve, *[v for v in VARS if v != var]) == q.answer_text, (seed, t)
    elif kind == "radius":
        plane, para = the_plane(t), the_surface(t)
        w = next(v for v in VARS if sympy.degree(para, v) == 1)
        curve = sympy.expand(para.subs(w, sympy.solve(plane, w)[0]))
        u, v = [x for x in VARS if x != w]
        assert oracle_conic(curve, u, v) == "circle"
        centre = sympy.solve([sympy.diff(curve, u), sympy.diff(curve, v)], [u, v])
        lead = sympy.Poly(curve, u, v).coeff_monomial(u * u)
        assert sympy.sqrt(-curve.subs(centre) / lead) == sympy.Rational(q.answer_text), (seed, t)
    else:
        f = next(_expr(r) for left, r in equations(t) if not left and r and _expr(r).free_symbols)
        point = [int(v.replace("−", "-")) for v in re.search(r"P = \(([^)]*)\)", t).group(1).split(", ")]
        axis = BY_NAME[re.search(r"(?:meets|crosses) the ([xyz])-axis", t).group(1)]
        c = f.subs(dict(zip(VARS, point)))
        root = sympy.Symbol("root", real=True)
        along = f.subs({v: (root if v == axis else 0) for v in VARS}) - c
        assert max(sympy.solve(along, root)) == sympy.Rational(q.answer_text), (seed, t)
        assert abs(point[VARS.index(axis)]) != sympy.Rational(q.answer_text)
    return kind


# -- wiring and generic checks ----------------------------------------------------------------------------------


def test_skill_wiring():
    support.check_wiring(SKILL, TOPIC, EXAM, ["ucv1-conic-sections-features"])
    assert SKILL.domain == "multivariable-and-vector-calculus" and SKILL.placement_eligible is False


@pytest.mark.parametrize("tier", support.TIERS)
def test_key_accepted_and_every_distractor_rejected(tier):
    support.check_tier(SKILL, TOPIC, tier, seeds=120)


@pytest.mark.parametrize("tier", support.TIERS)
def test_no_python_leakage_and_no_repeated_option(tier):
    for seed in range(150):
        q = make(TOPIC, seed=seed, difficulty=tier)
        shown = " ".join([q.text, q.answer_text, *q.distractors])
        assert not re.search(r"\bNone\b|\bnan\b|\*\*|sqrt|Rational|\boo\b|\bI\b", shown), (seed, shown)
        assert len({q.answer_text, *q.distractors}) == len(q.distractors) + 1


def test_the_reader_of_equations_is_sound():
    assert equations("Name the quadric surface −y²/16 − z²/36 = −1.") == [("−y²/16 − z²/36", "−1")]
    assert equations("What type of surface is given by x²/4 + y² = z?") == [("x²/4 + y²", "z")]
    assert equations("Let f(x, y, z) = 2x² − y². What is f = 4?") == [("", "2x² − y²"), ("", "4")]
    assert polys("The plane 5y = 7x − 5 is cut by x² + y² − z² = 0.")[0] == 5 * Y - 7 * X + 5


# -- keys against the oracle ------------------------------------------------------------------------------------


def test_easy_keys_match_the_oracle():
    kinds = Counter(check_question(make(TOPIC, seed=s, difficulty="easy"), s) for s in range(250))
    assert set(kinds) == {"name", "level", "trace"}, kinds
    assert min(kinds.values()) >= 25


def test_easy_names_every_type():
    names = {make(TOPIC, seed=s, difficulty="easy").answer_text for s in range(400)}
    assert {
        "ellipsoid",
        "hyperboloid of one sheet",
        "hyperboloid of two sheets",
        "cone",
        "elliptic paraboloid",
        "hyperbolic paraboloid",
        "elliptic cylinder",
        "hyperbolic cylinder",
        "parabolic cylinder",
        "circle",
        "ellipse",
        "hyperbola",
        "parabola",
        "single point",
        "empty set",
    } <= names


def test_medium_keys_match_the_oracle():
    kinds = Counter(check_question(make(TOPIC, seed=s, difficulty="medium"), s) for s in range(300))
    assert set(kinds) == {"name", "centre", "axis", "trace"}, kinds
    assert min(kinds.values()) >= 25


def test_medium_traces_are_the_degenerate_ones_and_names_meet_their_traps():
    degenerate, forgot = Counter(), 0
    for seed in range(400):
        q = make(TOPIC, seed=seed, difficulty="medium")
        if kind_of(q) == "trace":
            degenerate[q.answer_text] += 1
        forgot += "quad-forgot-completing-constant" in set(q.option_notes.values())
    assert set(degenerate) <= {"single point", "empty set", "pair of intersecting lines"}
    assert len(degenerate) == 3, degenerate
    assert forgot >= 5


def test_hard_keys_match_the_oracle():
    kinds = Counter(check_question(make(TOPIC, seed=s, difficulty="hard"), s) for s in range(300))
    assert set(kinds) == {"oblique", "radius", "intercept", "name"}, kinds
    assert min(kinds.values()) >= 25


def test_hard_names_are_the_cross_term_traps():
    names = Counter()
    for seed in range(300):
        q = make(TOPIC, seed=seed, difficulty="hard")
        if kind_of(q) == "name":
            names[q.answer_text] += 1
            assert "quad-cross-term-ignored" in set(q.option_notes.values()), (seed, q.text)
            assert re.search(r"[xyz][xyz]", q.text.split("?")[0].split(" is ")[-1]), q.text
    assert {"hyperboloid of one sheet", "hyperboloid of two sheets", "hyperbolic paraboloid"} <= set(names), names


def test_hard_oblique_covers_ellipse_parabola_hyperbola():
    answers = Counter(
        q.answer_text for q in (make(TOPIC, seed=s, difficulty="hard") for s in range(400)) if kind_of(q) == "oblique"
    )
    assert set(answers) == {"ellipse", "parabola", "hyperbola"}
    assert min(answers.values()) >= 15, answers


# -- the answer space, the options and the tiers ----------------------------------------------------------------


@pytest.mark.parametrize("tier", support.TIERS)
def test_answer_space_is_wide(tier):
    support.check_answer_space(TOPIC, tier, seeds=300, distinct=8, share=0.3)


def test_tier_shapes():
    support.check_tier_shapes_differ(TOPIC, seeds=150)


@pytest.mark.parametrize("tier", support.TIERS)
def test_the_key_is_not_given_away_by_position_or_length(tier):
    position, longest, n = Counter(), 0, 300
    for seed in range(n):
        q = make(TOPIC, seed=seed, difficulty=tier)
        options = mcq_options(q)
        position[options.index(q.answer_text)] += 1
        longest += (
            len(q.answer_text) == max(map(len, options)) and sum(len(o) == len(q.answer_text) for o in options) == 1
        )
    assert max(position.values()) <= 0.4 * n, position
    assert longest <= 0.4 * n, longest


def test_difficulty_is_reasoning_not_bigger_numbers():
    texts = {t: [make(TOPIC, seed=s, difficulty=t).text for s in range(150)] for t in support.TIERS}
    # xy terms and oblique planes only ever appear at the hard tier
    cross = re.compile(r"[0-9]?[xyz][xyz]\b")
    assert not any(cross.search(x.split("?")[0]) for x in texts["easy"] + texts["medium"])
    assert any(cross.search(x.split("?")[0]) for x in texts["hard"])
    oblique = re.compile(r"plane \d?[xyz] = [−0-9]*[xyz]")
    assert not any(oblique.search(x) for x in texts["easy"] + texts["medium"])
    assert any(oblique.search(x) for x in texts["hard"])
    # completing the square (linear terms next to squares) first appears at medium
    linear = re.compile(r"[xyz]² [+−] \d*[xyz]\b")
    assert not any(linear.search(x) for x in texts["easy"])
    assert any(linear.search(x) for x in texts["medium"])


# -- exam -------------------------------------------------------------------------------------------------------


def test_exam_shape_and_marks():
    q = make(EXAM, seed=3, difficulty="exam")
    assert [p.answer_kind for p in q.parts] == ["mcq", "number", "number", "number"]
    assert [p.follow_through for p in q.parts] == [None, None, None, "part_b*part_c"]
    assert [p.marks for p in q.parts] == [2, 2, 2, 3]


def test_exam_keys_match_independent_derivation():
    types = Counter()
    for seed in range(150):
        q = make(EXAM, seed=seed, difficulty="exam")
        f = the_surface(q.stem)
        assert oracle_name(f) == q.parts[0].answer_text
        types[q.parts[0].answer_text] += 1
        plane_var, k = coordinate_plane(q.stem)
        curve = sympy.expand(f.subs(plane_var, k))
        first, second = [v for v in VARS if v != plane_var]
        assert oracle_conic(curve, first, second) == "ellipse"
        for part, var, other in ((q.parts[1], first, second), (q.parts[2], second, first)):
            assert f"{var} direction" in part.prompt
            assert max(abs(r) for r in sympy.solve(curve.subs(other, 0), var)) == sympy.Rational(part.answer_text), (
                seed,
                q.stem,
            )
        assert sympy.Rational(q.parts[3].answer_text) == sympy.Rational(q.parts[1].answer_text) * sympy.Rational(
            q.parts[2].answer_text
        )
    assert set(types) == {"ellipsoid", "hyperboloid of one sheet", "hyperboloid of two sheets"}


def test_exam_grading_and_follow_through():
    support.check_exam_grading(SKILL, EXAM, [2, 2, 2, 3], seeds=80)
    support.check_follow_through(SKILL, EXAM)


def test_exam_answer_space():
    questions = [make(EXAM, seed=s, difficulty="exam") for s in range(200)]
    assert len({q.answer_text for q in questions}) >= 60
    assert len({Fraction(q.parts[3].answer_text) for q in questions}) >= 25
