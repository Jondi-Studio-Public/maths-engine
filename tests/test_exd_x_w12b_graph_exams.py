"""Tests the five straight-line graph exam chains (`exd_x_w12b_*`) against an oracle that reads only the figure and
the numbers the question states, never the generator's own answer."""

from __future__ import annotations

import math
import re
from fractions import Fraction as F

import pytest
import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from mathgen import figures, make
from mathgen.generators.pv1a_line_figure_build import read_intersection, read_line
from tests.exd_x_r2c_support import (
    canonical,
    check_answer_space,
    check_clean_and_deterministic,
    check_follow_through,
    check_grading,
    check_shapes,
    check_wired,
    exam,
)

X = sympy.Symbol("x")
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)

CHAINS = {
    "plot-linear-points": ("exd_x_w12b_plot_linear_points", "fg1_plotted_points"),
    "graph-y-mx": ("exd_x_w12b_graph_y_mx", "fg1_proportional_graph"),
    "graph-y-mx-plus-b": ("exd_x_w12b_graph_y_mx_plus_b", "fg1_line_table"),
    "gradient-from-graph": ("exd_x_w12b_gradient_from_graph", "fg1_gradient_from_graph"),
    "equation-from-graph": ("exd_x_w12b_equation_from_graph", "fg1_equation_from_graph"),
}
SHAPES = {
    "plot-linear-points": ("pair", "fraction", "fraction", "fraction"),
    "graph-y-mx": ("fraction", "expression", "fraction", "fraction"),
    "graph-y-mx-plus-b": ("fraction", "fraction", "expression", "fraction"),
    "gradient-from-graph": ("fraction", "expression", "pair", "number"),
    "equation-from-graph": ("expression", "expression", "fraction", "fraction"),
}
MARKS = {
    "plot-linear-points": [1, 2, 2, 2],
    "graph-y-mx": [2, 2, 1, 2],
    "graph-y-mx-plus-b": [1, 1, 2, 2],
    "gradient-from-graph": [2, 2, 1, 2],
    "equation-from-graph": [2, 2, 2, 1],
}
ids = list(CHAINS)


def _frac(text: str) -> F:
    return F(text.replace(" ", ""))


def _eq(text: str):
    """The right-hand side of ``y = ...`` as a SymPy expression."""
    return parse_expr(text.split("=")[1].replace("^", "**"), transformations=_TRANSFORMS, local_dict={"x": X})


def _same_eq(text: str, m: F, c: F) -> bool:
    return (
        sympy.expand(
            _eq(text) - (sympy.Rational(m.numerator, m.denominator) * X + sympy.Rational(c.numerator, c.denominator))
        )
        == 0
    )


def _pt(text: str):
    a, b = re.fullmatch(r"\((-?[\d/]+), (-?[\d/]+)\)", text).groups()
    return _frac(a), _frac(b)


def _stem_int(q, pattern: str) -> int:
    found = re.findall(pattern, " ".join(p.prompt for p in q.parts) + " " + q.stem)
    assert len(found) == 1, (pattern, q.stem)
    return int(found[0])


def _on_line(fig, m: F, c: F, labels=None):
    for p in fig["points"]:
        x, y = p["at"]
        assert y == m * x + c, (p, m, c)


def check_plot_linear_points(q):
    pts = {p["label"]: tuple(F(v) for v in p["at"]) for p in q.figure["points"]}
    (xa, ya), (xb, yb), (xc, yc) = pts["A"], pts["B"], pts["C"]
    m = (yc - ya) / (xc - xa)
    assert (yb - ya) == m * (xb - xa)
    x_far = _stem_int(q, r"where x = (-?\d+)")
    y_at = _stem_int(q, r"where y = (-?\d+)")
    got = canonical(q)
    assert _pt(got["a"]) == (xb, yb)
    assert _frac(got["b"]) == m
    assert _frac(got["c"]) == ya + m * (x_far - xa)
    assert _frac(got["d"]) == xa + (y_at - ya) / m
    assert not (win_contains(q.figure, x_far) or win_contains(q.figure, _frac(got["d"])))


def win_contains(fig, x) -> bool:
    return fig["x_range"][0] <= x <= fig["x_range"][1]


def check_graph_y_mx(q):
    m, c = read_line(q.figure, 0)
    assert c == 0
    (px, py) = [tuple(F(v) for v in p["at"]) for p in q.figure["points"]][0]
    assert py == m * px
    h = F(q.figure["lines"][1]["through"][0][1])
    assert q.figure["lines"][1]["through"][1][1] == h
    assert q.figure["lines"][1]["label"] == f"y = {h}"
    x_far = _stem_int(q, r"when x = (-?\d+)")
    got = canonical(q)
    assert _frac(got["a"]) == m
    assert _same_eq(got["b"], m, F(0))
    assert _frac(got["c"]) == m * x_far
    assert _frac(got["d"]) == read_intersection(q.figure, 0, 1)[0]
    assert not win_contains(q.figure, x_far)


def check_graph_y_mx_plus_b(q):
    m, c = read_line(q.figure)
    _on_line(q.figure, m, c)
    assert q.figure["x_range"][0] <= 0 <= q.figure["x_range"][1]
    got = canonical(q)
    assert _frac(got["a"]) == m and _frac(got["b"]) == c and c != 0
    assert _same_eq(got["c"], m, c)
    if "crosses the x-axis" in q.parts[3].prompt:
        assert _frac(got["d"]) == -c / m
    else:
        x_far = _stem_int(q, r"when x = (-?\d+)")
        assert _frac(got["d"]) == m * x_far + c
        assert not win_contains(q.figure, x_far)


def check_gradient_from_graph(q):
    m, c = read_line(q.figure)
    pts = {p["label"]: tuple(F(v) for v in p["at"]) for p in q.figure["points"]}
    (xa, ya), (xb, yb) = pts["A"], pts["B"]
    assert ya == m * xa + c and yb == m * xb + c
    got = canonical(q)
    assert _frac(got["a"]) == m
    assert _same_eq(got["b"], m, c)
    assert _pt(got["c"]) == ((xa + xb) / 2, (ya + yb) / 2)
    assert abs(F(got["d"]) - F(math.hypot(xb - xa, yb - ya)).limit_denominator(10**9)) <= F(5, 100)
    assert re.fullmatch(r"\d+\.\d", got["d"])


def check_equation_from_graph(q):
    m, c1 = read_line(q.figure)
    _on_line(q.figure, m, c1)
    p, qq = (int(v) for v in re.search(r"\((-?\d+), (-?\d+)\)", q.stem).groups())
    c2 = qq - m * p
    got = canonical(q)
    assert _same_eq(got["a"], m, c1)
    assert _same_eq(got["b"], m, c2)
    assert _frac(got["c"]) == -c2 / m
    assert _frac(got["d"]) == abs(c2 - c1) and c2 != c1 and c2 != 0


ORACLES = {
    "plot-linear-points": check_plot_linear_points,
    "graph-y-mx": check_graph_y_mx,
    "graph-y-mx-plus-b": check_graph_y_mx_plus_b,
    "gradient-from-graph": check_gradient_from_graph,
    "equation-from-graph": check_equation_from_graph,
}


@pytest.mark.parametrize("skill_id", ids)
def test_source_wired(skill_id):
    check_wired(skill_id, CHAINS[skill_id][0])


@pytest.mark.parametrize("skill_id", ids)
def test_prompt_variety_and_new_shapes_over_hard_tier(skill_id):
    topic, hard = CHAINS[skill_id]
    check_shapes(topic, hard)


@pytest.mark.parametrize("skill_id", ids)
def test_clean_and_deterministic(skill_id):
    check_clean_and_deterministic(CHAINS[skill_id][0])


@pytest.mark.parametrize("skill_id", ids)
def test_shape(skill_id):
    for seed in range(60):
        q = exam(CHAINS[skill_id][0], seed)
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert tuple(p.answer_kind for p in q.parts) == SHAPES[skill_id]
        assert [p.marks for p in q.parts] == MARKS[skill_id]
        assert q.parts[0].follow_through is None and any(p.follow_through for p in q.parts[1:])
        assert q.figure is not None and q.figure["kind"] == "cartesian"
        assert figures.validate(q.figure) == []
        figures.assert_no_leak(q.figure, [p.answer_text for p in q.parts])


@pytest.mark.parametrize("skill_id", ids)
def test_oracle_agrees_over_300_seeds(skill_id):
    topic = CHAINS[skill_id][0]
    for seed in range(300):
        ORACLES[skill_id](exam(topic, seed))


@pytest.mark.parametrize("skill_id", ids)
def test_grading_and_distractors(skill_id):
    check_grading(skill_id, CHAINS[skill_id][0], 120)


@pytest.mark.parametrize("skill_id", ids)
def test_answer_space(skill_id):
    check_answer_space(CHAINS[skill_id][0])


@pytest.mark.parametrize("skill_id", ids)
def test_unknown_tier_rejected(skill_id):
    with pytest.raises(ValueError):
        make(CHAINS[skill_id][0], seed=0, tier="nonsense")


def _eq_text(m: F, c: F) -> str:
    return f"y = {sympy.Rational(m.numerator, m.denominator)}*x + {sympy.Rational(c.numerator, c.denominator)}"


def test_follow_through_plot_linear_points():
    def carried(q):
        got = canonical(q)
        pts = {p["label"]: tuple(F(v) for v in p["at"]) for p in q.figure["points"]}
        xa, ya = pts["A"]
        wrong = _frac(got["b"]) + 1 if _frac(got["b"]) != -1 else F(2)
        x_far = _stem_int(q, r"where x = (-?\d+)")
        y_at = _stem_int(q, r"where y = (-?\d+)")
        later = {"c": str(ya + wrong * (x_far - xa)), "d": str(xa + (y_at - ya) / wrong)}
        return "b", str(wrong), later

    check_follow_through("plot-linear-points", CHAINS["plot-linear-points"][0], carried, 60)


def test_follow_through_graph_y_mx():
    def carried(q):
        k = _frac(canonical(q)["a"]) * 2
        x_far = _stem_int(q, r"when x = (-?\d+)")
        h = F(q.figure["lines"][1]["through"][0][1])
        return "a", str(k * 3), {"c": str(k * 3 * x_far), "d": str(h / (k * 3))}

    check_follow_through("graph-y-mx", CHAINS["graph-y-mx"][0], carried, 60)


def test_follow_through_graph_y_mx_plus_b():
    def carried(q):
        got = canonical(q)
        m, c = _frac(got["a"]), _frac(got["b"])
        x_far = None if "crosses the x-axis" in q.parts[3].prompt else _stem_int(q, r"when x = (-?\d+)")
        for wrong in (m + 1, m + 2, m * 2, m + 3):
            if wrong == 0:
                continue
            last = -c / wrong if x_far is None else wrong * x_far + c
            # A carried reply that merely repeats an earlier reply is not follow-through, so avoid those.
            if last not in (c, wrong, m, 0, 1, -1):
                return "a", str(wrong), {"c": _eq_text(wrong, c), "d": str(last)}
        raise AssertionError("no usable wrong gradient")

    check_follow_through("graph-y-mx-plus-b", CHAINS["graph-y-mx-plus-b"][0], carried, 60)


def test_follow_through_gradient_from_graph():
    def carried(q):
        m, c = read_line(q.figure)
        pts = {p["label"]: tuple(F(v) for v in p["at"]) for p in q.figure["points"]}
        (xa, ya), (xb, yb) = pts["A"], pts["B"]
        wrong = m * 2
        length = math.sqrt(float((xb - xa) ** 2) * (1 + float(wrong) ** 2))
        return "a", str(wrong), {"b": _eq_text(wrong, ya - wrong * xa), "d": f"{length:.1f}"}

    check_follow_through("gradient-from-graph", CHAINS["gradient-from-graph"][0], carried, 60)


def test_follow_through_equation_from_graph():
    def carried(q):
        m, c1 = read_line(q.figure)
        p, qq = (int(v) for v in re.search(r"\((-?\d+), (-?\d+)\)", q.stem).groups())
        wrong_m = m * 2
        c2 = qq - wrong_m * p
        gap = abs(c2 - c1)
        return "a", _eq_text(wrong_m, c1), {"b": _eq_text(wrong_m, c2), "c": str(-c2 / wrong_m), "d": str(gap)}

    check_follow_through("equation-from-graph", CHAINS["equation-from-graph"][0], carried, 60)
