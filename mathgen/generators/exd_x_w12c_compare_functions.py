"""Exam chain for comparing functions: a parabola drawn as a graph against a quadratic given as a vertex-form rule.

Function f is a parabola drawn through lattice points, with its turning point on the grid. Function g is given by a
rule ``g(x) = a(x - h)^2 + k`` that opens the same way. The two meet at two whole-number x-values. Set the way a
Year 9/10 exam sets it:

(a) the rule for f, read from its graph                                        expression, 2 marks
(b) the difference between the maximum (or minimum) values of f and g           number,     1 mark,  follow-through (a)
(c) the difference between their average rates of change over an interval       number,     2 marks, follow-through (a)
(d) the two x-values where f(x) = g(x)                                          set,        2 marks

## The algebra (checked on every question)

f and g are built backwards from the two meeting points: ``f(x) - g(x) = (a_f - a_g)(x - r1)(x - r2)`` fixes the sum of
the roots (so h_f and h_g are tied to a_f and a_g) and the product (so k_g follows from k_f). The roots are then
confirmed with SymPy's ``solve``, and f's rule is read back from the figure's sampled curve (the turning point is the
extreme sample; a is the drop or rise one column across). The average rate of change of ``a(x - h)^2 + k`` over
``[p, q]`` is ``a(p + q - 2h)``, which is also computed from the two function values.

## Distractors

* (a): the horizontal shift with the wrong sign, the turning point's height with the wrong sign, the leading
  coefficient lost.
* (b): the difference of the turning points' x-coordinates, the values added, the difference of the values at x = 0,
  one maximum alone.
* (c): the change over the interval without dividing by its length, the two rates added, one rate alone.
* (d): both roots with the wrong sign, one root with the wrong sign, the common y-value at each root.
"""

from __future__ import annotations

import sympy

from mathgen import figures
from mathgen.core import attempts, check, generator, join_terms
from mathgen.generators._exd_x_w12c_kit import redraw, expression_part, int_part, set_part
from mathgen.generators.fg1_graph_compare_build import vertex_form
from mathgen.generators.pv1a_line_figure_build import Window

_X = sympy.Symbol("x")
_ALT = "a parabola on a coordinate grid"

_OPEN = (
    "The graph shows function f, a parabola. Function g is given by g(x) = {g}.",
    "Function f is the parabola drawn on the grid. Function g is defined by g(x) = {g}.",
    "A parabola on the grid is the graph of function f. Another function is g(x) = {g}.",
)


def _curve(win: Window, a, h, k) -> dict:
    n = (win.u1 - win.u0) * 8 + 1  # step 1/8, so every whole-number x is an exact sample
    pieces = figures.sample_curve(lambda x: a * (x - h) ** 2 + k, win.u0, win.u1, (win.v0, win.v1), n=n)
    return figures.cartesian(x_range=win.x_range, y_range=win.y_range, alt=_ALT, grid=1, curves=[{"pieces": pieces}])


def _build(rng):
    """Rule for f from its graph, the gap between turning-point heights, between rates of change, and where f = g."""
    for _ in attempts(20000, "a parabola and a second quadratic that meet at two whole-number x-values"):
        sgn = rng.choice((-1, 1))
        a1, a2 = (sgn * m for m in rng.sample((1, 2, 3), 2))
        h1, h2 = rng.randint(-4, 5), rng.randint(-5, 6)
        if 0 in (h1, h2) or h1 == h2:
            continue
        s_num = 2 * a1 * h1 - 2 * a2 * h2
        if s_num % (a1 - a2):
            continue
        s = s_num // (a1 - a2)
        r1 = rng.randint(-8, 9)
        r2 = s - r1
        if r1 >= r2 or not -9 <= r2 <= 11:
            continue
        k1 = rng.choice([v for v in range(-5, 6) if v != 0])
        k2 = a1 * h1 * h1 + k1 - a2 * h2 * h2 - (a1 - a2) * r1 * r2
        if k2 == 0 or k2 == k1 or abs(k2) > 12:
            continue
        # a window with the turning point inside and a second lattice point of f visible
        u0 = h1 - rng.randint(3, 6)
        win_u = (u0, u0 + 9)
        v0 = k1 - rng.randint(5, 8) if sgn < 0 else k1 - rng.randint(1, 3)
        win = Window(win_u[0], win_u[1], v0, v0 + 9)

        def f(x):
            return a1 * (x - h1) ** 2 + k1

        def g(x):
            return a2 * (x - h2) ** 2 + k2

        visible = [x for x in range(win.u0, win.u1 + 1) if win.v0 <= f(x) <= win.v1]
        spans = [(p, q) for p in visible for q in visible if q - p in (2, 3, 4)]
        spans = [(p, q) for p, q in spans if a1 * (p + q - 2 * h1) != a2 * (p + q - 2 * h2)]
        if not spans or not (win.v0 <= f(h1 + 1) <= win.v1):
            continue
        p, q = rng.choice(spans)
        break

    fig = _curve(win, a1, h1, k1)

    # Second route 1: f's rule read back from the drawn curve.
    samples = {x: y for piece in fig["curves"][0]["pieces"] for x, y in map(tuple, piece)}
    extreme = (max if sgn < 0 else min)(samples.values())
    check(samples.get(h1) == extreme == k1, "the turning point is not where the rule puts it")
    check(samples[h1 + 1] - k1 == a1, "the leading coefficient does not match the drawn curve")
    # Second route 2: the meeting points by solving f(x) = g(x) with SymPy.
    f_expr, g_expr = a1 * (_X - h1) ** 2 + k1, a2 * (_X - h2) ** 2 + k2
    check(
        sorted(sympy.solve(sympy.Eq(f_expr, g_expr), _X)) == [r1, r2], "the meeting points are not the constructed ones"
    )
    rate_f, rate_g = (f(q) - f(p)) // (q - p), (g(q) - g(p)) // (q - p)
    check((f(q) - f(p)) % (q - p) == 0 and (g(q) - g(p)) % (q - p) == 0, "an average rate of change is not whole")
    check(rate_f == a1 * (p + q - 2 * h1) and rate_g == a2 * (p + q - 2 * h2), "the rates of change disagree")

    word = "maximum" if sgn < 0 else "minimum"
    f_text = vertex_form(a1, h1, k1)
    g_text = vertex_form(a2, h2, k2)
    stem = rng.choice(_OPEN).format(g=g_text)

    lead_wrong = sgn if abs(a1) > 1 else -a1
    part_a = expression_part(
        "Find the rule for f in the form f(x) = a(x - h)^2 + k. Give the expression that f(x) equals.",
        f_text,
        2,
        [
            f"The turning point is at ({h1}, {k1}), so h = {h1} and k = {k1}.",
            f"One column across from the turning point, f is {a1} {'above' if a1 > 0 else 'below'} {k1}, so a = {a1}.",
            f"f(x) = {f_text}.",
        ],
        [
            (vertex_form(a1, -h1, k1), "vertex-form-horizontal-sign"),
            (vertex_form(a1, h1, -k1), "vertex-form-vertical-sign"),
            (vertex_form(lead_wrong, h1, k1), "vertex-form-leading-coefficient-lost"),
        ],
    )

    diff_k = abs(k1 - k2)
    sign_k = 1 if k1 > k2 else -1  # (b) follows through with this sign, not abs()
    part_b = int_part(
        f"Find the difference between the {word} value of f and the {word} value of g.",
        diff_k,
        1,
        [
            f"f has {word} value {k1} (its turning point is ({h1}, {k1})).",
            f"g has {word} value {k2} (its turning point is ({h2}, {k2})).",
            f"The difference is {diff_k}.",
        ],
        [
            (abs(h1 - h2), "compares-the-x-coordinates-of-the-turning-points"),
            (abs(k1 + k2), "adds-instead-of-subtracts"),
            (abs(f(0) - g(0)), "compares-starting-values-only"),
            (abs(k1), "difference-not-taken"),
        ],
        lo=0,
        hi=40,
        follow_through=f"{sign_k}*(Subs(part_a,(x,),({h1},)) - ({k2}))",
    )

    diff_rate = abs(rate_f - rate_g)
    sign_r = 1 if rate_f > rate_g else -1
    part_c = int_part(
        f"Between x = {p} and x = {q}, find the difference between the average rate of change of f "
        "and the average rate of change of g. Give a positive number.",
        diff_rate,
        2,
        [
            f"f: from {f(p)} to {f(q)} over {q - p} across, an average rate of change of {rate_f}.",
            f"g: from {g(p)} to {g(q)} over {q - p} across, an average rate of change of {rate_g}.",
            f"The difference is {diff_rate}.",
        ],
        [
            (abs((f(q) - f(p)) - (g(q) - g(p))), "change-not-rate"),
            (abs(rate_f + rate_g), "rates-added-not-subtracted"),
            (abs(rate_f), "difference-not-taken"),
            (abs(rate_g), "difference-not-taken"),
        ],
        lo=0,
        hi=100,
        follow_through=(f"{sign_r}*((Subs(part_a,(x,),({q},)) - Subs(part_a,(x,),({p},)))/({q - p}) - ({rate_g}))"),
    )

    y1, y2 = f(r1), f(r2)
    poly = sympy.Poly(sympy.expand(f_expr - g_expr), _X)
    qb, qc = int(poly.coeff_monomial(_X)), int(poly.coeff_monomial(1))
    part_d = set_part(
        "Find the two values of x for which f(x) = g(x). Write them in braces, separated by a comma.",
        [r1, r2],
        2,
        [
            f"Set f(x) = g(x): {f_text} = {g_text}.",
            f"Expanding and collecting terms gives {join_terms([(a1 - a2, 'x^2'), (qb, 'x'), (qc, '')])} = 0.",
            f"This has solutions x = {r1} and x = {r2}.",
            f"Check: f({r1}) = g({r1}) = {y1} and f({r2}) = g({r2}) = {y2}.",
        ],
        [
            ([-r1, -r2], "sign-flip-both-roots"),
            ([y1, y2], "swapped-coordinates"),
            ([r1, -r2], "sign-flip-one-root"),
            ([h1, h2], "compares-the-x-coordinates-of-the-turning-points"),
        ],
    )
    return dict(text=stem, figure=fig, parts=[part_a, part_b, part_c, part_d])


@generator("exd_x_w12c_compare_functions", level=13, presets={"exam": {"tier": "exam"}})
def exd_x_w12c_compare_functions(rng, tier: str = "exam"):
    """Rule for f from its graph, the gap between turning-point heights, between rates of change, and where f = g."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return redraw(rng, _build)
