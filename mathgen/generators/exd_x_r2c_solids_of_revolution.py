"""Exam chain for volumes of solids of revolution: a parabola and a line, rotated about a horizontal axis.

A curve ``y = c x^2 + d`` and a line ``y = s x + w`` meet at two points with whole-number x-coordinates
``u < v``. The region between them is rotated about the x-axis or about a horizontal line ``y = -q`` below it,
and the question is set the way a senior exam sets it:

(a) the x-coordinate of the left-hand intersection            number,     1 mark
(b) the x-coordinate of the right-hand intersection           number,     1 mark
(c) the area A(x) of the cross-section at x                   expression, 2 marks
(d) the exact volume of the solid                             expression, 3 marks, follow-through from (a), (b)

## Construction: choose the intersections first

The curve and the line are built from ``c``, ``d``, ``u`` and ``v``: the line is ``y = c(u + v)x + (d - c u v)``,
so that ``c x^2 + d - (s x + w) = c (x - u)(x - v)``. With ``0 <= u < v``, ``c > 0`` and ``d >= 0`` the curve
lies on or above the x-axis, and the line lies above the curve between the intersections, so the line is the
outer radius and the curve the inner radius and the solid is a washer solid. The axis of rotation is the x-axis
or the line ``y = -q``; the radii are the heights above the axis, so ``q`` is added to each.

The volume is ``pi * integral_u^v [ (s x + w + q)^2 - (c x^2 + d + q)^2 ] dx``. It is worked in exact
rational arithmetic and re-derived with SymPy's integrator. The follow-through for (d) is that same integral's
antiderivative evaluated at the learner's own (a) and (b).

## Distractors

* (a), (b): the sign of the root flipped, the other root, the y-coordinate at the intersection.
* (c): the square of the difference of the radii, the radii measured from the x-axis when the axis is lower,
  the radii in the wrong order, and the area without pi.
* (d): the volume of each of those wrong cross-sections.
"""

from __future__ import annotations

from sympy import Poly, Rational, Symbol, pi

from mathgen.answers import Distractor
from mathgen.core import check, generator, join_terms, latex_answer, phrase, pick_distractors

_X = Symbol("x")

_STEMS = (
    "The region between the curve y = {curve} and the line y = {line} is rotated through one full turn about {axis} "
    "to form a solid.",
    "A solid is formed by rotating about {axis} the finite region enclosed by the curve y = {curve} and the line "
    "y = {line}.",
    "Let R be the region enclosed by the curve y = {curve} and the line y = {line}. R is rotated about {axis}.",
)


def _factored(c, u, v) -> str:
    """c(x - u)(x - v) as it is written on paper: no leading 1, and x rather than (x - 0)."""

    def factor(r):
        return "x" if r == 0 else f"(x - {r})"

    return f"{'' if c == 1 else c}{factor(u)}{factor(v)}"


def _typed(expr) -> str:
    return str(expr).replace("**", "^")


def _int_part(prompt, value, marks, working, wrong):
    """A whole-number part: `wrong` is a list of (value, misconception id)."""
    answer_text = str(value)
    candidates = [Distractor(str(v), tag, int(v)) for v, tag in wrong if v is not None]
    return dict(
        prompt=prompt,
        answer_kind="number",
        answer=int(value),
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        marks=marks,
        working=working,
        distractors=pick_distractors(answer_text, candidates),
    )


def _boole(poly: Poly, lo: int, hi: int):
    """Exact integral of a polynomial of degree at most 5 from Boole's rule on five equally spaced points."""
    h = Rational(hi - lo, 4)
    f = [poly.eval(lo + k * h) for k in range(5)]
    return 2 * h * (7 * f[0] + 32 * f[1] + 12 * f[2] + 32 * f[3] + 7 * f[4]) / 45


def _volume_text(value_over_pi) -> str:
    return _typed(value_over_pi * pi)


@generator("exd_x_r2c_solids_of_revolution", level=17, presets={"exam": {"tier": "exam"}})
def exd_x_r2c_solids_of_revolution(rng, tier: str = "exam"):
    """Intersections of a parabola and a line, the cross-section, and the exact volume of revolution."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    c = rng.choice([1, 1, 2])
    d = rng.randint(0, 3)
    u = rng.randint(0, 2)
    v = u + rng.randint(1, 3)
    q = rng.choice([0, 0, 1, 2, 3])
    s, w = c * (u + v), d - c * u * v

    curve = Poly(c * _X**2 + d, _X)
    line = Poly(s * _X + w, _X)
    check(curve - line == Poly(c * (_X - u) * (_X - v), _X), "the intersections are not at u and v")
    check(all(line.eval(t) >= curve.eval(t) >= 0 for t in (u, v, Rational(2 * u + v, 3))), "bad region")

    outer, inner = line + q, curve + q
    washer = outer**2 - inner**2  # A(x) / pi
    anti = washer.integrate()
    volume_over_pi = anti.eval(v) - anti.eval(u)
    check(volume_over_pi == _boole(washer, u, v), "volume disagrees with the independent quadrature")
    check(volume_over_pi > 0, "the volume is not positive")
    check(all(washer.eval(t) > 0 for t in (Rational(2 * u + v, 3), Rational(u + 2 * v, 3))), "outer is not outer")

    curve_text = join_terms([(c, "x^2"), (d, "")])
    line_text = join_terms([(s, "x"), (w, "")])
    axis_text = "the x-axis" if q == 0 else f"the line y = -{q}"
    stem = phrase(rng, _STEMS, curve=curve_text, line=line_text, axis=axis_text)

    # ---- (a), (b): the intersections --------------------------------------------------------------------------
    y_at = line.eval
    part_a = _int_part(
        "Find the x-coordinate of the point where the line meets the curve on the left.",
        u,
        1,
        [f"{curve_text} = {line_text} gives {_factored(c, u, v)} = 0 after collecting terms, so x = {u} or x = {v}."],
        [
            (-u, "sign-flip-both-roots"),
            (-v, "sign-flip-both-roots"),
            (v, "wrong-root-chosen"),
            (y_at(u), "y-value-used-as-x-limit"),
            (u + v, "ignored-constant-term"),
        ],
    )
    part_b = _int_part(
        "Find the x-coordinate of the point where the line meets the curve on the right.",
        v,
        1,
        [f"The other root of {_factored(c, u, v)} = 0 is x = {v}."],
        [
            (-v, "sign-flip-both-roots"),
            (-u, "sign-flip-both-roots"),
            (u, "wrong-root-chosen"),
            (y_at(v), "y-value-used-as-x-limit"),
            (u + v, "ignored-constant-term"),
        ],
    )

    # ---- (c): the cross-section --------------------------------------------------------------------------------
    area = pi * washer.as_expr()
    area_text = _typed(area)
    wrong_areas = [
        ("washer-squared-difference", (outer - inner) ** 2),
        ("outer-inner-radii-swapped", -washer),
        ("forgot-pi-in-volume", washer),
    ]
    if q:
        wrong_areas.insert(1, ("axis-offset-ignored", line**2 - curve**2))
    wrong_areas = [(tag, e) for tag, e in wrong_areas if e != washer]
    area_cands = []
    for tag, e in wrong_areas:
        shown = e.as_expr() if tag == "forgot-pi-in-volume" else pi * e.as_expr()
        area_cands.append(Distractor(_typed(shown), tag, _typed(shown)))
    part_c = dict(
        prompt="Find an expression for A(x), the area of the cross-section of the solid perpendicular to the axis of "
        "rotation at x, for x between the two intersections.",
        answer_kind="expression",
        answer=str(area),
        answer_text=area_text,
        answer_latex=latex_answer(area),
        marks=2,
        working=[
            f"Each cross-section is a washer. Outer radius {_typed(outer.as_expr())}, inner radius {_typed(inner.as_expr())}.",
            f"A(x) = π[(outer)² - (inner)²] = {area_text}.",
        ],
        distractors=pick_distractors(area_text, area_cands),
    )

    # ---- (d): the volume ---------------------------------------------------------------------------------------
    volume = volume_over_pi * pi
    volume_text = _volume_text(volume_over_pi)
    d_cands = []
    for tag, e in wrong_areas:
        wrong_anti = e.integrate()
        wrong_volume = wrong_anti.eval(v) - wrong_anti.eval(u)
        shown = wrong_volume if tag == "forgot-pi-in-volume" else wrong_volume * pi
        d_cands.append(Distractor(_typed(shown), tag, _typed(shown)))
    a_sym, b_sym = Symbol("part_a"), Symbol("part_b")
    follow = pi * (anti.as_expr().subs(_X, b_sym) - anti.as_expr().subs(_X, a_sym))
    part_d = dict(
        prompt="Hence find the exact volume of the solid.",
        answer_kind="expression",
        answer=str(volume),
        answer_text=volume_text,
        answer_latex=latex_answer(volume),
        marks=3,
        working=[
            f"V = ∫ from {u} to {v} of A(x) dx = π ∫ ({_typed(washer.as_expr())}) dx.",
            f"V = {volume_text}.",
        ],
        distractors=pick_distractors(volume_text, d_cands),
        follow_through=str(follow),
    )
    return dict(text=stem, parts=[part_a, part_b, part_c, part_d])
