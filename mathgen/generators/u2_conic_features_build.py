"""Generates `u2_conic_features` questions and the exam chain `exd_x_u2_conic_features`.

``ucv1-conic-sections-features`` (``geometry``, answer ``number``; individual answers are whole numbers or fractions):
the features of an ellipse, hyperbola or parabola read from its equation, with the square completed when the equation
is in general form.

* easy: a conic centred at the origin in standard form: the eccentricity of an ellipse or hyperbola, or the focus or
  directrix of a parabola y^2 = 4px (x^2 = 4py) with the sign of p to read.
* medium: an ellipse or hyperbola in general form, A x^2 + B y^2 + D x + E y + F = 0, whose axes are horizontal or
  vertical: complete the square to find the distance between the foci.
* hard: the directrix of a parabola in general form, or the directrix of an ellipse or hyperbola in general form
  (x = h +- a^2/c, so the answer is usually a fraction).
* exam: an ellipse in general form. (a) the centre, (b) the distance c from the centre to a focus, (c) the
  eccentricity (follows through from (b)), (d) the distance from the centre to a directrix (follows through from (b)).

Every feature is re-derived by building the general-form equation from the standard form (so the learner's route,
completing the square, runs the other way) and checking the equation against SymPy's expansion; foci are checked by the
defining distance property on a point of the curve.
"""

from __future__ import annotations

from fractions import Fraction
from math import gcd

import sympy

from ..answers import Distractor
from ..core import attempts, check, generator, rational_text
from ._u2_kit import int_d, keep, frac_d, redraws

TOPIC = "u2_conic_features"
EXAM_TOPIC = "exd_x_u2_conic_features"

X, Y = sympy.symbols("x y")

#: (a, b, c) with a^2 = b^2 + c^2 (ellipse: a > b) and with c^2 = a^2 + b^2 (hyperbola).
_ELLIPSE_TRIPLES = [(5, 3, 4), (5, 4, 3), (10, 6, 8), (10, 8, 6), (13, 12, 5), (13, 5, 12), (17, 15, 8), (17, 8, 15)]
_SMALL_ELLIPSE = [(5, 3, 4), (5, 4, 3), (10, 6, 8), (10, 8, 6)]
_HYPERBOLA_TRIPLES = [(3, 4, 5), (4, 3, 5), (5, 12, 13), (12, 5, 13), (6, 8, 10), (8, 6, 10), (8, 15, 17), (15, 8, 17)]
_SMALL_HYPERBOLA = [(3, 4, 5), (4, 3, 5), (6, 8, 10), (8, 6, 10)]


def _term(coef, mono, first):
    """One term of a polynomial: ' + 3x', ' - y', or without the sign when it comes first."""
    if coef == 0:
        return ""
    mag = abs(coef)
    body = ("" if mag == 1 and mono else str(mag)) + mono
    if first:
        return ("-" if coef < 0 else "") + body
    return (" - " if coef < 0 else " + ") + body


def poly_text(terms):
    """'9x² + 25y² - 18x + 100y - 116' from [(coef, monomial)] with the constant monomial ''."""
    out, first = "", True
    for coef, mono in terms:
        piece = _term(coef, mono, first)
        if piece:
            out += piece
            first = False
    return out


def general_form(kind, a, b, h, k, vertical):
    """(A, B, D, E, F) of the integer equation with no common factor, for the conic with these parameters."""
    if kind == "ellipse":
        sa, sb = (b * b, a * a) if not vertical else (a * a, b * b)  # coefficient of x², of y²
        # b²(x-h)² + a²(y-k)² = a²b² (horizontal major axis); the roles swap for a vertical one.
        expr = sa * (X - h) ** 2 + sb * (Y - k) ** 2 - a * a * b * b
    else:
        if not vertical:
            expr = b * b * (X - h) ** 2 - a * a * (Y - k) ** 2 - a * a * b * b
        else:
            expr = -a * a * (X - h) ** 2 + b * b * (Y - k) ** 2 - a * a * b * b
    poly = sympy.Poly(sympy.expand(expr), X, Y)
    coeffs = (
        poly.coeff_monomial(X**2),
        poly.coeff_monomial(Y**2),
        poly.coeff_monomial(X),
        poly.coeff_monomial(Y),
        poly.coeff_monomial(1),
    )
    g = 0
    for c in coeffs:
        g = gcd(g, int(c))
    coeffs = tuple(int(c) // g for c in coeffs)
    check(poly.coeff_monomial(X * Y) == 0, "an xy term appeared")
    return coeffs


def equation_text(coeffs):
    A, B, D, E, F = coeffs
    return poly_text([(A, "x²"), (B, "y²"), (D, "x"), (E, "y"), (F, "")]) + " = 0"


def _check_foci(kind, a, b, c, h, k, vertical):
    """A point on the curve has focal distances summing (ellipse) or differing (hyperbola) by 2a."""
    if kind == "ellipse":
        px, py = (h + b, k) if not vertical else (h, k + a)
        # a point at the end of the minor / major axis along the other direction
        pts = [(h + a, k), (h - a, k)] if not vertical else [(h, k + a), (h, k - a)]
        f1, f2 = ((h + c, k), (h - c, k)) if not vertical else ((h, k + c), (h, k - c))
        for qx, qy in pts + [(px, py)]:
            if (qx, qy) == (px, py) and vertical:
                continue
            d = sympy.sqrt((qx - f1[0]) ** 2 + (qy - f1[1]) ** 2) + sympy.sqrt((qx - f2[0]) ** 2 + (qy - f2[1]) ** 2)
            check(sympy.simplify(d - 2 * a) == 0 if (qx, qy) in pts else True, "focal distance sum differs")
    else:
        pts = [(h + a, k), (h - a, k)] if not vertical else [(h, k + a), (h, k - a)]
        f1, f2 = ((h + c, k), (h - c, k)) if not vertical else ((h, k + c), (h, k - c))
        for qx, qy in pts:
            d1 = sympy.sqrt((qx - f1[0]) ** 2 + (qy - f1[1]) ** 2)
            d2 = sympy.sqrt((qx - f2[0]) ** 2 + (qy - f2[1]) ** 2)
            check(abs(d1 - d2) == 2 * a, "focal distance difference differs")


def _on_curve(kind, a, b, h, k, vertical):
    """The standard-form equation holds at the vertices (sanity check on the general form)."""
    coeffs = general_form(kind, a, b, h, k, vertical)
    A, B, D, E, F = coeffs
    vertex = (h + a, k) if not vertical else (h, k + a)
    if kind == "ellipse" and vertical is False:
        vertex = (h + a, k)
    check(A * vertex[0] ** 2 + B * vertex[1] ** 2 + D * vertex[0] + E * vertex[1] + F == 0, "vertex not on curve")
    return coeffs


def _nonzero_centre(rng):
    return rng.choice([v for v in range(-5, 6) if v]), rng.choice([v for v in range(-5, 6) if v])


# -- easy ----------------------------------------------------------------------------------------------------------


def _easy_ellipse(rng):
    a, b, c = rng.choice(_SMALL_ELLIPSE + [(13, 12, 5), (13, 5, 12)])
    vertical = rng.random() < 0.5
    e = Fraction(c, a)
    sa, sb = (b * b, a * a) if not vertical else (a * a, b * b)  # denominators under x², y²
    cands = [
        frac_d(Fraction(b, a), "conic-eccentricity-uses-b"),
        frac_d(Fraction(a, c), "conic-eccentricity-inverted"),
        frac_d(Fraction(c, b), "conic-eccentricity-denominator-b"),
        frac_d(Fraction(a * a - b * b, a * a), "conic-eccentricity-squared"),
        frac_d(
            Fraction(c * c + b * b, a * a) if False else Fraction(b * b + a * a, a * a),
            "conic-ellipse-hyperbola-formula",
        ),
    ]
    text = f"Find the eccentricity of the ellipse x²/{sa} + y²/{sb} = 1."
    working = [
        f"The larger denominator is a² = {a * a} and the smaller is b² = {b * b}, so c² = a² - b² = {c * c} and c = {c}.",
        f"e = c/a = {c}/{a}" + (f" = {rational_text(e)}." if e.denominator != c else "."),
    ]
    return dict(
        text=text,
        answer_kind="fraction",
        answer=_frac(e),
        answer_text=rational_text(e),
        answer_latex=rational_text(e),
        working=working,
        distractors=keep(rational_text(e), cands, 4),
    )


def _frac(v):
    from ..answers import exact_value

    return exact_value(Fraction(v))


def _easy_hyperbola(rng):
    a, b, c = rng.choice(_SMALL_HYPERBOLA + [(5, 12, 13), (12, 5, 13)])
    e = Fraction(c, a)
    cands = [
        frac_d(Fraction(a, c), "conic-eccentricity-inverted"),
        frac_d(Fraction(b, a), "conic-eccentricity-uses-b"),
        frac_d(Fraction(abs(a * a - b * b), a * a), "conic-ellipse-hyperbola-formula"),
        frac_d(Fraction(c, b), "conic-eccentricity-denominator-b"),
    ]
    if rng.random() < 0.4:
        text = f"Find the eccentricity of the hyperbola y²/{a * a} - x²/{b * b} = 1."
    else:
        text = f"Find the eccentricity of the hyperbola x²/{a * a} - y²/{b * b} = 1."
    working = [
        f"Here a² = {a * a} and b² = {b * b}, so for a hyperbola c² = a² + b² = {c * c} and c = {c}.",
        f"e = c/a = {c}/{a}" + (f" = {rational_text(e)}." if e.denominator != a else "."),
    ]
    return dict(
        text=text,
        answer_kind="fraction",
        answer=_frac(e),
        answer_text=rational_text(e),
        answer_latex=rational_text(e),
        working=working,
        distractors=keep(rational_text(e), cands, 4),
    )


def _easy_parabola(rng):
    p = rng.choice([-5, -4, -3, -2, 2, 3, 4, 5, 6])
    horizontal = rng.random() < 0.5
    four_p = 4 * p
    var, other = ("y", "x") if horizontal else ("x", "y")
    equation = f"{var}² = {four_p}{other}"
    ask_focus = rng.random() < 0.5
    if ask_focus:
        text = f"The parabola {equation} has its focus on the {other}-axis. Find the {other}-coordinate of the focus."
        value = p
        cands = [
            int_d(four_p, "conic-parabola-4p-as-p"),
            int_d(-p, "conic-parabola-focus-directrix"),
            int_d(2 * p, "conic-parabola-4p-as-p"),
            int_d(p * p, "conic-parabola-4p-as-p"),
        ]
        working = [f"{var}² = 4p{other} with 4p = {four_p}, so p = {p}.", f"The focus is at {other} = p = {p}."]
    else:
        text = (
            f"The parabola {equation} has a directrix perpendicular to the {other}-axis. "
            f"Find the {other}-coordinate of the directrix."
        )
        value = -p
        cands = [
            int_d(p, "conic-parabola-focus-directrix"),
            int_d(-four_p, "conic-parabola-4p-as-p"),
            int_d(-2 * p, "conic-parabola-4p-as-p"),
            int_d(four_p, "conic-parabola-4p-as-p"),
        ]
        working = [f"{var}² = 4p{other} with 4p = {four_p}, so p = {p}.", f"The directrix is {other} = -p = {-p}."]
    return dict(
        text=text,
        answer_kind="number",
        answer=value,
        answer_text=str(value),
        answer_latex=str(value),
        working=working,
        distractors=keep(str(value), cands, 4),
    )


def _easy(rng):
    return rng.choice([_easy_ellipse, _easy_hyperbola, _easy_parabola])(rng)


# -- medium --------------------------------------------------------------------------------------------------------


def _medium(rng):
    kind = rng.choice(["ellipse", "hyperbola"])
    a, b, c = rng.choice((_SMALL_ELLIPSE + [(13, 12, 5), (13, 5, 12)]) if kind == "ellipse" else _SMALL_HYPERBOLA)
    vertical = rng.random() < 0.5
    h, k = _nonzero_centre(rng)
    coeffs = _on_curve(kind, a, b, h, k, vertical)
    _check_foci(kind, a, b, c, h, k, vertical)
    shape = "an ellipse" if kind == "ellipse" else "a hyperbola"
    standard = _standard_text(kind, a, b, h, k, vertical)
    if rng.random() < 0.6:
        value = Fraction(2 * c)
        text = f"The curve {equation_text(coeffs)} is {shape}. Find the distance between its foci."
        cands = [
            int_d(c, "conic-focal-distance-half"),
            int_d(2 * a, "conic-axis-length-not-focal"),
            int_d(2 * b, "conic-axis-length-not-focal"),
            int_d(2 * c * c, "conic-focal-distance-squared"),
            int_d(
                2 * int((a * a + b * b if kind == "ellipse" else abs(a * a - b * b)) ** 0.5 + 0.5),
                "conic-ellipse-hyperbola-formula",
            ),
        ]
        cands = [x for x in cands if int(x.text) != 2 * c]
        working = [
            f"Completing the square gives {standard}.",
            f"{'a² - b²' if kind == 'ellipse' else 'a² + b²'} = {c * c}, so c = {c} and the foci are {2 * c} apart.",
        ]
        return dict(
            text=text,
            answer_kind="number",
            answer=int(value),
            answer_text=str(int(value)),
            answer_latex=str(int(value)),
            working=working,
            distractors=keep(str(int(value)), cands, 4),
        )
    value = Fraction(2 * b * b, a)
    text = (
        f"The curve {equation_text(coeffs)} is {shape}. "
        "Find the length of its latus rectum, the chord through a focus perpendicular to the axis of the foci."
    )
    cands = [
        frac_d(Fraction(b * b, a), "conic-latus-rectum-half"),
        frac_d(Fraction(2 * a * a, b), "conic-latus-rectum-swapped"),
        frac_d(Fraction(2 * b, a), "conic-latus-rectum-b-not-squared"),
        frac_d(Fraction(2 * c * c, a), "conic-latus-rectum-uses-c"),
    ]
    working = [
        f"Completing the square gives {standard}, so a = {a} and b = {b}.",
        f"The latus rectum has length 2b²/a = 2 × {b * b}/{a} = {rational_text(value)}.",
    ]
    return dict(
        text=text,
        answer_kind="fraction",
        answer=_frac(value),
        answer_text=rational_text(value),
        answer_latex=rational_text(value),
        working=working,
        distractors=keep(rational_text(value), cands, 4),
    )


def _shifted(var, centre):
    if centre == 0:
        return var
    return f"({var} {'-' if centre > 0 else '+'} {abs(centre)})"


def _standard_text(kind, a, b, h, k, vertical):
    sx, sy = (a * a, b * b) if not vertical else (b * b, a * a)
    if kind == "ellipse":
        # a is the semi-major axis: under x for horizontal, under y for vertical.
        sx, sy = (a * a, b * b) if not vertical else (b * b, a * a)
        return f"{_shifted('x', h)}²/{sx} + {_shifted('y', k)}²/{sy} = 1"
    if not vertical:
        return f"{_shifted('x', h)}²/{a * a} - {_shifted('y', k)}²/{b * b} = 1"
    return f"{_shifted('y', k)}²/{a * a} - {_shifted('x', h)}²/{b * b} = 1"


# -- hard ----------------------------------------------------------------------------------------------------------


def _hard_parabola(rng):
    p = rng.choice([-4, -3, -2, -1, 1, 2, 3, 4])
    h, k = _nonzero_centre(rng)
    horizontal = rng.random() < 0.5
    if horizontal:  # (y-k)² = 4p(x-h)
        expr = sympy.expand((Y - k) ** 2 - 4 * p * (X - h))
        terms = [(1, "y²"), (-4 * p, "x"), (-2 * k, "y"), (k * k + 4 * p * h, "")]
        poly = sympy.Poly(expr, X, Y)
        check(
            [poly.coeff_monomial(Y**2), poly.coeff_monomial(X), poly.coeff_monomial(Y), poly.coeff_monomial(1)]
            == [t[0] for t in terms],
            "parabola coefficients differ",
        )
        text = f"The parabola {poly_text(terms)} = 0 has its axis parallel to the x-axis. Find the equation of its directrix as x = ... and give the value."
        text = f"The parabola {poly_text(terms)} = 0 has its axis parallel to the x-axis. Its directrix is the line x = d. Find d."
        value = h - p
        cands = [
            int_d(h + p, "conic-parabola-focus-directrix"),
            int_d(-p, "conic-parabola-focus-directrix"),
            int_d(-h - p, "conic-completing-square-sign"),
            int_d(h - 4 * p, "conic-parabola-4p-as-p"),
        ]
        working = [
            f"Complete the square: (y - ({k}))² = {4 * p}(x - ({h})), so 4p = {4 * p}, p = {p} and the vertex is ({h}, {k}).",
            f"The directrix is x = {h} - ({p}) = {value}.",
        ]
    else:  # (x-h)² = 4p(y-k)
        expr = sympy.expand((X - h) ** 2 - 4 * p * (Y - k))
        terms = [(1, "x²"), (-2 * h, "x"), (-4 * p, "y"), (h * h + 4 * p * k, "")]
        poly = sympy.Poly(expr, X, Y)
        check(
            [poly.coeff_monomial(X**2), poly.coeff_monomial(X), poly.coeff_monomial(Y), poly.coeff_monomial(1)]
            == [t[0] for t in terms],
            "parabola coefficients differ",
        )
        text = f"The parabola {poly_text(terms)} = 0 has its axis parallel to the y-axis. Its directrix is the line y = d. Find d."
        value = k - p
        cands = [
            int_d(k + p, "conic-parabola-focus-directrix"),
            int_d(-p, "conic-parabola-focus-directrix"),
            int_d(-k - p, "conic-completing-square-sign"),
            int_d(k - 4 * p, "conic-parabola-4p-as-p"),
        ]
        working = [
            f"Complete the square: (x - ({h}))² = {4 * p}(y - ({k})), so 4p = {4 * p}, p = {p} and the vertex is ({h}, {k}).",
            f"The directrix is y = {k} - ({p}) = {value}.",
        ]
    return dict(
        text=text,
        answer_kind="number",
        answer=value,
        answer_text=str(value),
        answer_latex=str(value),
        working=working,
        distractors=keep(str(value), cands, 4),
    )


def _hard_directrix(rng):
    kind = rng.choice(["ellipse", "hyperbola"])
    a, b, c = rng.choice(_ELLIPSE_TRIPLES if kind == "ellipse" else _HYPERBOLA_TRIPLES)
    vertical = rng.random() < 0.5
    if max(a, b) > 8:  # keep the coefficients modest
        a, b, c = rng.choice(_SMALL_ELLIPSE if kind == "ellipse" else _SMALL_HYPERBOLA)
    h, k = _nonzero_centre(rng)
    coeffs = _on_curve(kind, a, b, h, k, vertical)
    _check_foci(kind, a, b, c, h, k, vertical)
    offset = Fraction(a * a, c)
    centre = k if vertical else h
    value = centre + offset
    which = "upper" if vertical else "right"
    axis = "y" if vertical else "x"
    cands = [
        frac_d(centre - offset, "conic-directrix-wrong-side"),
        frac_d(offset, "conic-directrix-forgot-centre"),
        frac_d(
            centre + Fraction(a, 1) / (Fraction(c, a))
            if False
            else centre + Fraction(a * a, a * a - 0) * 0 + Fraction(c * c, a),
            "conic-directrix-uses-c-squared",
        ),
        frac_d(centre + Fraction(a, c), "conic-directrix-uses-a-over-c"),
        frac_d(centre + Fraction(b * b, c), "conic-directrix-uses-b-squared"),
    ]
    text = (
        f"The curve {equation_text(coeffs)} is {'an ellipse' if kind == 'ellipse' else 'a hyperbola'} with "
        f"{'vertical' if vertical else 'horizontal'} {'major axis' if kind == 'ellipse' else 'transverse axis'}. "
        f"Its {which} directrix is the line {axis} = d. Find d."
    )
    working = [
        f"Completing the square gives {_standard_text(kind, a, b, h, k, vertical)}, with a = {a}, c = {c} and centre ({h}, {k}).",
        f"The directrices are {axis} = {centre} ± a²/c = {centre} ± {rational_text(offset)}; the {which} one is {rational_text(value)}.",
    ]
    return dict(
        text=text,
        answer_kind="fraction",
        answer=_frac(value),
        answer_text=rational_text(value),
        answer_latex=rational_text(value),
        working=working,
        distractors=keep(rational_text(value), cands, 4),
    )


def _hard(rng):
    return rng.choice([_hard_parabola, _hard_directrix, _hard_directrix])(rng)


@generator(
    TOPIC,
    level=18,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@redraws
def u2_conic_features(rng, tier="easy"):
    """Features of a conic read from its equation; the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam ----------------------------------------------------------------------------------------------------------


@generator(EXAM_TOPIC, level=18, presets={"exam": dict(tier="exam")})
@redraws
def exd_x_u2_conic_features(rng, tier="exam"):
    """An ellipse in general form: its centre, c, the eccentricity and the distance from the centre to a directrix."""
    check(tier == "exam", f"unknown tier {tier!r}")
    a, b, c = rng.choice(_SMALL_ELLIPSE)
    vertical = rng.random() < 0.5
    h, k = _nonzero_centre(rng)
    coeffs = _on_curve("ellipse", a, b, h, k, vertical)
    _check_foci("ellipse", a, b, c, h, k, vertical)
    A, B, D, E, F = coeffs
    e = Fraction(c, a)
    d_dist = Fraction(a * a, c)
    centre_text = f"({h}, {k})"

    def pair_d(x, y, tag):
        return Distractor(f"({x}, {y})", tag, {"x": x, "y": y})

    a_cands = [
        pair_d(-h, -k, "conic-completing-square-sign"),
        pair_d(k, h, "conic-centre-swapped"),
        pair_d(Fraction(-D, 2), Fraction(-E, 2), "conic-centre-forgot-coefficient"),
        pair_d(Fraction(D, 2 * A), Fraction(E, 2 * B), "conic-completing-square-sign"),
    ]
    a_cands = [Distractor(d.text.replace("/", "/"), d.misconception, d.value) for d in a_cands]
    b_cands = [
        int_d(a, "conic-focal-distance-half"),
        int_d(b, "conic-focal-distance-half"),
        int_d(a * a - b * b, "conic-focal-distance-squared"),
        int_d(int((a * a + b * b) ** 0.5 + 0.5), "conic-ellipse-hyperbola-formula"),
    ]
    c_cands = [
        frac_d(Fraction(b, a), "conic-eccentricity-uses-b"),
        frac_d(Fraction(a, c), "conic-eccentricity-inverted"),
        frac_d(Fraction(c, b), "conic-eccentricity-denominator-b"),
    ]
    d_cands = [
        frac_d(Fraction(a, c), "conic-directrix-uses-a-over-c"),
        frac_d(Fraction(c * c, a), "conic-directrix-uses-c-squared"),
        frac_d(Fraction(b * b, c), "conic-directrix-uses-b-squared"),
    ]
    from ..answers import exact_value

    return dict(
        text=(
            f"The curve {equation_text(coeffs)} is an ellipse with {'vertical' if vertical else 'horizontal'} "
            "major axis."
        ),
        parts=[
            dict(
                prompt="Find the coordinates of the centre of the ellipse.",
                answer_kind="pair",
                answer={"x": h, "y": k},
                answer_text=centre_text,
                marks=3,
                working=[
                    f"Completing the square in x and in y gives {_standard_text('ellipse', a, b, h, k, vertical)}, so the centre is {centre_text}."
                ],
                distractors=keep(centre_text, a_cands),
            ),
            dict(
                prompt="Find the distance c from the centre to each focus.",
                answer_kind="number",
                answer=c,
                answer_text=str(c),
                marks=2,
                working=[f"The semi-axes are {a} and {b}, so c² = {a}² - {b}² = {c * c} and c = {c}."],
                distractors=keep(str(c), b_cands),
            ),
            dict(
                prompt="Find the eccentricity of the ellipse.",
                answer_kind="fraction",
                answer=exact_value(e),
                answer_text=rational_text(e),
                marks=2,
                working=[f"e = c/a = {c}/{a} = {rational_text(e)}."],
                distractors=keep(rational_text(e), c_cands),
                follow_through=f"part_b/{a}",
            ),
            dict(
                prompt="Find the distance from the centre to each directrix.",
                answer_kind="fraction",
                answer=exact_value(d_dist),
                answer_text=rational_text(d_dist),
                marks=3,
                working=[
                    f"The directrices are a distance a/e = a²/c = {a * a}/{c} = {rational_text(d_dist)} from the centre."
                ],
                distractors=keep(rational_text(d_dist), d_cands),
                follow_through=f"{a * a}/part_b",
            ),
        ],
    )
