"""Generates `r3h_separable_de` questions (easy, medium, hard) and the exam chain `exd_x_r3h_separable_de`.

``separable-de-particular-solution`` (``differential-equations``, answer ``expression``): solve ``dy/dx = g(x) h(y)``
by separating the variables, integrating, and using a point on the curve to fix the constant, then giving ``y``
explicitly. ``y`` is on the right-hand side of the equation, so the work is in integrating ``1/h(y)`` correctly and
in inverting the result. This is the senior-secondary skill: no partial fractions and no logistic-type quotients.

Each equation is built from its solution: ``H(y) = G(x) + C`` where ``H' = 1/h`` and ``G' = g``.

* easy: ``h(y) = y`` or ``h(y) = 1/y`` with ``g`` a power of x, through a point on the y-axis.
* medium: ``h`` is y, 1/y or e^(-y) and ``g`` is a trigonometric, exponential or reciprocal function.
* hard: ``h`` is y^2, 1 + y^2, 1/y or e^(-y) with ``g`` needing integration by parts, or y^2 and 1 + y^2 with a
  trigonometric ``g``.
* exam (``exd_x_r3h_separable_de``): ``dy/dx = g(x)/y`` or ``g(x) y^2`` in four parts: (a) the antiderivative of
  ``g``, (b) the constant in the separated form (follows through from (a)), (c) ``y(x)``, (d) ``y`` at a stated x
  to 3 decimal places (follows through from (c)).

The solution is checked by substituting it into the equation numerically at several points either side of the
initial point, and the initial condition is checked directly.
"""

from __future__ import annotations

import sympy

from ..answers import Distractor
from ..core import attempts, check, generator, pick_distractors, rounds_unambiguously
from ._r3h_kit import (
    _values,
    X,
    Redraw,
    antiderivative,
    build_with_redraws,
    expr_part,
    fraction_part,
    frac_text,
    machine,
    number_part,
    plain,
    real_on,
    same_function,
    say_marked,
    tidy_form,
)

x, y = X, sympy.Symbol("y")
_R = sympy.Rational
E = sympy.E


class _Fam:
    """h(y), the integral H of 1/h, its inverse, and the family's own characteristic wrong turns."""

    def __init__(self, name, h_text, h, H, inv, wrongs):
        self.name, self.h_text, self.h, self.H, self.inv, self.wrongs = name, h_text, h, H, inv, wrongs


def _wrap(text: str) -> str:
    return f"({text})" if (" " in text and not text.startswith("(")) else text


def _wrongs_exp(G, x0, y0, c):
    """ln y = G + C: the constant added after the exponential, h integrated instead of 1/h, the constant dropped."""
    y0, g0 = sympy.nsimplify(y0), G.subs(x, x0)
    return [
        ("r3h-separable-constant-after-inverse", sympy.exp(G) + (y0 - sympy.exp(g0))),
        ("r3h-separable-wrong-integral", sympy.sqrt(2 * (G + y0**2 / 2 - g0))),
        ("r3h-separable-constant-dropped", sympy.exp(G)),
    ]


def _wrongs_sqrt(G, x0, y0, c):
    """y^2/2 = G + C: the half lost, h integrated instead of 1/h, the constant dropped, the wrong root."""
    y0, g0 = sympy.nsimplify(y0), G.subs(x, x0)
    return [
        ("r3h-separable-integration-factor-lost", sympy.sqrt(G + y0**2 - g0)),
        ("r3h-separable-wrong-integral", y0 * sympy.exp(G - g0)),
        ("r3h-separable-constant-dropped", sympy.sqrt(2 * G)),
        ("r3h-separable-root-sign", -sympy.sqrt(2 * (G + c))),
    ]


def _wrongs_logexp(G, x0, y0, c):
    """e^y = G + C: h integrated instead of 1/h, the constant dropped, the logarithm taken term by term."""
    g0 = G.subs(x, x0)
    return [
        ("r3h-separable-wrong-integral", -sympy.log(-(G - 1 - g0))),
        ("r3h-separable-constant-dropped", sympy.log(G)),
        ("r3h-separable-constant-after-inverse", sympy.log(G) + (y0 - sympy.log(g0))),
    ]


def _wrongs_recip(G, x0, y0, c):
    """-1/y = G + C: the sign lost, the constant dropped, h integrated instead of 1/h."""
    y0, g0 = sympy.nsimplify(y0), G.subs(x, x0)
    return [
        ("r3h-separable-sign-lost", 1 / (G + 1 / y0 - g0)),
        ("r3h-separable-constant-dropped", -1 / G),
        ("r3h-separable-wrong-integral", sympy.real_root(3 * (G + y0**3 / 3 - g0), 3)),
    ]


def _wrongs_tan(G, x0, y0, c):
    """arctan y = G + C: the inverse applied backwards, the constant dropped, no inverse taken at all."""
    return [
        ("r3h-separable-inverse-backwards", sympy.atan(G + c)),
        ("r3h-separable-constant-dropped", sympy.tan(G)),
        ("r3h-separable-forgot-to-invert", G + c),
    ]


_FAMILIES = {
    "y": _Fam("y", "{g} y", y, sympy.log(y), sympy.exp, _wrongs_exp),
    "recip": _Fam("recip", "{g}/y", 1 / y, y**2 / 2, lambda r: sympy.sqrt(2 * r), _wrongs_sqrt),
    "exp": _Fam("exp", "{g} e^(-y)", sympy.exp(-y), sympy.exp(y), sympy.log, _wrongs_logexp),
    "sq": _Fam("sq", "{g} y^2", y**2, -1 / y, lambda r: -1 / r, _wrongs_recip),
    "tan": _Fam("tan", "{g} (1 + y^2)", 1 + y**2, sympy.atan(y), sympy.tan, _wrongs_tan),
}

_G_EASY = [lambda a, n: a * x**n]
_TRIG = [sympy.cos(x), sympy.sin(x)]
_EXPS = [sympy.exp(x), sympy.exp(-x), sympy.exp(2 * x)]
_PARTS = [x * sympy.exp(x), x * sympy.cos(x), x * sympy.sin(x), x * sympy.exp(-x)]


def _g_choices(tier):
    if tier == "easy":
        return [(a * x**n, 0) for a in (1, 2, 3, -1, -2) for n in (1, 2, 3)]
    if tier == "medium":
        out = [(a * g, 0) for g in _TRIG + _EXPS for a in (1, 2, 3)]
        out += [(a / x, 1) for a in (1, 2, 3)]
        return out
    if tier == "hard":
        return [(a * g, 0) for g in _PARTS for a in (1, 2)] + [(a * g, 0) for g in _TRIG for a in (1, 2, 3)]
    return [(a * x**n, 0) for a in (1, 2, 3) for n in (1, 2)] + [(a * g, 0) for g in _TRIG for a in (1, 2, 3)]


_FAMILY_BY_TIER = {
    "easy": ["y", "recip"],
    "medium": ["y", "recip", "exp"],
    "hard": ["sq", "tan", "recip", "exp"],
    "exam": ["recip", "sq"],
}
_Y0 = {
    "y": [1, 2, 3, 4, _R(1, 2)],
    "recip": [1, 2, 3, 4],
    "exp": [0],
    "sq": [1, 2, -1, -2, _R(1, 2)],
    "tan": [0],
}


class _Eq:
    def __init__(self, fam, g, x0, y0, G, c, sol):
        self.fam, self.g, self.x0, self.y0, self.G, self.c, self.sol = fam, g, x0, y0, G, c, sol

    @property
    def rhs(self):
        if self.fam.name == "recip":
            return plain(self.g / y)
        return self.fam.h_text.format(g=_wrap(plain(self.g)))


def _points(x0):
    return [x0 + _R(k, 10) for k in (-2, 1, 2, 3, 4, 5)]


def _draw(rng, tier, family=None):
    for _ in attempts(600, "a separable equation with a clean explicit solution"):
        fam = _FAMILIES[family or rng.choice(_FAMILY_BY_TIER[tier])]
        g, shifted = rng.choice(_g_choices(tier))
        # hard questions that put g(x) over y or e^y must need integration by parts, so they differ from medium
        if tier == "hard" and fam.name in ("recip", "exp") and x not in sympy.Mul.make_args(g.as_independent(x)[1]):
            continue
        x0 = sympy.Integer(1 if shifted else 0)
        if fam.name == "exp":
            y0 = 0
        else:
            y0 = rng.choice(_Y0[fam.name])
        G = antiderivative(g)
        if G.has(sympy.Integral):
            continue
        # H(y0) = G(x0) + c
        c = sympy.simplify(sympy.nsimplify(fam.H.subs(y, y0)) - G.subs(x, x0))
        if c.has(sympy.zoo, sympy.nan) or (fam.name != "y" and not c.is_Rational):
            continue
        try:
            sol = fam.inv(G + c)
        except (TypeError, ValueError):
            continue
        # real and finite on a window around the start, and the branch through (x0, y0)
        pts = tuple(_points(x0))
        if not real_on(sol, x, pts, 60) or any(abs(v.real) < 1e-6 for v in _values(sol, x, pts)):
            continue
        if sympy.N(sol.subs(x, x0) - y0, 20) != 0 and abs(sympy.N(sol.subs(x, x0) - y0, 20)) > 1e-12:
            continue
        if fam.name in ("recip", "y") and y0 <= 0:
            continue
        if sol.is_polynomial(x):
            continue
        return _Eq(fam, g, x0, sympy.nsimplify(y0), G, c, sol)
    raise AssertionError("unreachable")


def _residual_ok(eq: _Eq) -> bool:
    sol = eq.sol
    lhs = sympy.diff(sol, x)
    rhs = eq.g * eq.fam.h.subs(y, sol)
    return same_function(lhs, rhs, x, tuple(_points(eq.x0))) and abs(sympy.N(sol.subs(x, eq.x0) - eq.y0, 20)) < 1e-12


def _options(answer, wrongs, eq):
    kept = []
    pts = tuple(_points(eq.x0)[1:4])
    for tag, w in wrongs:
        w = sympy.sympify(w)
        if w.has(sympy.zoo, sympy.nan, sympy.oo, sympy.I):
            continue
        try:
            w = tidy_form(w)
            text = plain(w)
            if any(bad in text for bad in ("Piecewise", "Abs(", "re(", "im(", "sign(")) or text.count("sqrt") > 1:
                continue
            if not real_on(w, x, tuple(_points(eq.x0)[1:])):
                continue
            if same_function(w, answer, x, pts) or any(same_function(w, o, x, pts) for _, o in kept):
                continue
        except (TypeError, ValueError, ZeroDivisionError, OverflowError):
            continue
        kept.append((tag, w))
    return pick_distractors(plain(answer), [Distractor(plain(w), tag, machine(w)) for tag, w in kept], 4)


def _stem(eq: _Eq, rng) -> tuple[str, str]:
    x0, y0 = eq.x0, eq.y0
    return say_marked(
        rng.choice(
            [
                "Solve the differential equation «dy/dx = {rhs}» given that «y = {y0}» when «x = {x0}». Give «y» in terms of «x».",
                "A curve satisfies «dy/dx = {rhs}» and passes through the point «({x0}, {y0})». Find «y» as a function of «x».",
            ]
        ).format(rhs=eq.rhs, x0=x0, y0=frac_text(y0))
    )


@generator(
    "r3h_separable_de",
    level=18,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r3h_separable_de(rng, tier="easy"):
    for _ in attempts(80, "a separable question with enough distractors"):
        eq = _draw(rng, tier)
        check(_residual_ok(eq), "the solution does not satisfy the equation")
        sol = tidy_form(eq.sol)
        if not same_function(sol, eq.sol, x, tuple(_points(eq.x0))):
            continue  # the tidy rewrite changed the function (a branch of a root): draw again
        dist = _options(sol, eq.fam.wrongs(eq.G, eq.x0, eq.y0, eq.c), eq)
        if len(dist) >= 2:
            break
    stem, latex = _stem(eq, rng)
    return dict(
        text=stem,
        latex=latex,
        answer=machine(sol),
        answer_text=plain(sol),
        answer_kind="expression",
        distractors=dist,
        working=[
            f"Separate the variables and integrate: H(y) = {plain(eq.G)} + C.",
            f"Use the point ({eq.x0}, {frac_text(eq.y0)}) to find C = {plain(eq.c)}, so y = {plain(sol)}.",
        ],
    )


# ---------------------------------------------------------------------------------------------------------------
# exam chain


def _exam_once(rng):
    eq = _draw(rng, "exam")
    check(_residual_ok(eq), "the solution does not satisfy the equation")
    sol = tidy_form(eq.sol)
    fam, G, g = eq.fam, eq.G, eq.g
    x1 = eq.x0 + rng.choice([_R(1, 2), _R(1), _R(3, 2)]) if eq.x0 == 0 else eq.x0 + rng.choice([_R(1, 2), _R(1)])
    try:
        value = float(sympy.N(sol.subs(x, x1), 30))
    except (TypeError, ValueError, ZeroDivisionError):
        raise Redraw
    if abs(sympy.im(sympy.N(sol.subs(x, x1), 20))) > 1e-12 or not (0.05 <= abs(value) <= 40):
        raise Redraw
    if not rounds_unambiguously(value, 3, 0.1):
        raise Redraw
    # the separated form shown to the learner, with H(y) known and G(x) not
    sep = "y^2/2" if fam.name == "recip" else "-1/y"
    c = eq.c
    x1_text = f"{float(x1):g}"
    x1_ft = frac_text(x1)
    ga = G.subs(x, eq.x0)
    # (a) the antiderivative G with constant zero
    gp = _wrap(plain(g))
    a_wrongs = [
        ("r3h-antiderivative-differentiated", sympy.diff(g, x)),
        ("r3h-antiderivative-sign", -G),
        ("r3h-antiderivative-times-x", x * g),
    ]
    # (b) the constant
    h0 = fam.H.subs(y, eq.y0)
    c_wrongs = [
        ("r3h-separable-constant-sign", -c),
        ("r3h-separable-constant-ignores-antiderivative", h0),
        ("r3h-separable-constant-adds-antiderivative", h0 + ga),
        ("r3h-separable-constant-uses-integrand", h0 - g.subs(x, eq.x0)),
        ("r3h-separable-constant-is-antiderivative-value", ga),
    ]
    # (c) the solution
    sol_wrongs = fam.wrongs(G, eq.x0, eq.y0, c)
    sol_options = _options(sol, sol_wrongs, eq)
    # (d) the value
    d_wrongs = []
    for d in sol_options:
        try:
            val = _values(sympy.sympify(d.value), x, [x1])[0]
        except (TypeError, ValueError, ZeroDivisionError):
            continue
        if abs(val.imag) < 1e-12:
            d_wrongs.append((d.misconception, val.real))
    stem, latex = say_marked(
        f"A curve satisfies «dy/dx = {eq.rhs}» and passes through the point «({eq.x0}, {frac_text(eq.y0)})»."
    )
    sub = f"Subs(part_c, (x, e), ({x1_ft}, E))"
    parts = [
        expr_part(
            "The right-hand side is a product of a function of «x» alone, «g(x)», and a function of «y» alone. Find «G(x) = ∫ g(x) dx», taking the constant of integration to be 0.",
            G,
            2,
            [f"g(x) = {gp}, so G(x) = {plain(G)}."],
            a_wrongs,
        ),
        fraction_part(
            f"Separating the variables and integrating gives «{sep} = G(x) + C». Use the point «({eq.x0}, {frac_text(eq.y0)})» to find «C».",
            c,
            1,
            [f"C = {frac_text(h0)} - ({plain(ga)}) = {frac_text(c)}."],
            c_wrongs,
            follow_through=f"({frac_text(h0)}) - Subs(part_a, (x, e), ({frac_text(eq.x0)}, E))",
        ),
        expr_part(
            "Hence find «y» as a function of «x».", sol, 3, [f"y = {plain(sol)}."], sol_wrongs, options=sol_options
        ),
        number_part(
            "Find «y» when «x = {x1}», correct to 3 decimal places.",
            value,
            3,
            1,
            [f"y({x1_text}) = {value:.3f}."],
            d_wrongs,
            follow_through=sub,
            sensible_range=(0, None) if value > 0 and eq.y0 > 0 else None,
            x1=x1_text,
        ),
    ]
    # reuse the sol option list for part (c) (built by expr_part from the same wrongs); sanity-check it
    check(len(parts[2]["distractors"]) >= 2, "too few distractors for the solution")
    return dict(text=stem, latex=latex, parts=parts)


@generator("exd_x_r3h_separable_de", level=18, presets={"exam": dict(tier="exam")})
def exd_x_r3h_separable_de(rng, tier="exam"):
    """Exam tier: separable equation, antiderivative, constant, solution, value."""
    check(tier == "exam", f"unknown tier {tier!r}")
    return build_with_redraws(rng, _exam_once)
