"""HSC trigonometric substitution: `nx_trig_sub_hsc` (easy, medium, hard) and the exam chain `exd_nx_trig_sub_hsc`.

Skill ``trig-substitution-hsc-tan-and-convert`` (``integration``, answer ``expression``). The university skill
``ucv1-trig-substitution-integrals`` and the Year 12 ``definite-integral-substitution-changed-limits`` (x = a sin(theta) for
the root and x^2/root from 0 up to a sin(alpha)) already exist; this one is the Year 12 framing of the whole family, with
the tangent substitution, the conversion steps on their own, and lower limits that are not 0.

* easy: given the substitution, (a) the constant k in the converted integrand (k dtheta, k cos^2(theta) dtheta, ...), (b) the
  constant in a piece of the conversion (dx, a^2 + x^2, a^2 - x^2, the root), or (c) the new lower or upper limit.
* medium: the exact value, the substitution given: 1/(a^2 + x^2) between limits a tan(alpha), 1/sqrt(a^2 - x^2) between
  limits a sin(alpha), and the quarter circle sqrt(a^2 - x^2) from 0 to a.
* hard: the exact value, no substitution given: 1/(a^2 + x^2)^2, sqrt(a^2 - x^2) and x^2/sqrt(a^2 - x^2) between two limits,
  and 1/(a^2 - x^2)^(3/2).
* exam (``exd_nx_trig_sub_hsc``): the integral of 1/(a^2 + x^2)^2 (x = a tan(theta)) or of sqrt(a^2 - x^2) (x = a sin(theta))
  from 0 to b: (a) the constant k with k cos^2(theta) dtheta, (b) the new upper limit, (c) the integral of cos^2(theta) up to
  that limit, following through from (b), (d) the value, following through from (a) and (c).

Every exact value is compared with Simpson's rule on the original integral; every constant k is re-derived by substituting
the substitution into the integrand and dividing out the trigonometric part at sample angles.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction

import sympy
from sympy import Rational as R
from sympy import atan, cos, pi, sin, sqrt, tan

from ..core import check, generator, latex_from_inline
from ._r3b_kit import (
    Cand,
    Itg,
    Redraw,
    exact_fields,
    integral_math,
    integral_text,
    part,
    pick_expr,
    retry,
    say,
    x,
)
from ._spread_kit import NEAR, mixed, uses_rng

th = sympy.Symbol("th", positive=True)

_ALPHAS = (R(0), pi / 6, pi / 4, pi / 3)


def _tidy(line: str) -> str:
    line = re.sub(r"sqrt\((\d+)\)", r"√\1", line)
    line = re.sub(r"(?<=[\d)])\*(?=[A-Za-z√(])", "", line)
    line = re.sub(r"(?<![A-Za-z])pi(?![A-Za-z])", "π", line)
    return line.replace("*", " × ")


def _lim(v) -> str:
    """A limit of integration in a stem or working step."""
    from ._r3b_kit import shown

    return _tidy(shown(sympy.sympify(v)))


def _neat(v):
    """An exact value with the fractions gathered: (2*pi - 3*sqrt(3))/24 rather than a nest of brackets."""
    return sympy.factor_terms(sympy.together(sympy.expand(sympy.simplify(v))))


def _sc(a):
    return sin(a) * cos(a)


# -- the integrals ------------------------------------------------------------------------------------------------


class Kind:
    """One integrand: its text, its antiderivative in theta and the angle family (``tan`` or ``sin``)."""

    def __init__(self, name, fam, f, text, latex_fn, G, piece, k_value, form):
        self.name, self.fam, self.f, self.text, self.G = name, fam, f, text, G
        self.latex_fn, self.piece, self.k_value, self.form = latex_fn, piece, k_value, form


def _integrand(kind: str, a):
    sa = a * a
    if kind == "tan1":
        return 1 / (sa + x**2), f"1/({sa} + x^2)"
    if kind == "tan2":
        return 1 / (sa + x**2) ** 2, f"1/({sa} + x^2)^2"
    if kind == "sinroot":
        return sqrt(sa - x**2), f"√({sa} - x^2)"
    if kind == "sinx2":
        return x**2 / sqrt(sa - x**2), f"x^2/√({sa} - x^2)"
    if kind == "sin32":
        return 1 / (sa - x**2) ** R(3, 2), f"1/({sa} - x^2)^(3/2)"
    if kind == "sininv":
        return 1 / sqrt(sa - x**2), f"1/√({sa} - x^2)"
    raise KeyError(kind)  # pragma: no cover


_FAMILY = {"tan1": "tan", "tan2": "tan", "sinroot": "sin", "sinx2": "sin", "sin32": "sin", "sininv": "sin"}


def _G(kind: str, a, t):
    """The antiderivative in theta (the integrand after substitution is integrated), evaluated at the angle t."""
    if kind == "tan1":
        return t / a
    if kind == "tan2":
        return (t + _sc(t)) / (2 * a**3)
    if kind == "sinroot":
        return a * a * (t + _sc(t)) / 2
    if kind == "sinx2":
        return a * a * (t - _sc(t)) / 2
    if kind == "sin32":
        return tan(t) / (a * a)
    if kind == "sininv":
        return t
    raise KeyError(kind)  # pragma: no cover


# constant k and trigonometric form of the converted integrand: integrand * dx = k * form(theta) dtheta
_CONVERTED = {
    "tan1": (lambda a: 1 / a, "1", None),
    "tan2": (lambda a: 1 / a**3, "cos^2(θ)", cos(th) ** 2),
    "sinroot": (lambda a: a * a, "cos^2(θ)", cos(th) ** 2),
    "sinx2": (lambda a: a * a, "sin^2(θ)", sin(th) ** 2),
    "sin32": (lambda a: 1 / (a * a), "sec^2(θ)", 1 / cos(th) ** 2),
    "sininv": (lambda a: R(1), "1", None),
}


def _sub_expr(fam, a):
    return a * tan(th) if fam == "tan" else a * sin(th)


def _check_converted(kind, a):
    """Second route to k: f(x(theta)) dx/dtheta, divided by the stated trigonometric form, is the constant k."""
    f, _ = _integrand(kind, a)
    fam = _FAMILY[kind]
    xs = _sub_expr(fam, a)
    k_fn, _, form = _CONVERTED[kind]
    k = k_fn(a)
    for t in (0.3, 0.7, 1.1):
        g = f.subs(x, xs) * sympy.diff(xs, th)
        got = float(sympy.N(g.subs(th, t)))
        want = float(sympy.N(k * (form.subs(th, t) if form is not None else 1)))
        check(math.isclose(got, want, rel_tol=1e-9), f"converted integrand differs for {kind}")


def _hint(fam, a):
    coef = "" if a == 1 else str(a)
    name = "tan" if fam == "tan" else "sin"
    return f"x = {coef}{name}(θ)", rf"\(x = {coef}\{name}\theta\)"


def _fn_angle(fam, v, a):
    """The angle whose a-fold trig value is v."""
    r = sympy.nsimplify(v / a)
    return sympy.atan(r) if fam == "tan" else sympy.asin(r)


def _xval(fam, a, alpha):
    """The x-limit that a substitution of family `fam` sends to the angle alpha."""
    return sympy.simplify(a * (tan(alpha) if fam == "tan" else sin(alpha)))


# -- stems --------------------------------------------------------------------------------------------------------

_EXACT = (
    "Find the exact value of {I}.",
    "Evaluate {I}, giving an exact answer.",
)
_EXACT_HINT = (
    "Use the substitution {H} to find the exact value of {I}.",
    "Find the exact value of {I} by using the substitution {H}.",
    "Evaluate {I} exactly. Use the substitution {H}.",
)
_EXACT_FREE = (
    "Find the exact value of {I}. Choose a suitable trigonometric substitution.",
    "Evaluate {I}, giving an exact answer. A trigonometric substitution will help.",
    "By a suitable trigonometric substitution, find the exact value of {I}.",
)
_K_STEMS = (
    "Using the substitution {H}, the integral {I} becomes {J}, where k is a constant. Find k.",
    "The substitution {H} turns {I} into {J} for a constant k. What is k?",
    "After the substitution {H}, {I} can be written as {J}. Find the constant k.",
)
_LIMIT_STEMS = (
    "The integral {I} is evaluated using the substitution {H}. Find the new {which} limit for θ.",
    "A student uses {H} on {I}. What {which} limit of θ replaces the limit of x? Give it in radians, exactly.",
    "Use {H} to change the limits of {I} from x to θ. State the {which} limit in radians.",
)
_PIECE_STEMS = (
    "With the substitution {H}, {P} Find k.",
    "The substitution {H} gives {P} What is the constant k?",
)


def _integral(rng, f_text, f_value, lo=None, hi=None):
    f = Itg(f_value, text=f_text)
    if lo is None:
        return _tidy(integral_text(f)), integral_math(f)
    return _tidy(integral_text(f, lo, hi)), integral_math(f, lo, hi)


# -- options ------------------------------------------------------------------------------------------------------


def _quad(value, f, lo, hi):
    """Second route to an exact definite integral: adaptive Gauss-Legendre / tanh-sinh quadrature of the integrand as
    given in x (the integrands here may have an infinite slope or a blow-up at an end point, where Simpson's rule fails)."""
    import mpmath

    g = sympy.lambdify(x, f, "mpmath")
    approx = mpmath.quad(g, [mpmath.mpf(str(sympy.N(lo, 30))), mpmath.mpf(str(sympy.N(hi, 30)))])
    exact = mpmath.mpf(str(sympy.N(value, 30)))
    check(abs(approx - exact) < mpmath.mpf("1e-9") * max(1, abs(exact)), f"quadrature {approx} disagrees with {exact}")


def _span(f, lo, hi):
    """Bounds for the options of a definite integral: positive, and at most three times the largest the integrand allows."""
    import mpmath

    g = sympy.lambdify(x, f, "mpmath")
    top = mpmath.quad(g, [mpmath.mpf(str(sympy.N(lo, 20))), mpmath.mpf(str(sympy.N(hi, 20)))])
    return R(0), R(int(float(top) * 300), 100) + 1


def _pick_values(key, cands):
    """Three wrong exact values (value, tag pairs) for a constant such as 1/a^3."""
    key = sympy.sympify(key)
    near = [Cand(key * m, NEAR, const=False) for m in (R(1, 2), R(2), R(3), R(1, 3), R(3, 2), R(2, 3), R(4))]
    near += [Cand(key + d, NEAR, const=False) for d in (R(1, 2), R(1), R(2), R(1, 3))]
    named = [Cand(v, t, const=False) for v, t in cands if v is not None]
    out = pick_expr(key, False, mixed(key, named, fillers=near, lo=0, hi=None))
    # the key must not be the only whole number (or the only fraction) among its options
    if not any((sympy.sympify(d.text).is_integer) == key.is_integer for d in out):
        raise Redraw("the key is the only option of its shape")
    return out


def _angle_options(key, cands):
    """Three wrong angles for a limit of theta: between 0 and a quarter turn (a limit past pi/2 cannot arise from
    x = a tan or a sin of a positive angle that the question allows), with the key's place among them drawn at random."""
    key = sympy.sympify(key)
    near = [
        Cand(key + d, NEAR, const=False)
        for d in (pi / 12, -pi / 12, pi / 6, -pi / 6, pi / 24, -pi / 24, pi / 8, -pi / 8)
    ]
    cands = [Cand(c.expr, c.tag, const=False) for c in cands if c.expr != 0]
    return pick_expr(key, False, mixed(key, cands, fillers=near, lo=0, hi=pi / 2))


def _no_const(cands):
    return [Cand(c.expr, c.tag, const=False) for c in cands if c is not None]


def _options(key, cands, rng, *, lo, hi):
    key = sympy.sympify(key)
    near = [Cand(key + R(k, d), NEAR, const=False) for d in (2, 3, 4, 6, 8, 12) for k in (-1, 1)]
    near += [Cand(key * m, NEAR, const=False) for m in (R(1, 2), R(2, 3), R(3, 2), 2)]
    return pick_expr(key, False, mixed(key, _no_const(cands), fillers=near, lo=lo, hi=hi))


def _value_slips(kind, a, alpha, beta, lo_x, hi_x):
    """Named wrong exact values for the definite integral from angle alpha to beta."""
    G = lambda t: _G(kind, a, t)  # noqa: E731
    out = []
    if kind == "tan1":
        out = [
            Cand((beta - alpha), "nx-tsub-reciprocal-a-dropped"),
            Cand((beta - alpha) / a**2, "nx-tsub-dx-factor-a-dropped"),
            Cand((hi_x - lo_x) / a, "nx-tsub-limits-not-changed"),
            Cand(a * (beta - alpha), "nx-tsub-a-multiplied-not-divided"),
            Cand(G(beta) if alpha != 0 else None, "nx-tsub-lower-limit-ignored"),
            Cand(-(beta - alpha) / a, "nx-tsub-limits-swapped"),
        ]
    elif kind == "tan2":
        out = [
            Cand((beta + _sc(beta) - alpha - _sc(alpha)) / (2 * a), "nx-tsub-power-of-a-wrong"),
            Cand((beta + _sc(beta) - alpha - _sc(alpha)) / (2 * a**4), "nx-tsub-dx-factor-a-dropped"),
            Cand((beta + _sc(beta) - alpha - _sc(alpha)) / a**3, "nx-tsub-half-dropped"),
            Cand((beta - _sc(beta) - alpha + _sc(alpha)) / (2 * a**3), "nx-tsub-identity-sign"),
            Cand((beta - alpha) / (2 * a**3), "nx-tsub-identity-term-dropped"),
            Cand((beta - alpha) / a, "nx-tsub-power-not-squared"),
            Cand(G(beta) if alpha != 0 else None, "nx-tsub-lower-limit-ignored"),
        ]
    elif kind in ("sinroot", "sinx2"):
        sgn = 1 if kind == "sinroot" else -1
        base = lambda t: t + sgn * _sc(t)  # noqa: E731
        out = [
            Cand(a * (base(beta) - base(alpha)) / 2, "nx-tsub-amplitude-not-squared"),
            Cand(a * a * (beta - sgn * _sc(beta) - alpha + sgn * _sc(alpha)) / 2, "nx-tsub-identity-sign"),
            Cand(a * a * (beta - alpha) / 2, "nx-tsub-identity-term-dropped"),
            Cand(a * a * (base(beta) - base(alpha)), "nx-tsub-half-dropped"),
            Cand(G(beta) if alpha != 0 else None, "nx-tsub-lower-limit-ignored"),
            Cand(a**3 * (base(beta) - base(alpha)) / 2, "nx-tsub-amplitude-cubed"),
        ]
    elif kind == "sin32":
        out = [
            Cand((sin(beta) - sin(alpha)) / a**2, "nx-tsub-sec-squared-integrated-as-sin"),
            Cand((tan(beta) - tan(alpha)) / a, "nx-tsub-power-of-a-wrong"),
            Cand((beta - alpha) / a**2, "nx-tsub-sec-squared-integrated-as-theta"),
            Cand(tan(beta) - tan(alpha), "nx-tsub-reciprocal-a-dropped"),
            Cand(G(beta) if alpha != 0 else None, "nx-tsub-lower-limit-ignored"),
        ]
    else:  # sininv
        out = [
            Cand((beta - alpha) / a, "nx-tsub-reciprocal-a-added"),
            Cand(pi / 2 - beta - (pi / 2 - alpha), "nx-tsub-limits-swapped"),
            Cand(beta if alpha != 0 else None, "nx-tsub-lower-limit-ignored"),
            Cand(a * (beta - alpha), "nx-tsub-a-multiplied-not-divided"),
            Cand(pi / 2 - beta, "nx-tsub-complement-angle"),
        ]
    return [c for c in out if c.expr is not None]


def _make_definite(rng, kind, a, alpha, beta, text_style):
    """A definite-integral question for `kind` between the angles alpha and beta (the x-limits are derived)."""
    fam = _FAMILY[kind]
    f, f_text = _integrand(kind, a)
    lo_x, hi_x = _xval(fam, a, alpha), _xval(fam, a, beta)
    value = _neat(_G(kind, a, beta) - _G(kind, a, alpha))
    _quad(value, f, lo_x, hi_x)
    _check_converted(kind, a)
    floor, ceiling = _span(f, lo_x, hi_x)
    ints_I, ints_M = _integral(rng, f_text, f, lo_x, hi_x)
    H = _hint(fam, a)
    cands = _value_slips(kind, a, alpha, beta, lo_x, hi_x)
    if text_style == "hint":
        text, latex = say(rng, _EXACT_HINT, {"I": ints_I, "H": H[0]}, {"I": ints_M, "H": H[1]})
    elif text_style == "free":
        text, latex = say(rng, _EXACT_FREE, {"I": ints_I}, {"I": ints_M})
    else:
        text, latex = say(rng, _EXACT, {"I": ints_I}, {"I": ints_M})
    k_fn, form_text, _ = _CONVERTED[kind]
    k = k_fn(a)
    ang = lambda t: _lim(t)  # noqa: E731
    sub = f"x = {'' if a == 1 else a}{'tan' if fam == 'tan' else 'sin'}(θ)"
    conv = f"{'' if a == 1 else a}{'sec^2(θ)' if fam == 'tan' else 'cos(θ)'}"
    lead = " ".join(part for part in (str(k) if k != 1 else "", form_text if form_text != "1" else "") if part) or "1"
    working = [
        f"Let {sub}, so dx = {conv} dθ.",
        f"When x = {_lim(lo_x)}, θ = {ang(alpha)}; when x = {_lim(hi_x)}, θ = {ang(beta)}.",
        f"The integral becomes the integral of {lead} dθ from {ang(alpha)} to {ang(beta)}.",
        f"Its value is {_lim(value)}.",
    ]
    return dict(
        text=text,
        latex=latex,
        working=[_tidy(w) for w in working],
        distractors=_options(value, cands, rng, lo=floor, hi=ceiling),
        **exact_fields(value),
    )


# -- easy ---------------------------------------------------------------------------------------------------------

_K_KINDS = ("tan1", "tan2", "sinroot", "sinx2", "sin32")
_FORM_LATEX = {
    "1": r"\int k\,\mathrm{d}\theta",
    "cos^2(θ)": r"\int k\cos^2\theta\,\mathrm{d}\theta",
    "sin^2(θ)": r"\int k\sin^2\theta\,\mathrm{d}\theta",
    "sec^2(θ)": r"\int k\sec^2\theta\,\mathrm{d}\theta",
}


def _easy_k(rng):
    kind = rng.choice(_K_KINDS)
    fam = _FAMILY[kind]
    a = sympy.Integer(rng.choice([2, 3, 4, 5, 6] if kind != "sin32" else [2, 3, 4, 5]))
    f, f_text = _integrand(kind, a)
    _check_converted(kind, a)
    k_fn, form, _ = _CONVERTED[kind]
    k = k_fn(a)
    H = _hint(fam, a)
    I_text, I_math = _integral(rng, f_text, f)
    J_text = f"∫ k {form} dθ" if form != "1" else "∫ k dθ"
    J_math = rf"\({_FORM_LATEX[form]}\)"
    text, latex = say(rng, _K_STEMS, {"I": I_text, "H": H[0], "J": J_text}, {"I": I_math, "H": H[1], "J": J_math})
    cands = {
        "tan1": [
            (a, "nx-tsub-a-multiplied-not-divided"),
            (1 / a**2, "nx-tsub-dx-factor-a-dropped"),
            (R(1), "nx-tsub-reciprocal-a-dropped"),
            (a**2, "nx-tsub-substitution-not-squared"),
        ],
        "tan2": [
            (1 / a**4, "nx-tsub-dx-factor-a-dropped"),
            (1 / a**2, "nx-tsub-power-of-a-wrong"),
            (1 / a, "nx-tsub-power-not-squared"),
            (a**3, "nx-tsub-a-multiplied-not-divided"),
        ],
        "sinroot": [
            (a, "nx-tsub-amplitude-not-squared"),
            (a**3, "nx-tsub-amplitude-cubed"),
        ],
        "sinx2": [
            (a, "nx-tsub-amplitude-not-squared"),
            (a**3, "nx-tsub-amplitude-cubed"),
        ],
        "sin32": [
            (1 / a, "nx-tsub-power-of-a-wrong"),
            (a**2, "nx-tsub-a-multiplied-not-divided"),
            (1 / a**3, "nx-tsub-dx-factor-a-dropped"),
            (R(1), "nx-tsub-reciprocal-a-dropped"),
        ],
    }[kind]
    sub = f"x = {a}{'tan' if fam == 'tan' else 'sin'}(θ)"
    working = [
        f"With {sub}, dx = {a}{'sec^2(θ)' if fam == 'tan' else 'cos(θ)'} dθ.",
        {
            "tan1": f"The denominator {a * a} + x^2 = {a * a} sec^2(θ), so the integrand is 1/({a * a} sec^2(θ)) × {a} sec^2(θ) = 1/{a}.",
            "tan2": f"The denominator ({a * a} + x^2)^2 = {a**4} sec^4(θ), so the integrand is {a} sec^2(θ)/({a**4} sec^4(θ)) = (1/{a**3}) cos^2(θ).",
            "sinroot": f"√({a * a} - x^2) = {a}cos(θ), so the integrand is {a}cos(θ) × {a}cos(θ) = {a * a} cos^2(θ).",
            "sinx2": f"x^2 = {a * a} sin^2(θ) and √({a * a} - x^2) = {a}cos(θ), so the integrand is {a * a} sin^2(θ)/({a}cos(θ)) × {a}cos(θ) = {a * a} sin^2(θ).",
            "sin32": f"({a * a} - x^2)^(3/2) = {a**3} cos^3(θ), so the integrand is {a}cos(θ)/({a**3} cos^3(θ)) = (1/{a * a}) sec^2(θ).",
        }[kind],
        f"So k = {k}.",
    ]
    return dict(
        text=text,
        latex=latex,
        working=[_tidy(w) for w in working],
        distractors=_pick_values(k, cands),
        **exact_fields(k),
    )


_PIECES = (
    # (family, plain claim, latex claim, k as function of a)
    (
        "tan",
        "dx = k sec^2(θ) dθ.",
        r"\(\mathrm{d}x = k\sec^2\theta\,\mathrm{d}\theta\).",
        lambda a: a,
        [("a2", "nx-tsub-substitution-not-squared"), ("one", "nx-tsub-derivative-factor-wrong")],
    ),
    (
        "sin",
        "dx = k cos(θ) dθ.",
        r"\(\mathrm{d}x = k\cos\theta\,\mathrm{d}\theta\).",
        lambda a: a,
        [("a2", "nx-tsub-substitution-not-squared"), ("one", "nx-tsub-derivative-factor-wrong")],
    ),
    (
        "tan",
        "{S} + x^2 = k sec^2(θ).",
        r"\({S} + x^2 = k\sec^2\theta\).",
        lambda a: a * a,
        [("a", "nx-tsub-amplitude-not-squared"), ("a3", "nx-tsub-amplitude-cubed")],
    ),
    (
        "sin",
        "{S} - x^2 = k cos^2(θ).",
        r"\({S} - x^2 = k\cos^2\theta\).",
        lambda a: a * a,
        [("a", "nx-tsub-amplitude-not-squared"), ("a3", "nx-tsub-amplitude-cubed")],
    ),
    (
        "sin",
        "√({S} - x^2) = k cos(θ).",
        r"\(\sqrt{{S} - x^2} = k\cos\theta\).",
        lambda a: a,
        [("a2", "nx-tsub-substitution-not-squared"), ("one", "nx-tsub-derivative-factor-wrong")],
    ),
    (
        "tan",
        "√({S} + x^2) = k sec(θ).",
        r"\(\sqrt{{S} + x^2} = k\sec\theta\).",
        lambda a: a,
        [("a2", "nx-tsub-substitution-not-squared"), ("one", "nx-tsub-derivative-factor-wrong")],
    ),
)


def _easy_piece(rng):
    fam, plain, latex_claim, k_fn, wrong = rng.choice(_PIECES)
    a = sympy.Integer(rng.choice([2, 3, 4, 5, 6, 7]))
    k = k_fn(a)
    S = a * a
    claim = plain.replace("{S}", str(S))
    lclaim = latex_claim.replace("{S}", str(S))
    H = _hint(fam, a)
    pattern = rng.choice(_PIECE_STEMS)
    text = pattern.format(H=H[0], P=claim)
    latex = latex_from_inline(pattern.format(H=H[1], P=lclaim))
    values = {"a2": a * a, "one": R(1), "a": a, "a3": a**3}
    cands = [(values[v] if v != "a" or k != a else None, t) for v, t in wrong]
    cands += [(2 * a, "nx-tsub-derivative-factor-wrong"), (a + 1, "nx-tsub-derivative-factor-wrong")]
    sub = f"x = {a}{'tan' if fam == 'tan' else 'sin'}(θ)"
    working = [f"With {sub}: x^2 = {a * a}{'tan' if fam == 'tan' else 'sin'}^2(θ).", f"So k = {k}."]
    if "dx" in claim:
        working = [f"Differentiate {sub}: dx = {a}{'sec^2(θ)' if fam == 'tan' else 'cos(θ)'} dθ.", f"So k = {k}."]
    elif "-" in claim:
        working = (
            [f"{S} - x^2 = {S} - {S}sin^2(θ) = {S}cos^2(θ).", f"So k = {k}."]
            if "√" not in claim
            else [
                f"{S} - x^2 = {S}cos^2(θ), so the root is {a}cos(θ) (cos(θ) is positive for 0 ≤ θ ≤ π/2).",
                f"So k = {k}.",
            ]
        )
    else:
        working = (
            [f"{S} + x^2 = {S} + {S}tan^2(θ) = {S}sec^2(θ).", f"So k = {k}."]
            if "√" not in claim
            else [f"{S} + x^2 = {S}sec^2(θ), so the root is {a}sec(θ).", f"So k = {k}."]
        )
    return dict(
        text=text,
        latex=latex,
        working=working,
        distractors=_pick_values(k, cands),
        **exact_fields(k),
    )


def _easy_limit(rng):
    kind = rng.choice(["tan1", "tan2", "sinroot", "sininv", "sin32"])
    fam = _FAMILY[kind]
    a = sympy.Integer(rng.choice([3, 6] if fam == "tan" else [2, 3, 4, 5, 6]))
    alphas = (
        (R(0), pi / 6, pi / 4, pi / 3)
        if kind in ("tan1", "tan2", "sininv", "sin32")
        else (R(0), pi / 6, pi / 4, pi / 3, pi / 2)
    )
    if fam == "tan" and a % 3:
        a = sympy.Integer(3)
    lo_a = rng.choice(alphas[:-1]) if rng.random() < 0.4 else R(0)
    hi_a = rng.choice([t for t in alphas if t > lo_a])
    lo_x, hi_x = _xval(fam, a, lo_a), _xval(fam, a, hi_a)
    f, f_text = _integrand(kind, a)
    _check_converted(kind, a) if kind != "sininv" else None
    which = rng.choice(["upper", "lower"]) if lo_a != 0 else "upper"
    ask_x, ask_a = (hi_x, hi_a) if which == "upper" else (lo_x, lo_a)
    H = _hint(fam, a)
    I_text, I_math = _integral(rng, f_text, f, lo_x, hi_x)
    text, latex = say(rng, _LIMIT_STEMS, {"I": I_text, "H": H[0]}, {"I": I_math, "H": H[1]}, {"which": which})
    key = sympy.sympify(ask_a)
    comp = pi / 2 - key
    cands = [
        Cand(comp, "nx-tsub-complement-angle"),
        Cand(sympy.sympify(ask_x), "nx-tsub-limit-not-converted"),
        Cand(2 * key, "nx-tsub-limit-double-angle"),
        Cand(key / 2, "nx-tsub-limit-half-angle"),
    ]
    cands = [Cand(c.expr, c.tag, const=False) for c in cands if c.expr != 0]
    sub = f"x = {a}{'tan' if fam == 'tan' else 'sin'}(θ)"
    trig = "tan" if fam == "tan" else "sin"
    working = [
        f"Rearrange {sub}: {trig}(θ) = x/{a}.",
        f"When x = {_lim(ask_x)}, {trig}(θ) = {_lim(sympy.nsimplify(ask_x / a))}, so θ = {_lim(key)}.",
    ]
    return dict(
        text=text,
        latex=latex,
        working=[_tidy(w) for w in working],
        distractors=_angle_options(key, cands),
        **exact_fields(key),
    )


def _easy(rng):
    return rng.choice([_easy_k, _easy_k, _easy_piece, _easy_limit, _easy_limit])(rng)


# -- medium and hard ----------------------------------------------------------------------------------------------


def _pair(rng, angles, lo_zero):
    for _ in range(50):
        lo = R(0) if lo_zero or rng.random() < 0.4 else rng.choice(angles)
        hi = rng.choice(angles)
        if hi != lo and hi > lo:
            return lo, hi
    return R(0), angles[1]


def _medium(rng):
    kind = rng.choice(["tan1", "tan1", "sininv", "quarter"])
    if kind == "quarter":
        a = sympy.Integer(rng.choice([2, 3, 4, 5, 6, 8]))
        return _make_definite(rng, "sinroot", a, R(0), pi / 2, "hint")
    if kind == "tan1":
        a = sympy.Integer(rng.choice([3, 6, 3, 2, 4, 5]))
        angles = [R(0), pi / 4, pi / 3] if a % 3 else [R(0), pi / 6, pi / 4, pi / 3]
        lo, hi = _pair(rng, angles, False)
        return _make_definite(rng, "tan1", a, lo, hi, "hint")
    a = sympy.Integer(rng.choice([2, 3, 4, 5, 6]))
    lo, hi = _pair(rng, [R(0), pi / 6, pi / 4, pi / 3], False)
    return _make_definite(rng, "sininv", a, lo, hi, "hint")


def _hard(rng):
    kind = rng.choice(["tan2", "tan2", "sinroot", "sinx2", "sin32"])
    if _FAMILY[kind] == "tan":
        a = sympy.Integer(rng.choice([1, 2, 3, 3, 6]))
        angles = [R(0), pi / 4, pi / 3] if a % 3 else [R(0), pi / 6, pi / 4, pi / 3]
        lo, hi = _pair(rng, angles, rng.random() < 0.5)
    else:
        a = sympy.Integer(rng.choice([1, 2, 3, 4, 5, 6]))
        angles = [R(0), pi / 6, pi / 4, pi / 3] + ([pi / 2] if kind in ("sinroot", "sinx2") else [])
        lo, hi = _pair(rng, angles, rng.random() < 0.4)
    return _make_definite(rng, kind, a, lo, hi, "free")


_TIERS = {"easy": _easy, "medium": _medium, "hard": _hard}


@generator("nx_trig_sub_hsc", level=15, presets={t: {"tier": t} for t in _TIERS})
@uses_rng
@retry
def nx_trig_sub_hsc(rng, tier: str = "easy"):
    """HSC trigonometric substitution; the tiers are described in the module docstring."""
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    return _TIERS[tier](rng)


# -- exam ---------------------------------------------------------------------------------------------------------


@generator("exd_nx_trig_sub_hsc", level=15, presets={"exam": {}})
@uses_rng
@retry
def exd_nx_trig_sub_hsc(rng, tier="exam"):
    """Exam chain: k, the new upper limit, the integral of cos^2 up to it (from the limit), the value (from k and that)."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    fam_kind = rng.choice(["tan2", "sinroot"])
    fam = _FAMILY[fam_kind]
    if fam == "tan":
        a = sympy.Integer(rng.randint(1, 20))
        beta = rng.choice([pi / 4, pi / 3, pi / 6])
    else:
        a = sympy.Integer(rng.randint(1, 20))
        beta = rng.choice([pi / 6, pi / 4, pi / 3])
    f, f_text = _integrand(fam_kind, a)
    _check_converted(fam_kind, a)
    hi_x = _xval(fam, a, beta)
    k = _CONVERTED[fam_kind][0](a)
    inner = beta / 2 + sin(2 * beta) / 4
    value = sympy.simplify(k * inner)
    check(
        sympy.simplify(value - (_G(fam_kind, a, beta) - _G(fam_kind, a, 0))) == 0,
        "value differs from the antiderivative",
    )
    _quad(value, f, 0, hi_x)
    H = _hint(fam, a)
    I_text, I_math = _integral(rng, f_text, f, 0, hi_x)
    stem = f"The integral {I_text} is to be found using the substitution {H[0]}."
    stem_latex = latex_from_inline(f"The integral {I_math} is to be found using the substitution {H[1]}.")
    sub = f"x = {a}{'tan' if fam == 'tan' else 'sin'}(θ)"
    trig = "tan" if fam == "tan" else "sin"
    k_cands = (
        [
            (1 / a**4, "nx-tsub-dx-factor-a-dropped"),
            (1 / a**2, "nx-tsub-power-of-a-wrong"),
            (1 / a, "nx-tsub-power-not-squared"),
            (a**3, "nx-tsub-a-multiplied-not-divided"),
        ]
        if fam == "tan"
        else [
            (a, "nx-tsub-amplitude-not-squared"),
            (a**3, "nx-tsub-amplitude-cubed"),
        ]
    )
    b_cands = [
        Cand(pi / 2 - beta, "nx-tsub-complement-angle", const=False),
        Cand(hi_x, "nx-tsub-limit-not-converted", const=False),
        Cand(2 * beta, "nx-tsub-limit-double-angle", const=False),
        Cand(beta / 2, "nx-tsub-limit-half-angle", const=False),
    ]
    c_cands = [
        Cand(beta / 2 - sin(2 * beta) / 4, "nx-tsub-identity-sign"),
        Cand(beta + sin(2 * beta) / 2, "nx-tsub-half-dropped"),
        Cand(beta / 2, "nx-tsub-identity-term-dropped"),
        Cand(cos(beta) ** 3 / 3 - R(1, 3), "nx-tsub-power-rule-on-trig"),
    ]
    d_cands = [
        Cand(inner, "nx-tsub-constant-not-applied"),
        Cand(inner / k, "nx-tsub-constant-inverted"),
        Cand(k * (beta / 2 - sin(2 * beta) / 4), "nx-tsub-identity-sign"),
        Cand(k * beta / 2, "nx-tsub-identity-term-dropped"),
        Cand(k * (beta + sin(2 * beta) / 2), "nx-tsub-half-dropped"),
    ]
    c_cands = [Cand(c.expr, c.tag, const=False) for c in c_cands]
    pa = part(
        "Write the integral in terms of θ as the integral of k cos^2(θ) dθ. Find the constant k.",
        "Write the integral in terms of \\(\\theta\\) as \\(\\int k\\cos^2\\theta\\,\\mathrm{d}\\theta\\). Find the constant \\(k\\).",
        2,
        [
            _tidy(f"With {sub}, dx = {a}{'sec^2(θ)' if fam == 'tan' else 'cos(θ)'} dθ."),
            f"The integrand becomes k cos^2(θ) with k = {k}.",
        ],
        exact_fields(k),
        _pick_values(k, k_cands),
    )
    pb = part(
        "Find the upper limit of θ that replaces x = " + _lim(hi_x) + ". (The lower limit is 0.)",
        "Find the upper limit of \\(\\theta\\) that replaces \\(x = "
        + sympy.latex(hi_x)
        + "\\). (The lower limit is \\(0\\).)",
        1,
        [_tidy(f"{trig}(θ) = x/{a} = {_lim(sympy.nsimplify(hi_x / a))}, so θ = {_lim(beta)}.")],
        exact_fields(beta),
        _angle_options(beta, b_cands),
    )
    pc = part(
        "Using your upper limit from the previous part, find the exact value of the integral of cos^2(θ) dθ from 0 to that limit.",
        "Using your upper limit from the previous part, find the exact value of \\(\\int \\cos^2\\theta\\,\\mathrm{d}\\theta\\) from \\(0\\) to that limit.",
        2,
        [
            "cos^2(θ) = (1 + cos 2θ)/2, so the integral is θ/2 + sin(2θ)/4 evaluated from 0 to the limit.",
            _tidy(f"= {_lim(inner)}."),
        ],
        exact_fields(inner),
        _options(inner, c_cands, rng, lo=-1, hi=pi / 2),
        follow_through="part_b/2 + sin(2*part_b)/4",
    )
    pd = part(
        "Hence find the exact value of the original integral.",
        "Hence find the exact value of the original integral.",
        1,
        [_tidy(f"The value is k × (the integral of cos^2): {k} × ({_lim(inner)}) = {_lim(value)}.")],
        exact_fields(value),
        _options(value, [Cand(c.expr, c.tag, False) for c in d_cands], rng, lo=-1, hi=100),
        follow_through="part_a*part_c",
    )
    check(isinstance(pa, dict), "")
    return dict(text=stem, latex=stem_latex, parts=[pa, pb, pc, pd])
