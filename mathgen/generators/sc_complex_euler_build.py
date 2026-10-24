"""Complex numbers in Cartesian, polar and exponential form, answered with exact typed values:
`sc_complex_euler` (easy, medium, hard) and the exam chain `sc_complex_euler_exam`.

Skill: ``complex-exponential-form-conversions``. The existing ``ucv1-complex-exponential-form`` asks only for an argument,
as a multiple of pi. This skill converts between a + bi, r(cos θ + i sin θ) and r e^(iθ) and uses the exponential form to
multiply, divide and raise to powers; every answer is a real exact value (a real part, an imaginary part, a whole
number), typed as an expression, so the question cannot be answered by retyping the number it gives.

* easy: Re or Im of r(cos θ + i sin θ) or of r e^(iθ) for a standard angle in any quadrant; or the real value of a
  combination of e^(iπ), e^(2iπ) and constants (Euler's identity).
* medium: Re or Im of a product z1 z2 or a quotient z1/z2 of two numbers in exponential form (the arguments add or
  subtract; the moduli multiply or divide).
* hard: Re or Im of z^n where z is given as a + bi (find r and θ, raise to the power, convert back); or the real value of
  c(e^(iθ) + e^(-iθ)) or c(e^(iθ) - e^(-iθ))/i (Euler's formulas for cosine and sine).
* exam (``sc_complex_euler_exam``): z = r e^(iθ). (a) Re z, (b) Im z, (c) Re z^n and (d) Im z^n from (a) and (b) by
  expanding (a + bi)^n (n = 2 or 3), (e) |z^n| from (c) and (d).

Every key is checked against SymPy's exact complex arithmetic: the Cartesian form of the exponential form, the product
or quotient expanded in Cartesian form, and (a + bi)^n expanded by the binomial theorem.
"""

from __future__ import annotations

import re
import math
from fractions import Fraction

import sympy
from sympy import I, Rational, cos, exp, pi, sin, sqrt

from ..core import attempts, check, generator
from ._spread_kit import NEAR, mixed, uses_rng
from ._fy1cd_kit import F, pi_text, say, surd
from ._r3b_kit import (
    Cand,
    Redraw,
    exact_fields,
    expr_text,
    part,
    pick_expr,
    retry,
)

_ANGLES = [
    Fraction(s * a, b) for a, b in ((1, 6), (1, 4), (1, 3), (1, 2), (2, 3), (3, 4), (5, 6), (1, 1)) for s in (1, -1)
]
_ANGLES = sorted(set(_ANGLES) - {Fraction(-1)})  # pi and -pi are the same direction; keep pi
_SIXTHS = [p for p in _ANGLES if (p * 6).denominator == 1]
_QUARTERS = [p for p in _ANGLES if (p * 4).denominator == 1]
_MODULI = (1, 2, 3, 4, 5, 6, 8)


def _tidy(line: str) -> str:
    """A worked step with SymPy's spelling made learner-readable: sqrt(3) as √3, 3*pi as 3pi, any other * as ×."""
    line = re.sub(r"sqrt\((\d+)\)", r"√\1", line)
    line = re.sub(r"(?<=[\d)])\*(?=[A-Za-z√(])", "", line)
    return line.replace("*", " × ")


def _sym(p: Fraction):
    return Rational(p.numerator, p.denominator) * pi


def _unit(p: Fraction):
    """cos θ and sin θ for θ = pπ as exact SymPy numbers."""
    return sympy.cos(_sym(p)), sympy.sin(_sym(p))


def _is_close(a, b) -> bool:
    return abs(sympy.N(a - b, 30)) < sympy.Float("1e-20")


# -- spelling ------------------------------------------------------------------------------------------------------


def _surd_text(value) -> tuple[str, str]:
    """An exact real number of the form q or q√2 or q√3, plain and TeX (the sign included)."""
    value = sympy.nsimplify(value)
    for root in (1, 2, 3):
        coef = sympy.nsimplify(value / sympy.sqrt(root))
        if coef.is_Rational:
            return surd(Fraction(int(coef.p), int(coef.q)), root)
    raise Redraw(f"cannot write {value} as q, q√2 or q√3")


def _cartesian(a, b) -> F:
    """a + bi as a formula."""
    pa, ta = _surd_text(a) if a != 0 else ("", "")
    if b == 0:
        return F(pa or "0", ta or "0")
    if b in (1, -1):
        pb, tb = "i", "i"
    else:
        body, body_t = _surd_text(abs(b))
        pb, tb = f"{body} i", f"{body_t}\\,i"
    neg = b < 0
    if not pa:
        return F(("-" if neg else "") + pb, ("-" if neg else "") + tb)
    return F(f"{pa} {'-' if neg else '+'} {pb}", f"{ta} {'-' if neg else '+'} {tb}")


def _angle(p: Fraction) -> tuple[str, str]:
    return pi_text(p)


def _exp(r: int, p: Fraction) -> F:
    """r e^(iθ) with θ = pπ."""
    head = "" if r == 1 else str(r)
    sign = "-" if p < 0 else ""
    plain, tex = _angle(abs(p))
    return F(f"{head}e^({sign}i{plain})", f"{head}e^{{{sign}i\\,{tex}}}")


def _polar(r: int, p: Fraction) -> F:
    plain, tex = _angle(p)
    if r == 1:
        return F(f"cos({plain}) + i sin({plain})", f"\\cos {tex} + i\\sin {tex}")
    return F(f"{r}(cos({plain}) + i sin({plain}))", f"{r}\\left(\\cos {tex} + i\\sin {tex}\\right)")


def _exp_or_polar(rng, r, p):
    return (_exp if rng.random() < 0.5 else _polar)(r, p)


def _stem(*items):
    text, latex = say(*items)
    return dict(text=text, latex=latex)


def _writable(value) -> bool:
    """Is `value` of the form q, q sqrt(2) or q sqrt(3) with q rational (the shapes `_surd_text` can spell)?"""
    value = sympy.sympify(value)
    return any((value / sympy.sqrt(root)).is_Rational for root in (1, 2, 3))


def _around(size, cc, p, *, limit=None, key=None):
    """`size` times cos (or sin) of the angles a slip away from pπ, and (given the key) the key out by a half or a whole:
    values of the same shape as a real or imaginary part, none larger than the number's own modulus (a part cannot
    exceed it)."""
    out = []
    for d in (
        Fraction(1, 6),
        Fraction(-1, 6),
        Fraction(1, 4),
        Fraction(-1, 4),
        Fraction(1, 3),
        Fraction(-1, 3),
        Fraction(1, 2),
        Fraction(-1, 2),
        Fraction(1),
    ):
        v = sympy.sympify(size * cc(p + d))
        if _writable(v) and (limit is None or abs(v) <= limit):
            out.append(v)
    if key is not None:
        key = sympy.sympify(key)
        out += [key + Rational(k, 2) for k in (-2, -1, 1, 2) if limit is None or abs(key + Rational(k, 2)) <= limit]
    return out


def _options(key, cands, near, *, want=3, nonzero=False):
    """Wrong exact values with a random number of them below the key: the named slips first, then plain near misses
    (`near`) on a side that has too few. A part that later parts follow through from offers no 0 (`nonzero`)."""
    key = sympy.sympify(key)

    def shape(
        v,
    ):  # whole, fraction or surd: a filler shares the key's shape, so the key is not the only one of its kind
        return "int" if v.is_Integer else "rational" if v.is_Rational else "surd"

    fillers = [
        Cand(sympy.sympify(v), NEAR, const=False)
        for v in near
        if v is not None and not (nonzero and v == 0) and shape(sympy.sympify(v)) == shape(key)
    ]
    pool = [c for c in cands if c is not None and c.expr is not None]
    picked = []
    for _ in range(8):  # redraw the split until some option shares the key's shape (a whole key beside only fractions)
        picked = pick_expr(
            key,
            False,
            mixed(
                key,
                pool,
                fillers=fillers,
                lo=None,
                hi=None,
                unbounded="open",
            ),
            minimum=want,
        )
        if any(shape(sympy.sympify(d.value if not isinstance(d.value, str) else d.text)) == shape(key) for d in picked):
            break
    return picked


def _scalar(expr, cands, rng, *, want=3, near=()):
    """Answer fields and options for an exact real value."""
    fields = exact_fields(expr)
    picked = _options(expr, [Cand(c, t, const=False) for c, t in cands if c is not None], near, want=want)
    return fields, picked


def _nonzero(cands):
    """Options for a part that later parts follow through from: a reply of 0 earns no follow-through, so none is offered."""
    return [c for c in cands if c is not None and sympy.nsimplify(c.expr) != 0]


def _re_or_im(name):
    """The function p -> cos(pπ) for "Re", sin(pπ) for "Im"."""
    index = 0 if name == "Re" else 1
    return lambda p: _unit(p)[index]


# ------------------------------------------------------------------------------------------------ easy


def _easy_convert(rng):
    for _ in attempts(500, "a nonzero part"):
        p = rng.choice(_ANGLES)
        r = rng.choice(_MODULI)
        c, s = _unit(p)
        part_name = rng.choice(("Re", "Im"))
        value = r * (c if part_name == "Re" else s)
        # a part near the full modulus (r cos 0, r cos pi/6) leaves almost no room on that side for wrong values, which
        # cannot exceed r: its key would nearly always be the largest or the least option
        if value != 0 and abs(value) <= Rational(3, 4) * r:
            break
    z = _exp_or_polar(rng, r, p)
    z2 = _exp(r, p)
    check(
        _is_close(sympy.expand_complex(r * sympy.exp(I * _sym(p))) - (r * c + I * r * s), 0),
        "e^(iθ) differs from cos + i sin",
    )
    other = r * (s if part_name == "Re" else c)
    ref = r * (abs(c) if part_name == "Re" else abs(s))
    cands = [
        (other, "sc-cx-cos-sin-swapped"),
        (c if part_name == "Re" else s, "sc-cx-modulus-dropped"),
        (ref if ref != value else None, "sc-cx-quadrant-sign-lost"),
        (r + (c if part_name == "Re" else s), "sc-cx-modulus-added"),
        (-value, "sc-cx-sign-reversed"),
    ]
    name = "real part" if part_name == "Re" else "imaginary part"
    st = _stem("Let z = ", z, f". Find the {name} of z, {part_name}(z), as an exact value.")
    working = [
        f"r e^(iθ) = r(cos θ + i sin θ), with r = {r} and θ = {_angle(p)[0]}.",
        f"{part_name}(z) = r {'cos' if part_name == 'Re' else 'sin'} θ = {r} × {sympy.simplify(c if part_name == 'Re' else s)} = {sympy.simplify(value)}.",
    ]
    fields, options = _scalar(
        sympy.nsimplify(value), cands, rng, near=_around(r, _re_or_im(part_name), p, limit=r, key=value)
    )
    return dict(**st, working=working, distractors=options, **fields)


_EULER_TERMS = {
    ("e^(iπ)", r"e^{i\pi}"): -1,
    ("e^(-iπ)", r"e^{-i\pi}"): -1,
    ("e^(2iπ)", r"e^{2i\pi}"): 1,
    ("e^(-2iπ)", r"e^{-2i\pi}"): 1,
    ("e^(3iπ)", r"e^{3i\pi}"): -1,
}
_ODD = [k for k, v in _EULER_TERMS.items() if v == -1]
_EVEN = [k for k, v in _EULER_TERMS.items() if v == 1]


def _easy_euler(rng):
    for _ in attempts(500, "a nonzero value"):
        a, b = rng.randint(2, 9), rng.randint(2, 9)
        c0 = rng.randint(-9, 9)
        odd, even = rng.choice(_ODD), rng.choice(_EVEN)
        value = a * _EULER_TERMS[odd] + b * _EULER_TERMS[even] + c0
        if value != 0 and abs(value) <= 25 and value not in (a, b):
            break
    terms = [(a, odd), (b, even)]
    if rng.random() < 0.5:
        terms.reverse()
    plain = " + ".join(f"{k}{t[0]}" for k, t in terms)
    tex = " + ".join(f"{k}{t[1]}" for k, t in terms)
    if c0:
        plain += f" {'+' if c0 > 0 else '-'} {abs(c0)}"
        tex += f" {'+' if c0 > 0 else '-'} {abs(c0)}"
    # second route: SymPy evaluates each exponential from its printed exponent
    total = 0
    for k, t in terms:
        mult = {"e^(iπ)": 1, "e^(-iπ)": -1, "e^(2iπ)": 2, "e^(-2iπ)": -2, "e^(3iπ)": 3}[t[0]]
        total += k * sympy.simplify(sympy.exp(I * pi * mult))
    check(sympy.simplify(total + c0 - value) == 0, "Euler's identity evaluation differs")
    s_odd, s_even = _EULER_TERMS[odd], _EULER_TERMS[even]
    cands = [
        (-a * s_odd + b * s_even + c0, "sc-cx-euler-pi-as-one"),
        (a * s_odd - b * s_even + c0, "sc-cx-euler-two-pi-as-minus-one"),
        (-a * s_odd - b * s_even + c0, "sc-cx-euler-values-swapped"),
        (a * s_odd + b * s_even if c0 else None, "sc-cx-euler-constant-dropped"),
    ]
    st = _stem("Evaluate ", F(plain, tex), ", giving a real number.")
    working = [
        "Euler's identity: e^(iπ) = cos π + i sin π = -1 and e^(2iπ) = cos 2π + i sin 2π = 1.",
        "e^(3iπ) = -1 and e^(-iπ) = -1 (a whole number of half turns on the real axis), and e^(-2iπ) = 1.",
        f"The value is {sympy.simplify(total + c0)}.",
    ]
    fields, options = _scalar(
        sympy.Integer(value), cands, rng, near=[sympy.Integer(value + k) for k in (-4, -3, -2, -1, 1, 2, 3, 4)]
    )
    return dict(**st, working=working, distractors=options, **fields)


def _easy(rng):
    return (_easy_convert if rng.random() < 0.6 else _easy_euler)(rng)


# ------------------------------------------------------------------------------------------------ medium


def _medium(rng):
    op = rng.choice(("product", "quotient"))
    family = rng.choice((_SIXTHS, _QUARTERS))
    for _ in attempts(1000, "a standard angle"):
        p1, p2 = rng.choice(family), rng.choice(family)
        if op == "product":
            r1, r2 = rng.choice((1, 2, 3, 4)), rng.choice((1, 2, 3, 4))
            r, p = r1 * r2, p1 + p2
        else:
            r2 = rng.choice((1, 2, 3))
            r1 = r2 * rng.choice((2, 3, 4, 5))
            r, p = Rational(r1, r2), p1 - p2
        q = (p + 1) % 2 - 1  # bring into (-1, 1]
        p_red = Fraction(q) if q != -1 else Fraction(1)
        if (p_red * 12).denominator != 1 or p_red == 0 or p1 == p2 and op == "quotient":
            continue
        c, s = _unit(p_red)
        part_name = rng.choice(("Re", "Im"))
        value = sympy.nsimplify(r * (c if part_name == "Re" else s))
        if value != 0 and abs(value) < 100:
            break
    z1, z2 = _exp(r1, p1), _exp(r2, p2)
    num = r1 * sympy.exp(I * _sym(p1))
    den = r2 * sympy.exp(I * _sym(p2))
    second = sympy.expand_complex(num * den if op == "product" else num / den)
    check(
        _is_close(sympy.re(second) if part_name == "Re" else sympy.im(second), value),
        "Cartesian product or quotient differs",
    )
    cc = _re_or_im(part_name)
    other = _re_or_im("Im" if part_name == "Re" else "Re")
    if op == "product":
        p_wrong = p1 - p2
        cands = [
            (r * cc(p_wrong), "cx-product-subtracts-args"),
            ((r1 + r2) * cc(p_red), "sc-cx-moduli-added"),
            (r * other(p_red), "sc-cx-cos-sin-swapped"),
            (r * cc(p1), "sc-cx-second-argument-ignored"),
        ]
        verb = (
            "Find the real part of the product z1 z2"
            if part_name == "Re"
            else "Find the imaginary part of the product z1 z2"
        )
        working = [
            f"Multiply the moduli, {r1} × {r2} = {r}, and add the arguments: {_angle(p1)[0]} + ({_angle(p2)[0]}) = {_angle(p1 + p2)[0]}.",
            f"z1 z2 = {r} e^(i{_angle(p_red)[0]}), so {part_name}(z1 z2) = {'' if r == 1 else f'{r} '}{'cos' if part_name == 'Re' else 'sin'}({_angle(p_red)[0]}) = {value}.",
        ]
    else:
        p_wrong = p1 + p2
        cands = [
            (r * cc(p_wrong), "cx-quotient-adds-args"),
            (r1 * r2 * cc(p_red), "sc-cx-moduli-not-divided"),
            (r * other(p_red), "sc-cx-cos-sin-swapped"),
            (r * cc(p1), "sc-cx-second-argument-ignored"),
            (r * cc(p2 - p1), "sc-cx-quotient-arguments-reversed"),
        ]
        verb = (
            "Find the real part of the quotient z1/z2"
            if part_name == "Re"
            else "Find the imaginary part of the quotient z1/z2"
        )
        working = [
            f"Divide the moduli, {r1}/{r2} = {r}, and subtract the arguments: {_angle(p1)[0]} - ({_angle(p2)[0]}) = {_angle(p1 - p2)[0]}.",
            f"z1/z2 = {r} e^(i{_angle(p_red)[0]}), so {part_name}(z1/z2) = {'' if r == 1 else f'{r} '}{'cos' if part_name == 'Re' else 'sin'}({_angle(p_red)[0]}) = {value}.",
        ]
    st = _stem("Let z1 = ", z1, " and z2 = ", z2, f". {verb}, as an exact value.")
    fields, options = _scalar(value, cands, rng, near=_around(r, cc, p_red, limit=abs(r), key=value))
    return dict(**st, working=working, distractors=options, **fields)


# ------------------------------------------------------------------------------------------------ hard

# (a, b, r, argument/pi): Cartesian forms whose modulus and argument are exact
_CARTESIAN = [
    (1, 1, sqrt(2), Fraction(1, 4)),
    (-1, 1, sqrt(2), Fraction(3, 4)),
    (1, -1, sqrt(2), Fraction(-1, 4)),
    (-1, -1, sqrt(2), Fraction(-3, 4)),
    (sqrt(3), 1, 2, Fraction(1, 6)),
    (-sqrt(3), 1, 2, Fraction(5, 6)),
    (sqrt(3), -1, 2, Fraction(-1, 6)),
    (-sqrt(3), -1, 2, Fraction(-5, 6)),
    (1, sqrt(3), 2, Fraction(1, 3)),
    (-1, sqrt(3), 2, Fraction(2, 3)),
    (1, -sqrt(3), 2, Fraction(-1, 3)),
    (-1, -sqrt(3), 2, Fraction(-2, 3)),
]


def _hard_power(rng):
    for _ in attempts(1000, "a clear power"):
        a, b, r, p = rng.choice(_CARTESIAN)
        k = rng.choice((1, 1, 2))
        n = rng.randint(3, 8)
        modulus = k * r
        if sympy.N(modulus**n) > 2000:
            continue
        z = (k * a) + I * (k * b)
        direct = sympy.expand(z**n)
        c, s = _unit(n * p)
        via = sympy.nsimplify(modulus**n * (c + I * s))
        check(_is_close(direct, via), "(a + bi)^n differs from r^n e^(inθ)")
        part_name = rng.choice(("Re", "Im"))
        value = sympy.nsimplify(sympy.re(direct) if part_name == "Re" else sympy.im(direct))
        if value != 0:
            break
    cc, ss = (c, s) if part_name == "Re" else (s, c)
    wrong_angle = _unit(p)
    cands = [
        (modulus * n * (c if part_name == "Re" else s), "cx-power-as-multiplication"),
        (modulus**n * (wrong_angle[0] if part_name == "Re" else wrong_angle[1]), "cx-arg-power-ignored"),
        (modulus * (c if part_name == "Re" else s), "sc-cx-modulus-not-raised"),
        (abs(value) if value < 0 else None, "sc-cx-quadrant-sign-lost"),
        (modulus**n * ss, "sc-cx-cos-sin-swapped"),
    ]
    za = _cartesian(k * a, k * b)
    nsup = {3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸"}[n]
    verb = "real part" if part_name == "Re" else "imaginary part"
    st = _stem("Let z = ", za, f". Find the {verb} of z{nsup}, as an exact value.")
    mod_text = _surd_text(modulus)[0]
    working = [
        f"|z| = {mod_text} and arg z = {_angle(p)[0]}, so z = {mod_text} e^(i{_angle(p)[0]}).",
        f"z{nsup} = ({mod_text}){nsup} e^(i{_angle(n * p)[0]}) = {_surd_text(modulus**n)[0]} (cos({_angle(n * p)[0]}) + i sin({_angle(n * p)[0]})).",
        f"{part_name}(z{nsup}) = {_surd_text(modulus**n)[0]} × {'cos' if part_name == 'Re' else 'sin'}({_angle(n * p)[0]}) = {value}.",
    ]
    fields, options = _scalar(
        value, cands, rng, near=_around(modulus**n, _re_or_im(part_name), n * p, limit=modulus**n, key=value)
    )
    return dict(**st, working=working, distractors=options, **fields)


def _hard_euler_trig(rng):
    """c(e^(iθ) + e^(-iθ)) = 2c cos θ and c(e^(iθ) - e^(-iθ))/i = 2c sin θ, both real."""
    kind = rng.choice(("cosine", "sine"))
    for _ in attempts(500, "a nonzero value"):
        p = rng.choice([q for q in _ANGLES if q > 0])
        c = rng.choice((1, 2, 3, 4, 5, 6))
        cp, sp = _unit(p)
        value = sympy.nsimplify(2 * c * (cp if kind == "cosine" else sp))
        # a value at the very end of its range (2c cos pi = -2c) has nothing beyond it to offer on one side: keep the
        # key at least a quarter of the range from either end, so options below and above it are both possible
        if value != 0 and abs(float(value)) <= 1.5 * c:
            break
    ang = _angle(p)
    head = "" if c == 1 else f"{c}"
    if kind == "cosine":
        formula = F(
            f"{head}(e^(i{ang[0]}) + e^(-i{ang[0]}))", f"{head}\\left(e^{{i\\,{ang[1]}}} + e^{{-i\\,{ang[1]}}}\\right)"
        )
        total = c * (sympy.exp(I * _sym(p)) + sympy.exp(-I * _sym(p)))
        main, swap = cp, sp
    else:
        formula = F(
            f"{head}(e^(i{ang[0]}) - e^(-i{ang[0]}))/i" if c != 1 else f"(e^(i{ang[0]}) - e^(-i{ang[0]}))/i",
            f"{head}\\,\\frac{{e^{{i\\,{ang[1]}}} - e^{{-i\\,{ang[1]}}}}}{{i}}",
        )
        total = c * (sympy.exp(I * _sym(p)) - sympy.exp(-I * _sym(p))) / I
        main, swap = sp, cp
    check(_is_close(sympy.expand_complex(total), value), "the Euler combination differs from 2c cos or 2c sin")
    cands = [
        (c * main, "sc-cx-euler-factor-two-dropped"),
        (2 * c * swap, "sc-cx-cos-sin-swapped"),
        (0, "sc-cx-conjugate-terms-cancel"),
        (2 * c, "sc-cx-angle-ignored"),
        (-value, "sc-cx-sign-reversed"),
    ]
    st = _stem("Evaluate ", formula, ", giving an exact real value.")
    working = [
        f"e^(iθ) = cos θ + i sin θ and e^(-iθ) = cos θ - i sin θ, with θ = {ang[0]}.",
        "Their sum is 2 cos θ (the sine terms cancel)."
        if kind == "cosine"
        else "Their difference is 2i sin θ (the cosine terms cancel), and dividing by i leaves 2 sin θ.",
        f"The value is {2 * c} {'cos' if kind == 'cosine' else 'sin'}({ang[0]}) = {value}.",
    ]
    fields, options = _scalar(
        value, cands, rng, near=_around(2 * c, _re_or_im("Re" if kind == "cosine" else "Im"), p, limit=2 * c, key=value)
    )
    return dict(**st, working=working, distractors=options, **fields)


def _hard(rng):
    return rng.choice((_hard_power, _hard_power, _hard_euler_trig))(rng)


_TIERS = {"easy": _easy, "medium": _medium, "hard": _hard}
_PRESETS = {t: {"tier": t} for t in _TIERS}


@generator("sc_complex_euler", level=18, presets=_PRESETS)
@uses_rng
@retry
def sc_complex_euler(rng, tier: str = "easy"):
    """Cartesian, polar and exponential form of a complex number: converting, multiplying, dividing and powers."""
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    question = _TIERS[tier](rng)
    question["working"] = [_tidy(w) for w in question["working"]]
    return question


# ------------------------------------------------------------------------------------------------ exam


@generator("sc_complex_euler_exam", level=18, presets={"exam": {}})
@uses_rng
@retry
def sc_complex_euler_exam(rng, tier="exam"):
    """Exam chain: Re z and Im z from r e^(iθ), then Re, Im and the modulus of z^n from them by following through."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(1000, "a clear chain"):
        p = rng.choice([q for q in _ANGLES if q.denominator > 1 and (q.denominator != 2)])
        r = rng.choice((2, 3, 4, 5, 6, 7, 8, 9, 10, 12))
        n = rng.choice((2, 3))
        c, s = _unit(p)
        a, b = sympy.nsimplify(r * c), sympy.nsimplify(r * s)
        power = sympy.expand((a + I * b) ** n) if a != 0 and b != 0 else 0
        if a != 0 and b != 0 and sympy.re(power) != 0 and sympy.im(power) != 0:
            break
    power = sympy.expand((a + I * b) ** n)
    re_n, im_n = sympy.nsimplify(sympy.re(power)), sympy.nsimplify(sympy.im(power))
    cn, sn = _unit(n * p)
    check(_is_close(re_n, r**n * cn) and _is_close(im_n, r**n * sn), "z^n differs from r^n e^(inθ)")
    check(_is_close(sympy.sqrt(re_n**2 + im_n**2), r**n), "|z^n| differs from r^n")
    nsup = {2: "²", 3: "³"}[n]
    st = _stem("Let z = ", _exp(r, p), ".")
    pa = part(
        "Find the real part of z, Re(z), as an exact value.",
        "Find the real part of z, Re(z), as an exact value.",
        1,
        [f"z = {r}(cos({_angle(p)[0]}) + i sin({_angle(p)[0]})), so Re(z) = {r} cos({_angle(p)[0]}) = {a}."],
        exact_fields(a),
        _options(
            a,
            _nonzero(
                [
                    Cand(r * s, "sc-cx-cos-sin-swapped", const=False),
                    Cand(c, "sc-cx-modulus-dropped", const=False),
                    Cand(abs(a), "sc-cx-quadrant-sign-lost", const=False) if a < 0 else None,
                    Cand(-a, "sc-cx-sign-reversed", const=False),
                ]
            ),
            _around(r, _re_or_im("Re"), p, limit=r, key=a),
            nonzero=True,
        ),
    )
    pb = part(
        "Find the imaginary part of z, Im(z), as an exact value.",
        "Find the imaginary part of z, Im(z), as an exact value.",
        1,
        [f"Im(z) = {r} sin({_angle(p)[0]}) = {b}."],
        exact_fields(b),
        _options(
            b,
            _nonzero(
                [
                    Cand(r * c, "sc-cx-cos-sin-swapped", const=False),
                    Cand(s, "sc-cx-modulus-dropped", const=False),
                    Cand(abs(b), "sc-cx-quadrant-sign-lost", const=False) if b < 0 else None,
                    Cand(-b, "sc-cx-sign-reversed", const=False),
                ]
            ),
            _around(r, _re_or_im("Im"), p, limit=r, key=b),
            nonzero=True,
        ),
    )
    if n == 2:
        re_form, im_form = "a^2 - b^2", "2ab"
        ft_re, ft_im = "part_a**2 - part_b**2", "2*part_a*part_b"
        re_wrong = [
            Cand(a**2 + b**2, "sc-cx-i-squared-as-one", const=False),
            Cand(a**2, "sc-cx-imaginary-part-dropped", const=False),
            Cand(-(a**2 - b**2), "sc-cx-sign-reversed", const=False),
            Cand(2 * a * b, "sc-cx-parts-swapped-in-expansion", const=False),
        ]
        im_wrong = [
            Cand(a * b, "sc-cx-cross-term-not-doubled", const=False),
            Cand(a**2 + b**2, "sc-cx-i-squared-as-one", const=False),
            Cand(2 * a * b * -1, "sc-cx-sign-reversed", const=False),
            Cand(a**2 - b**2, "sc-cx-parts-swapped-in-expansion", const=False),
        ]
    else:
        re_form, im_form = "a^3 - 3ab^2", "3a^2 b - b^3"
        ft_re, ft_im = "part_a**3 - 3*part_a*part_b**2", "3*part_a**2*part_b - part_b**3"
        re_wrong = [
            Cand(a**3 + 3 * a * b**2, "sc-cx-i-squared-as-one", const=False),
            Cand(a**3 - a * b**2, "sc-cx-binomial-coefficient-dropped", const=False),
            Cand(a**3, "sc-cx-imaginary-part-dropped", const=False),
            Cand(3 * a**2 * b - b**3, "sc-cx-parts-swapped-in-expansion", const=False),
        ]
        im_wrong = [
            Cand(3 * a**2 * b + b**3, "sc-cx-i-cubed-as-i", const=False),
            Cand(a**2 * b - b**3, "sc-cx-binomial-coefficient-dropped", const=False),
            Cand(3 * a**2 * b, "sc-cx-imaginary-part-dropped", const=False),
            Cand(a**3 - 3 * a * b**2, "sc-cx-parts-swapped-in-expansion", const=False),
        ]
    pc = part(
        f"Expand (a + bi){nsup} with a = Re(z) and b = Im(z), and use your answers to (a) and (b) to find Re(z{nsup}).",
        f"Expand (a + bi){nsup} with a = Re(z) and b = Im(z), and use your answers to (a) and (b) to find Re(z{nsup}).",
        2,
        [f"(a + bi){nsup} has real part {re_form} (using i^2 = -1), so Re(z{nsup}) = {re_n}."],
        exact_fields(re_n),
        _options(re_n, _nonzero(re_wrong), _around(r**n, _re_or_im("Re"), n * p, limit=r**n, key=re_n), nonzero=True),
        follow_through=ft_re,
    )
    pd = part(
        f"Use your answers to (a) and (b) to find the imaginary part of z{nsup}, Im(z{nsup}).",
        f"Use your answers to (a) and (b) to find the imaginary part of z{nsup}, Im(z{nsup}).",
        2,
        [f"(a + bi){nsup} has imaginary part {im_form}, so Im(z{nsup}) = {im_n}."],
        exact_fields(im_n),
        _options(im_n, _nonzero(im_wrong), _around(r**n, _re_or_im("Im"), n * p, limit=r**n, key=im_n), nonzero=True),
        follow_through=ft_im,
    )
    modulus = sympy.Integer(r**n)
    pe = part(
        f"Use your answers to (c) and (d) to find |z{nsup}|, the modulus of z{nsup}.",
        f"Use your answers to (c) and (d) to find |z{nsup}|, the modulus of z{nsup}.",
        2,
        [
            f"|z{nsup}| = √(({re_n})² + ({im_n})²) = {modulus}, which is {r}{nsup}, as the exponential form r{nsup} e^(i{n}θ) says."
        ],
        exact_fields(modulus),
        _options(
            modulus,
            [
                Cand(sympy.nsimplify(abs(re_n) + abs(im_n)), "sc-cx-modulus-sum-not-root", const=False),
                Cand(modulus**2, "cx-modulus-forgot-root", const=False),
                Cand(sympy.Integer(r), "sc-cx-modulus-not-raised", const=False),
                Cand(sympy.Integer(n * r), "cx-power-as-multiplication", const=False),
                Cand(sympy.Integer(r) ** (n + 1), "cx-power-exponent-off-by-one", const=False),
                Cand(sympy.Integer(r) ** (n - 1), "cx-power-exponent-off-by-one", const=False),
            ],
            [modulus + k for k in (-3, -2, -1, 1, 2, 3)] + [modulus + r, modulus - r],
        ),
        follow_through="sqrt(part_c**2 + part_d**2)",
    )
    for each in (pa, pb, pc, pd, pe):
        each["working"] = [_tidy(w) for w in each["working"]]
    return dict(**st, parts=[pa, pb, pc, pd, pe])
