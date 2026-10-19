"""Generates `w5a_complex_log` and `exd_x_w5a_complex_log` questions: the principal logarithm, complex powers and branch cuts.

One second-year skill, ``uc2-complex-log-powers-branches`` (domain ``complex-analysis``). Every answer is an exact
value typed as an expression. Each question is built backwards from a number written in modulus-argument form with a
standard angle, so the principal argument is known exactly before the number is written out in Cartesian form. The
finished value is then confirmed against ``cmath`` (a different route from the exact angle arithmetic).

Tiers:

- ``easy``: ``Log z`` of a number in any quadrant or on an axis, and ``z^(bi)`` for ``z`` = i, -i or -1 (a real answer
  such as e^(-π/2)).
- ``medium``: the principal n-th root of a number whose argument wraps (for example ``(-2 + 2i)^3`` has principal square
  root 1 - i, not -1 + i); the failure of ``Log(z1 z2) = Log z1 + Log z2``; the limit of ``Log(±(z - a))`` at a point of
  its cut from above or below; ``z^(a + bi)`` for ``z`` = i, -i or -1.
- ``hard``: the jump of the principal ``z^(p/q)`` across the negative real axis; ``Log(z^n) - n Log z``; the principal
  ``z^(p/q)`` when ``n`` times the argument leaves (-π, π]; ``z^w`` for a Gaussian-integer ``w``.
- ``exam`` (topic ``exd_x_w5a_complex_log``): a four-part chain on ``f(z) = Log(z - a)``: its value at a point, where it
  is discontinuous (an inequality), a one-sided limit at a point of the cut, and a difference of the two values.
"""

from __future__ import annotations

import cmath
import math
import random
from fractions import Fraction as Fr
from typing import NamedTuple

import sympy
from sympy import Interval, oo

from ..answers import Distractor
from ..core import attempts, check, generator
from ..grading.inequality import set_text
from ._w5a_support import (
    shift,
    Link,
    X,
    astr,
    atext,
    exam_dict,
    ineq_part,
    msay,
    numeric_close,
    same_val,
    single,
)

TOPIC = "w5a_complex_log"
EXAM_TOPIC = "exd_x_w5a_complex_log"

PI = sympy.pi
I = sympy.I  # noqa: E741

_LOG = X("Log", r"\operatorname{Log}")


# -- numbers in modulus-argument form ------------------------------------------------------------------------------------


class Num(NamedTuple):
    x: X  # the number written out, for a prompt
    z: object  # the exact SymPy value
    theta: Fr  # the principal argument as a multiple of pi
    r: object  # the exact modulus


def _surd(e) -> tuple[bool, str, str]:
    """(negative, plain, tex) of a real number c or c*sqrt(r) with integer c."""
    e = sympy.sympify(e)
    c, rest = e.as_coeff_Mul()
    check(c.is_Integer, "integer coefficient")
    neg = c < 0
    c = abs(int(c))
    if rest == 1:
        return neg, str(c), str(c)
    check(rest.is_Pow and rest.exp == sympy.Rational(1, 2), "a plain square root")
    r = int(rest.base)
    lead_p = "" if c == 1 else str(c)
    return neg, f"{lead_p}√{r}", f"{lead_p}\\sqrt{{{r}}}"


def cpx_x(z) -> X:
    """A number with integer or integer-times-surd parts: -1 + √3 i, 2 - 2i, -3i, 5."""
    z = sympy.expand(sympy.sympify(z))
    re_, im_ = z.as_real_imag()
    pieces_p, pieces_t = [], []
    if re_ != 0:
        neg, p, t = _surd(re_)
        pieces_p.append(("-" if neg else "") + p)
        pieces_t.append(("-" if neg else "") + t)
    if im_ != 0:
        neg, p, t = _surd(im_)
        unit_p = "i" if p == "1" else (p + " i" if "√" in p else p + "i")
        unit_t = "i" if t == "1" else (t + r"\,i" if "sqrt" in t else t + "i")
        if pieces_p:
            pieces_p.append((" - " if neg else " + ") + unit_p)
            pieces_t.append(("-" if neg else "+") + unit_t)
        else:
            pieces_p.append(("-" if neg else "") + unit_p)
            pieces_t.append(("-" if neg else "") + unit_t)
    if not pieces_p:
        return X("0", "0")
    return X("".join(pieces_p), "".join(pieces_t))


def cpx_safe(z) -> X:
    """`cpx_x`, falling back to the generic exact text for sums of surds."""
    try:
        return cpx_x(z)
    except Exception:
        t = atext(sympy.expand(z))
        return X(t, t)


def _paren(x: X) -> X:
    return X(f"({x.p})", f"\\left({x.t}\\right)")


def _principal(t: Fr) -> tuple[Fr, int]:
    """(principal argument in (-1, 1], the multiple m of 2 removed) for an angle t, a multiple of pi."""
    m = math.ceil((t - 1) / 2)
    return t - 2 * m, m


_BASE = {
    # (re sign, im sign) -> argument, as a multiple of pi, for each family
    "45": {(1, 1): Fr(1, 4), (-1, 1): Fr(3, 4), (1, -1): Fr(-1, 4), (-1, -1): Fr(-3, 4)},
    "60": {(1, 1): Fr(1, 3), (-1, 1): Fr(2, 3), (1, -1): Fr(-1, 3), (-1, -1): Fr(-2, 3)},
    "30": {(1, 1): Fr(1, 6), (-1, 1): Fr(5, 6), (1, -1): Fr(-1, 6), (-1, -1): Fr(-5, 6)},
}


def _make_num(rng: random.Random, axes: bool = True, scale: tuple[int, int] = (1, 3), families=None) -> Num:
    kinds = list(families or ["45", "60", "30"]) + (["axis"] if axes else [])
    kind = rng.choice(kinds)
    a = rng.randint(*scale)
    if kind == "axis":
        theta, z = rng.choice([(Fr(1, 2), a * I), (Fr(-1, 2), -a * I), (Fr(1), sympy.Integer(-a))])
        return Num(cpx_x(z), z, theta, sympy.Integer(a))
    sx, sy = rng.choice([1, -1]), rng.choice([1, -1])
    theta = _BASE[kind][(sx, sy)]
    if kind == "45":
        z = a * (sx + sy * I)
        r = a * sympy.sqrt(2)
    elif kind == "60":
        z = a * (sx + sy * sympy.sqrt(3) * I)
        r = sympy.Integer(2 * a)
    else:
        z = a * (sx * sympy.sqrt(3) + sy * I)
        r = sympy.Integer(2 * a)
    return Num(cpx_x(z), z, theta, r)


def _cn(v) -> complex:
    return complex(sympy.N(sympy.sympify(v), 20))


def _log_value(n: Num):
    return sympy.log(n.r) + I * PI * sympy.Rational(n.theta.numerator, n.theta.denominator)


def exp_text(arg) -> str:
    """e^(a + bi) for an exact exponent a + bi, with a and b real and written out."""
    re_, im_ = sympy.expand(arg).as_real_imag()
    re_, im_ = sympy.expand(re_), sympy.expand(im_)
    parts = []
    if re_ != 0:
        parts.append(atext(re_))
    if im_ != 0:
        neg = im_.could_extract_minus_sign()
        mag = atext(-im_ if neg else im_)
        term = "i" if mag == "1" else (f"{mag}i" if mag.replace(".", "").isdigit() else f"({mag})i")
        parts.append((" - " if neg else " + ") + term if parts else ("-" if neg else "") + term)
    return "e^(" + "".join(parts) + ")" if parts else "1"


def _tr(t: Fr):
    return sympy.Rational(t.numerator, t.denominator)


def _i_pow(k: int):
    return I ** (k % 4)


# -- easy ----------------------------------------------------------------------------------------------------------------


def _easy_log(rng: random.Random) -> dict:
    n = _make_num(rng)
    value = _log_value(n)
    numeric_close(cmath.log(_cn(n.z)), _cn(value), "Log z")
    th = _tr(n.theta)
    logr = sympy.log(n.r)
    # the naive arctan(y/x) differs from the principal argument by pi when the real part is negative
    naive = n.theta - 1 if n.theta > Fr(1, 2) else n.theta + 1 if n.theta < Fr(-1, 2) else n.theta + 1
    wrongs = [
        (logr + I * PI * _tr(naive), "cx2-log-arctan-quadrant"),
        (logr - I * PI * th, "cx2-log-argument-sign"),
        (logr + I * PI * (th + 2), "cx2-log-argument-not-principal"),
        (n.r + I * PI * th, "cx2-log-modulus-not-logged"),
        (2 * logr + I * PI * th, "cx2-log-modulus-squared"),
        (logr + I * PI * (th - 2), "cx2-log-argument-not-principal"),
    ]
    arg_p = f"{atext(th * PI)}" if th != 0 else "0"
    working = [
        f"Write z in modulus-argument form: |z| = {atext(n.r)} and, taking the argument in (-π, π], Arg z = {arg_p}.",
        f"Log z = ln|z| + i Arg z = {atext(value)}.",
    ]
    zx = n.x
    lead = [
        lambda: msay(
            "Find the principal value ",
            X(f"Log z", r"\operatorname{Log} z"),
            " for ",
            X(f"z = {zx.p}", f"z = {zx.t}"),
            ", in exact form.",
        ),
        lambda: msay(
            "Evaluate ",
            X(f"Log({zx.p})", rf"\operatorname{{Log}}\left({zx.t}\right)"),
            ", where Log is the principal branch with -π < Arg ≤ π.",
        ),
        lambda: msay("Write the principal logarithm of ", X(zx.p, zx.t), " as a + bi, with a and b exact."),
    ][rng.randrange(3)]
    return single(lead(), value, working, wrongs)


_UNIT = [(Fr(1, 2), "i", "i"), (Fr(-1, 2), "(-i)", r"\left(-i\right)"), (Fr(1), "(-1)", r"\left(-1\right)")]


def _easy_unit_power(rng: random.Random) -> dict:
    theta, bp, bt = rng.choice(_UNIT)
    b = rng.choice([1, 2, 3, -1, -2])
    ex = X(f"{b}i" if abs(b) != 1 else ("i" if b == 1 else "-i"), f"{b}i" if abs(b) != 1 else ("i" if b == 1 else "-i"))
    th = _tr(theta)
    value = sympy.exp(-b * th * PI)
    numeric_close(
        cmath.exp(1j * b * cmath.log(_cn({Fr(1, 2): I, Fr(-1, 2): -I, Fr(1): -1}[theta]))), _cn(value), "z^(bi)"
    )
    wrongs = [
        (sympy.exp(b * th * PI), "cx2-power-exponent-sign"),
        (
            sympy.exp(-b * (th + (2 if theta < 0 else -2 if theta > Fr(1, 2) else 2)) * PI),
            "cx2-power-nonprincipal-branch",
        ),
        (sympy.Integer(1), "cx2-power-modulus-only"),
        (sympy.exp(-2 * b * th * PI), "cx2-power-argument-doubled"),
        (sympy.exp(-b * th * PI / 2), "cx2-power-argument-doubled"),
        (-value, "calc-sign-slip"),
    ]
    working = [
        f"z^w = exp(w Log z). Here |z| = 1 and Arg z = {atext(th * PI)}, so Log z = {atext(I * th * PI)}.",
        f"Then w Log z = ({b}i)({atext(I * th * PI)}) = {atext(-b * th * PI)}, and z^w = {atext(value)}.",
    ]
    base = X(bp, bt)
    expo = X(f"({ex.p})", f"{{{ex.t}}}")
    lead = [
        lambda: msay("Find the principal value of ", X(f"{base.p}^{expo.p}", f"{base.t}^{expo.t}"), ", in exact form."),
        lambda: msay("Using the principal branch, evaluate ", X(f"{base.p}^{expo.p}", f"{base.t}^{expo.t}"), "."),
        lambda: msay(
            "Show that ",
            X(f"{base.p}^{expo.p}", f"{base.t}^{expo.t}"),
            " is real, and give its value as an exact expression.",
        ),
    ][rng.randrange(3)]
    return single(lead(), value, working, wrongs)


# -- medium --------------------------------------------------------------------------------------------------------------


def _root_cases():
    return [2, 3, 4]


def _medium_root(rng: random.Random) -> dict:
    for _ in attempts(500, "a principal root"):
        n = rng.choice(_root_cases())
        u = _make_num(rng, axes=False if n != 4 else True, scale=(1, 3))
        z = sympy.expand(u.z**n)
        t_total = u.theta * n
        t_z, m = _principal(t_total)
        if n == 3 and m != 0:
            continue
        if z == 0:
            continue
        break
    # the principal root is u times a root of unity of order n: (-1)^m for n = 2, (-i)^m for n = 4
    if n == 2:
        root = u.z * (-1) ** m
    elif n == 4:
        root = sympy.expand(u.z * _i_pow(-m))
    else:
        root = u.z
    root = sympy.expand(root)
    numeric_close(_cn(z) ** (1 / n) if False else cmath.exp(cmath.log(_cn(z)) / n), _cn(root), "principal root")
    wrongs = [
        (-root, "cx2-root-other-branch"),
        (sympy.expand(u.z), "cx2-root-argument-not-reduced")
        if m != 0
        else (sympy.expand(root * _i_pow(1)), "cx2-root-other-branch"),
        (sympy.expand(root * _i_pow(1)), "cx2-root-other-branch"),
        (sympy.expand(sympy.conjugate(root)), "cx2-log-argument-sign"),
        (abs(root), "cx2-root-modulus-only"),
        (sympy.expand(root * _i_pow(-1)), "cx2-root-other-branch"),
    ]
    zx = cpx_x(z)
    ang = u.theta
    working = [
        f"The principal value is z^(1/{n}) = exp(Log z / {n}). Here |z| = {atext(abs(sympy.sympify(z)) if False else sympy.sqrt(sympy.expand(z * sympy.conjugate(z))))}.",
        f"Arg z = {atext(_tr(t_z) * PI)} (reduced into (-π, π]), so z^(1/{n}) has argument {atext(_tr(t_z / n) * PI)}.",
        f"Hence z^(1/{n}) = {atext(root)}.",
    ]
    nth = {2: "square root", 3: "cube root", 4: "fourth root"}[n]
    del ang
    lead = [
        lambda: msay(
            "Find the principal value of ",
            X(f"({zx.p})^(1/{n})", rf"\left({zx.t}\right)^{{1/{n}}}"),
            ", in the form a + bi.",
        ),
        lambda: msay(
            "Using the principal branch, evaluate ",
            X(f"z^(1/{n})", rf"z^{{1/{n}}}"),
            " for ",
            X(f"z = {zx.p}", f"z = {zx.t}"),
            ".",
        ),
        lambda: msay(
            "Let ",
            X(f"z = {zx.p}", f"z = {zx.t}"),
            ". Find the principal ",
            nth,
            " ",
            X(f"z^(1/{n})", rf"z^{{1/{n}}}"),
            ", defined as exp(Log(z)/" + str(n) + ").",
        ),
    ][rng.randrange(3)]
    return single(lead(), root, working, wrongs, fallback_tag="cx2-root-other-branch")


def _medium_log_product(rng: random.Random) -> dict:
    for _ in attempts(500, "a pair with a wrap"):
        n1 = _make_num(rng, axes=True)
        n2 = _make_num(rng, axes=True)
        _t, m = _principal(n1.theta + n2.theta)
        if m == 0 and rng.random() < 0.65:
            continue
        break
    value = -2 * PI * I * m
    z1, z2 = n1.z, n2.z
    prod = sympy.expand(z1 * z2)
    numeric_close(
        cmath.log(_cn(prod)) - cmath.log(_cn(z1)) - cmath.log(_cn(z2)),
        _cn(value),
        "Log(z1 z2) - Log z1 - Log z2",
    )
    if m == 0:
        wrongs = [
            (2 * PI * I, "cx2-log-product-wrong-multiple"),
            (-2 * PI * I, "cx2-log-product-wrong-multiple"),
            (PI * I, "cx2-log-product-wrong-multiple"),
            (4 * PI * I, "cx2-log-product-wrong-multiple"),
        ]
    else:
        wrongs = [
            (sympy.Integer(0), "cx2-log-product-additive"),
            (2 * PI * I * m, "cx2-log-product-sign"),
            (-2 * PI * I * (m + 1), "cx2-log-product-wrong-multiple"),
            (-2 * PI * I * (m - 1), "cx2-log-product-wrong-multiple"),
            (-PI * I * m, "cx2-log-product-wrong-multiple"),
        ]
    t1, t2 = n1.theta, n2.theta
    tz, _m = _principal(t1 + t2)
    working = [
        f"Arg z1 = {atext(_tr(t1) * PI)} and Arg z2 = {atext(_tr(t2) * PI)}, so the arguments add to {atext(_tr(t1 + t2) * PI)}.",
        f"z1 z2 = {cpx_safe(prod).p}, whose principal argument is {atext(_tr(tz) * PI)}, so Arg(z1 z2) = Arg z1 + Arg z2 - 2π·({m}) and the moduli's logarithms add exactly.",
        f"Log(z1 z2) - Log z1 - Log z2 = {atext(value)}.",
    ]
    lhs = X(
        "Log(z1 z2) - Log z1 - Log z2", r"\operatorname{Log}(z_1 z_2) - \operatorname{Log} z_1 - \operatorname{Log} z_2"
    )
    pair = X(f"z1 = {n1.x.p} and z2 = {n2.x.p}", f"z_1 = {n1.x.t}\\text{{ and }}z_2 = {n2.x.t}")
    lead = [
        lambda: msay("Let ", pair, ". Evaluate ", lhs, ", using the principal logarithm."),
        lambda: msay("For ", pair, ", find the exact value of ", lhs, ". (Log is the principal branch.)"),
        lambda: msay(
            "The principal logarithm does not always satisfy Log(z1 z2) = Log z1 + Log z2. With ",
            pair,
            ", find ",
            lhs,
            ".",
        ),
    ][rng.randrange(3)]
    return single(lead(), value, working, wrongs)


def _medium_limit(rng: random.Random) -> dict:
    s = rng.choice([1, -1])
    a = rng.randint(-3, 3)
    d = rng.randint(1, 4)
    p = a - s * d  # s (p - a) = -d < 0, so p is on the cut
    direction = rng.choice([1, -1])  # approach from above (+1) or below (-1)
    sign = s * direction  # the sign of Im(s (z - a)) near p
    value = sympy.log(d) + I * PI * sign
    eps = 1e-9
    numeric_close(cmath.log(s * (complex(p, direction * eps) - a)), _cn(value), "one-sided limit", 1e-6)
    wrongs = [
        (sympy.log(d) - I * PI * sign, "cx2-cut-wrong-side"),
        (sympy.log(d), "cx2-cut-argument-dropped"),
        (sympy.log(d) + I * PI * 2 * sign, "cx2-log-argument-not-principal"),
        (sympy.log(d) + I * PI * sign * 0 + I * PI * sign / 2, "cx2-cut-argument-dropped"),
        (I * PI * sign, "cx2-log-modulus-not-logged"),
    ]
    inner = (
        X("z - " + str(a), f"z - {a}")
        if s == 1 and a > 0
        else X("z + " + str(-a), f"z + {-a}")
        if s == 1 and a < 0
        else X("z", "z")
        if s == 1
        else X(f"{a} - z" if a != 0 else "-z", f"{a} - z" if a != 0 else "-z")
    )
    fx = X(f"f(z) = Log({inner.p})", rf"f(z) = \operatorname{{Log}}\left({inner.t}\right)")
    side = "above" if direction == 1 else "below"
    half = "Im z > 0" if direction == 1 else "Im z < 0"
    working = [
        f"At z = {p} the argument of the logarithm is {atext(s * (p - a))} = -{d}, which is on the cut of Log (the negative real axis).",
        f"For Im z {'>' if direction == 1 else '<'} 0 the quantity {inner.p} has imaginary part of sign {'+' if sign == 1 else '-'}, so its argument tends to {'π' if sign == 1 else '-π'}.",
        f"The limit is ln {d} {'+' if sign == 1 else '-'} πi = {atext(value)}.",
    ]
    lead = [
        lambda: msay(
            "Let ",
            fx,
            " with the principal branch. Find the limit of f(z) as z tends to ",
            X(str(p), str(p)),
            f" from {side} the real axis (through points with {half}).",
        ),
        lambda: msay(
            "For ",
            fx,
            " (principal Log), evaluate ",
            X(f"lim f(z) as z → {p}, {half}", rf"\lim_{{z \to {p},\ {half}}} f(z)"),
            ".",
        ),
        lambda: msay(
            "The function ",
            fx,
            f" is discontinuous across a ray. Find the value f takes as z approaches {p} from {side}.",
        ),
    ][rng.randrange(3)]
    return single(lead(), value, working, wrongs, fallback_tag="cx2-cut-wrong-side")


def _medium_unit_power(rng: random.Random) -> dict:
    theta, bp, bt = rng.choice(_UNIT)
    a = rng.choice([1, 2, 3, -1, -2])
    b = rng.choice([1, 2, -1, -2])
    th = _tr(theta)
    w = a + b * I
    value = sympy.expand(sympy.exp(-b * th * PI) * sympy.exp(I * a * th * PI))
    zval = {Fr(1, 2): I, Fr(-1, 2): -I, Fr(1): sympy.Integer(-1)}[theta]
    numeric_close(cmath.exp(_cn(w) * cmath.log(_cn(zval))), _cn(value), "z^w")
    wrongs = [
        (sympy.exp(-b * th * PI), "cx2-power-real-part-ignored"),
        (sympy.expand(sympy.exp(b * th * PI) * sympy.exp(I * a * th * PI)), "cx2-power-exponent-sign"),
        (
            sympy.expand(
                sympy.exp(-b * (th + (2 if theta < 0 else -2 if theta > Fr(1, 2) else 2)) * PI)
                * sympy.exp(I * a * (th + (2 if theta < 0 else -2 if theta > Fr(1, 2) else 2)) * PI)
            ),
            "cx2-power-nonprincipal-branch",
        ),
        (sympy.expand(sympy.exp(I * a * th * PI)), "cx2-power-imaginary-part-ignored"),
        (sympy.expand(sympy.exp(-b * th * PI) * sympy.exp(-I * a * th * PI)), "cx2-log-argument-sign"),
        (sympy.expand(-value), "calc-sign-slip"),
    ]
    wtx = cpx_x(w)
    working = [
        f"z^w = exp(w Log z) with Log z = {atext(I * th * PI)} (|z| = 1, Arg z = {atext(th * PI)}).",
        f"w Log z = ({wtx.p})({atext(I * th * PI)}) = {atext(sympy.expand(w * I * th * PI))}, so z^w = {atext(value)}.",
    ]
    lead = [
        lambda: msay("Find the principal value of ", X(f"{bp}^({wtx.p})", f"{bt}^{{{wtx.t}}}"), ", in exact form."),
        lambda: msay(
            "Using the principal branch, evaluate ",
            X(f"{bp}^({wtx.p})", f"{bt}^{{{wtx.t}}}"),
            " as an exact expression.",
        ),
        lambda: msay(
            "Write ",
            X(f"{bp}^({wtx.p})", f"{bt}^{{{wtx.t}}}"),
            " in the form (real number) times (a power of i), where the principal logarithm is used.",
        ),
    ][rng.randrange(3)]
    return single(lead(), value, working, wrongs, fallback_tag="cx2-power-nonprincipal-branch")


# -- hard ----------------------------------------------------------------------------------------------------------------

_JUMPS = [(1, 2), (1, 3), (2, 3), (1, 4), (3, 4), (1, 6), (5, 6), (3, 2), (5, 2), (4, 3), (5, 4), (7, 6)]


def _hard_jump(rng: random.Random) -> dict:
    p, q = rng.choice(_JUMPS)
    rho = rng.choice([2, 3] if q <= 4 else [2])
    r = rho**q
    expo = sympy.Rational(p, q)
    value = sympy.simplify(2 * I * rho**p * sympy.sin(PI * expo))
    eps = 1e-10

    def f(z):
        return cmath.exp(float(expo) * cmath.log(z))

    numeric_close(f(complex(-r, eps)) - f(complex(-r, -eps)), _cn(value), "jump", 1e-6)
    wrongs = [
        (sympy.simplify(rho**p * sympy.sin(PI * expo) * I), "cx2-jump-factor-two-dropped"),
        (sympy.simplify(-value), "cx2-jump-sign-reversed"),
        (sympy.simplify(2 * rho**p * sympy.cos(PI * expo)), "cx2-jump-sum-not-difference"),
        (sympy.simplify(2 * rho**p * sympy.sin(PI * expo)), "cx2-jump-i-dropped"),
        (sympy.simplify(2 * I * sympy.Integer(r) * sympy.sin(PI * expo)), "cx2-jump-modulus-power-slip"),
        (sympy.simplify(2 * I * rho**p * sympy.cos(PI * expo)), "cx2-jump-sum-not-difference"),
    ]
    # the stated exponent as p/q
    ex = X(f"{p}/{q}", rf"\frac{{{p}}}{{{q}}}")
    working = [
        f"Just above the cut z = -{r} + iε has argument → π and just below it → -π, with |z| = {r}, so |z|^({p}/{q}) = {rho}^{p} = {rho**p}.",
        f"f(-{r} + i0) - f(-{r} - i0) = {rho**p}(e^(iπ·{p}/{q}) - e^(-iπ·{p}/{q})) = {rho**p}·2i sin({p}π/{q}) = {atext(value)}.",
    ]
    lead = [
        lambda: msay(
            "Let ",
            X(f"f(z) = z^({p}/{q})", rf"f(z) = z^{{{p}/{q}}}"),
            " with the principal branch (Arg z in (-π, π]). Find ",
            X(
                f"lim [f(-{r} + iε) - f(-{r} - iε)]",
                rf"\lim_{{\varepsilon \to 0^+}} \left[f(-{r} + i\varepsilon) - f(-{r} - i\varepsilon)\right]",
            ),
            " as ε → 0 from above.",
        ),
        lambda: msay(
            "The principal ",
            X(f"z^({p}/{q})", rf"z^{{{p}/{q}}}"),
            f" jumps across the negative real axis. Find the size and direction of the jump at z = -{r}, that is f(-{r} + i0) - f(-{r} - i0), exactly.",
        ),
        lambda: msay(
            "Define ",
            X(f"f(z) = exp(({p}/{q}) Log z)", rf"f(z) = \exp\left(\frac{{{p}}}{{{q}}} \operatorname{{Log}} z\right)"),
            f". Compute the difference between the values of f just above and just below the cut at z = -{r}.",
        ),
    ][rng.randrange(3)]
    del ex
    return single(lead(), value, working, wrongs, fallback_tag="cx2-jump-sign-reversed")


def _hard_log_power(rng: random.Random) -> dict:
    for _ in attempts(500, "a wrapping power"):
        n = rng.choice([2, 3, 4, 5, -2, -3])
        num = _make_num(rng, axes=rng.random() < 0.25)
        _t, m = _principal(num.theta * n)
        if m == 0:
            continue
        break
    value = -2 * PI * I * m
    zn = sympy.expand(num.z**n)
    numeric_close(cmath.log(_cn(zn)) - n * cmath.log(_cn(num.z)), _cn(value), "Log(z^n) - n Log z")
    wrongs = [
        (sympy.Integer(0), "cx2-log-power-rule-assumed"),
        (2 * PI * I * m, "cx2-log-product-sign"),
        (-2 * PI * I * (m + 1), "cx2-log-product-wrong-multiple"),
        (-2 * PI * I * (m - 1), "cx2-log-product-wrong-multiple"),
        (-2 * PI * I * n * m, "cx2-log-product-wrong-multiple"),
        (-2 * PI * I, "cx2-log-product-wrong-multiple"),
    ]
    zx = num.x
    lhs_p = f"Log(z^{n}) - {n} Log z" if n > 0 else f"Log(z^({n})) + {-n} Log z"
    lhs_t = (
        rf"\operatorname{{Log}}(z^{{{n}}}) - {n}\operatorname{{Log}}\, z"
        if n > 0
        else rf"\operatorname{{Log}}(z^{{{n}}}) + {-n}\operatorname{{Log}}\, z"
    )
    ztn = cpx_x(zn) if n > 0 else X(atext(zn), atext(zn))
    working = [
        f"Arg z = {atext(_tr(num.theta) * PI)}, so n·Arg z = {atext(_tr(num.theta * n) * PI)}, which is outside (-π, π].",
        f"z^({n}) = {ztn.p}, whose principal argument is {atext(_tr(_t) * PI)} = n Arg z - 2π·({m}).",
        f"Therefore {lhs_p} = -2πi·({m}) = {atext(value)}.",
    ]
    lead = [
        lambda: msay("For ", X(f"z = {zx.p}", f"z = {zx.t}"), f", evaluate {lhs_p} using the principal logarithm."),
        lambda: msay(
            "Let ",
            X(f"z = {zx.p}", f"z = {zx.t}"),
            ". The identity Log(zⁿ) = n Log z can fail. Find ",
            X(lhs_p, lhs_t),
            ".",
        ),
        lambda: msay("Find the exact value of ", X(lhs_p, lhs_t), " when ", X(f"z = {zx.p}", f"z = {zx.t}"), "."),
    ][rng.randrange(3)]
    return single(lead(), value, working, wrongs, fallback_tag="cx2-log-product-wrong-multiple")


def _hard_fractional_power(rng: random.Random) -> dict:
    for _ in attempts(800, "a wrapping fractional power"):
        q = rng.choice([2, 4, 4])
        p = rng.choice([3, 5, 7]) if q == 4 else rng.choice([3, 5])
        u = _make_num(rng, axes=False, families=["45", "60", "30"], scale=(1, 2))
        z = sympy.expand(u.z**q)
        _t, m = _principal(u.theta * q)
        if m == 0:
            continue
        break
    # exp((p/q) Log z) = u^p * exp(-2 pi i p m / q)
    ph = sympy.exp(-2 * PI * I * sympy.Rational(p * m, q))
    value = sympy.expand(sympy.expand(u.z**p) * ph)
    numeric_close(cmath.exp(p / q * cmath.log(_cn(z))), _cn(value), "z^(p/q)")
    unwrapped = sympy.expand(u.z**p)
    wrongs = [
        (unwrapped, "cx2-power-nonprincipal-branch"),
        (sympy.expand(-value), "calc-sign-slip"),
        (sympy.expand(sympy.conjugate(value)), "cx2-log-argument-sign"),
        (sympy.expand(unwrapped * _i_pow(1)), "cx2-power-nonprincipal-branch"),
        (sympy.expand(unwrapped * _i_pow(-1)), "cx2-power-nonprincipal-branch"),
        (sympy.expand(value * I), "cx2-power-nonprincipal-branch"),
    ]
    zx = cpx_x(z)
    working = [
        f"z = {zx.p} has modulus {atext(sympy.sqrt(sympy.expand(z * sympy.conjugate(z))))} and principal argument {atext(_tr(_t) * PI)} (reduced from {atext(_tr(u.theta * q) * PI)}).",
        f"z^({p}/{q}) = exp(({p}/{q}) Log z) has modulus {atext(sympy.sqrt(sympy.expand(value * sympy.conjugate(value))))} and argument {atext(_tr(_t * Fr(p, q)) * PI)}, so z^({p}/{q}) = {atext(value)}.",
    ]
    lead = [
        lambda: msay(
            "Find the principal value of ",
            X(f"({zx.p})^({p}/{q})", rf"\left({zx.t}\right)^{{{p}/{q}}}"),
            ", in the form a + bi.",
        ),
        lambda: msay(
            "Using the principal branch (z^w = exp(w Log z)), evaluate ",
            X(f"z^({p}/{q})", rf"z^{{{p}/{q}}}"),
            " for ",
            X(f"z = {zx.p}", f"z = {zx.t}"),
            ".",
        ),
        lambda: msay(
            "Let ",
            X(f"z = {zx.p}", f"z = {zx.t}"),
            f". Compute z^({p}/{q}), defined as exp(({p}/{q}) Log z), in exact form.",
        ),
    ][rng.randrange(3)]
    return single(lead(), value, working, wrongs, fallback_tag="cx2-power-nonprincipal-branch")


_WS = [(1, 1), (2, 1), (1, -1), (3, 1), (-1, 1), (1, 2), (2, -1), (5, 1), (4, -1), (3, -1), (-2, 1)]


def _hard_unit_circle_power(rng: random.Random) -> dict:
    """z^(a+bi) for z on the unit circle at a standard angle: a real factor exp(-b*theta) times cos + i sin."""
    for _ in attempts(500, "a clean unit-circle power"):
        n = _make_num(rng, axes=False, scale=(1, 1))
        a, b = rng.choice(_WS)
        th = _tr(n.theta)
        ang = a * th * PI
        # cos and sin of a*theta must be a standard value: a*theta a multiple of pi/6 or pi/4
        if (a * n.theta * 12) % 1 != 0 or (a * n.theta * 12) % 2 != 0 and (a * n.theta * 12) % 3 != 0:
            continue
        break
    value = sympy.exp(-b * th * PI) * (sympy.cos(ang) + I * sympy.sin(ang))
    zc = sympy.expand(n.z / n.r)
    numeric_close(cmath.exp((a + b * 1j) * cmath.log(_cn(zc))), _cn(value), "z^w")
    other = th + (2 if th < 0 else -2)
    wrongs = [
        (sympy.exp(-b * th * PI), "cx2-power-real-part-ignored"),
        (sympy.exp(b * th * PI) * (sympy.cos(ang) + I * sympy.sin(ang)), "cx2-power-exponent-sign"),
        (
            sympy.exp(-b * other * PI) * (sympy.cos(a * other * PI) + I * sympy.sin(a * other * PI)),
            "cx2-power-nonprincipal-branch",
        ),
        (sympy.cos(ang) + I * sympy.sin(ang), "cx2-power-imaginary-part-ignored"),
        (sympy.exp(-b * th * PI) * (sympy.cos(ang) - I * sympy.sin(ang)), "cx2-log-argument-sign"),
        (
            sympy.exp(-a * th * PI) * (sympy.cos(b * th * PI) + I * sympy.sin(b * th * PI)),
            "cx2-power-exponent-conjugated",
        ),
    ]
    zt = n.x
    if n.r == sympy.sqrt(2) * 1:
        base = X(f"({zt.p})/√2", rf"\frac{{{zt.t}}}{{\sqrt{{2}}}}")
    else:
        base = X(f"({zt.p})/2", rf"\frac{{{zt.t}}}{{2}}")
    wtx = cpx_x(a + b * I)
    working = [
        f"|z| = 1 and Arg z = {atext(th * PI)}, so Log z = {atext(I * th * PI)}.",
        f"w Log z = ({wtx.p})({atext(I * th * PI)}) = {atext(sympy.expand((a + b * I) * I * th * PI))}.",
        f"Hence z^w = e^({atext(-b * th * PI)}) (cos({atext(ang)}) + i sin({atext(ang)})) = {atext(value)}.",
    ]
    lead = [
        lambda: msay(
            "Let ",
            X(f"z = {base.p}", f"z = {base.t}"),
            ". Find the principal value of ",
            X(f"z^({wtx.p})", rf"z^{{{wtx.t}}}"),
            ", in exact form.",
        ),
        lambda: msay(
            "Using the principal branch, evaluate ",
            X(f"({base.p})^({wtx.p})", rf"\left({base.t}\right)^{{{wtx.t}}}"),
            ".",
        ),
        lambda: msay(
            "The number ",
            X(f"z = {base.p}", f"z = {base.t}"),
            " has modulus 1. Find ",
            X(f"z^({wtx.p})", rf"z^{{{wtx.t}}}"),
            " (principal value) in the form (real number)(a + bi) with exact parts.",
        ),
    ][rng.randrange(3)]
    return single(lead(), value, working, wrongs, fallback_tag="cx2-power-nonprincipal-branch")


_TIERS = {
    "easy": [_easy_log, _easy_log, _easy_unit_power],
    "medium": [_medium_root, _medium_log_product, _medium_limit, _medium_unit_power],
    "hard": [_hard_jump, _hard_log_power, _hard_fractional_power, _hard_unit_circle_power],
}


@generator(
    TOPIC,
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def w5a_complex_log(rng: random.Random, tier: str = "easy") -> dict:
    """A principal logarithm, complex power or branch-cut value, exact."""
    check(tier in _TIERS, f"unknown tier {tier!r}")
    return rng.choice(_TIERS[tier])(rng)


# -- exam: a four-part chain -----------------------------------------------------------------------------------------------


@generator(EXAM_TOPIC, level=19, presets={"exam": dict(tier="exam")})
def exd_x_w5a_complex_log(rng: random.Random, tier: str = "exam") -> dict:
    """f(z) = Log(z - a): its value at a point, where it is discontinuous, a one-sided limit on the cut, a difference."""
    check(tier == "exam", f"unknown tier {tier!r}")
    a = rng.randint(-2, 2)
    s = rng.randint(1, 3)
    num = _make_num(rng, axes=False, scale=(1, 2), families=["45", "60"])
    point = sympy.expand(a + num.z)
    below = a - s
    va = _log_value(num)
    vc = sympy.log(s) - I * PI
    vd = sympy.expand(va - vc)
    numeric_close(cmath.log(_cn(point) - a), _cn(va), "Log(p - a)")
    numeric_close(cmath.log(complex(below - a, -1e-9)), _cn(vc), "one-sided limit", 1e-6)
    if a == 0:
        inner, inner_t = "z", "z"
    elif a > 0:
        inner, inner_t = f"z - {a}", f"z - {a}"
    else:
        inner, inner_t = f"z + {-a}", f"z + {-a}"
    fx = X(f"f(z) = Log({inner})", rf"f(z) = \operatorname{{Log}}\left({inner_t}\right)")
    stem = msay("Let ", fx, ", where Log is the principal logarithm, with -π < Arg ≤ π.")
    px = cpx_x(point)
    th = _tr(num.theta)
    wa = Link(
        msay("Find f(", X(px.p, px.t), ") in exact form."),
        va,
        "expression",
        3,
        [
            f"{shift(a)} = {num.x.p}, so |{shift(a)}| = {atext(num.r)} and Arg = {atext(th * PI)}.",
            f"f = ln|{shift(a)}| + i Arg({shift(a)}) = {atext(va)}.",
        ],
        [
            (
                sympy.log(num.r) + I * PI * (th + (-1 if num.theta > Fr(1, 2) else 1 if num.theta < Fr(-1, 2) else 1)),
                "cx2-log-arctan-quadrant",
            ),
            (sympy.log(num.r) - I * PI * th, "cx2-log-argument-sign"),
            (sympy.log(num.r) + I * PI * (th + 2), "cx2-log-argument-not-principal"),
            (num.r + I * PI * th, "cx2-log-modulus-not-logged"),
        ],
    )
    cut = Interval(-oo, a)
    cut_text = set_text(cut)
    wb = ineq_part(
        msay("f is not continuous (or not defined) at some real numbers z = x. Find the set of such x."),
        cut_text,
        2,
        [
            f"Log is discontinuous on the non-positive real axis, so {shift(a)} must be real and ≤ 0 for f to jump.",
            f"That is {shift(a, 'x')} ≤ 0, so x ≤ {a} (including x = {a}, where f is undefined).",
        ],
        [
            (set_text(Interval.open(-oo, a)), "cx2-cut-endpoint"),
            (set_text(Interval(a, oo)), "cx2-cut-wrong-direction"),
            (set_text(Interval.open(a, oo)), "cx2-cut-wrong-direction"),
            (set_text(Interval(-oo, -a)) if a != 0 else set_text(Interval(-oo, 1)), "cx2-cut-wrong-direction"),
        ],
    )
    wc = Link(
        msay(
            "Find the limit of f(z) as z tends to ",
            X(str(below), str(below)),
            " from below the real axis (through points with Im z < 0).",
        ),
        vc,
        "expression",
        3,
        [
            f"At z = {below} the quantity {shift(a)} equals -{s}, which lies on the cut.",
            f"From below, Arg({shift(a)}) → -π, so the limit is ln {s} - πi = {atext(vc)}.",
        ],
        [
            (sympy.log(s) + I * PI, "cx2-cut-wrong-side"),
            (sympy.log(s), "cx2-cut-argument-dropped"),
            (sympy.log(s) - 2 * I * PI, "cx2-log-argument-not-principal"),
            (I * PI * -1, "cx2-log-modulus-not-logged"),
        ],
    )
    wd = Link(
        msay("Hence find f(", X(px.p, px.t), ") minus the limit found in part (c)."),
        vd,
        "expression",
        2,
        [f"{atext(va)} - ({atext(vc)}) = {atext(vd)}."],
        [
            (sympy.expand(va + vc), "cx2-values-added"),
            (sympy.expand(vc - va), "calc-sign-slip"),
            (sympy.expand(va - sympy.log(s) - I * PI), "cx2-cut-wrong-side"),
            (sympy.expand(va), "cx2-values-added"),
        ],
        ft="part_a - part_c",
    )
    return exam_dict(stem, [wa, wb, wc, wd])
