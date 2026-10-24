"""Shared machinery for the graph-transformation builds (`gt_*` generators).

Not a generator module (the leading underscore keeps the package loader from importing it). It holds:

* the family ``y = a F(bx + c) + d`` that every combination of translations, dilations and reflections of a base graph
  stays inside, with the exact update each transformation makes to ``(a, b, c, d)``;
* the typed spelling of such an equation (``-3*(x - 2)^2 + 1``), built from the parameters and not by SymPy, so a
  shift of ``sin`` by pi/2 is written as the shift and not quietly turned into ``cos``;
* the wording of a transformation in a stem, and the wrong chains a learner follows by mistake;
* a numeric reader for a typed equation, used to check every key a second way.

For the trigonometric bases the horizontal numbers (``c``, shifts) are multiples of pi.
"""

from __future__ import annotations

import functools
import math
import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Callable

import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)

from ..core import check
from ._sr2b_kit import TooFewDistractors, pi_text  # noqa: F401  (re-exported)

Frac = Fraction
Params = tuple  # (a, b, c, d) as Fractions


@dataclass(frozen=True)
class Family:
    key: str  # the base graph as the stem writes it: "x^2"
    kind: str  # poly, exp, trig, recip
    fn: Callable[[float], float]  # the base function on floats
    wrap: Callable[[str], str]  # the typed base applied to an argument string
    even: bool = False  # symmetric about the y-axis, so a reflection in it changes nothing

    @property
    def trig(self) -> bool:
        return self.kind == "trig"


def _w(arg: str) -> str:
    return arg if arg == "x" else f"({arg})"


def _pow(n: int):
    return lambda arg: f"{_w(arg)}^{n}"


def _exp(base: str):
    return lambda arg: f"{base}^{_w(arg)}"


FAMILIES = {
    f.key: f
    for f in (
        Family("x^2", "poly", lambda u: u * u, _pow(2), even=True),
        Family("x^3", "poly", lambda u: u**3, _pow(3)),
        Family("2^x", "exp", lambda u: 2.0**u, _exp("2")),
        Family("3^x", "exp", lambda u: 3.0**u, _exp("3")),
        Family("e^x", "exp", math.exp, _exp("e")),
        Family("sin(x)", "trig", math.sin, lambda arg: f"sin({arg})"),
        Family("cos(x)", "trig", math.cos, lambda arg: f"cos({arg})", even=True),
        Family("tan(x)", "trig", math.tan, lambda arg: f"tan({arg})"),
        Family("1/x", "recip", lambda u: 1.0 / u, lambda arg: f"1/{_w(arg)}"),
    )
}

# ------------------------------------------------------------------------------------------- the transformations

#: ("tx", h) moves right by h; ("ty", k) moves up by k; ("rx",) and ("ry",) reflect in the axes; ("dy", c) dilates
#: from the x-axis by factor c (y becomes cy); ("dx", k) dilates from the y-axis by factor k (x becomes kx).


def apply_op(p: Params, op: tuple) -> Params:
    a, b, c, d = p
    kind = op[0]
    if kind == "tx":  # g(x - h): the argument b x + c becomes b x - b h + c
        return a, b, c - b * op[1], d
    if kind == "ty":
        return a, b, c, d + op[1]
    if kind == "rx":
        return -a, b, c, -d
    if kind == "ry":
        return a, -b, c, d
    if kind == "dy":
        return a * op[1], b, c, d * op[1]
    if kind == "dx":  # g(x / k)
        return a, b / op[1], c, d
    raise ValueError(f"unknown transformation {op!r}")


def compose(ops) -> Params:
    p: Params = (Frac(1), Frac(1), Frac(0), Frac(0))
    for op in ops:
        p = apply_op(p, op)
    return p


def move_point(pt: tuple, ops, unit=1) -> tuple:
    """The image of a point (x, y) under the transformations in order: the geometric route, not the equation's.
    `unit` is the length of a horizontal shift of 1 (pi for the trigonometric bases, whose shifts are multiples of pi)."""
    x, y = pt
    for op in ops:
        kind = op[0]
        if kind == "tx":
            x = x + op[1] * unit
        elif kind == "ty":
            y = y + op[1]
        elif kind == "rx":
            y = -y
        elif kind == "ry":
            x = -x
        elif kind == "dy":
            y = y * op[1]
        elif kind == "dx":
            x = x * op[1]
    return x, y


def op_phrase(op: tuple, trig: bool = False) -> str:
    """The transformation as a stem words it."""
    kind = op[0]

    def amount(v: Frac) -> str:
        v = abs(Frac(v))
        return pi_text(v).replace("pi", "π") if trig else frac_text(v)

    if kind == "tx":
        return f"translated {amount(op[1])} unit{'s' if op[1] not in (1, -1) or trig else ''} to the {'right' if op[1] > 0 else 'left'}"
    if kind == "ty":
        return (
            f"translated {frac_text(abs(op[1]))} unit{'s' if abs(op[1]) != 1 else ''} {'up' if op[1] > 0 else 'down'}"
        )
    if kind == "rx":
        return "reflected in the x-axis"
    if kind == "ry":
        return "reflected in the y-axis"
    if kind == "dy":
        return f"dilated by a factor of {frac_text(op[1])} from the x-axis"
    if kind == "dx":
        return f"dilated by a factor of {frac_text(op[1])} from the y-axis"
    raise ValueError(op)


def frac_text(v) -> str:
    v = Frac(v)
    return str(v.numerator) if v.denominator == 1 else f"{v.numerator}/{v.denominator}"


def ops_sentence(ops, trig: bool = False) -> str:
    """'translated 2 units to the right, then reflected in the x-axis' (first, second, then third)."""
    phrases = [op_phrase(o, trig) for o in ops]
    if len(phrases) == 1:
        return phrases[0]
    if len(phrases) == 2:
        return f"{phrases[0]}, then {phrases[1]}"
    return ", then ".join(phrases[:-1]) + f", and finally {phrases[-1]}"


# -------------------------------------------------------------------------------------------------- typed spelling


def shift_of(p: Params) -> Frac:
    """The horizontal shift h of the argument written b(x - h): h = -c / b."""
    return -p[2] / p[1]


def arg_text(p: Params, trig: bool) -> str:
    """The argument ``b x + c`` as typed: ``x - 2``, ``2*(x + 1)``, ``(x - 2)/3``, ``-x + 3``, ``x/2``."""
    _, b, c, _ = p
    if c == 0:
        if b == 1:
            return "x"
        if b == -1:
            return "-x"
        if b.denominator == 1:
            return f"{b.numerator}*x"
        if b.numerator in (1, -1):
            return f"{'-' if b < 0 else ''}x/{b.denominator}"
        return f"{b.numerator}*x/{b.denominator}"
    h = shift_of(p)
    amt = pi_text(abs(h)) if trig else frac_text(abs(h))
    inner = f"x {'-' if h > 0 else '+'} {amt}"
    if b == 1:
        return inner
    if b == -1:
        camt = pi_text(abs(c)) if trig else frac_text(abs(c))
        return f"-x {'+' if c > 0 else '-'} {camt}"
    if b.denominator == 1:
        return f"{b.numerator}*({inner})"
    if b.numerator in (1, -1):
        return f"{'-' if b < 0 else ''}({inner})/{b.denominator}"
    return f"{b.numerator}*({inner})/{b.denominator}"


def normalise(fam: Family, p: Params) -> Params:
    """Fold the horizontal dilation into the leading factor where the base allows it: a (b(x - h))^n is
    a b^n (x - h)^n and a / (b(x - h)) is (a / b) / (x - h), so a power or reciprocal is written in its plain form."""
    a, b, c, d = p
    if fam.kind == "poly":
        n = int(fam.key[-1])
        return a * b**n, Frac(1), c / b, d
    if fam.kind == "recip":
        return a / b, Frac(1), c / b, d
    return p


def equation_text(fam: Family, p: Params, strict: bool = True) -> str:
    """The typed equation's right-hand side, e.g. ``-3*(x - 2)^2 + 1``. `strict` requires a whole-number d."""
    p = normalise(fam, p)
    a, _, _, d = p
    if strict:
        check(d.denominator == 1, f"the vertical shift {d} is not a whole number")
    arg = arg_text(p, fam.trig)
    num, den = a.numerator, a.denominator
    if fam.kind == "recip":
        body = f"{num}/{_w(arg)}" if den == 1 else f"{num}/({den}*{_w(arg)})"
    else:
        lead = "" if num == 1 else "-" if num == -1 else f"{num}*"
        body = f"{lead}{fam.wrap(arg)}" + (f"/{den}" if den != 1 else "")
    if d == 0:
        return body
    return f"{body} {'+' if d > 0 else '-'} {frac_text(abs(d))}"


def tidy(text: str) -> str:
    """No doubled brackets or '+ -' in a built equation."""
    return text.replace("+ -", "- ")


# ------------------------------------------------------------------------------------------------ numeric reading

_X = sympy.Symbol("x")


@functools.lru_cache(maxsize=8192)
def typed_function(text: str) -> Callable[[float], float]:
    """A typed equation's right-hand side as a function of x on floats (the second route; SymPy parses the text)."""
    src = re.sub(r"\be\^", "E^", text).replace("^", "**")
    expr = parse_expr(src, local_dict={"x": _X, "pi": sympy.pi, "E": sympy.E}, transformations=_TRANSFORMS)
    f = sympy.lambdify(_X, expr, modules=["math"])
    return f


def value_at(p: Params, fam: Family, u: float) -> float:
    """a F(b u + c) + d on floats; for trig, u is in units of pi and c is a multiple of pi."""
    a, b, c, d = (float(v) for v in p)
    arg = b * u + c
    return a * fam.fn(math.pi * arg if fam.trig else arg) + d


def functions_match(fam: Family, p: Params, text: str, xs) -> bool:
    """Whether the typed text and the parameters give the same values at every sample (floats)."""
    g = typed_function(text)
    for u in xs:
        try:
            want = value_at(p, fam, u)
            got = g(math.pi * u if fam.trig else u)
        except (ZeroDivisionError, OverflowError, ValueError):
            continue
        if not (math.isfinite(want) and math.isfinite(got)):
            continue
        if abs(want - got) > 1e-6 * (1 + abs(want)):
            return False
    return True


def same_function(t1: str, t2: str, xs=tuple(0.37 * i - 2.1 for i in range(12))) -> bool:
    """Whether two typed equations agree at a spread of x values (undefined points skipped)."""
    f, g = typed_function(t1), typed_function(t2)
    seen = 0
    for u in xs:
        try:
            a, b = f(u), g(u)
        except (ZeroDivisionError, OverflowError, ValueError):
            continue
        if not (math.isfinite(a) and math.isfinite(b)):
            continue
        seen += 1
        if abs(a - b) > 1e-7 * (1 + abs(a)):
            return False
    check(seen >= 4, "two equations compared at too few points")
    return True


def check_by_points(fam: Family, ops, text: str) -> None:
    """The geometric route: move points of the base graph by the transformations and require each to lie on the
    typed equation."""
    g = typed_function(text)
    points = 0
    xs = [Frac(k, 4) for k in range(-9, 10)] if not fam.trig else [Frac(k, 6) for k in range(-12, 13)]
    for u in xs:
        arg = math.pi * float(u) if fam.trig else float(u)
        try:
            y = fam.fn(arg)
        except (ZeroDivisionError, OverflowError, ValueError):
            continue
        if not math.isfinite(y) or abs(y) > 1e6:
            continue
        if fam.key == "tan(x)" and abs(math.cos(arg)) < 1e-3:
            continue
        x1, y1 = move_point((arg, y), ops, math.pi if fam.trig else 1.0)
        try:
            got = g(x1)
        except (ZeroDivisionError, OverflowError, ValueError):
            continue
        check(abs(got - y1) <= 1e-6 * (1 + abs(y1)), f"the point {(x1, y1)} is not on {text}")
        points += 1
    check(points >= 8, "too few points checked")


# ---------------------------------------------------------------------------------------------- wrong chains

#: (misconception id, how the chain is wrong). Each builds one wrong chain from the true one.


def _flip_first(ops, kinds):
    out, done = [], False
    for op in ops:
        if not done and op[0] in kinds:
            out.append((op[0], -op[1]))
            done = True
        else:
            out.append(op)
    return out if done else None


def wrong_chains(ops) -> list[tuple[str, list]]:
    """(id, chain) pairs: the transformations done with one common slip, most likely slip first."""
    ops = list(ops)
    out: list[tuple[str, list]] = []
    flipped = _flip_first(ops, ("tx",))
    if flipped:
        out.append(("gt-horizontal-shift-sign", flipped))
    flipped = _flip_first(ops, ("ty",))
    if flipped:
        out.append(("gt-vertical-shift-sign", flipped))
    for i in range(len(ops) - 1):
        swapped = ops[:i] + [ops[i + 1], ops[i]] + ops[i + 2 :]
        out.append(("gt-order-swapped", swapped))
    inverted = []
    done = False
    for op in ops:
        if not done and op[0] in ("dx", "dy"):
            inverted.append((op[0], 1 / op[1]))
            done = True
        else:
            inverted.append(op)
    if done:
        out.append(("gt-dilation-factor-inverted", inverted))
    axis = []
    done = False
    for op in ops:
        if not done and op[0] in ("rx", "ry"):
            axis.append(("ry",) if op[0] == "rx" else ("rx",))
            done = True
        else:
            axis.append(op)
    if done:
        out.append(("gt-reflection-wrong-axis", axis))
    return out


__all__ = [
    "FAMILIES",
    "Family",
    "TooFewDistractors",
    "apply_op",
    "arg_text",
    "check_by_points",
    "compose",
    "equation_text",
    "normalise",
    "frac_text",
    "functions_match",
    "move_point",
    "op_phrase",
    "ops_sentence",
    "same_function",
    "shift_of",
    "typed_function",
    "value_at",
    "wrong_chains",
]
