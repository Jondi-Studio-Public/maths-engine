"""Generates `gt_inverse_trig` (easy, medium, hard) and the exam chain `gt_inverse_trig_exam`: exact values, domain,
range and graph features of arcsin, arccos and arctan, typed in radians in terms of pi.

Skill ``inverse-trig-values-and-graph-features`` (``trigonometry``, answer ``expression``, which also grades the fractions
and surds that composites give).

* easy: the exact value of ``arcsin``, ``arccos`` or ``arctan`` at a standard value, with the argument negative about
  half the time. The learner must give the angle in the principal range (``-pi/2`` to ``pi/2`` for arcsin and arctan,
  ``0`` to ``pi`` for arccos), not another angle with the same sine, cosine or tangent.
* medium: a feature of ``y = a arcsin(bx + d) + c``, ``y = a arccos(bx) + c`` or ``y = a arctan(bx) + c``: the greatest
  value, the upper horizontal asymptote, the largest x in the domain, or the exact value of y at a given x.
* hard: a composite. ``sin(arccos(u))``, ``cos(arcsin(u))``, ``tan(arcsin(u))`` or ``tan(arccos(u))`` for a Pythagorean
  ``u`` (a fraction), ``sin(2 arcsin(u))``, or ``arccos(cos(t))``, ``arcsin(sin(t))``, ``arctan(tan(t))`` for an angle ``t``
  outside the principal range (the answer is the principal value, not ``t``).
* exam (``gt_inverse_trig_exam``): for ``f(x) = a arcsin(bx) + c`` with c chosen so f has a nice zero, (a) the largest x in
  the domain, (b) the greatest value of f, (c) the exact value of f at a given x, and (d) the x-intercept, which follows
  through from (a): it is a fixed fraction of the largest x.

Every angle is built as an exact multiple of pi from a table of standard values, then checked against the library's
``math`` functions and against SymPy's exact evaluation.
"""

from __future__ import annotations

import math
from fractions import Fraction as Frac

import numpy as np
import sympy as sp

from ..answers import Distractor, exact_value
from ..core import attempts, check, generator, phrase, rational_text
from ._spread_kit import NEAR, pick, uses_rng
from ._sr2b_kit import int_dist, number_fields, pi_latex, pi_shown, pi_text, retry_short, text_dist, value_latex

# the argument, as typed, and its exact value; the principal values (in units of pi) are checked below
_S2, _S3 = sp.sqrt(2), sp.sqrt(3)
_SIN = [  # (typed argument, exact value, arcsin in units of pi) for arguments 0 to 1
    ("0", sp.Integer(0), Frac(0)),
    ("1/2", sp.Rational(1, 2), Frac(1, 6)),
    ("sqrt(2)/2", _S2 / 2, Frac(1, 4)),
    ("sqrt(3)/2", _S3 / 2, Frac(1, 3)),
    ("1", sp.Integer(1), Frac(1, 2)),
]
_TAN = [  # (typed argument, exact value, arctan in units of pi)
    ("0", sp.Integer(0), Frac(0)),
    ("sqrt(3)/3", _S3 / 3, Frac(1, 6)),
    ("1", sp.Integer(1), Frac(1, 4)),
    ("sqrt(3)", _S3, Frac(1, 3)),
]


def _flt(expr) -> float:
    return float(sp.N(expr, 30))


def _arcsin(u: float) -> Frac:
    return Frac(math.asin(u) / math.pi).limit_denominator(24)


def _arccos(u: float) -> Frac:
    return Frac(math.acos(u) / math.pi).limit_denominator(24)


def _arctan(u: float) -> Frac:
    return Frac(math.atan(u) / math.pi).limit_denominator(24)


def _check_pi(value: Frac, exact: sp.Expr) -> None:
    check(abs(float(value) - _flt(exact / sp.pi)) < 1e-12, "the exact value differs from the table")


def _pt(coeff) -> str:
    """A multiple of pi as typed: 5pi/6."""
    return pi_text(Frac(coeff))


def _dist(coeff, tag) -> Distractor:
    t = _pt(coeff)
    return Distractor(t, tag, t)


def _expr_fields(coeff: Frac) -> dict:
    t = _pt(coeff)
    return dict(answer_kind="expression", answer=t, answer_text=t, answer_latex=pi_latex(coeff))


def _typed(sign: int, arg: str) -> str:
    """-sqrt(3)/2 style: the argument with its sign."""
    return f"-{arg}" if sign < 0 and arg != "0" else arg


def _stem_pi(coeff: Frac) -> str:
    return pi_shown(coeff)


# ------------------------------------------------------------------------------------------------------- easy


def _easy(rng):
    kind = rng.choice(("arcsin", "arccos", "arctan"))
    sign = rng.choice((-1, 1, -1))
    if kind == "arctan":
        arg, exact, base = rng.choice(_TAN[1:])
        value = sign * base
        u = sign * _flt(exact)
        check(_arctan(u) == value, "arctan value differs")
        _check_pi(value, sign * sp.atan(exact))
        wrong = [
            (-value, "gti-sign-dropped"),
            (1 - base if sign < 0 else 1 - value, "gti-second-quadrant-angle"),
            (2 - base if sign < 0 else value + 1, "gti-angle-outside-range"),
        ]
        fname = "arctan"
    else:
        arg, exact, base = rng.choice(_SIN if kind == "arcsin" else _SIN[:4])
        if arg == "0":
            sign = 1
        u = sign * _flt(exact)
        if kind == "arcsin":
            value = sign * base
            check(_arcsin(u) == value, "arcsin value differs")
            _check_pi(value, sign * sp.asin(exact))
            wrong = [
                (-value, "gti-sign-dropped"),
                (Frac(2) + value if sign < 0 else 1 - value, "gti-angle-outside-range"),
                (Frac(1, 2) - base, "gti-arcsin-arccos-swapped"),
                (1 - value if sign < 0 else 1 + value, "gti-second-quadrant-angle"),
            ]
        else:
            value = Frac(1, 2) - base if sign > 0 else Frac(1, 2) + base
            check(_arccos(u) == value, "arccos value differs")
            _check_pi(value, sp.acos(sign * exact))
            wrong = [
                (-(Frac(1, 2) - base) if sign < 0 else Frac(1, 2) - base, "gti-negative-angle-for-arccos"),
                (
                    Frac(1, 2) - base if sign < 0 else base,
                    "gti-sign-dropped" if sign < 0 else "gti-arcsin-arccos-swapped",
                ),
                (Frac(1, 2) + base + 1 if sign < 0 else Frac(1, 2) + base, "gti-angle-outside-range"),
                (base, "gti-arcsin-arccos-swapped"),
            ]
        fname = kind
    # a half-turn or a whole turn either way: an angle outside the range with the right trig value, on each side of the key
    wrong += [(value + d, "gti-angle-outside-range") for d in (1, -1, 2, -2)]
    cands = [_dist(v, t) for v, t in wrong if v != value]
    ans = _pt(value)
    argtext = _typed(sign, arg)
    stem = phrase(
        rng,
        (
            "Find the exact value of {f}({a}). Give your answer in radians, in terms of pi.",
            "What is {f}({a})? Give an exact value in radians.",
            "State the exact value of {f}({a}) in radians.",
        ),
        f=fname,
        a=argtext,
    )
    lo, hi = ("0", "π") if kind == "arccos" else ("-π/2", "π/2")
    return dict(
        text=stem,
        # the options are angles in units of pi: arccos lives in [0, 1], arcsin and arctan in (-1/2, 1/2); the named
        # slips are the very mistakes of leaving the range (outside it, or the wrong branch), so they may stray further
        distractors=pick(
            ans,
            cands,
            lo=0 if kind == "arccos" else -Frac(1, 2) if kind == "arcsin" else -Frac(5, 12),  # arcsin reaches +-pi/2
            hi=1 if kind == "arccos" else Frac(1, 2) if kind == "arcsin" else Frac(5, 12),
            slip_range=(-2, 3) if kind == "arccos" else (-3, 3),
        ),
        working=[
            f"{fname}(u) is the angle between {lo} and {hi} (inclusive) whose {'sine' if kind == 'arcsin' else 'cosine' if kind == 'arccos' else 'tangent'} is u.",
            f"{argtext} = {'sin' if kind == 'arcsin' else 'cos' if kind == 'arccos' else 'tan'}({_stem_pi(value)}), and {_stem_pi(value)} is in that range.",
            f"So {fname}({argtext}) = {_stem_pi(value)}.",
        ],
        **_expr_fields(value),
    )


# ----------------------------------------------------------------------------------------------------- medium


def _lead(a: int, name: str, arg: str) -> str:
    return f"{'' if a == 1 else '-' if a == -1 else a} {name}({arg})".strip() if abs(a) == 1 else f"{a} {name}({arg})"


def _shift_text(c: Frac) -> str:
    """The constant as written after the function: - π/2, + π/3 (nothing for zero)."""
    if c == 0:
        return ""
    return f" {'+' if c > 0 else '-'} {pi_shown(abs(c))}"


def _x_text(b: int) -> str:
    return "x" if b == 1 else f"{b}x"


def _medium_greatest(rng):
    kind = rng.choice(("arcsin", "arccos"))
    a = rng.choice((2, 3, 4, -2, -3) if kind == "arcsin" else (2, 3, 4))
    b = rng.choice((2, 3, 4))
    c = rng.choice([Frac(v) for v in (-1, Frac(-1, 2), Frac(1, 2), Frac(1, 3), Frac(1, 4), Frac(-1, 3), 1)])
    half = Frac(1, 2) if kind == "arcsin" else Frac(1)
    top = abs(a) * half + c
    # the second route: scan the function on its domain
    n = 20001
    ys = [
        a * (math.asin(t) if kind == "arcsin" else math.acos(t)) + float(c) * math.pi
        for t in (-1 + 2 * i / (n - 1) for i in range(n))
    ]
    check(abs(max(ys) / math.pi - float(top)) < 1e-6, "greatest value differs from the scan")
    wrong = [
        (abs(a) * half, "gti-range-ignores-shift"),
        (
            abs(a) * Frac(1) + c if kind == "arcsin" else abs(a) * Frac(1, 2) + c,
            "gti-arcsin-range-and-arccos-range-swapped",
        ),
        (c - abs(a) * half if kind == "arcsin" else c, "gti-least-for-greatest"),
        (abs(a) * half - c, "gti-shift-sign"),
    ]
    cands = [_dist(v, t) for v, t in wrong if v != top]
    ans = _pt(top)
    fn = f"{_lead(a, kind, _x_text(b))}{_shift_text(c)}"
    stem = phrase(
        rng,
        (
            "Find the greatest value of y = {f}. Give an exact value in terms of pi.",
            "The graph of y = {f} has a greatest value. State it exactly, in terms of pi.",
        ),
        f=fn,
    )
    span = "-π/2 to π/2" if kind == "arcsin" else "0 to π"
    return dict(
        text=stem,
        distractors=pick(ans, cands, lo=c - abs(a), hi=c + 2 * abs(a), slip_range=(-abs(a) - 2, abs(a) + 2)),
        working=[
            f"{kind} takes values from {span}, so {_lead(a, kind, _x_text(b))} has greatest value {_stem_pi(abs(a) * half)}.",
            f"Adding the shift gives the greatest value {_stem_pi(top)}.",
        ],
        **_expr_fields(top),
    )


def _medium_asymptote(rng):
    a = rng.choice((2, 3, 4, -2, -3))
    b = rng.choice((1, 2, 3))
    c = rng.choice([Frac(v) for v in (-1, Frac(-1, 2), Frac(1, 2), Frac(1, 3), Frac(1, 4), Frac(-1, 3), 1)])
    top = abs(a) * Frac(1, 2) + c
    check(abs(abs(a) * math.atan(1e9) + float(c) * math.pi - float(top) * math.pi) < 1e-6, "asymptote differs")
    wrong = [
        (abs(a), "gti-asymptote-uses-pi"),
        (abs(a) * Frac(1, 2), "gti-range-ignores-shift"),
        (c - abs(a) * Frac(1, 2), "gti-least-for-greatest"),
        (abs(a) * Frac(1, 2) - c, "gti-shift-sign"),
    ]
    cands = [_dist(v, t) for v, t in wrong if v != top]
    ans = _pt(top)
    fn = f"{_lead(a, 'arctan', _x_text(b))}{_shift_text(c)}"
    stem = phrase(
        rng,
        (
            "The graph of y = {f} has two horizontal asymptotes. Find the equation of the upper one, y = k, by giving k exactly in terms of pi.",
            "Give the exact value of k in the equation y = k of the upper horizontal asymptote of the graph of y = {f}.",
        ),
        f=fn,
    )
    return dict(
        text=stem,
        distractors=pick(ans, cands, lo=c - abs(a), hi=c + abs(a), slip_range=(-abs(a) - 2, abs(a) + 2)),
        working=[
            "As x gets very large, arctan approaches π/2 from below and, as x gets very large and negative, approaches -π/2.",
            f"{_lead(a, 'arctan', _x_text(b))} therefore lies between {_stem_pi(-abs(a) * Frac(1, 2))} and {_stem_pi(abs(a) * Frac(1, 2))}, and the shift gives the upper asymptote y = {_stem_pi(top)}.",
        ],
        **_expr_fields(top),
    )


def _medium_domain(rng):
    for _ in attempts(500, "an argument bx + d with a whole-number shift"):
        a = rng.choice((2, 3, -2, -3))
        b = rng.choice((2, 3, 4, 5))
        d = rng.choice((-3, -2, -1, 1, 2, 3))
        c = rng.choice([Frac(v) for v in (-1, Frac(-1, 2), Frac(1, 2), Frac(1, 3), 1)])
        top = Frac(1 - d, b)
        if top != 0:
            break
    check(Frac(-1 - d, b) < top, "the largest x is not the right-hand end")
    for t in (top, Frac(-1 - d, b)):
        check(abs(b * float(t) + d) <= 1 + 1e-12, "an end point is outside the domain")
    check(b * (float(top) + 1e-6) + d > 1, "a larger x is still in the domain")
    wrong = [
        (Frac(1 + d, b), "gti-shift-sign"),
        (Frac(1, b), "gti-domain-ignores-shift"),
        (Frac(-1 - d, b), "gti-least-for-greatest"),
        (Frac(b * (1 - d)), "gti-multiplies-by-b"),
    ]
    cands = []
    for v, t in wrong:
        if v != top:
            txt = rational_text(v)
            cands.append(Distractor(txt, t, txt))
    ans = rational_text(top)
    fn = f"{a} arcsin({b}x {'+' if d > 0 else '-'} {abs(d)}){_shift_text(c)}"
    stem = phrase(
        rng,
        (
            "The function f is given by f(x) = {f}. Find the largest value of x in the domain of f.",
            "For f(x) = {f}, state the largest value of x for which f is defined.",
        ),
        f=fn,
    )
    near = [
        Distractor(rational_text(top + Frac(k, b)), NEAR, rational_text(top + Frac(k, b))) for k in (-2, -1, 1, 2)
    ]  # a slip of one or two steps of 1/b either way
    return dict(
        text=stem,
        distractors=pick(ans, cands, fillers=near, lo=None, hi=None, unbounded="open"),
        working=[
            "arcsin(u) is defined only for -1 ≤ u ≤ 1.",
            f"So -1 ≤ {b}x {'+' if d > 0 else '-'} {abs(d)} ≤ 1, which gives {rational_text(Frac(-1 - d, b))} ≤ x ≤ {ans}.",
            f"The largest value of x is {ans}.",
        ],
        answer_kind="expression",
        answer=ans,
        answer_text=ans,
        answer_latex=value_latex(top),
    )


def _medium_value(rng):
    for _ in attempts(200, "a value at a standard argument"):
        a = rng.choice((2, 3, 4, -2, -3))
        b = rng.choice((2, 3, 4))
        c = rng.choice([Frac(v) for v in (-1, Frac(-1, 2), Frac(1, 2), Frac(1, 3), Frac(1, 4), 1)])
        u = rng.choice((Frac(1, 2), Frac(-1, 2), Frac(1), Frac(-1)))
        base = _arcsin(float(u))
        value = a * base + c
        if value != 0:
            break
    x0 = u / b
    check(abs(a * math.asin(b * float(x0)) + float(c) * math.pi - float(value) * math.pi) < 1e-9, "value differs")
    wrong = [
        (a * _arccos(float(u)) + c, "gti-arcsin-arccos-swapped"),
        (a * abs(base) + c, "gti-sign-dropped"),
        (a * base, "gti-forgets-the-shift"),
        (a * base - c, "gti-shift-sign"),
    ]
    cands = [_dist(v, t) for v, t in wrong if v != value]
    ans = _pt(value)
    fn = f"{a} arcsin({b}x){_shift_text(c)}"
    stem = phrase(
        rng,
        (
            "Let f(x) = {f}. Find the exact value of f({x0}). Give your answer in terms of pi.",
            "For f(x) = {f}, find f({x0}) exactly, in terms of pi.",
        ),
        f=fn,
        x0=rational_text(x0),
    )
    return dict(
        text=stem,
        distractors=pick(ans, cands, lo=-abs(a) - 2, hi=abs(a) + 2, slip_range=(-abs(a) - 3, abs(a) + 3)),
        working=[
            f"{b} × {rational_text(x0)} = {rational_text(u)}, so f({rational_text(x0)}) = {a} arcsin({rational_text(u)}){_shift_text(c)}.",
            f"arcsin({rational_text(u)}) = {_stem_pi(base)}, so f = {a} × {_stem_pi(base)}{_shift_text(c)} = {_stem_pi(value)}.",
        ],
        **_expr_fields(value),
    )


# ------------------------------------------------------------------------------------------------------- hard

_TRIPLES = [(3, 4, 5), (5, 12, 13), (8, 15, 17), (7, 24, 25), (20, 21, 29)]


def _frac_dist(v: Frac, tag: str) -> Distractor:
    t = rational_text(v)
    return Distractor(t, tag, t)


def _near_fracs(v: Frac, q: int) -> list[Distractor]:
    """Plain slips a step either side of a ratio: one or two steps of 1/denominator (of 1/q when that is very fine)."""
    d = v.denominator if v.denominator <= 40 else q
    out = []
    for k in (-3, -2, -1, 1, 2, 3):
        w = v + Frac(k, d)
        if w != 0 and abs(w) <= 3:
            out.append(_frac_dist(w, NEAR))
    return out


def _frac_fields(v: Frac) -> dict:
    t = rational_text(v)
    return dict(answer_kind="expression", answer=t, answer_text=t, answer_latex=value_latex(v))


def _hard_ratio(rng):
    p, r, q = rng.choice(_TRIPLES)
    if rng.random() < 0.5:
        p, r = r, p
    sign = rng.choice((-1, 1))
    u = Frac(sign * p, q)  # argument; the other leg is r
    form = rng.choice(("sin_arccos", "cos_arcsin", "tan_arcsin", "tan_arccos", "sin_double"))
    uf = float(u)
    if form == "sin_arccos":
        value, stem_fn = Frac(r, q), f"sin(arccos({rational_text(u)}))"
        check(abs(math.sin(math.acos(uf)) - float(value)) < 1e-12, "sin(arccos) differs")
        wrong = [
            (u, "gti-returns-the-argument"),
            (-value, "gti-sign-of-root"),
            (Frac(q, r), "gti-reciprocal"),
            (Frac(r, p), "gti-wrong-ratio"),
        ]
        why = f"Let θ = arccos({rational_text(u)}), an angle from 0 to π, so sin θ is not negative. sin θ = √(1 - ({rational_text(u)})²) = {rational_text(value)}."
    elif form == "cos_arcsin":
        value, stem_fn = Frac(r, q), f"cos(arcsin({rational_text(u)}))"
        check(abs(math.cos(math.asin(uf)) - float(value)) < 1e-12, "cos(arcsin) differs")
        wrong = [
            (u, "gti-returns-the-argument"),
            (-value, "gti-sign-of-root"),
            (Frac(q, r), "gti-reciprocal"),
            (Frac(r, p), "gti-wrong-ratio"),
        ]
        why = f"Let θ = arcsin({rational_text(u)}), an angle from -π/2 to π/2, so cos θ is not negative. cos θ = √(1 - ({rational_text(u)})²) = {rational_text(value)}."
    elif form == "tan_arcsin":
        value, stem_fn = Frac(sign * p, r), f"tan(arcsin({rational_text(u)}))"
        check(abs(math.tan(math.asin(uf)) - float(value)) < 1e-12, "tan(arcsin) differs")
        wrong = [
            (u, "gti-returns-the-argument"),
            (-value, "gti-sign-of-root"),
            (1 / value, "gti-reciprocal"),
            (Frac(r, q), "gti-wrong-ratio"),
        ]
        why = f"Let θ = arcsin({rational_text(u)}), so sin θ = {rational_text(u)} and cos θ = {rational_text(Frac(r, q))} (positive on the range). tan θ = sin θ ÷ cos θ = {rational_text(value)}."
    elif form == "tan_arccos":
        value, stem_fn = Frac(r, sign * p), f"tan(arccos({rational_text(u)}))"
        check(abs(math.tan(math.acos(uf)) - float(value)) < 1e-12, "tan(arccos) differs")
        wrong = [
            (u, "gti-returns-the-argument"),
            (-value, "gti-sign-of-root"),
            (1 / value, "gti-reciprocal"),
            (Frac(r, q), "gti-wrong-ratio"),
        ]
        why = f"Let θ = arccos({rational_text(u)}), so cos θ = {rational_text(u)} and sin θ = {rational_text(Frac(r, q))} (positive on the range). tan θ = sin θ ÷ cos θ = {rational_text(value)}."
    else:
        value = Frac(2 * sign * p * r, q * q)
        stem_fn = f"sin(2 arcsin({rational_text(u)}))"
        check(abs(math.sin(2 * math.asin(uf)) - float(value)) < 1e-12, "sin(2 arcsin) differs")
        wrong = [
            (2 * u, "gti-doubles-the-sine-only"),
            (Frac(sign * p * r, q * q), "gti-forgets-the-factor-two"),
            (-value, "gti-sign-of-root"),
            (Frac(2 * p * r, q * q) if sign < 0 else Frac(2 * sign * p, q), "gti-doubles-the-sine-only"),
        ]
        why = f"Let θ = arcsin({rational_text(u)}): sin θ = {rational_text(u)} and cos θ = {rational_text(Frac(r, q))}. sin 2θ = 2 sin θ cos θ = {rational_text(value)}."
    cands = [_frac_dist(v, t) for v, t in wrong if v != value and v.denominator != 0]
    ans = rational_text(value)
    stem = phrase(
        rng,
        (
            "Find the exact value of {e}.",
            "Evaluate {e} exactly, without a calculator.",
        ),
        e=stem_fn,
    )
    return dict(
        text=stem,
        distractors=pick(
            ans,
            cands,
            fillers=_near_fracs(value, q),
            lo=None,
            hi=None,
            unbounded="open",
        ),
        working=[why, f"So {stem_fn} = {ans}."],
        **_frac_fields(value),
    )


_ANGLES = [Frac(n, d) for d in (3, 4, 6) for n in range(-11, 12) if n != 0 and math.gcd(abs(n), d) == 1]


def _principal(kind: str, t: Frac) -> Frac:
    """The principal value (units of pi) of arccos(cos t), arcsin(sin t) or arctan(tan t) from the library's math."""
    x = float(t) * math.pi
    return {
        "arccos": lambda: _arccos(math.cos(x)),
        "arcsin": lambda: _arcsin(math.sin(x)),
        "arctan": lambda: _arctan(math.tan(x)),
    }[kind]()


def _hard_composite(rng):
    for _ in attempts(500, "an angle outside the principal range"):
        kind = rng.choice(("arccos", "arcsin", "arctan"))
        t = rng.choice(_ANGLES)
        if kind == "arccos":
            outside = not (0 <= t <= 1)
        elif kind == "arcsin":
            outside = not (-Frac(1, 2) <= t <= Frac(1, 2))
        else:
            outside = not (-Frac(1, 2) < t < Frac(1, 2)) and (t * 2).denominator != 1
        if outside and abs(t) <= 2 and (kind != "arctan" or (t * 2).denominator != 1):
            value = _principal(kind, t)
            if value != t:
                break
    inner = {"arccos": "cos", "arcsin": "sin", "arctan": "tan"}[kind]
    # the second route: SymPy's exact evaluation of the composite
    exact = {"arccos": sp.acos, "arcsin": sp.asin, "arctan": sp.atan}[kind](
        {"cos": sp.cos, "sin": sp.sin, "tan": sp.tan}[inner](sp.pi * sp.Rational(t.numerator, t.denominator))
    )
    check(
        sp.simplify(exact - sp.pi * sp.Rational(value.numerator, value.denominator)) == 0,
        "SymPy disagrees with the principal value",
    )
    reduced = t - 2 * math.floor(t / 2) if kind != "arctan" else t - math.floor(t)
    wrong = [
        (t, "gti-composite-cancels"),
        (-t, "gti-negative-angle"),
        (t - 2 * math.floor(t / 2) if kind != "arctan" else t - math.floor(t), "gti-angle-reduced-not-principal"),
        (1 - value, "gti-second-quadrant-angle"),
        (value + 1 if kind != "arctan" else value - 1, "gti-angle-outside-range"),
    ]
    cands = [_dist(v, tag) for v, tag in wrong if v != value]
    ans = _pt(value)
    e = f"{kind}({inner}({_pt(t)}))"
    lo, hi = ("0", "π") if kind == "arccos" else ("-π/2", "π/2")
    stem = phrase(
        rng,
        (
            "Find the exact value of {e}. Give your answer in radians, in terms of pi.",
            "Evaluate {e} exactly, in radians.",
        ),
        e=e,
    )
    return dict(
        text=stem,
        distractors=pick(ans, cands, lo=-2, hi=2, slip_range=(-3, 3)),
        working=[
            f"{kind}(u) gives the angle between {lo} and {hi} whose {inner} is u, so the answer is not just {_stem_pi(t)} when {_stem_pi(t)} is outside that range.",
            f"{inner}({_stem_pi(t)}) = {inner}({_stem_pi(value)}), and {_stem_pi(value)} is in the range, so {e} = {_stem_pi(value)}.",
        ],
        **_expr_fields(value),
    )


@generator(
    "gt_inverse_trig",
    level=15,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@uses_rng
@retry_short
def gt_inverse_trig(rng, tier="easy"):
    """An exact value or graph feature of an inverse trigonometric function; the tiers are in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return rng.choice([_medium_greatest, _medium_asymptote, _medium_domain, _medium_value])(rng)
    if tier != "hard":
        raise ValueError(f"unknown tier {tier!r}")
    return rng.choice([_hard_ratio, _hard_composite])(rng)


# -------------------------------------------------------------------------------------------------------- exam


@generator("gt_inverse_trig_exam", level=16, presets={"exam": dict(tier="exam")})
@uses_rng
@retry_short
def gt_inverse_trig_exam(rng, tier="exam"):
    """Exam chain on f(x) = a arcsin(bx) + c: largest x in the domain, greatest value, f at a given x, the x-intercept."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    a = rng.choice((2, 3, 4, 5, 6, 8))
    b = rng.choice((2, 3, 4, 5))
    # f has its zero where b x = s. Not s = 1: then the greatest value is 0 and the zero is the largest x, so (b) would
    # always be 0 and (d) would repeat (a).
    s = rng.choice((Frac(1, 2), Frac(-1, 2)))
    c = -a * _arcsin(float(s))  # in units of pi
    check(abs(a * math.asin(float(s)) + float(c) * math.pi) < 1e-12, "f does not vanish at the chosen point")
    u = rng.choice([v for v in (Frac(1, 2), Frac(-1, 2), Frac(1), Frac(-1)) if v != s])
    top_x = Frac(1, b)
    greatest = Frac(a, 2) + c
    f_at = a * _arcsin(float(u)) + c
    x0 = u / b
    zero = s / b
    # second route: scan f over its domain for its greatest value and its zero
    n = 40001
    xs = np.linspace(-1 / b, 1 / b, n)
    ys = a * np.arcsin(np.clip(b * xs, -1.0, 1.0)) + float(c) * math.pi
    check(abs(ys.max() / math.pi - float(greatest)) < 1e-6, "greatest value differs from the scan")
    root = int(np.argmin(np.abs(ys)))
    check(abs(xs[root] - float(zero)) < 1e-4, "zero differs from the scan")
    fn = f"{a} arcsin({_x_text(b)}){_shift_text(c)}"
    # (a)
    a_wrong = [
        Distractor(rational_text(Frac(b)), "gti-multiplies-by-b", exact_value(Frac(b))),
        Distractor("1", "gti-domain-ignores-b", 1),
        Distractor(rational_text(-top_x), "gti-least-for-greatest", exact_value(-top_x)),
        # plain near misses on both sides of the key, so it is not always the middle option
        Distractor(rational_text(2 * top_x), NEAR, exact_value(2 * top_x)),
        Distractor(rational_text(Frac(1, b + 1)), NEAR, exact_value(Frac(1, b + 1))),
        Distractor(rational_text(Frac(1, 2 * b)), NEAR, exact_value(Frac(1, 2 * b))),
    ]
    a_wrong = [d for d in a_wrong if d.text != rational_text(top_x)]
    # (b)
    b_wrong = [
        _dist(Frac(a, 2), "gti-range-ignores-shift"),
        _dist(Frac(a) + c, "gti-arcsin-range-and-arccos-range-swapped"),
        _dist(c - Frac(a, 2), "gti-least-for-greatest"),
        _dist(Frac(a, 2) - c, "gti-shift-sign"),
    ]
    # (c)
    c_wrong = [
        _dist(a * abs(_arcsin(float(u))) + c, "gti-sign-dropped"),
        _dist(a * _arcsin(float(u)), "gti-forgets-the-shift"),
        _dist(a * _arccos(float(u)) + c, "gti-arcsin-arccos-swapped"),
        _dist(a * _arcsin(float(u)) - c, "gti-shift-sign"),
    ]
    # (d)
    d_wrong = [
        Distractor(rational_text(-zero), "gti-sign-of-root", exact_value(-zero)),
        Distractor(rational_text(s), "gti-domain-ignores-b", exact_value(s)),
        Distractor(rational_text(s * b), "gti-multiplies-by-b", exact_value(s * b)),
        Distractor(rational_text(top_x), "gti-least-for-greatest", exact_value(top_x)),
    ]
    d_wrong = [d for d in d_wrong if Frac(d.text) != zero]
    parts = [
        dict(
            prompt="Find the largest value of x in the domain of f.",
            marks=1,
            answer_kind="fraction",
            answer=exact_value(top_x),
            answer_text=rational_text(top_x),
            answer_latex=value_latex(top_x),
            working=[
                f"arcsin needs -1 ≤ {_x_text(b)} ≤ 1, so -{rational_text(top_x)} ≤ x ≤ {rational_text(top_x)}.",
                f"The largest x is {rational_text(top_x)}.",
            ],
            distractors=pick(
                rational_text(top_x),
                a_wrong,
                lo=None,
                hi=None,
                unbounded="open",
            ),
        ),
        dict(
            prompt="Find the greatest value of f. Give an exact value in terms of pi.",
            marks=2,
            working=[
                f"arcsin takes values up to π/2, so {a} arcsin({_x_text(b)}) has greatest value {_stem_pi(Frac(a, 2))}.",
                f"With the shift, the greatest value of f is {_stem_pi(greatest)}.",
            ],
            distractors=pick(
                _pt(greatest),
                [d for d in b_wrong if d.text != _pt(greatest)],
                lo=c - abs(a),
                hi=c + 2 * abs(a),  # arccos reaches |a| above the shift: leave room above it too
                slip_range=(-abs(a) - 2, abs(a) + 2),
            ),
            **_expr_fields(greatest),
        ),
        dict(
            prompt=f"Find the exact value of f({rational_text(x0)}). Give your answer in terms of pi.",
            marks=2,
            working=[
                f"{b} × {rational_text(x0)} = {rational_text(u)}, and arcsin({rational_text(u)}) = {_stem_pi(_arcsin(float(u)))}.",
                f"f({rational_text(x0)}) = {a} × {_stem_pi(_arcsin(float(u)))}{_shift_text(c)} = {_stem_pi(f_at)}.",
            ],
            distractors=pick(
                _pt(f_at),
                [d for d in c_wrong if d.text != _pt(f_at)],
                lo=-abs(a) - 2,
                hi=abs(a) + 2,
                slip_range=(-abs(a) - 3, abs(a) + 3),
            ),
            **_expr_fields(f_at),
        ),
        dict(
            prompt="Find the x-intercept of the graph of f.",
            marks=2,
            answer_kind="fraction",
            answer=exact_value(zero),
            answer_text=rational_text(zero),
            answer_latex=value_latex(zero),
            follow_through=f"({rational_text(s)})*part_a",
            working=[
                f"f(x) = 0 when arcsin({_x_text(b)}) = {_stem_pi(_arcsin(float(s)))}, that is {_x_text(b)} = {rational_text(s)}.",
                f"So x = {rational_text(zero)}, which is {rational_text(s)} times the largest x.",
            ],
            distractors=pick(
                rational_text(zero),
                d_wrong,
                lo=-top_x,
                hi=top_x,
                slip_range=(-b * top_x - 1, b * top_x + 1),
                fillers=[
                    Distractor(rational_text(v), NEAR, exact_value(v))
                    # plain slips stay inside the domain, where an x-intercept has to be
                    for v in (zero / 2, -top_x, 3 * zero / 2, -zero / 2, -3 * zero / 2)
                    if v != zero and abs(v) <= top_x
                ],
            ),
        ),
    ]
    stem = phrase(
        rng,
        ("The function f is given by f(x) = {f}.", "Consider the function f(x) = {f}."),
        f=fn,
    )
    return dict(text=stem, parts=parts)
