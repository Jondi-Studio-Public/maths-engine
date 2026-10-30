"""Generates `w5b_pde_discriminant` and `exd_x_w5b_pde_discriminant`: the discriminant of a second-order PDE as a number.

One second-year skill, ``pde-discriminant-values-and-canonical-form`` (domain ``fourier-analysis-and-pdes``). Its
neighbour ``classify-constant-coefficient-pde`` is a three-way choice (hyperbolic / parabolic / elliptic); this skill
asks for the VALUES behind the classification, typed rather than chosen: the discriminant itself, the parameter value
at which the type changes, how many integer parameter values give one type (or the smallest or largest such integer),
a characteristic slope, and the coefficient of the mixed term after the standard change of variables.

The equation is  A u_xx + B u_xy + C u_yy + (lower-order terms) = rhs,  written in one of several surface forms
(``u_xx`` or ``∂²u/∂x²`` notation, terms in different orders, the whole equation multiplied by -1, a right-hand side
that is not 0). The lower-order terms and the right-hand side are there to be ignored. Where a convention could be
ambiguous the stem spells it out ("B is the coefficient of u_xy", "the mixed term is 2b u_xy", "the characteristics
satisfy A y'^2 - B y' + C = 0").

Tiers:

- ``easy``: the value of B² - 4AC (or b² - AC when the mixed term is written 2b u_xy). Integer answers.
- ``medium``: a parameter k sits in one or two coefficients. Either the positive value of k at which the type changes
  (a terminating decimal; for A = k, C = k + c0 it comes from a Pythagorean pair), or the COUNT of integers in a stated
  range for which the equation is hyperbolic (or elliptic), or the smallest or largest such integer, where a parabolic
  boundary value may fall on an integer.
- ``hard``: a hyperbolic equation built backwards from two rational characteristic slopes s1, s2 (so
  B = A(s1 + s2), C = A s1 s2): the larger or the smaller of the two slopes, with the coefficients sometimes linear in a parameter k at a
  stated value, or the coefficient Q = -A (s1 - s2)² of w_ξη after ξ = y - s1 x, η = y - s2 x (an exact fraction).
- ``exam`` (topic ``exd_x_w5b_pde_discriminant``): one equation with B = p k, four typed parts: the discriminant at a
  given k, how many integers in -N..N make the equation elliptic, how many make it hyperbolic, and the set of
  characteristic slopes at a hyperbolic k whose discriminant is a perfect square.

Second routes inside `make()`: the type counts are made by evaluating the discriminant at every integer, the slopes
are checked in the characteristic equation, and the canonical coefficient is checked by the chain rule on
ξ and η, with no use of the formula -D/A.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction as Fr

from ..answers import Distractor, exact_value
from ..core import (
    GRADER_TOLERANCE,
    attempts,
    check,
    exact_decimal_string,
    generator,
    latex_answer,
    numeric_answer,
    pick_distractors,
    rational_text,
    round_half_up,
    stated_precision_string,
)
from ..grading.typed import set_answer_text
from ._spread_kit import _RNG, uses_rng
from ._w5b_pde_cp_balance_kit import fair

TOPIC = "w5b_pde_discriminant"
EXAM_TOPIC = "exd_x_w5b_pde_discriminant"

FRACTION_FORMAT = " Give your answer as a fraction in lowest terms or a whole number."
SET_FORMAT = " Give each slope as a fraction in lowest terms or a whole number."

Lin = tuple  # (c0, c1): c0 + c1 k

_RHS = ["0", "0", "0", "0", "x + y", "sin(x)", "e^y", "x y"]
_SUBSCRIPT = {"xx": "u_xx", "xy": "u_xy", "yy": "u_yy", "x": "u_x", "y": "u_y", "u": "u"}
_PARTIAL = {"xx": "∂²u/∂x²", "xy": "∂²u/∂x∂y", "yy": "∂²u/∂y²", "x": "∂u/∂x", "y": "∂u/∂y", "u": "u"}
_ORDERS = [
    ("xx", "xy", "yy", "x", "y", "u"),
    ("xx", "xy", "yy", "x", "y", "u"),
    ("yy", "xy", "xx", "y", "x", "u"),
    ("xy", "xx", "yy", "u", "x", "y"),
    ("x", "xx", "y", "xy", "yy", "u"),
]


def lin_eval(c: Lin, k) -> Fr:
    return Fr(c[0]) + Fr(c[1]) * Fr(k)


def lin_neg(c: Lin) -> Lin:
    return (-c[0], -c[1])


def _lin_body(c: Lin) -> tuple[bool, str]:
    """(negative, text without the sign) for c0 + c1 k; a bracketed sum is never negative."""
    c0, c1 = c
    if c1 == 0:
        return c0 < 0, str(abs(c0))
    if c0 == 0:
        mag = abs(c1)
        return c1 < 0, ("k" if mag == 1 else f"{mag}k")
    lead = "k" if abs(c1) == 1 else f"{abs(c1)}k"
    lead = ("-" if c1 < 0 else "") + lead
    tail = f" + {c0}" if c0 > 0 else f" - {-c0}"
    return False, f"({lead}{tail})"


def equation(coefs: dict, layout: dict) -> tuple[str, str]:
    """(left-hand side text, right-hand side text) for the stored layout."""
    sym = _PARTIAL if layout["partial"] else _SUBSCRIPT
    parts: list[str] = []
    for name in layout["order"]:
        c = coefs.get(name)
        if c is None or (c[0] == 0 and c[1] == 0):
            continue
        c = lin_neg(c) if layout["negate"] else c
        neg, body = _lin_body(c)
        coef = "" if body == "1" else body + (" " if body.endswith("k") else "")
        term = f"{coef}{sym[name]}"
        if not parts:
            parts.append(("-" if neg else "") + term)
        else:
            parts.append(f"{'-' if neg else '+'} {term}")
    check(parts, "an empty equation")
    return " ".join(parts), layout["rhs"]


def _layout(rng: random.Random) -> dict:
    return dict(
        negate=rng.random() < 0.3,
        partial=rng.random() < 0.3,
        order=rng.choice(_ORDERS),
        rhs=rng.choice(_RHS),
    )


def _noise(rng: random.Random, k_free: bool = True) -> dict:
    out = {"x": (rng.choice([-5, -3, -2, 1, 2, 3, 4]), 0), "y": (rng.choice([-4, -2, -1, 0, 0, 2, 5]), 0)}
    out["u"] = (rng.choice([-3, -1, 0, 0, 1, 2, 6]), 0)
    return out


def _dec(value: Fr, places: int = 4) -> str:
    return exact_decimal_string(value, places)


def dnum(v, tag: str | None) -> Distractor:
    v = Fr(v)
    return Distractor(str(v.numerator) if v.denominator == 1 else _dec(v, 4), tag, exact_value(v))


def numeric_dists(answer_text: str, cands: list, n: int = 4) -> list:
    """Distractors spelled like the answer: whole numbers beside a whole-number answer, and beside a decimal answer
    every option at the answer's own number of decimal places (rounded half up), as a `number` skill's options are."""
    dp = len(answer_text.partition(".")[2]) if "." in answer_text else None
    out = []
    answer = Fr(answer_text)
    fallback = [(answer + off, None) for off in (1, -1, 2, -2, 3, -3, 4, -4, 6, -6) if answer + off >= 0 or answer < 0]
    for v, tag in [*cands, *fallback]:
        v = Fr(v)
        if dp is None:
            if v.denominator != 1:
                continue
            out.append(dnum(v, tag))
        else:
            r = round_half_up(v, dp)
            out.append(Distractor(stated_precision_string(r, dp), tag, exact_value(r)))
    rng = _RNG.get()
    if rng is None:
        return pick_distractors(answer_text, out, n, tolerance=GRADER_TOLERANCE)
    return fair(answer_text, pick_distractors(answer_text, out, n + 6, tolerance=GRADER_TOLERANCE), n, rng)


def _sign_pair(rng) -> int:
    return rng.choice([1, -1])


def _pm(v) -> str:
    """A signed constant as an operator and a magnitude: '+ 3' or '- 3'."""
    return f"+ {v}" if v >= 0 else f"- {-v}"


def _den(v: int) -> str:
    return str(v) if v > 0 else f"({v})"


def _frac_text(f: Fr) -> str:
    return rational_text(f)


def set_fields(items: list) -> dict:
    text = set_answer_text(items)
    return dict(answer_kind="set", answer=list(items), answer_text=text, answer_latex="plain:" + text)


def set_dists(key: list, cands: list, n: int = 4) -> list:
    """Wrong sets, each given as (items, tag): distinct from the key and from each other, as unordered sets."""
    seen = {tuple(sorted(str(x) for x in key))}
    out = []
    for items, tag in cands:
        norm = tuple(sorted(str(x) for x in items))
        if not items or norm in seen:
            continue
        seen.add(norm)
        out.append(Distractor(set_answer_text(items), tag, list(items)))
        if len(out) == n:
            break
    return out


def _fmt_item(f: Fr) -> str:
    return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"


# -- phrases -----------------------------------------------------------------------------------------------------------


def _syms(layout: dict) -> tuple:
    s = _PARTIAL if layout["partial"] else _SUBSCRIPT
    return s["xx"], s["xy"], s["yy"]


def _coef_clause(layout: dict) -> str:
    xx, xy, yy = _syms(layout)
    return f"where A, B and C are the coefficients of {xx}, {xy} and {yy}"


# -- easy --------------------------------------------------------------------------------------------------------------


def _easy(rng: random.Random) -> dict:
    layout = _layout(rng)
    half = rng.random() < 0.3
    for _ in attempts(800, "an easy discriminant"):
        a = rng.choice([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6])
        c = rng.choice([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6])
        b = rng.choice([-8, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 8])
        if rng.random() < 0.12:
            a = 0
        elif rng.random() < 0.12:
            c = 0
        if half and b % 2:
            continue
        d = b * b - 4 * a * c
        if d == 0 or abs(d) > 70:
            continue
        break
    coefs = dict(xx=(a, 0), xy=(b, 0), yy=(c, 0))
    coefs.update({k: (v[0], 0) for k, v in _noise(rng).items()})
    lhs, rhs = equation(coefs, layout)
    xx, xy, yy = _syms(layout)
    shown = -1 if layout["negate"] else 1  # the sign of B as it is printed: a negated layout prints -b
    if half:
        value = Fr(b * b - 4 * a * c, 4)
        bb = b // 2
        check(value == bb * bb - a * c, "b^2 - AC and the discriminant disagree")
        ask = [
            f"Write {lhs} = {rhs} in the form A{xx} + 2b{xy} + C{yy} + (lower-order terms) = rhs, and find the value of b² - AC.",
            f"In {lhs} = {rhs}, the mixed term is written 2b{xy}. Find b² - AC, where A and C are the coefficients of {xx} and {yy}.",
            f"Take the equation {lhs} = {rhs} with mixed term 2b{xy}, and compute b² - AC.",
        ]
        wrongs = [
            (Fr(b * b - 4 * a * c), "pde-mixed-coefficient-convention"),
            (Fr(bb * bb - 4 * a * c), "pde-mixed-coefficient-convention"),
            (Fr(b * b - a * c), "pde-mixed-coefficient-convention"),
            (Fr(a * c - bb * bb), "reversed-discriminant-sign"),
            (Fr(bb * bb + a * c), "pde-4ac-added"),
            (Fr(shown * bb - a * c), "pde-b-not-squared"),
        ]
    else:
        value = Fr(b * b - 4 * a * c)
        ask = [
            f"Consider {lhs} = {rhs}. Here A, B and C are the coefficients of {xx}, {xy} and {yy}. Find the value of B² - 4AC.",
            f"For the equation {lhs} = {rhs}, use B² - 4AC, where B is the coefficient of {xy}, to find the discriminant.",
            f"Compute the discriminant B² - 4AC of the second-order part of {lhs} = {rhs} (A, B and C multiply {xx}, {xy} and {yy}).",
        ]
        e1 = coefs["y"][0]
        d1 = coefs["x"][0]
        wrongs = [
            (Fr(b * b - a * c), "dropped-factor-4-in-discriminant"),
            (Fr(4 * a * c - b * b), "reversed-discriminant-sign"),
            (Fr(shown * b - 4 * a * c), "pde-b-not-squared"),
            (Fr(b * b + 4 * a * c), "pde-4ac-added"),
            (Fr(b * b - 4 * a * (e1 if e1 else d1)), "pde-lower-order-in-discriminant"),
            (Fr(b * b - 4 * d1 * c), "pde-lower-order-in-discriminant"),
            (Fr(b * b // 4 - a * c) if b % 2 == 0 else Fr(b * b - 2 * a * c), "pde-mixed-coefficient-convention"),
        ]
    wrongs += [(value + 4, None), (value - 3, None), (value * 2, None)]
    stem = ask[rng.randrange(3)]
    answer = _fmt_item(value)
    dist = numeric_dists(answer, wrongs)
    check(len(dist) >= 3, "fewer than three distractors")
    return dict(
        text=stem,
        latex="",
        answer=numeric_answer(value),
        answer_text=answer,
        answer_latex=latex_answer(answer),
        answer_kind="number",
        working=[
            f"A = {a}, B = {b}, C = {c}"
            + (
                " (read from the equation as written; the lower-order terms play no part)"
                if not layout["negate"]
                else " (all signs are reversed, so read them as written)"
            )
            + ".",
            (
                f"b = {b // 2}, so b² - AC = {(b // 2) ** 2} - ({a})({c}) = {answer}."
                if half
                else f"B² - 4AC = {b * b} - 4({a})({c}) = {answer}."
            ),
        ],
        distractors=dist,
    )


# -- medium ------------------------------------------------------------------------------------------------------------

_PYTH = [(3, 4, 5), (4, 3, 5), (5, 12, 13), (12, 5, 13), (6, 8, 10), (8, 6, 10), (8, 15, 17), (15, 8, 17), (9, 12, 15)]
_CS = [1, 2, 4, 5, 8, 10, 20, 25]


def _medium_param_a(rng: random.Random) -> dict:
    """A = k: parabolic when k = B^2/(4C)."""
    layout = _layout(rng)
    for _ in attempts(500, "a clean parabolic value"):
        c = rng.choice(_CS)
        b = rng.choice([2, 3, 4, 5, 6, 7, 8, 9, 10, 12])
        k0 = Fr(b * b, 4 * c)
        try:
            _dec(k0)
        except ValueError:
            continue
        if not (Fr(1, 2) <= k0 <= 30):
            continue
        break
    sign = rng.choice([1, -1])
    coefs = dict(xx=(0, 1), xy=(sign * b, 0), yy=(c, 0))
    coefs.update(_noise(rng))
    lhs, rhs = equation(coefs, layout)
    xx, xy, yy = _syms(layout)
    disc = lambda k: Fr(b * b) - 4 * Fr(k) * c  # noqa: E731
    check(disc(k0) == 0, "not parabolic at k0")
    check(disc(k0 - Fr(1, 10)) > 0 > disc(k0 + Fr(1, 10)), "type does not change at k0")
    k_t = _dec(k0)
    ask = [
        f"The equation {lhs} = {rhs} has a constant k in one coefficient. For what positive value of k is it parabolic? Give k as a decimal.",
        f"Consider {lhs} = {rhs}. The discriminant B² - 4AC uses the coefficients of {xx}, {xy} and {yy}. Find the value of k at which the equation changes type (it is parabolic there). Give k as a decimal.",
        f"For {lhs} = {rhs}, find the value of k > 0 for which B² - 4AC = 0. Give k as a decimal.",
    ]
    wrongs = [
        (Fr(b, 4 * c), "pde-b-not-squared"),
        (Fr(b * b, c), "dropped-factor-4-in-discriminant"),
        (Fr(b * b, 2 * c), None),
        (Fr(4 * c, b * b), None),
        (k0 + 1, None),
        (k0 / 2, None),
        (Fr(2 * b * b, c), None),
    ]
    dist = numeric_dists(k_t, wrongs)
    check(len(dist) >= 3, "fewer than three distractors")
    return dict(
        text=ask[rng.randrange(3)],
        latex="",
        answer=numeric_answer(k0),
        answer_text=k_t,
        answer_latex=latex_answer(k_t),
        answer_kind="number",
        working=[
            f"A = k, B = {sign * b}, C = {c}, so B² - 4AC = {b * b} - {4 * c}k.",
            f"Setting this to 0 gives k = {b * b}/{4 * c} = {k_t}.",
        ],
        distractors=dist,
    )


def _medium_param_ac(rng: random.Random) -> dict:
    """A = k, C = k + c0: parabolic where 4k² + 4 c0 k - B² = 0, the positive root (h - c0)/2."""
    layout = _layout(rng)
    c0, b, h = rng.choice(_PYTH)
    c0 *= rng.choice([1, -1])
    b *= rng.choice([1, -1])
    k0 = Fr(h - c0, 2)
    check(k0 > 0, "the positive root must be positive")
    coefs = dict(xx=(0, 1), xy=(b, 0), yy=(c0, 1))
    coefs.update(_noise(rng))
    lhs, rhs = equation(coefs, layout)

    def disc(k: Fr) -> Fr:
        return Fr(b * b) - 4 * k * (k + c0)

    check(disc(k0) == 0, "not parabolic at the positive root")
    check(disc(Fr(-c0 - h, 2)) == 0, "the other root is wrong")
    k_t = _dec(k0)
    ask = [
        f"In {lhs} = {rhs}, the constant k appears in two coefficients. Find the positive value of k at which the equation is parabolic. Give k as a decimal.",
        f"The equation {lhs} = {rhs} changes type at one positive value of k. Find it, as a decimal.",
        f"Consider {lhs} = {rhs}, with B² - 4AC computed from the coefficients of the three second derivatives. For which k > 0 does B² - 4AC vanish? Give k as a decimal.",
    ]
    k_other = Fr(-c0 - h, 2)
    wrongs = [
        (-k_other, "pde-parameter-negative-values-missed"),
        (Fr(h + c0, 2), None),
        (k0 * 2, None),
        (Fr(h, 2), None),
        (Fr(b * b, 4 * (c0 if c0 else 1)), "forgot-square-root"),
        (k0 + Fr(1, 2), None),
    ]
    dist = numeric_dists(k_t, wrongs)
    check(len(dist) >= 3, "fewer than three distractors")
    return dict(
        text=ask[rng.randrange(3)],
        latex="",
        answer=numeric_answer(k0),
        answer_text=k_t,
        answer_latex=latex_answer(k_t),
        answer_kind="number",
        working=[
            f"A = k, B = {b}, C = k {_pm(c0)}, so B² - 4AC = {b * b} - 4k(k {_pm(c0)}).",
            f"B² - 4AC = 0 gives 4k² {_pm(4 * c0)}k - {b * b} = 0, whose roots are k = ({-c0} ± {h})/2. The positive one is k = {k_t}.",
        ],
        distractors=dist,
    )


def _family(rng: random.Random):
    """(coefs, d, disc_text, lo, hi): a parameter family whose discriminant d(k) changes sign inside lo..hi."""
    family = rng.choice(["b", "b", "a"])
    for _ in attempts(800, "a parameter family"):
        if family == "b":
            p = rng.choice([1, 1, 2, 3])
            a = rng.choice([1, 2, 3, 4, 5, 6])
            c = rng.choice([1, 2, 3, 4, 5, 6, 8, 9])
            sgn = rng.choice([1, -1])
            a, c = sgn * a, sgn * c
            n = rng.choice([5, 6, 7, 8])
            if not (2 <= (4 * a * c) / (p * p) <= (n - 1.5) ** 2):
                continue
            coefs = dict(xx=(a, 0), xy=(0, p), yy=(c, 0))

            def d(k, p=p, a=a, c=c):
                return p * p * k * k - 4 * a * c

            return coefs, d, f"{p * p}k² - {4 * a * c}", -n, n
        b = rng.choice([2, 3, 4, 5, 6, 7])
        c = rng.choice([1, 2, 3, 4, -1, -2])
        n = rng.choice([6, 8, 10])
        lo, hi = rng.choice([(-n // 2, n), (1, n), (-n, n)])
        if not (lo + 1 < b * b / (4 * c) < hi - 1):
            continue
        coefs = dict(xx=(0, 1), xy=(b, 0), yy=(c, 0))

        def d(k, b=b, c=c):
            return b * b - 4 * k * c

        return coefs, d, f"{b * b} {_pm(-4 * c)}k", lo, hi
    raise AssertionError("unreachable")


def _medium_count(rng: random.Random) -> dict:
    layout = _layout(rng)
    mode = rng.choice(["count", "count", "edge"])
    for _ in attempts(300, "a count or edge question"):
        coefs, d, disc_text, lo, hi = _family(rng)
        ks = list(range(lo, hi + 1))
        types = {k: ("hyperbolic" if d(k) > 0 else "elliptic" if d(k) < 0 else "parabolic") for k in ks}
        want = rng.choice(["hyperbolic", "elliptic"])
        other = "elliptic" if want == "hyperbolic" else "hyperbolic"
        cn = lambda t: sum(1 for k in ks if types[k] == t)  # noqa: E731
        if mode == "count":
            if 2 <= cn(want) < len(ks) - 2:
                break
        else:
            pos = [k for k in ks if k >= 1]
            pset = [k for k in pos if types[k] == want]
            if pset and len(pset) < len(pos) and (hi in pset) != (1 in pset):
                edge = min(pset) if hi in pset else max(pset)
                if 1 < edge < hi:
                    break
    para = [k for k in ks if types[k] == "parabolic"]
    coefs.update(_noise(rng))
    lhs, rhs = equation(coefs, layout)
    xx, xy, yy = _syms(layout)
    if mode == "count":
        value = cn(want)
        wrongs = [
            (cn(other), "pde-type-condition-reversed"),
            (cn(want) + len(para), "included-the-parabolic-value"),
            (len(ks) - cn(want), "pde-type-condition-reversed"),
            (sum(1 for k in ks if k > 0 and types[k] == want), "pde-parameter-negative-values-missed"),
            (cn(want) + 1, None),
            (cn(want) - 1, None),
        ]
        ask = [
            f"The equation {lhs} = {rhs} contains a whole-number constant k. How many integers k with {lo} ≤ k ≤ {hi} make it {want}?",
            f"Consider {lhs} = {rhs}. Use B² - 4AC with A, B, C the coefficients of {xx}, {xy}, {yy}. For how many integers k from {lo} to {hi} is the equation {want}?",
            f"For {lhs} = {rhs}, count the integers k in the range {lo} ≤ k ≤ {hi} that make it {want}.",
        ]
        if not para:
            wrongs = [w for w in wrongs if w[1] != "included-the-parabolic-value"]
        if all(k >= 0 for k in ks if types[k] == want) or not any(k > 0 for k in ks):
            wrongs = [w for w in wrongs if w[1] != "pde-parameter-negative-values-missed"]
        working = [
            f"The discriminant is {disc_text}.",
            f"It is {'positive' if want == 'hyperbolic' else 'negative'} for {value} of the {len(ks)} integers"
            + (
                f"; the value{'s' if len(para) > 1 else ''} {', '.join(str(k) for k in para)} give 0 (parabolic) and are not counted."
                if para
                else "."
            ),
        ]
    else:
        smallest = hi in pset
        value = min(pset) if smallest else max(pset)
        check(1 < value < hi or (value not in (1, hi)), "the answer sits on the end of the range")
        adj = value - 1 if smallest else value + 1
        wrongs = [
            *([(para[0], "included-the-parabolic-value")] if para and para[0] > 0 else []),
            (adj, None),
            (value + (1 if smallest else -1) * 2, None),
            (
                (max if smallest else min)(
                    [k for k in pos if types[k] == ("elliptic" if want == "hyperbolic" else "hyperbolic")]
                    or [value + 3]
                ),
                "pde-type-condition-reversed",
            ),
        ]
        which = "smallest" if smallest else "largest"
        ask = [
            f"The equation {lhs} = {rhs} contains a whole-number constant k. Among the positive integers k with k ≤ {hi}, what is the {which} one that makes it {want}?",
            f"Consider {lhs} = {rhs}. Use B² - 4AC with A, B, C the coefficients of {xx}, {xy}, {yy}. Find the {which} positive integer k not above {hi} for which the equation is {want}.",
            f"For {lhs} = {rhs}, find the {which} integer k with 1 ≤ k ≤ {hi} that makes it {want}.",
        ]
        working = [
            f"The discriminant is {disc_text}.",
            f"Among k = 1, ..., {hi} it is {'positive' if want == 'hyperbolic' else 'negative'} exactly for {', '.join(str(k) for k in pset)}"
            + (f" (k = {para[0]} gives 0, which is parabolic)" if para and para[0] > 0 else "")
            + f", so the {which} is {value}.",
        ]
    answer = str(value)
    wrongs += [(value + 2, None), (value - 2, None), (value + 3, None), (value - 1, None), (value + 1, None)]
    dist = numeric_dists(answer, [(Fr(v), t) for v, t in wrongs if v >= 0])
    check(len(dist) >= 3, "fewer than three distractors")
    return dict(
        text=ask[rng.randrange(3)],
        latex="",
        answer=value,
        answer_text=answer,
        answer_latex=answer,
        answer_kind="number",
        working=working,
        distractors=dist,
    )


def _medium(rng: random.Random) -> dict:
    return rng.choice([_medium_param_a, _medium_param_ac, _medium_count, _medium_count])(rng)


# -- hard --------------------------------------------------------------------------------------------------------------

_SLOPES = [Fr(p, q) for q in (1, 2, 3) for p in range(-7, 8) if p and math.gcd(abs(p), q) == 1]

_EXAM_SLOPES = [Fr(p, q) for q in (1, 2, 3, 4) for p in range(-7 * q, 7 * q + 1) if p and math.gcd(abs(p), q) == 1]


def _hyperbolic(rng: random.Random, lo: int = 1, hi: int = 4) -> tuple:
    """(A, B, C, s1, s2) with A s^2 - B s + C = A (s - s1)(s - s2), integer coefficients, s1 != s2, s1 s2 != 0."""
    for _ in attempts(2000, "a hyperbolic equation with clean slopes"):
        s1, s2 = rng.sample(_SLOPES, 2)
        if abs(s1) > 5 or abs(s2) > 5:
            continue
        a = rng.choice(range(lo, hi + 1)) * rng.choice([1, 1, -1])
        b, c = a * (s1 + s2), a * s1 * s2
        if b.denominator != 1 or c.denominator != 1:
            continue
        if abs(b) > 40 or abs(c) > 40:
            continue
        return a, int(b), int(c), s1, s2
    raise AssertionError("unreachable")


def _check_slopes(a: int, b: int, c: int, s1: Fr, s2: Fr) -> None:
    for s in (s1, s2):
        check(a * s * s - b * s + c == 0, "slope does not satisfy the characteristic equation")
    check(b * b - 4 * a * c > 0, "equation is not hyperbolic")
    check(s1 != s2, "slopes coincide")


def _xy(s: Fr) -> str:
    m = abs(s)
    body = (
        "x"
        if m == 1
        else (
            f"{m.numerator}x"
            if m.denominator == 1
            else (f"x/{m.denominator}" if m.numerator == 1 else f"{m.numerator}x/{m.denominator}")
        )
    )
    return f"y {'-' if s > 0 else '+'} {body}"


def _hard_slopes(rng: random.Random, with_k: bool) -> dict:
    layout = _layout(rng)
    a, b, c, s1, s2 = _hyperbolic(rng)
    _check_slopes(a, b, c, s1, s2)
    k_star = None
    coefs: dict = {}
    if with_k:
        k_star = rng.choice([1, 2, 3, 4, -1, -2])
        slopes_k = [rng.choice([-2, -1, 0, 1, 2]) for _ in range(3)]
        if not any(slopes_k):
            slopes_k[rng.randrange(3)] = rng.choice([-2, -1, 1, 2])
        for (name, val), c1 in zip((("xx", a), ("xy", b), ("yy", c)), slopes_k):
            coefs[name] = (val - c1 * k_star, c1)
        for name in ("xx", "xy", "yy"):
            check(lin_eval(coefs[name], k_star) == {"xx": a, "xy": b, "yy": c}[name], "parameter substitution")
    else:
        coefs = dict(xx=(a, 0), xy=(b, 0), yy=(c, 0))
    coefs.update(_noise(rng))
    lhs, rhs = equation(coefs, layout)
    big, small = max(s1, s2), min(s1, s2)
    larger = rng.random() < 0.5
    value, other = (big, small) if larger else (small, big)
    which = "larger" if larger else "smaller"
    root = Fr(abs((s1 - s2) * a))  # sqrt(D) = |A (s1 - s2)|
    check(root * root == Fr(b * b - 4 * a * c), "discriminant is not the square of A (s1 - s2)")
    sgn = 1 if larger else -1
    # with A > 0 the larger slope uses +sqrt(D); with A < 0 it uses -sqrt(D)
    wrongs = [
        (other, "pde-slope-wrong-root"),
        (-value, "pde-slope-sign-flipped"),
        ((Fr(b) + sgn * root) / (2 * c), "pde-slope-denominator-c"),
        (Fr(b) + sgn * root * (1 if a > 0 else -1), "pde-slope-factor-2a-dropped"),
        (1 / value, "pde-slope-reciprocal"),
        (value + 1, None),
        (value * 2, None),
    ]
    chars = "The characteristic curves y = y(x) satisfy A(dy/dx)² - B(dy/dx) + C = 0, " + _coef_clause(layout) + "."
    if with_k:
        ask = [
            f"{chars} Let k = {k_star} in the equation {lhs} = {rhs}. Find the {which} of the two characteristic slopes dy/dx.",
            f"Put k = {k_star} in the equation {lhs} = {rhs}, which is then hyperbolic. {chars} What is the {which} of its two slopes dy/dx?",
            f"The equation {lhs} = {rhs} is taken at k = {k_star}. {chars} Find the {which} slope dy/dx.",
        ]
    else:
        ask = [
            f"{chars} The equation {lhs} = {rhs} is hyperbolic. Find the {which} of the two slopes dy/dx.",
            f"Consider {lhs} = {rhs}. {chars} What is the {which} of the two characteristic slopes dy/dx?",
            f"Find the {which} characteristic slope dy/dx of {lhs} = {rhs}. {chars}",
        ]
    text_key = rational_text(value)
    out, seen = [], {value}
    for v, tag in wrongs:
        v = Fr(v)
        if v in seen:
            continue
        seen.add(v)
        out.append(Distractor(rational_text(v), tag, exact_value(v)))
    dist = pick_distractors(text_key, out, 4)
    check(len(dist) >= 3, "fewer than three distractors")
    working = [
        f"At the stated parameter value A = {a}, B = {b}, C = {c}." if with_k else f"A = {a}, B = {b}, C = {c}.",
        f"B² - 4AC = {b * b - 4 * a * c} = {int(root)}², so dy/dx = ({b} ± {int(root)})/{_den(2 * a)}, giving {rational_text(small)} and {rational_text(big)}; the {which} is {text_key}.",
    ]
    return dict(
        text=ask[rng.randrange(3)] + FRACTION_FORMAT,
        latex="",
        answer=exact_value(value),
        answer_text=text_key,
        answer_latex=latex_answer(text_key),
        answer_kind="fraction",
        working=working,
        distractors=dist,
    )


def _chain_rule_mixed(a: int, b: int, c: int, s1: Fr, s2: Fr) -> Fr:
    """Coefficient of w_{xi eta} when xi = y - s1 x and eta = y - s2 x, by the chain rule."""
    xi_x, xi_y, eta_x, eta_y = -s1, Fr(1), -s2, Fr(1)
    # u_xx -> 2 xi_x eta_x w_xieta, u_xy -> (xi_x eta_y + xi_y eta_x) w_xieta, u_yy -> 2 xi_y eta_y w_xieta
    return 2 * a * xi_x * eta_x + b * (xi_x * eta_y + xi_y * eta_x) + 2 * c * xi_y * eta_y


def _hard_canonical(rng: random.Random) -> dict:
    layout = _layout(rng)
    layout["negate"] = False  # the sign of A is already random, and Q depends on the coefficients as displayed
    a, b, c, s1, s2 = _hyperbolic(rng)
    _check_slopes(a, b, c, s1, s2)
    # the diagonal coefficients vanish on the characteristics
    check(a * s1 * s1 - b * s1 + c == 0 and a * s2 * s2 - b * s2 + c == 0, "xi, eta are not characteristic")
    q = _chain_rule_mixed(a, b, c, s1, s2)
    d = Fr(b * b - 4 * a * c)
    check(q == -d / a, "chain rule and -D/A disagree")
    check(q == -a * (s1 - s2) ** 2, "chain rule and -A (s1 - s2)^2 disagree")
    coefs = dict(xx=(a, 0), xy=(b, 0), yy=(c, 0))
    coefs.update(_noise(rng))
    lhs, rhs = equation(coefs, layout)
    first, second = (s1, s2) if rng.random() < 0.5 else (s2, s1)
    ask = [
        f"Let u(x, y) = w(ξ, η) with ξ = {_xy(first)} and η = {_xy(second)}. The equation {lhs} = {rhs} becomes P w_ξξ + Q w_ξη + R w_ηη + (lower-order terms) = rhs. Find Q.",
        f"Substitute ξ = {_xy(first)}, η = {_xy(second)} into {lhs} = {rhs}, writing u(x, y) = w(ξ, η). Find the coefficient Q of w_ξη in the result.",
        f"After the change of variables ξ = {_xy(first)}, η = {_xy(second)}, the equation {lhs} = {rhs} takes the form P w_ξξ + Q w_ξη + R w_ηη + ... = rhs. What is Q?",
    ]
    key = q
    wrongs = [
        (d / a, "pde-canonical-coefficient-sign"),
        (-d, "pde-canonical-coefficient-undivided"),
        (d, "pde-canonical-coefficient-undivided"),
        (-d / (2 * a), None),
        (-d / c if c else d, None),
        (-q, "pde-canonical-coefficient-sign"),
        (q * 2, None),
        (Fr(2 * a) * s1 * s2, "pde-canonical-coefficient-undivided"),
    ]
    text_key = rational_text(key)
    out, seen = [], {key}
    for v, tag in wrongs:
        v = Fr(v)
        if v in seen:
            continue
        seen.add(v)
        out.append(Distractor(rational_text(v), tag, exact_value(v)))
    dist = pick_distractors(text_key, out, 4)
    check(len(dist) >= 3, "fewer than three distractors")
    return dict(
        text=ask[rng.randrange(3)] + FRACTION_FORMAT,
        latex="",
        answer=exact_value(key),
        answer_text=text_key,
        answer_latex=latex_answer(text_key),
        answer_kind="fraction",
        working=[
            f"A = {a}, B = {b}, C = {c}, so the characteristic slopes are {_fmt_item(s1)} and {_fmt_item(s2)}: the new variables are the characteristic ones, and P = R = 0.",
            f"Q = 2A ξx ηx + B(ξx ηy + ξy ηx) + 2C ξy ηy = {text_key} (equivalently -(B² - 4AC)/A).",
        ],
        distractors=dist,
    )


def _hard(rng: random.Random) -> dict:
    kind = rng.random()
    if kind < 0.35:
        return _hard_slopes(rng, False)
    if kind < 0.6:
        return _hard_slopes(rng, True)
    return _hard_canonical(rng)


_TIERS = {"easy": _easy, "medium": _medium, "hard": _hard}


@generator(
    TOPIC,
    level=17,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@uses_rng
def w5b_pde_discriminant(rng: random.Random, tier: str = "easy") -> dict:
    """The value behind the classification of a second-order PDE: a number, a set, or a fraction."""
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    return _TIERS[tier](rng)


# -- exam ------------------------------------------------------------------------------------------------------------


@generator(EXAM_TOPIC, level=17, presets={"exam": dict(tier="exam")})
@uses_rng
def exd_x_w5b_pde_discriminant(rng: random.Random, tier: str = "exam") -> dict:
    """B = p k: the discriminant at k, the integers k that make it elliptic and hyperbolic, the characteristic slopes."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    layout = _layout(rng)
    for _ in attempts(3000, "an exam equation"):
        s1, s2 = rng.sample(_EXAM_SLOPES, 2)
        if abs(s1) > 7 or abs(s2) > 7:
            continue
        a = rng.choice([1, 2, 3, 4, 5, 6, 8, 12, -1, -2, -3, -4, -5, -8])
        b2, c = a * (s1 + s2), a * s1 * s2
        if b2.denominator != 1 or c.denominator != 1 or b2 == 0:
            continue
        b2, c = int(b2), int(c)
        divisors = [p for p in (1, 2, 3, 4, 5, 6, 7, 8) if b2 % p == 0 and abs(b2 // p) <= 14 and b2 // p != 0]
        if not divisors or abs(c) > 150:
            continue
        p = rng.choice(divisors)
        k2 = b2 // p
        if a * c <= 0:
            continue  # keep both elliptic and hyperbolic values in range
        n = rng.choice([5, 6, 7, 8, 9])
        if not (1.5 <= (4 * a * c) / (p * p) < (n - 1) ** 2):
            continue
        k1 = rng.choice([k for k in range(-7, 8) if k not in (0, k2)])
        break
    else:
        raise AssertionError("unreachable")

    def disc(k: int) -> int:
        return p * p * k * k - 4 * a * c

    d_at = disc(k1)
    ks = range(-n, n + 1)
    ell = sum(1 for k in ks if disc(k) < 0)
    hyp = sum(1 for k in ks if disc(k) > 0)
    par = sum(1 for k in ks if disc(k) == 0)
    check(ell + hyp + par == 2 * n + 1 and ell >= 1 and hyp >= 1, "type counts")
    _check_slopes(a, p * k2, c, s1, s2)
    coefs = dict(xx=(a, 0), xy=(0, p), yy=(c, 0))
    coefs.update(_noise(rng))
    lhs, rhs = equation(coefs, layout)
    xx, xy, yy = _syms(layout)
    stem_variants = [
        f"Consider {lhs} = {rhs}, where k is a constant. A, B and C are the coefficients of {xx}, {xy} and {yy}, and the discriminant is B² - 4AC.",
        f"The equation {lhs} = {rhs} involves a whole-number constant k. Classify it by B² - 4AC, {_coef_clause(layout)}.",
        f"A second-order equation is {lhs} = {rhs}, with constant k. Its discriminant is B² - 4AC, {_coef_clause(layout)}.",
    ]
    key_slopes = sorted([s1, s2])
    slopes_fields = set_fields([_fmt_item(s) for s in key_slopes])
    root = abs(p * k2) if False else abs(a * (s1 - s2))
    # distractors
    da = numeric_dists(
        str(d_at),
        [
            (-d_at, "reversed-discriminant-sign"),
            (Fr(p * k1 * k1 * p) - a * c, "dropped-factor-4-in-discriminant"),
            (Fr(p * k1 * k1) - 4 * a * c, "forgot-to-square-coefficient"),
            (Fr(p * p - 4 * a * c), "forgot-parameter-in-discriminant"),
            (Fr(p * p * k1 * k1 + 4 * a * c), "pde-4ac-added"),
            (Fr(d_at) + p, None),
        ],
    )
    db = numeric_dists(
        str(ell),
        [
            (hyp, "pde-type-condition-reversed"),
            (ell + par, "included-the-parabolic-value"),
            (ell + 1, None),
            (ell - 1, None),
            (Fr(2 * n + 1 - hyp), None),
            (sum(1 for k in range(1, n + 1) if disc(k) < 0), "pde-parameter-negative-values-missed"),
        ],
    )
    dc = numeric_dists(
        str(hyp),
        [
            (ell, "pde-type-condition-reversed"),
            (hyp + par, "included-the-parabolic-value"),
            (sum(1 for k in range(1, n + 1) if disc(k) > 0), "pde-parameter-negative-values-missed"),
            (hyp + 1, None),
            (hyp - 1, None),
            (2 * n + 1 - ell, "included-the-parabolic-value"),
        ],
    )
    s_wrongs = [
        ([-s for s in key_slopes], "pde-slope-sign-flipped"),
        ([(Fr(p * k2) + sg * root) / (2 * c) for sg in (1, -1)], "pde-slope-denominator-c"),
        ([Fr(p * k2) + sg * root for sg in (1, -1)], "pde-slope-factor-2a-dropped"),
        ([1 / s for s in key_slopes], "pde-slope-reciprocal"),
        ([s + 1 for s in key_slopes], None),
    ]
    dd = set_dists(
        [_fmt_item(s) for s in key_slopes],
        [([_fmt_item(Fr(x)) for x in items], tag) for items, tag in s_wrongs],
    )
    check(len(da) >= 2 and len(db) >= 2 and len(dc) >= 2 and len(dd) >= 3, "too few distractors")
    chars = "The characteristic curves y = y(x) satisfy A(dy/dx)² - B(dy/dx) + C = 0."
    parts = [
        dict(
            prompt=f"Find the value of B² - 4AC when k = {k1}.",
            answer_kind="number",
            answer=d_at,
            answer_text=str(d_at),
            answer_latex=str(d_at),
            marks=2,
            working=[f"A = {a}, B = {p}k = {p * k1}, C = {c}, so B² - 4AC = {(p * k1) ** 2} - 4({a})({c}) = {d_at}."],
            distractors=da,
        ),
        dict(
            prompt=f"How many integers k with {-n} ≤ k ≤ {n} make the equation elliptic?",
            answer_kind="number",
            answer=ell,
            answer_text=str(ell),
            answer_latex=str(ell),
            marks=2,
            sensible_range=(0, 2 * n + 1),
            working=[f"B² - 4AC = {p * p}k² - {4 * a * c} < 0 for {ell} of the {2 * n + 1} integers."],
            distractors=db,
        ),
        dict(
            prompt=f"How many integers k with {-n} ≤ k ≤ {n} make the equation hyperbolic?",
            answer_kind="number",
            answer=hyp,
            answer_text=str(hyp),
            answer_latex=str(hyp),
            marks=2,
            sensible_range=(0, 2 * n + 1),
            working=[f"B² - 4AC > 0 for {hyp} of them; {par} give 0 (parabolic) and are in neither count."],
            distractors=dc,
        ),
        dict(
            prompt=f"{chars} For k = {k2}, find the two slopes dy/dx of the characteristics as a set.{SET_FORMAT}",
            marks=3,
            working=[
                f"At k = {k2}: A = {a}, B = {p * k2}, C = {c}; B² - 4AC = {(p * k2) ** 2 - 4 * a * c} = {root}².",
                f"dy/dx = ({p * k2} ± {root})/{_den(2 * a)} gives {slopes_fields['answer_text']}.",
            ],
            distractors=dd,
            **slopes_fields,
        ),
    ]
    return dict(text=stem_variants[rng.randrange(3)], latex="", parts=parts)
