"""Generates `cp1_factor_remainder_proof` items: prove the identities behind the factor theorem and the remainder
theorem, or apply them.

Topic ``cp1_factor_remainder_proof`` (open-response proof, ``level=14``). PROVE items are chains of equal
expressions: x^m - a^m factorised, P(x) - P(a) written as (x - a) times a quotient, and P(x) rewritten as
(x - a)Q(x) + R by dividing one term at a time (so R = P(a) is visible). APPLY items are numeric chains that find a
remainder by substituting into P.

Tiers: x^m - a^m; P(x) - P(a) for a cubic with numbers; division step by step for a cubic or quartic with numbers;
division with a general first coefficient, a general cubic with symbols p, q, r and a, and x^m + a^m for odd m.

The generator never runs a full CAS chain check. Every claim is confirmed with SymPy's polynomial division, and every
reference chain numerically at fixed irregular points, with the `math` module only.
"""

from __future__ import annotations

import functools
from fractions import Fraction

import sympy as sp

from ..core import check, generator
from . import _chain_proof_kit as kit
from ._chain_proof_kit import M, T

_x = sp.Symbol("x")


def _poly(coeffs: list[int]) -> str:
    """The polynomial with the given coefficients (highest power first) as a sympy-style string."""
    degree = len(coeffs) - 1
    return str(sp.Poly(coeffs, _x).as_expr()) if degree >= 0 else "0"


def _identity(key, tier, ref, lead, *, symbols=None, domain="Take a to be a real number.", jump_ok=False, given=None):
    return dict(
        key=key,
        tier=tier,
        kind="identity",
        symbols=symbols or {"x": "real", "a": "real"},
        ref=ref,
        lead=lead,
        domain=domain,
        jump_ok=jump_ok,
        given_step=given or "states what is being divided by (x - a) and the quotient and remainder it produces",
    )


def _join_signed(items: list[tuple[int, str]]) -> str:
    """'3*x**2*(x-a) - 2*x*(x-a) + ...' from (coefficient, factor text) pairs, dropping zero coefficients."""
    out = ""
    for coef, factor in items:
        if coef == 0:
            continue
        mag = abs(coef)
        body = factor if mag == 1 else f"{mag}*{factor}"
        if not out:
            out = ("-" if coef < 0 else "") + body
        else:
            out += (" - " if coef < 0 else " + ") + body
    return out or "0"


def _mono(a: str, i: int, power_x: int) -> str:
    return str(sp.sympify(f"{a}**{i}*x**{power_x}"))


def _power_difference(m: int, a: str, tier: str = "easy") -> dict:
    quotient = " + ".join(_mono(a, i, m - 1 - i) for i in range(m))
    start = f"x**{m} - {a}**{m}"
    target = f"(x-{a})*({quotient})"
    middle = " + ".join(f"{_mono(a, i, m - 1 - i)}*(x-{a})" for i in range(m))
    numeric = a.isdigit()
    return _identity(
        f"power_difference_{m}_{a}",
        tier,
        [start, middle, target],
        [
            T(f"The factor theorem says x - {a} is a factor of x^{m} - {a}^{m}. Prove it by showing that "),
            M(start),
            T(" = "),
            M(target),
            T("."),
        ],
        symbols={"x": "real"} if numeric else {"x": "real", "a": "real"},
        domain="Take x to be a real number." if numeric else "Take a to be a real number.",
    )


def _cubic_difference(coeffs: list[int], a: int) -> dict:
    poly = sp.Poly(coeffs, _x)
    value = int(poly.eval(a))
    check(value != 0, "cubic_difference: P(a) = 0 makes the difference trivial")
    quotient, rem = sp.div(poly - value, sp.Poly([1, -a], _x))
    check(rem.is_zero, "cubic_difference: P(x) - P(a) is not divisible by x - a")
    p_text = _poly(coeffs)
    xa = f"(x-{a})" if a >= 0 else f"(x+{-a})"
    shown_xa = xa.replace("x-", "x - ").replace("x+", "x + ")
    start = f"{p_text} - ({value})"
    target = f"{xa}*({quotient.as_expr()})"
    middle = _poly([*coeffs[:-1], coeffs[-1] - value])
    return _identity(
        f"cubic_difference_{'_'.join(map(str, coeffs))}_{a}",
        "medium",
        [start, middle, target],
        [
            T("Let P(x) = "),
            M(p_text),
            T(f". Then P({a}) = {value}. Show that P(x) - P({a}) is divisible by {shown_xa}, by showing that "),
            M(start),
            T(" = "),
            M(target),
            T("."),
        ],
        symbols={"x": "real"},
        domain="Take x to be a real number.",
    )


def _division_chain(coeffs: list[int], a: int, tier: str = "hard") -> dict:
    degree = len(coeffs) - 1
    b = [coeffs[0]]
    for c in coeffs[1:]:
        b.append(c + a * b[-1])
    quotient, remainder = b[:-1], b[-1]
    poly = sp.Poly(coeffs, _x)
    q_poly, r_poly = sp.div(poly, sp.Poly([1, -a], _x))
    check(int(poly.eval(a)) == remainder, "division_chain: the remainder is not P(a)")
    check(list(q_poly.all_coeffs()) == quotient and r_poly.as_expr() == remainder, "division_chain: quotient is wrong")
    xa = f"(x-{a})" if a >= 0 else f"(x+{-a})"
    shown_xa = xa.replace("x-", "x - ").replace("x+", "x + ")
    p_text = _poly(coeffs)

    def peeled(count: int) -> str:
        items = []
        for j in range(count):
            power = degree - 1 - j
            factor = f"x**{power}*{xa}" if power > 1 else (f"x*{xa}" if power == 1 else xa)
            items.append((quotient[j], factor))
        return _join_signed(items)

    lines = [p_text]
    for i in range(1, degree + 1):
        rest = f"({_poly([b[i], *coeffs[i + 1 :]])})" if i < degree else f"({remainder})"
        lines.append(f"{peeled(i)} + {rest}")
    target = f"{xa}*({_poly(quotient)}) + ({remainder})"
    lines.append(target)
    ref = [ln for i, ln in enumerate(lines) if i == 0 or sp.sympify(ln) != sp.sympify(lines[i - 1])]
    kind = "quartic" if degree == 4 else "cubic"
    return _identity(
        f"division_{kind}_{'_'.join(map(str, coeffs))}_{a}",
        tier,
        ref,
        [
            T("Let P(x) = "),
            M(p_text),
            T(f". Divide one term at a time to show that P(x) = {shown_xa}Q(x) + R with Q(x) = "),
            M(_poly(quotient)),
            T(f" and R = {remainder}; that is, show that "),
            M(p_text),
            T(" = "),
            M(target),
            T(f". This R is P({a}), the remainder theorem for the divisor {shown_xa}."),
        ],
        symbols={"x": "real"},
        domain="Take x to be a real number.",
        jump_ok=len(ref) >= 4,
        given="states that the quotient is built one term at a time and that the constant left over is the remainder",
    )


def _general_cubic() -> dict:
    start = "x**3 + p*x**2 + q*x + r"
    b1, b0 = "(a+p)", "(q+a*(a+p))"
    rem = "(r+a*(q+a*(a+p)))"
    ref = [
        start,
        f"x**2*(x-a) + {b1}*x**2 + q*x + r",
        f"x**2*(x-a) + {b1}*x*(x-a) + {b0}*x + r",
        f"x**2*(x-a) + {b1}*x*(x-a) + {b0}*(x-a) + {rem}",
        f"(x-a)*(x**2 + {b1}*x + {b0}) + {rem}",
    ]
    return _identity(
        "general_cubic",
        "exam",
        ref,
        [
            T(
                "Prove the remainder theorem for a cubic. Divide P(x) = x^3 + px^2 + qx + r by (x - a) one term at a time to show that "
            ),
            M(start),
            T(" = "),
            M(ref[-1]),
            T(", and so the remainder is P(a) = a^3 + pa^2 + qa + r."),
        ],
        symbols={"x": "real", "a": "real", "p": "real", "q": "real", "r": "real"},
        jump_ok=True,
    )


def _general_quartic_numeric(a: int) -> dict:
    coeffs = [1, 0, -7, 2, 6] if a == 2 else [1, 3, 0, -5, 4]
    return _division_chain(coeffs, a, "exam")


def _odd_power_sum(m: int) -> dict:
    sign = lambda i: "-" if i % 2 else ""  # noqa: E731
    quotient = " ".join(
        (f"{'+' if i else ''} {sign(i)}{_mono('a', i, m - 1 - i)}" if i % 2 == 0 else f"- {_mono('a', i, m - 1 - i)}")
        for i in range(m)
    ).strip()
    quotient = quotient.replace("+  ", "+ ")
    start = f"x**{m} + a**{m}"
    target = f"(x+a)*({quotient})"
    middle = " ".join(
        (f"+ {_mono('a', i, m - 1 - i)}*(x+a)" if i % 2 == 0 else f"- {_mono('a', i, m - 1 - i)}*(x+a)")
        for i in range(m)
    )
    middle = middle[2:] if middle.startswith("+ ") else middle
    return _identity(
        f"odd_power_sum_{m}",
        "exam",
        [start, middle, target],
        [
            T(
                f"For an odd power, x + a is a factor of x^{m} + a^{m} (because x = -a makes it zero). Prove it by showing that "
            ),
            M(start),
            T(" = "),
            M(target),
            T("."),
        ],
        symbols={"x": "real", "a": "real"},
    )


# --- APPLY items -------------------------------------------------------------------------------------------------
def _eval_text(coeffs: list[int], value: str) -> str:
    degree = len(coeffs) - 1
    parts = []
    for i, c in enumerate(coeffs):
        power = degree - i
        if c == 0:
            continue
        coef = "" if abs(c) == 1 and power >= 1 else f"{abs(c)}*"
        term = f"{coef}({value})**{power}" if power > 1 else (f"{coef}({value})" if power == 1 else f"{abs(c)}")
        parts.append(("-" if c < 0 else "+") + term)
    text = "".join(parts)
    return text[1:] if text.startswith("+") else text


def _apply_remainder(coeffs: list[int], a: int) -> dict:
    poly = sp.Poly(coeffs, _x)
    value = int(poly.eval(a))
    check(int(sp.rem(poly, sp.Poly([1, -a], _x)).as_expr()) == value, "apply_remainder: remainder is not P(a)")
    shown = f"(x - {a})" if a >= 0 else f"(x + {-a})"
    ref = [_eval_text(coeffs, str(a)), f"{value}"]
    return dict(
        key=f"apply_remainder_{'_'.join(map(str, coeffs))}_{a}",
        tier="apply_easy",
        lead=[
            T("Use the remainder theorem to find the remainder when P(x) = "),
            M(_poly(coeffs)),
            T(f" is divided by {shown}."),
        ],
        ref=ref,
        wrong_final=f"{value + 3}",
        answer_text=f"remainder = {value}",
        given_step=f"states that the remainder on division by {shown} is P({a}) and substitutes",
    )


def _apply_linear(coeffs: list[int], num: int, den: int) -> dict:
    point = Fraction(num, den)
    value = sum(Fraction(c) * point ** (len(coeffs) - 1 - i) for i, c in enumerate(coeffs))
    text = f"{num}/{den}"
    expr = _eval_text(coeffs, text)
    poly_value = value.numerator if value.denominator == 1 else f"{value.numerator}/{value.denominator}"
    ref = [expr, f"{poly_value}"]
    divisor = f"({den}x - {num})" if num >= 0 else f"({den}x + {-num})"
    return dict(
        key=f"apply_linear_{'_'.join(map(str, coeffs))}_{num}_{den}",
        tier="apply_exam",
        lead=[
            T("Use the remainder theorem to find the remainder when P(x) = "),
            M(_poly(coeffs)),
            T(
                f" is divided by {divisor}. The remainder is P at the value of x that makes the divisor zero, here x = {num}/{den}."
            ),
        ],
        ref=ref,
        wrong_final=(f"{value + 1}".replace(" ", "")),
        answer_text=f"remainder = {poly_value}",
        given_step="finds the value of x that makes the divisor zero and substitutes it into P",
    )


def _build_pools() -> dict[str, list[dict]]:
    pools: dict[str, list[dict]] = {k: [] for k in ("easy", "medium", "hard", "exam", "apply_easy", "apply_exam")}
    pools["easy"] += [_power_difference(m, "a") for m in (2, 3, 4, 5)]
    pools["easy"] += [_power_difference(m, str(a)) for m, a in ((3, 2), (3, 3), (4, 2), (3, 4), (4, 3), (5, 2))]
    for coeffs, a in (
        ([2, -3, 1, -5], 2),
        ([1, -4, 2, 6], 3),
        ([3, 2, -7, 1], 1),
        ([1, 0, -5, 4], 2),
        ([2, 1, -5, 3], -1),
        ([1, 3, -2, -7], -2),
        ([4, -1, 0, 7], 1),
        ([1, -2, -5, 7], 3),
        ([5, 0, -3, 2], 2),
        ([2, -5, 4, -1], 3),
    ):
        pools["medium"].append(_cubic_difference(coeffs, a))
    for coeffs, a in (
        ([2, -3, 1, -5], 2),
        ([1, -4, 1, 6], 3),
        ([3, 2, -7, 1], 1),
        ([1, 0, -6, 4], -2),
        ([2, 1, -5, 3], 2),
        ([1, 3, -2, -8], -1),
        ([2, -1, 3, 4], 3),
        ([1, -2, 0, 5], 2),
        ([3, 0, 1, -9], 1),
        ([1, 5, 2, -6], -3),
    ):
        pools["hard"].append(_division_chain(coeffs, a, "hard"))
    pools["exam"] += [_general_cubic(), _general_quartic_numeric(2), _general_quartic_numeric(-1)]
    pools["exam"] += [_division_chain([2, -1, 0, 3, -4], 2, "exam"), _division_chain([1, 2, -3, 0, 5], -2, "exam")]
    pools["exam"] += [_odd_power_sum(m) for m in (3, 5)]
    for coeffs, a in (
        ([2, -3, 1, -5], 2),
        ([1, -4, 1, 6], 3),
        ([3, 2, -7, 1], 1),
        ([1, 0, -6, 4], -2),
        ([2, 1, -5, 3], 2),
        ([1, 3, -2, -8], -1),
        ([2, -5, 4, -1], 3),
        ([1, 5, 2, -6], -3),
    ):
        pools["apply_easy"].append(_apply_remainder(coeffs, a))
    for coeffs, num, den in (
        ([2, -3, 1, -5], 1, 2),
        ([4, 0, -2, 3], 1, 2),
        ([3, 1, 0, -4], 2, 3),
        ([2, 5, -1, 6], -1, 2),
        ([6, -1, -4, 2], 1, 3),
        ([4, 4, -3, 1], -1, 2),
    ):
        pools["apply_exam"].append(_apply_linear(coeffs, num, den))
    return pools


@functools.lru_cache(maxsize=None)
def pools() -> dict[str, list[dict]]:
    """The families, built on first use (building them runs SymPy, which the package import should not pay for)."""
    return _build_pools()


@generator(
    "cp1_factor_remainder_proof",
    level=14,
    presets={
        "easy": dict(variant="easy"),
        "medium": dict(variant="medium"),
        "hard": dict(variant="hard"),
        "exam": dict(variant="exam"),
    },
)
def cp1_factor_remainder_proof(rng, variant="easy"):
    """`senior-prove-and-apply-the-factor-theorem-and-the`. Every question carries a marking dict (`Question.answer`):
    the working as a chain of equal expressions, the reference chain, the key steps and the traps."""
    if variant not in ("easy", "medium", "hard", "exam"):
        raise ValueError(f"unknown variant: {variant}")
    if variant in ("easy", "exam") and rng.random() < 0.4:
        return kit.apply_item(rng, kit.pick(rng, pools()[f"apply_{variant}"]))
    return kit.prove_item(rng, kit.pick(rng, pools()[variant]))
