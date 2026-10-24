"""Exam tier for solving a simple polynomial equation with complex roots.

The practice tiers solve a monic quadratic whose roots are ``p +- mi`` with
integer ``p`` and ``m``. The exam tier uses a leading coefficient ``a`` of 2
to 5, so the roots are ``(P +- Mi) / a`` and the real and imaginary parts are
usually fractions, and (in the ``rearranged`` form) the equation is given with
terms on both sides so the learner must collect it into standard form before
applying the quadratic formula.

Built backwards from the root ``(P + Mi) / a``: the quadratic is
``a z^2 - 2P z + (P^2 + M^2) / a = 0``, so ``a`` must divide ``P^2 + M^2``.
The key is the root with positive imaginary part, written ``p + qi``.
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..core import check, generator, join_terms, paren_negative, pick_distractors


def _part(value: Fraction, unit: str = "") -> str:
    """One signed-free component such as ``3``, ``3/2``, ``i``, ``3i/2``."""
    n, d = abs(value.numerator), value.denominator
    if unit:
        head = unit if n == 1 else f"{n}{unit}"
        return head if d == 1 else f"{head}/{d}"
    return str(n) if d == 1 else f"{n}/{d}"


def _complex_text(re_: Fraction, im: Fraction) -> str:
    """``p + qi`` with fractional parts, e.g. ``-3/2 + 5i/2`` or ``1/2 - i``."""
    imag = _part(abs(im), "i")
    if re_ == 0:
        return imag if im > 0 else f"-{imag}"
    real = ("-" if re_ < 0 else "") + _part(re_)
    return f"{real} {'+' if im > 0 else '-'} {imag}"


def _pick_root(rng: random.Random):
    while True:
        a = rng.choice([2, 3, 4, 5])
        big_p = rng.randint(-9, 9)
        big_m = rng.randint(1, 9)
        if (big_p * big_p + big_m * big_m) % a != 0:
            continue
        if big_p % a == 0 and big_m % a == 0:
            continue  # a whole-number root: no longer a change from the practice tiers
        if big_p == 0:
            continue
        return a, big_p, big_m


def _build(rng: random.Random, form: str) -> dict:
    a, big_p, big_m = _pick_root(rng)
    b = -2 * big_p
    c = (big_p * big_p + big_m * big_m) // a
    p, m = Fraction(big_p, a), Fraction(big_m, a)
    # Independent check: substitute the root into the displayed quadratic with exact arithmetic on
    # (real, imag) pairs.
    zr, zi = p, m
    sq_r, sq_i = zr * zr - zi * zi, 2 * zr * zi
    check(a * sq_r + b * zr + c == 0 and a * sq_i + b * zi == 0, "root does not satisfy the quadratic")

    lhs = join_terms([(a, "z²"), (b, "z"), (c, "")])
    if form == "standard":
        eq = f"{lhs} = 0"
        working = [f"a = {a}, b = {b}, c = {c}"]
    else:
        # Split the linear and constant parts across the equals sign, with an extra z term and constant on
        # the right that must be collected.
        e = rng.choice([v for v in range(-4, 5) if v != 0])
        f = rng.choice([v for v in range(-6, 7) if v != 0])
        left = join_terms([(a, "z²"), (b + e, "z"), (c + f, "")])
        right = join_terms([(e, "z"), (f, "")])
        eq = f"{left} = {right}"
        working = [f"Collect on the left: {lhs} = 0", f"a = {a}, b = {b}, c = {c}"]
    working += [
        f"z = ({-b} ± √({paren_negative(b)}² - 4({a})({c}))) / {2 * a}",
        f"= ({big_p} ± {big_m}i) / {a}",
    ]
    answer_text = _complex_text(p, m)
    # Distractors: each a named slip.
    conj = _complex_text(p, -m)
    sign = _complex_text(-p, m)
    only_real_divided = _complex_text(p, Fraction(big_m))  # divides the real part by a but not the imaginary
    no_shift = _complex_text(Fraction(0), m)  # drops the real part
    distractors = pick_distractors(answer_text, [conj, sign, only_real_divided, no_shift], n=3)
    check(len(distractors) == 3, "too few distractors")
    return dict(
        text=f"Solve {eq} for z. State the root with positive imaginary part in the form a + bi, "
        "giving a and b as exact fractions where needed.",
        answer=answer_text,
        answer_text=answer_text,
        working=working,
        distractors=distractors,
    )


@generator("exd_x_f3_polyeq", level=12, presets={"exam": dict(tier="exam")})
def exd_x_f3_polyeq(rng: random.Random, tier: str = "exam", form: str | None = None) -> dict:
    """Non-monic quadratic (optionally rearranged) with non-real roots; give the root with Im > 0."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return _build(rng, form or rng.choice(["standard", "rearranged"]))
