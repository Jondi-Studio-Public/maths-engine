"""Exam tier for solving trigonometric equations: a quadratic in sin(x) or cos(x).

The practice tiers are always ``f(a(x - b)) = c``. Here the equation is a quadratic in ``sin(x)`` or ``cos(x)``,
so the learner must choose to treat it as a quadratic, factorise, discard any factor whose value lies outside
[-1, 1], and then solve what is left over 0 <= x < 360. Built forwards from the two values of the trig function.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

from ..core import check, generator, join_terms, pick_distractors

_HALF = Fraction(1, 2)
# value of the trig function -> solutions in [0, 360), by function
_SOLUTIONS = {
    "sin": {Fraction(1): (90,), -Fraction(1): (270,), _HALF: (30, 150), -_HALF: (210, 330), Fraction(0): (0, 180)},
    "cos": {Fraction(1): (0,), -Fraction(1): (180,), _HALF: (60, 300), -_HALF: (120, 240), Fraction(0): (90, 270)},
}
_VALID = list(_SOLUTIONS["sin"])
_EXTRANEOUS = [Fraction(2), -Fraction(2), Fraction(3, 2), -Fraction(3, 2), Fraction(3), -Fraction(3)]


def _poly_coeffs(r1: Fraction, r2: Fraction) -> tuple[int, int, int]:
    """Integer (A, B, C) with A s^2 + B s + C = A (s - r1)(s - r2), A > 0, in lowest terms."""
    den = r1.denominator * r2.denominator
    a = den
    b = -(r1 + r2) * den
    c = r1 * r2 * den
    g = math.gcd(math.gcd(int(a), int(b)), int(c))
    return int(a) // g, int(b) // g, int(c) // g


def _fmt_set(values) -> str:
    parts = [f"{v}°" for v in sorted(values)]
    if len(parts) == 1:
        return f"x = {parts[0]}"
    return f"x = {', '.join(parts[:-1])} and {parts[-1]}"


@generator("exd_x_f5_trigquad", level=14, presets={"exam": dict(tier="exam")})
def exd_x_f5_trigquad(rng: random.Random, tier: str = "exam") -> dict:
    """Solve a quadratic in sin(x) or cos(x) for 0 <= x < 360 (degrees)."""
    if tier != "exam":
        raise ValueError(f"unknown tier: {tier}")
    fn = rng.choice(["sin", "cos"])
    other_fn = "cos" if fn == "sin" else "sin"
    while True:
        while True:
            r1 = rng.choice(_VALID)
            r2 = rng.choice(_VALID + _EXTRANEOUS * 2)
            if r1 == r2:
                continue
            found = {d for r in (r1, r2) if r in _SOLUTIONS[fn] for d in _SOLUTIONS[fn][r]}
            if len(found) >= 2:
                break
        a, b, c = _poly_coeffs(r1, r2)
        power = f"{fn}^2(x)"
        lin = f"{fn}(x)"
        if rng.random() < 0.5:
            eq = f"{join_terms([(a, power), (b, lin), (c, '')])} = 0"
        else:
            # one term moved across the equals sign
            eq = f"{join_terms([(a, power), (b, lin)])} = {-c}"
        text = f"Find all solutions of {eq} for 0 <= x < 360, where x is in degrees."

        valid_roots = [r for r in (r1, r2) if r in _SOLUTIONS[fn]]
        check(len(valid_roots) >= 1, "no admissible root")
        solutions = sorted({d for r in valid_roots for d in _SOLUTIONS[fn][r]})
        # independent check: substitute every integer degree and compare with the constructed set
        trig = math.sin if fn == "sin" else math.cos
        brute = [d for d in range(360) if abs(a * trig(math.radians(d)) ** 2 + b * trig(math.radians(d)) + c) < 1e-9]
        check(brute == solutions, "constructed solutions disagree with brute force")
        answer_text = _fmt_set(solutions)

        sign_flipped = sorted({d for r in (r1, r2) if -r in _SOLUTIONS[fn] for d in _SOLUTIONS[fn][-r]})
        principal_only = sorted({_SOLUTIONS[fn][r][0] for r in valid_roots})
        wrong_function = sorted({d for r in valid_roots for d in _SOLUTIONS[other_fn][r]})
        # divides both sides by the trig function, which silently loses the root where it equals 0
        divided_out = sorted({d for r in valid_roots if r != 0 for d in _SOLUTIONS[fn][r]})
        cands = []
        for s in (divided_out, sign_flipped, principal_only, wrong_function):
            if s:
                cands.append(_fmt_set(s))
        distractors = pick_distractors(answer_text, cands, n=3)
        if len(distractors) == 3:
            break
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer_text,
        answer_text=answer_text,
        answer_latex=f"plain:{answer_text}",
        working=[
            f"Factorise as a quadratic in {fn}(x): the values are {fn}(x) = {r1} and {fn}(x) = {r2}.",
            f"Only values in [-1, 1] have solutions, so {answer_text}.",
        ],
        distractors=distractors,
    )
