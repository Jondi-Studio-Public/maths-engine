"""Exam-tier source for `f10-exponent-laws-positive-zero`: powers of different bases rewritten over one base.

The practice tiers apply the product and quotient laws to powers that already share a base. An exam question
hides the common base: `8^3 x 4^2` only combines once both are written as powers of 2. A zero power in the
expression (any non-zero base to the power 0 is 1) is a further trap. Part (a) asks for the index `k` when the
expression is written as `c x b^k`; part (b) asks for the value, following through from the learner's `k`.

Built backwards from the base, the indices of the two powers and the divisor, so `k` is exact. The printed
expression is re-evaluated with exact integers and fractions, a different route from the index arithmetic.
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..core import check, draw_until, generator, latex_answer, pick_distractors

_FAMILIES = {2: (2, 3, 4, 5), 3: (2, 3, 4), 5: (2, 3)}  # base -> indices e where b^e is the larger written base
_ZERO_BASES = [7, 11, 13, 6, 9, 10, 12, 15]


def _power_text(base: int, exp: int) -> str:
    return f"{base}^{exp}"


@generator("exd_x_n3_common_base_powers", level=8, presets={"exam": dict(tier="exam")})
def exd_x_n3_common_base_powers(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: rewrite powers of 4, 8, 9, 25 ... over one base, then find the index and the value."""
    check(tier == "exam", f"unknown tier {tier!r}")

    def draw(r):
        b = r.choice([2, 2, 2, 3, 5])
        e1, e2 = r.choice(_FAMILIES[b]), r.choice(_FAMILIES[b])
        p, q, d = r.randint(2, 5), r.randint(1, 4), r.randint(1, 6)
        c = r.choice([2, 3, 4, 5, 6, 7, 9, 10, 11, 12])
        z = r.choice(_ZERO_BASES)
        return b, e1, e2, p, q, d, c, z

    def accept(t):
        b, e1, e2, p, q, d, c, z = t
        k = e1 * p + e2 * q - d
        return e1 != e2 and 2 <= k <= {2: 18, 3: 10, 5: 7}[b] and z != b

    b, e1, e2, p, q, d, c, z = draw_until(rng, draw, accept, what="bases and indices")
    k = e1 * p + e2 * q - d
    shown = f"{c} × {_power_text(b**e1, p)} × {_power_text(b**e2, q)} ÷ {_power_text(b, d)} × {_power_text(z, 0)}"
    exact = Fraction(c) * Fraction(b**e1) ** p * Fraction(b**e2) ** q / Fraction(b) ** d * Fraction(z) ** 0
    value = c * b**k
    check(exact == value, "printed expression disagrees with the value")
    check(exact.denominator == 1, "value is not an integer")

    a_text = str(k)
    a_cands = [
        e1 * p + e2 * q + d,  # divides by adding the index
        p + q - d,  # indices combined without writing the bases over one base
        e1 * p + e2 * q,  # ignores the divisor
        e1 * p * e2 * q - d,  # multiplies the indices of the two powers
    ]
    a_dist = pick_distractors(a_text, [str(v) for v in a_cands if v > 0], 3)
    b_text = str(value)
    b_cands = []
    for kk in (e1 * p + e2 * q + d, p + q - d, e1 * p + e2 * q):
        if kk > 0:
            b_cands.append(str(c * b**kk))
    b_cands += [str((c * b) ** k), str(c + b**k), str(b**k)]
    b_dist = pick_distractors(b_text, b_cands, 3)
    check(len(a_dist) >= 2 and len(b_dist) >= 2, "too few distractors")

    return dict(
        text=rng.choice(["Let N = {t}.", "Consider the number N = {t}.", "A calculation gives N = {t}."]).format(
            t=shown
        ),
        parts=[
            dict(
                prompt=f"Write N in the form {c} × {b}^k and find the whole number k.",
                answer_kind="number",
                answer=k,
                answer_text=a_text,
                answer_latex=latex_answer(a_text),
                marks=3,
                working=[
                    f"{b**e1} = {b}^{e1} and {b**e2} = {b}^{e2}, and {z}^0 = 1.",
                    f"Indices: {e1} × {p} + {e2} × {q} − {d} = {k}.",
                ],
                distractors=a_dist,
            ),
            dict(
                prompt="Hence find the value of N.",
                answer_kind="number",
                answer=value,
                answer_text=b_text,
                answer_latex=latex_answer(b_text),
                marks=1,
                sensible_range=(0, None),
                follow_through=f"{c}*{b}**part_a",
                working=[f"N = {c} × {b}^{k} = {b_text}."],
                distractors=b_dist,
            ),
        ],
    )
