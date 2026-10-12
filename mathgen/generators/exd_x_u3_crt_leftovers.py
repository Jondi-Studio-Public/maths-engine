"""Exam-tier source for `ucv1-chinese-remainder-two-congruences`: combining two leftover counts step by step.

A count leaves remainder r1 in groups of m1 and r2 in groups of m2, with m1 and m2 coprime. The practice tiers hand
over the two congruences; here the learner writes the number as r1 + m1 t and has to find t. That takes three
linked steps, one per part: the inverse of m1 modulo m2 (a), the multiplier t (b), then the count itself (c).
Parts (b) and (c) carry `follow_through`, so a wrong inverse in (a) used correctly still earns both.

Built backwards from the count, and checked by brute force over the full period m1 m2.
"""

from __future__ import annotations

import math

from mathgen.answers import Distractor
from mathgen.core import attempts, check, generator, latex_answer, pick_distractors

_SETTINGS = [
    ("A baker in Orange packs rolls", "rolls", "box", "boxes"),
    ("A fruit grower near Mildura packs peaches", "peaches", "tray", "trays"),
    ("A teacher in Alice Springs packs pencils", "pencils", "pack", "packs"),
]


@generator("exd_x_u3_crt_leftovers", level=14, presets={"exam": dict(tier="exam")})
def exd_x_u3_crt_leftovers(rng, tier: str = "exam") -> dict:
    """Exam tier: the inverse of m1 mod m2, the multiplier t, then the least count."""
    check(tier == "exam", f"unknown tier {tier!r}")
    who, thing, one, many = rng.choice(_SETTINGS)
    for _ in attempts(1000, "coprime group sizes with clean, distinct candidate errors"):
        m1, m2 = rng.randint(5, 15), rng.randint(5, 15)
        if m1 == m2 or math.gcd(m1, m2) != 1:
            continue
        r1, r2 = rng.randint(1, m1 - 1), rng.randint(1, m2 - 1)
        diff = (r2 - r1) % m2
        inv = pow(m1, -1, m2)
        t = inv * diff % m2
        n = r1 + m1 * t
        if {inv, t, n} & {m1, m2, r1, r2}:
            continue
        if diff == 0 or t in (0, 1) or inv in (1, m2 - 1) or n == r2 or n <= max(r1, r2):
            continue
        wrong_inv = pow(m2, -1, m1)
        if len({inv, wrong_inv, m1 % m2, m2 - inv}) < 4:
            continue
        if len({t, inv * r2 % m2, diff, inv * (r1 - r2) % m2}) < 4:
            continue
        if len({n, r1 + r2, r1 + m1 * diff, r2, r1}) < 5:
            continue
        break
    solutions = [x for x in range(m1 * m2) if x % m1 == r1 and x % m2 == r2]
    check(solutions == [n], "brute force disagrees with the constructed count")

    stem = (
        f"{who} into {many}. When the {thing} are packed {m1} to a {one}, {r1} {'is' if r1 == 1 else 'are'} left over. "
        f"When the same {thing} are packed {m2} to a {one} instead, {r2} {'is' if r2 == 1 else 'are'} left over. "
        f"The number of {thing} is N = {r1} + {m1}t for some whole number t."
    )

    def tagged(value: int, ident: str | None):
        return Distractor(str(value), ident, value)

    a_cand = [
        tagged(wrong_inv, "inverse-of-wrong-modulus"),
        tagged(m2 - inv, None),
        tagged(m1 % m2, None),
    ]
    b_cand = [
        tagged(inv * r2 % m2, "difference-of-remainders-ignored"),
        tagged(diff, None),
        tagged(inv * (r1 - r2) % m2, "sign-error-moving-term"),
    ]
    c_cand = [
        tagged(r1 + r2, "added-remainders"),
        tagged(r1 + m1 * diff, None),
        tagged(r2, "solved-one-congruence-only"),
        tagged(r1, "solved-one-congruence-only"),
    ]
    a_d = pick_distractors(str(inv), a_cand, 3)
    b_d = pick_distractors(str(t), b_cand, 3)
    c_d = pick_distractors(str(n), c_cand, 3)
    check(len(a_d) >= 2 and len(b_d) >= 2 and len(c_d) >= 2, "too few distractors")

    return dict(
        text=stem,
        parts=[
            dict(
                prompt=f"Find the multiplicative inverse of {m1} modulo {m2}: the whole number k from 1 to {m2 - 1} "
                f"with {m1}k ≡ 1 (mod {m2}).",
                answer_kind="number",
                answer=inv,
                answer_text=str(inv),
                answer_latex=latex_answer(str(inv)),
                marks=2,
                sensible_range=(0, None),
                working=[f"{m1} × {inv} = {m1 * inv} = {m1 * inv // m2} × {m2} + 1, so the inverse is {inv}."],
                distractors=a_d,
            ),
            dict(
                prompt=f"Using your answer to (a), find the smallest whole number t ≥ 0 for which N leaves "
                f"remainder {r2} when divided by {m2}.",
                answer_kind="number",
                answer=t,
                answer_text=str(t),
                answer_latex=latex_answer(str(t)),
                marks=3,
                follow_through=f"({r2 - r1})*part_a - {m2}*floor(({r2 - r1})*part_a/{m2})",
                working=[
                    f"{r1} + {m1}t ≡ {r2} (mod {m2}), so {m1}t ≡ {r2 - r1} ≡ {diff} (mod {m2}).",
                    f"t ≡ {inv} × {diff} = {inv * diff} ≡ {t} (mod {m2}).",
                ],
                distractors=b_d,
            ),
            dict(
                prompt=f"Using your answer to (b), find the smallest possible number of {thing}.",
                answer_kind="number",
                answer=n,
                answer_text=str(n),
                answer_latex=latex_answer(str(n)),
                marks=1,
                sensible_range=(0, None),
                follow_through=f"{r1} + {m1}*part_b",
                working=[f"N = {r1} + {m1} × {t} = {n}. Check: {n} = {n // m2} × {m2} + {n % m2}."],
                distractors=c_d,
            ),
        ],
    )
