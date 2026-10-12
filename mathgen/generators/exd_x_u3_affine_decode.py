"""Exam-tier source for `modular-multiplicative-inverse`: decoding a message scrambled by an affine map.

Whole-number codes 0 to m - 1 are scrambled by C = a P + c (mod m). Decoding needs the inverse of a, then a
shift and a multiplication. The learner has to choose the order of those steps: subtract c first, then multiply
by the inverse. Part (a) finds the inverse; parts (b) and (c) decode two scrambled codes and carry
`follow_through`, so a wrong inverse from (a) used correctly still earns both decoding parts.

Built backwards from the plain codes, then checked against Python's three-argument `pow` inverse.
"""

from __future__ import annotations

import math

from mathgen.answers import Distractor
from mathgen.core import attempts, check, generator, latex_answer, pick_distractors

_CONTEXTS = [
    "A hardware shop in Ballarat scrambles its stock codes, the whole numbers from 0 to {top}, with the rule",
    "A library in Armidale scrambles its shelf codes, the whole numbers from 0 to {top}, with the rule",
    "A scouts group in Rockhampton scrambles the whole numbers from 0 to {top} in its message code with the rule",
]


@generator("exd_x_u3_affine_decode", level=14, presets={"exam": dict(tier="exam")})
def exd_x_u3_affine_decode(rng, tier: str = "exam") -> dict:
    """Exam tier: the inverse of a modulo m, then two codes decoded with it."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(1000, "a unit, shift and codes with distinct candidate errors"):
        m = rng.randint(21, 60)
        a = rng.randint(3, m - 2)
        c = rng.randint(2, m - 2)
        if math.gcd(a, m) != 1:
            continue
        inv = pow(a, -1, m)
        p1, p2 = rng.sample(range(2, m), 2)
        c1, c2 = (a * p1 + c) % m, (a * p2 + c) % m
        if inv in (1, m - 1) or a == inv or len({p1, p2, c1, c2}) < 4 or c1 < c or c2 < c:
            continue
        if len({(inv * c1) % m, (a * (c1 - c)) % m, (c1 - c) % m, ((c1 + c) * inv) % m, p1}) < 5:
            continue
        if len({(inv * c2) % m, (a * (c2 - c)) % m, (c2 - c) % m, ((c2 + c) * inv) % m, p2}) < 5:
            continue
        break
    check(a * inv % m == 1, "inverse does not multiply to 1")
    check((inv * (c1 - c)) % m == p1 and (inv * (c2 - c)) % m == p2, "decoding disagrees with the plain codes")

    stem = (
        _CONTEXTS[rng.randrange(len(_CONTEXTS))].format(top=m - 1)
        + f" C = {a}P + {c} (mod {m}), where P is the plain code and C is the scrambled code."
    )

    def tagged(value: int, ident: str | None):
        return Distractor(str(value), ident, value)

    a_cand = [
        tagged(m - a, "additive-inverse-for-multiplicative"),
        tagged(m - inv, None),
        tagged((inv * 2) % m, None),
    ]

    def decode_cands(code: int):
        return [
            tagged((inv * code) % m, "inverse-not-applied-to-shifted"),
            tagged(((code + c) * inv) % m, "sign-error-moving-term"),
            tagged((a * (code - c)) % m, None),
            tagged((code - c) % m, None),
        ]

    a_d = pick_distractors(str(inv), a_cand, 3)
    b_d = pick_distractors(str(p1), decode_cands(c1), 3)
    c_d = pick_distractors(str(p2), decode_cands(c2), 3)
    check(len(a_d) >= 2 and len(b_d) >= 3 and len(c_d) >= 3, "too few distractors")

    def ft(code: int) -> str:
        shifted = f"({code - c})*part_a"
        return f"{shifted} - {m}*floor(({shifted})/{m})"

    return dict(
        text=stem,
        parts=[
            dict(
                prompt=f"Find the multiplicative inverse of {a} modulo {m}: the whole number k from 1 to {m - 1} "
                f"with {a}k ≡ 1 (mod {m}).",
                answer_kind="number",
                answer=inv,
                answer_text=str(inv),
                answer_latex=latex_answer(str(inv)),
                marks=2,
                sensible_range=(0, None),
                working=[f"{a} × {inv} = {a * inv} = {a * inv // m} × {m} + 1, so the inverse is {inv}."],
                distractors=a_d,
            ),
            dict(
                prompt=f"Using your answer to (a), find the plain code P that was scrambled to C = {c1}.",
                answer_kind="number",
                answer=p1,
                answer_text=str(p1),
                answer_latex=latex_answer(str(p1)),
                marks=3,
                follow_through=ft(c1),
                working=[
                    f"C - {c} ≡ {a}P, so P ≡ {inv}({c1} - {c}) = {inv} × {c1 - c} (mod {m}).",
                    f"{inv * (c1 - c)} = {inv * (c1 - c) // m} × {m} + {p1}, so P = {p1}.",
                ],
                distractors=b_d,
            ),
            dict(
                prompt=f"Using your answer to (a), find the plain code P that was scrambled to C = {c2}.",
                answer_kind="number",
                answer=p2,
                answer_text=str(p2),
                answer_latex=latex_answer(str(p2)),
                marks=2,
                follow_through=ft(c2),
                working=[f"P ≡ {inv}({c2} - {c}) = {inv} × {c2 - c} ≡ {p2} (mod {m})."],
                distractors=c_d,
            ),
        ],
    )
