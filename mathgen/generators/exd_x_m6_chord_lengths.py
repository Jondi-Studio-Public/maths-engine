"""Exam question: two chords of a circle meet inside it, with the parts of one chord given in terms of x.

A three-part chain. Part (a) sets up and solves the quadratic that the intersecting chords theorem gives
(AP x PB = CP x PD) and rejects the negative root, part (b) uses x to find the whole length of chord AB, and
part (c) compares AB with the other chord. Parts (b) and (c) carry `follow_through`.

Two stem forms: one part is x and the other is a multiple of x plus a constant, or both parts are x plus a
constant. Built backwards from a whole-number x, so the product is a whole number, and CP and PD are drawn as a
factor pair of that product, so every key is an integer.
"""

from __future__ import annotations

import random
from fractions import Fraction

from mathgen.core import attempts, check, generator

from .exd_x_m6_support import TooFewDistractors, number_part, shows_no_key, tagged


def _linear(coeff: int, const: int) -> str:
    lead = "x" if coeff == 1 else f"{coeff}x"
    return f"{lead} + {const}"


@generator("exd_x_m6_chord_lengths", level=12, presets={"exam": dict(tier="exam")})
def exd_x_m6_chord_lengths(rng: random.Random, tier: str = "exam") -> dict:
    """Solve for x from intersecting chords, then find a chord length and compare two chords."""
    check(tier == "exam", "exd_x_m6_chord_lengths: only the exam tier exists")
    for _ in attempts(1000, "a pair of chords whose figures work out exactly"):
        try:
            question = _attempt(rng)
        except TooFewDistractors:
            continue
        if question is not None:
            return question


def _attempt(rng: random.Random) -> dict | None:
    x = rng.randint(2, 12)
    form = rng.choice(("multiple", "shifted"))
    if form == "multiple":
        m, k = rng.choice((1, 2, 3)), rng.randint(1, 8)
        ap_text, pb_text = "x", _linear(m, k)
        pb = m * x + k
        ap = x
        p_off, q_off = 0, k  # AP = 1x + 0, PB = m x + k
        ap_coeff, pb_coeff = 1, m
        whole_ft = f"{1 + m}*part_a + {k}"
    else:
        p_off, q_off = rng.randint(1, 6), rng.randint(1, 8)
        if p_off == q_off:
            return None
        ap_coeff = pb_coeff = 1
        ap_text, pb_text = _linear(1, p_off), _linear(1, q_off)
        ap, pb = x + p_off, x + q_off
        whole_ft = f"2*part_a + {p_off + q_off}"
    product = ap * pb
    pairs = [(a, product // a) for a in range(2, 61) if product % a == 0 and 2 <= product // a <= 60]
    pairs = [(a, b) for a, b in pairs if a != b]
    if not pairs:
        return None
    cp, pd = rng.choice(pairs)
    chord_ab, chord_cd = ap + pb, cp + pd
    if chord_ab == chord_cd:
        return None
    check(ap_coeff * x + p_off == ap and pb_coeff * x + q_off == pb and ap * pb == cp * pd, "chords: figures disagree")
    # The other root of the quadratic is negative, so x is the only usable solution.
    other_root = -Fraction(q_off if form == "multiple" else p_off + q_off, m if form == "multiple" else 1) - x
    check(other_root < 0, "chords: second root should be negative")

    stem = (
        f"Two chords AB and CD of a circle intersect at a point P inside the circle. "
        f"AP = ({ap_text}) cm, PB = ({pb_text}) cm, CP = {cp} cm and PD = {pd} cm."
    ).replace("(x)", "x")
    prompts = (
        "Find the value of x.",
        "Hence find the length of the chord AB, in centimetres.",
        "By how many centimetres do the lengths of the two chords differ?",
    )
    if not shows_no_key([stem, *prompts], [x, chord_ab, abs(chord_ab - chord_cd)]):
        return None

    if form == "multiple":
        sum_rule = Fraction(cp + pd - k, 1 + m)
        crossed = Fraction(k * pd, cp - m * pd) if cp != m * pd else None
    else:
        sum_rule = Fraction(cp + pd - p_off - q_off, 2)
        crossed = Fraction(q_off * pd - p_off * cp, cp - pd)
    a_cands = [
        tagged(sum_rule, "chord-segments-added"),
        tagged(crossed, "chord-products-crossed") if crossed is not None else None,
        tagged(other_root, "negative-root-as-length", allow_negative=True),
    ]
    b_cands = [
        tagged(Fraction(pb), "segment-for-whole-chord"),
        tagged(Fraction(ap), "segment-for-whole-chord"),
        tagged(Fraction(product)),  # reports the product that was set up
    ]
    diff = abs(chord_ab - chord_cd)
    c_cands = [
        tagged(Fraction(chord_ab + chord_cd)),  # adds the lengths
        tagged(Fraction(chord_ab)),  # reports one chord
        tagged(Fraction(chord_cd)),
    ]
    return dict(
        text=stem,
        parts=[
            number_part(
                prompts[0],
                Fraction(x),
                a_cands,
                marks=2,
                working=[
                    "Intersecting chords: AP × PB = CP × PD.",
                    f"({ap_text})({pb_text}) = {cp} × {pd} = {product}; the positive solution is x = {x}.",
                ],
            ),
            number_part(
                prompts[1],
                Fraction(chord_ab),
                b_cands,
                marks=1,
                follow_through=whole_ft,
                working=[f"AB = AP + PB = {ap} + {pb} = {chord_ab} cm."],
            ),
            number_part(
                prompts[2],
                Fraction(diff),
                c_cands,
                marks=1,
                follow_through=f"Abs(({whole_ft}) - {chord_cd})",
                working=[f"CD = {cp} + {pd} = {chord_cd} cm.", f"Difference = |{chord_ab} - {chord_cd}| = {diff} cm."],
            ),
        ],
    )
