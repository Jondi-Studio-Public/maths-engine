"""Exam-tier source for `hypergeometric-probability`: an acceptance-sampling rule applied to a sample drawn without
replacement.

A batch of items contains some faulty ones. An inspector samples a few at random, without replacement, and the batch
passes only if the sample holds at most one faulty item. The learner

1. finds the probability that the sample has no faulty item,
2. finds the probability that it has exactly one, and
3. uses both, with the rule, to find the probability that the batch is rejected.

The method is the learner's to choose: the rule has to be read to see which outcomes matter, and the final part
needs the two earlier results (or the complement of their sum).
"""

from __future__ import annotations

import random
from fractions import Fraction
from math import comb

import sympy

from ..answers import exact_value
from ..core import attempts, check, generator, latex_answer, pick_distractors, rational_text
from .exd_x_u6_support import keys_clear, tagged, within_unit_interval

_ITEMS = [
    ("A carton of {N} phone chargers contains {D} that are faulty.", "chargers", "carton"),
    ("A box of {N} bicycle lights holds {D} that are faulty.", "lights", "box"),
    ("A crate of {N} cricket balls includes {D} with a flaw in the seam.", "balls", "crate"),
    ("A tray of {N} solar-panel connectors contains {D} that are defective.", "connectors", "tray"),
]
_RULES = [
    "The {unit} is accepted if the sample contains at most one faulty item.",
    "The {unit} passes only when no more than one item in the sample is faulty.",
    "The {unit} is accepted when fewer than two of the sampled items are faulty.",
]


def _build(rng: random.Random):
    N = rng.randint(10, 24)
    D = rng.randint(3, 7)
    n = rng.randint(3, 5)
    total = comb(N, n)
    a = Fraction(comb(N - D, n), total)
    b = Fraction(D * comb(N - D, n - 1), total)
    c = 1 - a - b
    if not (0 < a < 1 and 0 < b < 1 and 0 < c < 1) or len({a, b, c}) < 3:
        return None
    # The follow-through machinery restates a result with `nsimplify`, which fails for a few rationals with a large
    # denominator; leave those out so every part can be followed through.
    if not sympy.nsimplify(sympy.Rational(c.numerator, c.denominator)).is_Rational:
        return None
    check(c == Fraction(sum(comb(D, k) * comb(N - D, n - k) for k in range(2, min(D, n) + 1)), total))

    setting, noun, unit = rng.choice(_ITEMS)
    stem = (
        f"{setting.format(N=N, D=D)} An inspector chooses {n} {noun} at random, without replacement. "
        f"{rng.choice(_RULES).format(unit=unit)}"
    )
    p1 = "Find the probability that the sample contains no faulty items."
    p2 = "Find the probability that the sample contains exactly one faulty item."
    p3 = f"Hence find the probability that the {unit} is rejected."
    if not keys_clear(" ".join([stem, p1, p2, p3]), [a, b, c]):
        return None

    p_bad, p_ok = Fraction(D, N), Fraction(N - D, N)
    a_wr = p_ok**n
    a_stale = Fraction(1)
    for i in range(n):
        a_stale *= Fraction(N - D - i, N)
    b_wr = n * p_bad * p_ok ** (n - 1)
    b_one_order = p_bad
    for i in range(n - 1):
        b_one_order *= Fraction(N - D - i, N - 1 - i)
    a_d = pick_distractors(
        rational_text(a),
        within_unit_interval(
            [
                tagged(rational_text(a_wr), "uses-with-replacement-model", a_wr),
                tagged(rational_text(a_stale), "forgot-to-update-total", a_stale),
                tagged(rational_text(1 - a), "gave-complement-probability", 1 - a),
            ]
        ),
        3,
    )
    b_d = pick_distractors(
        rational_text(b),
        within_unit_interval(
            [
                tagged(rational_text(b_wr), "uses-with-replacement-model", b_wr),
                tagged(rational_text(b_one_order), "omits-combination-factor", b_one_order),
                tagged(rational_text(1 - b), "gave-complement-probability", 1 - b),
            ]
        ),
        3,
    )
    c_d = pick_distractors(
        rational_text(c),
        within_unit_interval(
            [
                tagged(rational_text(a + b), "gave-complement-probability", a + b),
                tagged(rational_text(1 - a), "ignored-acceptance-rule", 1 - a),
                tagged(rational_text(1 - b), "ignored-acceptance-rule", 1 - b),
            ]
        ),
        3,
    )
    if min(len(a_d), len(b_d), len(c_d)) < 2:
        return None

    def part(prompt, value, marks, distractors, working, follow_through=None):
        entry = dict(
            prompt=prompt,
            answer_kind="fraction",
            answer=exact_value(value),
            answer_text=rational_text(value),
            answer_latex=latex_answer(rational_text(value)),
            marks=marks,
            sensible_range=(0, 1),
            working=working,
            distractors=distractors,
        )
        if follow_through:
            entry["follow_through"] = follow_through
        return entry

    return dict(
        text=stem,
        parts=[
            part(
                p1,
                a,
                2,
                a_d,
                [f"C({N - D},{n}) / C({N},{n}) = {comb(N - D, n)}/{total} = {rational_text(a)}."],
            ),
            part(
                p2,
                b,
                2,
                b_d,
                [f"{D} × C({N - D},{n - 1}) / C({N},{n}) = {D * comb(N - D, n - 1)}/{total} = {rational_text(b)}."],
            ),
            part(
                p3,
                c,
                2,
                c_d,
                [
                    "Rejected means two or more faulty items.",
                    f"1 - {rational_text(a)} - {rational_text(b)} = {rational_text(c)}.",
                ],
                follow_through="1 - part_a - part_b",
            ),
        ],
    )


@generator("exd_x_u6_hypergeometric_acceptance", level=14, presets={"exam": dict(tier="exam")})
def exd_x_u6_hypergeometric_acceptance(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: sampling without replacement against an acceptance rule, as a three-part chain."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(1000, "a batch with enough distinct distractors"):
        question = _build(rng)
        if question is not None:
            return question
