"""Exam tier for "evaluate powers".

The practice tiers evaluate one power of an integer. Here a quantity is multiplied by the same whole number in
every period, so the learner must see that the amount after h periods is a starting amount times a power, evaluate
it, and then keep using powers to compare later periods. Three parts:

(a) the amount after h periods (2 marks);
(b) the increase during the next period (2 marks, follow-through from (a));
(c) the amount after k further periods (2 marks, follow-through from (a)).

The growth factor is 2, 3 or 4 and each context has its own size limit. Everything is an exact integer.
"""

from __future__ import annotations

from mathgen.answers import Distractor
from mathgen.core import check, draw_until, generator, latex_answer, phrase, pick_distractors

# (thing, period, scenario sentence with {c} {b}, question wording for the amount, growth factors, largest total).
# Each context is one where repeated multiplication really happens, with a factor and a size limit that fit it.
_SCENES = (
    (
        "bacteria",
        "hour",
        "A culture in a Brisbane laboratory starts with {c} bacteria. The number of bacteria is multiplied by {b} "
        "every hour.",
        "How many bacteria are in the culture after {h} hours?",
        (3, 4),
        5_000_000,
    ),
    (
        "cells",
        "hour",
        "A sample in a Perth research lab starts with {c} cells. The number of cells is multiplied by {b} "
        "every hour as the cells divide.",
        "How many cells are in the sample after {h} hours?",
        (2, 3),
        2_000_000,
    ),
    (
        "infected computers",
        "day",
        "A computer virus is found on {c} computers in a large company's network. From then on the number of infected computers is "
        "multiplied by {b} at the end of each day as it spreads through email.",
        "How many computers are infected after {h} days?",
        (2, 3),
        40_000,
    ),
    (
        "people",
        "hour",
        "A chain message is first sent to {c} people in Hobart. The number of people who have received it is "
        "multiplied by {b} every hour as each person forwards it.",
        "How many people have received the message after {h} hours?",
        (3, 4),
        200_000,
    ),
)
_STARTS = (2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30, 40, 50)


def _draw(rng, factors):
    c = rng.choice(_STARTS)
    b = rng.choice(factors)
    h = rng.randint(3, 8)
    k = rng.choice((2, 3))
    return c, b, h, k


def _acceptable(d, largest) -> bool:
    c, b, h, k = d
    return 200 <= c * b**h <= largest // 4 and c * b ** (h + k) <= largest


@generator("exd_x_n2_powers_growth", level=10, presets={"exam": dict(tier="exam")})
def exd_x_n2_powers_growth(rng, tier="exam"):
    """Repeated multiplication as a power, then increase and later amounts; see the module docstring."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    thing, period, scene, ask, factors, largest = rng.choice(_SCENES)
    c, b, h, k = draw_until(
        rng,
        lambda r: _draw(r, factors),
        lambda d: _acceptable(d, largest),
        what="a growth chain of moderate size",
    )
    a = c * b**h
    inc = a * (b - 1)
    later = a * b**k
    check(inc == c * b ** (h + 1) - a and later == c * b ** (h + k), "chain disagrees with the closed form")
    stem = phrase(rng, (scene,), c=c, b=b)
    ask = ask.format(h=h)

    def part(prompt, key, marks, work, cands, ft=None):
        shown = [Distractor(str(v), tag, v) if tag else str(v) for v, tag in cands]
        out = dict(
            prompt=prompt,
            answer_kind="number",
            answer=key,
            answer_text=str(key),
            answer_latex=latex_answer(str(key)),
            marks=marks,
            sensible_range=(0, None),
            working=work,
            distractors=pick_distractors(str(key), shown, 3),
        )
        if ft:
            out["follow_through"] = ft
        return out

    parts = [
        part(
            ask,
            a,
            2,
            [f"After {h} {period}s: {c} × {b}^{h} = {c} × {b**h} = {a}."],
            [
                (c * b * h, "multiplied-base-exponent"),
                ((c * b) ** h, "exponent-applied-to-coefficient"),
                (c * b ** (h - 1), "off-by-one-power"),
                (b**h, "dropped-starting-amount"),
            ],
        ),
        part(
            f"By how many {thing} does the number increase during the next {period}, after your answer to (a)?",
            inc,
            2,
            [f"New total = {a} × {b} = {a * b}; increase = {a * b} - {a} = {inc}."],
            [
                (a * b, "gave-new-total-not-increase"),
                (b**h * (b - 1), "dropped-starting-amount"),
                (a * b - c, None),
            ],
            ft=f"part_a*{b - 1}",
        ),
        part(
            f"How many {thing} will there be after a further {k} {period}s, starting from your answer to (a)?",
            later,
            2,
            [f"Further growth is × {b}^{k} = × {b**k}: {a} × {b**k} = {later}."],
            [
                (a * b * k, "multiplied-base-exponent"),
                (a * b, "one-step-for-two"),
                (a + b**k, "added-instead-of-multiplied-by-power"),
                (c * b**k, None),
            ],
            ft=f"part_a*{b**k}",
        ),
    ]
    return dict(text=stem, parts=parts)
