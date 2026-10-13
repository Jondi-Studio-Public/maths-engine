"""Exam tier for "operate with fractions".

The practice tiers combine fractions with one or more operations written out as a sum. Here the fractions are
buried in a story: a first fraction of a whole is used, then a second fraction of *what is left*, and the amount
that remains is known. The learner has to see that the second fraction is of the remainder, combine the two, and
then work backwards from the known remainder. Three parts:

(a) the fraction of the whole left over (2 marks, a fraction);
(b) the size of the original whole (2 marks, follow-through from (a));
(c) the amount used in the second step (2 marks, follow-through from (b)).

Built backwards from the whole, chosen as a multiple of every denominator that appears, so each amount is an
exact whole number.
"""

from __future__ import annotations

import math
from fractions import Fraction

from mathgen.answers import Distractor
from mathgen.core import (
    check,
    draw_until,
    generator,
    latex_answer,
    phrase,
    pick_distractors,
)
from mathgen.slots import NAMES

_FRACTIONS = tuple(Fraction(a, b) for b in (3, 4, 5, 6, 8) for a in range(1, b) if math.gcd(a, b) == 1)

# (prefix, suffix) around an amount, then the three scenario sentences.
_SCENES = (
    dict(
        pre="$",
        post="",
        stem=(
            "{name} saved some money. {name} spent {f1} of the savings on a bike helmet, then {f2} of what was left "
            "on a drink bottle. After that, {name} had {pre}{x}{post} left.",
            "{name} had some pocket money. {f1} of it went on a book, and then {f2} of the money that remained "
            "was spent on lunch. {name} was left with {pre}{x}{post}.",
        ),
        ask_whole="How much money did {name} have to begin with, in dollars?",
        ask_second="How much was spent in the second purchase, in dollars?",
    ),
    dict(
        pre="",
        post=" litres",
        stem=(
            "A full water tank on a farm near Dubbo is used in two stages. {f1} of the water is used on the garden, "
            "then {f2} of the water still in the tank is used for the stock. The tank then holds{post_x}.",
        ),
        ask_whole="How many litres did the tank hold when it was full?",
        ask_second="How many litres were used for the stock?",
    ),
    dict(
        pre="",
        post=" metres",
        stem=(
            "A tailor in Ballarat cuts {f1} of a roll of fabric for curtains, then {f2} of the fabric still on the "
            "roll for cushions. {x} metres of fabric are left on the roll.",
        ),
        ask_whole="How many metres of fabric were on the roll at the start?",
        ask_second="How many metres were cut for cushions?",
    ),
)


def _frac_text(f: Fraction) -> str:
    return f"{f.numerator}/{f.denominator}"


def _draw(rng):
    f1, f2 = rng.choice(_FRACTIONS), rng.choice(_FRACTIONS)
    if f1 == f2:
        return None
    rest = 1 - f1
    left = rest * (1 - f2)
    lcm = math.lcm(rest.denominator, left.denominator, f1.denominator, f2.denominator)
    whole = lcm * rng.randint(1, 40)
    if not 120 <= whole <= 1500:
        return None
    return f1, f2, Fraction(whole)


@generator("exd_x_n2_fraction_remainders", level=10, presets={"exam": dict(tier="exam")})
def exd_x_n2_fraction_remainders(rng, tier="exam"):
    """A fraction of a fraction left over, then the whole from the remainder; see the module docstring."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    f1, f2, whole = draw_until(rng, _draw, lambda d: d is not None, what="two fractions with a whole-number chain")
    rest = 1 - f1
    left_frac = rest * (1 - f2)
    left = whole * left_frac
    second = whole * rest * f2
    check(left.denominator == 1 and second.denominator == 1 and whole * left_frac == left, "amounts are not whole")
    scene = rng.choice(_SCENES)
    name = rng.choice(NAMES)
    x = int(left)
    stem = phrase(
        rng,
        scene["stem"],
        name=name,
        f1=_frac_text(f1),
        f2=_frac_text(f2),
        x=f"{x:,}",
        pre=scene["pre"],
        post=scene["post"],
        post_x=f" {x:,} litres",
    )

    def frac_part():
        key = _frac_text(left_frac)
        cands = [
            Distractor(_frac_text(1 - f1 - f2), "second-fraction-of-original", 1 - f1 - f2)
            if 1 - f1 - f2 > 0
            else None,
            Distractor(_frac_text(f1 + f2), "added-fractions-spent-as-remaining", f1 + f2) if f1 + f2 < 1 else None,
            Distractor(_frac_text(rest), "forgot-second-purchase", rest),
            Distractor(_frac_text(1 - f2), "ignored-second-fraction", 1 - f2),
        ]
        return dict(
            prompt="What fraction of the original amount was left? Give the answer as a fraction in simplest form.",
            answer_kind="fraction",
            answer={"p": left_frac.numerator, "q": left_frac.denominator},
            answer_text=key,
            answer_latex=latex_answer(key),
            marks=2,
            sensible_range=(0, 1),
            working=[
                f"After the first use, {_frac_text(rest)} is left.",
                f"The second use takes {_frac_text(f2)} of that, so {_frac_text(1 - f2)} of it is left: "
                f"{_frac_text(rest)} × {_frac_text(1 - f2)} = {key}.",
            ],
            distractors=pick_distractors(key, [c for c in cands if c], 3),
        )

    def num_part(prompt, key, marks, work, cands, ft):
        # The key is a whole number, so every option is too: a wrong value that is not whole (39.38 against 280)
        # would give the key away by its format, and rounding it would misreport what the slip produces.
        shown = [Distractor(str(int(v)), tag, int(v)) if tag else str(int(v)) for v, tag in cands if v.denominator == 1]
        return dict(
            prompt=prompt,
            answer_kind="number",
            answer=int(key),
            answer_text=str(int(key)),
            answer_latex=latex_answer(str(int(key))),
            marks=marks,
            sensible_range=(0, None),
            working=work,
            distractors=pick_distractors(str(int(key)), shown, 3),
            follow_through=ft,
        )

    parts = [
        frac_part(),
        num_part(
            scene["ask_whole"].format(name=name),
            whole,
            2,
            [
                f"{x} is {_frac_text(left_frac)} of the original, so the original is {x} ÷ {_frac_text(left_frac)} = {int(whole)}."
            ],
            [
                (x * left_frac, "multiplied-by-fraction-to-find-whole"),
                (x / rest, "ignored-second-fraction"),
                *([(x / (1 - f1 - f2), "second-fraction-of-original")] if 1 - f1 - f2 > 0 else []),
                (x * (1 - f2), None),
                (x / (1 - f2), None),  # scales back with the second fraction only
            ],
            f"{x}/part_a",
        ),
        num_part(
            scene["ask_second"],
            second,
            2,
            [f"The second use is {_frac_text(f2)} of {_frac_text(rest)} of {int(whole)} = {int(second)}."],
            [
                (whole * f2, "second-fraction-of-original"),
                (whole * f1, None),
                (Fraction(x), None),
            ],
            f"part_b*{rest}*{f2}".replace("/", "/")
            if False
            else f"part_b*({rest.numerator}/{rest.denominator})*({f2.numerator}/{f2.denominator})",
        ),
    ]
    return dict(text=stem, parts=parts)
