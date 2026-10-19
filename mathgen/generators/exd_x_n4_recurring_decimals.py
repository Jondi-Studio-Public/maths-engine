"""Exam-tier sources for `f10-terminating-recurring-decimals` and `recurring-decimal-to-fraction`.

Both are chains in which each conversion feeds a final calculation done with fractions, so the learner has to
choose the method for each decimal and then combine the results exactly.

`exd_x_n4_decimal_sum` gives one terminating and one recurring decimal:

    (a) the terminating decimal as a fraction, (b) the recurring decimal as a fraction, (c) their sum or difference.

`exd_x_n4_recurring_mixed` gives a purely recurring decimal y and a decimal x that has the same recurring block
after some non-repeating digits:

    (a) y as a fraction, (b) x as a fraction (using the fact that x is the digits before the block plus y, scaled),
    (c) the difference between x and y.

Every canonical answer is an exact fraction in lowest terms, built from the chosen digits, and every part from (b)
on follows through from the parts before it.
"""

from __future__ import annotations

import math
from fractions import Fraction

import sympy

from mathgen.answers import Distractor, exact_value
from mathgen.core import (
    attempts,
    check,
    check_equivalent,
    equals_chain,
    generator,
    latex_answer,
    pick_distractors,
    rational_text,
)


def _states_exactly(*values: Fraction) -> bool:
    """Whether follow-through marking can state each value exactly: the marker simplifies a value before restating
    it, and for a few fractions with large parts that simplification returns a surd instead of the fraction."""
    return all(sympy.nsimplify(sympy.Rational(v.numerator, v.denominator)).is_Rational for v in values)


def _fraction_text(value: Fraction) -> str:
    return rational_text(value)


def _dist(value: Fraction, tag: str) -> Distractor:
    return Distractor(_fraction_text(value), tag, exact_value(value))


def _recurring_digits(block: int, width: int, times: int = 3) -> str:
    return f"{block:0{width}d}" * times


def _repeat_phrase(block: int, width: int) -> str:
    digits = f"{block:0{width}d}"
    return f"the digit {digits} repeats forever" if width == 1 else f"the digits {digits} repeat forever"


def _recurring_value(lead: int, lead_width: int, block: int, width: int) -> Fraction:
    """The value of 0.<lead><block repeating>, worked out as a geometric series (an independent route)."""
    series = Fraction(block, 10 ** (lead_width + width)) / (1 - Fraction(1, 10**width))
    return Fraction(lead, 10**lead_width) + series


def _fraction_part(prompt, value, *, marks, distractors, working, follow_through=None, sensible_range=None):
    part = dict(
        prompt=prompt,
        answer_kind="fraction",
        answer=exact_value(value),
        answer_text=_fraction_text(value),
        answer_latex=latex_answer(_fraction_text(value)),
        marks=marks,
        working=working,
        distractors=distractors,
    )
    if follow_through is not None:
        part["follow_through"] = follow_through
    if sensible_range is not None:
        part["sensible_range"] = sensible_range
    return part


def _draw_block(rng, width: int) -> int:
    """A repeating block that is a genuine block of that width (not all nines, not one digit repeated)."""
    if width == 1:
        return rng.randint(1, 8)
    for _ in attempts(1000, "a two-digit block"):
        block = rng.randint(10, 98)
        if block % 11:
            return block
    raise AssertionError("unreachable")


@generator("exd_x_n4_decimal_sum", level=9, presets={"exam": {"tier": "exam"}})
def exd_x_n4_decimal_sum(rng, tier: str = "exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(1000, "a terminating and a recurring decimal with a clean combined answer"):
        places = rng.choice((2, 3))
        numerator = rng.randint(10 ** (places - 1), 10**places - 1)
        if numerator % 10 == 0 or math.gcd(numerator, 10**places) == 1:
            continue  # want a decimal that really needs reducing
        width = rng.choice((1, 2))
        block = _draw_block(rng, width)
        terminating = Fraction(numerator, 10**places)
        recurring = Fraction(block, 10**width - 1)
        if recurring.denominator == 1 or terminating == recurring:
            continue
        op = rng.choice(("sum", "difference"))
        if op == "sum":
            total = terminating + recurring
            expression, follow = "t + r", "part_a+part_b"
        elif recurring > terminating:
            total = recurring - terminating
            expression, follow = "r − t", "part_b-part_a"
        else:
            total = terminating - recurring
            expression, follow = "t − r", "part_a-part_b"
        if total.denominator == 1 or not _states_exactly(total):
            continue
        shown_t = f"0.{numerator:0{places}d}"
        shown_r = f"0.{_recurring_digits(block, width)}..."
        stem = f"Let t = {shown_t} and r = {shown_r} ({_repeat_phrase(block, width)})."
        prompts = (
            "Write t as a fraction in lowest terms.",
            "Write r as a fraction in lowest terms.",
            f"Hence find {expression} as a fraction in lowest terms.",
        )

        digits_shown = int(_recurring_digits(block, width))
        truncated_r = Fraction(digits_shown, 10 ** (3 * width))
        cands_a = [
            Distractor(
                _fraction_text(Fraction(numerator, 10**places - 1)),
                "misclassifies-terminating-as-recurring",
                exact_value(Fraction(numerator, 10**places - 1)),
            ),
            _dist(Fraction(numerator, 10 ** (places + 1)), "decimal-place-value-miscounted"),
        ]
        cands_b = [
            _dist(truncated_r, "treats-recurring-as-terminating"),
            _dist(Fraction(block, 10**width), "treats-recurring-as-terminating"),
        ]
        if width == 2:
            cands_b.append(_dist(Fraction(block, 9), "wrong-repeat-length"))
        sloppy_r = truncated_r
        if op == "sum":
            cands_c = [
                Distractor(
                    f"{terminating.numerator + recurring.numerator}/{terminating.denominator + recurring.denominator}",
                    "added-numerators-and-denominators",
                    exact_value(
                        Fraction(
                            terminating.numerator + recurring.numerator,
                            terminating.denominator + recurring.denominator,
                        )
                    ),
                ),
                _dist(terminating + sloppy_r, "treats-recurring-as-terminating"),
            ]
        else:
            cands_c = [
                _dist(-total, "subtracted-in-wrong-order"),
                _dist(terminating + recurring, "added-instead-of-subtracted"),
                _dist(abs(terminating - sloppy_r), "treats-recurring-as-terminating"),
            ]
        a_text, b_text, c_text = (_fraction_text(v) for v in (terminating, recurring, total))
        d_a = pick_distractors(a_text, cands_a, 3)
        d_b = pick_distractors(b_text, cands_b, 3)
        d_c = pick_distractors(c_text, cands_c, 3)
        if min(len(d_a), len(d_b), len(d_c)) >= 2:
            break

    check_equivalent(
        _recurring_value(0, 0, block, width), recurring, "recurring value disagrees with the geometric series"
    )
    check_equivalent(Fraction(shown_t), terminating, "terminating value disagrees with its digits")
    check(total == (terminating + recurring if op == "sum" else abs(recurring - terminating)), "combined value is off")

    return dict(
        text=stem,
        parts=[
            _fraction_part(
                prompts[0],
                terminating,
                marks=1,
                distractors=d_a,
                working=[f"{equals_chain(shown_t, f'{numerator}/{10**places}', a_text)}."],
            ),
            _fraction_part(
                prompts[1],
                recurring,
                marks=2,
                distractors=d_b,
                working=[
                    f"The repeating block has {width} digit{'s' if width != 1 else ''}, "
                    f"so {equals_chain('r', f'{block}/{10**width - 1}', b_text)}."
                ],
            ),
            _fraction_part(
                prompts[2],
                total,
                marks=2,
                distractors=d_c,
                follow_through=follow,
                working=[f"{expression} = {a_text} and {b_text} combined over a common denominator = {c_text}."],
            ),
        ],
    )


@generator("exd_x_n4_recurring_mixed", level=9, presets={"exam": {"tier": "exam"}})
def exd_x_n4_recurring_mixed(rng, tier: str = "exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(1000, "a mixed recurring decimal with a clean difference"):
        width = rng.choice((1, 2))
        lead_width = rng.choice((1, 2))
        block = _draw_block(rng, width)
        lead = rng.randint(1, 9) if lead_width == 1 else rng.randint(10, 99)
        if f"{lead:0{lead_width}d}"[-width:] == f"{block:0{width}d}":
            continue  # the lead would run into the block and shorten the repeat
        y = Fraction(block, 10**width - 1)
        x = (lead + y) / 10**lead_width
        if y.denominator == 1 or x == y:
            continue
        if x > y:
            expression, follow, gap = "x − y", "part_b-part_a", x - y
        else:
            expression, follow, gap = "y − x", "part_a-part_b", y - x
        if gap.denominator == 1 or not _states_exactly(x, gap):
            continue
        lead_text = f"{lead:0{lead_width}d}"
        y_shown = f"0.{_recurring_digits(block, width)}..."
        x_shown = f"0.{lead_text}{_recurring_digits(block, width)}..."
        lead_phrase = (
            f"the digit {lead_text} is followed by" if lead_width == 1 else f"the digits {lead_text} are followed by"
        )
        block_phrase = (
            f"the digit {block:0{width}d} repeating forever"
            if width == 1
            else f"the digits {block:0{width}d} repeating forever"
        )
        stem = f"Let y = {y_shown} ({_repeat_phrase(block, width)}) and x = {x_shown} ({lead_phrase} {block_phrase})."
        prompts = (
            "Write y as a fraction in lowest terms.",
            "Hence write x as a fraction in lowest terms.",
            f"Hence find {expression} as a fraction in lowest terms.",
        )
        whole_block = int(f"{lead_text}{block:0{width}d}")
        digits_shown = int(f"{lead_text}{_recurring_digits(block, width)}")
        cands_a = [
            _dist(Fraction(int(_recurring_digits(block, width)), 10 ** (3 * width)), "treats-recurring-as-terminating"),
            _dist(Fraction(block, 10**width), "treats-recurring-as-terminating"),
        ]
        if width == 2:
            cands_a.append(_dist(Fraction(block, 9), "wrong-repeat-length"))
        cands_b = [
            _dist(Fraction(whole_block, 10 ** (lead_width + width) - 1), "ignores-non-repeating-lead"),
            _dist(y / 10**lead_width, "dropped-lead-digits"),
            _dist(Fraction(digits_shown, 10 ** (lead_width + 3 * width)), "treats-recurring-as-terminating"),
        ]
        cands_c = [
            _dist(-gap, "subtracted-in-wrong-order"),
            _dist(x + y, "added-instead-of-subtracted"),
        ]
        a_text, b_text, c_text = (_fraction_text(v) for v in (y, x, gap))
        d_a = pick_distractors(a_text, cands_a, 3)
        d_b = pick_distractors(b_text, cands_b, 3)
        d_c = pick_distractors(c_text, cands_c, 3)
        if min(len(d_a), len(d_b), len(d_c)) >= 2:
            break

    check_equivalent(_recurring_value(0, 0, block, width), y, "y disagrees with the geometric series")
    check_equivalent(_recurring_value(lead, lead_width, block, width), x, "x disagrees with the geometric series")
    check(gap == abs(x - y), "difference is off")

    return dict(
        text=stem,
        parts=[
            _fraction_part(
                prompts[0],
                y,
                marks=2,
                distractors=d_a,
                working=[f"{equals_chain('y', f'{block}/{10**width - 1}', a_text)}."],
            ),
            _fraction_part(
                prompts[1],
                x,
                marks=2,
                distractors=d_b,
                follow_through=f"({lead}+part_a)/{10**lead_width}",
                working=[
                    f"x is the digits {lead_text} followed by y, so x = ({lead} + y) / {10**lead_width} = {b_text}."
                ],
            ),
            _fraction_part(
                prompts[2],
                gap,
                marks=1,
                distractors=d_c,
                follow_through=follow,
                working=[f"{expression} = {c_text}."],
            ),
        ],
    )
