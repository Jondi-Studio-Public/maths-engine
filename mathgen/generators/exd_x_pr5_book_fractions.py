"""Exam tier for `f10-fraction-add-subtract-equivalent`: three fractions of one whole, then what is left.

A task (reading a book, painting a fence, a charity ride) is done in three sessions, each a different fraction of
the whole, with unrelated denominators. The learner adds the first two fractions, subtracts all three from the
whole, and turns the fraction that is left into a count.

* (a) the fraction done in the first two sessions (2 marks);
* (b) the fraction still to do after the third session: 1 - (a) - the third fraction (2 marks, follow-through
  from (a));
* (c) how much that is in real units, the whole being a stated number of pages, metres or kilometres (1 mark,
  follow-through from (b)).

Built backwards from the whole number of units: it is a multiple of the denominator of the fraction left.
"""

from __future__ import annotations

import random
from fractions import Fraction

from mathgen.answers import Distractor
from mathgen.core import attempts, check, generator, latex_answer, numeric_answer, pick_distractors, rational_text

_DENOMS = [2, 3, 4, 5, 6, 8, 10, 12]

# (stem pattern, question for (c), unit noun). {f1}, {f2}, {f3} are fractions of the whole; {total} the whole.
_SETTINGS = [
    (
        "Priya is reading a novel of {total} pages. On Friday she reads {f1} of the book, on Saturday {f2} of the "
        "book and on Sunday {f3} of the book.",
        "the fraction of the book she has read after Saturday",
        "the fraction of the book she still has to read after Sunday",
        "How many pages does she still have to read after Sunday?",
    ),
    (
        "A work crew paints a fence {total} metres long beside the Murray River. On Monday they paint {f1} of the "
        "fence, on Tuesday {f2} of the fence and on Wednesday {f3} of the fence.",
        "the fraction of the fence painted by the end of Tuesday",
        "the fraction of the fence still unpainted after Wednesday",
        "How many metres of the fence are still unpainted after Wednesday?",
    ),
    (
        "A charity bike ride along the Great Ocean Road is {total} kilometres long. The riders cover {f1} of the "
        "distance on day one, {f2} of the distance on day two and {f3} of the distance on day three.",
        "the fraction of the ride completed after day two",
        "the fraction of the ride still to go after day three",
        "How many kilometres of the ride are still to go after day three?",
    ),
]


class _Retry(Exception):
    pass


def _text(f: Fraction) -> str:
    return rational_text(f)


def _attempt(rng: random.Random) -> dict:
    for _ in attempts(1000, "three fractions of a whole with a positive remainder"):
        dens = [rng.choice(_DENOMS) for _ in range(3)]
        fracs = [Fraction(rng.randint(1, d - 1), d) for d in dens]
        if len({f for f in fracs}) < 3 or any(f.denominator != d for f, d in zip(fracs, dens)):
            continue
        f1, f2, f3 = fracs
        first_two = f1 + f2
        left = 1 - first_two - f3
        if left < Fraction(1, 12) or first_two.denominator == 1 or left.denominator == 1:
            continue
        if first_two in (f1, f2, f3) or left in (f1, f2, f3, first_two):
            continue
        total = left.denominator * rng.randint(2, 40)
        remaining = left * total
        if not 60 <= total <= 480 or remaining.denominator != 1:
            continue
        shown = {total, *(f.numerator for f in fracs), *(f.denominator for f in fracs)}
        if int(remaining) in shown:
            continue
        break

    pattern, first_prompt, left_prompt, ask_units = rng.choice(_SETTINGS)
    stem = pattern.format(total=total, f1=_text(f1), f2=_text(f2), f3=_text(f3))

    n1, d1, n2, d2 = f1.numerator, f1.denominator, f2.numerator, f2.denominator
    wrong_a = Fraction(n1 + n2, d1 + d2)
    d_a = pick_distractors(
        _text(first_two),
        [
            Distractor(_text(wrong_a), "adds-numerators-and-denominators", wrong_a),
            Distractor(
                _text(Fraction(n1 + n2, max(d1, d2))),
                "adds-numerators-and-denominators",
                Fraction(n1 + n2, max(d1, d2)),
            ),
            Distractor(_text(abs(f1 - f2)), "story-operation-reversed", abs(f1 - f2)),
            Distractor(_text(1 - first_two), "answers-amount-used-not-remaining", 1 - first_two),
        ],
        3,
    )
    d_b = pick_distractors(
        _text(left),
        [
            Distractor(_text(1 - first_two), "omitted-a-term", 1 - first_two),
            Distractor(_text(first_two + f3), "answers-amount-used-not-remaining", first_two + f3),
            Distractor(_text(abs(f3 - first_two)), "story-operation-reversed", abs(f3 - first_two)),
            Distractor(
                _text(Fraction(1) - Fraction(n1 + n2 + f3.numerator, d1 + d2 + f3.denominator)),
                "adds-numerators-and-denominators",
                1 - Fraction(n1 + n2 + f3.numerator, d1 + d2 + f3.denominator),
            ),
        ],
        3,
    )
    done = (1 - left) * total
    d_c = pick_distractors(
        str(int(remaining)),
        [
            Distractor(str(int(done)), "answers-amount-used-not-remaining", done),
            Distractor(str(int((1 - first_two) * total)), "omitted-a-term", (1 - first_two) * total),
            Distractor(str(int(total // left.denominator)), "ignores-numerator", Fraction(total, left.denominator)),
        ],
        3,
    )
    if min(len(d_a), len(d_b), len(d_c)) < 3:
        raise _Retry
    check(first_two + f3 + left == 1, "fractions do not make one whole")

    return dict(
        text=stem,
        parts=[
            dict(
                prompt=f"What is {first_prompt}? Give your answer as a fraction in simplest form.",
                answer_kind="fraction",
                answer={"p": first_two.numerator, "q": first_two.denominator},
                answer_text=_text(first_two),
                answer_latex=latex_answer(_text(first_two)),
                marks=2,
                sensible_range=(0, 1),
                working=[f"{_text(f1)} + {_text(f2)} = {_text(first_two)}, using a common denominator."],
                distractors=d_a,
            ),
            dict(
                prompt=f"What is {left_prompt}? Give your answer as a fraction in simplest form.",
                answer_kind="fraction",
                answer={"p": left.numerator, "q": left.denominator},
                answer_text=_text(left),
                answer_latex=latex_answer(_text(left)),
                marks=2,
                sensible_range=(0, 1),
                follow_through=f"1 - part_a - {f3.numerator}/{f3.denominator}",
                working=[f"1 - {_text(first_two)} - {_text(f3)} = {_text(left)}."],
                distractors=d_b,
            ),
            dict(
                prompt=ask_units,
                answer_kind="number",
                answer=numeric_answer(remaining),
                answer_text=str(int(remaining)),
                answer_latex=latex_answer(str(int(remaining))),
                marks=1,
                sensible_range=(0, None),
                follow_through=f"part_b*{total}",
                working=[f"{_text(left)} of {total} = {int(remaining)}."],
                distractors=d_c,
            ),
        ],
    )


def _build(rng: random.Random) -> dict:
    for _ in attempts(200, "a draw with three distinct distractors per part"):
        try:
            return _attempt(rng)
        except _Retry:
            continue


@generator("exd_x_pr5_book_fractions", level=6, presets={"exam": dict(tier="exam")})
def exd_x_pr5_book_fractions(rng: random.Random, tier: str = "exam") -> dict:
    """Three unrelated-denominator fractions of one whole: the first two added, the remainder found, then counted."""
    check(tier == "exam", "exd_x_pr5_book_fractions: only the exam tier exists")
    return _build(rng)
