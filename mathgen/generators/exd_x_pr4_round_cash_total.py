"""Exam question on rounding to a stated accuracy: cash payments rounded to the nearest 5 cents.

Australian shops round a cash total to the nearest 5 cents (card payments are not rounded). Part (a) finds the
exact total of two lines of purchases, part (b) rounds it to the nearest 5 cents, and part (c) finds the change
from a note. The exact total never ends in 0 or 5 cents, so the rounding always matters, and it is rounded up
in some questions and down in others.
"""

from __future__ import annotations

import random
from fractions import Fraction

from mathgen.answers import Distractor
from mathgen.core import (
    attempts,
    check,
    generator,
    numeric_answer,
    pick_distractors,
    round_half_up,
    stated_precision_string,
)

# (place, buyer names, item one, item two, price range one in cents, price range two in cents)
_SHOPS = (
    ("a cafe in Fitzroy", "flat whites", "muffins", (390, 560), (450, 690)),
    ("a bakery in Parramatta", "meat pies", "vanilla slices", (420, 650), (310, 540)),
    ("a school canteen in Toowoomba", "toasted sandwiches", "fruit cups", (360, 520), (210, 380)),
    ("a bookshop in Geelong", "notebooks", "pens", (480, 790), (130, 290)),
    ("a sports club kiosk in Mackay", "sausage rolls", "iced teas", (350, 480), (330, 470)),
)
_NAMES = ("Ravi", "Chloe", "Jack", "Mia", "Hamid", "Ella", "Sione", "Grace")
_NOTES = (20, 50, 100)


def _money(cents: int | Fraction) -> str:
    return stated_precision_string(Fraction(cents, 100), 2)


def _nearest_five(cents: int) -> int:
    return 5 * int(round_half_up(Fraction(cents, 5), 0))


@generator("exd_x_pr4_round_cash_total", level=6, presets={"exam": dict(tier="exam")})
def exd_x_pr4_round_cash_total(rng: random.Random, tier: str = "exam") -> dict:
    check(tier == "exam", "exd_x_pr4_round_cash_total: only the exam tier exists")
    place, item1, item2, range1, range2 = rng.choice(_SHOPS)
    for _ in attempts(1000, "a total that is not a multiple of 5 cents"):
        q1, q2 = rng.randint(2, 4), rng.randint(2, 5)
        if q1 == q2:
            continue
        p1, p2 = rng.randint(*range1), rng.randint(*range2)
        total = q1 * p1 + q2 * p2
        if total % 5 and p1 % 5 and p2 % 5:
            break
    note = next(n for n in _NOTES if total < 100 * n)
    cash = _nearest_five(total)

    # Independent route: the last digit of the total decides the rounding.
    last = total % 10
    expected = total - last + {1: 0, 2: 0, 3: 5, 4: 5, 5: 5, 6: 5, 7: 5, 8: 10, 9: 10, 0: 0}[last]
    check(cash == expected and cash != total, "cash rounding disagrees")
    change = 100 * note - cash

    total_text, cash_text, change_text = _money(total), _money(cash), _money(change)
    up = total - total % 5 + 5
    down = total - total % 5
    nearest_ten = 10 * int(round_half_up(Fraction(total, 10), 0))

    def d(cents, misconception):
        return Distractor(_money(cents), misconception, Fraction(cents, 100))

    a_candidates = [
        d(q2 * p1 + q1 * p2, "uses-wrong-quantity"),
        d(p1 + p2, "uses-wrong-quantity"),
        d(q1 * p1 + p2, "uses-wrong-quantity"),
    ]
    b_candidates = [
        d(nearest_ten, "rounds-to-nearest-ten-cents"),
        d(up, "rounds-cash-total-up"),
        d(down, "rounded-down-not-to-nearest"),
        d(total, None),
    ]
    c_candidates = [
        d(100 * note - total, "change-from-unrounded-total"),
        d(100 * note - up, "rounds-cash-total-up"),
        d(100 * note - nearest_ten, "rounds-to-nearest-ten-cents"),
        d(100 * note - down, "rounded-down-not-to-nearest"),
    ]
    a_d = pick_distractors(total_text, a_candidates, 3, tolerance=Fraction(1, 200))
    b_d = pick_distractors(cash_text, b_candidates, 3, tolerance=Fraction(1, 200))
    c_d = pick_distractors(change_text, c_candidates, 3, tolerance=Fraction(1, 200))
    check(min(len(a_d), len(b_d), len(c_d)) >= 2, "too few distractors")

    buyer = rng.choice(_NAMES)
    stem = (
        f"At {place}, {buyer} buys {q1} {item1} at ${_money(p1)} each and {q2} {item2} at ${_money(p2)} each. "
        f"{rng.choice(('Cash payments are rounded to the nearest 5 cents.', 'A cash total is rounded to the nearest 5 cents, but a card total is not.'))} "
        f"{buyer} pays in cash with a ${note} note."
    )
    return dict(
        text=stem,
        parts=[
            dict(
                prompt="Find the exact total cost, in dollars.",
                answer_kind="number",
                answer=numeric_answer(Fraction(total, 100)),
                answer_text=total_text,
                marks=2,
                sensible_range=(0, None),
                working=[
                    f"{q1} × {_money(p1)} + {q2} × {_money(p2)} = {_money(q1 * p1)} + {_money(q2 * p2)} = {total_text}."
                ],
                distractors=a_d,
            ),
            dict(
                prompt="How much is paid in cash, in dollars?",
                answer_kind="number",
                answer=numeric_answer(Fraction(cash, 100)),
                answer_text=cash_text,
                marks=1,
                sensible_range=(0, None),
                follow_through="5*floor(20*part_a + 1/2)/100",
                working=[f"{total_text} is {'closer to ' + cash_text}, the nearest multiple of 5 cents."],
                distractors=b_d,
            ),
            dict(
                prompt="How much change is given, in dollars?",
                answer_kind="number",
                answer=numeric_answer(Fraction(change, 100)),
                answer_text=change_text,
                marks=1,
                sensible_range=(0, None),
                follow_through=f"{note} - part_b",
                working=[f"{note} - {cash_text} = {change_text}."],
                distractors=c_d,
            ),
        ],
    )
