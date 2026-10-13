"""Exam-tier source for `solve-simultaneous-equations`: a priced-orders problem that must be set up and solved.

Two receipts from the same shop each list a number of two items and a total. The learner has to form the pair of
equations, eliminate one unknown for the first price, substitute for the second, and then use both prices for a
third order. Part (b) and part (c) carry `follow_through`, so a slip in (a) still earns the later marks when the
learner uses their own price correctly.

Built backwards: both prices are drawn first (in multiples of 5 cents), so every total and every answer is an
exact amount of money and the key is exact.
"""

from __future__ import annotations

from fractions import Fraction

from mathgen.answers import Distractor
from mathgen.core import (
    GRADER_TOLERANCE,
    attempts,
    check,
    generator,
    latex_answer,
    numeric_answer,
    pick_distractors,
    round_half_up,
    stated_precision_string,
)

_SHOPS = [
    ("a bakery in Fremantle", ("meat pie", "meat pies"), ("sausage roll", "sausage rolls")),
    ("a canteen in Geelong", ("sandwich", "sandwiches"), ("juice", "juices")),
    ("a cafe in Newcastle", ("flat white", "flat whites"), ("muffin", "muffins")),
    ("a kiosk at Bondi Beach", ("ice cream", "ice creams"), ("bottle of water", "bottles of water")),
    (
        "a stall at the Adelaide Central Market",
        ("punnet of strawberries", "punnets of strawberries"),
        ("loaf of sourdough", "loaves of sourdough"),
    ),
    ("a school tuckshop in Toowoomba", ("wrap", "wraps"), ("carton of milk", "cartons of milk")),
]


def _money(value: Fraction) -> str:
    return stated_precision_string(round_half_up(value, 2), 2)


def _count(noun: tuple[str, str], n: int) -> str:
    return f"{n} {noun[0] if n == 1 else noun[1]}"


def _tag(text_value: Fraction, ident: str):
    rounded = round_half_up(text_value, 2)
    return Distractor(_money(rounded), ident, rounded)


@generator("exd_x_u3_simultaneous_prices", level=9, presets={"exam": dict(tier="exam")})
def exd_x_u3_simultaneous_prices(rng, tier: str = "exam") -> dict:
    """Exam tier: set up and solve a 2x2 system from two receipts, then price a third order."""
    check(tier == "exam", f"unknown tier {tier!r}")
    shop, first, second = rng.choice(_SHOPS)
    for _ in attempts(1000, "two receipts with a clean, distinct system"):
        x = Fraction(rng.randint(40, 180), 20)  # 2.00 .. 9.00 in 5c steps
        y = Fraction(rng.randint(40, 180), 20)
        p, q, r, s = (rng.randint(1, 6) for _ in range(4))
        det = p * s - q * r
        n, m = rng.randint(2, 7), rng.randint(2, 7)
        if x == y or abs(det) < 2 or p == r or q == s or n == m or p * s == q * r:
            continue
        # The unscaled-subtraction slip must differ from the key, and every candidate must be a sensible price.
        total_one, total_two = p * x + q * y, r * x + s * y
        if total_one == total_two:
            continue
        stem_numbers = {str(v) for v in (p, q, r, s, n, m)} | {_money(total_one), _money(total_two)}
        if {_money(x), _money(y), _money(n * x + m * y)} & stem_numbers:
            continue
        break

    # Independent route: Cramer's rule from the printed totals.
    x_back = (total_one * s - total_two * q) / det
    y_back = (p * total_two - r * total_one) / det
    check((x_back, y_back) == (x, y), "the system does not return the drawn prices")
    cost = n * x + m * y

    stem = (
        f"At {shop}, one receipt shows {_count(first, p)} and {_count(second, q)} for ${_money(total_one)}. "
        f"A second receipt shows {_count(first, r)} and {_count(second, s)} for ${_money(total_two)}."
    )

    def sane(v: Fraction) -> bool:
        return v > 0

    a_cand = [
        _tag(y, "swapped-unknowns"),
        _tag((total_one * s + total_two * q) / (p * s + q * r), "elimination-wrong-operation"),
        _tag(total_one / p, "ignored-other-unknown"),
    ]
    if r != p:
        a_cand.append(_tag((total_two - total_one) / (r - p), "elimination-without-matching-coefficients"))
    b_cand = [
        _tag(x, "swapped-unknowns"),
        _tag((total_one - x) / q, "dropped-coefficient-in-substitution"),
        _tag(total_one / q, "ignored-other-unknown"),
        _tag((p * total_two + r * total_one) / (p * s + q * r), "elimination-wrong-operation"),
    ]
    c_cand = [
        _tag(n * y + m * x, "swapped-unknowns"),
        _tag(n * x, "ignored-other-unknown"),
        _tag(m * y, "ignored-other-unknown"),
    ]
    x_text, y_text, cost_text = _money(x), _money(y), _money(cost)
    a_d = pick_distractors(x_text, [c for c in a_cand if sane(c.value)], 3, tolerance=GRADER_TOLERANCE)
    b_d = pick_distractors(y_text, [c for c in b_cand if sane(c.value)], 3, tolerance=GRADER_TOLERANCE)
    c_d = pick_distractors(cost_text, [c for c in c_cand if sane(c.value)], 3, tolerance=GRADER_TOLERANCE)
    check(a_d and b_d and c_d, "a part has no distractor")

    return dict(
        text=stem,
        parts=[
            dict(
                prompt=f"Find the price of 1 {first[0]}, in dollars.",
                answer_kind="number",
                answer=numeric_answer(x),
                answer_text=x_text,
                answer_latex=latex_answer(x_text),
                marks=3,
                sensible_range=(0, None),
                working=[
                    f"Let a = price of a {first[0]} and b = price of a {second[0]}. "
                    f"{p}a + {q}b = {_money(total_one)} and {r}a + {s}b = {_money(total_two)}.",
                    "Make one coefficient match, subtract to eliminate one unknown, and solve for a.",
                    f"a = {x_text}.",
                ],
                distractors=a_d,
            ),
            dict(
                prompt=f"Hence find the price of 1 {second[0]}, in dollars.",
                answer_kind="number",
                answer=numeric_answer(y),
                answer_text=y_text,
                answer_latex=latex_answer(y_text),
                marks=2,
                sensible_range=(0, None),
                follow_through=f"({_money(total_one)} - {p}*part_a)/{q}",
                working=[
                    f"Substitute a into the first receipt: b = ({_money(total_one)} - {p} × {x_text}) ÷ {q} = {y_text}."
                ],
                distractors=b_d,
            ),
            dict(
                prompt=f"Using your prices from (a) and (b), find the cost of {_count(first, n)} and "
                f"{_count(second, m)}, in dollars.",
                answer_kind="number",
                answer=numeric_answer(cost),
                answer_text=cost_text,
                answer_latex=latex_answer(cost_text),
                marks=1,
                sensible_range=(0, None),
                follow_through=f"{n}*part_a + {m}*part_b",
                working=[f"{n} × {x_text} + {m} × {y_text} = {cost_text}."],
                distractors=c_d,
            ),
        ],
    )
