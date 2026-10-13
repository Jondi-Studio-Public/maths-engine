"""Exam tier for "use leading-digit approximation to obtain estimates of calculations".

The practice tiers round the operands of a bare expression and evaluate it. Here the calculation sits in a short
story (a total cost, or a quantity shared out), and the learner estimates it, finds the exact value, and then
judges the estimate by its percentage error. Three parts:

(a) the leading-digit estimate (2 marks);
(b) the exact value (1 mark);
(c) the percentage error of the estimate, as a percentage of the exact value, to the nearest whole number
    (2 marks, follow-through from (a) and (b)).

Two structures are used, a sum of two products and a product shared by a divisor. Both are built so the exact
value and the estimate are whole numbers, and the percentage error is kept at least 0.02 clear of a rounding
boundary and between 3% and 60%.
"""

from __future__ import annotations

from fractions import Fraction

from mathgen.answers import Distractor
from mathgen.core import (
    check,
    check_distinct,
    draw_until,
    generator,
    latex_answer,
    phrase,
    pick_distractors,
    round_half_up,
)
from mathgen.slots import NAMES

_SUM_STEMS = (
    "{name} runs a stall at a Perth market. On Saturday {name} sells {a} {item1} at ${b} each and {c} {item2} "
    "at ${d} each.",
    "A school in Hobart orders {a} {item1} at ${b} each and {c} {item2} at ${d} each for a fundraiser.",
    "{name} is planning a community event in Darwin and needs {a} {item1} costing ${b} each and {c} {item2} "
    "costing ${d} each.",
)
_SUM_ITEMS = (
    ("pies", "sausage rolls"),
    ("T-shirts", "caps"),
    ("plants", "pots"),
    ("notebooks", "pens"),
    ("raffle tickets", "stickers"),
    ("folding chairs", "tables"),
)
_SUM_ASK = (
    "the total amount, in dollars",
    "the total cost, in dollars",
)
_SHARE_STEMS = (
    "A bakery in Adelaide makes {a} loaves a day for {b} days. The loaves are then shared equally among {c} stores.",
    "A farm near Toowoomba packs {a} eggs into each of {b} crates. The eggs are then divided equally among {c} "
    "supermarkets.",
    "A printer in Geelong produces {a} pages an hour for {b} hours. The pages are bound into {c} equal batches.",
)


def _d(value: int, tag: str | None) -> Distractor:
    return Distractor(str(value), tag, value)


def _lead(n: int) -> int:
    """`n` rounded to its own leading digit (to the nearest 10^k, where 10^k is its leading place)."""
    p = 10 ** (len(str(n)) - 1)
    return int(round_half_up(Fraction(n, p), 0)) * p


def _floor_lead(n: int) -> int:
    p = 10 ** (len(str(n)) - 1)
    return n // p * p


def _tens(n: int) -> int:
    return int(round_half_up(Fraction(n, 10), 0)) * 10


def _plain(n: int) -> bool:
    """Not already a single non-zero digit followed by zeros (so rounding changes it)."""
    return n % (10 ** (len(str(n)) - 1)) != 0


def _pct(est: int, exact: int) -> Fraction:
    return Fraction(100 * abs(est - exact), exact)


def _draw(rng):
    if rng.random() < 0.5:
        a, b, c, d = rng.randint(120, 880), rng.randint(12, 88), rng.randint(110, 690), rng.randint(12, 79)
        shape = "sum"
        values = (a, b, c, d)
        est = _lead(a) * _lead(b) + _lead(c) * _lead(d)
        exact = a * b + c * d
    else:
        a, b = rng.randint(120, 880), rng.randint(12, 88)
        divisors = [x for x in range(12, 99) if (a * b) % x == 0 and x % 10 != 0]
        if not divisors:
            return None
        c = rng.choice(divisors)
        shape = "share"
        values = (a, b, c)
        if (_lead(a) * _lead(b)) % _lead(c):
            return None
        est = _lead(a) * _lead(b) // _lead(c)
        exact = a * b // c
    return shape, values, est, exact


def _acceptable(d) -> bool:
    if d is None:
        return False
    shape, values, est, exact = d
    if not all(_plain(v) for v in values) or est == exact:
        return False
    if shape == "share" and (exact in values or any(values[i] % values[2] == 0 for i in (0, 1))):
        return False
    pct = _pct(est, exact)
    return 3 < pct < 60 and abs((pct % 1) - Fraction(1, 2)) > Fraction(1, 50)


@generator("exd_x_n2_leading_digit_error", level=10, presets={"exam": dict(tier="exam")})
def exd_x_n2_leading_digit_error(rng, tier="exam"):
    """Estimate, exact value, percentage error; see the module docstring."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    shape, values, est, exact = draw_until(rng, _draw, _acceptable, what="an estimate with a clear percentage error")
    pct = int(round_half_up(_pct(est, exact), 0))
    name = rng.choice(NAMES)
    if shape == "sum":
        a, b, c, d = values
        item1, item2 = rng.choice(_SUM_ITEMS)
        stem = phrase(rng, _SUM_STEMS, name=name, a=a, b=b, c=c, d=d, item1=item1, item2=item2)
        what = rng.choice(_SUM_ASK)
        a_prompt = f"Round every number to its leading digit to estimate {what}. Write the estimate as a whole number."
        b_prompt = f"Calculate the exact value of {what}."
        a_work = [f"Estimate = {_lead(a)} × {_lead(b)} + {_lead(c)} × {_lead(d)} = {est}."]
        b_work = [f"Exact = {a} × {b} + {c} × {d} = {exact}."]
        # Slips
        a_cands = [
            Distractor(
                str(_floor_lead(a) * _floor_lead(b) + _floor_lead(c) * _floor_lead(d)), "truncates-instead-of-rounds"
            ),
            _d(_tens(a) * _tens(b) + _tens(c) * _tens(d), "same-fixed-place-for-every-operand"),
            _d((_lead(a) * _lead(b) + _lead(c)) * _lead(d), "left-to-right-instead-of-precedence"),
            _d(_lead(a) * _lead(b), "forgot-second-purchase"),
        ]
        b_cands = [
            _d(est, "estimate-reported-as-exact"),
            _d((a + c) * (b + d), None),
            _d(a * b + c + d, "left-to-right-instead-of-precedence"),
        ]
    else:
        a, b, c = values
        stem = phrase(rng, _SHARE_STEMS, a=a, b=b, c=c)
        what = "the number each one receives"
        a_prompt = f"Round every number to its leading digit to estimate {what}. Write the estimate as a whole number."
        b_prompt = f"Calculate the exact value of {what}."
        a_work = [f"Estimate = {_lead(a)} × {_lead(b)} ÷ {_lead(c)} = {est}."]
        b_work = [f"Exact = {a} × {b} ÷ {c} = {exact}."]
        a_cands = [
            _d(_lead(a) * _lead(b) * _lead(c), "multiplied-where-division-needed"),
            _d(_floor_lead(a) * _floor_lead(b) // _floor_lead(c), "truncates-instead-of-rounds"),
            _d(_tens(a) * _tens(b) // _tens(c), "same-fixed-place-for-every-operand"),
            _d(_lead(a) * _lead(b), "forgot-to-share"),
        ]
        b_cands = [
            _d(est, "estimate-reported-as-exact"),
            _d(a * b * c, "multiplied-where-division-needed"),
            _d(a * b, None),
        ]
    c_prompt = (
        "Find the percentage error of your estimate, as a percentage of the exact value. Give the answer to the "
        "nearest whole number."
    )
    c_work = [
        f"Error = |{est} - {exact}| = {abs(est - exact)}; ({abs(est - exact)} ÷ {exact}) × 100 = {pct}% (nearest whole)."
    ]
    err = abs(est - exact)
    c_cands = [
        _d(int(round_half_up(Fraction(100 * err, est), 0)), "error-over-estimate-not-exact"),
        _d(err, "gave-change-not-percent"),
        _d(int(round_half_up(Fraction(100 * est, exact), 0)), "new-as-percent-of-original"),
    ]
    check(est != exact and pct > 0, "estimate must differ from the exact value")

    def part(prompt, key, marks, work, cands, ft=None):
        text = str(key)
        picked = pick_distractors(text, cands, 3)
        check_distinct(key, *[int(str(x)) for x in picked], msg="a distractor equals the answer")
        out = dict(
            prompt=prompt,
            answer_kind="number",
            answer=key,
            answer_text=text,
            answer_latex=latex_answer(text),
            marks=marks,
            sensible_range=(0, None),
            working=work,
            distractors=picked,
        )
        if ft:
            out["follow_through"] = ft
        return out

    parts = [
        part(a_prompt, est, 2, a_work, a_cands),
        part(b_prompt, exact, 1, b_work, b_cands),
        part(c_prompt, pct, 2, c_work, c_cands, ft="100*Abs(part_a - part_b)/part_b"),
    ]
    return dict(text=stem, parts=parts)
