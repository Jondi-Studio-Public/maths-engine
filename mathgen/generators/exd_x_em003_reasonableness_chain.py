"""Exam chain for ``senior-ascertain-the-reasonableness-of-answers-to-arithmetic-calculations`` (ACMEM003): estimate a
product by rounding to one significant figure, judge a calculator answer against the estimate, find the exact value
and measure how far the estimate was out.

One calculation, a quantity times a rate (or a length times a width), is set in an Australian everyday or trade
context. Set the way a school exam sets it:

(a) the estimate, each number rounded to 1 significant figure          number, 2 marks
(b) whether a stated calculator answer is reasonable, by the estimate  mcq,    1 mark
(c) the exact value                                                    number, 1 mark
(d) the estimate's difference from the exact value, as a percentage    number, 2 marks, follow-through (a), (c)

## Construction

Both numbers are two or three digits and are drawn so that neither rounds on a tie. The estimate is between
5% and 40% out, so the percentage in (d) is a real calculation and the stated answer in (b) is judged by size, not
by matching digits. The stated answer in (b) is one of three kinds, each equally likely: within 8% of the estimate
(reasonable), or about ten times the estimate, or about a tenth of it. The oracle in the test re-derives the
verdict from the estimate alone.

## Distractors, each one error of its own part

* (a): only one number rounded, the rounded numbers added, rounding to the nearest ten, truncating instead of
  rounding, a power of ten out.
* (b): the three other verdicts, each the error of a learner who wants digits to match, who accepts a wrong
  magnitude, who reverses the direction, or who thinks the exact answer is needed first.
* (c): a power of ten out, the numbers added, the estimate given as the exact value, a partial product left unshifted.
* (d): the difference taken as a percentage of the estimate, the ratio instead of the difference, the raw difference,
  a decimal place out.
"""

from __future__ import annotations

from fractions import Fraction

from mathgen.core import attempts, check, generator, phrase, round_half_up
from mathgen.generators._exd_x_w12a_kit import mcq_part, number_part

TOPIC = "exd_x_em003_reasonableness_chain"

# lead: two {a} and {b} slots, the product the context's total; unit is how the answer is asked for.
# lo/hi: the draw ranges of a and b, chosen so the total is a plausible size for the situation.
_CONTEXTS = (
    dict(
        lead="An electrician works {a} hours on a rewire and charges ${b} per hour. The total charge is worked out as {a} × {b}.",
        total="total charge",
        unit="in dollars",
        money=True,
        a=(12, 48),
        b=(65, 135),
    ),
    dict(
        lead="A landscaper lays {a} square metres of turf at ${b} per square metre. The cost is worked out as {a} × {b}.",
        total="cost of the turf",
        unit="in dollars",
        money=True,
        a=(35, 190),
        b=(14, 48),
    ),
    dict(
        lead="A tiler orders {a} boxes of floor tiles at ${b} a box. The bill is worked out as {a} × {b}.",
        total="bill",
        unit="in dollars",
        money=True,
        a=(14, 85),
        b=(38, 95),
    ),
    dict(
        lead="A caterer prepares a wedding meal for {a} guests at ${b} per head. The quote is worked out as {a} × {b}.",
        total="quote",
        unit="in dollars",
        money=True,
        a=(45, 180),
        b=(28, 85),
    ),
    dict(
        lead="A farmer buys {a} round hay bales at ${b} each. The cost is worked out as {a} × {b}.",
        total="cost of the hay",
        unit="in dollars",
        money=True,
        a=(60, 290),
        b=(14, 38),
    ),
    dict(
        lead="A rectangular paddock is {a} m long and {b} m wide. Its area is worked out as {a} × {b}.",
        total="area of the paddock",
        unit="in square metres",
        money=False,
        a=(35, 280),
        b=(24, 95),
    ),
    dict(
        lead="A casual worker puts in {a} hours over a fortnight at ${b} per hour. The pay is worked out as {a} × {b}.",
        total="fortnight's pay",
        unit="in dollars",
        money=True,
        a=(38, 96),
        b=(26, 48),
    ),
    dict(
        lead="A removalist has {a} vans on the road, and each drives {b} km in a week. The total distance is worked out as {a} × {b}.",
        total="total distance",
        unit="in kilometres",
        money=False,
        a=(12, 48),
        b=(130, 480),
    ),
    dict(
        lead="A builder buys {a} sheets of plasterboard at ${b} a sheet. The cost is worked out as {a} × {b}.",
        total="cost of the plasterboard",
        unit="in dollars",
        money=True,
        a=(16, 90),
        b=(14, 32),
    ),
    dict(
        lead="A school buys {a} classroom chairs at ${b} each. The total is worked out as {a} × {b}.",
        total="total price",
        unit="in dollars",
        money=True,
        a=(40, 160),
        b=(45, 130),
    ),
    dict(
        lead="A concreter pours {a} cubic metres of concrete at ${b} per cubic metre. The charge is worked out as {a} × {b}.",
        total="charge",
        unit="in dollars",
        money=True,
        a=(12, 60),
        b=(180, 320),
    ),
    dict(
        lead="A family hires a campervan for {a} days at ${b} per day. The hire cost is worked out as {a} × {b}.",
        total="hire cost",
        unit="in dollars",
        money=True,
        a=(12, 48),
        b=(120, 210),
    ),
)

_EST_ASK = (
    "Round each number in the calculation to 1 significant figure, then multiply, to estimate the {total}.",
    "Estimate the {total} by rounding both numbers to 1 significant figure first.",
    "Without a calculator, estimate the {total}: round each number to 1 significant figure and multiply.",
)
_VERDICT_ASK = (
    "A colleague's answer to the calculation is {s}. Using your estimate, is that answer reasonable?",
    "A student's answer to the calculation is {s}. Judging by your estimate, which statement is correct?",
    "A friend works the calculation out as {s}. Compare this with your estimate: is it reasonable?",
)
_EXACT_ASK = (
    "Work out the exact {total}, {unit}.",
    "Calculate the exact {total}, {unit}.",
)
_PERCENT_ASK = (
    "By what percentage does your estimate differ from the exact value? Give the percentage of the exact value, to the "
    "nearest whole number.",
    "Find the difference between your estimate and the exact value as a percentage of the exact value, to the nearest "
    "whole number.",
)

_REASONABLE = "Reasonable: the answer is about the size of the estimate"
_LARGE = "Not reasonable: the answer is about 10 times too large"
_SMALL = "Not reasonable: the answer is about 10 times too small"
_UNSURE = "It cannot be judged until the exact answer has been worked out"
_OPTIONS = (_REASONABLE, _LARGE, _SMALL, _UNSURE)

_WRONG_FOR = {
    _REASONABLE: (
        (_LARGE, "est-demands-exact-match"),
        (_SMALL, "est-demands-exact-match"),
        (_UNSURE, "est-needs-exact-answer"),
    ),
    _LARGE: (
        (_REASONABLE, "est-accepts-wrong-magnitude"),
        (_SMALL, "est-reverses-direction"),
        (_UNSURE, "est-needs-exact-answer"),
    ),
    _SMALL: (
        (_REASONABLE, "est-accepts-wrong-magnitude"),
        (_LARGE, "est-reverses-direction"),
        (_UNSURE, "est-needs-exact-answer"),
    ),
}


def _lead_digits(n: int) -> int:
    """The number of digits after the first: the power of ten of the place the first digit sits in."""
    return len(str(n)) - 1


def _round1(n: int) -> int:
    """`n` rounded to 1 significant figure, ties away from zero."""
    scale = 10 ** _lead_digits(n)
    return int(round_half_up(Fraction(n, scale))) * scale


def _trunc1(n: int) -> int:
    """`n` with every digit after the first replaced by zero."""
    scale = 10 ** _lead_digits(n)
    return n // scale * scale


def _is_tie(n: int) -> bool:
    """Whether 1-significant-figure rounding of `n` sits exactly halfway between two values."""
    scale = 10 ** _lead_digits(n)
    return n % scale * 2 == scale


def _unshifted_partial(a: int, b: int) -> int:
    """a × b with each partial product a × digit of b added without its place value."""
    return sum(a * int(d) for d in str(b))


def _draw(rng, ctx):
    for _ in attempts(3000, "two numbers whose estimate is 5% to 40% out and does not round on a tie"):
        a = rng.randint(*ctx["a"])
        b = rng.randint(*ctx["b"])
        if _is_tie(a) or _is_tie(b):
            continue
        exact = a * b
        est = _round1(a) * _round1(b)
        gap = Fraction(abs(est - exact), exact)
        if not Fraction(5, 100) <= gap <= Fraction(40, 100):
            continue
        pct = gap * 100
        if abs(pct % 1 - Fraction(1, 2)) <= Fraction(1, 20):
            continue
        if a % 10 == 0 or b % 10 == 0:
            continue
        return a, b, exact, est, int(round_half_up(pct))


def _stated(rng, kind: str, exact: int, est: int) -> int:
    """A calculator answer of the given kind, within 2% to 8% of the estimate (reasonable), of ten times it or of a
    tenth of it, and never equal to the exact value. Every reasonable answer is within 8% of the estimate, and every
    other is at least 8 times or at most an eighth of it, so no stated answer is a borderline call."""
    base = {"ok": Fraction(est), "large": Fraction(est * 10), "small": Fraction(est, 10)}[kind]
    for _ in attempts(100, "a stated answer that is not the exact value"):
        nudge = rng.choice((-1, 1)) * rng.randint(2, 8)
        value = int(round_half_up(base * (100 + nudge) / 100))
        if value != exact and value >= 10:
            return value


def _money(ctx, n: int) -> str:
    return f"${n:,}" if ctx["money"] else f"{n:,}"


@generator(TOPIC, level=13, presets={"exam": {"tier": "exam"}})
def exd_x_em003_reasonableness_chain(rng, tier: str = "exam"):
    """An estimate by 1 significant figure, a verdict on a stated answer, the exact value and the percentage error."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(200, "a reasonableness chain with three distractors per part"):
        ctx = rng.choice(_CONTEXTS)
        a, b, exact, est, pct = _draw(rng, ctx)
        kind = rng.choice(("ok", "large", "small"))
        stated = _stated(rng, kind, exact, est)
        key = {"ok": _REASONABLE, "large": _LARGE, "small": _SMALL}[kind]
        ra, rb = _round1(a), _round1(b)
        gap = abs(est - exact)
        part_a = number_part(
            phrase(rng, _EST_ASK, total=ctx["total"]),
            est,
            2,
            [f"{a} rounds to {ra} and {b} rounds to {rb}.", f"Estimate = {ra} × {rb} = {est}."],
            [
                (ra * b, "est-rounds-one-factor-only"),
                (a * rb, "est-rounds-one-factor-only"),
                (ra + rb, "est-adds-rounded-factors"),
                (est * 10, "est-estimate-zeros-miscounted"),
                (round(a, -1) * round(b, -1), "est-rounds-to-nearest-ten"),
                (_trunc1(a) * _trunc1(b), "est-truncates-instead-of-rounding"),
                (est // 10, "est-estimate-zeros-miscounted"),
            ],
        )
        part_b = mcq_part(
            phrase(rng, _VERDICT_ASK, s=_money(ctx, stated)),
            key,
            1,
            [f"The estimate is {est}. The stated answer, {stated}, is {_ratio_words(kind)} the estimate."],
            list(_WRONG_FOR[key]),
        )
        part_c = number_part(
            phrase(rng, _EXACT_ASK, total=ctx["total"], unit=ctx["unit"]),
            exact,
            1,
            [f"{a} × {b} = {exact}."],
            [
                (exact * 10, "est-exact-place-value-shift"),
                (a + b, "est-adds-instead-of-multiplying"),
                (est, "est-estimate-given-as-exact"),
                (_unshifted_partial(a, b), "est-partial-product-unshifted"),
                (exact // 10, "est-exact-place-value-shift"),
            ],
        )
        part_d = number_part(
            phrase(rng, _PERCENT_ASK),
            pct,
            2,
            [
                f"Difference = |{est} - {exact}| = {gap}.",
                f"Percentage of the exact value = {gap} ÷ {exact} × 100 = {pct}% to the nearest whole number.",
            ],
            [
                (round_half_up(Fraction(gap * 100, est)), "est-percent-of-estimate"),
                (round_half_up(Fraction(est * 100, exact)), "est-ratio-for-difference"),
                (gap, "est-raw-difference-for-percent"),
                (pct * 10, "est-percentage-decimal-shift"),
                (round_half_up(Fraction(pct, 10)), "est-percentage-decimal-shift"),
            ],
            follow="Abs(part_c-part_a)/part_c*100",
        )
        if None in (part_a, part_b, part_c, part_d):
            continue
        check(est != exact, "the estimate equals the exact value")
        return dict(
            text=ctx["lead"].format(a=a, b=b),
            parts=[part_a, part_b, part_c, part_d],
        )


def _ratio_words(kind: str) -> str:
    return {"ok": "close to", "large": "about 10 times", "small": "about a tenth of"}[kind]
