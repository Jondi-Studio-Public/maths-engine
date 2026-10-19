"""Repaying a student (HELP) debt from a stated repayment table: `r3e_hecs_help` (easy, medium, hard) and the exam
chain `exd_x_r3e_hecs_help`.

``hecs-help-repayment`` (``financial-discrete``, answer ``number``). Every question prints its repayment scale. Two
kinds of scale appear, and the question says which: a rate on the WHOLE repayment income (the rate of the band the
income falls in), or a rate on the slice of income ABOVE each threshold (as with tax brackets). The indexation rate
and the order of repayment and indexation are stated.

* easy: the repayment, from a whole-income table.
* medium: the balance after the repayment and then the indexation of what is left.
* hard: how many yearly repayments clear a debt that is indexed after each repayment; or the repayment under a
  slice-above-threshold scale, on a repayment income built from taxable income, reportable fringe benefits and
  reportable super.
* exam (``exd_x_r3e_hecs_help``): repayment income, repayment, the balance after repayment and the balance after
  indexation, each following through from the one before.

The number of repayments is checked a second way, from the closed form of the balance after k years.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import attempts, check, generator
from ._exd_distinct_parts import distinct_part_answers
from ._r3e_kit import PEOPLE, Banded, dec, dollars, listing, number_part, percent, single_number

_INDEX = [2, 3, 4, 5, 6]


def _whole_scale(rng):
    """(thresholds, rates): nil below the first threshold, then rates r0, r0+1, ... on the whole income."""
    first = 50000 + 1000 * rng.randint(0, 8)
    step = 1000 * rng.randint(6, 10)
    r0 = rng.choice([1, 2])
    tops = [first + step * k for k in range(5)]
    rates = [r0 + k for k in range(5)]
    return tops, rates


def _whole_rate(income, tops, rates):
    rate = Fraction(0)
    for t, r in zip(tops, rates):
        if income >= t:
            rate = Fraction(r)
    return rate


def _whole_text(tops, rates):
    rows = [(f"Below {dollars(tops[0])}", "Nil")]
    for i, (t, r) in enumerate(zip(tops, rates)):
        if i + 1 < len(tops):
            rows.append((f"{dollars(t)} up to but not including {dollars(tops[i + 1])}", f"{r}%"))
        else:
            rows.append((f"{dollars(t)} or more", f"{r}%"))
    return (
        "The compulsory repayment is a percentage of the whole repayment income, at the rate for the band the income "
        "falls in. The bands, by repayment income, are: " + listing(rows)
    )


def _slice_scale(rng):
    t1 = rng.choice([52000, 54000, 56000, 60000])
    t2 = t1 + 1000 * rng.choice([30, 35, 40, 45])
    t3 = t2 + 1000 * rng.choice([40, 50, 60])
    return Banded([(0, 0), (t1, rng.choice([10, 15])), (t2, rng.choice([17, 18])), (t3, rng.choice([20, 22]))])


def _slice_text(sc):
    return "The compulsory repayment is worked out like this, by repayment income: " + listing(sc.table_rows())


def _income_in_whole(rng, tops, rates, band):
    low = tops[band]
    high = tops[band + 1] if band + 1 < len(tops) else low + 25000
    return low + 100 * rng.randint(5, (high - low) // 100 - 5)


def _easy(rng):
    tops, rates = _whole_scale(rng)
    band = rng.randint(0, 3)
    income = _income_in_whole(rng, tops, rates, band)
    rate = _whole_rate(income, tops, rates)
    repayment = Fraction(income) * rate / 100
    check(rate == rates[band], "band disagrees")
    name, job = rng.choice(PEOPLE)
    text = (
        f"{name} is {job} with a HELP debt and a repayment income of {dollars(income)} for the year. "
        f"{_whole_text(tops, rates)} How much is {name}'s compulsory repayment for the year? Give your answer in "
        "dollars, to the nearest cent."
    )
    lower = rates[band - 1] if band else 0
    higher = rates[band + 1] if band + 1 < len(rates) else rates[band] + 1
    slips = [
        (Fraction(income) * (lower or 1) / 100, "r3e-lower-band-rate-used"),
        (Fraction(income) * higher / 100, "r3e-next-band-rate-used"),
        (Fraction(income - tops[band]) * rate / 100, "r3e-rate-on-excess-over-threshold"),
        (Fraction(income) * rate / 1000, "r3e-percent-decimal-slip"),
        (Fraction(income) * sum(rates[: band + 1]) / 100, "r3e-band-rates-summed"),
    ]
    working = [f"The income falls in the {rate}% band.", f"{rate}% of {dollars(income)} = {dollars(repayment, 2)}."]
    return single_number(text, repayment, 2, working, slips)


def _medium(rng):
    tops, rates = _whole_scale(rng)
    band = rng.randint(0, 3)
    income = _income_in_whole(rng, tops, rates, band)
    rate = _whole_rate(income, tops, rates)
    repayment = Fraction(income) * rate / 100
    debt = 500 * rng.randint(24, 120)
    idx = rng.choice(_INDEX)
    after = debt - repayment
    final = after * (100 + idx) / 100
    name, job = rng.choice(PEOPLE)
    text = (
        f"{name} is {job} with a HELP debt of {dollars(debt)} and a repayment income of {dollars(income)} for the year. "
        f"{_whole_text(tops, rates)} After {name} makes the compulsory repayment, the balance that remains is "
        f"indexed by {idx}%. What is the debt after the repayment and the indexation? Give your answer in dollars, "
        "to the nearest cent."
    )
    slips = [
        (debt * (100 + idx) / 100 - repayment, "r3e-indexation-before-repayment"),
        (after, "r3e-indexation-left-out"),
        (debt * (100 + idx) / 100, "r3e-repayment-left-out"),
        (after + debt * idx / 100, "r3e-indexation-on-original-debt"),
        (after + repayment * idx / 100, "r3e-indexation-on-repayment"),
    ]
    working = [
        f"Repayment = {rate}% of {dollars(income)} = {dollars(repayment, 2)}.",
        f"Balance after repayment = {dollars(debt)} - {dollars(repayment, 2)} = {dollars(after, 2)}.",
        f"After indexation = {dollars(after, 2)} x {dec(Fraction(100 + idx, 100))} = {dollars(final, 2)}.",
    ]
    return single_number(text, final, 2, working, slips)


def _repayments_needed(debt, repayment, idx):
    g = 1 + Fraction(idx, 100)
    balance = Fraction(debt)
    for n in range(1, 61):
        balance -= repayment
        if balance <= 0:
            break
        balance *= g
    else:
        raise AssertionError("debt does not clear")
    for k in range(n):  # closed form: balance after k full years, then does repayment k+1 clear it?
        after_k = Fraction(debt) * g**k - repayment * g * (g**k - 1) / (g - 1)
        cleared = after_k <= repayment
        check(cleared == (k == n - 1), "closed form disagrees with the run")
    return n


def _hard_years(rng):
    for _ in attempts(500, "a debt that clears in a sensible number of years"):
        debt = 500 * rng.randint(16, 90)
        repayment = 500 * rng.randint(2, 8)
        idx = rng.choice(_INDEX)
        if repayment <= debt * idx / 100 * 1.3:
            continue
        n = _repayments_needed(debt, repayment, idx)
        if 4 <= n <= 12:
            break
    name, job = rng.choice(PEOPLE)
    text = (
        f"{name} is {job} with a HELP debt of {dollars(debt)}. {name}'s income will stay about the same, so the "
        f"compulsory repayment will be {dollars(repayment)} each year. After each yearly repayment, the balance that "
        f"remains is indexed by {idx}%. The last repayment can be smaller than the others. How many yearly repayments "
        "will it take to clear the debt?"
    )
    no_index = -(-debt // repayment)
    g = 1 + Fraction(idx, 100)
    balance = Fraction(debt)
    for alt in range(1, 61):  # indexed BEFORE each repayment instead
        balance = balance * g - repayment
        if balance <= 0:
            break
    slips = [
        (no_index, "r3e-indexation-ignored"),
        (n + 1, "r3e-extra-repayment-counted"),
        (n - 1, "r3e-one-repayment-too-few"),
        (alt, "r3e-indexation-before-repayment"),
        (
            -(-debt // (repayment - debt * idx // 100)) if repayment > debt * idx // 100 else n + 2,
            "r3e-indexation-on-original-debt",
        ),
    ]
    balance = Fraction(debt)
    working = []
    for year in range(1, n + 1):
        balance -= repayment
        if balance <= 0:
            working.append(f"Year {year}: the repayment of {dollars(repayment)} clears what is left.")
            break
        balance *= g
        working.append(f"Year {year}: after repayment and {idx}% indexation the debt is {dollars(balance, 2)}.")
    return single_number(text, n, 0, working, slips)


def _hard_slice(rng):
    sc = _slice_scale(rng)
    name, job = rng.choice(PEOPLE)
    band = rng.choice([1, 2, 3])
    income = 100 * rng.randint(int(sc.bands[band][0]) // 100 + 20, int(sc.bands[band][0]) // 100 + 250)
    if band + 1 < len(sc.bands):
        income = min(income, int(sc.bands[band + 1][0]) - 2000)
    fringe = 100 * rng.randint(5, 40)
    rsuper = 500 * rng.randint(2, 12)
    taxable = income - fringe - rsuper
    repayment = sc.amount(income)
    text = (
        f"{name} is {job} with a HELP debt. This year {name}'s taxable income was {dollars(taxable)}, with "
        f"{dollars(fringe)} of reportable fringe benefits and {dollars(rsuper)} of reportable super contributions. "
        "Repayment income is the sum of these three amounts. "
        f"{_slice_text(sc)} How much is {name}'s compulsory repayment? Give your answer in dollars, to the nearest cent."
    )
    r = sc.bands[sc.band_index(income)][1]
    slips = [
        (sc.amount(taxable), "r3e-taxable-income-only-used"),
        (Fraction(income) * r / 100, "r3e-rate-on-whole-income"),
        (sc.amount(income - fringe), "r3e-super-left-out-of-repayment-income"),
        (sc.amount(income - rsuper), "r3e-fringe-left-out-of-repayment-income"),
        (sc.amount(income) - sc.slice_amounts(income)[sc.band_index(income) - 1], "r3e-middle-band-left-out"),
    ]
    working = [
        f"Repayment income = {dollars(taxable)} + {dollars(fringe)} + {dollars(rsuper)} = {dollars(income)}.",
    ]
    for i, (low, rate) in enumerate(sc.bands):
        amt = sc.slice_amounts(income)[i]
        if rate and amt:
            working.append(f"{percent(rate)}% of the slice from {dollars(low)}: {dollars(amt, 2)}.")
    working.append(f"Repayment = {dollars(repayment, 2)}.")
    return single_number(text, repayment, 2, working, slips)


@generator(
    "r3e_hecs_help",
    level=9,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r3e_hecs_help(rng, tier="easy"):
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return rng.choice([_hard_years, _hard_slice])(rng)


# -- exam -------------------------------------------------------------------------------------------------------


@generator("exd_x_r3e_hecs_help", level=14, presets={"exam": dict(tier="exam")})
@distinct_part_answers
def exd_x_r3e_hecs_help(rng, tier="exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    sc = _slice_scale(rng)
    name, job = rng.choice(PEOPLE)
    band = rng.choice([1, 2, 3])
    income = 100 * rng.randint(int(sc.bands[band][0]) // 100 + 20, int(sc.bands[band][0]) // 100 + 250)
    if band + 1 < len(sc.bands):
        income = min(income, int(sc.bands[band + 1][0]) - 2000)
    fringe = 100 * rng.randint(5, 40)
    rsuper = 500 * rng.randint(2, 12)
    taxable = income - fringe - rsuper
    repayment = sc.amount(income)
    base = int(repayment // 500)
    debt = 500 * rng.randint(base + 6, base + 70)
    idx = rng.choice([2, 3, 4, 5, 6])
    after = debt - repayment
    final = after * (100 + idx) / 100
    check(repayment < debt, "repayment exceeds the debt")
    r = sc.marginal_rate(income)
    text = (
        f"{name} is {job} with a HELP debt of {dollars(debt)}. This year {name}'s taxable income was {dollars(taxable)}, "
        f"with {dollars(fringe)} of reportable fringe benefits and {dollars(rsuper)} of reportable super "
        "contributions. Repayment income is the sum of these three amounts. After the compulsory repayment is made, "
        f"the balance that remains is indexed by {idx}%. {_slice_text(sc)}"
    )
    parts = [
        number_part(
            "Find the repayment income, in dollars.",
            income,
            2,
            1,
            [f"{dollars(taxable)} + {dollars(fringe)} + {dollars(rsuper)} = {dollars(income)}."],
            [
                (taxable, "r3e-taxable-income-only-used"),
                (taxable + fringe, "r3e-super-left-out-of-repayment-income"),
                (taxable + rsuper, "r3e-fringe-left-out-of-repayment-income"),
                (taxable - fringe - rsuper, "r3e-add-ons-subtracted"),
            ],
        ),
        number_part(
            "Hence find the compulsory repayment, in dollars.",
            repayment,
            2,
            3,
            [f"Repayment = {dollars(repayment, 2)}."],
            [
                (Fraction(income) * r / 100, "r3e-rate-on-whole-income"),
                (sc.amount(taxable), "r3e-taxable-income-only-used"),
                (income - sc.bands[sc.band_index(income)][0], "r3e-rate-not-applied-to-excess"),
                (repayment - sc.slice_amounts(income)[sc.band_index(income) - 1], "r3e-middle-band-left-out"),
            ],
            follow_through=sc.expression("part_a"),
        ),
        number_part(
            "Find the balance of the debt after the repayment is made, before indexation, in dollars.",
            after,
            2,
            1,
            [f"{dollars(debt)} - {dollars(repayment, 2)} = {dollars(after, 2)}."],
            [
                (debt + repayment, "r3e-repayment-added-to-debt"),
                (repayment, "r3e-repayment-given-as-balance"),
            ],
            follow_through=f"{debt} - part_b",
        ),
        number_part(
            "Find the balance of the debt after indexation, in dollars.",
            final,
            2,
            2,
            [f"{dollars(after, 2)} x {dec(Fraction(100 + idx, 100))} = {dollars(final, 2)}."],
            [
                (after + debt * idx / 100, "r3e-indexation-on-original-debt"),
                (after + after * idx / 1000, "r3e-percent-decimal-slip"),
                (after, "r3e-indexation-left-out"),
                (debt * (100 + idx) / 100 - repayment, "r3e-indexation-before-repayment"),
            ],
            follow_through=f"part_c*{100 + idx}/100",
        ),
    ]
    return dict(text=text, parts=parts)
