"""Medium and hard tiers for ten consumer-arithmetic and statistics skills that had only an easy tier and an exam.

Topics (each has `medium` and `hard` presets; the skill keeps its existing easy source):

* ``w3_ci_tiers``        compound-interest: interest earned, sub-annual compounding; the amount to invest now, a rate
                         that changes, compound against simple.
* ``w3_si_tiers``        simple-interest: total amount, a term in months; the rate, the time or the principal.
* ``w3_wage_tiers``      calculate-wages: hours and minutes, the hourly rate, a week of shifts; casual loading, tax
                         withheld, the hours needed, two jobs.
* ``w3_ot_tiers``        calculate-overtime: weekly pay with time and a half, the overtime pay alone, double time;
                         two overtime rates, the overtime hours from a pay, the hourly rate from a pay.
* ``w3_unit_tiers``      unit-pricing: the cost of some items, the difference between two packs, the price per 100 g;
                         the saving by buying the better value, the percentage cheaper.
* ``w3_flatloan_tiers``  flat-loan-repayment: interest charged, total repaid; monthly repayment after a deposit, the
                         annual rate, the term.
* ``w3_movavg_tiers``    moving-average: the k-th 3- or 5-point mean; a missing value, the change when the window
                         moves, a centred 4-point mean.
* ``w3_mean_tiers``      calculate-mean: decimals and a frequency table; the missing value, a combined mean, the
                         mean after one more value.
* ``w3_netpath_tiers``   shortest-network-path: a six-town road list; a seven-town list, a route through a town, a
                         road closed.

Every key is built from exact `Fraction` arithmetic (or an exhaustive route search) and every tagged distractor is a
believable slip that lands at least two units of the last place away from the key. Money answers are always two decimal
places; counts and years are whole numbers; a rate is one decimal place.
"""

from __future__ import annotations

from fractions import Fraction
from math import ceil, floor

from ..core import VerificationError, attempts, check, generator
from ._w3_kit import DP1, DP2, MONEY, WHOLE, dollars, frac, pct_text
from ._w6a_kit import TooFewDistractors, enough

_CI = "compound-interest"
_SI = "simple-interest"
_WG = "calculate-wages"
_OT = "calculate-overtime"
_UP = "unit-pricing"
_FL = "flat-loan-repayment"
_MA = "moving-average"
_MN = "calculate-mean"
_NP = "shortest-network-path"

MISCONCEPTIONS = [
    # compound interest
    ("w3ci-balance-for-interest", "Gives the final balance where the interest earned was asked.", _CI),
    ("w3ci-simple-instead", "Finds simple interest, adding the same amount every year.", _CI),
    ("w3ci-one-year", "Gives one year's interest only.", _CI),
    ("w3ci-wrong-years", "Compounds for one year too many or too few.", _CI),
    (
        "w3ci-annual-rate-each-period",
        "Uses the annual rate in every compounding period instead of the rate for one period.",
        _CI,
    ),
    ("w3ci-periods-as-years", "Counts the years instead of the number of compounding periods.", _CI),
    ("w3ci-annual-compounding", "Compounds once a year although the interest is added more often.", _CI),
    ("w3ci-multiplies-by-amount", "Multiplies the target amount by the growth factor instead of dividing.", _CI),
    ("w3ci-simple-discount", "Takes the simple interest off the target amount to find the amount to invest.", _CI),
    ("w3ci-first-rate-throughout", "Uses the first rate for the whole term when the rate changes.", _CI),
    ("w3ci-average-rate", "Uses the average of the two rates for the whole term.", _CI),
    ("w3ci-difference-reversed", "Gives one kind of interest where the difference between the two was asked.", _CI),
    # simple interest
    ("w3si-compounded", "Compounds the interest every year instead of using simple interest.", _SI),
    ("w3si-adds-principal", "Adds the principal to the interest where only the interest was asked.", _SI),
    ("w3si-off-by-one", "Counts one year too many.", _SI),
    ("w3si-forgets-principal", "Gives the interest where the total amount was asked.", _SI),
    ("w3si-months-as-years", "Uses the number of months as the number of years.", _SI),
    ("w3si-month-not-divided", "Divides the months by 100 or by 10 instead of by 12.", _SI),
    ("w3si-rate-not-percent", "Leaves the rate as a whole number and does not divide it by 100.", _SI),
    ("w3si-ignores-time", "Leaves the time out of the working.", _SI),
    (
        "w3si-time-for-rate",
        "Divides the interest by the principal and by the rate but not by the time (or the reverse).",
        _SI,
    ),
    (
        "w3si-multiplies-for-inverse",
        "Multiplies by the other quantities where the interest should be divided by them.",
        _SI,
    ),
    ("w3si-total-for-interest", "Uses the total amount as the interest when working back.", _SI),
    # wages
    ("w3wg-adds-instead", "Adds the rate and the hours instead of multiplying them.", _WG),
    ("w3wg-misses-a-shift", "Leaves one day's hours out of the total.", _WG),
    ("w3wg-first-day-repeated", "Uses the first day's hours for every day.", _WG),
    ("w3wg-extra-hour", "Counts one hour too many.", _WG),
    ("w3wg-loading-only", "Gives the loading alone, not the pay including it.", _WG),
    ("w3wg-tax-as-dollars", "Takes the tax rate off as a number of dollars.", _WG),
    ("w3wg-tax-ignored", "Gives the gross pay and does not take off tax.", _WG),
    ("w3wg-rate-misread", "Divides by a rate that is a dollar out.", _WG),
    ("w3wg-rates-difference", "Subtracts the two rates instead of the two pays.", _WG),
    ("w3wg-hours-difference", "Multiplies the difference in hours by one rate.", _WG),
    ("w3wg-minutes-as-decimal", "Writes 30 minutes as 0.30 of an hour (and 15 minutes as 0.15).", _WG),
    ("w3wg-ignores-minutes", "Leaves the minutes out of the hours worked.", _WG),
    ("w3wg-rounds-up-hour", "Rounds the part hour up to a whole hour.", _WG),
    ("w3wg-rate-inverted", "Divides the hours by the pay instead of the pay by the hours.", _WG),
    ("w3wg-loading-as-percent-only", "Adds the loading as a dollar amount instead of a percentage of the rate.", _WG),
    ("w3wg-loading-ignored", "Leaves the casual loading out.", _WG),
    ("w3wg-tax-as-pay", "Gives the tax withheld where the pay after tax was asked.", _WG),
    ("w3wg-tax-added", "Adds the tax to the gross pay instead of taking it off.", _WG),
    ("w3wg-floor-hours", "Rounds the hours needed down, so the target is not reached.", _WG),
    ("w3wg-pay-as-hours", "Divides the pay by the wrong number of hours.", _WG),
    ("w3wg-single-job", "Gives the pay from one job only.", _WG),
    ("w3wg-sum-not-difference", "Adds the two pays where the difference was asked.", _WG),
    # overtime
    ("w3ot-one-hour", "Pays one hour of overtime only.", _OT),
    ("w3ot-extra-hour", "Counts one overtime hour too many.", _OT),
    ("w3ot-wrong-multiplier", "Uses double time where time and a half applies.", _OT),
    ("w3ot-all-hours-multiplied", "Applies the overtime multiplier to every hour worked.", _OT),
    ("w3ot-overtime-at-ordinary", "Pays the overtime hours at the ordinary rate.", _OT),
    ("w3ot-hours-not-subtracted", "Uses all the hours worked as the overtime hours.", _OT),
    ("w3ot-ordinary-only", "Leaves out the overtime pay.", _OT),
    ("w3ot-overtime-only", "Gives the overtime pay where the whole week's pay was asked.", _OT),
    ("w3ot-second-rate-throughout", "Pays every overtime hour at the higher rate.", _OT),
    ("w3ot-first-rate-throughout", "Pays every overtime hour at the lower rate.", _OT),
    (
        "w3ot-pay-over-rate",
        "Divides the pay by the ordinary rate only and does not remove the ordinary hours first.",
        _OT,
    ),
    (
        "w3ot-multiplier-forgotten-on-division",
        "Divides by the hours worked and ignores that some are paid at a higher rate.",
        _OT,
    ),
    ("w3ot-ordinary-as-total", "Treats the whole pay as overtime pay.", _OT),
    # unit pricing
    ("w3up-miscounts-hundreds", "Miscounts the number of 100 g lots in the packet.", _UP),
    ("w3up-unit-price-given", "Gives one of the unit prices where the difference was asked.", _UP),
    ("w3up-total-as-unit", "Gives the price of the pack where the price of one was asked.", _UP),
    ("w3up-inverted", "Divides the quantity by the price instead of the price by the quantity.", _UP),
    ("w3up-adds-instead", "Adds the unit price to itself instead of multiplying by the number of items.", _UP),
    ("w3up-difference-of-totals", "Subtracts the two pack prices instead of the two unit prices.", _UP),
    ("w3up-per-gram", "Gives the price per gram where the price per 100 grams was asked.", _UP),
    ("w3up-per-kg-as-100g", "Gives the price per kilogram where the price per 100 grams was asked.", _UP),
    ("w3up-saving-per-item", "Gives the saving on one item where the saving on all of them was asked.", _UP),
    ("w3up-wrong-pack", "Prices all the items at the dearer pack's unit price.", _UP),
    ("w3up-percent-of-cheaper", "Finds the percentage using the cheaper unit price as the starting amount.", _UP),
    ("w3up-percent-points", "Gives the difference in cents as the percentage.", _UP),
    # flat loan
    ("w3fl-compounded", "Compounds the interest instead of charging flat interest.", _FL),
    ("w3fl-one-year-of-payments", "Counts one year of repayments only.", _FL),
    ("w3fl-off-by-one", "Counts one year too many or too few.", _FL),
    ("w3fl-ignores-interest", "Divides the amount borrowed alone by the number of months.", _FL),
    ("w3fl-interest-as-total", "Gives the interest where the total repaid was asked.", _FL),
    ("w3fl-one-year-interest", "Finds one year's interest only.", _FL),
    ("w3fl-total-as-interest", "Gives the total repaid where the interest was asked.", _FL),
    ("w3fl-deposit-ignored", "Charges interest on the price although a deposit is paid first.", _FL),
    ("w3fl-deposit-not-subtracted", "Forgets to take the deposit off the price before adding interest.", _FL),
    ("w3fl-months-as-years", "Divides by the number of years instead of the number of months.", _FL),
    ("w3fl-rate-without-time", "Finds the rate without allowing for the number of years.", _FL),
    ("w3fl-rate-from-total", "Uses the total repaid, not the interest, to find the rate.", _FL),
    ("w3fl-term-in-months", "Gives the number of months where the number of years was asked.", _FL),
    ("w3fl-term-from-total", "Uses the total repaid, not the interest, to find the term.", _FL),
    # moving average
    ("w3ma-missing-wrong-window", "Multiplies the average by the wrong window size.", _MA),
    ("w3ma-wrong-window", "Averages a window of the wrong size.", _MA),
    (
        "w3ma-shifted-window",
        "Averages the window that starts at the stated point instead of the one centred on it.",
        _MA,
    ),
    ("w3ma-total-as-mean", "Gives the total of the window without dividing.", _MA),
    ("w3ma-whole-series", "Averages the whole series instead of one window.", _MA),
    ("w3ma-change-not-divided", "Gives the change in the window total without dividing by the window size.", _MA),
    (
        "w3ma-change-as-new-value",
        "Gives the difference between the values leaving and entering as the change in the mean.",
        _MA,
    ),
    ("w3ma-missing-not-from-total", "Treats the moving average as the missing value.", _MA),
    (
        "w3ma-missing-no-total",
        "Subtracts the known values from the average without multiplying it by the window size first.",
        _MA,
    ),
    ("w3ma-centred-unweighted", "Gives the end values of a centred 4-point mean the same weight as the others.", _MA),
    ("w3ma-centred-simple", "Averages only the middle four values.", _MA),
    # mean
    ("w3mn-known-mean", "Gives the mean of the known values as the missing value.", _MN),
    ("w3mn-total-needed", "Gives the total needed where the last score was asked.", _MN),
    ("w3mn-weights-swapped", "Weights each mean by the other group's size.", _MN),
    ("w3mn-new-value-left-out", "Leaves the new number out of the total.", _MN),
    ("w3mn-divides-by-wrong-count", "Divides the total by the wrong number of values.", _MN),
    ("w3mn-freq-unweighted", "Averages the scores without using the frequencies.", _MN),
    ("w3mn-freq-divides-by-rows", "Divides the weighted total by the number of rows in the table.", _MN),
    ("w3mn-freq-sums-frequencies", "Gives the total of the frequencies as the mean.", _MN),
    ("w3mn-mean-of-means", "Averages the two group means without weighting them by group size.", _MN),
    ("w3mn-missing-is-mean", "Gives the target mean as the missing value.", _MN),
    (
        "w3mn-missing-no-total",
        "Subtracts the known values from the mean without multiplying the mean by the number of values.",
        _MN,
    ),
    (
        "w3mn-missing-wrong-count",
        "Multiplies the target mean by one value too few when finding the missing total.",
        _MN,
    ),
    ("w3mn-new-value-averaged", "Averages the old mean with the new value.", _MN),
    ("w3mn-new-count-forgotten", "Divides the new total by the old number of values.", _MN),
    # network
    ("w3np-fewest-roads", "Chooses the route with the fewest roads instead of the shortest total.", _NP),
    ("w3np-greedy", "Always takes the shortest road next and does not compare whole routes.", _NP),
    ("w3np-longest", "Gives the longest route instead of the shortest.", _NP),
    ("w3np-second-shortest", "Gives the second shortest route.", _NP),
    ("w3np-skips-stop", "Ignores that the route must pass through the stated town.", _NP),
    ("w3np-two-legs-shortest-sum-error", "Joins the two shortest legs even when they overlap or share a road.", _NP),
    ("w3np-closed-ignored", "Keeps using the road that has been closed.", _NP),
    (
        "w3np-drops-the-detour",
        "Takes the closed road's length off the old shortest route instead of finding a new route.",
        _NP,
    ),
]


def _draw(rng, pool):
    """One question from a random draw function in `pool`, redrawing when a draw is degenerate (a rounding near-tie, a
    key that is not in whole cents, too few distractors)."""
    for _ in attempts(200, "a usable draw"):
        try:
            return rng.choice(pool)(rng)
        except (TooFewDistractors, VerificationError):
            continue
    raise AssertionError  # pragma: no cover


def _yrs(n):
    return f"{n} year" if n == 1 else f"{n} years"


def _hhmm(h, m):
    return f"{h} h {m} min" if m else f"{h} h"


# -- compound interest ------------------------------------------------------------------------------------------------


def _growth(rate_pct, periods):
    return (1 + frac(rate_pct) / 100) ** periods


def _rate_text(r):
    return pct_text(r)


_CI_RATES = (2, 3, 4, 5, 6, 8, Fraction(5, 2), Fraction(7, 2), Fraction(9, 2))


def _ci_interest(rng):
    p = rng.randint(2, 40) * 250
    r = rng.choice(_CI_RATES)
    n = rng.randint(2, 4)
    bal = p * _growth(r, n)
    interest = bal - p
    d = MONEY.dist(
        interest,
        [
            (bal, "w3ci-balance-for-interest"),
            (p * frac(r) / 100 * n, "w3ci-simple-instead"),
            (p * frac(r) / 100, "w3ci-one-year"),
            (p * _growth(r, n + 1) - p, "w3ci-wrong-years"),
            (p * _growth(r, n - 1) - p, "w3ci-wrong-years"),
        ],
    )
    enough(d, 3)
    text = (
        f"{dollars(p)} is invested at {_rate_text(r)}% p.a., with the interest compounded annually for {n} years. "
        f"How much interest is earned in total? Give your answer to the nearest cent."
    )
    working = [
        f"Balance = {dollars(p)} × (1 + {_rate_text(r)}/100)^{n} = ${MONEY.text(bal)}.",
        f"Interest = balance − principal = ${MONEY.text(interest)}.",
    ]
    return dict(text=text, working=working, distractors=d, **MONEY.fields(interest), answer_kind="number")


_FREQ = (("half-yearly", 2), ("quarterly", 4), ("monthly", 12))


def _ci_periods(rng):
    p = rng.randint(2, 40) * 250
    name, m = rng.choice(_FREQ[:2])
    r = rng.choice((4, 6, 8, 10, 12))
    n = rng.randint(1, 3)
    per = frac(r) / m
    bal = p * (1 + per / 100) ** (m * n)
    d = MONEY.dist(
        bal,
        [
            (p * _growth(r, n), "w3ci-annual-compounding"),
            (p * _growth(r, m * n), "w3ci-annual-rate-each-period"),
            (p * (1 + per / 100) ** n, "w3ci-periods-as-years"),
            (p * (1 + per / 100) ** (m * n + 1), "w3ci-wrong-years"),
        ],
    )
    enough(d, 3)
    text = (
        f"{dollars(p)} is invested at {r}% p.a., with the interest compounded {name}. What is the balance after {n} "
        f"{'year' if n == 1 else 'years'}? Give your answer to the nearest cent."
    )
    working = [
        f"Rate per period = {r}% ÷ {m} = {pct_text(per) if (per * 10).denominator == 1 else float(per)}%; periods = {m} × {n} = {m * n}.",
        f"Balance = ${MONEY.text(bal)}.",
    ]
    return dict(text=text, working=working, distractors=d, **MONEY.fields(bal), answer_kind="number")


def _ci_present(rng):
    r = rng.choice((10, 20))
    n = rng.randint(2, 3) if r == 10 else 2
    p = rng.randint(2, 30) * 1000
    amount = p * _growth(r, n)
    check((amount * 100).denominator == 1, "target not a whole number of cents")
    d = WHOLE.dist(
        p,
        [
            (amount * _growth(r, n), "w3ci-multiplies-by-amount"),
            (amount - amount * frac(r) / 100 * n, "w3ci-simple-discount"),
            (amount / _growth(r, n - 1), "w3ci-wrong-years"),
            (amount / _growth(r, n + 1), "w3ci-wrong-years"),
        ],
    )
    enough(d, 3)
    text = (
        f"Mia wants {dollars(amount)} in {n} years. The account pays {r}% p.a., with the interest compounded annually. "
        f"How much must she invest now? Give your answer to the nearest dollar."
    )
    working = [f"Amount = P × (1 + {r}/100)^{n}, so P = {dollars(amount)} ÷ {_growth(r, n)} = {dollars(p)}."]
    return dict(text=text, working=working, distractors=d, **WHOLE.fields(p), answer_kind="number")


def _ci_two_rates(rng):
    p = rng.randint(2, 40) * 250
    r1, r2 = rng.sample((2, 3, 4, 5, 6, 8), 2)
    n1, n2 = rng.randint(1, 3), rng.randint(1, 3)
    bal = p * _growth(r1, n1) * _growth(r2, n2)
    d = MONEY.dist(
        bal,
        [
            (p * _growth(r1, n1 + n2), "w3ci-first-rate-throughout"),
            (p * _growth(Fraction(r1 + r2, 2), n1 + n2), "w3ci-average-rate"),
            (p * _growth(r2, n1 + n2), "w3ci-first-rate-throughout"),
            (p * _growth(r1, n1) * _growth(r2, n2 + 1), "w3ci-wrong-years"),
        ],
    )
    enough(d, 3)
    text = (
        f"{dollars(p)} is invested with the interest compounded annually. The rate is {r1}% p.a. for the first {n1} "
        f"{'year' if n1 == 1 else 'years'} and {r2}% p.a. for the next {n2} {'year' if n2 == 1 else 'years'}. "
        f"What is the balance at the end? Give your answer to the nearest cent."
    )
    working = [
        f"After {_yrs(n1)}: {dollars(p)} × (1 + {r1}/100)^{n1}.",
        f"Then multiply by (1 + {r2}/100)^{n2} to get ${MONEY.text(bal)}.",
    ]
    return dict(text=text, working=working, distractors=d, **MONEY.fields(bal), answer_kind="number")


def _ci_vs_simple(rng):
    p = rng.randint(2, 40) * 500
    r = rng.choice((3, 4, 5, 6, 8, 10))
    n = rng.randint(3, 6)
    comp = p * _growth(r, n) - p
    simp = p * frac(r) / 100 * n
    diff = comp - simp
    d = MONEY.dist(
        diff,
        [
            (comp, "w3ci-difference-reversed"),
            (simp, "w3ci-difference-reversed"),
            (p * _growth(r, n + 1) - p - simp, "w3ci-wrong-years"),
            (p * _growth(r, n - 1) - p - simp, "w3ci-wrong-years"),
        ],
    )
    enough(d, 3)
    text = (
        f"{dollars(p)} is invested for {n} years at {r}% p.a. Account A pays simple interest and Account B pays compound "
        f"interest, compounded annually. How much more interest does Account B earn than Account A? "
        f"Give your answer to the nearest cent."
    )
    working = [
        f"Simple interest = ${MONEY.text(simp)}; compound interest = ${MONEY.text(comp)}.",
        f"Difference = ${MONEY.text(diff)}.",
    ]
    return dict(text=text, working=working, distractors=d, **MONEY.fields(diff), answer_kind="number")


@generator(
    "w3_ci_tiers",
    level=11,
    presets={"medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def w3_ci_tiers(rng, tier="medium"):
    """Compound interest, medium and hard."""
    pool = {"medium": (_ci_interest, _ci_periods), "hard": (_ci_present, _ci_two_rates, _ci_vs_simple)}.get(tier)
    if pool is None:
        raise ValueError(f"unknown tier {tier!r}")
    return _draw(rng, pool)


# -- simple interest --------------------------------------------------------------------------------------------------


def _si_amount(rng):
    p = rng.randint(2, 40) * 250
    r = rng.choice((3, 4, 5, 6, 8, Fraction(7, 2), Fraction(9, 2)))
    n = rng.randint(2, 6)
    i = p * frac(r) / 100 * n
    total = p + i
    d = MONEY.dist(
        total,
        [
            (i, "w3si-forgets-principal"),
            (p * frac(r) * n, "w3si-rate-not-percent"),
            (p + p * frac(r) / 100, "w3si-ignores-time"),
            (p * (1 + frac(r) / 100) ** n, "w3si-compounded"),
        ],
    )
    enough(d, 3)
    text = (
        f"{dollars(p)} is invested at {pct_text(r)}% p.a. simple interest for {n} years. "
        f"What is the total amount at the end of the {n} years?"
    )
    working = [
        f"I = Prt = {dollars(p)} × {pct_text(r)}/100 × {n} = {dollars(i)}.",
        f"Total = {dollars(p)} + {dollars(i)} = {dollars(total)}.",
    ]
    return dict(text=text, working=working, distractors=d, **MONEY.fields(total), answer_kind="number")


def _si_months(rng):
    p = rng.randint(2, 40) * 250
    r = rng.choice((3, 4, 5, 6, 8, 9))
    months = rng.choice((6, 9, 15, 18, 21, 30, 42))
    i = p * frac(r) / 100 * Fraction(months, 12)
    d = MONEY.dist(
        i,
        [
            (p * frac(r) / 100 * months, "w3si-months-as-years"),
            (p * frac(r) / 100 * Fraction(months, 10), "w3si-month-not-divided"),
            (p * frac(r) / 100, "w3si-ignores-time"),
            (p * frac(r) / 100 * Fraction(months, 12) + p, "w3si-adds-principal"),
        ],
    )
    enough(d, 3)
    text = f"Find the simple interest earned on {dollars(p)} at {r}% p.a. for {months} months."
    working = [
        f"Time = {months} ÷ 12 = {float(Fraction(months, 12))} years.",
        f"I = {dollars(p)} × {r}/100 × {months}/12 = ${MONEY.text(i)}.",
    ]
    return dict(text=text, working=working, distractors=d, **MONEY.fields(i), answer_kind="number")


def _si_rate(rng):
    p = rng.randint(2, 40) * 500
    r = rng.choice([Fraction(k, 2) for k in range(5, 21)])
    n = rng.randint(2, 6)
    i = p * r / 100 * n
    d = DP1.dist(
        r,
        [
            (i * 100 / p, "w3si-ignores-time"),
            (i / (p * n), "w3si-rate-not-percent"),
            ((i + p) * 100 / (p * n), "w3si-total-for-interest"),
            (i * 100 / (p * (n + 1)), "w3si-off-by-one"),
        ],
        extremes=1,
    )
    enough(d, 3)
    text = (
        f"{dollars(p)} earns {dollars(i)} simple interest over {n} years. "
        f"What is the annual interest rate, as a percentage? Give your answer to 1 decimal place."
    )
    working = [f"I = Prt, so r = I ÷ (P × t) = {dollars(i)} ÷ ({dollars(p)} × {n}) = {pct_text(r)}%."]
    return dict(text=text, working=working, distractors=d, **DP1.fields(r), answer_kind="number")


def _si_time(rng):
    p = rng.randint(2, 40) * 500
    r = rng.choice((3, 4, 5, 6, 8, 10))
    n = rng.randint(2, 9)
    i = p * frac(r) / 100 * n
    d = WHOLE.dist(
        n,
        [
            (i / (p * frac(r) / 100) * 12, "w3si-month-not-divided"),
            ((i + p) / (p * frac(r) / 100), "w3si-total-for-interest"),
            (n + 1, "w3si-off-by-one"),
            (n - 1, "w3si-off-by-one"),
        ],
    )
    enough(d, 3)
    text = (
        f"{dollars(p)} is invested at {r}% p.a. simple interest and earns {dollars(i)}. "
        f"For how many years was it invested?"
    )
    working = [f"t = I ÷ (P × r) = {dollars(i)} ÷ ({dollars(p)} × {r}/100) = {n} years."]
    return dict(text=text, working=working, distractors=d, **WHOLE.fields(n), answer_kind="number")


def _si_principal(rng):
    p = rng.randint(2, 60) * 250
    r = rng.choice((3, 4, 5, 6, 8))
    n = rng.randint(2, 6)
    i = p * frac(r) / 100 * n
    d = WHOLE.dist(
        p,
        [
            (i * frac(r) * n / 100, "w3si-multiplies-for-inverse"),
            (i * 100 / r, "w3si-ignores-time"),
            (i * 100 / (r * (n + 1)), "w3si-off-by-one"),
            (i / (frac(r) * n), "w3si-rate-not-percent"),
        ],
    )
    enough(d, 3)
    text = (
        f"An investment at {r}% p.a. simple interest earns {dollars(i)} in {n} years. "
        f"How much was invested? Give your answer to the nearest dollar."
    )
    working = [f"P = I ÷ (r × t) = {dollars(i)} ÷ ({r}/100 × {n}) = {dollars(p)}."]
    return dict(text=text, working=working, distractors=d, **WHOLE.fields(p), answer_kind="number")


@generator("w3_si_tiers", level=11, presets={"medium": dict(tier="medium"), "hard": dict(tier="hard")})
def w3_si_tiers(rng, tier="medium"):
    """Simple interest, medium and hard."""
    pool = {"medium": (_si_amount, _si_months), "hard": (_si_rate, _si_time, _si_principal)}.get(tier)
    if pool is None:
        raise ValueError(f"unknown tier {tier!r}")
    return _draw(rng, pool)


# -- wages ------------------------------------------------------------------------------------------------------------


def _wg_hm(rng):
    if rng.random() < 0.5:
        rate = Fraction(rng.randint(15, 45))
        minutes = rng.choice((15, 30, 45))
    else:
        rate = Fraction(rng.randint(30, 90), 2)
        minutes = 30
    h = rng.randint(3, 9)
    hours = h + Fraction(minutes, 60)
    pay = rate * hours
    d = MONEY.dist(
        pay,
        [
            (rate * (h + Fraction(minutes, 100)), "w3wg-minutes-as-decimal"),
            (rate * h, "w3wg-ignores-minutes"),
            (rate * (h + 1), "w3wg-rounds-up-hour"),
            (rate + h + Fraction(minutes, 60), "w3wg-adds-instead"),
        ],
    )
    enough(d, 3)
    text = f"Sam is paid {dollars(rate)} per hour and works for {_hhmm(h, minutes)}. How much does Sam earn?"
    working = [
        f"Hours = {h} + {minutes}/60 = {float(hours)}.",
        f"Pay = {dollars(rate)} × {float(hours)} = {dollars(pay)}.",
    ]
    return dict(text=text, working=working, distractors=d, **MONEY.fields(pay), answer_kind="number")


def _wg_rate(rng):
    rate = Fraction(rng.randint(60, 180), 4)
    hours = rng.randint(5, 38)
    pay = rate * hours
    d = MONEY.dist(
        rate,
        [
            (pay / (hours + 1), "w3wg-pay-as-hours"),
            (pay / (hours - 1), "w3wg-pay-as-hours"),
            (pay / (hours + 2), "w3wg-pay-as-hours"),
            (pay / (hours - 2), "w3wg-pay-as-hours"),
        ],
    )
    enough(d, 3)
    text = f"Ali earned {dollars(pay)} for {hours} hours of work. What is Ali's hourly rate of pay?"
    working = [f"Rate = {dollars(pay)} ÷ {hours} = ${MONEY.text(rate)} per hour."]
    return dict(text=text, working=working, distractors=d, **MONEY.fields(rate), answer_kind="number")


def _wg_week(rng):
    rate = Fraction(rng.randint(15, 40))
    days = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")
    k = rng.randint(3, 5)
    chosen = days[:k]
    hrs = [Fraction(rng.randint(8, 18), 2) for _ in chosen]
    total = sum(hrs)
    pay = rate * total
    shifts = ", ".join(f"{day} {float(h):g} h" for day, h in zip(chosen, hrs))
    d = MONEY.dist(
        pay,
        [
            (rate * (total - hrs[0]), "w3wg-misses-a-shift"),
            (rate * (total + 1), "w3wg-extra-hour"),
            (rate + total, "w3wg-adds-instead"),
            (rate * hrs[0] * k, "w3wg-first-day-repeated"),
        ],
    )
    enough(d, 3)
    text = f"Jo is paid {dollars(rate)} per hour. This week Jo worked: {shifts}. What is Jo's pay for the week?"
    working = [f"Total hours = {float(total):g}.", f"Pay = {dollars(rate)} × {float(total):g} = {dollars(pay)}."]
    return dict(text=text, working=working, distractors=d, **MONEY.fields(pay), answer_kind="number")


def _wg_casual(rng):
    base = Fraction(rng.randint(16, 32))
    loading = rng.choice((20, 25))
    hours = rng.randint(6, 30)
    pay = base * (1 + Fraction(loading, 100)) * hours
    d = MONEY.dist(
        pay,
        [
            (base * hours, "w3wg-loading-ignored"),
            ((base + loading) * hours, "w3wg-loading-as-percent-only"),
            (base * hours * Fraction(loading, 100), "w3wg-loading-only"),
            (base * (1 + Fraction(loading, 100)) * (hours + 1), "w3wg-extra-hour"),
        ],
    )
    enough(d, 3)
    text = (
        f"A casual worker's base rate is {dollars(base)} per hour, plus a {loading}% casual loading. "
        f"How much does the worker earn in {hours} hours?"
    )
    working = [
        f"Hourly rate with loading = {dollars(base)} × {1 + Fraction(loading, 100)} = ${MONEY.text(base * (1 + Fraction(loading, 100)))}.",
        f"Pay = hourly rate × {hours} = {dollars(pay)}.",
    ]
    return dict(text=text, working=working, distractors=d, **MONEY.fields(pay), answer_kind="number")


def _wg_net(rng):
    rate = Fraction(rng.randint(18, 42))
    hours = rng.randint(20, 38)
    tax = rng.choice((10, 15, 19, 20, 25, 30))
    gross = rate * hours
    net = gross * (1 - Fraction(tax, 100))
    d = MONEY.dist(
        net,
        [
            (gross * Fraction(tax, 100), "w3wg-tax-as-pay"),
            (gross * (1 + Fraction(tax, 100)), "w3wg-tax-added"),
            (gross, "w3wg-tax-ignored"),
            (gross - tax, "w3wg-tax-as-dollars"),
        ],
    )
    enough(d, 3)
    text = (
        f"Pat earns {dollars(rate)} per hour and works {hours} hours. {tax}% of the gross pay is withheld as tax. "
        f"How much does Pat take home?"
    )
    working = [
        f"Gross = {dollars(gross)}.",
        f"Tax = {tax}% of {dollars(gross)}; take-home = {dollars(gross)} × {1 - Fraction(tax, 100)} = {dollars(net)}.",
    ]
    return dict(text=text, working=working, distractors=d, **MONEY.fields(net), answer_kind="number")


def _wg_hours_needed(rng):
    rate = Fraction(rng.randint(15, 40))
    hours = rng.randint(5, 18)
    target = rate * hours - rng.randint(1, int(rate) - 1)
    need = ceil(target / rate)
    check(target / rate != int(target / rate), "exact")
    d = WHOLE.dist(
        need,
        [
            (floor(target / rate), "w3wg-floor-hours"),
            (need + 1, "w3wg-extra-hour"),
            (target / (rate - 1), "w3wg-rate-misread"),
            (target / (rate + 1), "w3wg-rate-misread"),
        ],
    )
    enough(d, 3)
    text = (
        f"Kim is paid {dollars(rate)} per hour and wants to earn at least {dollars(target)}. "
        f"What is the smallest whole number of hours Kim must work?"
    )
    working = [
        f"{dollars(target)} ÷ {dollars(rate)} = {float(target / rate):.2f} hours.",
        f"Rounded up to a whole number of hours: {need}.",
    ]
    return dict(text=text, working=working, distractors=d, **WHOLE.fields(need), answer_kind="number")


def _wg_two_jobs(rng):
    r1, r2 = Fraction(rng.randint(18, 30)), Fraction(rng.randint(24, 44))
    h1, h2 = rng.randint(10, 24), rng.randint(6, 18)
    p1, p2 = r1 * h1, r2 * h2
    diff = abs(p1 - p2)
    check(diff > 10, "pays too close")
    name1, name2 = ("the cafe", "the warehouse")
    d = MONEY.dist(
        diff,
        [
            (p1 + p2, "w3wg-sum-not-difference"),
            (p1 if p1 > p2 else p2, "w3wg-single-job"),
            (abs(r1 - r2), "w3wg-rates-difference"),
            (abs(h1 - h2) * r1, "w3wg-hours-difference"),
        ],
    )
    enough(d, 3)
    text = (
        f"Lee is paid {dollars(r1)} per hour at {name1} and works {h1} hours there. Lee is paid {dollars(r2)} per hour at "
        f"{name2} and works {h2} hours there. How much more does Lee earn from the job that pays more in total?"
    )
    working = [f"{name1.capitalize()}: {dollars(p1)}; {name2}: {dollars(p2)}.", f"Difference = {dollars(diff)}."]
    return dict(text=text, working=working, distractors=d, **MONEY.fields(diff), answer_kind="number")


@generator("w3_wage_tiers", level=11, presets={"medium": dict(tier="medium"), "hard": dict(tier="hard")})
def w3_wage_tiers(rng, tier="medium"):
    """Wages, medium and hard."""
    pool = {"medium": (_wg_hm, _wg_rate, _wg_week), "hard": (_wg_casual, _wg_net, _wg_hours_needed, _wg_two_jobs)}.get(
        tier
    )
    if pool is None:
        raise ValueError(f"unknown tier {tier!r}")
    return _draw(rng, pool)


# -- overtime ---------------------------------------------------------------------------------------------------------


def _ot_rate(rng):
    return Fraction(rng.randint(30, 80), 2)


def _ot_total(rng):
    rate = _ot_rate(rng)
    o = rng.randint(2, 10)
    ordinary = 38
    pay = rate * ordinary + rate * Fraction(3, 2) * o
    d = MONEY.dist(
        pay,
        [
            (rate * (ordinary + o) * Fraction(3, 2), "w3ot-all-hours-multiplied"),
            (rate * (ordinary + o), "w3ot-overtime-at-ordinary"),
            (rate * ordinary, "w3ot-ordinary-only"),
            (rate * ordinary + rate * Fraction(3, 2), "w3ot-one-hour"),
        ],
    )
    enough(d, 3)
    text = (
        f"Dee's ordinary rate is {dollars(rate)} per hour for the first 38 hours of the week. Hours after that are paid at "
        f"time and a half. Dee works {ordinary + o} hours. What is Dee's pay for the week?"
    )
    working = [
        f"Ordinary pay = 38 × {dollars(rate)} = {dollars(rate * 38)}.",
        f"Overtime = {o} h × 1.5 × {dollars(rate)} = {dollars(rate * Fraction(3, 2) * o)}.",
        f"Total = {dollars(pay)}.",
    ]
    return dict(text=text, working=working, distractors=d, **MONEY.fields(pay), answer_kind="number")


def _ot_only(rng):
    rate = _ot_rate(rng)
    worked = rng.randint(40, 50)
    o = worked - 38
    ot = rate * Fraction(3, 2) * o
    d = MONEY.dist(
        ot,
        [
            (rate * Fraction(3, 2) * worked, "w3ot-all-hours-multiplied"),
            (rate * o, "w3ot-overtime-at-ordinary"),
            (rate * 38 + ot, "w3ot-overtime-only"),
            (rate * Fraction(3, 2) * (o + 1), "w3ot-extra-hour"),
        ],
    )
    enough(d, 3)
    text = (
        f"A worker earns {dollars(rate)} per hour for the first 38 hours and time and a half after that. "
        f"This week the worker is on duty for {worked} hours. How much of the week's pay is overtime pay?"
    )
    working = [f"Overtime hours = {worked} − 38 = {o}.", f"Overtime pay = {o} × 1.5 × {dollars(rate)} = {dollars(ot)}."]
    return dict(text=text, working=working, distractors=d, **MONEY.fields(ot), answer_kind="number")


def _ot_double(rng):
    rate = _ot_rate(rng)
    base_h = rng.randint(30, 38)
    sun = rng.randint(3, 8)
    pay = rate * base_h + rate * 2 * sun
    d = MONEY.dist(
        pay,
        [
            (rate * (base_h + sun) * 2, "w3ot-all-hours-multiplied"),
            (rate * (base_h + sun), "w3ot-overtime-at-ordinary"),
            (rate * base_h, "w3ot-ordinary-only"),
            (rate * base_h + rate * Fraction(3, 2) * sun, "w3ot-first-rate-throughout"),
        ],
    )
    enough(d, 3)
    text = (
        f"Ravi earns {dollars(rate)} per hour from Monday to Saturday. Sunday work is paid at double time. "
        f"Ravi works {base_h} hours from Monday to Saturday and {sun} hours on Sunday. What is Ravi's total pay?"
    )
    working = [
        f"Monday to Saturday: {base_h} × {dollars(rate)} = {dollars(rate * base_h)}.",
        f"Sunday: {sun} × 2 × {dollars(rate)} = {dollars(rate * 2 * sun)}.",
        f"Total = {dollars(pay)}.",
    ]
    return dict(text=text, working=working, distractors=d, **MONEY.fields(pay), answer_kind="number")


def _ot_tiered(rng):
    rate = _ot_rate(rng)
    o = rng.randint(5, 10)
    first = 3
    pay = rate * 38 + rate * Fraction(3, 2) * first + rate * 2 * (o - first)
    d = MONEY.dist(
        pay,
        [
            (rate * 38 + rate * 2 * o, "w3ot-second-rate-throughout"),
            (rate * 38 + rate * Fraction(3, 2) * o, "w3ot-first-rate-throughout"),
            (rate * 38 + rate * (o), "w3ot-overtime-at-ordinary"),
            (rate * (38 + o) * 2, "w3ot-all-hours-multiplied"),
        ],
    )
    enough(d, 3)
    text = (
        f"A worker's rate is {dollars(rate)} per hour for 38 hours. The next 3 hours are paid at time and a half and every "
        f"hour after that at double time. The worker works {38 + o} hours in the week. What is the week's pay?"
    )
    working = [
        f"Ordinary: {dollars(rate * 38)}.",
        f"Time and a half: 3 × 1.5 × {dollars(rate)} = {dollars(rate * Fraction(9, 2))}.",
        f"Double time: {o - first} × 2 × {dollars(rate)} = {dollars(rate * 2 * (o - first))}.",
        f"Total = {dollars(pay)}.",
    ]
    return dict(text=text, working=working, distractors=d, **MONEY.fields(pay), answer_kind="number")


def _ot_hours_back(rng):
    rate = _ot_rate(rng)
    o = rng.randint(2, 12)
    pay = rate * 38 + rate * Fraction(3, 2) * o
    d = WHOLE.dist(
        o,
        [
            (pay / rate - 38, "w3ot-pay-over-rate"),
            (pay / rate, "w3ot-pay-over-rate"),
            (o + 38, "w3ot-hours-not-subtracted"),
            ((pay - rate * 38) / rate, "w3ot-overtime-at-ordinary"),
            (o + 1, "w3ot-extra-hour"),
        ],
    )
    enough(d, 3)
    text = (
        f"A worker earns {dollars(rate)} per hour for the first 38 hours of the week and time and a half after that. "
        f"One week the worker is paid {dollars(pay)}. How many hours of overtime were worked?"
    )
    working = [
        f"Ordinary pay = {dollars(rate * 38)}.",
        f"Overtime pay = {dollars(pay)} − {dollars(rate * 38)} = {dollars(rate * Fraction(3, 2) * o)}.",
        f"Overtime hours = {dollars(rate * Fraction(3, 2) * o)} ÷ ({dollars(rate)} × 1.5) = {o}.",
    ]
    return dict(text=text, working=working, distractors=d, **WHOLE.fields(o), answer_kind="number")


def _ot_rate_back(rng):
    rate = Fraction(rng.randint(60, 160), 4)
    o = rng.randint(2, 12)
    pay = rate * 38 + rate * Fraction(3, 2) * o
    check((pay * 100).denominator == 1, "pay not whole cents")
    d = MONEY.dist(
        rate,
        [
            (pay / (38 + o), "w3ot-multiplier-forgotten-on-division"),
            (pay / 38, "w3ot-ordinary-only"),
            (pay / (38 + 2 * o), "w3ot-wrong-multiplier"),
            (pay / (38 + Fraction(3, 2) * (o + 1)), "w3ot-extra-hour"),
        ],
    )
    enough(d, 3)
    text = (
        f"A worker is paid {dollars(pay)} for a week of {38 + o} hours. The first 38 hours are at the ordinary rate and "
        f"the other {o} are at time and a half. What is the ordinary hourly rate?"
    )
    working = [
        f"Pay = rate × (38 + 1.5 × {o}) = rate × {38 + Fraction(3, 2) * o}.",
        f"Rate = {dollars(pay)} ÷ {38 + Fraction(3, 2) * o} = ${MONEY.text(rate)}.",
    ]
    return dict(text=text, working=working, distractors=d, **MONEY.fields(rate), answer_kind="number")


@generator("w3_ot_tiers", level=11, presets={"medium": dict(tier="medium"), "hard": dict(tier="hard")})
def w3_ot_tiers(rng, tier="medium"):
    """Overtime, medium and hard."""
    pool = {"medium": (_ot_total, _ot_only, _ot_double), "hard": (_ot_tiered, _ot_hours_back, _ot_rate_back)}.get(tier)
    if pool is None:
        raise ValueError(f"unknown tier {tier!r}")
    return _draw(rng, pool)


# -- unit pricing -----------------------------------------------------------------------------------------------------

_ITEMS = ("pens", "muffins", "apples", "tickets", "batteries", "bottles")


def _up_cost_for_n(rng):
    k = rng.choice((4, 6, 8, 10, 12, 20))
    unit = Fraction(rng.randint(8, 80), 20)
    price = unit * k
    n = rng.choice([x for x in (5, 7, 9, 15, 18, 25, 30, 36) if x != k])
    item = rng.choice(_ITEMS)
    cost = unit * n
    d = MONEY.dist(
        cost,
        [
            (price, "w3up-total-as-unit"),
            (Fraction(k) / price * n, "w3up-inverted"),
            (price + n, "w3up-adds-instead"),
            (unit * (n + k), "w3up-adds-instead"),
        ],
    )
    enough(d, 3)
    text = f"A pack of {k} {item} costs {dollars(price)}. At the same price per item, how much do {n} {item} cost?"
    working = [
        f"Price of one = {dollars(price)} ÷ {k} = ${MONEY.text(unit)}.",
        f"{n} × ${MONEY.text(unit)} = {dollars(cost)}.",
    ]
    return dict(text=text, working=working, distractors=d, **MONEY.fields(cost), answer_kind="number")


def _up_difference(rng):
    k1, k2 = rng.sample((4, 5, 8, 10, 12, 20, 25), 2)
    u1 = Fraction(rng.randint(40, 120), 100)
    u2 = u1 + Fraction(rng.choice((-1, 1)) * rng.randint(5, 30), 100)
    p1, p2 = u1 * k1, u2 * k2
    check(u2 > 0 and u1 != u2, "units")
    diff_cents = abs(u1 - u2) * 100
    item = rng.choice(_ITEMS)
    d = WHOLE.dist(
        diff_cents,
        [
            (abs(p1 - p2) * 100, "w3up-difference-of-totals"),
            (u1 * 100, "w3up-unit-price-given"),
            (u2 * 100, "w3up-unit-price-given"),
        ],
    )
    enough(d, 3)
    text = (
        f"A pack of {k1} {item} costs {dollars(p1)}. A pack of {k2} {item} costs {dollars(p2)}. "
        f"What is the difference between the two prices per item, in cents?"
    )
    working = [
        f"{dollars(p1)} ÷ {k1} = ${MONEY.text(u1)} per item; {dollars(p2)} ÷ {k2} = ${MONEY.text(u2)} per item.",
        f"Difference = {int(diff_cents)} cents.",
    ]
    return dict(text=text, working=working, distractors=d, **WHOLE.fields(diff_cents), answer_kind="number")


def _up_per100(rng):
    grams = rng.choice((250, 400, 500, 750, 800, 1200, 1500, 2000))
    per100 = Fraction(rng.randint(40, 400), 100)
    price = per100 * Fraction(grams, 100)
    check((price * 100).denominator == 1, "price not whole cents")
    d = MONEY.dist(
        per100,
        [
            (price / (Fraction(grams, 1000)), "w3up-per-kg-as-100g"),
            (price, "w3up-total-as-unit"),
            (price / (Fraction(grams, 100) + 1), "w3up-miscounts-hundreds"),
            (price / (Fraction(grams, 100) - 1), "w3up-miscounts-hundreds"),
        ],
    )
    enough(d, 3)
    shown = f"{grams} g" if grams < 1000 else f"{grams / 1000:g} kg"
    text = f"A {shown} packet of rice costs {dollars(price)}. What is its price per 100 g, in dollars?"
    working = [
        f"{grams} g is {Fraction(grams, 100)} lots of 100 g.",
        f"Price per 100 g = {dollars(price)} ÷ {Fraction(grams, 100)} = ${MONEY.text(per100)}.",
    ]
    return dict(text=text, working=working, distractors=d, **MONEY.fields(per100), answer_kind="number")


def _up_saving(rng):
    k1, k2 = rng.choice(((4, 12), (5, 10), (6, 24), (10, 30), (8, 24)))
    u_small = Fraction(rng.randint(50, 120), 100)
    u_big = u_small - Fraction(rng.randint(5, 20), 100)
    p_small, p_big = u_small * k1, u_big * k2
    n = k2 * rng.randint(1, 3)
    check(n % k1 == 0, "divisibility")
    saving = (u_small - u_big) * n
    item = rng.choice(_ITEMS)
    d = MONEY.dist(
        saving,
        [
            (u_small - u_big, "w3up-saving-per-item"),
            (abs(p_small - p_big), "w3up-difference-of-totals"),
            (u_small * n, "w3up-wrong-pack"),
            ((u_small - u_big) * (n + k2), "w3up-wrong-pack"),
        ],
    )
    enough(d, 3)
    text = (
        f"{item.capitalize()} come in a pack of {k1} for {dollars(p_small)} or a pack of {k2} for {dollars(p_big)}. "
        f"How much is saved by buying {n} {item} at the cheaper price per item instead of the dearer price per item?"
    )
    working = [
        f"Small pack: ${MONEY.text(u_small)} each; large pack: ${MONEY.text(u_big)} each.",
        f"Saving = {n} × (${MONEY.text(u_small)} − ${MONEY.text(u_big)}) = {dollars(saving)}.",
    ]
    return dict(text=text, working=working, distractors=d, **MONEY.fields(saving), answer_kind="number")


def _up_percent_cheaper(rng):
    u_dear = Fraction(rng.randint(20, 60), 10)
    pct = rng.choice((5, 10, 12, 15, 20, 25, 30, 40))
    u_cheap = u_dear * (1 - Fraction(pct, 100))
    check((u_cheap * 100).denominator == 1, "cheap price not whole cents")
    k = 10
    d = WHOLE.dist(
        pct,
        [
            ((u_dear - u_cheap) * 100, "w3up-percent-points"),
            ((u_dear - u_cheap) / u_cheap * 100, "w3up-percent-of-cheaper"),
            (u_cheap / u_dear * 100, "w3up-percent-points"),
            (pct + 5, "w3up-wrong-pack"),
        ],
    )
    enough(d, 3)
    text = (
        f"Brand A costs {dollars(u_dear * k)} for {k} items. Brand B costs {dollars(u_cheap * k)} for {k} items. "
        f"Brand B is what percentage cheaper than Brand A, per item?"
    )
    working = [
        f"Brand A: ${MONEY.text(u_dear)} each; brand B: ${MONEY.text(u_cheap)} each.",
        f"({MONEY.text(u_dear)} − {MONEY.text(u_cheap)}) ÷ {MONEY.text(u_dear)} × 100 = {pct}%.",
    ]
    return dict(text=text, working=working, distractors=d, **WHOLE.fields(pct), answer_kind="number")


@generator("w3_unit_tiers", level=8, presets={"medium": dict(tier="medium"), "hard": dict(tier="hard")})
def w3_unit_tiers(rng, tier="medium"):
    """Unit pricing, medium and hard."""
    pool = {
        "medium": (_up_cost_for_n, _up_difference, _up_per100),
        "hard": (_up_saving, _up_percent_cheaper, _up_difference),
    }.get(tier)
    if pool is None:
        raise ValueError(f"unknown tier {tier!r}")
    return _draw(rng, pool)


# -- flat-rate loans --------------------------------------------------------------------------------------------------


def _fl_interest(rng):
    p = rng.randint(5, 50) * 1000
    r = rng.choice((3, 4, 5, 6, 7, 8, Fraction(9, 2)))
    n = rng.randint(2, 7)
    i = p * frac(r) / 100 * n
    d = MONEY.dist(
        i,
        [
            (p + i, "w3fl-total-as-interest"),
            (p * frac(r) / 100, "w3fl-one-year-interest"),
            (p * frac(r) / 100 * (n + 1), "w3fl-off-by-one"),
            (p * frac(r) / 100 * (n - 1), "w3fl-off-by-one"),
        ],
    )
    enough(d, 3)
    text = f"A {dollars(p)} loan charges flat interest of {pct_text(r)}% of the amount borrowed per year for {n} years. How much interest is charged in total?"
    working = [f"Interest = {dollars(p)} × {pct_text(r)}/100 × {n} = {dollars(i)}."]
    return dict(text=text, working=working, distractors=d, **MONEY.fields(i), answer_kind="number")


def _fl_total(rng):
    p = rng.randint(5, 50) * 1000
    r = rng.choice((3, 4, 5, 6, 7, 8))
    n = rng.randint(2, 7)
    i = p * frac(r) / 100 * n
    total = p + i
    d = MONEY.dist(
        total,
        [
            (i, "w3fl-interest-as-total"),
            (p + p * frac(r) / 100, "w3fl-one-year-interest"),
            (p * (1 + frac(r) / 100) ** n, "w3fl-compounded"),
            (p + i / n * (n + 1), "w3fl-off-by-one"),
        ],
    )
    enough(d, 3)
    text = f"A {dollars(p)} loan charges flat interest of {r}% of the amount borrowed per year for {n} years. What is the total amount to be repaid?"
    working = [f"Interest = {dollars(i)}.", f"Total = {dollars(p)} + {dollars(i)} = {dollars(total)}."]
    return dict(text=text, working=working, distractors=d, **MONEY.fields(total), answer_kind="number")


def _fl_deposit(rng):
    price = rng.randint(10, 60) * 1000
    dep_pct = rng.choice((10, 20, 25))
    deposit = price * Fraction(dep_pct, 100)
    r = rng.choice((4, 5, 6, 7, 8))
    n = rng.randint(2, 5)
    loan = price - deposit
    monthly = loan * (1 + frac(r) / 100 * n) / (12 * n)
    d = MONEY.dist(
        monthly,
        [
            (price * (1 + frac(r) / 100 * n) / (12 * n), "w3fl-deposit-ignored"),
            (loan * (1 + frac(r) / 100 * n) / n, "w3fl-months-as-years"),
            (loan / (12 * n), "w3fl-ignores-interest"),
            (price * (1 + frac(r) / 100 * n) / (12 * n) - deposit, "w3fl-deposit-not-subtracted"),
        ],
    )
    enough(d, 3)
    text = (
        f"A car costs {dollars(price)}. The buyer pays a {dep_pct}% deposit and borrows the rest on a flat-interest loan of {r}% per year "
        f"of the amount borrowed, over {n} years, repaid in equal monthly instalments. What is each monthly repayment? Give your answer to the nearest cent."
    )
    working = [
        f"Deposit = {dollars(deposit)}; amount borrowed = {dollars(loan)}.",
        f"Interest = {dollars(loan)} × {r}/100 × {n} = {dollars(loan * frac(r) / 100 * n)}.",
        f"Monthly = {dollars(loan * (1 + frac(r) / 100 * n))} ÷ {12 * n} = ${MONEY.text(monthly)}.",
    ]
    return dict(text=text, working=working, distractors=d, **MONEY.fields(monthly), answer_kind="number")


def _fl_rate(rng):
    p = rng.randint(5, 50) * 1000
    r = rng.choice([Fraction(k, 2) for k in range(6, 19)])
    n = rng.randint(2, 6)
    i = p * r / 100 * n
    total = p + i
    monthly = total / (12 * n)
    check((monthly * 100).denominator == 1, "repayment not whole cents")
    d = DP1.dist(
        r,
        [
            (i / p * 100, "w3fl-rate-without-time"),
            (total / p * 100, "w3fl-rate-from-total"),
            ((total / p - 1) * 100 / (n + 1), "w3fl-off-by-one"),
        ],
    )
    enough(d, 3)
    text = (
        f"A {dollars(p)} loan is repaid with {12 * n} equal monthly payments of {dollars(monthly)}. "
        f"What annual flat interest rate was charged, as a percentage of the amount borrowed? Give your answer to 1 decimal place."
    )
    working = [
        f"Total repaid = {12 * n} × {dollars(monthly)} = {dollars(total)}.",
        f"Interest = {dollars(total)} − {dollars(p)} = {dollars(i)}.",
        f"Rate = {dollars(i)} ÷ ({dollars(p)} × {n}) × 100 = {pct_text(r)}%.",
    ]
    return dict(text=text, working=working, distractors=d, **DP1.fields(r), answer_kind="number")


def _fl_term(rng):
    p = rng.randint(5, 50) * 1000
    r = rng.choice((4, 5, 6, 8))
    n = rng.randint(2, 7)
    i = p * frac(r) / 100 * n
    d = WHOLE.dist(
        n,
        [
            (12 * n, "w3fl-term-in-months"),
            ((i + p) / (p * frac(r) / 100), "w3fl-term-from-total"),
            (n + 1, "w3fl-off-by-one"),
            (n - 1, "w3fl-off-by-one"),
        ],
    )
    enough(d, 3)
    text = f"A {dollars(p)} loan charges flat interest of {r}% of the amount borrowed per year, and {dollars(i)} interest is charged in total. For how many years is the loan?"
    working = [
        f"Interest per year = {dollars(p * frac(r) / 100)}.",
        f"Years = {dollars(i)} ÷ {dollars(p * frac(r) / 100)} = {n}.",
    ]
    return dict(text=text, working=working, distractors=d, **WHOLE.fields(n), answer_kind="number")


def _fl_from_monthly(rng):
    p = rng.randint(5, 50) * 1000
    r = rng.choice((4, 5, 6, 8))
    n = rng.randint(2, 6)
    total = p * (1 + frac(r) / 100 * n)
    monthly = total / (12 * n)
    check((monthly * 100).denominator == 1, "repayment not whole cents")
    interest = total - p
    d = MONEY.dist(
        interest,
        [
            (total, "w3fl-interest-as-total"),
            (monthly * 12 - p, "w3fl-one-year-of-payments"),
            (interest / n, "w3fl-one-year-interest"),
            (monthly * 12 * (n + 1) - p, "w3fl-off-by-one"),
        ],
    )
    enough(d, 3)
    text = (
        f"A {dollars(p)} loan is repaid with {12 * n} equal monthly payments of {dollars(monthly)}. "
        f"How much interest is charged in total?"
    )
    working = [
        f"Total repaid = {12 * n} × {dollars(monthly)} = {dollars(total)}.",
        f"Interest = {dollars(total)} − {dollars(p)} = {dollars(interest)}.",
    ]
    return dict(text=text, working=working, distractors=d, **MONEY.fields(interest), answer_kind="number")


@generator("w3_flatloan_tiers", level=11, presets={"medium": dict(tier="medium"), "hard": dict(tier="hard")})
def w3_flatloan_tiers(rng, tier="medium"):
    """Flat-rate loans, medium and hard."""
    pool = {"medium": (_fl_interest, _fl_total, _fl_from_monthly), "hard": (_fl_deposit, _fl_rate, _fl_term)}.get(tier)
    if pool is None:
        raise ValueError(f"unknown tier {tier!r}")
    return _draw(rng, pool)


# -- moving average ---------------------------------------------------------------------------------------------------

_MA_LABELS = (
    "weekly sales, in dollars",
    "daily visitors",
    "monthly rainfall, in millimetres",
    "cans collected each week",
)


def _ma_kth(rng):
    w = rng.choice((3, 5))
    length = rng.randint(w + 3, 9)
    series = [rng.randint(8, 60) for _ in range(length)]
    half = w // 2
    k = rng.randint(half, length - half - 1)  # zero-based centre
    window = series[k - half : k + half + 1]
    # make the window total divisible by w by changing the last value of the window
    window[-1] += (-sum(window)) % w
    series[k - half : k + half + 1] = window
    mean = Fraction(sum(window), w)
    shifted = Fraction(sum(series[k : k + w]), w) if k + w <= length else Fraction(sum(series[k - w + 1 : k + 1]), w)
    wrong_w = w + 2 if k - half - 1 >= 0 and k + half + 1 < length else w - 2
    wrong_win = series[max(0, k - wrong_w // 2) : k + wrong_w // 2 + 1]
    d = WHOLE.dist(
        mean,
        [
            (shifted, "w3ma-shifted-window"),
            (sum(window), "w3ma-total-as-mean"),
            (Fraction(sum(series), length), "w3ma-whole-series"),
            (Fraction(sum(wrong_win), len(wrong_win)), "w3ma-wrong-window"),
        ],
    )
    enough(d, 3)
    label = rng.choice(_MA_LABELS)
    text = (
        f"The {label} for {length} consecutive periods were: {', '.join(map(str, series))}. "
        f"What is the {w}-point moving average centred on period {k + 1}?"
    )
    working = [
        f"The {w} values centred on period {k + 1} are {', '.join(map(str, window))}.",
        f"Mean = {sum(window)} ÷ {w} = {mean}.",
    ]
    return dict(text=text, working=working, distractors=d, **WHOLE.fields(mean), answer_kind="number")


def _ma_missing(rng):
    w = rng.choice((3, 4, 5))
    vals = [rng.randint(10, 60) for _ in range(w)]
    mean = Fraction(sum(vals), w)
    check(mean.denominator == 1, "mean not whole")
    missing_idx = rng.randrange(w)
    missing = vals[missing_idx]
    known = [v for i, v in enumerate(vals) if i != missing_idx]
    d = WHOLE.dist(
        missing,
        [
            (mean, "w3ma-missing-not-from-total"),
            (mean - sum(known), "w3ma-missing-no-total"),
            (mean * (w - 1) - sum(known), "w3ma-missing-no-total"),
            (mean * (w + 1) - sum(known), "w3ma-missing-wrong-window"),
        ],
        allow_negative=True,
    )
    enough(d, 3)
    shown = ["?" if i == missing_idx else str(v) for i, v in enumerate(vals)]
    text = f"The {w}-point moving average of the values {', '.join(shown)} is {mean}. What is the missing value?"
    working = [
        f"Total of the {w} values = {mean} × {w} = {mean * w}.",
        f"Missing = {mean * w} − {sum(known)} = {missing}.",
    ]
    return dict(text=text, working=working, distractors=d, **WHOLE.fields(missing), answer_kind="number")


def _ma_change(rng):
    w = rng.choice((3, 4, 5))
    out = rng.randint(10, 40)
    delta = w * rng.randint(1, 6) * rng.choice((-1, 1))
    new = out + delta
    check(new > 0, "new value positive")
    mean_before = rng.randint(20, 50)
    change = Fraction(delta, w)
    d = WHOLE.dist(
        change,
        [
            (delta, "w3ma-change-not-divided"),
            (-change, "w3ma-change-as-new-value"),
            (Fraction(delta, w + 1), "w3ma-wrong-window"),
            (Fraction(delta, w - 1), "w3ma-wrong-window"),
        ],
        allow_negative=True,
    )
    enough(d, 3)
    text = (
        f"A {w}-point moving average has the value {mean_before}. The next period, the value {out} drops out of the window and the new value "
        f"{new} comes in. By how much does the moving average change? (Give a negative answer if it falls.)"
    )
    working = [
        f"Change in the window total = {new} − {out} = {delta}.",
        f"Change in the mean = {delta} ÷ {w} = {change}.",
    ]
    return dict(text=text, working=working, distractors=d, **WHOLE.fields(change), answer_kind="number")


def _ma_centred4(rng):
    v = [rng.randint(10, 50) for _ in range(5)]
    # centred 4-point: (v0/2 + v1 + v2 + v3 + v4/2) / 4 = (v0 + 2(v1+v2+v3) + v4) / 8 ; force numerator divisible by 4
    v[4] += (-(v[0] + 2 * (v[1] + v[2] + v[3]) + v[4])) % 4
    value = Fraction(v[0] + 2 * (v[1] + v[2] + v[3]) + v[4], 8)
    d = DP1.dist(
        value,
        [
            (Fraction(sum(v), 5), "w3ma-whole-series"),
            (Fraction(sum(v[:4]), 4), "w3ma-shifted-window"),
            (Fraction(sum(v[1:4]), 3), "w3ma-centred-simple"),
            (Fraction(sum(v[1:5]), 4), "w3ma-shifted-window"),
            (Fraction(v[0] + v[1] + v[2] + v[3] + v[4], 4), "w3ma-centred-unweighted"),
        ],
    )
    enough(d, 3)
    text = (
        f"Five consecutive quarterly values are {', '.join(map(str, v))}. A 4-point moving average centred on the third value gives the "
        f"end values half weight: (½ × first + second + third + fourth + ½ × fifth) ÷ 4. What is it? Give your answer to 1 decimal place."
    )
    working = [f"(½ × {v[0]} + {v[1]} + {v[2]} + {v[3]} + ½ × {v[4]}) ÷ 4 = {float(value):g}."]
    return dict(text=text, working=working, distractors=d, **DP1.fields(value), answer_kind="number")


@generator("w3_movavg_tiers", level=11, presets={"medium": dict(tier="medium"), "hard": dict(tier="hard")})
def w3_movavg_tiers(rng, tier="medium"):
    """Moving average, medium and hard."""
    pool = {"medium": (_ma_kth,), "hard": (_ma_missing, _ma_change, _ma_centred4)}.get(tier)
    if pool is None:
        raise ValueError(f"unknown tier {tier!r}")
    return _draw(rng, pool)


# -- mean -------------------------------------------------------------------------------------------------------------


def _mn_decimal(rng):
    n = rng.choice((5, 10))
    vals = [Fraction(rng.randint(40, 400), 10) for _ in range(n)]
    mean = sum(vals) / n
    d = DP2.dist(
        mean,
        [
            (sum(vals), "w3mn-divides-by-wrong-count"),
            (sum(vals) / (n - 1), "w3mn-divides-by-wrong-count"),
            (sum(vals) / (n + 1), "w3mn-divides-by-wrong-count"),
        ],
    )
    enough(d, 3)
    text = f"A class timed {n} runners over 100 m, in seconds: {', '.join(f'{float(v):.1f}' for v in vals)}. What is the mean time, in seconds, to 2 decimal places?"
    text = text.replace("A class timed", "A coach timed")
    working = [f"Total = {float(sum(vals)):.1f}.", f"Mean = {float(sum(vals)):.1f} ÷ {n} = {DP2.text(mean)}."]
    return dict(text=text, working=working, distractors=d, **DP2.fields(mean), answer_kind="number")


def _mn_freq(rng):
    start = rng.randint(0, 3)
    scores = list(range(start, start + 5))
    rows = len(scores)
    for _ in range(40):
        freqs = [rng.randint(1, 9) for _ in range(rows)]
        if sum(freqs) in (10, 20, 25, 40, 50):
            break
    else:
        check(False, "no frequency set")
    total_f = sum(freqs)
    total = sum(s * f for s, f in zip(scores, freqs))
    mean = Fraction(total, total_f)
    d = DP2.dist(
        mean,
        [
            (Fraction(sum(scores), rows), "w3mn-freq-unweighted"),
            (Fraction(total, rows), "w3mn-freq-divides-by-rows"),
            (Fraction(total_f, rows), "w3mn-freq-sums-frequencies"),
            (Fraction(total, total_f + 1), "w3mn-divides-by-wrong-count"),
        ],
    )
    enough(d, 3)
    table = "; ".join(f"score {s}: {f}" for s, f in zip(scores, freqs))
    text = f"The scores in a quiz and how many students got each score were: {table}. What is the mean score, to 2 decimal places?"
    working = [
        f"Total of scores = {' + '.join(f'{s}×{f}' for s, f in zip(scores, freqs))} = {total}.",
        f"Students = {total_f}.",
        f"Mean = {total} ÷ {total_f} = {DP2.text(mean)}.",
    ]
    return dict(text=text, working=working, distractors=d, **DP2.fields(mean), answer_kind="number")


def _mn_missing(rng):
    n = rng.randint(4, 6)
    target = rng.randint(55, 85)
    known = [rng.randint(40, 100) for _ in range(n - 1)]
    need = target * n - sum(known)
    check(0 < need <= 100, "need out of range")
    d = WHOLE.dist(
        need,
        [
            (target, "w3mn-missing-is-mean"),
            (Fraction(sum(known), n - 1), "w3mn-known-mean"),
            (target * n, "w3mn-total-needed"),
            (target * (n - 1) - sum(known), "w3mn-missing-wrong-count"),
            (target * (n + 1) - sum(known), "w3mn-divides-by-wrong-count"),
        ],
    )
    enough(d, 3)
    text = (
        f"A student scored {', '.join(map(str, known))} in the first {n - 1} tests. What must the student score in test {n} "
        f"to have a mean of {target} over all {n} tests?"
    )
    working = [
        f"Total needed = {target} × {n} = {target * n}.",
        f"Score needed = {target * n} − {sum(known)} = {need}.",
    ]
    return dict(text=text, working=working, distractors=d, **WHOLE.fields(need), answer_kind="number")


def _mn_combined(rng):
    n1, n2 = rng.sample((10, 15, 20, 25, 30), 2)
    m1, m2 = rng.randint(50, 85), rng.randint(50, 85)
    check(m1 != m2, "means equal")
    total = n1 * m1 + n2 * m2
    mean = Fraction(total, n1 + n2)
    d = DP1.dist(
        mean,
        [
            (Fraction(m1 + m2, 2), "w3mn-mean-of-means"),
            (Fraction(total, 2 * max(n1, n2)), "w3mn-divides-by-wrong-count"),
            (Fraction(total, n1 + n2 + 1), "w3mn-divides-by-wrong-count"),
            (Fraction(m1 * n2 + m2 * n1, n1 + n2), "w3mn-weights-swapped"),
        ],
    )
    enough(d, 3)
    text = (
        f"Class A has {n1} students with a mean mark of {m1}. Class B has {n2} students with a mean mark of {m2}. "
        f"What is the mean mark of all {n1 + n2} students? Give your answer to 1 decimal place."
    )
    working = [f"Total marks = {n1} × {m1} + {n2} × {m2} = {total}.", f"Mean = {total} ÷ {n1 + n2} = {DP1.text(mean)}."]
    return dict(text=text, working=working, distractors=d, **DP1.fields(mean), answer_kind="number")


def _mn_new_value(rng):
    n = rng.randint(4, 9)
    m = rng.randint(20, 60)
    x = rng.randint(10, 90)
    new = Fraction(n * m + x, n + 1)
    d = DP1.dist(
        new,
        [
            (Fraction(m + x, 2), "w3mn-new-value-averaged"),
            (Fraction(n * m + x, n), "w3mn-new-count-forgotten"),
            (Fraction(m * n, n + 1), "w3mn-new-value-left-out"),
            (Fraction(n * m + x, n + 2), "w3mn-divides-by-wrong-count"),
        ],
    )
    enough(d, 3)
    text = (
        f"The mean of {n} numbers is {m}. A new number, {x}, is added to the list. What is the mean of the {n + 1} numbers? "
        f"Give your answer to 1 decimal place."
    )
    working = [
        f"Original total = {n} × {m} = {n * m}.",
        f"New total = {n * m} + {x} = {n * m + x}.",
        f"New mean = {n * m + x} ÷ {n + 1} = {DP1.text(new)}.",
    ]
    return dict(text=text, working=working, distractors=d, **DP1.fields(new), answer_kind="number")


@generator("w3_mean_tiers", level=7, presets={"medium": dict(tier="medium"), "hard": dict(tier="hard")})
def w3_mean_tiers(rng, tier="medium"):
    """Mean, medium and hard."""
    pool = {"medium": (_mn_decimal, _mn_freq), "hard": (_mn_missing, _mn_combined, _mn_new_value)}.get(tier)
    if pool is None:
        raise ValueError(f"unknown tier {tier!r}")
    return _draw(rng, pool)


# -- shortest network path (text only) ---------------------------------------------------------------------------------


def _random_graph(rng, names, extra):
    """A connected weighted graph: a random spanning tree plus `extra` more edges. {frozenset({a, b}): weight}."""
    order = list(names)
    rng.shuffle(order)
    edges = {}
    for i in range(1, len(order)):
        j = rng.randrange(i)
        edges[frozenset((order[i], order[j]))] = rng.randint(2, 15)
    attempts_left = 200
    while extra and attempts_left:
        attempts_left -= 1
        a, b = rng.sample(names, 2)
        key = frozenset((a, b))
        if key not in edges:
            edges[key] = rng.randint(2, 15)
            extra -= 1
    return edges


def _simple_paths(edges, start, end):
    adj = {}
    for e, w in edges.items():
        a, b = tuple(e)
        adj.setdefault(a, []).append((b, w))
        adj.setdefault(b, []).append((a, w))
    out = []

    def go(node, seen, length, route):
        if node == end:
            out.append((length, tuple(route)))
            return
        for nxt, w in adj.get(node, ()):
            if nxt not in seen:
                seen.add(nxt)
                route.append(nxt)
                go(nxt, seen, length + w, route)
                route.pop()
                seen.remove(nxt)

    go(start, {start}, 0, [start])
    return sorted(out)


def _edge_list(edges):
    return ", ".join(f"{''.join(sorted(e))} = {w}" for e, w in sorted(edges.items(), key=lambda kv: sorted(kv[0])))


def _greedy_length(edges, start, end):
    adj = {}
    for e, w in edges.items():
        a, b = tuple(e)
        adj.setdefault(a, []).append((w, b))
        adj.setdefault(b, []).append((w, a))
    node, seen, total = start, {start}, 0
    while node != end:
        options = sorted((w, n) for w, n in adj[node] if n not in seen)
        if not options:
            return None
        w, node = options[0]
        seen.add(node)
        total += w
    return total


def _np_plain(rng):
    names = "ABCDEF"
    edges = _random_graph(rng, names, 4)
    check(frozenset("AF") not in edges, "direct road")
    paths = _simple_paths(edges, "A", "F")
    best = paths[0][0]
    check(len(paths) >= 3 and paths[1][0] > best and len(paths[0][1]) >= 4, "unique shortest")
    fewest = min(paths, key=lambda p: (len(p[1]), p[0]))
    d = WHOLE.dist(
        best,
        [
            (fewest[0], "w3np-fewest-roads"),
            (_greedy_length(edges, "A", "F"), "w3np-greedy"),
            (paths[1][0], "w3np-second-shortest"),
            (paths[-1][0], "w3np-longest"),
            (best + 3, "w3np-greedy"),
        ],
    )
    enough(d, 3)
    text = f"Six towns A to F are joined by roads of these lengths in km: {_edge_list(edges)}. What is the length of the shortest route from A to F?"
    route = "-".join(paths[0][1])
    working = [f"Compare every route from A to F; the shortest is {route}, {best} km."]
    return dict(text=text, working=working, distractors=d, **WHOLE.fields(best), answer_kind="number")


def _np_via(rng):
    names = "ABCDEFG"
    edges = _random_graph(rng, names, 5)
    check(frozenset("AG") not in edges, "direct road")
    paths = _simple_paths(edges, "A", "G")
    via = rng.choice("CDE")
    through = [p for p in paths if via in p[1]]
    check(through and paths[0][0] < through[0][0], "route through the town is not the overall shortest")
    best = through[0][0]
    check(len(through) < 2 or through[1][0] > best, "unique via route")
    fewest = min(paths, key=lambda p: (len(p[1]), p[0]))
    d = WHOLE.dist(
        best,
        [
            (paths[0][0], "w3np-skips-stop"),
            (fewest[0], "w3np-fewest-roads"),
            (through[1][0] if len(through) > 1 else None, "w3np-second-shortest"),
            (
                min(l for l, _ in _simple_paths(edges, "A", via)) + min(l for l, _ in _simple_paths(edges, via, "G")),
                "w3np-two-legs-shortest-sum-error",
            ),
            (through[-1][0], "w3np-longest"),
        ],
    )
    enough(d, 3)
    text = (
        f"Seven towns A to G are joined by roads of these lengths in km: {_edge_list(edges)}. A bus must go from A to G and stop at "
        f"town {via} on the way. What is the length of the shortest such route? (A route cannot visit a town twice.)"
    )
    working = [
        f"List the routes from A to G that pass through {via}; the shortest is {'-'.join(through[0][1])}, {best} km."
    ]
    return dict(text=text, working=working, distractors=d, **WHOLE.fields(best), answer_kind="number")


def _np_closed(rng):
    names = "ABCDEFG"
    edges = _random_graph(rng, names, 5)
    check(frozenset("AG") not in edges, "direct road")
    paths = _simple_paths(edges, "A", "G")
    best_len, best_route = paths[0]
    check(len(paths) >= 2 and paths[1][0] > best_len and len(best_route) >= 4, "unique shortest")
    k = rng.randrange(len(best_route) - 1)
    closed = frozenset((best_route[k], best_route[k + 1]))
    remaining = {e: w for e, w in edges.items() if e != closed}
    new_paths = _simple_paths(remaining, "A", "G")
    check(new_paths, "graph disconnected")
    new_best = new_paths[0][0]
    check(new_best > best_len and (len(new_paths) < 2 or new_paths[1][0] > new_best), "closure changes nothing or ties")
    fewest = min(new_paths, key=lambda p: (len(p[1]), p[0]))
    d = WHOLE.dist(
        new_best,
        [
            (best_len, "w3np-closed-ignored"),
            (best_len - edges[closed] if best_len > edges[closed] else None, "w3np-drops-the-detour"),
            (fewest[0], "w3np-fewest-roads"),
            (new_paths[1][0] if len(new_paths) > 1 else None, "w3np-second-shortest"),
            (new_paths[-1][0], "w3np-longest"),
        ],
    )
    enough(d, 3)
    a, b = sorted(closed)
    text = (
        f"Seven towns A to G are joined by roads of these lengths in km: {_edge_list(edges)}. The road {a}{b} is closed for repairs. "
        f"What is the length of the shortest route from A to G now?"
    )
    working = [
        f"Remove {a}{b} ({edges[closed]} km) and compare the remaining routes; the shortest is {'-'.join(new_paths[0][1])}, {new_best} km."
    ]
    return dict(text=text, working=working, distractors=d, **WHOLE.fields(new_best), answer_kind="number")


@generator("w3_netpath_tiers", level=11, presets={"medium": dict(tier="medium"), "hard": dict(tier="hard")})
def w3_netpath_tiers(rng, tier="medium"):
    """Shortest route over a road list (no figure), medium and hard."""
    pool = {"medium": (_np_plain,), "hard": (_np_via, _np_closed)}.get(tier)
    if pool is None:
        raise ValueError(f"unknown tier {tier!r}")
    return _draw(rng, pool)
