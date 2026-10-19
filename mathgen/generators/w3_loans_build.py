"""Generates the `w3_*` loan and annuity topics: interest-only loans and future-value annuity table lookups.

* ``interest-only-loan`` (``w3_interest_only_loan``, exam ``exd_x_w3_interest_only_loan``): the borrower pays only the
  interest, so the amount owing stays the same until it is repaid. Easy: the yearly interest, the monthly payment, the
  interest over several years. Medium: the total repaid when the loan is cleared at the end, the loan from a monthly
  payment, the rate from a monthly payment. Hard: a rate that changes part-way, a rate rise that changes the payment by
  a known amount, a lump sum that reduces what is owing part-way through.
* ``future-value-annuity-table`` (``w3_annuity_table``, exam ``exd_x_w3_annuity_table``, NSW Standard 2 style): the
  future value of equal contributions read from a table of factors for $1 per period. Easy: one lookup, the rate given
  per period. Medium: the payment that reaches a target, the interest earned. Hard: an annual rate paid in monthly or
  quarterly contributions, so the rate per period and the number of periods have to be found before the lookup, either to
  find the future value or the payment that reaches a target.

Loans are built from whole multiples of $12,000 (so the monthly interest is a whole number of cents). Table factors are
the exact future-value factors rounded to 4 places; contributions are multiples of $100, so every product is an exact
number of cents and no learner rounding is involved.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import attempts, check, generator, round_half_up
from ._w3_kit import DP1, MONEY, WHOLE, art, dollars, pct_text
from ._w6a_kit import TooFewDistractors, enough, part, retry_draw

IOL_TOPIC = "w3_interest_only_loan"
IOL_EXAM = "exd_x_w3_interest_only_loan"
ANN_TOPIC = "w3_annuity_table"
ANN_EXAM = "exd_x_w3_annuity_table"

_IOL = "interest-only-loan"
_ANN = "future-value-annuity-table"

MISCONCEPTIONS = [
    ("iol-rate-not-converted", "Multiplies the loan by the percentage without dividing by 100.", _IOL),
    ("iol-divides-by-rate", "Divides the loan by the rate instead of finding that percentage of it.", _IOL),
    ("iol-decimal-slip", "Misplaces the decimal point when finding the interest.", _IOL),
    ("iol-annual-for-monthly", "Gives the interest for a whole year where the monthly payment was asked.", _IOL),
    ("iol-multiplies-by-twelve", "Multiplies the yearly interest by 12 instead of dividing it by 12.", _IOL),
    ("iol-principal-share", "Divides the loan by 12 and calls that the monthly interest.", _IOL),
    ("iol-one-year-only", "Gives one year's interest where the interest over several years was asked.", _IOL),
    ("iol-extra-year", "Counts one year too many when adding up the interest.", _IOL),
    (
        "iol-months-as-years",
        "Multiplies the monthly payment by the number of years instead of the number of months.",
        _IOL,
    ),
    (
        "iol-total-repaid-for-interest",
        "Gives the total repaid (loan plus interest) where the interest was asked.",
        _IOL,
    ),
    (
        "iol-forgets-principal",
        "Gives only the interest paid where the total repaid, including the loan, was asked.",
        _IOL,
    ),
    (
        "iol-compound",
        "Compounds the interest each year, although the amount owing never changes on an interest-only loan.",
        _IOL,
    ),
    (
        "iol-forgets-twelve",
        "Divides the payment by the rate without first converting the monthly payment to a yearly amount.",
        _IOL,
    ),
    ("iol-multiplies-by-rate", "Multiplies the yearly interest by the rate instead of dividing by it.", _IOL),
    ("iol-annual-as-loan", "Takes the yearly interest as the amount borrowed.", _IOL),
    ("iol-rate-from-annual-payment", "Treats the monthly payment as the yearly interest when finding the rate.", _IOL),
    ("iol-rate-inverted", "Divides the loan by the yearly interest instead of the other way round.", _IOL),
    ("iol-first-rate-throughout", "Uses the first interest rate for the whole term of the loan.", _IOL),
    ("iol-second-rate-throughout", "Uses the later interest rate for the whole term of the loan.", _IOL),
    ("iol-average-rate", "Averages the two rates, although they apply for different lengths of time.", _IOL),
    ("iol-swapped-years", "Swaps the number of years at each rate.", _IOL),
    ("iol-rise-as-payment", "Gives the rise in the payment where the original payment was asked.", _IOL),
    ("iol-loan-for-payment", "Gives the amount borrowed where the original monthly payment was asked.", _IOL),
    (
        "iol-rise-times-rate",
        "Multiplies the rise in the payment by the rate instead of dividing by the change in rate.",
        _IOL,
    ),
    ("iol-lump-ignored", "Ignores the lump sum paid off the loan.", _IOL),
    ("iol-lump-off-interest", "Takes the lump sum off the interest instead of off the amount owing.", _IOL),
    ("iol-lump-whole-term", "Applies the lump sum to the whole term, as if it had been paid on the first day.", _IOL),
    ("iol-lump-after-only", "Counts only the interest after the lump sum was paid.", _IOL),
    ("ann-wrong-period-row", "Reads the factor for one period more or fewer than the contributions made.", _ANN),
    ("ann-wrong-rate-row", "Reads the factor for the next rate up or down in the table.", _ANN),
    ("ann-no-interest", "Multiplies the contribution by the number of periods, ignoring the interest.", _ANN),
    ("ann-factor-only", "Gives the table factor itself, forgetting to multiply by the contribution.", _ANN),
    ("ann-lump-sum-growth", "Grows a single deposit instead of using the annuity factor.", _ANN),
    ("ann-divides-by-factor", "Divides the contribution by the factor instead of multiplying.", _ANN),
    ("ann-multiplies-for-payment", "Multiplies the target by the factor instead of dividing by it.", _ANN),
    ("ann-divides-by-periods", "Divides the target by the number of periods, ignoring the interest.", _ANN),
    ("ann-future-value-for-interest", "Gives the future value where the interest earned was asked.", _ANN),
    ("ann-contributions-for-interest", "Gives the total contributed where the interest earned was asked.", _ANN),
    ("ann-years-as-periods", "Uses the number of years as the number of periods.", _ANN),
    (
        "ann-rate-not-divided",
        "Divides the annual rate by the wrong number of periods per year, so reads the wrong row of the table.",
        _ANN,
    ),
    ("ann-difference-sign", "Adds the two future values instead of finding their difference.", _ANN),
    ("ann-contributions-total-only", "Gives the total contributed for one option only.", _ANN),
]

_PRINCIPAL_UNIT = 12000
_IOL_RATES = (Fraction(7, 2), 4, Fraction(9, 2), 5, Fraction(11, 2), 6, Fraction(13, 2), 7, Fraction(15, 2), 8)
_LOAN_PURPOSES = (
    "an investment property",
    "a block of units",
    "a rural property",
    "a shop",
    "a farm",
    "a holiday home",
)


def _loan(rng, lo=6, hi=60):
    return Fraction(rng.randint(lo, hi) * _PRINCIPAL_UNIT)


def _rate(rng):
    return Fraction(rng.choice(_IOL_RATES))


def _intro(rng, loan, rate):
    purpose = rng.choice(_LOAN_PURPOSES)
    return (
        f"A bank lends {dollars(loan)} to buy {purpose} on an interest-only loan at {pct_text(rate)}% per annum. "
        f"Only the interest is paid, so the amount owing stays at {dollars(loan)}."
    )


# -- interest-only: easy ------------------------------------------------------------------------------------------------


def _iol_annual(rng):
    loan, r = _loan(rng), _rate(rng)
    annual = loan * r / 100
    d = MONEY.dist(
        annual,
        [
            (loan * r, "iol-rate-not-converted"),
            (annual / 10, "iol-decimal-slip"),
            (loan / r, "iol-divides-by-rate"),
            (loan + annual, "iol-total-repaid-for-interest"),
            (annual / 12, "iol-annual-for-monthly"),
        ],
    )
    enough(d, 3)
    text = _intro(rng, loan, r) + " How much interest is paid in one year, in dollars?"
    working = [f"Yearly interest = {pct_text(r)}% of {dollars(loan)} = {dollars(annual)}."]
    return dict(text=text, working=working, distractors=d, **MONEY.fields(annual), answer_kind="number")


def _iol_monthly(rng):
    loan, r = _loan(rng), _rate(rng)
    annual = loan * r / 100
    monthly = annual / 12
    check(monthly == loan * r / 1200 and (monthly * 100).denominator == 1, "monthly")
    d = MONEY.dist(
        monthly,
        [
            (annual, "iol-annual-for-monthly"),
            (annual * 12, "iol-multiplies-by-twelve"),
            (loan / 12, "iol-principal-share"),
            (monthly / 10, "iol-decimal-slip"),
            (monthly * 10, "iol-decimal-slip"),
        ],
    )
    enough(d, 3)
    text = (
        _intro(rng, loan, r)
        + " The interest is paid in equal monthly payments. What is each monthly payment, in dollars?"
    )
    working = [
        f"Yearly interest = {dollars(annual)}.",
        f"Monthly payment = {dollars(annual)} / 12 = {dollars(monthly)}.",
    ]
    return dict(text=text, working=working, distractors=d, **MONEY.fields(monthly), answer_kind="number")


def _iol_total_interest(rng):
    loan, r = _loan(rng), _rate(rng)
    n = rng.randint(3, 12)
    annual = loan * r / 100
    total = annual * n
    monthly = annual / 12
    d = MONEY.dist(
        total,
        [
            (annual, "iol-one-year-only"),
            (annual * (n + 1), "iol-extra-year"),
            (monthly * n, "iol-months-as-years"),
            (loan + total, "iol-total-repaid-for-interest"),
            (loan * ((1 + r / 100) ** n - 1), "iol-compound"),
        ],
    )
    enough(d, 3)
    text = _intro(rng, loan, r) + f" How much interest is paid altogether over {n} years, in dollars?"
    working = [f"Yearly interest = {dollars(annual)}.", f"Over {n} years: {n} × {dollars(annual)} = {dollars(total)}."]
    return dict(text=text, working=working, distractors=d, **MONEY.fields(total), answer_kind="number")


# -- interest-only: medium ----------------------------------------------------------------------------------------------


def _iol_total_repaid(rng):
    loan, r = _loan(rng), _rate(rng)
    n = rng.randint(3, 10)
    interest = loan * r / 100 * n
    total = loan + interest
    d = MONEY.dist(
        total,
        [
            (interest, "iol-forgets-principal"),
            (loan + loan * r / 100, "iol-one-year-only"),
            (loan * (1 + r / 100) ** n, "iol-compound"),
            (loan + loan * r / 100 * (n + 1), "iol-extra-year"),
            (loan * r / 1200 * n + loan, "iol-months-as-years"),
        ],
    )
    enough(d, 3)
    text = (
        _intro(rng, loan, r)
        + f" At the end of {n} years the borrower repays the {dollars(loan)}. How much has the borrower paid altogether, including the interest, in dollars?"
    )
    working = [
        f"Interest = {n} × {dollars(loan * r / 100)} = {dollars(interest)}.",
        f"Total paid = {dollars(interest)} + {dollars(loan)} = {dollars(total)}.",
    ]
    return dict(text=text, working=working, distractors=d, **MONEY.fields(total), answer_kind="number")


def _iol_loan_from_payment(rng):
    k = rng.randint(6, 60)
    loan = Fraction(k * _PRINCIPAL_UNIT)
    r = _rate(rng)
    monthly = loan * r / 1200
    d = MONEY.dist(
        loan,
        [
            (monthly * 12, "iol-annual-as-loan"),
            (monthly * 100 / r, "iol-forgets-twelve"),
            (monthly * 12 * r / 100, "iol-multiplies-by-rate"),
            (monthly * 1200 * r, "iol-multiplies-by-rate"),
            (monthly * 12 * 10 / r, "iol-decimal-slip"),
        ],
    )
    enough(d, 3)
    text = (
        f"A borrower has an interest-only loan at {pct_text(r)}% per annum and pays {dollars(monthly)} in interest each month. "
        f"How much was borrowed, in dollars?"
    )
    working = [
        f"Yearly interest = 12 × {dollars(monthly)} = {dollars(monthly * 12)}.",
        f"Loan = {dollars(monthly * 12)} / {pct_text(r)}% = {dollars(loan)}.",
    ]
    return dict(text=text, working=working, distractors=d, **MONEY.fields(loan), answer_kind="number")


def _iol_rate_from_payment(rng):
    loan = _loan(rng)
    r = _rate(rng)
    monthly = loan * r / 1200
    d = DP1.dist(
        r,
        [
            (monthly * 100 / loan, "iol-rate-from-annual-payment"),
            (loan / monthly / 12, "iol-rate-inverted"),
            (r / 12, "iol-annual-for-monthly"),
            (r * 12, "iol-multiplies-by-twelve"),
            (r + Fraction(1, 2), "iol-decimal-slip"),
        ],
    )
    enough(d, 3)
    text = (
        f"A borrower has an interest-only loan of {dollars(loan)} and pays {dollars(monthly)} in interest each month. "
        f"What is the interest rate per annum, as a percentage? Give the answer to 1 decimal place."
    )
    working = [
        f"Yearly interest = 12 × {dollars(monthly)} = {dollars(monthly * 12)}.",
        f"Rate = {dollars(monthly * 12)} / {dollars(loan)} × 100 = {pct_text(r)}%.",
    ]
    return dict(text=text, working=working, distractors=d, **DP1.fields(r), answer_kind="number")


# -- interest-only: hard ------------------------------------------------------------------------------------------------


def _iol_rate_change(rng):
    loan = _loan(rng)
    r1 = _rate(rng)
    for _ in attempts(100, "two different rates"):
        r2 = _rate(rng)
        if abs(r2 - r1) >= 1:
            break
    n = rng.randint(6, 12)
    k = rng.randint(2, n - 2)
    if 2 * k == n:
        raise TooFewDistractors("equal split")
    total = loan * (r1 * k + r2 * (n - k)) / 100
    d = MONEY.dist(
        total,
        [
            (loan * r1 * n / 100, "iol-first-rate-throughout"),
            (loan * r2 * n / 100, "iol-second-rate-throughout"),
            (loan * (r1 + r2) / 2 * n / 100, "iol-average-rate"),
            (loan * (r1 * (n - k) + r2 * k) / 100, "iol-swapped-years"),
        ],
    )
    enough(d, 3)
    text = (
        f"A bank lends {dollars(loan)} on an interest-only loan for {n} years. The interest rate is {pct_text(r1)}% per annum for the "
        f"first {k} years and {pct_text(r2)}% per annum for the remaining {n - k} years. How much interest is paid over the {n} years, in dollars?"
    )
    working = [
        f"First {k} years: {k} × {pct_text(r1)}% of {dollars(loan)} = {dollars(loan * r1 * k / 100)}.",
        f"Last {n - k} years: {n - k} × {pct_text(r2)}% of {dollars(loan)} = {dollars(loan * r2 * (n - k) / 100)}.",
        f"Total = {dollars(total)}.",
    ]
    return dict(text=text, working=working, distractors=d, **MONEY.fields(total), answer_kind="number")


def _iol_rate_rise(rng):
    k = rng.randint(6, 60)
    loan = Fraction(k * _PRINCIPAL_UNIT)
    r1 = _rate(rng)
    rise = Fraction(rng.choice((1, 2, 3, 4))) / 2
    r2 = r1 + rise
    pay1 = loan * r1 / 1200
    extra = loan * rise / 1200
    d = MONEY.dist(
        pay1,
        [
            (extra, "iol-rise-as-payment"),
            (loan, "iol-loan-for-payment"),
            (extra * r1, "iol-rise-times-rate"),
            (extra * 12, "iol-annual-for-monthly"),
            (loan * r2 / 1200, "iol-rise-as-payment"),
        ],
    )
    enough(d, 3)
    text = (
        f"A borrower has an interest-only loan. The interest rate is {pct_text(r1)}% per annum. When the rate rises by {pct_text(rise)} percentage "
        f"points, the monthly payment increases by {dollars(extra)}. What was the monthly payment before the rise, in dollars?"
    )
    working = [
        f"Extra yearly interest = 12 × {dollars(extra)} = {dollars(extra * 12)}, which is {pct_text(rise)}% of the loan, so the loan is {dollars(loan)}.",
        f"Original monthly payment = {dollars(loan)} × {pct_text(r1)}% / 12 = {dollars(pay1)}.",
    ]
    return dict(text=text, working=working, distractors=d, **MONEY.fields(pay1), answer_kind="number")


def _iol_lump_sum(rng):
    loan = _loan(rng, 10, 60)
    r = _rate(rng)
    n = rng.randint(6, 12)
    k = rng.randint(2, n - 2)
    lump = Fraction(rng.randint(1, 5) * _PRINCIPAL_UNIT)
    if lump >= loan / 2:
        raise TooFewDistractors("lump sum too big")
    total = loan * r * k / 100 + (loan - lump) * r * (n - k) / 100
    d = MONEY.dist(
        total,
        [
            (loan * r * n / 100, "iol-lump-ignored"),
            (loan * r * n / 100 - lump, "iol-lump-off-interest"),
            ((loan - lump) * r * n / 100, "iol-lump-whole-term"),
            ((loan - lump) * r * (n - k) / 100, "iol-lump-after-only"),
        ],
    )
    enough(d, 3)
    text = (
        f"A borrower takes out an interest-only loan of {dollars(loan)} at {pct_text(r)}% per annum for {n} years. After {k} years the borrower pays "
        f"{dollars(lump)} off the amount owing, and the interest rate does not change. How much interest is paid over the {n} years, in dollars?"
    )
    working = [
        f"First {k} years: {k} × {dollars(loan * r / 100)} = {dollars(loan * r * k / 100)}.",
        f"After the lump sum the loan is {dollars(loan - lump)}; the last {n - k} years: {n - k} × {dollars((loan - lump) * r / 100)} = {dollars((loan - lump) * r * (n - k) / 100)}.",
        f"Total = {dollars(total)}.",
    ]
    return dict(text=text, working=working, distractors=d, **MONEY.fields(total), answer_kind="number")


_IOL_EASY = (_iol_annual, _iol_monthly, _iol_total_interest)
_IOL_MEDIUM = (_iol_total_repaid, _iol_loan_from_payment, _iol_rate_from_payment)
_IOL_HARD = (_iol_rate_change, _iol_rate_rise, _iol_lump_sum)


@generator(
    IOL_TOPIC,
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def w3_interest_only_loan(rng, tier="easy"):
    """Interest-only loans; the tiers are described in the module docstring."""
    pool = {"easy": _IOL_EASY, "medium": _IOL_MEDIUM, "hard": _IOL_HARD}.get(tier)
    if pool is None:
        raise ValueError(f"unknown tier {tier!r}")
    return retry_draw(rng, rng.choice(pool))


@generator(IOL_EXAM, level=19, presets={"exam": dict(tier="exam")})
def exd_x_w3_interest_only_loan(rng, tier="exam"):
    """Exam chain: monthly payment, interest over the term, total repaid, payment after a lump sum."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")

    def build(rng):
        loan, r = _loan(rng, 10, 60), _rate(rng)
        n = rng.randint(4, 10)
        lump = Fraction(rng.randint(1, 5) * _PRINCIPAL_UNIT)
        if lump >= loan / 2:
            raise TooFewDistractors("lump sum too big")
        annual = loan * r / 100
        monthly = annual / 12
        total_interest = annual * n
        total_repaid = total_interest + loan
        new_monthly = (loan - lump) * r / 1200
        a_d = MONEY.dist(
            monthly,
            [
                (annual, "iol-annual-for-monthly"),
                (annual * 12, "iol-multiplies-by-twelve"),
                (loan / 12, "iol-principal-share"),
                (monthly / 10, "iol-decimal-slip"),
            ],
        )
        b_d = MONEY.dist(
            total_interest,
            [
                (annual, "iol-one-year-only"),
                (annual * (n + 1), "iol-extra-year"),
                (monthly * n, "iol-months-as-years"),
                (loan * ((1 + r / 100) ** n - 1), "iol-compound"),
            ],
        )
        c_d = MONEY.dist(
            total_repaid,
            [
                (total_interest, "iol-forgets-principal"),
                (loan * (1 + r / 100) ** n, "iol-compound"),
                (loan + annual, "iol-one-year-only"),
                (loan + monthly * n, "iol-months-as-years"),
            ],
        )
        d_d = MONEY.dist(
            new_monthly,
            [
                (monthly, "iol-lump-ignored"),
                (monthly - lump / 12, "iol-lump-off-interest"),
                (monthly - lump, "iol-lump-off-interest"),
                ((loan - lump) * r / 100, "iol-annual-for-monthly"),
            ],
        )
        for x in (a_d, b_d, c_d, d_d):
            enough(x, 3)
        stem = _intro(rng, loan, r)
        return dict(
            text=stem,
            parts=[
                part(
                    "The interest is paid in equal monthly payments. Find each payment, in dollars.",
                    "number",
                    MONEY.fields(monthly),
                    1,
                    [f"{dollars(annual)} / 12 = {dollars(monthly)}."],
                    a_d,
                    sensible_range=(0, None),
                ),
                part(
                    f"How much interest is paid over {n} years, in dollars?",
                    "number",
                    MONEY.fields(total_interest),
                    2,
                    [f"{n} × 12 × {dollars(monthly)} = {dollars(total_interest)}."],
                    b_d,
                    follow_through=f"12*{n}*part_a",
                    sensible_range=(0, None),
                ),
                part(
                    f"At the end of {n} years the loan is repaid in full. How much has the borrower paid altogether, in dollars?",
                    "number",
                    MONEY.fields(total_repaid),
                    1,
                    [f"{dollars(total_interest)} + {dollars(loan)} = {dollars(total_repaid)}."],
                    c_d,
                    follow_through=f"part_b + {loan}",
                    sensible_range=(0, None),
                ),
                part(
                    f"Instead, the borrower pays {dollars(lump)} off the amount owing at the start. The rate stays the same. Find the new monthly payment, in dollars.",
                    "number",
                    MONEY.fields(new_monthly),
                    2,
                    [
                        f"New loan = {dollars(loan - lump)}; payment = {dollars(loan - lump)} × {pct_text(r)}% / 12 = {dollars(new_monthly)}."
                    ],
                    d_d,
                    sensible_range=(0, None),
                ),
            ],
        )

    return retry_draw(rng, build)


# -- annuity table ----------------------------------------------------------------------------------------------------


def factor(rate_pct, periods: int) -> Fraction:
    """The future value of $1 paid at the end of each period, rounded to 4 places (the table's entry)."""
    i = Fraction(rate_pct) / 100
    return round_half_up(((1 + i) ** periods - 1) / i, 4)


def _fmt4(x: Fraction) -> str:
    return f"{float(x):.4f}"


def _table_text(rows, label="per period") -> str:
    """rows: [(rate_pct, [periods...])] -> a one-line table the learner reads a cell from."""
    pieces = []
    for rate, periods in rows:
        cells = ", ".join(f"{n} periods {_fmt4(factor(rate, n))}" for n in periods)
        pieces.append(f"at {pct_text(rate)}% {label}: {cells}")
    return "Future value factors for $1 paid at the end of each period: " + "; ".join(pieces) + "."


_FACTOR_RATES = (2, 3, 4, 5, 6)
_AMOUNTS = tuple(range(100, 1501, 100))
_SAVING_PURPOSES = (
    "a deposit on a car",
    "a trip overseas",
    "a new kitchen",
    "a boat",
    "a future school fund",
    "a caravan",
)


def _pick_rows(rng, rate, n, per_rate=3):
    rates = sorted({rate, *rng.sample([x for x in _FACTOR_RATES if x != rate], 2)})
    start = rng.randint(max(2, n - 2), n)
    periods = list(range(start, start + per_rate))
    check(n in periods, "the needed period is in the table")
    return [(x, periods) for x in rates]


def _ann_one(rng):
    rate, n = rng.choice(_FACTOR_RATES), rng.randint(5, 12)
    pay = Fraction(rng.choice(_AMOUNTS))
    rows = _pick_rows(rng, rate, n)
    f = factor(rate, n)
    fv = pay * f
    check((fv * 100).denominator == 1, "exact cents")
    lower = factor(rate, n - 1) if n - 1 in rows[0][1] else factor(rate, n + 1)
    other = next(x for x, _ in rows if x != rate)
    d = MONEY.dist(
        fv,
        [
            (pay * lower, "ann-wrong-period-row"),
            (pay * factor(other, n), "ann-wrong-rate-row"),
            (pay * n, "ann-no-interest"),
            (pay * (1 + Fraction(rate, 100)) ** n, "ann-lump-sum-growth"),
            (f, "ann-factor-only"),
        ],
    )
    enough(d, 3)
    text = (
        f"{rng.choice(('Priya', 'Tom', 'Aisha', 'Mei', 'Jack', 'Leila'))} saves for {rng.choice(_SAVING_PURPOSES)} by paying {dollars(pay)} into an account at the end of "
        f"each period for {n} periods. The interest rate is {pct_text(rate)}% per period. {_table_text(rows)} "
        f"Use the table to find the future value of the savings, in dollars."
    )
    working = [
        f"From the table, the factor for {pct_text(rate)}% and {n} periods is {_fmt4(f)}.",
        f"Future value = {dollars(pay)} × {_fmt4(f)} = {dollars(fv)}.",
    ]
    return dict(text=text, working=working, distractors=d, **MONEY.fields(fv), answer_kind="number")


def _ann_payment(rng):
    rate, n = rng.choice(_FACTOR_RATES), rng.randint(5, 12)
    pay = Fraction(rng.choice(_AMOUNTS))
    rows = _pick_rows(rng, rate, n)
    f = factor(rate, n)
    target = pay * f
    d = MONEY.dist(
        pay,
        [
            (target * f, "ann-multiplies-for-payment"),
            (target / n, "ann-divides-by-periods"),
            (
                target / factor(rate, n - 1) if n - 1 in rows[0][1] else target / factor(rate, n + 1),
                "ann-wrong-period-row",
            ),
            (pay * f / 10, "ann-multiplies-for-payment"),
            (target / factor(next(x for x, _ in rows if x != rate), n), "ann-wrong-rate-row"),
        ],
    )
    enough(d, 3)
    text = (
        f"A couple want to save exactly {dollars(target)} for {rng.choice(_SAVING_PURPOSES)} by making equal payments at the end of each period for {n} periods. "
        f"The interest rate is {pct_text(rate)}% per period. {_table_text(rows)} "
        f"Use the table to find the payment needed each period, in dollars."
    )
    working = [
        f"From the table, the factor for {pct_text(rate)}% and {n} periods is {_fmt4(f)}.",
        f"Payment = {dollars(target)} / {_fmt4(f)} = {dollars(pay)}.",
    ]
    return dict(text=text, working=working, distractors=d, **MONEY.fields(pay), answer_kind="number")


def _ann_interest(rng):
    rate, n = rng.choice(_FACTOR_RATES), rng.randint(5, 12)
    pay = Fraction(rng.choice(_AMOUNTS))
    rows = _pick_rows(rng, rate, n)
    f = factor(rate, n)
    fv = pay * f
    interest = fv - pay * n
    d = MONEY.dist(
        interest,
        [
            (fv, "ann-future-value-for-interest"),
            (pay * n, "ann-contributions-for-interest"),
            (fv - pay * (n + 1), "ann-wrong-period-row"),
            (fv - pay * (n - 1), "ann-wrong-period-row"),
            (pay * (f - 1), "ann-wrong-period-row"),
        ],
    )
    enough(d, 3)
    text = (
        f"{rng.choice(('Sam', 'Nina', 'Omar', 'Kate', 'Ravi', 'Zoe'))} pays {dollars(pay)} at the end of each period for {n} periods into an account paying "
        f"{pct_text(rate)}% per period. {_table_text(rows)} "
        f"Use the table to find how much interest is earned altogether, in dollars."
    )
    working = [
        f"Future value = {dollars(pay)} × {_fmt4(f)} = {dollars(fv)}; contributions = {n} × {dollars(pay)} = {dollars(pay * n)}.",
        f"Interest = {dollars(fv)} - {dollars(pay * n)} = {dollars(interest)}.",
    ]
    return dict(text=text, working=working, distractors=d, **MONEY.fields(interest), answer_kind="number")


_PER_YEAR = {"monthly": 12, "quarterly": 4, "half-yearly": 2}
_CONV_RATES = {"monthly": (6, 12, 18, 24), "quarterly": (4, 8, 12, 16), "half-yearly": (4, 6, 8, 10)}


def _conv_case(rng, kind=None):
    kind = kind or rng.choice(("monthly", "quarterly", "half-yearly"))
    per = _PER_YEAR[kind]
    annual = rng.choice(_CONV_RATES[kind])
    i = Fraction(annual, per)
    years = rng.randint(2, 6)
    n = years * per
    return kind, per, annual, i, years, n


def _wrong_rate(kind, annual) -> Fraction:
    """The per-period rate a learner gets by dividing the annual rate by the wrong number of periods per year."""
    return Fraction(annual, {"monthly": 4, "quarterly": 2, "half-yearly": 4}[kind])


def _conv_rows(rng, kind, annual, i, n, per):
    """Table rows: the right rate per period, the rate a wrong division gives, and one more; periods around n."""
    rates = {i, _wrong_rate(kind, annual)}
    for cand in (i + Fraction(1, 2), Fraction(1), Fraction(2), Fraction(3)):
        if len(rates) < 3 and cand > 0:
            rates.add(cand)
    periods = [n - per, n, n + per] if n - per >= 2 else [n, n + per, n + 2 * per]
    return [(r, periods) for r in sorted(rates)]


def _ann_convert(rng):
    kind, per, annual, i, years, n = _conv_case(rng)
    pay = Fraction(rng.choice(_AMOUNTS))
    rows = _conv_rows(rng, kind, annual, i, n, per)
    f = factor(i, n)
    fv = pay * f
    wrong = _wrong_rate(kind, annual)
    near = next(r for r, _ in rows if r not in (i, wrong))
    d = MONEY.dist(
        fv,
        [
            (pay * factor(wrong, n), "ann-rate-not-divided"),
            (pay * factor(i, n - per) if n - per in rows[0][1] else pay * factor(i, n + per), "ann-wrong-period-row"),
            (pay * factor(near, n), "ann-wrong-rate-row"),
            (pay * n, "ann-no-interest"),
        ],
    )
    enough(d, 3)
    text = (
        f"{rng.choice(('Hana', 'Ben', 'Imogen', 'Dev', 'Claire', 'Luca'))} makes {kind} payments of {dollars(pay)} into an account that pays {annual}% per annum, with interest "
        f"added {kind}. The payments are made at the end of each period for {years} years. {_table_text(rows)} "
        f"Use the table to find the future value after {years} years, in dollars."
    )
    working = [
        f"Rate per period = {annual}% / {per} = {pct_text(i)}%; number of periods = {years} × {per} = {n}.",
        f"Factor = {_fmt4(f)}; future value = {dollars(pay)} × {_fmt4(f)} = {dollars(fv)}.",
    ]
    return dict(text=text, working=working, distractors=d, **MONEY.fields(fv), answer_kind="number")


def _ann_convert_payment(rng):
    kind, per, annual, i, years, n = _conv_case(rng)
    pay = Fraction(rng.choice(_AMOUNTS))
    rows = _conv_rows(rng, kind, annual, i, n, per)
    f = factor(i, n)
    target = pay * f
    wrong = _wrong_rate(kind, annual)
    near = next(r for r, _ in rows if r not in (i, wrong))
    d = MONEY.dist(
        pay,
        [
            (target / factor(wrong, n), "ann-rate-not-divided"),
            (target / factor(near, n), "ann-wrong-rate-row"),
            (
                target / factor(i, n - per) if n - per in rows[0][1] else target / factor(i, n + per),
                "ann-wrong-period-row",
            ),
            (target / n, "ann-divides-by-periods"),
            (target * f, "ann-multiplies-for-payment"),
        ],
    )
    enough(d, 3)
    text = (
        f"{rng.choice(('Hana', 'Ben', 'Imogen', 'Dev', 'Claire', 'Luca'))} wants {dollars(target)} in {years} years and will make equal {kind} payments at the end of each period "
        f"into an account that pays {annual}% per annum, with interest added {kind}. {_table_text(rows)} "
        f"Use the table to find the payment needed each period, in dollars."
    )
    working = [
        f"Rate per period = {annual}% / {per} = {pct_text(i)}%; number of periods = {years} × {per} = {n}.",
        f"Factor = {_fmt4(f)}; payment = {dollars(target)} / {_fmt4(f)} = {dollars(pay)}.",
    ]
    return dict(text=text, working=working, distractors=d, **MONEY.fields(pay), answer_kind="number")


_ANN_EASY = (_ann_one,)
_ANN_MEDIUM = (_ann_payment, _ann_interest)
_ANN_HARD = (_ann_convert, _ann_convert_payment)


@generator(
    ANN_TOPIC,
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def w3_annuity_table(rng, tier="easy"):
    """Future value of an annuity from a table of factors; the tiers are described in the module docstring."""
    pool = {"easy": _ANN_EASY, "medium": _ANN_MEDIUM, "hard": _ANN_HARD}.get(tier)
    if pool is None:
        raise ValueError(f"unknown tier {tier!r}")
    return retry_draw(rng, rng.choice(pool))


@generator(ANN_EXAM, level=19, presets={"exam": dict(tier="exam")})
def exd_x_w3_annuity_table(rng, tier="exam"):
    """Exam chain: future value from the table, total contributed, interest earned, payment for a target."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")

    def build(rng):
        kind, per, annual, i, years, n = _conv_case(rng)
        pay = Fraction(rng.choice(_AMOUNTS))
        pay2 = Fraction(rng.choice(_AMOUNTS))
        rows = _conv_rows(rng, kind, annual, i, n, per)
        f = factor(i, n)
        fv = pay * f
        contributed = pay * n
        interest = fv - contributed
        target = pay2 * f
        wrong = _wrong_rate(kind, annual)
        near = next(r for r, _ in rows if r not in (i, wrong))
        a_d = MONEY.dist(
            fv,
            [
                (pay * factor(wrong, n), "ann-rate-not-divided"),
                (
                    pay * factor(i, n - per) if n - per in rows[0][1] else pay * factor(i, n + per),
                    "ann-wrong-period-row",
                ),
                (pay * factor(near, n), "ann-wrong-rate-row"),
                (pay * n, "ann-no-interest"),
            ],
        )
        b_d = MONEY.dist(
            contributed,
            [
                (pay * years, "ann-years-as-periods"),
                (fv, "ann-future-value-for-interest"),
                (pay * (n + 1), "ann-wrong-period-row"),
                (pay * (n - 1), "ann-wrong-period-row"),
            ],
        )
        c_d = MONEY.dist(
            interest,
            [
                (fv, "ann-future-value-for-interest"),
                (contributed, "ann-contributions-for-interest"),
                (fv + contributed, "ann-difference-sign"),
                (fv - pay * years, "ann-years-as-periods"),
            ],
        )
        d_d = MONEY.dist(
            pay2,
            [
                (target * f, "ann-multiplies-for-payment"),
                (target / n, "ann-divides-by-periods"),
                (target / factor(near, n), "ann-wrong-rate-row"),
                (
                    target / factor(i, n - per) if n - per in rows[0][1] else target / factor(i, n + per),
                    "ann-wrong-period-row",
                ),
            ],
        )
        for x in (a_d, b_d, c_d, d_d):
            enough(x, 3)
        stem = (
            f"{rng.choice(('Hana', 'Ben', 'Imogen', 'Dev', 'Claire', 'Luca'))} makes {kind} payments of {dollars(pay)} into an account that pays {annual}% per annum, "
            f"with interest added {kind}, for {years} years. {_table_text(rows)}"
        )
        return dict(
            text=stem,
            parts=[
                part(
                    "Use the table to find the future value of the account, in dollars.",
                    "number",
                    MONEY.fields(fv),
                    2,
                    [
                        f"{pct_text(i)}% per period, {n} periods: factor {_fmt4(f)}; {dollars(pay)} × {_fmt4(f)} = {dollars(fv)}."
                    ],
                    a_d,
                    sensible_range=(0, None),
                ),
                part(
                    "How much has been paid into the account altogether, in dollars?",
                    "number",
                    MONEY.fields(contributed),
                    1,
                    [f"{n} × {dollars(pay)} = {dollars(contributed)}."],
                    b_d,
                    sensible_range=(0, None),
                ),
                part(
                    "How much interest has been earned, in dollars?",
                    "number",
                    MONEY.fields(interest),
                    1,
                    [f"{dollars(fv)} - {dollars(contributed)} = {dollars(interest)}."],
                    c_d,
                    follow_through=f"part_a - {contributed}",
                    sensible_range=(0, None),
                ),
                part(
                    f"Using the same account and term, what payment each period would give a future value of {dollars(target)}, in dollars?",
                    "number",
                    MONEY.fields(pay2),
                    2,
                    [f"Payment = {dollars(target)} / {_fmt4(f)} = {dollars(pay2)}."],
                    d_d,
                    sensible_range=(0, None),
                ),
            ],
        )

    return retry_draw(rng, build)
