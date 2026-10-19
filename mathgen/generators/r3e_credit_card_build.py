"""Credit-card statements: `r3e_credit_card` (easy, medium, hard) and the exam chain `exd_x_r3e_credit_card`.

``credit-card-interest-statement`` (``financial-discrete``, answer ``number``). Every question states the annual
rate, how interest is charged (daily, at one 365th of the annual rate, on the balance owing) and the minimum-payment
rule.

* easy: the interest on a closing balance left unpaid for a stated number of days.
* medium: the minimum payment under a "greater of a fixed sum and a percentage" rule; or the interest on what is left
  after the minimum has been paid.
* hard: how many monthly payments clear a card (interest each month at one twelfth of the annual rate, charged
  before the payment); the balance after three payments; or the total interest paid over the life of the debt.
* exam (``exd_x_r3e_credit_card``): the minimum payment, the balance after it, the interest on that balance for a
  stated number of days, and the balance owing after that interest.

Months to clear and total interest are rebuilt a second way from the closed form of the balance after k payments.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import attempts, check, generator
from ._exd_distinct_parts import distinct_part_answers
from ._r3e_kit import PEOPLE, dec, dollars, number_part, number_text, percent, single_number

_RATES = [Fraction("14.6"), Fraction("18.25"), Fraction("19.99"), Fraction("21.9"), Fraction("24.99")]
_DAYS = [21, 25, 28, 30, 31, 35, 40]


def _daily_interest(balance, rate, days):
    return Fraction(balance) * rate / 100 / 365 * days


def _min_payment(balance, fixed, pct):
    return max(Fraction(fixed), Fraction(balance) * pct / 100)


def _easy(rng):
    name, job = rng.choice(PEOPLE)
    balance = 50 * rng.randint(20, 160)
    rate = rng.choice(_RATES)
    days = rng.choice(_DAYS)
    interest = _daily_interest(balance, rate, days)
    check(interest == Fraction(balance) * (rate / 365 / 100) * days, "interest disagrees")
    text = (
        f"{name} is {job}. {name}'s credit card statement shows a closing balance of {dollars(balance)}, and "
        f"{name} pays nothing off it for {days} days. The interest rate is {percent(rate)}% a year, and interest is "
        "charged each day at one 365th of the yearly rate on the balance owing. How much interest is charged over "
        f"the {days} days? Give your answer in dollars, to the nearest cent."
    )
    slips = [
        (Fraction(balance) * rate / 100, "r3e-days-ignored"),
        (Fraction(balance) * rate / 100 / 12, "r3e-monthly-rate-used-for-days"),
        (Fraction(balance) * rate / 100 / 360 * days, "r3e-year-taken-as-360-days"),
        (Fraction(balance) * rate / 100 / 365, "r3e-one-day-only"),
        (Fraction(balance) * rate / 10000 / 365 * days, "r3e-percent-decimal-slip"),
    ]
    working = [
        f"Interest for one day = {dollars(balance)} x {percent(rate)}% / 365.",
        f"For {days} days: {dollars(balance)} x {dec(rate / 100)} / 365 x {days} = {dollars(interest, 2)}.",
    ]
    return single_number(text, interest, 2, working, slips)


def _medium_min(rng):
    name, job = rng.choice(PEOPLE)
    pct = rng.choice([Fraction(2), Fraction(3), Fraction(5)])
    fixed = rng.choice([20, 25, 30])
    # half the time the percentage rule wins, half the time the fixed sum does
    if rng.random() < 0.5:
        balance = 10 * rng.randint(int(fixed * 100 / pct / 10) + 20, int(fixed * 100 / pct / 10) + 400)
    else:
        balance = 10 * rng.randint(10, int(fixed * 100 / pct / 10) - 5)
    payment = _min_payment(balance, fixed, pct)
    text = (
        f"{name} is {job}. {name}'s credit card statement shows a closing balance of {dollars(balance)}. The minimum "
        f"payment is {percent(pct)}% of the closing balance or {dollars(fixed)}, whichever is greater. What is the "
        "minimum payment? Give your answer in dollars, to the nearest cent."
    )
    pct_amount = Fraction(balance) * pct / 100
    slips = [
        (min(pct_amount, Fraction(fixed)), "r3e-lesser-minimum-chosen"),
        (pct_amount + fixed, "r3e-minimum-rules-added"),
        (Fraction(balance) * pct / 1000, "r3e-percent-decimal-slip"),
        (Fraction(balance) - pct_amount, "r3e-percentage-taken-off-instead"),
        (Fraction(balance) / pct, "r3e-percentage-divided-not-multiplied"),
    ]
    working = [
        f"{percent(pct)}% of {dollars(balance)} = {dollars(pct_amount, 2)}.",
        f"The greater of {dollars(pct_amount, 2)} and {dollars(fixed)} is {dollars(payment, 2)}.",
    ]
    return single_number(text, payment, 2, working, slips)


def _medium_after(rng):
    name, job = rng.choice(PEOPLE)
    pct = rng.choice([Fraction(2), Fraction(3), Fraction(5)])
    fixed = rng.choice([20, 25, 30])
    balance = 10 * rng.randint(int(fixed * 100 / pct / 10) + 20, int(fixed * 100 / pct / 10) + 300)
    rate = rng.choice(_RATES)
    days = rng.choice([28, 30, 31])
    payment = _min_payment(balance, fixed, pct)
    left = Fraction(balance) - payment
    interest = _daily_interest(left, rate, days)
    text = (
        f"{name} is {job}. {name}'s credit card statement shows a closing balance of {dollars(balance)}. {name} pays "
        f"the minimum payment, which is {percent(pct)}% of the closing balance or {dollars(fixed)}, whichever is "
        f"greater. The rest is left unpaid for {days} days at {percent(rate)}% a year, with interest charged daily at "
        "one 365th of the yearly rate on the balance owing. How much interest is charged on the unpaid balance? Give "
        "your answer in dollars, to the nearest cent."
    )
    slips = [
        (_daily_interest(balance, rate, days), "r3e-interest-on-whole-balance"),
        (_daily_interest(payment, rate, days), "r3e-interest-on-payment"),
        (_daily_interest(left, rate, 365), "r3e-days-ignored"),
        (_daily_interest(left, rate, days) / 12 * 365 / days, "r3e-monthly-rate-used-for-days"),
        (_daily_interest(left, rate / 10, days), "r3e-percent-decimal-slip"),
    ]
    working = [
        f"Minimum payment = {dollars(payment, 2)}, so the unpaid balance is {dollars(balance)} - {dollars(payment, 2)} = {dollars(left, 2)}.",
        f"Interest = {dollars(left, 2)} x {dec(rate / 100)} / 365 x {days} = {dollars(interest, 2)}.",
    ]
    return single_number(text, interest, 2, working, slips)


def _months_to_clear(balance, payment, rate):
    """Payments needed when interest at rate/12 is added each month before the payment."""
    i = Fraction(rate) / 1200
    g = 1 + i
    owing = Fraction(balance)
    for n in range(1, 241):
        owing = owing * g - payment
        if owing <= 0:
            break
    else:
        raise AssertionError("card never clears")
    for k in range(n):  # closed form: balance after k payments, then does the next payment clear it?
        after_k = Fraction(balance) * g**k - payment * (g**k - 1) / i
        check((after_k * g <= payment) == (k == n - 1), "closed form disagrees with the run")
    return n


def _payments_balance(balance, payment, rate, k):
    i = Fraction(rate) / 1200
    g = 1 + i
    owing = Fraction(balance)
    for _ in range(k):
        owing = owing * g - payment
    closed = Fraction(balance) * g**k - payment * (g**k - 1) / i
    check(owing == closed, "balance disagrees with the closed form")
    return owing


def _hard_months(rng):
    for _ in attempts(500, "a card that clears in a sensible time"):
        balance = 100 * rng.randint(10, 60)
        payment = 50 * rng.randint(4, 20)
        rate = rng.choice(_RATES)
        if payment <= Fraction(balance) * rate / 1200 * 1.5:
            continue
        n = _months_to_clear(balance, payment, rate)
        if 4 <= n <= 24:
            break
    name, job = rng.choice(PEOPLE)
    text = (
        f"{name} is {job} with a credit card balance of {dollars(balance)} and makes no more purchases. Each month, "
        f"interest of one twelfth of the yearly rate of {percent(rate)}% is added to the balance, and then {name} "
        f"pays {dollars(payment)}. The last payment can be smaller. How many monthly payments does it take to clear "
        "the card?"
    )
    simple = -(-balance // payment)
    slips = [
        (simple, "r3e-interest-ignored"),
        (n + 1, "r3e-extra-payment-counted"),
        (n - 1, "r3e-one-payment-too-few"),
        (
            -(-balance // (payment - int(Fraction(balance) * rate / 100)))
            if payment > Fraction(balance) * rate / 100
            else n + 2,
            "r3e-yearly-interest-each-month",
        ),
    ]
    i = Fraction(rate) / 1200
    owing = Fraction(balance)
    working = []
    for m in range(1, n + 1):
        owing = owing * (1 + i) - payment
        if owing <= 0:
            working.append(f"Month {m}: the payment clears the balance.")
            break
        working.append(f"After payment {m} the balance is {dollars(owing, 2)}.")
    return single_number(text, n, 0, working, slips)


def _hard_balance(rng):
    name, job = rng.choice(PEOPLE)
    for _ in attempts(500, "a card still owing after three payments"):
        balance = 100 * rng.randint(15, 80)
        payment = 50 * rng.randint(6, 24)
        rate = rng.choice(_RATES)
        owing = _payments_balance(balance, payment, rate, 3)
        if owing > payment:
            break
    i = Fraction(rate) / 1200
    text = (
        f"{name} is {job} with a credit card balance of {dollars(balance)} and makes no more purchases. Each month, "
        f"interest of one twelfth of the yearly rate of {percent(rate)}% is added to the balance, and then {name} "
        f"pays {dollars(payment)}. Do not round until the final answer. What is the balance after the third monthly "
        "payment? Give your answer in dollars, to the nearest cent."
    )
    slips = [
        (Fraction(balance) * (1 + i) ** 3 - payment, "r3e-payments-not-applied-each-month"),
        (Fraction(balance) - 3 * payment + Fraction(balance) * i * 3, "r3e-interest-on-original-balance"),
        (Fraction(balance) - 3 * payment, "r3e-interest-ignored"),
        (_payments_balance(balance, payment, rate, 2), "r3e-one-payment-too-few"),
        (_payments_balance(balance, payment, rate / 12, 3), "r3e-monthly-rate-divided-twice"),
    ]
    owing_1 = Fraction(balance) * (1 + i) - payment
    working = [
        f"Monthly rate = {percent(rate)}% / 12.",
        f"After payment 1: {dollars(balance)} x {dec(1 + i)} - {dollars(payment)} = {dollars(owing_1, 2)}.",
        f"Repeating twice more gives {dollars(owing, 2)}.",
    ]
    return single_number(text, owing, 2, working, slips)


def _hard_interest(rng):
    for _ in attempts(500, "a card that clears in a sensible time"):
        balance = 100 * rng.randint(10, 45)
        payment = 50 * rng.randint(4, 14)
        rate = rng.choice(_RATES)
        if payment <= Fraction(balance) * rate / 1200 * 1.5:
            continue
        n = _months_to_clear(balance, payment, rate)
        if 3 <= n <= 14:
            break
    i = Fraction(rate) / 1200
    g = 1 + i
    owing = Fraction(balance)
    for _ in range(n - 1):
        owing = owing * g - payment
    final = owing * g
    total_paid = payment * (n - 1) + final
    total_interest = total_paid - balance
    # closed form: total interest = (n-1) payments + final payment - balance
    check(final == Fraction(balance) * g**n - payment * (g**n - g) / i, "last payment disagrees")
    name, job = rng.choice(PEOPLE)
    text = (
        f"{name} is {job} with a credit card balance of {dollars(balance)} and makes no more purchases. Each month, "
        f"interest of one twelfth of the yearly rate of {percent(rate)}% is added to the balance, and then {name} "
        f"pays {dollars(payment)}, except for the last payment, which is just enough to clear the card. Do not round "
        "until the final answer. How much interest does the card charge in total? Give your answer in dollars, to the "
        "nearest cent."
    )
    slips = [
        (payment * n - balance, "r3e-last-payment-taken-as-full"),
        (Fraction(balance) * i * n, "r3e-interest-on-original-balance"),
        (total_paid, "r3e-total-paid-given-as-interest"),
        (final, "r3e-last-payment-given-as-interest"),
        (payment * (n - 1) - balance, "r3e-last-payment-left-out"),
    ]
    working = [
        f"It takes {n} payments. The first {n - 1} are {dollars(payment)} each and the last is {dollars(final, 2)}.",
        f"Total paid = {dollars(payment * (n - 1))} + {dollars(final, 2)} = {dollars(total_paid, 2)}.",
        f"Interest = {dollars(total_paid, 2)} - {dollars(balance)} = {dollars(total_interest, 2)}.",
    ]
    return single_number(text, total_interest, 2, working, slips)


@generator(
    "r3e_credit_card",
    level=9,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r3e_credit_card(rng, tier="easy"):
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return rng.choice([_medium_min, _medium_after])(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return rng.choice([_hard_months, _hard_balance, _hard_interest])(rng)


# -- exam -------------------------------------------------------------------------------------------------------


@generator("exd_x_r3e_credit_card", level=14, presets={"exam": dict(tier="exam")})
@distinct_part_answers
def exd_x_r3e_credit_card(rng, tier="exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    name, job = rng.choice(PEOPLE)
    pct = rng.choice([Fraction(2), Fraction(3), Fraction(5)])
    fixed = rng.choice([20, 25, 30])
    balance = 10 * rng.randint(int(fixed * 100 / pct / 10) + 20, int(fixed * 100 / pct / 10) + 300)
    rate = rng.choice(_RATES)
    days = rng.choice([28, 30, 31])
    payment = _min_payment(balance, fixed, pct)
    left = Fraction(balance) - payment
    interest = _daily_interest(left, rate, days)
    owing = left + interest
    pct_amount = Fraction(balance) * pct / 100
    text = (
        f"{name} is {job}. {name}'s credit card statement shows a closing balance of {dollars(balance)}. The minimum "
        f"payment is {percent(pct)}% of the closing balance or {dollars(fixed)}, whichever is greater. {name} pays "
        f"only the minimum. The rest is left unpaid for {days} days at {percent(rate)}% a year, with interest "
        "charged daily at one 365th of the yearly rate on the balance owing."
    )
    rate_num = f"{rate.numerator}/{rate.denominator * 36500}"
    parts = [
        number_part(
            "Find the minimum payment, in dollars.",
            payment,
            2,
            2,
            [
                f"{percent(pct)}% of {dollars(balance)} = {dollars(pct_amount, 2)}; the greater of that and {dollars(fixed)} is {dollars(payment, 2)}."
            ],
            [
                (min(pct_amount, Fraction(fixed)), "r3e-lesser-minimum-chosen"),
                (pct_amount + fixed, "r3e-minimum-rules-added"),
                (Fraction(balance) * pct / 1000, "r3e-percent-decimal-slip"),
            ],
        ),
        number_part(
            "Find the balance left unpaid after the minimum payment, in dollars.",
            left,
            2,
            1,
            [f"{dollars(balance)} - {dollars(payment, 2)} = {dollars(left, 2)}."],
            [
                (Fraction(balance) + payment, "r3e-payment-added-to-balance"),
                (payment, "r3e-payment-given-as-balance"),
                (Fraction(balance), "r3e-payment-left-out"),
            ],
            follow_through=f"{balance} - part_a",
        ),
        number_part(
            f"Find the interest charged on the unpaid balance over the {days} days, in dollars.",
            interest,
            2,
            2,
            [f"{dollars(left, 2)} x {dec(rate / 100)} / 365 x {days} = {dollars(interest, 2)}."],
            [
                (_daily_interest(balance, rate, days), "r3e-interest-on-whole-balance"),
                (_daily_interest(left, rate, 365), "r3e-days-ignored"),
                (left * rate / 100 / 360 * days, "r3e-year-taken-as-360-days"),
                (_daily_interest(left, rate / 10, days), "r3e-percent-decimal-slip"),
            ],
            follow_through=f"part_b*{rate_num}*{days}",
        ),
        number_part(
            "Find the amount owing at the end of the period, in dollars.",
            owing,
            2,
            1,
            [f"{dollars(left, 2)} + {dollars(interest, 2)} = {dollars(owing, 2)}."],
            [
                (left - interest, "r3e-interest-subtracted"),
                (left, "r3e-interest-left-out"),
                (Fraction(balance) + interest, "r3e-interest-added-to-whole-balance"),
            ],
            follow_through="part_b + part_c",
        ),
    ]
    return dict(text=text, parts=parts)
