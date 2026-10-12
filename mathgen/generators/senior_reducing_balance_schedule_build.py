"""Generates `senior-use-a-recurrence-relation-to-model-a-reducing` questions (ACMGM097): complete the blank
cells of a reducing-balance loan schedule built from the recurrence A(n+1) = A(n)(1 + r) - R. `answer: table`;
every blank is a dollar amount to the cent.

Scope: one loan, repaid by equal instalments R at the end of each period, with interest r per period charged on
the balance and rounded to the nearest cent each period. The schedule has a row per period (opening balance,
interest, repayment, closing balance), the opening balance of each period being the closing balance of the one
before. The blank cells are always a block at the foot of the table, so a blank can only be reached by running
the recurrence forward from given figures (never read back off a neighbouring given cell). Every figure is
built in whole cents with exact `Fraction` arithmetic; ties at half a cent are never drawn, so the rounding
rule cannot be read two ways.

Ladder (reasoning escalation, not size):
  easy:   the monthly rate and repayment are stated in words; the learner completes the last row, one step of
          the recurrence (interest on the opening balance, then closing balance).
  medium: the loan is given as the recurrence A(n+1) = 1.007 A(n) - R with A(0); the learner reads the rate off
          the multiplier and runs the recurrence for two or three periods, each opening balance being the
          closing balance the learner has just found.
  hard:   an annual rate with interest added monthly or quarterly. The learner first converts it to a rate per
          period, then runs the recurrence for two to four periods and totals the interest and the repayments.
  exam:   neither the rate nor the repayment is stated. The first row shows an opening balance, the interest
          for the period and the closing balance. The learner recovers the rate (interest / opening balance)
          and the repayment (opening + interest - closing), then continues the schedule and totals it.
"""

from __future__ import annotations

import random
from fractions import Fraction

from mathgen.core import check, draw_until, generator, phrase, round_half_up

from ._senior_table_kit import build_answer, finish_wrong, grid_text, money

TIERS = ("easy", "medium", "hard", "exam")

_PURPOSES = [
    "car loan",
    "home renovation loan",
    "caravan loan",
    "solar panel loan",
    "personal loan",
    "boat loan",
    "tractor loan",
    "ute loan",
]

_BORROWERS = ["Priya", "Tom", "Mei", "Sam", "Aroha", "Liam", "Fatima", "Jack", "Nadia", "Ben"]

_COLUMNS = ["Opening balance ($)", "Interest ($)", "Repayment ($)", "Closing balance ($)"]


def _dollars(cents) -> Fraction:
    return Fraction(cents, 100)


def _stem_money(cents: int) -> str:
    """A whole-dollar or dollars-and-cents amount for running text, with a thousands comma."""
    dollars, rem = divmod(int(cents), 100)
    return f"${dollars:,}" if rem == 0 else f"${dollars:,}.{rem:02d}"


def _rate_text(tenths: int) -> str:
    """A rate given in tenths of a percent (e.g. 7 -> "0.7", 72 -> "7.2") with trailing zeros dropped."""
    return f"{tenths / 10:.2f}".rstrip("0").rstrip(".")


def _schedule(principal: int, rate: Fraction, repayment: int, periods: int):
    """[(opening, interest, repayment, closing)] in cents, interest rounded half-up each period."""
    rows, balance = [], principal
    for _ in range(periods):
        interest = int(round_half_up(Fraction(balance) * rate, 0))
        closing = balance + interest - repayment
        rows.append((balance, interest, repayment, closing))
        balance = closing
    return rows


def _no_ties(principal: int, rate: Fraction, repayment: int, periods: int) -> bool:
    """True when no period's interest sits exactly half-way between two cents, and the loan stays open."""
    balance = principal
    for _ in range(periods):
        raw = Fraction(balance) * rate
        if (raw - (raw.numerator // raw.denominator)) == Fraction(1, 2):
            return False
        balance = balance + int(round_half_up(raw, 0)) - repayment
        if balance <= repayment:
            return False
    return True


def _draw_loan(rng, principals, rates, periods):
    """(principal cents, rate per period as a Fraction, repayment cents) with a balance that reduces."""

    def draw(r):
        principal = r.randrange(*principals) * 100
        rate = r.choice(rates)
        interest = principal * rate / 100  # dollars
        low = -(-int(interest * Fraction(13, 10)) // 10) * 10 + 10
        high = int((interest + Fraction(principal, 100 * 3 * periods)) // 10) * 10
        return principal, rate, r.randrange(low, max(high, low) + 1, 10) * 100

    return draw_until(
        rng,
        draw,
        lambda v: v[2] > v[0] * v[1] * Fraction(12, 10) and _no_ties(v[0], v[1], v[2], periods),
        what="a reducing-balance loan that stays open",
    )


def _row_labels(word: str, periods: int) -> list[str]:
    return [f"{word} {k}" for k in range(1, periods + 1)]


def _cells(schedule, word: str, with_totals: bool):
    """Table rows (strings/ints as shown) from the schedule in cents."""
    labels = _row_labels(word, len(schedule))
    rows = [[labels[i], *[money(_dollars(v)) for v in schedule[i]]] for i in range(len(schedule))]
    if with_totals:
        rows.append(
            [
                "Total",
                "-",
                money(_dollars(sum(r[1] for r in schedule))),
                money(_dollars(sum(r[2] for r in schedule))),
                "-",
            ]
        )
    return rows


def _block(periods: int, first_blank_row: int, extra=()):
    """Blank cells for rows first_blank_row.. (1-based): interest, closing, and every opening after the first."""
    cells = set(extra)
    for k in range(first_blank_row, periods + 1):
        cells |= {(k - 1, 2), (k - 1, 4)}
        if k > first_blank_row:
            cells.add((k - 1, 1))
    return cells


def _wrong_options(schedule, rate, annual_rate, blanks, principal):
    """{key: [(misconception id, wrong value in dollars)]} for the blank cells, one error at a time."""
    options: dict[str, list] = {}
    for r, c in sorted(blanks):
        if r >= len(schedule):
            continue
        opening, interest, repay, closing = schedule[r]
        if c == 2:
            opts = [
                ("charges-interest-on-the-original-balance", Fraction(principal) * rate),
                ("charges-interest-after-taking-off-the-repayment", Fraction(opening - repay) * rate),
            ]
            if annual_rate is not None:
                opts.append(("uses-the-annual-rate-each-period", Fraction(opening) * annual_rate))
        elif c == 4:
            opts = [
                ("forgets-the-interest-in-the-closing-balance", Fraction(opening - repay)),
                ("adds-the-repayment-to-the-balance", Fraction(opening + interest + repay)),
                ("forgets-the-repayment-in-the-closing-balance", Fraction(opening + interest)),
            ]
        elif c == 1 and r > 0:
            prev = schedule[r - 1]
            opts = [
                ("takes-the-repayment-off-without-adding-interest", Fraction(prev[0] - prev[2])),
                ("carries-the-previous-opening-balance-forward", Fraction(prev[0])),
            ]
        elif c == 3:
            opts = [("finds-the-repayment-without-the-interest", Fraction(opening - closing))]
        else:
            opts = []
        options[f"{r},{c}"] = [(mid, v / 100) for mid, v in opts]
    return options


def _totals_wrong(schedule, n_rows, blanks, options):
    if (n_rows, 2) in blanks:
        total = sum(r[1] for r in schedule)
        options[f"{n_rows},2"] = [("leaves-out-a-period-from-the-total", _dollars(total - schedule[0][1]))]
    if (n_rows, 3) in blanks:
        total = sum(r[2] for r in schedule)
        options[f"{n_rows},3"] = [("leaves-out-a-period-from-the-total", _dollars(total - schedule[0][2]))]


# --------------------------------------------------------------------------------------------------
# tiers
# --------------------------------------------------------------------------------------------------


def _easy(rng):
    periods = rng.choice([3, 4, 5])
    rate_pm = rng.choice([5, 6, 7, 8, 9, 10, 12])  # thousandths per month
    principal, rate, repayment = _draw_loan(rng, (4000, 18001, 500), [Fraction(rate_pm, 1000)], periods)
    sched = _schedule(principal, rate, repayment, periods)
    rows = _cells(sched, "Month", False)
    blanks = _block(periods, periods)
    answer = build_answer(["Month", *_COLUMNS], rows, blanks)
    options = _wrong_options(sched, rate, None, blanks, principal)
    who, purpose = rng.choice(_BORROWERS), rng.choice(_PURPOSES)
    stem = phrase(
        rng,
        [
            "{who} took out a {purpose} of {p}. At the end of each month, interest of {r}% of the balance is added "
            "and a repayment of {R} is made, so A(n+1) = A(n)(1 + r) - R, where A(n) is the balance after n months. "
            "The interest is rounded to the nearest cent each month. The table shows the loan for {n} months. "
            "Complete the last row, with each amount in dollars to the cent. Fill in each ? with a number.",
            "{who}'s {purpose} of {p} is repaid monthly. Each month the balance is increased by {r}% interest "
            "and then reduced by a repayment of {R} (A(n+1) = A(n)(1 + r) - R), with the interest rounded to the "
            "nearest cent. The table follows the loan for {n} months. Complete the last row, with each amount in "
            "dollars to the cent. Fill in each ? with a number.",
        ],
        who=who,
        purpose=purpose,
        p=_stem_money(principal),
        r=_rate_text(rate_pm),
        R=_stem_money(repayment),
        n=periods,
    )
    return answer, stem, options, sched, rate


def _medium(rng):
    periods = 4
    rate_pm = rng.choice([5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    principal, rate, repayment = _draw_loan(rng, (6000, 30001, 500), [Fraction(rate_pm, 1000)], periods)
    sched = _schedule(principal, rate, repayment, periods)
    rows = _cells(sched, "Month", False)
    blanks = _block(periods, rng.choice([2, 3]))
    answer = build_answer(["Month", *_COLUMNS], rows, blanks)
    options = _wrong_options(sched, rate, None, blanks, principal)
    who, purpose = rng.choice(_BORROWERS), rng.choice(_PURPOSES)
    multiplier = Fraction(1) + rate
    stem = phrase(
        rng,
        [
            "{who}'s {purpose} is modelled by the recurrence relation A(n+1) = {m} x A(n) - {R}, with A(0) = {p}, "
            "where A(n) is the balance in dollars after n monthly repayments. The table shows the schedule, with "
            "interest rounded to the nearest cent each month. Complete the blank cells, with each amount in dollars "
            "to the cent. Fill in each ? with a number.",
            "The balance of {who}'s {purpose} after n monthly repayments is A(n), where A(0) = {p} and "
            "A(n+1) = {m} x A(n) - {R}. The table shows the schedule, with interest rounded to the nearest cent "
            "each month. Complete the blank cells, with each amount in dollars to the cent. Fill in each ? with a "
            "number.",
        ],
        who=who,
        purpose=purpose,
        m=f"{float(multiplier):.3f}",
        R=_stem_money(repayment).lstrip("$").replace(",", ""),
        p=principal // 100,
    )
    return answer, stem, options, sched, rate


def _hard(rng):
    monthly = rng.random() < 0.65
    periods_per_year = 12 if monthly else 4
    word = "Month" if monthly else "Quarter"
    annual_tenths = rng.choice([48, 54, 60, 66, 72, 78, 84, 90, 96] if monthly else [40, 48, 56, 60, 64, 72, 80])
    rate = Fraction(annual_tenths, 1000 * periods_per_year)
    periods = 5
    principal, rate, repayment = _draw_loan(rng, (10000, 45001, 1000), [rate], periods)
    sched = _schedule(principal, rate, repayment, periods)
    rows = _cells(sched, word, True)
    blanks = _block(periods, rng.choice([2, 3]), extra=[(periods, 2), (periods, 3)])
    answer = build_answer([word, *_COLUMNS], rows, blanks)
    options = _wrong_options(sched, rate, Fraction(annual_tenths, 1000), blanks, principal)
    _totals_wrong(sched, periods, blanks, options)
    who, purpose = rng.choice(_BORROWERS), rng.choice(_PURPOSES)
    stem = phrase(
        rng,
        [
            "{who} has a {purpose} of {p} at {a}% per annum, with interest calculated and added {freq}. The balance "
            "A(n) after n {unit}s follows A(n+1) = A(n)(1 + r) - R, where r is the interest rate per {unit} and "
            "R = {R} is the {unit}ly repayment. Interest is rounded to the nearest cent each {unit}. The table shows "
            "the first {n} {unit}s. Complete the blank cells, with each amount in dollars to the cent. Fill in each "
            "? with a number.",
            "A {purpose} of {p} for {who} charges {a}% per annum, with interest added {freq}, and is repaid by {R} "
            "each {unit}. The balance follows A(n+1) = A(n)(1 + r) - R, where r is the rate per {unit}, with the "
            "interest rounded to the nearest cent each {unit}. The table shows the first {n} {unit}s. Complete the "
            "blank cells, with each amount in dollars to the cent. Fill in each ? with a number.",
        ],
        who=who,
        purpose=purpose,
        p=_stem_money(principal),
        a=_rate_text(annual_tenths),
        freq="monthly" if monthly else "quarterly",
        unit=word.lower(),
        R=_stem_money(repayment),
        n=periods,
    )
    return answer, stem, options, sched, rate


def _exam(rng):
    periods = 3
    rate_pm = rng.choice([4, 5, 6, 7, 8, 9, 10, 11, 12])
    principal, rate, repayment = _draw_loan(rng, (6000, 30001, 500), [Fraction(rate_pm, 1000)], periods)
    check(Fraction(principal) * rate == int(Fraction(principal) * rate), "first interest must be whole cents")
    sched = _schedule(principal, rate, repayment, periods)
    rows = _cells(sched, "Month", True)
    # Row 1 shows opening, interest and closing; its repayment is left out so the learner must recover it.
    blanks = {(0, 3), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4), (3, 2), (3, 3)}
    answer = build_answer(["Month", *_COLUMNS], rows, blanks)
    options = _wrong_options(sched, rate, None, blanks, principal)
    _totals_wrong(sched, periods, blanks, options)
    who, purpose = rng.choice(_BORROWERS), rng.choice(_PURPOSES)
    stem = phrase(
        rng,
        [
            "{who} repays a {purpose} with the same repayment at the end of every month, and the interest rate per "
            "month is the same every month. Interest is rounded to the nearest cent each month, and the balance "
            "follows A(n+1) = A(n)(1 + r) - R. The table shows the first month of the schedule, in which the "
            "repayment is left out. Use it to find r and R, then complete the blank cells, with each amount in "
            "dollars to the cent. Fill in each ? with a number.",
            "A {purpose} taken out by {who} is repaid by a fixed amount at the end of each month, at a fixed "
            "interest rate per month, with interest rounded to the nearest cent each month (A(n+1) = A(n)(1 + r) - "
            "R). Neither r nor R is stated, but the first month of the schedule shows the opening balance, the "
            "interest and the closing balance. Work out r and R, then complete the blank cells, with each amount in "
            "dollars to the cent. Fill in each ? with a number.",
        ],
        who=who,
        purpose=purpose,
    )
    return answer, stem, options, sched, rate


_BUILDERS = {"easy": _easy, "medium": _medium, "hard": _hard, "exam": _exam}


def _recheck(answer: dict, sched, rate: Fraction) -> None:
    """Run the recurrence again from the first opening balance and compare every cell of the finished table."""
    rows = [list(r) for r in answer["rows"]]
    for key, value in answer["blanks"].items():
        r, c = map(int, key.split(","))
        rows[r][c] = value
    balance = int(Fraction(rows[0][1]) * 100)
    repayment = int(Fraction(rows[0][3]) * 100)
    interest_total = repaid_total = 0
    for k in range(len(sched)):
        interest = int(round_half_up(Fraction(balance) * rate, 0))
        closing = balance + interest - repayment
        expected = [money(_dollars(v)) for v in (balance, interest, repayment, closing)]
        check(rows[k][1:5] == expected, "schedule row disagrees with the recurrence")
        interest_total += interest
        repaid_total += repayment
        balance = closing
    if len(rows) > len(sched):
        check(rows[-1][2] == money(_dollars(interest_total)), "total interest disagrees")
        check(rows[-1][3] == money(_dollars(repaid_total)), "total repaid disagrees")


@generator(
    "senior_reducing_balance_schedule",
    level=13,
    presets={t: dict(tier=t) for t in TIERS},
)
def senior_reducing_balance_schedule(rng: random.Random, tier: str = "easy") -> dict:
    if tier not in _BUILDERS:
        raise ValueError(f"unknown tier {tier!r}")
    answer, stem, options, sched, rate = _BUILDERS[tier](rng)
    _recheck(answer, sched, rate)
    answer_text, distractors, notes = finish_wrong(rng, answer, options)
    working = [
        f"Month {k + 1}: interest {money(_dollars(s[1]))}, closing balance {money(_dollars(s[3]))}"
        for k, s in enumerate(sched)
    ]
    return dict(
        text=f"{stem} {grid_text(answer)}",
        answer=answer,
        answer_text=answer_text,
        working=working,
        distractors=distractors,
        option_notes=notes,
    )
