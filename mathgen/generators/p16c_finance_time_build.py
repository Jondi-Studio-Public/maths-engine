"""Generates `p16c_loan_recurrence_balance`, `p16c_compound_effect`, `p16c_worker_wage_table` and
`p16c_gmt_dateline` questions on loan balances, compounding, wage tables and time zones.

Four NEW senior finance/time skills.

- ``p16c_loan_recurrence_balance`` (senior-use-technology-and-a-recurrence-relation-to-model,
  ACMEM174) -- the outstanding balance of a reducing-balance loan after a stated number of
  periods, computed by repeated application of the loan's own recurrence relation
  B(n+1) = B(n)(1 + r) - R. Separated from ``flat-loan-repayment`` (a flat, non-compounding
  interest charge divided evenly across the term -- no recurrence, no period-by-period
  balance at all) and from ``ucv1-recurrence-two-step`` (an abstract second-order recurrence
  with no money and no repayment subtraction).

- ``p16c_compound_effect`` (senior-use-technology-to-investigate-the-effect-of-the, ACMEM173)
  -- the EFFECT of a changed compounding frequency or interest rate on future value: the
  dollar difference between two future values (or, at the exam tier, the percentage by which
  one exceeds the other). Separated from ``compound-interest`` (a single balance under ONE
  stated annual-compounding rate) precisely because the core operation is the *difference*
  between two scenarios, a quantity ``compound-interest`` never computes.

- ``p16c_worker_wage_table`` (senior-use-a-spreadsheet-to-display-examples-of-the, ACMGM009)
  -- reads one named worker's row (or, at the exam tier, two rows) out of a small
  multi-worker wage table and computes the weekly total. Separated from ``calculate-wages``
  (one worker, one rate, one hours figure, no table, no row to pick) by the table itself:
  every tier here requires picking the right row (and, from medium on, the right of several
  rate columns) before any arithmetic starts.

- ``p16c_gmt_dateline`` (senior-solve-problems-involving-greenwich-mean-time-and-the,
  ACMEM164) -- a flight's local arrival day and time, computed from each place's stated GMT
  offset, explicitly crossing the International Date Line. Both shipping time-zone skills
  (``senior-solve-problems-associated-with-time-zones-for-example``,
  ``senior-solve-problems-involving-time-zones-in-australia-and``) return a plain number of
  hours and never touch the calendar date; this skill's answer is always a day-and-time
  string and the calendar date changing is the whole point. ``answer: mcq`` (options are
  day+time strings, plain text, never LaTeX -- see the module's ``_mcq`` helper).

Every skill works in exact ``Fraction`` arithmetic throughout and rounds only the FINAL
displayed value, formatted with
``stated_precision_string(value, 2)`` so the real grader's 2dp tolerance
(``stated_precision_tolerance(2) == Fraction(1, 200)``) always matches what is displayed.
"""

from __future__ import annotations

from fractions import Fraction

from mathgen.core import (
    check,
    check_distinct,
    check_equivalent,
    generator,
    pick_distractors,
    round_half_up,
    stated_precision_string,
    stated_precision_tolerance,
)
from mathgen.slots import NAMES

_TOL2 = stated_precision_tolerance(2)  # Fraction(1, 200) -- 2dp money tolerance


def _money(value: Fraction) -> str:
    """Round `value` (exact dollars) half-up to the cent and display at
    exactly 2dp -- the minimum standalone version of the queued
    `money_cents_rounded_2dp` helper (see module docstring)."""
    return stated_precision_string(round_half_up(value, 2), 2)


def _money_signed(value: Fraction) -> str:
    v = round_half_up(value, 2)
    sign = "-" if v < 0 else ""
    return sign + stated_precision_string(abs(v), 2)


def _drop_collisions_money(answer_rounded: Fraction, money_candidates: list[str]) -> list[str]:
    """Drop any money-string candidate whose rounded-to-the-cent value
    equals the answer's -- a handful of this module's misconception
    formulas can coincide with the answer for an unrelated reason on a rare
    seed (e.g. two independently-drawn workers sharing the same hourly rate),
    which is an incidental numeric coincidence, not a construction bug.
    `pick_distractors` would silently drop the same candidate at serve time
    (its own tolerance-based dedup); doing it here too, before the stricter
    `check_distinct` calls, keeps those calls meaningful (a real, systematic
    formula bug still fails on most seeds) rather than spuriously failing a
    build over one unlucky draw."""
    return [c for c in money_candidates if round_half_up(_parse_money(c), 2) != answer_rounded]


def _ceil_frac(value: Fraction) -> Fraction:
    """Smallest integer >= `value`, exact (no float roundoff)."""
    return Fraction(-(-value.numerator // value.denominator))


def _fmt_pct(p: Fraction) -> str:
    """ "1.25" not "1.25000..." -- p is a small Fraction of percent points."""
    if p.denominator == 1:
        return str(p.numerator)
    return str(float(p)).rstrip("0").rstrip(".") if float(p) == round(float(p), 4) else str(p)


# =============================================================================
# Skill 1: p16c_loan_recurrence_balance (ACMEM174)
# =============================================================================

_LOAN_RATES = (Fraction(1, 2), Fraction(3, 4), Fraction(1, 1), Fraction(5, 4), Fraction(3, 2))
_LOAN_R_MULT = (3, 4, 5, 6)

# The answer is stored to the cent, so the question says so: the grader marks a stated value exactly.
_TO_THE_CENT = " Give your answer to the nearest cent."

_LOAN_FORWARD_TEMPLATES = (
    "A loan of ${p} is charged {r}% interest each period. A regular repayment of ${R} is "
    "made at the end of every period, following the loan's own recurrence relation "
    "B(n+1) = B(n)(1 + r) - R.{extra}Find the outstanding balance after {n} periods.",
    "A borrower owes ${p}. Each period, interest of {r}% is added to the balance and then a "
    "fixed repayment of ${R} is subtracted -- the loan's own recurrence relation.{extra}What "
    "is the balance remaining after {n} periods?",
    "Under a reducing-balance loan of ${p} at {r}% interest per period, with a fixed "
    "repayment of ${R} made every period, use the recurrence relation to find the balance "
    "remaining after {n} periods.{extra}",
)


def _iterate_balance(principal, r: Fraction, R: Fraction, n: int, extra_at=None, extra=Fraction(0)):
    b = Fraction(principal)
    for i in range(1, n + 1):
        pay = R + (extra if i == extra_at else 0)
        b = b * (1 + r) - pay
    return b


def _stream_balance(principal, r: Fraction, R: Fraction, n: int, extra_at=None, extra=Fraction(0)):
    """Independent route: the future value of the principal MINUS the future
    value of every outflow (a payment made `k` periods before the end still
    grows by `(1 + r) ** (n - k)`). Mathematically the same recurrence, but a
    genuinely different code path from `_iterate_balance`'s forward loop, so
    a bug in either (an off-by-one bound, a misplaced `extra`) is caught."""
    b = Fraction(principal) * (1 + r) ** n
    for i in range(1, n + 1):
        pay = R + (extra if i == extra_at else 0)
        b -= pay * (1 + r) ** (n - i)
    return b


def _loan_forward(rng, n_lo: int, n_hi: int, lump: bool):
    principal = rng.randint(8, 60) * 1000
    rate_pct = rng.choice(_LOAN_RATES)
    r = rate_pct / 100
    n = rng.randint(n_lo, n_hi)
    interest0 = Fraction(principal) * r
    R = _ceil_frac(interest0) * rng.choice(_LOAN_R_MULT)

    extra_at, extra = None, Fraction(0)
    extra_text = " "
    if lump:
        extra_at = rng.randint(2, n - 1)
        extra = R * rng.choice((Fraction(1, 2), Fraction(1, 1), Fraction(3, 2)))
        extra_text = f" In period {extra_at}, an extra one-off repayment of ${_money(extra)} is also made. "

    balance = _iterate_balance(principal, r, R, n, extra_at, extra)
    balance_check = _stream_balance(principal, r, R, n, extra_at, extra)
    check_equivalent(balance, balance_check, "loan balance: loop and payment-stream routes disagree")

    template = rng.choice(_LOAN_FORWARD_TEMPLATES)
    text = template.format(
        p=principal, r=_fmt_pct(rate_pct), R=(int(R) if R.denominator == 1 else float(R)), extra=extra_text, n=n
    )
    text = text.rstrip() + _TO_THE_CENT

    answer_text = _money(balance)
    candidates = [
        _money(Fraction(principal) * (1 + r) ** n),  # forgot the repayment entirely
        _money(_wrong_order_balance(principal, r, R, n)),  # repays before growing, not after
        _money(Fraction(principal) - R * n),  # linear, ignores compounding entirely
    ]
    if lump:
        candidates += [
            _money(_iterate_balance(principal, r, R, n)),  # forgot the lump sum
            _money(_iterate_balance(principal, r, R, n, n, extra)),  # lump at the wrong (last) period
        ]
    candidates = _drop_collisions_money(round_half_up(balance, 2), candidates)
    check_distinct(
        round_half_up(balance, 2),
        *[round_half_up(_parse_money(c), 2) for c in candidates],
        msg="loan balance distractor equals the answer",
    )
    distractors = pick_distractors(answer_text, candidates, n=3, tolerance=_TOL2)
    return dict(
        text=text,
        answer=float(round_half_up(balance, 2)),
        answer_text=answer_text,
        answer_latex=f"\\${answer_text}",
        distractors=distractors,
        working=[f"Applying B(n+1) = B(n)(1 + r) - R for {n} periods gives ${answer_text}."],
    )


def _wrong_order_balance(principal, r: Fraction, R: Fraction, n: int) -> Fraction:
    b = Fraction(principal)
    for _ in range(n):
        b = (b - R) * (1 + r)
    return b


def _parse_money(s: str) -> Fraction:
    neg = s.startswith("-")
    s = s.lstrip("-")
    whole, frac = s.split(".")
    return (Fraction(-1) if neg else Fraction(1)) * (Fraction(int(whole)) + Fraction(int(frac), 100))


def _closed_balance(principal, r: Fraction, R: Fraction, k: int) -> Fraction:
    if k == 0:
        return Fraction(principal)
    if r == 0:
        return Fraction(principal) - R * k
    return Fraction(principal) * (1 + r) ** k - R * ((1 + r) ** k - 1) / r


_LOAN_FIND_N_TEMPLATES = (
    "A reducing-balance loan of ${p} is charged {r}% interest each period, with a fixed "
    "repayment of ${R} made every period, following B(n+1) = B(n)(1 + r) - R. After how many "
    "whole periods does the balance first fall to ${T} or below?",
    "A loan of ${p} at {r}% interest per period is repaid at a fixed ${R} every period, using "
    "the recurrence relation B(n+1) = B(n)(1 + r) - R. What is the smallest whole number of "
    "periods after which the balance is ${T} or less?",
    "Applying the recurrence relation B(n+1) = B(n)(1 + r) - R to a ${p} loan charged {r}% "
    "interest per period with a ${R} regular repayment, find the first whole period at which "
    "the balance drops to ${T} or below.",
)


def _loan_find_n(rng):
    principal = rng.randint(8, 60) * 1000
    rate_pct = rng.choice(_LOAN_RATES)
    r = rate_pct / 100
    interest0 = Fraction(principal) * r
    R = _ceil_frac(interest0) * rng.choice(_LOAN_R_MULT)
    frac_target = rng.choice((Fraction(1, 2), Fraction(3, 5), Fraction(7, 10), Fraction(4, 5)))
    T = round_half_up(Fraction(principal) * frac_target, 0)

    n_correct, b = 0, Fraction(principal)
    while b > T:
        b = b * (1 + r) - R
        n_correct += 1
        check(n_correct < 1000, "runaway search: this loan never reaches the target balance")
    check(n_correct >= 2, "search resolved too quickly to be a genuine search")

    check(_closed_balance(principal, r, R, n_correct) <= T, "closed form disagrees with the search at n")
    check(_closed_balance(principal, r, R, n_correct - 1) > T, "closed form disagrees with the search at n-1")

    r_disp = int(R) if R.denominator == 1 else float(R)
    template = rng.choice(_LOAN_FIND_N_TEMPLATES)
    text = template.format(p=principal, r=_fmt_pct(rate_pct), R=r_disp, T=int(T))
    answer_text = str(n_correct)
    d_over = n_correct + 1
    d_under = max(0, n_correct - 1)
    d_linear = int(_ceil_frac((Fraction(principal) - T) / R))
    raw = [v for v in (d_over, d_under, d_linear) if v != n_correct]
    check_distinct(n_correct, *raw, msg="loan find-n distractor equals the answer")
    distractors = pick_distractors(answer_text, [str(v) for v in raw], n=3)
    return dict(
        text=text,
        answer=n_correct,
        answer_text=answer_text,
        answer_latex=answer_text,
        distractors=distractors,
        working=[
            f"Applying the recurrence period by period, the balance first falls to ${int(T)} or "
            f"below after {n_correct} periods."
        ],
    )


_LOAN_RATE_CHANGE_TEMPLATES = (
    "A reducing-balance loan of ${p} is charged {r1}% interest per period for the first {n1} "
    "periods, with a fixed repayment of ${R} made every period. After that, the interest rate "
    "changes to {r2}% per period (the repayment stays ${R}). Using the recurrence relation "
    "B(n+1) = B(n)(1 + r) - R throughout, find the balance after {ntotal} periods in total.",
    "For the first {n1} periods, a ${p} reducing-balance loan is charged {r1}% interest per "
    "period, with a fixed ${R} repaid every period, following B(n+1) = B(n)(1 + r) - R. The "
    "rate then changes to {r2}% per period for the remaining periods (same ${R} repayment). "
    "Find the balance after {ntotal} periods in total.",
    "A loan of ${p} follows the recurrence B(n+1) = B(n)(1 + r) - R, with a ${R} repayment "
    "every period throughout. The interest rate is {r1}% per period for the first {n1} "
    "periods, then {r2}% per period for the rest. Find the balance after {ntotal} periods.",
)


def _loan_rate_change(rng):
    principal = rng.randint(8, 60) * 1000
    rate1_pct, rate2_pct = rng.sample(list(_LOAN_RATES), 2)  # always two DISTINCT rates
    r1, r2 = rate1_pct / 100, rate2_pct / 100
    n1 = rng.randint(2, 4)
    n2 = rng.randint(2, 4)
    ntotal = n1 + n2
    interest0 = Fraction(principal) * r1
    R = _ceil_frac(interest0) * rng.choice(_LOAN_R_MULT)

    phase1_balance = _iterate_balance(principal, r1, R, n1)
    final_balance = _iterate_balance(phase1_balance, r2, R, n2)
    # Independent route: the payment-stream formula run separately over each
    # phase, carrying phase 1's closing balance into phase 2 as its own
    # starting principal -- a genuinely different code path from the
    # single forward loop above, so a boundary/off-by-one bug in either is
    # caught.
    phase1_check = _stream_balance(principal, r1, R, n1)
    check_equivalent(phase1_balance, phase1_check, "rate-change loan: phase-1 loop/stream routes disagree")
    final_check = _stream_balance(phase1_balance, r2, R, n2)
    check_equivalent(final_balance, final_check, "rate-change loan: phase-2 loop/stream routes disagree")

    template = rng.choice(_LOAN_RATE_CHANGE_TEMPLATES)
    text = template.format(
        p=principal,
        r1=_fmt_pct(rate1_pct),
        r2=_fmt_pct(rate2_pct),
        R=(int(R) if R.denominator == 1 else float(R)),
        n1=n1,
        ntotal=ntotal,
    )
    text = text.rstrip() + _TO_THE_CENT

    answer_text = _money(final_balance)
    candidates = [
        _money(_iterate_balance(principal, r1, R, ntotal)),  # never applied the rate change
        _money(_iterate_balance(principal, r2, R, ntotal)),  # applied the new rate from period 1
        _money(_iterate_balance(principal, r2, R, n2)),  # reset to principal at the change,
        # forgetting phase 1's balance
    ]
    candidates = _drop_collisions_money(round_half_up(final_balance, 2), candidates)
    check_distinct(
        round_half_up(final_balance, 2),
        *[round_half_up(_parse_money(c), 2) for c in candidates],
        msg="loan rate-change distractor equals the answer",
    )
    distractors = pick_distractors(answer_text, candidates, n=3, tolerance=_TOL2)
    return dict(
        text=text,
        answer=float(round_half_up(final_balance, 2)),
        answer_text=answer_text,
        answer_latex=f"\\${answer_text}",
        distractors=distractors,
        working=[
            f"After {n1} periods at {_fmt_pct(rate1_pct)}%, the balance is ${_money(phase1_balance)}. "
            f"Carrying that forward as the starting balance for {n2} more periods at "
            f"{_fmt_pct(rate2_pct)}% gives ${answer_text}."
        ],
    )


_LOAN_PRESETS = {
    "easy": {"variant": "forward"},
    "medium": {"variant": "lump_sum"},
    "hard": {"variant": "find_n"},
    "exam": {"variant": "rate_change"},
}


@generator("p16c_loan_recurrence_balance", level=13, presets=_LOAN_PRESETS)
def loan_recurrence_balance(rng, variant="forward"):
    if variant == "forward":
        return _loan_forward(rng, 2, 3, lump=False)
    if variant == "lump_sum":
        return _loan_forward(rng, 4, 6, lump=True)
    if variant == "find_n":
        return _loan_find_n(rng)
    if variant == "rate_change":
        return _loan_rate_change(rng)
    raise ValueError(f"unknown variant: {variant!r}")


# =============================================================================
# Skill 2: p16c_compound_effect (ACMEM173)
# =============================================================================

_CE_RATES = (4, 5, 6, 8, 10)
_CE_FREQ_NAME = {1: "annually", 2: "semi-annually", 4: "quarterly", 12: "monthly"}

_CE_DIFF_TEMPLATES = (
    "An investment of ${p} earns a nominal {r}% p.a. interest rate. Find the difference "
    "between its future value after {t} years when compounded {fa} and its future value "
    "after {t} years when compounded {fb}, to the nearest cent.",
    "A loan of ${p} at a nominal annual rate of {r}% runs for {t} years. By how much (to the "
    "nearest cent) does its future value change if the compounding is done {fb} instead of {fa}?",
    "Compare an investment of ${p} over {t} years: compounded {fa}, and separately compounded "
    "{fb}, both at a nominal {r}% p.a. What is the difference between the two resulting future "
    "values, to the nearest cent?",
)

_CE_RATE_TEMPLATES = (
    "An investment of ${p} is compounded {fa} over {t} years. Find the difference (to the "
    "nearest cent) between its future value at a nominal {ra}% p.a. and its future value at a "
    "nominal {rb}% p.a.",
    "A loan of ${p} runs for {t} years, compounded {fa}. By how much (to the nearest cent) is "
    "the future value at {rb}% p.a. greater than the future value at {ra}% p.a.?",
    "Compare two nominal annual rates, {ra}% and {rb}%, both compounded {fa} on a principal of "
    "${p} over {t} years. What is the difference between the two future values, to the nearest "
    "cent?",
)

_CE_PCT_TEMPLATES = (
    "An investment of ${p} over {t} years is compounded {fa} at a nominal {ra}% p.a., and "
    "separately compounded {fb} at a nominal {rb}% p.a. By what percentage (to 1 decimal "
    "place) is the larger future value greater than the smaller?",
    "Two ways of growing ${p} over {t} years are compared: compounded {fa} at {ra}% p.a., and "
    "compounded {fb} at {rb}% p.a. Express the larger resulting future value as a percentage "
    "increase over the smaller, to 1 decimal place.",
)


def _fv_pow(principal, r: Fraction, freq: int, years: int) -> Fraction:
    return Fraction(principal) * (1 + r / freq) ** (freq * years)


def _fv_loop(principal, r: Fraction, freq: int, years: int) -> Fraction:
    """Independent route: an explicit period-by-period loop instead of the
    closed-form power, so an off-by-one in either the exponent or the loop
    bound is caught."""
    b = Fraction(principal)
    for _ in range(freq * years):
        b = b * (1 + r / freq)
    return b


def _ce_freq_diff(rng, freq_a: int, freq_b: int, t_lo: int, t_hi: int):
    principal = rng.randint(5, 50) * 1000
    r = Fraction(rng.choice(_CE_RATES), 100)
    t = rng.randint(t_lo, t_hi)
    fv_a = _fv_pow(principal, r, freq_a, t)
    fv_b = _fv_pow(principal, r, freq_b, t)
    check_equivalent(fv_a, _fv_loop(principal, r, freq_a, t), "future value: pow and loop routes disagree (a)")
    check_equivalent(fv_b, _fv_loop(principal, r, freq_b, t), "future value: pow and loop routes disagree (b)")
    diff = fv_b - fv_a  # freq_b has more periods per year -> always the larger future value for r > 0
    check(diff > 0, "the more frequent compounding did not produce the larger future value")

    template = rng.choice(_CE_DIFF_TEMPLATES)
    text = template.format(
        p=principal,
        r=r * 100 if r.denominator == 1 else float(r * 100),
        t=t,
        fa=_CE_FREQ_NAME[freq_a],
        fb=_CE_FREQ_NAME[freq_b],
    )
    answer_text = _money(diff)
    same_freq_diff = Fraction(0)  # "used the same frequency both sides" -> no effect at all
    off_by_one_year = _fv_pow(principal, r, freq_b, max(t - 1, 1)) - fv_a  # one fewer year on the freq_b side
    candidates = [
        _money(same_freq_diff),
        _money_signed(fv_a - fv_b),  # sign flipped
        _money(off_by_one_year),
    ]
    candidates = _drop_collisions_money(round_half_up(diff, 2), candidates)
    check_distinct(
        round_half_up(diff, 2),
        *[round_half_up(_parse_money(c), 2) for c in candidates],
        msg="compounding-effect distractor equals the answer",
    )
    distractors = pick_distractors(answer_text, candidates, n=3, tolerance=_TOL2)
    return dict(
        text=text,
        answer=float(round_half_up(diff, 2)),
        answer_text=answer_text,
        answer_latex=f"\\${answer_text}",
        distractors=distractors,
        working=[
            f"FV compounded {_CE_FREQ_NAME[freq_b]} minus FV compounded {_CE_FREQ_NAME[freq_a]} "
            f"= ${_money(fv_b)} - ${_money(fv_a)} = ${answer_text}."
        ],
    )


def _ce_rate_diff(rng):
    principal = rng.randint(5, 50) * 1000
    freq = 12
    t = rng.randint(2, 4)
    r1, r2 = sorted(rng.sample(_CE_RATES, 2))
    ra, rb = Fraction(r1, 100), Fraction(r2, 100)
    fv_a = _fv_pow(principal, ra, freq, t)
    fv_b = _fv_pow(principal, rb, freq, t)
    check_equivalent(fv_a, _fv_loop(principal, ra, freq, t), "future value: pow and loop routes disagree (rate a)")
    check_equivalent(fv_b, _fv_loop(principal, rb, freq, t), "future value: pow and loop routes disagree (rate b)")
    diff = fv_b - fv_a
    check(diff > 0, "the higher rate did not produce the larger future value")

    template = rng.choice(_CE_RATE_TEMPLATES)
    text = template.format(p=principal, t=t, fa=_CE_FREQ_NAME[freq], ra=r1, rb=r2)
    answer_text = _money(diff)
    candidates = [
        _money(Fraction(0)),  # "the rate change has no effect" (used the same rate both sides)
        _money_signed(fv_a - fv_b),  # sign flipped
        _money(Fraction(principal) * (rb - ra) * t),  # simple-interest-style linear misconception
    ]
    candidates = _drop_collisions_money(round_half_up(diff, 2), candidates)
    check_distinct(
        round_half_up(diff, 2),
        *[round_half_up(_parse_money(c), 2) for c in candidates],
        msg="compounding-effect (rate) distractor equals the answer",
    )
    distractors = pick_distractors(answer_text, candidates, n=3, tolerance=_TOL2)
    return dict(
        text=text,
        answer=float(round_half_up(diff, 2)),
        answer_text=answer_text,
        answer_latex=f"\\${answer_text}",
        distractors=distractors,
        working=[
            f"FV at {r2}% minus FV at {r1}% (both compounded {_CE_FREQ_NAME[freq]}) "
            f"= ${_money(fv_b)} - ${_money(fv_a)} = ${answer_text}."
        ],
    )


def _ce_percent_effect(rng):
    principal = rng.randint(5, 50) * 1000
    t = rng.randint(2, 4)
    freq_a, freq_b = rng.sample((1, 2, 4, 12), 2)
    r1, r2 = rng.sample(_CE_RATES, 2)
    ra, rb = Fraction(r1, 100), Fraction(r2, 100)
    fv_a = _fv_pow(principal, ra, freq_a, t)
    fv_b = _fv_pow(principal, rb, freq_b, t)
    check_equivalent(fv_a, _fv_loop(principal, ra, freq_a, t), "future value: pow and loop routes disagree (pct a)")
    check_equivalent(fv_b, _fv_loop(principal, rb, freq_b, t), "future value: pow and loop routes disagree (pct b)")
    larger, smaller = (fv_b, fv_a) if fv_b > fv_a else (fv_a, fv_b)
    pct = (larger - smaller) / smaller * 100
    pct_rounded = round_half_up(pct, 1)

    template = rng.choice(_CE_PCT_TEMPLATES)
    text = template.format(p=principal, t=t, fa=_CE_FREQ_NAME[freq_a], fb=_CE_FREQ_NAME[freq_b], ra=r1, rb=r2)
    answer_text = stated_precision_string(pct_rounded, 1)
    dollar_diff_as_pct_of_principal = round_half_up((larger - smaller) / principal * 100, 1)
    pct_of_larger = round_half_up((larger - smaller) / larger * 100, 1)
    candidates_v = [dollar_diff_as_pct_of_principal, pct_of_larger]
    candidates = [stated_precision_string(v, 1) for v in candidates_v if v != pct_rounded]
    candidates.append(stated_precision_string(round_half_up(pct_rounded * 2, 1), 1))
    check_distinct(
        pct_rounded, *[Fraction(c) for c in candidates], msg="compounding-effect (percent) distractor equals the answer"
    )
    distractors = pick_distractors(answer_text, candidates, n=3, tolerance=stated_precision_tolerance(1))
    return dict(
        text=text,
        answer=float(pct_rounded),
        answer_text=answer_text,
        answer_latex=f"{answer_text}\\%",
        distractors=distractors,
        working=[
            f"The larger future value (${_money(larger)}) exceeds the smaller (${_money(smaller)}) "
            f"by {answer_text}% of the smaller."
        ],
    )


_CE_PRESETS = {
    "easy": {"variant": "annual_vs_semi"},
    "medium": {"variant": "quarterly_vs_monthly"},
    "hard": {"variant": "rate_effect"},
    "exam": {"variant": "percent_effect"},
}


@generator("p16c_compound_effect", level=13, presets=_CE_PRESETS)
def compound_effect(rng, variant="annual_vs_semi"):
    if variant == "annual_vs_semi":
        return _ce_freq_diff(rng, 1, 2, 2, 3)
    if variant == "quarterly_vs_monthly":
        return _ce_freq_diff(rng, 4, 12, 3, 5)
    if variant == "rate_effect":
        return _ce_rate_diff(rng)
    if variant == "percent_effect":
        return _ce_percent_effect(rng)
    raise ValueError(f"unknown variant: {variant!r}")


# =============================================================================
# Skill 3: p16c_worker_wage_table (ACMGM009)
# =============================================================================


def _draw_workers(rng, k: int, with_ot: bool, uniform_mult: bool):
    names = rng.sample(NAMES, k)
    rows = []
    for name in names:
        rate = rng.randint(18, 40)
        hours = rng.randint(15, 38)
        ot = 0
        mult = Fraction(3, 2)
        if with_ot and rng.random() < 0.7:
            ot = rng.randint(2, 8)
            mult = Fraction(3, 2) if uniform_mult else rng.choice((Fraction(3, 2), Fraction(2, 1)))
        rows.append({"name": name, "rate": rate, "hours": hours, "ot": ot, "mult": mult})
    return rows


def _wage(row) -> Fraction:
    return Fraction(row["rate"]) * row["hours"] + Fraction(row["rate"]) * row["mult"] * row["ot"]


def _wage_alt_route(row) -> Fraction:
    """Independent route: total hours paid at the ordinary rate PLUS the
    extra (mult - 1) loading on the overtime hours only, instead of a single
    rate*hours + rate*mult*overtime expression -- catches a slip in either
    formula."""
    total_hours = row["hours"] + row["ot"]
    base = Fraction(row["rate"]) * total_hours
    loading = Fraction(row["rate"]) * (row["mult"] - 1) * row["ot"]
    return base + loading


def _table_text(rows, with_ot: bool, per_row_mult: bool) -> str:
    lines = []
    for r in rows:
        if not with_ot:
            lines.append(f"{r['name']}: ${r['rate']}/h, {r['hours']} ordinary hours")
        elif per_row_mult:
            mult_txt = "1.5x" if r["mult"] == Fraction(3, 2) else "2x"
            ot_clause = f"{r['ot']} overtime hours at {mult_txt}" if r["ot"] > 0 else "0 overtime hours"
            lines.append(f"{r['name']}: ${r['rate']}/h, {r['hours']} ordinary hours, {ot_clause}")
        else:
            lines.append(f"{r['name']}: ${r['rate']}/h, {r['hours']} ordinary hours, {r['ot']} overtime hours")
    return "; ".join(lines)


_WAGE_TEMPLATES = (
    "A fast-food store's wage sheet lists each worker's hourly rate and hours this week: {table}."
    "{ot_note} What is {who}'s total wage for the week?",
    "The weekly wage spreadsheet for a fast-food store shows: {table}.{ot_note} Find {who}'s gross pay for the week.",
    "This week's wage sheet shows: {table}.{ot_note} From this sheet, calculate the total amount {who} earned.",
)

_WAGE_OT_NOTE = " Overtime hours are paid at 1.5 times the hourly rate."
_WAGE_DIFF_TEMPLATES = (
    "A fast-food store's wage sheet lists each worker's hourly rate and hours this week: {table}. "
    "Overtime hours are paid at the stated multiple of the hourly rate. How much MORE did {who_a} "
    "earn than {who_b} this week?",
    "This week's wage sheet shows: {table}. Overtime is paid at the stated multiple of the "
    "hourly rate. Find the difference between {who_a}'s wage and {who_b}'s wage.",
)


def _wage_single(rng, k: int, with_ot: bool, uniform_mult: bool, force_ot_for_answer: bool):
    for _ in range(50):
        rows = _draw_workers(rng, k, with_ot, uniform_mult)
        idx = rng.randrange(k)
        if not force_ot_for_answer or rows[idx]["ot"] > 0:
            break
    who = rows[idx]["name"]
    wage = _wage(rows[idx])
    check_equivalent(wage, _wage_alt_route(rows[idx]), "wage: two formulas disagree")

    per_row_mult = with_ot and not uniform_mult
    ot_note = "" if not with_ot else ("" if per_row_mult else _WAGE_OT_NOTE)
    table = _table_text(rows, with_ot, per_row_mult)
    template = rng.choice(_WAGE_TEMPLATES)
    text = template.format(table=table, ot_note=ot_note, who=who)

    answer_text = _money(wage)
    other = rows[(idx + 1) % k]
    own_rate, own_hours, own_ot, own_mult = (
        Fraction(rows[idx]["rate"]),
        rows[idx]["hours"],
        rows[idx]["ot"],
        rows[idx]["mult"],
    )
    candidates = [
        _money(_wage(other)),  # read the wrong row entirely
        _money(own_rate + own_hours),  # adds rate and hours instead of multiplying
    ]
    if other["rate"] != rows[idx]["rate"]:
        # uses another worker's rate with this worker's own hours
        candidates.append(
            _money(Fraction(other["rate"]) * own_hours + Fraction(other["rate"]) * other["mult"] * own_ot)
        )
    if own_ot > 0:
        candidates.append(_money(own_rate * own_hours))  # forgot overtime entirely
        candidates.append(_money(own_rate * (own_hours + own_ot)))  # overtime paid at the ordinary rate
    if per_row_mult and own_ot > 0:
        wrong_mult = Fraction(2, 1) if own_mult == Fraction(3, 2) else Fraction(3, 2)
        candidates.append(_money(own_rate * own_hours + own_rate * wrong_mult * own_ot))  # wrong multiplier
    candidates = _drop_collisions_money(round_half_up(wage, 2), candidates)
    check_distinct(
        round_half_up(wage, 2),
        *[round_half_up(_parse_money(c), 2) for c in candidates],
        msg="wage-table distractor equals the answer",
    )
    distractors = pick_distractors(answer_text, candidates, n=3, tolerance=_TOL2)
    return dict(
        text=text,
        answer=float(round_half_up(wage, 2)),
        answer_text=answer_text,
        answer_latex=f"\\${answer_text}",
        distractors=distractors,
        working=[f"Reading {who}'s own row and applying the stated rates: total wage = ${answer_text}."],
    )


def _wage_diff(rng):
    k = 4
    for _ in range(50):
        rows = _draw_workers(rng, k, with_ot=True, uniform_mult=False)
        ot_rows = [i for i, r in enumerate(rows) if r["ot"] > 0]
        if ot_rows:
            break
    a_idx = rng.choice(ot_rows)
    b_idx = rng.choice([i for i in range(k) if i != a_idx])
    wage_a = _wage(rows[a_idx])
    wage_b = _wage(rows[b_idx])
    check_equivalent(wage_a, _wage_alt_route(rows[a_idx]), "wage: two formulas disagree (a)")
    check_equivalent(wage_b, _wage_alt_route(rows[b_idx]), "wage: two formulas disagree (b)")
    if wage_b > wage_a:
        a_idx, b_idx = b_idx, a_idx
        wage_a, wage_b = wage_b, wage_a
    who_a, who_b = rows[a_idx]["name"], rows[b_idx]["name"]

    table = _table_text(rows, with_ot=True, per_row_mult=True)
    template = rng.choice(_WAGE_DIFF_TEMPLATES)
    text = template.format(table=table, who_a=who_a, who_b=who_b)

    diff = wage_a - wage_b
    answer_text = _money(abs(diff))
    ra, ha, oa, ma = (Fraction(rows[a_idx]["rate"]), rows[a_idx]["hours"], rows[a_idx]["ot"], rows[a_idx]["mult"])
    rb, hb, ob, mb = (Fraction(rows[b_idx]["rate"]), rows[b_idx]["hours"], rows[b_idx]["ot"], rows[b_idx]["mult"])
    candidates = [
        _money(wage_a + wage_b),  # summed instead of subtracted
        _money(abs(ra * ha - rb * hb)),  # ordinary pay only, both sides' overtime forgotten
        _money(abs(ra * hb - rb * ha)),  # rates swapped between the two workers
    ]
    if oa > 0:
        candidates.append(_money(abs(ra * ha - wage_b)))  # forgot a's overtime only
    if ob > 0:
        candidates.append(_money(abs(wage_a - rb * hb)))  # forgot b's overtime only
    candidates = _drop_collisions_money(round_half_up(abs(diff), 2), candidates)
    check_distinct(
        round_half_up(abs(diff), 2),
        *[round_half_up(_parse_money(c), 2) for c in candidates],
        msg="wage-difference distractor equals the answer",
    )
    distractors = pick_distractors(answer_text, candidates, n=3, tolerance=_TOL2)
    return dict(
        text=text,
        answer=float(round_half_up(abs(diff), 2)),
        answer_text=answer_text,
        answer_latex=f"\\${answer_text}",
        distractors=distractors,
        working=[f"{who_a}'s wage minus {who_b}'s wage: ${_money(wage_a)} - ${_money(wage_b)} = ${answer_text}."],
    )


_WAGE_PRESETS = {
    "easy": {"variant": "no_overtime"},
    "medium": {"variant": "uniform_overtime"},
    "hard": {"variant": "per_row_multiplier"},
    "exam": {"variant": "cross_row_diff"},
}


@generator("p16c_worker_wage_table", level=13, presets=_WAGE_PRESETS)
def worker_wage_table(rng, variant="no_overtime"):
    if variant == "no_overtime":
        return _wage_single(rng, 3, with_ot=False, uniform_mult=True, force_ot_for_answer=False)
    if variant == "uniform_overtime":
        return _wage_single(rng, 4, with_ot=True, uniform_mult=True, force_ot_for_answer=False)
    if variant == "per_row_multiplier":
        return _wage_single(rng, 4, with_ot=True, uniform_mult=False, force_ot_for_answer=True)
    if variant == "cross_row_diff":
        return _wage_diff(rng)
    raise ValueError(f"unknown variant: {variant!r}")


# =============================================================================
# Skill 4: p16c_gmt_dateline (ACMEM164) -- mcq, options are day+time strings
# =============================================================================

_DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")

# A route only genuinely crosses the International Date Line when one end is in Asia/Oceania (a
# positive GMT offset) and the other is in the Americas or the eastern Pacific (a negative GMT
# offset) -- flying between two places on the SAME side (e.g. Accra to Sydney, or Delhi to Adelaide)
# never crosses it, whatever the two offsets happen to be. No daylight saving anywhere here -- the
# kernel is GMT-referenced arithmetic across the Date Line, not zone/DST lookup (that is what the
# two shipping time-zone skills already ask).
#
# Each tuple is (west_place, west_offset, east_place, east_offset, hours).
_CROSSING_ROUTES = (
    ("Auckland", 12, "Los Angeles", -8, 12),
    ("Auckland", 12, "Honolulu", -10, 9),
    ("Auckland", 12, "Vancouver", -8, 13),
    ("Auckland", 12, "Buenos Aires", -3, 11),
    ("Auckland", 12, "Anchorage", -9, 10),
    ("Wellington", 12, "Los Angeles", -8, 12),
    ("Wellington", 12, "Honolulu", -10, 9),
    ("Wellington", 12, "Rio de Janeiro", -3, 17),
    ("Suva", 12, "Los Angeles", -8, 11),
    ("Suva", 12, "Denver", -7, 13),
    ("Suva", 12, "Honolulu", -10, 6),
    ("Suva", 12, "Rio de Janeiro", -3, 13),
    ("Apia", 13, "Honolulu", -10, 6),
    ("Apia", 13, "Los Angeles", -8, 9),
    ("Sydney", 10, "Los Angeles", -8, 14),
    ("Sydney", 10, "Honolulu", -10, 10),
    ("Sydney", 10, "Vancouver", -8, 15),
    ("Sydney", 10, "Buenos Aires", -3, 12),
    ("Sydney", 10, "Denver", -7, 15),
    ("Tokyo", 9, "Los Angeles", -8, 9),
    ("Tokyo", 9, "New York", -5, 13),
    ("Tokyo", 9, "Vancouver", -8, 9),
    ("Tokyo", 9, "Anchorage", -9, 7),
    ("Tokyo", 9, "Toronto", -5, 13),
    ("Singapore", 8, "Los Angeles", -8, 17),
    ("Singapore", 8, "New York", -5, 18),
    ("Singapore", 8, "Toronto", -5, 19),
    ("Perth", 8, "Honolulu", -10, 11),
    ("Perth", 8, "Los Angeles", -8, 17),
    ("Perth", 8, "Vancouver", -8, 16),
)
_CROSSING_ROUTES_HALF = (
    ("Adelaide", Fraction(19, 2), "St John's", Fraction(-7, 2), 20),
    ("Adelaide", Fraction(19, 2), "Marquesas Islands", Fraction(-19, 2), 10),
    ("Darwin", Fraction(19, 2), "St John's", Fraction(-7, 2), 19),
    ("Darwin", Fraction(19, 2), "Marquesas Islands", Fraction(-19, 2), 11),
)


def _pick_crossing_route(rng, table):
    """A random (place_from, off_from, place_to, off_to, hours) from `table`,
    with the departure/arrival direction chosen at random too (so both signs
    of the offset difference occur across draws)."""
    west, west_off, east, east_off, hours = rng.choice(table)
    if rng.random() < 0.5:
        return west, west_off, east, east_off, hours
    return east, east_off, west, west_off, hours


def _fmt_offset(offset) -> str:
    """A GMT offset as a learner would read it: "GMT+8", "GMT-3", "GMT+9:30",
    "GMT-3:30" -- never a bare Fraction repr like "19/2"."""
    off = Fraction(offset)
    sign = "+" if off >= 0 else "-"
    off = abs(off)
    whole = int(off)
    minutes = int((off - whole) * 60)
    if minutes == 0:
        return f"GMT{sign}{whole}"
    return f"GMT{sign}{whole}:{minutes:02d}"


def _fmt_time(total_minutes: int) -> str:
    day_idx = (total_minutes // 1440) % 7
    minute_of_day = total_minutes % 1440
    h, m = divmod(minute_of_day, 60)
    return f"{_DAYS[day_idx]} {h:02d}:{m:02d}"


def _to_minutes(day_idx: int, hour: int, minute: int) -> int:
    return day_idx * 1440 + hour * 60 + minute


def _offset_minutes(offset) -> int:
    return int(Fraction(offset) * 60)


def _gmt_leg(depart_minutes: int, offset_from, duration_minutes: int, offset_to) -> int:
    depart_utc = depart_minutes - _offset_minutes(offset_from)
    arrive_utc = depart_utc + duration_minutes
    return arrive_utc + _offset_minutes(offset_to)


def _mcq(text: str, answer_text: str, candidates: list, *, working) -> dict:
    distractors = pick_distractors(answer_text, candidates, n=3)
    return dict(
        text=text,
        answer=answer_text,
        answer_text=answer_text,
        answer_latex=f"plain:{answer_text}",
        distractors=distractors,
        working=working,
    )


_GMT_INSTRUCTION = (
    "Give your answer as the day and the 24-hour local time at the destination (for example, Tuesday 14:00)."
)


def _gmt_single_leg(rng, half_hour: bool, force_day_back: bool):
    table = _CROSSING_ROUTES_HALF if half_hour else _CROSSING_ROUTES
    for _ in range(400):
        place_from, off_from, place_to, off_to, duration_h = _pick_crossing_route(rng, table)
        day_idx = rng.randrange(7)
        hour = rng.randint(0, 23)
        minute = rng.choice((0, 15, 30, 45)) if half_hour else 0
        depart_minutes = _to_minutes(day_idx, hour, minute)
        arrive_minutes = _gmt_leg(depart_minutes, off_from, duration_h * 60, off_to)
        depart_local_day = depart_minutes // 1440
        arrive_local_day = arrive_minutes // 1440
        goes_back = arrive_local_day < depart_local_day
        if force_day_back and not goes_back:
            continue
        if not force_day_back and goes_back:
            continue
        break
    else:
        raise ValueError("could not draw a GMT/IDL scenario matching the requested day-direction")

    answer_text = _fmt_time(arrive_minutes)
    text = (
        f"A flight departs {place_from} ({_fmt_offset(off_from)}) at "
        f"{hour:02d}:{minute:02d} on {_DAYS[day_idx]}, and flies for {duration_h} hours to "
        f"{place_to} ({_fmt_offset(off_to)}), crossing the International Date "
        f"Line. Using GMT as the reference, find {place_to}'s local day and time of arrival. "
        f"{_GMT_INSTRUCTION}"
    )
    same_zone_wrong = _fmt_time(depart_minutes + duration_h * 60)  # ignored both zones/date line entirely
    swapped_zone_wrong = _fmt_time(_gmt_leg(depart_minutes, off_to, duration_h * 60, off_from))  # zones swapped
    off_by_one_day = _fmt_time(arrive_minutes + 1440)
    off_by_one_day_back = _fmt_time(arrive_minutes - 1440)
    candidates = [same_zone_wrong, swapped_zone_wrong, off_by_one_day, off_by_one_day_back]
    return _mcq(
        text,
        answer_text,
        candidates,
        working=[
            f"Departure in GMT: {_fmt_time(depart_minutes - _offset_minutes(off_from))}. "
            f"Add the {duration_h}h flight, then apply {place_to}'s GMT offset: {answer_text}."
        ],
    )


def _gmt_two_leg(rng):
    for _ in range(400):
        place_a, off_a, place_b, off_b, dur1 = _pick_crossing_route(rng, _CROSSING_ROUTES)
        # Leg 2 continues from place_b's OTHER crossing route (a different
        # table entry that still names place_b), so its duration is also a
        # real-world approximate figure, not an arbitrary draw. Leg 2's own
        # text never claims a Date Line crossing, so it need not cross one,
        # but every entry in this table does in fact cross -- it is simply
        # not asserted for leg 2.
        leg2_candidates = []
        for a, oa, b, ob, hours in _CROSSING_ROUTES:
            if a == place_b and b != place_a:
                leg2_candidates.append((b, ob, hours))
            elif b == place_b and a != place_a:
                leg2_candidates.append((a, oa, hours))
        if not leg2_candidates:
            continue
        place_c, off_c, dur2 = rng.choice(leg2_candidates)
        day_idx = rng.randrange(7)
        hour = rng.randint(0, 23)
        layover = rng.randint(1, 5)
        depart_minutes = _to_minutes(day_idx, hour, 0)
        arrive_b = _gmt_leg(depart_minutes, off_a, dur1 * 60, off_b)
        depart_b_local = arrive_b + layover * 60
        depart_b_utc = depart_b_local - _offset_minutes(off_b)
        arrive_c_utc = depart_b_utc + dur2 * 60
        arrive_c = arrive_c_utc + _offset_minutes(off_c)
        if arrive_c // 1440 != depart_minutes // 1440:
            break
    else:
        raise ValueError("could not draw a two-leg GMT scenario with a calendar-day change")

    answer_text = _fmt_time(arrive_c)
    text = (
        f"A traveller departs {place_a} ({_fmt_offset(off_a)}) at {hour:02d}:00 on "
        f"{_DAYS[day_idx]} and flies {dur1} hours to {place_b} ({_fmt_offset(off_b)}), "
        f"crossing the International Date Line. After a {layover}-hour layover in {place_b}'s local "
        f"time, a connecting flight takes {dur2} hours to {place_c} ({_fmt_offset(off_c)}). "
        f"Using GMT as the reference throughout, find {place_c}'s local day and time of "
        f"arrival. {_GMT_INSTRUCTION}"
    )
    ignore_layover = _fmt_time(_gmt_leg(depart_minutes, off_a, (dur1 + dur2) * 60, off_c))
    swap_leg2_zone = _fmt_time(_gmt_leg(arrive_b, off_a, layover * 60 + dur2 * 60, off_c))
    off_by_one_day = _fmt_time(arrive_c + 1440)
    off_by_one_day_back = _fmt_time(arrive_c - 1440)
    candidates = [ignore_layover, swap_leg2_zone, off_by_one_day, off_by_one_day_back]
    return _mcq(
        text,
        answer_text,
        candidates,
        working=[
            f"Leg 1 arrival in {place_b}: {_fmt_time(arrive_b)}. After the {layover}h "
            f"layover, leg 2 arrives in {place_c} at {answer_text}."
        ],
    )


_GMT_PRESETS = {
    "easy": {"variant": "single_forward"},
    "medium": {"variant": "single_half_hour"},
    "hard": {"variant": "single_day_back"},
    "exam": {"variant": "two_leg"},
}


@generator("p16c_gmt_dateline", level=13, presets=_GMT_PRESETS)
def gmt_dateline(rng, variant="single_forward"):
    if variant == "single_forward":
        return _gmt_single_leg(rng, half_hour=False, force_day_back=False)
    if variant == "single_half_hour":
        return _gmt_single_leg(rng, half_hour=True, force_day_back=False)
    if variant == "single_day_back":
        return _gmt_single_leg(rng, half_hour=False, force_day_back=True)
    if variant == "two_leg":
        return _gmt_two_leg(rng)
    raise ValueError(f"unknown variant: {variant!r}")
