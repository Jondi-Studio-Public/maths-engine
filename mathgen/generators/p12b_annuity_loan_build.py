"""Reducing-balance loan repayment and annuity/perpetuity present value.

Two skills, one shared vocabulary (periodic rate ``i``, number of periods
``n``, periodic payment ``A``), two separate generators -- the underlying
formula differs so they are never forced into one topic:

- ``p12b_reducing_balance_loan`` --
  ``senior-with-the-aid-of-a-financial-calculator-or`` (ACMGM098): the
  standard amortisation formula ``A = P*i / (1 - (1+i)**-n)``, the periodic
  repayment that exactly reduces ``P`` to zero over ``n`` periods at
  periodic rate ``i``.
- ``p12b_annuity_perpetuity`` --
  ``senior-with-the-aid-of-a-financial-calculator-or-acmgm100`` (ACMGM100):
  the present value of an ordinary annuity, ``PV = A * (1 - (1+i)**-n) / i``,
  with the perpetuity ``PV = A / i`` as the exact ``n -> infinity`` limit.

**Boundary:** no earlier module covers annuities, reducing-balance
amortisation or perpetuities. ``compound-interest`` computes ``A = P(1+r)^n`` from a
stated ``P`` (forward growth, no repayments); ``flat-loan-repayment``
computes a FLAT-interest repayment (interest on the ORIGINAL principal every
period, never reducing) -- neither is a reducing-balance amortisation or a
present-value annuity/perpetuity. ``senior-investigate-the-effect-of-the-
interest-rate-and`` computes the NUMBER OF PERIODS
for a FLAT-interest loan given a stated repayment; this module's ACMGM098
skill computes the REPAYMENT AMOUNT (never the period count) for a
REDUCING-balance loan (compound periodic rate, not flat) -- both the unknown
and the interest model differ.

**Exact arithmetic throughout, rounded once.** Every intermediate value is a
``fractions.Fraction``; the final ``answer_text`` is rounded to 2 d.p. (cents)
via ``round_half_up`` + ``stated_precision_string`` only at the point of
display, never re-rounded. ``(1+i)**-n`` is Python's own exact ``Fraction``
power (never a float truncated early).

**The perpetuity/finite-annuity tie boundary.** Perpetuity
is ALWAYS its own separate tier (easy, ``p12b_annuity_perpetuity``) with no
``n`` ever offered as a literal input; every finite-annuity tier (medium,
hard, exam) always carries an explicit ``n`` and is constructed so its PV is
never within a cent of the perpetuity PV at the same rate -- see
``_medium``'s in-construction ``check`` for the enforced minimum gap, proven
by construction on every seed rather than sampled after the fact.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import (
    check,
    check_equivalent,
    check_pairwise_distinct,
    generator,
    latex_answer,
    numeric_answer,
    phrase,
    pick_distractors,
    round_half_up,
    stated_precision_string,
    symbolically_equal,
)

_MAX_ATTEMPTS = 400

# 3.0% .. 12.0% p.a., in tenths of a percent, deliberately including
# non-round tenths so the periodic rate is not always a "nice" fraction.
_RATE_TENTHS = (30, 40, 45, 48, 50, 55, 60, 66, 70, 72, 80, 90, 96, 100, 108, 120)
_TERM_YEARS = (10, 15, 20, 25, 30)
_ACCUM_YEARS = (5, 10, 15, 20)


def _money(value: Fraction) -> Fraction:
    return round_half_up(value, 2)


def _money_text(value: Fraction) -> str:
    return stated_precision_string(_money(value), 2)


def _rate_text(rate_tenths: int) -> str:
    """Display a rate stored in tenths of a percent as a fixed 1-d.p.
    percentage string ("4.8", "10.0") -- never the raw `Fraction` (which
    would render an ugly, mathematically-correct-but-unreadable "24/5")."""
    return stated_precision_string(Fraction(rate_tenths, 10), 1)


def _pairwise_distinct(*values) -> bool:
    """The VERDICT form of pairwise distinctness, for deciding whether to resample a draw -- `symbolically_equal`
    per pair (n choose 2 comparisons), never a caught `VerificationError` used as control flow. Once a draw is
    accepted, `check_pairwise_distinct(*values, msg=...)` is still called unconditionally on it."""
    for idx, left in enumerate(values):
        for right in values[idx + 1 :]:
            if symbolically_equal(left, right):
                return False
    return True


# =============================================================================
# ACMGM098 -- p12b_reducing_balance_loan
# =============================================================================


def _periodic_rate(rate_tenths: int, periods_per_year: int) -> Fraction:
    return Fraction(rate_tenths, 10 * 100 * periods_per_year)


def _repayment(principal, i: Fraction, n: int) -> Fraction:
    """A = P*i / (1 - (1+i)**-n): the periodic payment that exactly
    amortises `principal` over `n` periods at periodic rate `i`."""
    growth = (1 + i) ** n
    return Fraction(principal) * i * growth / (growth - 1)


def _balance_after(principal, i: Fraction, repayment, k: int) -> Fraction:
    """Closed form: balance after k periods = FV(P) - FV(the k payments
    made) = P*(1+i)**k - A*((1+i)**k - 1)/i."""
    growth = (1 + i) ** k
    return Fraction(principal) * growth - Fraction(repayment) * (growth - 1) / i


def _max_principal(repayment, i: Fraction, n: int) -> Fraction:
    """A = P*i/(1-(1+i)**-n) rearranged for P: the largest principal a
    repayment `repayment` can amortise over `n` periods at rate `i`."""
    growth = (1 + i) ** n
    return Fraction(repayment) * (growth - 1) / (i * growth)


def _simulate_schedule(principal, i: Fraction, repayment, n: int) -> Fraction:
    """Independent-check route: simulate the amortisation SCHEDULE
    period-by-period (balance = balance*(1+i) - repayment) and return the
    final balance -- a genuinely different route from the closed form used
    for construction (a numeric probe of the closed form, not a hand-rolled
    equivalence test)."""
    balance = Fraction(principal)
    for _ in range(n):
        balance = balance * (1 + i) - Fraction(repayment)
    return balance


def _rbl_easy(rng):
    for _attempt in range(_MAX_ATTEMPTS):
        principal = rng.randrange(100_000, 600_001, 10_000)
        rate_tenths = rng.choice(_RATE_TENTHS)
        term_years = rng.choice(_TERM_YEARS)
        n = term_years * 12
        i = _periodic_rate(rate_tenths, 12)
        annual_rate = Fraction(rate_tenths, 10)
        rate_text = _rate_text(rate_tenths)

        repayment_exact = _repayment(principal, i, n)
        answer = _money(repayment_exact)

        # misconception: flat interest on the ORIGINAL principal every
        # period, never reducing -- confuses this skill with
        # flat-loan-repayment.
        total_flat = Fraction(principal) + Fraction(principal) * annual_rate * term_years / 100
        flat_repayment = _money(total_flat / n)

        # misconception: used the ANNUAL rate directly as the periodic
        # rate -- forgot to divide by periods-per-year.
        i_wrong = annual_rate / 100
        wrong_rate_repayment = _money(_repayment(principal, i_wrong, n))

        # misconception: off-by-one on n -- used total YEARS instead of
        # total MONTHS as the period count (same monthly rate `i`).
        off_by_one_repayment = _money(_repayment(principal, i, term_years))

        candidates = (flat_repayment, wrong_rate_repayment, off_by_one_repayment)
        if any(c <= 0 for c in candidates) or any(c == answer for c in candidates):
            continue

        if not _pairwise_distinct(answer, *candidates):
            continue
        break
    else:
        raise RuntimeError("p12b_reducing_balance_loan/easy: no valid draw found")

    check_pairwise_distinct(answer, *candidates, msg="easy: distractor collision")

    # Independent check: simulate the schedule using the ROUNDED (as
    # advertised) repayment and assert it zeros the balance within a tight,
    # explicit tolerance -- a genuinely different route (a step-by-step
    # numeric probe) from the closed-form construction above. The tolerance
    # is not an arbitrary guess: the repayment is off from its exact value
    # by at most half a cent (round-half-up to 2 d.p.), and each period's
    # error compounds like an annuity's future value, so the total drift at
    # period n is bounded by `0.005 * ((1+i)^n - 1)/i` -- the exact
    # future-value-of-an-annuity factor applied to the per-period rounding
    # error.
    final_balance = _simulate_schedule(principal, i, answer, n)
    growth = (1 + i) ** n
    tol = Fraction(1, 200) * (growth - 1) / i
    check(
        abs(final_balance) < tol,
        f"easy: schedule simulation did not zero out: P={principal}, rate={annual_rate}, "
        f"n={n}, A={answer}, final_balance={final_balance}",
    )

    text = phrase(
        rng,
        [
            "A home loan of ${p:,} is charged {r}% per annum, compounding monthly, and is "
            "repaid in equal monthly instalments over {t} years. Find the monthly repayment, "
            "in dollars to the nearest cent.",
            "A borrower takes out a ${p:,} home loan at {r}% per annum, compounding monthly, "
            "to be repaid in equal monthly instalments over {t} years. Calculate the monthly "
            "repayment, in dollars to the nearest cent.",
            "Consider a ${p:,} reducing-balance loan charged at {r}% per annum, compounding "
            "monthly, and repaid in equal monthly instalments over {t} years. What is the "
            "required monthly repayment, in dollars to the nearest cent?",
        ],
        p=principal,
        r=rate_text,
        t=term_years,
    )
    answer_text = _money_text(repayment_exact)
    working = [
        f"Monthly rate i = {rate_text}%/12 = {i}. n = {term_years} x 12 = {n} months.",
        f"A = P*i / (1 - (1+i)^-n) = {answer_text}.",
    ]
    distractors = pick_distractors(
        answer_text,
        [_money_text(c) for c in (flat_repayment, wrong_rate_repayment, off_by_one_repayment)],
    )
    return dict(
        text=text,
        latex=text,
        answer=numeric_answer(answer),
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        working=working,
        distractors=distractors,
    )


def _rbl_medium(rng):
    for _attempt in range(_MAX_ATTEMPTS):
        principal = rng.randrange(100_000, 600_001, 10_000)
        rate_tenths = rng.choice(_RATE_TENTHS)
        term_years = rng.choice(_TERM_YEARS)
        n = term_years * 12
        i = _periodic_rate(rate_tenths, 12)
        annual_rate = Fraction(rate_tenths, 10)
        rate_text = _rate_text(rate_tenths)
        repayment = _money(_repayment(principal, i, n))

        k_years = rng.randint(1, term_years - 1)
        k = k_years * 12

        balance_exact = _balance_after(principal, i, repayment, k)
        answer = _money(balance_exact)
        if answer <= 0:
            continue

        # misconception: forgets the reducing effect of repayments
        # entirely -- future value of P alone, no repayments subtracted.
        fv_only = _money(Fraction(principal) * (1 + i) ** k)

        # misconception: pays down the principal LINEARLY (evenly over the
        # whole term), ignoring interest and the true amortisation shape.
        linear_balance = _money(Fraction(principal) - Fraction(principal) * k / n)

        # misconception: off-by-one on n -- uses k_years directly as the
        # period count instead of k months.
        off_by_one = _money(_balance_after(principal, i, repayment, k_years))

        candidates = (fv_only, linear_balance, off_by_one)
        if any(c <= 0 for c in candidates) or any(c == answer for c in candidates):
            continue

        if not _pairwise_distinct(answer, *candidates):
            continue
        break
    else:
        raise RuntimeError("p12b_reducing_balance_loan/medium: no valid draw found")

    check_pairwise_distinct(answer, *candidates, msg="medium: distractor collision")

    # Independent check: simulate the schedule period-by-period (a loop) and
    # compare its ending balance after k periods against the closed form
    # used for construction -- two genuinely different derivation routes.
    via_schedule = _simulate_schedule(principal, i, repayment, k)
    check_equivalent(
        via_schedule,
        balance_exact,
        f"medium: schedule simulation disagrees with closed-form balance: P={principal}, i={i}, A={repayment}, k={k}",
    )

    # The question states P, rate and term AS WELL AS the (cents-rounded)
    # instalment -- redundant by the closed form, but deliberately so: a
    # learner using a calculator's AMORT/BAL function reads the balance off
    # the STATED instalment, not a freshly re-solved exact PMT, and the two
    # routes disagree by several dollars at a large k (the reducing balance
    # accumulates the half-cent-per-period rounding gap). Pinning "using
    # this instalment amount" removes that second, wrong-by-construction
    # reading rather than leaving it ambiguous.
    k_word = "year" if k_years == 1 else "years"
    text = phrase(
        rng,
        [
            "A home loan of ${p:,} is charged {r}% per annum, compounding monthly, and "
            "repaid with a monthly instalment of ${a} over {t} years. Using this instalment "
            "amount, find the outstanding balance after {k} {kw} of repayments, in dollars to "
            "the nearest cent.",
            "A borrower with a ${p:,} home loan at {r}% per annum, compounding monthly, pays "
            "a monthly instalment of ${a} over {t} years. Using this instalment amount, "
            "calculate the outstanding balance after {k} {kw} of repayments, in dollars to the "
            "nearest cent.",
            "Consider a ${p:,} loan, {r}% per annum compounding monthly, repaid with a "
            "monthly instalment of ${a} over {t} years. Using this instalment amount, what is "
            "the outstanding balance after {k} {kw} of repayments, in dollars to the nearest "
            "cent?",
        ],
        p=principal,
        r=rate_text,
        a=_money_text(repayment),
        t=term_years,
        k=k_years,
        kw=k_word,
    )
    answer_text = _money_text(balance_exact)
    working = [
        f"i = {rate_text}%/12 = {i}. k = {k_years} x 12 = {k} months.",
        f"Balance = P*(1+i)^k - A*((1+i)^k - 1)/i = {answer_text}.",
    ]
    distractors = pick_distractors(answer_text, [_money_text(c) for c in (fv_only, linear_balance, off_by_one)])
    return dict(
        text=text,
        latex=text,
        answer=numeric_answer(answer),
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        working=working,
        distractors=distractors,
    )


def _rbl_hard(rng):
    for _attempt in range(_MAX_ATTEMPTS):
        repayment = Fraction(rng.randrange(800, 3001, 50))
        rate_tenths = rng.choice(_RATE_TENTHS)
        term_years = rng.choice(_TERM_YEARS)
        n = term_years * 12
        i = _periodic_rate(rate_tenths, 12)
        annual_rate = Fraction(rate_tenths, 10)
        rate_text = _rate_text(rate_tenths)

        principal_exact = _max_principal(repayment, i, n)
        answer = _money(principal_exact)
        if answer <= 0:
            continue

        # misconception: flat-interest inverse -- treats the total repaid
        # (A*n) as principal plus simple interest on the ORIGINAL principal.
        total_repaid = repayment * n
        flat_principal = _money(total_repaid / (1 + annual_rate * term_years / 100))

        # misconception: used the ANNUAL rate directly as the periodic rate.
        i_wrong = annual_rate / 100
        wrong_rate_principal = _money(_max_principal(repayment, i_wrong, n))

        # misconception: off-by-one on n -- used term_years directly as the
        # period count instead of total months.
        off_by_one_principal = _money(_max_principal(repayment, i, term_years))

        candidates = (flat_principal, wrong_rate_principal, off_by_one_principal)
        if any(c <= 0 for c in candidates) or any(c == answer for c in candidates):
            continue

        if not _pairwise_distinct(answer, *candidates):
            continue
        break
    else:
        raise RuntimeError("p12b_reducing_balance_loan/hard: no valid draw found")

    check_pairwise_distinct(answer, *candidates, msg="hard: distractor collision")

    # Independent check: an algebraic inverse identity, a genuinely
    # different route from the construction formula -- feeding the
    # constructed principal back through the FORWARD repayment formula must
    # exactly reproduce the stated affordable repayment.
    check_equivalent(
        _repayment(principal_exact, i, n),
        repayment,
        f"hard: forward repayment(max principal) != stated repayment: P={principal_exact}, i={i}, n={n}, A={repayment}",
    )

    text = phrase(
        rng,
        [
            "A borrower can afford a monthly repayment of ${a}. A lender charges {r}% per "
            "annum, compounding monthly, over a {t}-year reducing-balance loan. Find the "
            "maximum principal the borrower can afford to borrow, in dollars to the nearest "
            "cent.",
            "A borrower can afford to repay ${a} per month. A lender offers {r}% per annum, "
            "compounding monthly, over a {t}-year reducing-balance loan. Calculate the "
            "maximum principal the borrower could be advanced, in dollars to the nearest "
            "cent.",
            "Given a maximum affordable monthly repayment of ${a}, and a lender charging {r}% "
            "per annum, compounding monthly, over a {t}-year reducing-balance loan, what is "
            "the largest principal the borrower can afford to borrow, in dollars to the "
            "nearest cent?",
        ],
        a=repayment,
        r=rate_text,
        t=term_years,
    )
    answer_text = _money_text(principal_exact)
    working = [
        f"i = {rate_text}%/12 = {i}. n = {term_years} x 12 = {n} months.",
        f"P = A*(1-(1+i)^-n)/i = {answer_text}.",
    ]
    distractors = pick_distractors(
        answer_text,
        [_money_text(c) for c in (flat_principal, wrong_rate_principal, off_by_one_principal)],
    )
    return dict(
        text=text,
        latex=text,
        answer=numeric_answer(answer),
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        working=working,
        distractors=distractors,
    )


def _rbl_exam(rng):
    for _attempt in range(_MAX_ATTEMPTS):
        principal = rng.randrange(100_000, 600_001, 10_000)
        rate_tenths = rng.choice(_RATE_TENTHS)
        term_years = rng.choice(_TERM_YEARS)
        annual_rate = Fraction(rate_tenths, 10)
        rate_text = _rate_text(rate_tenths)

        # Repayments are rounded to the nearest cent BEFORE the totals are
        # built (`repay_m`, `repay_f`) -- the natural learner path with a
        # financial calculator: solve PMT, read off the displayed
        # cents-rounded repayment, then multiply that stated figure by the
        # number of repayments. Using the unrounded exact repayment for a
        # 240-600 period total instead can disagree with this reading by
        # several dollars (up to half a cent per period), which is larger
        # than the stated cent precision -- so the exact value is never
        # used for the totals below, only for the earlier per-tier
        # repayment formulas that stay genuinely exact.
        n_monthly = term_years * 12
        i_monthly = _periodic_rate(rate_tenths, 12)
        repay_m = _money(_repayment(principal, i_monthly, n_monthly))

        n_fortnightly = term_years * 26
        i_fortnightly = _periodic_rate(rate_tenths, 26)
        repay_f = _money(_repayment(principal, i_fortnightly, n_fortnightly))

        total_monthly = repay_m * n_monthly
        total_fortnightly = repay_f * n_fortnightly
        diff_exact = total_fortnightly - total_monthly
        answer = _money(diff_exact)
        if answer == 0:
            continue

        # misconception: applies the MONTHLY repayment amount at the
        # fortnightly FREQUENCY -- forgets that a different frequency
        # changes the repayment amount itself.
        wrong_diff1 = _money(repay_m * n_fortnightly - total_monthly)

        # misconception: keeps the MONTHLY period count for both totals,
        # only changing the periodic rate to the fortnightly one.
        repay_fortnightly_wrong_n = _money(_repayment(principal, i_fortnightly, n_monthly))
        wrong_diff2 = _money(repay_fortnightly_wrong_n * n_monthly - total_monthly)

        # misconception: compares the PERIODIC repayment amounts directly
        # instead of the TOTAL amounts repaid over the loan's life.
        wrong_diff3 = _money(repay_f - repay_m)

        candidates = (wrong_diff1, wrong_diff2, wrong_diff3)
        if any(c == answer for c in candidates):
            continue

        if not _pairwise_distinct(answer, *candidates):
            continue
        break
    else:
        raise RuntimeError("p12b_reducing_balance_loan/exam: no valid draw found")

    check_pairwise_distinct(answer, *candidates, msg="exam: distractor collision")

    # Independent check: each frequency's own repayment must independently
    # amortise its own loan to (near) zero via schedule simulation -- the
    # same route as the easy tier's check, run twice, before trusting the
    # totals used to build the comparison.
    for i_val, n_val, repay_val in (
        (i_monthly, n_monthly, repay_m),
        (i_fortnightly, n_fortnightly, repay_f),
    ):
        final_balance = _simulate_schedule(principal, i_val, repay_val, n_val)
        growth_val = (1 + i_val) ** n_val
        tol_val = Fraction(1, 200) * (growth_val - 1) / i_val
        check(
            abs(final_balance) < tol_val,
            f"exam: schedule simulation did not zero out for n={n_val}, balance={final_balance}, tol={tol_val}",
        )

    rounding_note = "Round each repayment to the nearest cent before totalling."
    text = phrase(
        rng,
        [
            "A ${p:,} home loan at {r}% per annum is fully repaid over {t} years, compounding "
            "at the same nominal annual rate under two repayment frequencies: monthly (12 "
            "repayments per year) and fortnightly (26 repayments per year). {note} Find the "
            "difference between the total amount repaid under fortnightly repayments and the "
            "total amount repaid under monthly repayments (fortnightly total minus monthly "
            "total), in dollars to the nearest cent.",
            "A ${p:,} home loan, charged at {r}% per annum and fully repaid over {t} years, is "
            "compared under two repayment frequencies at the same nominal annual rate: monthly "
            "(12 repayments per year) and fortnightly (26 repayments per year). {note} "
            "Calculate the difference between the total amount repaid under fortnightly "
            "repayments and the total amount repaid under monthly repayments (fortnightly "
            "total minus monthly total), in dollars to the nearest cent.",
            "Compare monthly (12 repayments per year) and fortnightly (26 repayments per year) "
            "repayments, both compounding at the same nominal annual rate of {r}% per annum, "
            "for a ${p:,} home loan fully repaid over {t} years. {note} What is the difference "
            "between the total amount repaid under fortnightly repayments and the total amount "
            "repaid under monthly repayments (fortnightly total minus monthly total), in "
            "dollars to the nearest cent?",
        ],
        p=principal,
        r=rate_text,
        t=term_years,
        note=rounding_note,
    )
    answer_text = _money_text(diff_exact)
    working = [
        f"Monthly: A={_money_text(repay_m)}, n={n_monthly}, total={_money_text(total_monthly)}.",
        f"Fortnightly: A={_money_text(repay_f)}, n={n_fortnightly}, total={_money_text(total_fortnightly)}.",
        f"Difference = {answer_text}.",
    ]
    distractors = pick_distractors(answer_text, [_money_text(c) for c in (wrong_diff1, wrong_diff2, wrong_diff3)])
    return dict(
        text=text,
        latex=text,
        answer=numeric_answer(answer),
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        working=working,
        distractors=distractors,
    )


@generator(
    "p12b_reducing_balance_loan",
    level=12,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p12b_reducing_balance_loan(rng, tier="easy"):
    """ACMGM098: reducing-balance (amortising) loan repayment. easy: forward
    monthly repayment. medium: outstanding balance after k < n payments.
    hard: INVERSE -- maximum affordable principal for a stated repayment.
    exam: compares total repaid under monthly vs fortnightly repayment
    frequency at the same nominal rate. See the module docstring."""
    if tier == "easy":
        return _rbl_easy(rng)
    if tier == "medium":
        return _rbl_medium(rng)
    if tier == "hard":
        return _rbl_hard(rng)
    if tier == "exam":
        return _rbl_exam(rng)
    raise KeyError(f"p12b_reducing_balance_loan has no tier {tier!r}")


# =============================================================================
# ACMGM100 -- p12b_annuity_perpetuity
# =============================================================================


def _annuity_pv(payment, i: Fraction, n: int) -> Fraction:
    """PV = A * (1 - (1+i)**-n) / i: the amount to invest now to fund `n`
    future withdrawals of `payment` at periodic rate `i`."""
    growth = (1 + i) ** n
    return Fraction(payment) * (growth - 1) / (i * growth)


def _perpetuity_pv(payment, i: Fraction) -> Fraction:
    """PV = A / i exactly -- the n -> infinity limit of `_annuity_pv`, since
    (1+i)**-n -> 0 as n -> infinity for any i > 0 (a strictly-greater-than-1
    base raised to an unboundedly negative power), so the `growth`-scaled
    correction term in `_annuity_pv` vanishes entirely and no large-n
    numeric approximation is needed -- this is the exact closed form."""
    return Fraction(payment) / i


def _periodic_from_pv(pv, i: Fraction, n: int) -> Fraction:
    """A = PV*i / (1 - (1+i)**-n): the periodic income a lump sum `pv`
    sustains for `n` periods at rate `i` -- the same formula shape as
    `_repayment` above (an annuity's PV plays the role a loan's principal
    does), reused here under its annuity meaning (sustainable retirement
    income), never imported across modules."""
    growth = (1 + i) ** n
    return Fraction(pv) * i * growth / (growth - 1)


def _ann_easy(rng):
    for _attempt in range(_MAX_ATTEMPTS):
        rate_tenths = rng.choice(_RATE_TENTHS)
        i = _periodic_rate(rate_tenths, 12)
        annual_rate = Fraction(rate_tenths, 10)
        rate_text = _rate_text(rate_tenths)
        income = rng.randrange(500, 5001, 50)

        pv_exact = _perpetuity_pv(income, i)
        answer = _money(pv_exact)

        # misconception: applies the FINITE annuity formula (a plausible
        # term) to a question that is a perpetuity -- the exact confusion
        # the tie check below guards against.
        n_plausible = rng.choice(_TERM_YEARS) * 12
        finite_instead = _money(_annuity_pv(income, i, n_plausible))

        # misconception: used the ANNUAL rate directly as the periodic
        # rate (a 12x-scale error against the correct periodic PV).
        i_wrong = annual_rate / 100
        wrong_rate_pv = _money(_perpetuity_pv(income, i_wrong))

        # misconception: forgot to discount at all -- one year's income,
        # simple multiplication instead of the perpetuity relation.
        one_year_total = _money(Fraction(income) * 12)

        candidates = (finite_instead, wrong_rate_pv, one_year_total)
        if any(c <= 0 for c in candidates) or any(c == answer for c in candidates):
            continue

        if not _pairwise_distinct(answer, *candidates):
            continue
        break
    else:
        raise RuntimeError("p12b_annuity_perpetuity/easy: no valid draw found")

    check_pairwise_distinct(answer, *candidates, msg="easy: distractor collision")

    # Independent check: PV*i == A exactly -- the defining relation, a
    # different route from the division that produced PV.
    check(pv_exact * i == Fraction(income), f"easy: PV*i != A: PV={pv_exact}, i={i}, A={income}")
    # Tie-boundary guard: the perpetuity PV must never be
    # mistaken for a finite-term PV at any plausible term -- proven by
    # construction on every seed via the distinctness check above, which
    # already rejects a draw where `finite_instead == answer`.

    text = phrase(
        rng,
        [
            "An investor wants to withdraw ${a} every month, forever, without ever running "
            "out of the invested capital (a perpetuity). The investment earns {r}% per annum, "
            "compounding monthly. Find the amount that must be invested now, in dollars to "
            "the nearest cent.",
            "An investor plans to withdraw ${a} every month, indefinitely, without ever "
            "depleting the invested capital (a perpetuity). Given an investment earning {r}% "
            "per annum, compounding monthly, calculate the amount that must be invested now, "
            "in dollars to the nearest cent.",
            "To fund a perpetual monthly withdrawal of ${a}, with the investment earning {r}% "
            "per annum, compounding monthly, what amount must be invested now so the capital "
            "is never depleted, in dollars to the nearest cent?",
        ],
        a=income,
        r=rate_text,
    )
    answer_text = _money_text(pv_exact)
    working = [
        f"i = {rate_text}%/12 = {i}.",
        f"Perpetuity: PV = A/i = {answer_text}.",
    ]
    distractors = pick_distractors(
        answer_text, [_money_text(c) for c in (finite_instead, wrong_rate_pv, one_year_total)]
    )
    return dict(
        text=text,
        latex=text,
        answer=numeric_answer(answer),
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        working=working,
        distractors=distractors,
    )


def _ann_medium(rng):
    for _attempt in range(_MAX_ATTEMPTS):
        rate_tenths = rng.choice(_RATE_TENTHS)
        i = _periodic_rate(rate_tenths, 12)
        annual_rate = Fraction(rate_tenths, 10)
        rate_text = _rate_text(rate_tenths)
        income = rng.randrange(500, 5001, 50)
        term_years = rng.choice(_TERM_YEARS)
        n = term_years * 12

        pv_exact = _annuity_pv(income, i, n)
        answer = _money(pv_exact)

        # Tie-boundary proof, enforced by construction on
        # EVERY seed (not sampled after the fact): the finite-annuity PV
        # must never be within a cent of the perpetuity PV at the same
        # rate, so a learner can never confuse the two formulas and land on
        # an indistinguishable answer.
        perp_pv = _perpetuity_pv(income, i)
        if perp_pv - pv_exact <= Fraction(1, 100):
            continue

        # misconception: applies the PERPETUITY formula to this FINITE
        # question -- the exact confusion the guard above targets.
        perpetuity_instead = _money(perp_pv)

        # misconception: forgot to discount at all -- PV = A*n, simple
        # multiplication.
        naive_total = _money(Fraction(income) * n)

        # misconception: used the ANNUAL rate directly as the periodic
        # rate.
        i_wrong = annual_rate / 100
        wrong_rate_pv = _money(_annuity_pv(income, i_wrong, n))

        candidates = (perpetuity_instead, naive_total, wrong_rate_pv)
        if any(c <= 0 for c in candidates) or any(c == answer for c in candidates):
            continue

        if not _pairwise_distinct(answer, *candidates):
            continue
        break
    else:
        raise RuntimeError("p12b_annuity_perpetuity/medium: no valid draw found")

    check_pairwise_distinct(answer, *candidates, msg="medium: distractor collision")

    # Independent check: sum-of-discounted-payments route --
    # sum(A/(1+i)**k for k in 1..n) evaluated with exact Fraction
    # arithmetic, against the closed-form geometric-series formula used for
    # construction. Accumulated iteratively (multiplying by the constant
    # discount factor each step) rather than re-exponentiating from
    # scratch at every k.
    discounted_sum = Fraction(0)
    factor = Fraction(1)
    step = Fraction(1) / (1 + i)
    for _ in range(n):
        factor *= step
        discounted_sum += Fraction(income) * factor
    check_equivalent(
        discounted_sum,
        pv_exact,
        f"medium: sum-of-discounted-payments disagrees with closed form: A={income}, i={i}, n={n}",
    )

    text = phrase(
        rng,
        [
            "An investor wants to withdraw ${a} every month for {t} years, after which the "
            "fund is exhausted. The investment earns {r}% per annum, compounding monthly. "
            "Find the amount that must be invested now, in dollars to the nearest cent.",
            "An investor plans to withdraw ${a} every month for {t} years, at which point the "
            "fund will be exhausted. Given an investment earning {r}% per annum, compounding "
            "monthly, calculate the amount that must be invested now, in dollars to the "
            "nearest cent.",
            "To fund a monthly withdrawal of ${a} for {t} years, after which the fund is "
            "exhausted, with the investment earning {r}% per annum, compounding monthly, what "
            "amount must be invested now, in dollars to the nearest cent?",
        ],
        a=income,
        t=term_years,
        r=rate_text,
    )
    answer_text = _money_text(pv_exact)
    working = [
        f"i = {rate_text}%/12 = {i}. n = {term_years} x 12 = {n} months.",
        f"PV = A*(1-(1+i)^-n)/i = {answer_text}.",
    ]
    distractors = pick_distractors(
        answer_text, [_money_text(c) for c in (perpetuity_instead, naive_total, wrong_rate_pv)]
    )
    return dict(
        text=text,
        latex=text,
        answer=numeric_answer(answer),
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        working=working,
        distractors=distractors,
    )


def _ann_hard(rng):
    for _attempt in range(_MAX_ATTEMPTS):
        rate_tenths = rng.choice(_RATE_TENTHS)
        i = _periodic_rate(rate_tenths, 12)
        annual_rate = Fraction(rate_tenths, 10)
        rate_text = _rate_text(rate_tenths)
        income = rng.randrange(1000, 6001, 100)
        # A wider pool than `_TERM_YEARS` (5 values): `n_years` IS the
        # answer this tier draws, so a narrow pool would concentrate the
        # answer distribution directly (measured 23% top-answer share at
        # 200 seeds against `_TERM_YEARS`'s 5 values -- inside the 25%
        # ceiling but too close to it to be a comfortable margin; widened
        # to 25 values here, distinct from `_TERM_YEARS`, which stays as
        # the narrower "typical home-loan term" pool for the other tiers
        # and for this skill's own medium/exam tiers, where `n_years` is
        # only an input, not the answer).
        n_years = rng.randint(8, 32)
        n = n_years * 12

        pv_exact = _annuity_pv(income, i, n)
        pv = _money(pv_exact)
        if pv <= 0:
            continue

        answer = n_years

        # Rounding-margin guard: the DISPLAYED (cents-rounded) pv must not
        # be ambiguous between n_years and either neighbouring whole year --
        # otherwise a learner (or the grader) could round-trip to the wrong
        # n from the rounded pv alone.
        pv_prev = _money(_annuity_pv(income, i, (n_years - 1) * 12)) if n_years > 1 else None
        pv_next = _money(_annuity_pv(income, i, (n_years + 1) * 12))
        gaps = [abs(pv - pv_next)]
        if pv_prev is not None:
            gaps.append(abs(pv - pv_prev))
        if min(gaps) <= Fraction(1, 2):
            continue

        # misconception: forgot to discount at all -- treats PV as if it
        # were simply drawn down with no interest earned, so
        # naive_years = floor(PV / (12 * A)).
        naive_years = int(pv // (Fraction(income) * 12))

        # misconception: reports the number of MONTHS instead of years.
        n_in_months = n

        # misconception: halves the term.
        half_years = max(1, n_years // 2)

        candidates = (naive_years, n_in_months, half_years)
        if any(c <= 0 for c in candidates) or any(c == answer for c in candidates):
            continue

        if not _pairwise_distinct(answer, *candidates):
            continue
        break
    else:
        raise RuntimeError("p12b_annuity_perpetuity/hard: no valid draw found")

    check_pairwise_distinct(answer, *candidates, msg="hard: distractor collision")

    # Independent check: the defining algebraic identity
    # 1 - PV*i/A == (1+i)**-n, using the EXACT (unrounded) pv -- a
    # genuinely different route from constructing n forward and reading
    # PV(n) off the closed form, and one that needs no logarithm at all
    # since n was fixed at construction time.
    lhs = 1 - pv_exact * i / Fraction(income)
    rhs = (1 + i) ** (-n)
    check_equivalent(lhs, rhs, f"hard: PV/A/i algebraic identity failed: PV={pv_exact}, A={income}, i={i}, n={n}")

    text = phrase(
        rng,
        [
            "${pv} is invested now, earning {r}% per annum, compounding monthly. It funds a "
            "monthly withdrawal of ${a}, after which the fund is exactly exhausted. Find how "
            "many whole years the withdrawals last.",
            "An investor deposits ${pv} now, earning {r}% per annum, compounding monthly, and "
            "withdraws ${a} every month until the fund is exactly exhausted. Calculate how "
            "many whole years the withdrawals last.",
            "${pv} is invested at {r}% per annum, compounding monthly, to fund a monthly "
            "withdrawal of ${a} until the fund is exactly exhausted. How many whole years do "
            "the withdrawals last?",
        ],
        pv=_money_text(pv),
        r=rate_text,
        a=income,
    )
    answer_text = str(answer)
    working = [
        f"i = {rate_text}%/12 = {i}. Solve PV = A*(1-(1+i)^-n)/i for n, in months, then convert "
        f"to years: n = {n} months = {n_years} years.",
    ]
    distractors = pick_distractors(answer_text, [str(c) for c in (naive_years, n_in_months, half_years)])
    return dict(
        text=text,
        latex=text,
        answer=answer,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        working=working,
        distractors=distractors,
    )


def _ann_exam(rng):
    for _attempt in range(_MAX_ATTEMPTS):
        rate_tenths = rng.choice(_RATE_TENTHS)
        i = _periodic_rate(rate_tenths, 12)
        annual_rate = Fraction(rate_tenths, 10)
        rate_text = _rate_text(rate_tenths)
        principal = rng.randrange(50_000, 300_001, 10_000)
        m_years = rng.choice(_ACCUM_YEARS)
        n_years = rng.choice(_TERM_YEARS)
        m = m_years * 12
        n = n_years * 12

        fv_exact = Fraction(principal) * (1 + i) ** m
        income_exact = _periodic_from_pv(fv_exact, i, n)
        answer = _money(income_exact)
        if answer <= 0:
            continue

        # misconception: skips the accumulation stage entirely -- treats
        # the ORIGINAL principal as if it were already the retirement lump
        # sum.
        skip_accum = _money(_periodic_from_pv(principal, i, n))

        # misconception: grows the lump sum with SIMPLE (non-compounding)
        # interest instead of compounding it.
        fv_simple = Fraction(principal) * (1 + annual_rate * m_years / 100)
        simple_growth_income = _money(_periodic_from_pv(fv_simple, i, n))

        # misconception: swaps the accumulation period and the payout
        # period -- grows the lump sum for n years, then sizes the payout
        # over m years.
        fv_swapped = Fraction(principal) * (1 + i) ** n
        swapped_income = _money(_periodic_from_pv(fv_swapped, i, m))

        candidates = (skip_accum, simple_growth_income, swapped_income)
        if any(c <= 0 for c in candidates) or any(c == answer for c in candidates):
            continue

        if not _pairwise_distinct(answer, *candidates):
            continue
        break
    else:
        raise RuntimeError("p12b_annuity_perpetuity/exam: no valid draw found")

    check_pairwise_distinct(answer, *candidates, msg="exam: distractor collision")

    # Independent check, two stages, each via a genuinely different route:
    # (1) accumulation -- simulate the monthly compounding growth with a
    #     LOOP and compare against the closed-form fv used for construction.
    grown = Fraction(principal)
    for _ in range(m):
        grown *= 1 + i
    check_equivalent(
        grown, fv_exact, f"exam: loop-accumulated FV disagrees with closed form: P={principal}, i={i}, m={m}"
    )
    # (2) payout -- sum-of-discounted-payments route (as in the medium
    #     tier), confirming the computed income truly funds n withdrawals
    #     from the accumulated lump sum.
    discounted_sum = Fraction(0)
    factor = Fraction(1)
    step = Fraction(1) / (1 + i)
    for _ in range(n):
        factor *= step
        discounted_sum += income_exact * factor
    check_equivalent(
        discounted_sum,
        fv_exact,
        f"exam: sum-of-discounted-payments disagrees with accumulated FV: A={income_exact}, i={i}, n={n}",
    )

    text = phrase(
        rng,
        [
            "${p:,} is invested now at {r}% per annum, compounding monthly, for {m} years. At "
            "the end of that period the accumulated amount is used to fund a monthly "
            "retirement income for the following {n} years (compounding at the same monthly "
            "rate), after which the fund is exactly exhausted. Find the maximum sustainable "
            "monthly income, in dollars to the nearest cent.",
            "An investor deposits ${p:,} now, earning {r}% per annum compounding monthly, and "
            "leaves it to grow for {m} years. The accumulated amount then funds a monthly "
            "retirement income for the following {n} years, compounding at the same monthly "
            "rate, until the fund is exactly exhausted. Calculate the maximum sustainable "
            "monthly income, in dollars to the nearest cent.",
            "${p:,} is invested now at {r}% per annum, compounding monthly, and grows for {m} "
            "years before being used to fund a monthly retirement income for the following {n} "
            "years (compounding at the same monthly rate), after which the fund is exactly "
            "exhausted. What is the maximum sustainable monthly income, in dollars to the "
            "nearest cent?",
        ],
        p=principal,
        r=rate_text,
        m=m_years,
        n=n_years,
    )
    answer_text = _money_text(income_exact)
    working = [
        f"Stage 1: FV = P*(1+i)^m = {_money_text(fv_exact)} after {m_years} years.",
        f"Stage 2: A = FV*i/(1-(1+i)^-n) = {answer_text} over {n_years} years.",
    ]
    distractors = pick_distractors(
        answer_text,
        [_money_text(c) for c in (skip_accum, simple_growth_income, swapped_income)],
    )
    return dict(
        text=text,
        latex=text,
        answer=numeric_answer(answer),
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        working=working,
        distractors=distractors,
    )


@generator(
    "p12b_annuity_perpetuity",
    level=12,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p12b_annuity_perpetuity(rng, tier="easy"):
    """ACMGM100: annuities and perpetuities. easy: PERPETUITY (no n) --
    PV = A/i. medium: FINITE annuity -- PV = A*(1-(1+i)^-n)/i, with the
    tie-boundary guard against the perpetuity PV enforced by construction.
    hard: INVERSE -- given PV, i and A, find how many whole years n the
    income lasts. exam: two-stage accumulation-then-annuity -- grow a lump
    sum for m years, then treat it as the annuity PV for the following n
    years. See the module docstring."""
    if tier == "easy":
        return _ann_easy(rng)
    if tier == "medium":
        return _ann_medium(rng)
    if tier == "hard":
        return _ann_hard(rng)
    if tier == "exam":
        return _ann_exam(rng)
    raise KeyError(f"p12b_annuity_perpetuity has no tier {tier!r}")
