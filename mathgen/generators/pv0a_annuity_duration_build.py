"""Generates `pv0a_annuity_duration` questions (ACMGM099): the duration of an annuity modelled with a recurrence
relation.

Scope: "Given an annuity's opening balance, interest rate per
period and fixed withdrawal per period, iterate the recurrence to find how many full
withdrawals the annuity pays before its balance falls below one withdrawal." The answer is
ALWAYS a whole number of withdrawals (a count), never a balance -- that is the sharp
neighbour ``senior-use-technology-and-a-recurrence-relation-to-model``, which returns a
reducing-BALANCE loan's balance after a stated number of periods.

Model: ``A(0)`` is the opening balance; each period interest at rate ``r`` is added and then
the fixed withdrawal ``d`` is taken -- ``A(n+1) = A(n)(1 + r) - d``. Three readings of "how
many full withdrawals" are all made to agree, with margin, on every draw: (a) a withdrawal is
paid whenever ``A(k)(1 + r) >= d``; (b) count until the post-withdrawal balance first falls
below ``d``; (c) count until the next withdrawal would take the balance below zero -- (a) and
(c) are the SAME condition (``A(k)(1 + r) < d`` iff the next withdrawal goes negative), so the
only genuinely separate reading is (b), and construction enforces ``A(n-1) >= d + m`` (reading
(b)'s margin) and ``A(n)(1 + r) <= d - m`` (reading (a)/(c)'s margin), which -- combined with
the sequence being strictly decreasing whenever the perpetuity guard ``A(0)*r < d`` holds --
guarantees every A(k) for k < n clears its own margin too (monotonic decrease from a
decreasing start).

Construct to target: the answer (or, at hard, the target DIFFERENCE between two durations) is
drawn FIRST, uniformly, then the opening balance (or the changed parameter, at hard/exam) is
solved from the closed form ``A(n) = A(0)(1+r)^n - d((1+r)^n - 1)/r`` to land the residual
balance inside the safe margin window, rounded to a whole dollar and RE-VERIFIED by iterating
the recurrence from scratch (never trusting the closed form's own rounding).

Tiers: easy states the recurrence with its numbers substituted; medium states an annual rate
compounded monthly/quarterly with no formula, so the learner forms the per-period rate; hard
asks the EFFECT of changing exactly one of the opening balance, the rate or the withdrawal
(the name's own words) on the duration; exam carries a mid-life change (in the withdrawal or
the rate, after the k-th withdrawal) forward from the balance actually reached, not from the
opening balance again.
"""

from __future__ import annotations

from fractions import Fraction

from mathgen.core import (
    VerificationError,
    check,
    check_distinct,
    check_equivalent,
    generator,
    ordinal,
    pick_distractors,
    round_half_up,
)

_A0_LO, _A0_HI = Fraction(1000), Fraction(600_000)
_D_LO, _D_HI = Fraction(300), Fraction(9000)


def _margin(d: Fraction) -> Fraction:
    """At least $1 and at least 0.5% of the withdrawal -- the gap every
    boundary condition below must clear."""
    return max(Fraction(1), d * Fraction(1, 200))


def _fmt_exact(x: Fraction, dp: int) -> str:
    """`x` at up to `dp` decimal places, trailing zeros trimmed -- "1.5", not
    "1.50" or "1.5000". Every value passed here has a denominator dividing a
    power of 10 well within `dp` digits, so the float round-trip is exact."""
    s = f"{float(x):.{dp}f}".rstrip("0").rstrip(".")
    return s if s else "0"


def _fmt_pct(x: Fraction) -> str:
    return _fmt_exact(x, 4)


def _draw_withdrawal(rng, lo: int = 400, hi: int = 6000, step: int = 50) -> Fraction:
    return Fraction(rng.randrange(lo, hi + 1, step))


def _draw_freq_rate(rng):
    """(freq, lo, hi, r, r_pct) -- `r` is the PER-PERIOD rate; `freq` is 12
    (monthly) or 4 (quarterly), each drawn with a per-period-rate band scaled
    so the resulting nominal ANNUAL rate is the same 2.4%-7.2% range either
    way (0.20%-0.60% per month is exactly 0.60%-1.80% per quarter)."""
    freq = rng.choice((12, 4))
    lo, hi = (20, 60) if freq == 12 else (60, 180)
    hundredths = rng.randint(lo, hi)
    r = Fraction(hundredths, 10000)
    r_pct = Fraction(hundredths, 100)
    return freq, lo, hi, r, r_pct


def _freq_words(freq: int):
    return ("monthly", "month") if freq == 12 else ("quarterly", "quarter")


# =============================================================================
# The recurrence, iterated two genuinely different ways
# =============================================================================


def _iterate_annuity(a0, r: Fraction, d: Fraction, limit: int = 600):
    """Reading (a)/(c): count full withdrawals -- interest added, then the
    withdrawal taken -- until the next one cannot be paid in full.

    Returns ``(n, balance_after_n, balance_before_last_withdrawal,
    failed_grown_balance)``: ``balance_after_n`` is A(n); the third element is
    A(n-1) (the balance just before the LAST successful withdrawal's growth
    was applied -- reading (b)'s own quantity); the fourth is A(n)(1 + r),
    the grown balance that first failed to reach `d`."""
    n = 0
    b = Fraction(a0)
    prev_b = None
    grown = b * (1 + r)
    while grown >= d:
        prev_b = b
        b = grown - d
        n += 1
        check(n < limit, "runaway search: the annuity never depletes")
        grown = b * (1 + r)
    return n, b, prev_b, grown


def _iterate_annuity_reading_b(a0, r: Fraction, d: Fraction, limit: int = 600):
    """Reading (b), a genuinely different loop condition: count while the
    POST-withdrawal balance itself (not grown) is still >= d. Construction's
    margins make this always agree with reading (a)/(c) above; this is the
    generator's own internal cross-check (the oracle re-derives both readings
    again from the rendered text, independently of this function)."""
    n = 0
    b = Fraction(a0)
    while b >= d:
        b = b * (1 + r) - d
        n += 1
        check(n < limit, "runaway search (reading b): the annuity never depletes")
    return n, b


def _count_or_none(a0, r, d, limit: int = 600):
    """A misconception route: iterate reading (a) with possibly-wrong
    `a0`/`r`/`d`, WITHOUT raising -- returns None if the fund never depletes
    (a perpetuity, or the search runs past `limit`) so a bad distractor
    candidate is silently dropped rather than crashing the draw."""
    if r is None or r <= 0 or a0 * r >= d:
        return None
    b = Fraction(a0)
    for i in range(limit):
        grown = b * (1 + r)
        if grown < d:
            return i
        b = grown - d
    return None


# =============================================================================
# Construct-to-target solvers -- draw the answer first, solve the parameter
# =============================================================================


def _solve_a0_for_target(r: Fraction, d: Fraction, n: int):
    """Solve the opening balance A(0) (rounded to a whole dollar) so that
    iterating the recurrence gives EXACTLY `n` full withdrawals, with every
    margin cleared. Tries several target residual balances across the safe
    window before giving up (bounded, never runs away)."""
    if r <= 0 or n < 1:
        return None
    m = _margin(d)
    growth = (1 + r) ** n
    max_residual = (d - m) / (1 + r)
    if max_residual <= 0:
        return None
    for num in range(2, 19):
        frac = Fraction(num, 20)
        target_residual = frac * max_residual
        a0 = (target_residual + d * (growth - 1) / r) / growth
        a0_rounded = round_half_up(a0, 0)
        if a0_rounded < _A0_LO or a0_rounded > _A0_HI or a0_rounded * r >= d:
            continue
        try:
            n_actual, b_after, prev_b, fail_grown = _iterate_annuity(a0_rounded, r, d)
        except VerificationError:
            continue
        if n_actual != n or prev_b is None:
            continue
        if prev_b - d < m or d - fail_grown < m:
            continue
        return a0_rounded, n_actual, b_after
    return None


def _solve_d_for_target(a0: Fraction, r: Fraction, n: int):
    """Solve the fixed withdrawal `d` (rounded to a whole dollar), holding
    `a0` and `r` fixed, so the recurrence gives exactly `n` full withdrawals."""
    if r <= 0 or n < 1:
        return None
    growth = (1 + r) ** n
    factor = (growth - 1) / r
    if factor <= 0:
        return None
    for num in range(1, 40):
        frac = Fraction(num, 800)  # target residual as a small fraction of a0
        target_residual = frac * a0
        d = (a0 * growth - target_residual) / factor
        if d <= 0:
            continue
        d_rounded = round_half_up(d, 0)
        if d_rounded < _D_LO or d_rounded > _D_HI or a0 * r >= d_rounded:
            continue
        m = _margin(d_rounded)
        try:
            n_actual, b_after, prev_b, fail_grown = _iterate_annuity(a0, r, d_rounded)
        except VerificationError:
            continue
        if n_actual != n or prev_b is None:
            continue
        if prev_b - d_rounded < m or d_rounded - fail_grown < m:
            continue
        return d_rounded, n_actual, b_after
    return None


def _scan_rate_for_target(a0: Fraction, d: Fraction, n_target: int, lo: int, hi: int):
    """Scan candidate per-period rates (holding `a0` and `d` fixed) for one
    landing on exactly `n_target` full withdrawals with every margin cleared.
    A bounded scan, not a closed-form solve: the closed form is not linear in
    `r`, but the search space (hundredths of a percent) is small and every
    trial iterate is itself bounded (`limit=300`, far above any real n here)."""
    m = _margin(d)
    scan_lo = max(1, lo - 40)
    scan_hi = hi + 120
    for hundredths in range(scan_lo, scan_hi + 1):
        r2 = Fraction(hundredths, 10000)
        if a0 * r2 >= d:
            continue
        try:
            n_actual, b_after, prev_b, fail_grown = _iterate_annuity(a0, r2, d, limit=300)
        except VerificationError:
            continue
        if n_actual != n_target or prev_b is None:
            continue
        if prev_b - d < m or d - fail_grown < m:
            continue
        return r2, Fraction(hundredths, 100), n_actual
    return None


# =============================================================================
# easy -- recurrence stated with its numbers substituted
# =============================================================================

_EASY_TEMPLATES = (
    "An annuity begins with ${a0} invested. Each period, interest of {r}% is added to the "
    "balance and then a fixed ${d} is withdrawn -- the recurrence relation "
    "A(n+1) = A(n)({gf}) - {d}. A withdrawal counts as full only when the whole ${d} can be "
    "paid; the fund stops once it can no longer pay a full withdrawal. How many full "
    "withdrawals does the annuity pay? Give your answer as a whole number of withdrawals.",
    "A retirement fund of ${a0} earns {r}% interest every period, following "
    "A(n+1) = A(n)({gf}) - {d}: interest is added first, then the fixed ${d} withdrawal is "
    "taken. The fund stops once it can no longer pay a full ${d} withdrawal. Find the total "
    "number of full withdrawals the fund makes. Give your answer as a whole number of "
    "withdrawals.",
    "Using the recurrence relation A(n+1) = A(n)({gf}) - {d}, an annuity starts at ${a0}: {r}% "
    "interest is credited each period before the ${d} withdrawal is taken. A full withdrawal is "
    "one where the whole ${d} is paid. Find how many full withdrawals the annuity pays in "
    "total. Give your answer as a whole number of withdrawals.",
)


def _annuity_easy(rng):
    for _ in range(120):
        hundredths = rng.randint(20, 60)
        r = Fraction(hundredths, 10000)
        r_pct = Fraction(hundredths, 100)
        d = _draw_withdrawal(rng)
        n_target = rng.randint(6, 40)
        solved = _solve_a0_for_target(r, d, n_target)
        if solved:
            break
    else:
        raise ValueError("could not construct an easy annuity-duration question")
    a0, n, _b_after = solved
    n_check, _ = _iterate_annuity_reading_b(a0, r, d)
    check_equivalent(n, n_check, "annuity duration: readings (a) and (b) disagree (easy)")

    template = rng.choice(_EASY_TEMPLATES)
    text = template.format(a0=int(a0), r=_fmt_pct(r_pct), gf=_fmt_exact(1 + r, 4), d=int(d))

    answer_text = str(n)
    d_over = n + 1
    d_under = max(0, n - 1)
    d_linear = int(a0 // d)
    raw = [v for v in (d_over, d_under, d_linear) if v != n]
    check_distinct(n, *raw, msg="easy annuity-duration distractor equals the answer")
    distractors = pick_distractors(answer_text, [str(v) for v in raw], n=3)
    return dict(
        text=text,
        answer=n,
        answer_text=answer_text,
        answer_latex=answer_text,
        distractors=distractors,
        working=[
            f"Iterating A(n+1) = A(n)({_fmt_exact(1 + r, 4)}) - {int(d)} from ${int(a0)}, "
            f"the fund pays {n} full ${int(d)} withdrawals before it can no longer pay a "
            f"full withdrawal."
        ],
    )


# =============================================================================
# medium -- annual rate compounded monthly/quarterly, no formula given
# =============================================================================

_MEDIUM_TEMPLATES = (
    "A retirement fund starts with ${a0}, earning {annual}% per annum compounded {freq_adv} "
    "(interest is credited before each withdrawal). A fixed ${d} is withdrawn at the end of "
    "each {period}; a withdrawal counts as full only when the whole ${d} can be paid. Find the "
    "total number of full withdrawals the fund makes. Give your answer as a whole number of "
    "withdrawals.",
    "An annuity of ${a0} earns {annual}% per annum, compounded {freq_adv}, interest being added "
    "before the withdrawal each {period}. A fixed ${d} is withdrawn at the end of every "
    "{period} until a full withdrawal can no longer be paid. How many full withdrawals does the "
    "annuity make in total? Give your answer as a whole number of withdrawals.",
    "${a0} is invested in an annuity earning {annual}% per annum compounded {freq_adv}. At the "
    "end of each {period}, interest is credited and then a fixed ${d} is withdrawn. Find the "
    "number of full ${d} withdrawals the annuity can make before it can no longer pay a full "
    "withdrawal. Give your answer as a whole number of withdrawals.",
)


def _annuity_medium(rng):
    for _ in range(120):
        freq, lo, hi, r, r_pct = _draw_freq_rate(rng)
        d = _draw_withdrawal(rng)
        n_target = rng.randint(6, 40)
        solved = _solve_a0_for_target(r, d, n_target)
        if solved:
            break
    else:
        raise ValueError("could not construct a medium annuity-duration question")
    a0, n, _b_after = solved
    n_check, _ = _iterate_annuity_reading_b(a0, r, d)
    check_equivalent(n, n_check, "annuity duration: readings (a) and (b) disagree (medium)")

    annual_pct = r_pct * freq
    freq_adv, period = _freq_words(freq)
    template = rng.choice(_MEDIUM_TEMPLATES)
    text = template.format(a0=int(a0), annual=_fmt_pct(annual_pct), freq_adv=freq_adv, d=int(d), period=period)

    answer_text = str(n)
    candidates_int = [n + 1, max(0, n - 1), int(a0 // d)]
    wrong_n = _count_or_none(a0, annual_pct / 100, d)  # forgot to divide the annual rate by freq
    if wrong_n is not None:
        candidates_int.append(wrong_n)
    raw = [v for v in candidates_int if v != n]
    check_distinct(n, *raw, msg="medium annuity-duration distractor equals the answer")
    distractors = pick_distractors(answer_text, [str(v) for v in raw], n=3)
    return dict(
        text=text,
        answer=n,
        answer_text=answer_text,
        answer_latex=answer_text,
        distractors=distractors,
        working=[
            f"The per-{period} rate is {_fmt_pct(annual_pct)}%/{freq} = {_fmt_exact(r, 4)}; "
            f"iterating A(n+1) = A(n)(1 + {_fmt_exact(r, 4)}) - {int(d)} from ${int(a0)} "
            f"gives {n} full ${int(d)} withdrawals."
        ],
    )


# =============================================================================
# hard -- the EFFECT of one changed parameter on the duration
# =============================================================================

_HARD_TEMPLATES = {
    "amount": (
        "An annuity earning {annual}% per annum compounded {freq_adv} (interest credited before "
        "each withdrawal) pays a fixed ${d} at the end of every {period}, until a full "
        "withdrawal can no longer be paid. One version starts with ${a0_1} invested; another, "
        "otherwise identical, starts with ${a0_2}. By how many full withdrawals do the two "
        "versions' durations differ? Give your answer as a whole number of withdrawals.",
        "Two annuities earn {annual}% per annum compounded {freq_adv} and both pay a fixed ${d} "
        "at the end of every {period} until a full withdrawal can no longer be paid: one starts "
        "with ${a0_1} invested, the other with ${a0_2}. Find the difference between the two "
        "annuities' total numbers of full withdrawals. Give your answer as a whole number of "
        "withdrawals.",
    ),
    "rate": (
        "An annuity of ${a0} pays a fixed ${d} at the end of every {period} (interest credited "
        "before each withdrawal), until a full withdrawal can no longer be paid. Compare "
        "compounding at {annual1}% per annum against {annual2}% per annum, both compounded "
        "{freq_adv}. By how many full withdrawals do the two rates' durations differ? Give your "
        "answer as a whole number of withdrawals.",
        "Two versions of the same ${a0} annuity, each paying a fixed ${d} at the end of every "
        "{period}, are compounded {freq_adv}: one at {annual1}% per annum, the other at "
        "{annual2}% per annum. Find the difference between the two versions' total numbers of "
        "full withdrawals. Give your answer as a whole number of withdrawals.",
    ),
    "withdrawal": (
        "An annuity of ${a0}, earning {annual}% per annum compounded {freq_adv} (interest "
        "credited before each withdrawal), withdraws a fixed amount at the end of every "
        "{period} until a full withdrawal can no longer be paid. Compare withdrawing ${d1} each "
        "{period} against withdrawing ${d2} each {period}. By how many full withdrawals do the "
        "two withdrawal amounts' durations differ? Give your answer as a whole number of "
        "withdrawals.",
        "The same ${a0} annuity, earning {annual}% per annum compounded {freq_adv}, is compared "
        "under two fixed withdrawal amounts, ${d1} and ${d2}, each paid at the end of every "
        "{period}. Find the difference between the two withdrawal amounts' total numbers of "
        "full withdrawals. Give your answer as a whole number of withdrawals.",
    ),
}


def _hard_scenario(rng):
    for _ in range(200):
        freq, lo, hi, r1, r1_pct = _draw_freq_rate(rng)
        d1 = _draw_withdrawal(rng)
        n1_target = rng.randint(8, 35)
        solved1 = _solve_a0_for_target(r1, d1, n1_target)
        if not solved1:
            continue
        a0_1, n1, _ = solved1
        diff = rng.randint(1, 12)
        sign = rng.choice((1, -1))
        n2_target = n1 + sign * diff
        if n2_target < 6 or n2_target > 50:
            continue
        which = rng.choice(("amount", "rate", "withdrawal"))
        if which == "amount":
            solved2 = _solve_a0_for_target(r1, d1, n2_target)
            if not solved2:
                continue
            a0_2, n2, _ = solved2
            if a0_2 == a0_1:
                continue
            scenario = dict(which=which, freq=freq, r=r1, r_pct=r1_pct, d=d1, a0_1=a0_1, a0_2=a0_2, n1=n1, n2=n2)
        elif which == "withdrawal":
            solved2 = _solve_d_for_target(a0_1, r1, n2_target)
            if not solved2:
                continue
            d2, n2, _ = solved2
            if d2 == d1:
                continue
            scenario = dict(which=which, freq=freq, r=r1, r_pct=r1_pct, a0=a0_1, d1=d1, d2=d2, n1=n1, n2=n2)
        else:
            found = _scan_rate_for_target(a0_1, d1, n2_target, lo, hi)
            if found is None:
                continue
            r2, r2_pct, n2 = found
            if r2 == r1:
                continue
            scenario = dict(
                which=which, freq=freq, a0=a0_1, d=d1, r1=r1, r1_pct=r1_pct, r2=r2, r2_pct=r2_pct, n1=n1, n2=n2
            )
        if abs(scenario["n2"] - scenario["n1"]) < 1:
            continue
        return scenario
    raise ValueError("could not construct a hard annuity-duration effect question")


def _annuity_hard(rng):
    scenario = _hard_scenario(rng)
    freq_adv, period = _freq_words(scenario["freq"])
    which = scenario["which"]
    template = rng.choice(_HARD_TEMPLATES[which])

    if which == "amount":
        annual = scenario["r_pct"] * scenario["freq"]
        text = template.format(
            annual=_fmt_pct(annual),
            freq_adv=freq_adv,
            d=int(scenario["d"]),
            a0_1=int(scenario["a0_1"]),
            a0_2=int(scenario["a0_2"]),
            period=period,
        )
    elif which == "rate":
        annual1 = scenario["r1_pct"] * scenario["freq"]
        annual2 = scenario["r2_pct"] * scenario["freq"]
        lo_annual, hi_annual = sorted((annual1, annual2))
        text = template.format(
            a0=int(scenario["a0"]),
            d=int(scenario["d"]),
            freq_adv=freq_adv,
            annual1=_fmt_pct(lo_annual),
            annual2=_fmt_pct(hi_annual),
            period=period,
        )
    else:
        annual = scenario["r_pct"] * scenario["freq"]
        d_lo, d_hi = sorted((scenario["d1"], scenario["d2"]))
        text = template.format(
            a0=int(scenario["a0"]),
            annual=_fmt_pct(annual),
            freq_adv=freq_adv,
            d1=int(d_lo),
            d2=int(d_hi),
            period=period,
        )

    n1, n2 = scenario["n1"], scenario["n2"]
    diff = abs(n1 - n2)
    answer_text = str(diff)
    candidates_int = [n1, n2, diff + 1, max(0, diff - 1)]
    raw = [v for v in candidates_int if v != diff]
    check_distinct(diff, *raw, msg="hard annuity-duration-effect distractor equals the answer")
    distractors = pick_distractors(answer_text, [str(v) for v in raw], n=3)
    return dict(
        text=text,
        answer=diff,
        answer_text=answer_text,
        answer_latex=answer_text,
        distractors=distractors,
        working=[f"One version pays {n1} full withdrawals, the other {n2}; the durations differ by {diff}."],
    )


# =============================================================================
# exam -- a mid-life change carried forward from the balance actually reached
# =============================================================================

_EXAM_TEMPLATES = {
    "withdrawal": (
        "An annuity begins with ${a0}, earning {annual}% per annum compounded {freq_adv} "
        "(interest credited before each withdrawal). A fixed ${d1} is withdrawn at the end of "
        "each {period}. After the {k_ord} withdrawal, the withdrawal amount changes to ${d2} for "
        "every {period} after that (the rate stays the same); a withdrawal counts as full only "
        "when the full amount can be paid. Carrying the balance forward from the {k_ord} "
        "withdrawal, find the TOTAL number of full withdrawals the annuity makes. Give your "
        "answer as a whole number of withdrawals.",
        "${a0} is invested in an annuity earning {annual}% per annum compounded {freq_adv}, "
        "withdrawing a fixed ${d1} at the end of each {period} for the first {k} withdrawals; "
        "every withdrawal after the {k_ord} is ${d2} instead (the rate is unchanged). Using the "
        "balance carried forward after the {k_ord} withdrawal, find the total number of full "
        "withdrawals the annuity makes in all. Give your answer as a whole number of "
        "withdrawals.",
    ),
    "rate": (
        "An annuity begins with ${a0}, earning {annual1}% per annum compounded {freq_adv} "
        "(interest credited before each withdrawal), withdrawing a fixed ${d} at the end of "
        "each {period}. After the {k_ord} withdrawal, the rate changes to {annual2}% per annum "
        "compounded {freq_adv} for every {period} after that (the withdrawal stays the same); a "
        "withdrawal counts as full only when the full amount can be paid. Carrying the balance "
        "forward from the {k_ord} withdrawal, find the TOTAL number of full withdrawals the "
        "annuity makes. Give your answer as a whole number of withdrawals.",
        "${a0} is invested in an annuity, initially earning {annual1}% per annum compounded "
        "{freq_adv}, withdrawing a fixed ${d} at the end of each {period} for the first {k} "
        "withdrawals; every withdrawal after the {k_ord} earns {annual2}% per annum instead (the "
        "withdrawal is unchanged). Using the balance carried forward after the {k_ord} "
        "withdrawal, find the total number of full withdrawals the annuity makes in all. Give "
        "your answer as a whole number of withdrawals.",
    ),
}


def _exam_scenario(rng):
    for _ in range(200):
        freq, lo, hi, r1, r1_pct = _draw_freq_rate(rng)
        d1 = _draw_withdrawal(rng)
        total_target = rng.randint(10, 40)
        k = rng.randint(2, min(8, total_target - 3))
        n2_target = total_target - k
        n1_virtual = k + rng.randint(4, 15)
        solved1 = _solve_a0_for_target(r1, d1, n1_virtual)
        if not solved1:
            continue
        a0, n1_virtual_actual, _ = solved1
        if n1_virtual_actual < k + 1:
            continue
        # Forward-iterate the k REAL withdrawals under phase-1 terms, keeping
        # every intermediate margin (guaranteed: k < n1_virtual_actual, and
        # the balance is strictly decreasing, so every A(i) for i < n1_virtual
        # clears the same margin the construction already proved at n1_virtual).
        b = Fraction(a0)
        for _ in range(k):
            b = b * (1 + r1) - d1
        a_k = b

        which = rng.choice(("withdrawal", "rate"))
        if which == "withdrawal":
            solved2 = _solve_d_for_target(a_k, r1, n2_target)
            if not solved2:
                continue
            d2, n2, _ = solved2
            if d2 == d1:
                continue
            scenario = dict(
                which=which,
                freq=freq,
                a0=a0,
                r=r1,
                r_pct=r1_pct,
                d1=d1,
                d2=d2,
                k=k,
                k_ord=ordinal(k),
                n2=n2,
                n1_virtual=n1_virtual_actual,
            )
        else:
            found = _scan_rate_for_target(a_k, d1, n2_target, lo, hi)
            if found is None:
                continue
            r2, r2_pct, n2 = found
            if r2 == r1:
                continue
            scenario = dict(
                which=which,
                freq=freq,
                a0=a0,
                d=d1,
                r1=r1,
                r1_pct=r1_pct,
                r2=r2,
                r2_pct=r2_pct,
                k=k,
                k_ord=ordinal(k),
                n2=n2,
                n1_virtual=n1_virtual_actual,
            )
        total = k + n2
        if total < 6:
            continue
        scenario["total"] = total
        return scenario
    raise ValueError("could not construct an exam annuity-duration mid-life-change question")


def _annuity_exam(rng):
    scenario = _exam_scenario(rng)
    freq_adv, period = _freq_words(scenario["freq"])
    which = scenario["which"]
    template = rng.choice(_EXAM_TEMPLATES[which])
    k = scenario["k"]
    total = scenario["total"]
    answer_text = str(total)

    if which == "withdrawal":
        annual = scenario["r_pct"] * scenario["freq"]
        text = template.format(
            a0=int(scenario["a0"]),
            annual=_fmt_pct(annual),
            freq_adv=freq_adv,
            d1=int(scenario["d1"]),
            d2=int(scenario["d2"]),
            k=k,
            k_ord=ordinal(k),
            period=period,
        )
        no_change = scenario["n1_virtual"]
        restart_n = _count_or_none(scenario["a0"], scenario["r"], scenario["d2"])
    else:
        annual1 = scenario["r1_pct"] * scenario["freq"]
        annual2 = scenario["r2_pct"] * scenario["freq"]
        text = template.format(
            a0=int(scenario["a0"]),
            annual1=_fmt_pct(annual1),
            annual2=_fmt_pct(annual2),
            d=int(scenario["d"]),
            k=k,
            k_ord=ordinal(k),
            period=period,
            freq_adv=freq_adv,
        )
        no_change = scenario["n1_virtual"]
        restart_n = _count_or_none(scenario["a0"], scenario["r2"], scenario["d"])

    candidates_int = [no_change]
    if restart_n is not None:
        candidates_int.append(k + restart_n)
    candidates_int += [total + 1, max(0, total - 1)]
    raw = [v for v in candidates_int if v != total]
    check_distinct(total, *raw, msg="exam annuity-duration distractor equals the answer")
    distractors = pick_distractors(answer_text, [str(v) for v in raw], n=3)
    return dict(
        text=text,
        answer=total,
        answer_text=answer_text,
        answer_latex=answer_text,
        distractors=distractors,
        working=[
            f"After {k} withdrawals under the original terms, the carried balance continues "
            f"under the new terms for {scenario['n2']} more full withdrawals: "
            f"{k} + {scenario['n2']} = {total} in total."
        ],
    )


_PRESETS = {
    "easy": {"variant": "easy"},
    "medium": {"variant": "medium"},
    "hard": {"variant": "hard"},
    "exam": {"variant": "exam"},
}


@generator("pv0a_annuity_duration", level=13, presets=_PRESETS)
def annuity_duration(rng, variant="easy"):
    if variant == "easy":
        return _annuity_easy(rng)
    if variant == "medium":
        return _annuity_medium(rng)
    if variant == "hard":
        return _annuity_hard(rng)
    if variant == "exam":
        return _annuity_exam(rng)
    raise ValueError(f"unknown variant: {variant!r}")
