"""The first whole year a compound-interest balance overtakes a simple-interest one.

One skill: ``senior-use-technology-to-compare-numerically-and-graphically-the``
(ACMEM172), ``domain: financial-discrete``, ``answer: number`` (exact whole
year), ``prerequisites: [simple-interest]``.

Scope: given a principal, a simple interest rate and a
compound interest rate, find the first whole year in which the
compound-interest balance exceeds the simple-interest balance. The
graphical comparison is dropped; the numeric
comparison alone is graded.

Every prompt states: "balance at the end of year n" (or "amount owed" for a
loan), the compounding frequency, that no money is added or withdrawn, and a
horizon (within the first 40 years) the crossover always lies inside.

Four tiers, genuine reasoning escalation:

- ``same_principal`` (easy): the same principal, annual compounding, the
  compound rate strictly below the simple rate. Decision: compare the two
  balances year by year (or solve ``(1+c)^n > 1+sn``) and take the first
  year compound is ahead.
- ``nominal_compounding`` (medium): the compound account compounds monthly
  or quarterly at a stated NOMINAL annual rate; simple interest stays
  annual. Decision: convert the nominal rate to a per-period rate and use
  the correct number of periods per year before comparing.
- ``different_principals`` (hard): the compound account starts with LESS
  money than the simple account. Decision: the principal no longer cancels,
  so both amounts must be carried through the comparison.
- ``recovered_rates`` (exam): neither rate is stated directly -- the simple
  account's yearly dollar interest and the compound account's balance after
  one year are given instead (principals may differ). Decision: recover
  each rate from the growth data before comparing.

Same-principal phrasing (easy/medium) always says the amount goes into
EACH of the two accounts, never ambiguously "into two accounts". A loan context ("owes"/"loan"/"repay") is only plausible
when the crossover year is <= 20 years -- a debt left unpaid for 30+ years
is not one a teacher accepts -- so the hard and exam tiers draw their
template from an investment-context pool once the constructed crossover
year exceeds 20, and from either pool (by random choice) when it does not.

Construct-to-target throughout: the crossover year is drawn FIRST from a
wide range, then a compatible rate pair (and, for the harder tiers,
principal pair) is searched for by exact ``Fraction`` arithmetic -- every
percentage is a half-percent multiple, so every balance and every
comparison is exact, never a float. The near-tie guard (compound ahead by
at least $1 at year n, simple ahead by at least $1 at year n-1, and no
earlier year has compound >= simple) is checked on every accepted draw by
construction of the search itself: `_first_crossover_year` scans upward
from year 1 and returns the FIRST year compound exceeds simple, so "no
earlier year" is automatic, and the margin is checked explicitly before a
candidate is accepted.
"""

from __future__ import annotations

from fractions import Fraction

from mathgen.core import (
    check,
    check_equivalent,
    generator,
    pick_distractors,
)

_RESAMPLE_ATTEMPTS = 250
_TARGET_ATTEMPTS = 60
_HORIZON = 45  # scan cap; every stated horizon (40) sits well inside this


def _year_noun(n: int) -> str:
    return "year" if n == 1 else "years"


def _compound_factor(rate_pct: Fraction, periods: int) -> Fraction:
    """(1 + rate_pct/100) ** periods, exact -- rate_pct is a per-PERIOD rate."""
    return (1 + rate_pct / 100) ** periods


def _simple_factor(rate_pct: Fraction, years: int) -> Fraction:
    return 1 + rate_pct * years / 100


def _first_crossover_year(
    Ps: Fraction,
    s_pct: Fraction,
    Pc: Fraction,
    per_period_pct: Fraction,
    periods_per_year: int,
    horizon: int = _HORIZON,
) -> int | None:
    """Smallest whole year y in [1, horizon] at which the compound balance
    (principal Pc, per-period rate per_period_pct, periods_per_year periods
    each year) exceeds the simple balance (principal Ps, annual rate
    s_pct). None if no such year exists within the horizon. Because this
    scans upward from year 1 and returns on the FIRST hit, "no year before
    the returned one has compound >= simple" holds by construction."""
    for y in range(1, horizon + 1):
        compound_val = Pc * _compound_factor(per_period_pct, periods_per_year * y)
        simple_val = Ps * _simple_factor(s_pct, y)
        if compound_val > simple_val:
            return y
    return None


def _margins_ok(
    Ps: Fraction,
    s_pct: Fraction,
    Pc: Fraction,
    per_period_pct: Fraction,
    periods_per_year: int,
    year: int,
) -> bool:
    """At `year`, compound exceeds simple by >= $1; at `year - 1`, simple
    exceeds compound by >= $1 (the near-tie guard, checked every seed)."""
    cv_n = Pc * _compound_factor(per_period_pct, periods_per_year * year)
    sv_n = Ps * _simple_factor(s_pct, year)
    if cv_n - sv_n < 1:
        return False
    if year > 1:
        cv_p = Pc * _compound_factor(per_period_pct, periods_per_year * (year - 1))
        sv_p = Ps * _simple_factor(s_pct, year - 1)
        if sv_p - cv_p < 1:
            return False
    return True


def _locate_crossover_float(
    Ps: Fraction,
    s_pct: Fraction,
    Pc: Fraction,
    per_period_pct: Fraction,
    periods_per_year: int,
    horizon: int = _HORIZON,
) -> int | None:
    """Fast float-only LOCATOR for the search loop: same comparison as
    `_first_crossover_year`, but with plain floats instead of exact
    `Fraction` powers, so a monthly-compounding scan (540 periods at year
    45) costs microseconds instead of milliseconds. This is a candidate
    finder ONLY -- every accepted draw is re-confirmed exactly by
    `_margins_ok` (Fraction, years m-1 and m) before it is used, and by a
    single full exact `_first_crossover_year` scan via `check_equivalent`
    once the draw is accepted, so a float rounding slip can only ever cause
    a wasted search attempt, never a wrong shipped answer.

    "No year before the returned one has compound >= simple" holds by the
    same convexity argument as `_first_crossover_year`: f(n) = Pc*(1+c)^n -
    Ps*(1+sn) is convex in n, and f(0) = Pc - Ps <= 0 whenever Pc <= Ps (the
    same-principal and hard/exam tiers) or Pc == Ps (easy/medium). A convex
    function that is <= 0 at n=0 and crosses zero can cross at most once
    upward, so confirming f(m-1) <= 0 < f(m) EXACTLY (which `_margins_ok`
    does, with a $1 margin) is sufficient to rule out any earlier crossing
    -- it is not merely evidence, it is the proof, given convexity."""
    Psf, sf, Pcf, cf = float(Ps), float(s_pct), float(Pc), float(per_period_pct)
    for y in range(1, horizon + 1):
        compound_val = Pcf * (1 + cf / 100) ** (periods_per_year * y)
        simple_val = Psf * (1 + sf * y / 100)
        if compound_val > simple_val:
            return y
    return None


def _drop_other_adjacent(answer: int, adjacent_candidate: int, candidates: list[int]) -> list[int]:
    """fix 1: having chosen ONE of m-1 / m+1 as `adjacent_candidate`, drop the OTHER one from every other
    candidate source too -- a tier misconception (annual_compounding_assumed, principals_cancelled...) can
    independently land on m-1 or m+1 by coincidence, which would silently recreate the "middle of three
    consecutive integers" pattern even though only one adjacent value was deliberately offered."""
    banned = (answer + 1) if adjacent_candidate == answer - 1 else (answer - 1)
    return [c for c in candidates if c != banned]


def _dedupe_keep_order(answer: int, candidates: list[int]) -> list[int]:
    """Keep candidates in priority order, dropping any that equal the
    answer or an earlier-kept candidate -- a fixed priority
    per tier, skipping a candidate that collides: a colliding
    candidate is DROPPED, never a reason to redraw the whole scenario."""
    kept: list[int] = []
    seen = {answer}
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        kept.append(c)
    return kept


def _collides_with_printed(printed_values, *candidates: int) -> bool:
    """Whether any candidate equals a number the prompt itself prints (a
    principal, a rate, the horizon, a growth figure) -- guards lesson 9.
    `printed_values` is the exact set of numbers the chosen template will
    render (compared as `Fraction`, so a rate like 10.5 is compared as a
    value, never confused with a bare "10" or "5")."""
    printed = {Fraction(v) for v in printed_values}
    return any(Fraction(c) in printed for c in candidates)


def _interest_in_year_exceeds(
    Ps: Fraction,
    s_pct: Fraction,
    Pc: Fraction,
    per_period_pct: Fraction,
    periods_per_year: int,
    up_to_year: int,
) -> int | None:
    """First year y (1..up_to_year) at which the COMPOUND account's own
    interest earned DURING year y exceeds the (constant) simple interest
    earned during any one year -- the misconception of comparing yearly
    increments instead of balances."""
    simple_yearly = Ps * s_pct / 100
    prev = Pc
    for y in range(1, up_to_year + 1):
        cur = Pc * _compound_factor(per_period_pct, periods_per_year * y)
        if cur - prev > simple_yearly:
            return y
        prev = cur
    return None


# =============================================================================
# easy: same_principal (annual compounding both sides, same principal)
# =============================================================================

_SAME_PRINCIPAL_TEMPLATES = [
    "An investor deposits ${P} into each of two accounts and makes no further "
    "deposits or withdrawals. Account A earns simple interest at {s}% p.a. "
    "Account B earns compound interest at {c}% p.a., compounded annually. "
    "Within the first 40 years, find the first whole year at the end of which "
    "Account B's balance exceeds Account A's balance.",
    "Nadia invests ${P} in each of two accounts, with no further deposits or "
    "withdrawals. The first pays simple interest at {s}% p.a. The second pays "
    "compound interest at {c}% p.a., compounded annually. Within the first 40 "
    "years, find the first whole year at the end of which the second account's "
    "balance is greater than the first's.",
    "${P} is invested in each of two accounts, with no further deposits or "
    "withdrawals. Account X earns simple interest at {s}% p.a. Account Y earns "
    "compound interest at {c}% p.a., compounded annually. Within the first 40 "
    "years, find the first whole year at the end of which Account Y's balance "
    "exceeds Account X's balance.",
]


def _search_same_principal(rng, n_target: int):
    for _ in range(_RESAMPLE_ATTEMPTS):
        s_idx = rng.randint(4, 30)  # 2.0 .. 15.0 %, half-percent steps
        c_idx = rng.randint(2, s_idx - 1)  # c < s
        s_pct, c_pct = Fraction(s_idx, 2), Fraction(c_idx, 2)
        m = _locate_crossover_float(Fraction(1), s_pct, Fraction(1), c_pct, 1)
        if m != n_target:
            continue
        P = Fraction(rng.randint(1000, 50000))
        if not _margins_ok(P, s_pct, P, c_pct, 1, m):
            continue
        return P, s_pct, c_pct, m
    return None


def _build_same_principal(rng) -> dict:
    for _ in range(_TARGET_ATTEMPTS):
        n_target = rng.randint(3, 38)
        found = _search_same_principal(rng, n_target)
        if found is None:
            continue
        P, s_pct, c_pct, m = found

        s_idx, c_idx = int(s_pct * 2), int(c_pct * 2)
        # Offer at most one, chosen 50/50 -- both are real misconceptions (m-1: the last year simple
        # is still ahead or looks tied; m+1: counting the first FULL year after the crossover
        # instead of the crossover year itself).
        adjacent_candidate = m - 1 if rng.random() < 0.5 else m + 1
        interest_candidate = _interest_in_year_exceeds(P, s_pct, P, c_pct, 1, m)

        raw_candidates = [adjacent_candidate]
        if interest_candidate is not None:
            raw_candidates.append(interest_candidate)
        raw_candidates = _drop_other_adjacent(m, adjacent_candidate, raw_candidates)
        candidates = _dedupe_keep_order(m, raw_candidates)
        candidates = [c for c in candidates if c >= 1]

        s_disp0 = s_idx / 2 if s_idx % 2 else s_idx // 2
        c_disp0 = c_idx / 2 if c_idx % 2 else c_idx // 2
        if not candidates:
            continue
        if _collides_with_printed([int(P), s_disp0, c_disp0, 40], *candidates):
            continue
        break
    else:
        raise ValueError("pv0b_interest_crossover: no collision-free 'same_principal' draw found")

    # Independent route: the same forward scan re-derives m from P, s, c --
    # exact `Fraction` arithmetic, run once (not in the search loop).
    m_check = _first_crossover_year(P, s_pct, P, c_pct, 1)
    check_equivalent(
        Fraction(m_check),
        Fraction(m),
        "same_principal-tier: forward scan must reproduce the constructed crossover year",
    )

    P_int = int(P)
    s_disp = s_idx / 2 if s_idx % 2 else s_idx // 2
    c_disp = c_idx / 2 if c_idx % 2 else c_idx // 2
    template = _SAME_PRINCIPAL_TEMPLATES[rng.randrange(len(_SAME_PRINCIPAL_TEMPLATES))]
    text = template.format(P=P_int, s=s_disp, c=c_disp)
    answer_text = str(m)

    candidate_texts = [str(c) for c in candidates]
    distractors = pick_distractors(answer_text, candidate_texts, n=len(candidate_texts))
    check(len(distractors) >= 1, "same_principal tier expects at least 1 distinct distractor")

    return dict(
        text=text,
        latex=text,
        answer=m,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"Compare balances year by year: Account B's balance is P x (1 + {c_disp}/100)^n, "
            f"Account A's is P x (1 + {s_disp} x n / 100). The first whole year n at which "
            f"Account B's balance exceeds Account A's is n = {m}."
        ],
        distractors=distractors,
    )


# =============================================================================
# medium: nominal_compounding (compound side compounds monthly/quarterly)
# =============================================================================

_FREQ_WORDS = {4: "quarterly", 12: "monthly"}

_NOMINAL_TEMPLATES = [
    "An investor deposits ${P} into each of two accounts and makes no further "
    "deposits or withdrawals. Account A earns simple interest at {s}% p.a. "
    "Account B earns compound interest at a nominal rate of {c}% p.a., "
    "compounded {freq}. Within the first 40 years, find the first whole year "
    "at the end of which Account B's balance exceeds Account A's balance.",
    "Priya invests ${P} in each of two accounts, with no further deposits or "
    "withdrawals. The first pays simple interest at {s}% p.a. The second pays "
    "compound interest at a nominal rate of {c}% p.a., compounded {freq}. "
    "Within the first 40 years, find the first whole year at the end of which "
    "the second account's balance is greater than the first's.",
    "${P} is invested in each of two accounts, with no further deposits or "
    "withdrawals. Account X earns simple interest at {s}% p.a. Account Y earns "
    "compound interest at a nominal rate of {c}% p.a., compounded {freq}. "
    "Within the first 40 years, find the first whole year at the end of which "
    "Account Y's balance exceeds Account X's balance.",
]


def _search_nominal(rng, n_target: int, freq: int):
    for _ in range(_RESAMPLE_ATTEMPTS):
        s_idx = rng.randint(4, 30)
        c_idx = rng.randint(2, 40)  # nominal rate, half-percent steps up to 20%
        s_pct, c_nom = Fraction(s_idx, 2), Fraction(c_idx, 2)
        per_period = c_nom / freq
        m = _locate_crossover_float(Fraction(1), s_pct, Fraction(1), per_period, freq)
        if m != n_target:
            continue
        P = Fraction(rng.randint(1000, 50000))
        if not _margins_ok(P, s_pct, P, per_period, freq, m):
            continue
        return P, s_pct, c_nom, per_period, m
    return None


def _build_nominal_compounding(rng) -> dict:
    freq = rng.choice([4, 12])
    for _ in range(_TARGET_ATTEMPTS):
        n_target = rng.randint(3, 38)
        found = _search_nominal(rng, n_target, freq)
        if found is None:
            continue
        P, s_pct, c_nom, per_period, m = found

        s_idx, c_idx = int(s_pct * 2), int(c_nom * 2)
        adjacent_candidate = m - 1 if rng.random() < 0.5 else m + 1
        interest_candidate = _interest_in_year_exceeds(P, s_pct, P, per_period, freq, m)
        m_annual = _first_crossover_year(Fraction(1), s_pct, Fraction(1), c_nom, 1)
        annual_compounding_assumed = m_annual

        raw_candidates = [adjacent_candidate]
        if interest_candidate is not None:
            raw_candidates.append(interest_candidate)
        if annual_compounding_assumed is not None:
            raw_candidates.append(annual_compounding_assumed)
        raw_candidates = _drop_other_adjacent(m, adjacent_candidate, raw_candidates)
        candidates = _dedupe_keep_order(m, raw_candidates)
        candidates = [c for c in candidates if c >= 1]

        s_disp0 = s_idx / 2 if s_idx % 2 else s_idx // 2
        c_disp0 = c_idx / 2 if c_idx % 2 else c_idx // 2
        if not candidates:
            continue
        if _collides_with_printed([int(P), s_disp0, c_disp0, 40], *candidates):
            continue
        break
    else:
        raise ValueError("pv0b_interest_crossover: no collision-free 'nominal_compounding' draw found")

    m_check = _first_crossover_year(P, s_pct, P, per_period, freq)
    check_equivalent(
        Fraction(m_check),
        Fraction(m),
        "nominal_compounding-tier: forward scan (per-period rate) must reproduce the crossover year",
    )

    P_int = int(P)
    s_disp = s_idx / 2 if s_idx % 2 else s_idx // 2
    c_disp = c_idx / 2 if c_idx % 2 else c_idx // 2
    template = _NOMINAL_TEMPLATES[rng.randrange(len(_NOMINAL_TEMPLATES))]
    text = template.format(P=P_int, s=s_disp, c=c_disp, freq=_FREQ_WORDS[freq])
    answer_text = str(m)

    candidate_texts = [str(c) for c in candidates]
    distractors = pick_distractors(answer_text, candidate_texts, n=len(candidate_texts))
    check(len(distractors) >= 1, "nominal_compounding tier expects at least 1 distinct distractor")

    return dict(
        text=text,
        latex=text,
        answer=m,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"The nominal {c_disp}% p.a. compounded {_FREQ_WORDS[freq]} gives a per-period rate of "
            f"({c_disp}% ÷ {freq}), applied {freq} times a year. Comparing balances year by year, the "
            f"first whole year n at which Account B's balance exceeds Account A's is n = {m}."
        ],
        distractors=distractors,
    )


# =============================================================================
# hard: different_principals (compound side starts with LESS money)
# =============================================================================

_DIFFERENT_PRINCIPALS_LOAN_TEMPLATES = [
    "Yara owes ${Ps} on a simple-interest loan charged at {s}% p.a. Zane owes "
    "${Pc} on a compound-interest loan charged at {c}% p.a., compounded "
    "annually. Neither makes any repayments. Within the first 40 years, find "
    "the first whole year at the end of which Zane owes more than Yara.",
    "A simple-interest loan of ${Ps} is charged at {s}% p.a. A separate "
    "compound-interest loan of ${Pc} is charged at {c}% p.a., compounded "
    "annually. Neither loan is repaid. Within the first 40 years, find the "
    "first whole year at the end of which the amount owed on the "
    "compound-interest loan exceeds the amount owed on the simple-interest "
    "loan.",
    "Deng owes ${Ps} on a simple-interest loan at {s}% p.a. Wei owes ${Pc} on "
    "a compound-interest loan at {c}% p.a., compounded annually. Neither makes "
    "any repayments. Within the first 40 years, find the first whole year at "
    "the end of which Wei owes more than Deng.",
]

# Loan contexts are only plausible when the crossover falls within about 20
# years (a debt left unpaid for 30+ years is not one a teacher accepts); an investment context is used instead for longer
# crossovers, and both contexts remain available (by random choice) for
# crossovers of 20 years or less.
_DIFFERENT_PRINCIPALS_INVESTMENT_TEMPLATES = [
    "Yara invests ${Ps} in an account earning simple interest at {s}% p.a. "
    "Zane invests ${Pc} in a separate account earning compound interest at "
    "{c}% p.a., compounded annually. Neither makes any further deposits or "
    "withdrawals. Within the first 40 years, find the first whole year at the "
    "end of which Zane's balance exceeds Yara's balance.",
    "${Ps} is invested in an account earning simple interest at {s}% p.a. A "
    "separate amount of ${Pc} is invested in an account earning compound "
    "interest at {c}% p.a., compounded annually. Neither account has any "
    "further deposits or withdrawals. Within the first 40 years, find the "
    "first whole year at the end of which the balance of the compound-"
    "interest account exceeds the balance of the simple-interest account.",
    "Deng invests ${Ps} in an account earning simple interest at {s}% p.a. "
    "Wei invests ${Pc} in a separate account earning compound interest at "
    "{c}% p.a., compounded annually. Neither makes any further deposits or "
    "withdrawals. Within the first 40 years, find the first whole year at the "
    "end of which Wei's balance exceeds Deng's balance.",
]


def _search_different_principals(rng, n_target: int):
    for _ in range(_RESAMPLE_ATTEMPTS):
        s_idx = rng.randint(2, 30)
        c_idx = rng.randint(2, 40)
        s_pct, c_pct = Fraction(s_idx, 2), Fraction(c_idx, 2)
        Ps = Fraction(rng.randint(10000, 60000))
        gap = rng.randint(1000, 15000)
        Pc = Ps - gap
        if Pc < 1000:
            continue
        m = _locate_crossover_float(Ps, s_pct, Pc, c_pct, 1)
        if m != n_target:
            continue
        if not _margins_ok(Ps, s_pct, Pc, c_pct, 1, m):
            continue
        return Ps, s_pct, Pc, c_pct, m
    return None


def _build_different_principals(rng) -> dict:
    for _ in range(_TARGET_ATTEMPTS):
        n_target = rng.randint(3, 38)
        found = _search_different_principals(rng, n_target)
        if found is None:
            continue
        Ps, s_pct, Pc, c_pct, m = found

        s_idx, c_idx = int(s_pct * 2), int(c_pct * 2)
        adjacent_candidate = m - 1 if rng.random() < 0.5 else m + 1
        interest_candidate = _interest_in_year_exceeds(Ps, s_pct, Pc, c_pct, 1, m)
        m_cancelled = _first_crossover_year(Ps, s_pct, Ps, c_pct, 1)
        principals_cancelled = m_cancelled

        raw_candidates = [adjacent_candidate]
        if interest_candidate is not None:
            raw_candidates.append(interest_candidate)
        if principals_cancelled is not None:
            raw_candidates.append(principals_cancelled)
        raw_candidates = _drop_other_adjacent(m, adjacent_candidate, raw_candidates)
        candidates = _dedupe_keep_order(m, raw_candidates)
        candidates = [c for c in candidates if c >= 1]

        Ps_int0, Pc_int0 = int(Ps), int(Pc)
        s_disp0 = s_idx / 2 if s_idx % 2 else s_idx // 2
        c_disp0 = c_idx / 2 if c_idx % 2 else c_idx // 2
        if not candidates:
            continue
        if _collides_with_printed([Ps_int0, Pc_int0, s_disp0, c_disp0, 40], *candidates):
            continue
        break
    else:
        raise ValueError("pv0b_interest_crossover: no collision-free 'different_principals' draw found")

    m_check = _first_crossover_year(Ps, s_pct, Pc, c_pct, 1)
    check_equivalent(
        Fraction(m_check),
        Fraction(m),
        "different_principals-tier: forward scan must reproduce the constructed crossover year",
    )

    Ps_int, Pc_int = int(Ps), int(Pc)
    s_disp = s_idx / 2 if s_idx % 2 else s_idx // 2
    c_disp = c_idx / 2 if c_idx % 2 else c_idx // 2
    # Loan contexts are only plausible for a crossover within about 20 years; longer crossovers always use an investment
    # context, both contexts stay available when the crossover is <= 20.
    if m <= 20:
        pool = rng.choice([_DIFFERENT_PRINCIPALS_LOAN_TEMPLATES, _DIFFERENT_PRINCIPALS_INVESTMENT_TEMPLATES])
    else:
        pool = _DIFFERENT_PRINCIPALS_INVESTMENT_TEMPLATES
    template = pool[rng.randrange(len(pool))]
    text = template.format(Ps=Ps_int, Pc=Pc_int, s=s_disp, c=c_disp)
    answer_text = str(m)

    candidate_texts = [str(c) for c in candidates]
    distractors = pick_distractors(answer_text, candidate_texts, n=len(candidate_texts))
    check(len(distractors) >= 1, "different_principals tier expects at least 1 distinct distractor")

    return dict(
        text=text,
        latex=text,
        answer=m,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"The compound amount is {Pc_int} x (1 + {c_disp}/100)^n; the simple amount is "
            f"{Ps_int} x (1 + {s_disp} x n / 100). Because the principals differ, both must be "
            f"carried through the comparison. The first whole year n at which the compound amount "
            f"exceeds the simple amount is n = {m}."
        ],
        distractors=distractors,
    )


# =============================================================================
# exam: recovered_rates (rates derived from stated growth data, not stated)
# =============================================================================

_RECOVERED_RATES_LOAN_TEMPLATES = [
    "Priya's simple-interest loan of ${Ps} accrues ${X} in interest every year. "
    "Owen's compound-interest loan of ${Pc}, compounded annually, grows to "
    "${Y} in total owed after 1 year. Neither makes any repayments. Within the "
    "first 40 years, find the first whole year at the end of which Owen owes "
    "more than Priya.",
    "A simple-interest loan of ${Ps} accrues ${X} in interest every year. A "
    "separate compound-interest loan of ${Pc}, compounded annually, grows to a "
    "total of ${Y} owed after 1 year. Neither loan is repaid. Within the first "
    "40 years, find the first whole year at the end of which the amount owed "
    "on the compound-interest loan exceeds the amount owed on the "
    "simple-interest loan.",
    "Wei's simple-interest loan of ${Ps} accrues ${X} in interest every year. "
    "Zara's compound-interest loan of ${Pc}, compounded annually, grows to "
    "${Y} in total owed after 1 year. Neither makes any repayments. Within the "
    "first 40 years, find the first whole year at the end of which Zara owes "
    "more than Wei.",
]

# Loan contexts are only plausible for a crossover within about 20 years; longer crossovers use an investment context instead.
_RECOVERED_RATES_INVESTMENT_TEMPLATES = [
    "Priya's simple-interest investment of ${Ps} earns ${X} in interest every "
    "year. Owen's compound-interest investment of ${Pc}, compounded annually, "
    "grows to a balance of ${Y} after 1 year. Neither makes any further "
    "deposits or withdrawals. Within the first 40 years, find the first whole "
    "year at the end of which Owen's balance exceeds Priya's balance.",
    "A simple-interest investment of ${Ps} earns ${X} in interest every year. "
    "A separate compound-interest investment of ${Pc}, compounded annually, "
    "grows to a balance of ${Y} after 1 year. Neither investment has any "
    "further deposits or withdrawals. Within the first 40 years, find the "
    "first whole year at the end of which the balance of the "
    "compound-interest investment exceeds the balance of the "
    "simple-interest investment.",
    "Wei's simple-interest investment of ${Ps} earns ${X} in interest every "
    "year. Zara's compound-interest investment of ${Pc}, compounded annually, "
    "grows to a balance of ${Y} after 1 year. Neither makes any further "
    "deposits or withdrawals. Within the first 40 years, find the first whole "
    "year at the end of which Zara's balance exceeds Wei's balance.",
]


def _search_recovered_rates(rng, n_target: int):
    for _ in range(_RESAMPLE_ATTEMPTS):
        s_idx = rng.randint(2, 30)
        c_idx = rng.randint(2, 40)
        s_pct, c_pct = Fraction(s_idx, 2), Fraction(c_idx, 2)
        Ps = Fraction(rng.randint(20, 300) * 200)  # $4000-$60000, multiple of 200 -> X is a whole dollar
        Pc = Fraction(rng.randint(20, 300) * 200)
        if Pc >= Ps:
            continue
        m = _locate_crossover_float(Ps, s_pct, Pc, c_pct, 1)
        if m != n_target:
            continue
        if not _margins_ok(Ps, s_pct, Pc, c_pct, 1, m):
            continue
        X = Ps * s_pct / 100
        Y = Pc * (1 + c_pct / 100)
        if X.denominator != 1 or Y.denominator != 1:
            continue
        return Ps, s_pct, Pc, c_pct, X, Y, m
    return None


def _build_recovered_rates(rng) -> dict:
    for _ in range(_TARGET_ATTEMPTS):
        n_target = rng.randint(3, 38)
        found = _search_recovered_rates(rng, n_target)
        if found is None:
            continue
        Ps, s_pct, Pc, c_pct, X, Y, m = found

        adjacent_candidate = m - 1 if rng.random() < 0.5 else m + 1
        interest_candidate = _interest_in_year_exceeds(Ps, s_pct, Pc, c_pct, 1, m)
        m_cancelled = _first_crossover_year(Ps, s_pct, Ps, c_pct, 1)
        principals_cancelled = m_cancelled

        raw_candidates = [adjacent_candidate]
        if interest_candidate is not None:
            raw_candidates.append(interest_candidate)
        if principals_cancelled is not None:
            raw_candidates.append(principals_cancelled)
        raw_candidates = _drop_other_adjacent(m, adjacent_candidate, raw_candidates)
        candidates = _dedupe_keep_order(m, raw_candidates)
        candidates = [c for c in candidates if c >= 1]

        Ps_int0, Pc_int0, X_int0, Y_int0 = int(Ps), int(Pc), int(X), int(Y)
        if not candidates:
            continue
        if _collides_with_printed([Ps_int0, Pc_int0, X_int0, Y_int0, 40, 1], *candidates):
            continue
        break
    else:
        raise ValueError("pv0b_interest_crossover: no collision-free 'recovered_rates' draw found")

    # Independent route: recover s_pct and c_pct from the stated growth data
    # alone (never trust the construction direction), then re-scan.
    s_pct_recovered = 100 * X / Ps
    c_pct_recovered = 100 * (Y / Pc - 1)
    check_equivalent(s_pct_recovered, s_pct, "recovered_rates-tier: recovered simple rate must reproduce s_pct")
    check_equivalent(c_pct_recovered, c_pct, "recovered_rates-tier: recovered compound rate must reproduce c_pct")
    m_check = _first_crossover_year(Ps, s_pct_recovered, Pc, c_pct_recovered, 1)
    check_equivalent(
        Fraction(m_check),
        Fraction(m),
        "recovered_rates-tier: forward scan on recovered rates must reproduce the crossover year",
    )

    Ps_int, Pc_int, X_int, Y_int = int(Ps), int(Pc), int(X), int(Y)
    if m <= 20:
        pool = rng.choice([_RECOVERED_RATES_LOAN_TEMPLATES, _RECOVERED_RATES_INVESTMENT_TEMPLATES])
    else:
        pool = _RECOVERED_RATES_INVESTMENT_TEMPLATES
    template = pool[rng.randrange(len(pool))]
    text = template.format(Ps=Ps_int, Pc=Pc_int, X=X_int, Y=Y_int)
    answer_text = str(m)

    candidate_texts = [str(c) for c in candidates]
    distractors = pick_distractors(answer_text, candidate_texts, n=len(candidate_texts))
    check(len(distractors) >= 1, "recovered_rates tier expects at least 1 distinct distractor")

    return dict(
        text=text,
        latex=text,
        answer=m,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"Simple rate: {X_int} / {Ps_int} x 100 = {s_pct_recovered.numerator}/{s_pct_recovered.denominator}% "
            f"p.a. Compound rate: ({Y_int} / {Pc_int} - 1) x 100 = "
            f"{c_pct_recovered.numerator}/{c_pct_recovered.denominator}% p.a. Comparing balances year by "
            f"year, the first whole year n at which the compound amount exceeds the simple amount is n = {m}."
        ],
        distractors=distractors,
    )


@generator(
    "pv0b_interest_crossover",
    level=12,
    presets={
        "easy": dict(mode="same_principal"),
        "medium": dict(mode="nominal_compounding"),
        "hard": dict(mode="different_principals"),
        "exam": dict(mode="recovered_rates"),
    },
)
def pv0b_interest_crossover(rng, mode="same_principal"):
    """The first whole year a compound-interest balance overtakes a
    simple-interest one. easy: same principal, annual compounding both
    sides. medium: the compound side compounds monthly/quarterly at a
    stated NOMINAL rate. hard: the compound side starts with LESS
    principal. exam: neither rate is stated directly -- both are recovered
    from stated growth data."""
    if mode == "same_principal":
        return _build_same_principal(rng)
    if mode == "nominal_compounding":
        return _build_nominal_compounding(rng)
    if mode == "different_principals":
        return _build_different_principals(rng)
    if mode == "recovered_rates":
        return _build_recovered_rates(rng)
    raise ValueError(f"unknown mode: {mode!r}")
