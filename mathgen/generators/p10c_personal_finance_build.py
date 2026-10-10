"""Personal budget + government allowances and pensions.

Two generators, both senior ``acara-senior-v8-4``, domain ``arithmetic``,
``answer: number``:

- ``p10c_personal_budget`` (``senior-prepare-a-personal-budget-for-a-given-
  income``, ACMGM004) -- the dollar amount of weekly income remaining after
  fixed and discretionary spending. easy: two fixed expenses, subtract their
  sum. medium: three fixed expenses plus a two-item discretionary total, two
  separate category sums then subtract both. hard: fixed expenses plus
  discretionary spending stated as a PERCENTAGE of income, folding a
  percentage-of-income calculation into the same structure.

- ``p10c_government_allowance`` (``senior-calculate-payments-based-on-
  government-allowances-and-pensions``, ACMGM003) -- a fortnightly payment
  computed from a stated, explicitly hypothetical support-payment scheme's
  rules. easy: base rate plus a per-dependent-child rate. medium: adds a
  supplement stated as a WEEKLY rate that must be converted to fortnightly.
  hard: adds an income test -- a taper reducing the payment above an
  income-free threshold, constructed so the excess is an exact multiple of
  the taper step.

BOUNDARY -- ``p10c_government_allowance`` never encodes a real government
scheme's payment rate anywhere in this file. Every base rate, per-dependent
rate, supplement, threshold, taper step and taper rate is drawn from an
arbitrary numeric RANGE and stated fresh in the question text; every question
frames the scheme as an explicitly hypothetical "support payment scheme",
never named after or matched to an actual, dateable entitlement.

Both skills use exact integer-cents arithmetic throughout -- every income,
percentage and rate is a whole number of dollars or a whole percent, so no
generator here ever rounds a division forward to build its own canonical
answer (matching the ``p7d_rates_build.py`` precedent this module follows for
its currency answer shape: ``answer=<numeric>, answer_text=<"$..." string>,
answer_latex=<same string>``, via ``mathgen.core.currency_from_cents``).
"""

from __future__ import annotations

from fractions import Fraction

from mathgen.core import (
    VerificationError,
    check,
    check_distinct,
    check_equivalent,
    check_pairwise_distinct,
    currency_from_cents,
    generator,
    latex_answer,
    numeric_answer,
    pick_distractors,
)

_RESAMPLE_ATTEMPTS = 200


def _currency(dollars) -> str:
    """Whole dollars -> "$X.YY" via the shared cents formatter."""
    return currency_from_cents(int(dollars) * 100)


def _cents_currency(cents: int) -> str:
    return currency_from_cents(int(cents))


# =============================================================================
# Skill 1: p10c_personal_budget (ACMGM004)
# =============================================================================

_FIXED_LABELS = ["rent", "utility bills", "insurance"]
_DISCRETIONARY_LABELS = ["entertainment", "eating out"]


def _fixed_expenses_phrase(values: list[int]) -> str:
    parts = [f"{_currency(v)} for {label}" for v, label in zip(values, _FIXED_LABELS)]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return f"{parts[0]}, {parts[1]} and {parts[2]}"


def _verify_budget_easy_medium(I: int, fixed: list[int], discretionary: int, answer: int):
    """Independent check: recompute total spending as ONE combined figure via
    Python's ``sum()`` over the fixed-item list (construction instead
    accumulates a running total, in draw order, in a loop) plus the
    discretionary figure, then re-subtract from income -- a genuinely
    different route from the running-total accumulation construction uses."""
    total_spending = sum(fixed) + discretionary
    check_equivalent(I - total_spending, answer, "remaining amount must equal income minus total spending")


def _budget_easy(rng):
    for _ in range(_RESAMPLE_ATTEMPTS):
        I = rng.randint(300, 900)
        fixed = [rng.randint(20, 150), rng.randint(20, 150)]
        running_total = 0
        for f in fixed:
            running_total += f
        F = running_total
        answer = I - F
        if answer < 10:
            continue
        check(answer < 1000, "easy remaining must stay under $1000")

        forgot_last = I - fixed[0]  # drops the LAST fixed expense from the sum
        added_instead = I + F  # added instead of subtracted
        try:
            check_pairwise_distinct(answer, forgot_last, added_instead)
        except VerificationError:
            continue
        break
    else:
        raise ValueError("p10c_personal_budget: no collision-free easy draw found")

    _verify_budget_easy_medium(I, fixed, 0, answer)

    text = (
        f"A person earns {_currency(I)} per week. Their fixed weekly expenses are "
        f"{_fixed_expenses_phrase(fixed)}. How much of their weekly income remains "
        f"after paying these fixed expenses?"
    )
    answer_text = _currency(answer)
    candidates = [_currency(forgot_last), _currency(added_instead)]
    distractors = pick_distractors(answer_text, candidates, n=2)
    check(len(distractors) == 2, "easy tier expects 2 distinct distractors")

    return dict(
        text=text,
        latex=text,
        answer=answer,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text, plain=True),
        working=[
            f"Remaining = income - fixed expenses = {_currency(I)} - "
            f"({' + '.join(_currency(v) for v in fixed)}) = {answer_text}."
        ],
        distractors=distractors,
    )


def _budget_medium(rng):
    for _ in range(_RESAMPLE_ATTEMPTS):
        I = rng.randint(300, 900)
        fixed = [rng.randint(20, 150) for _ in range(3)]
        running_total = 0
        for f in fixed:
            running_total += f
        F = running_total
        disc_items = [rng.randint(10, 80), rng.randint(10, 80)]
        D = disc_items[0] + disc_items[1]
        answer = I - F - D
        if answer < 10:
            continue
        check(answer < 1000, "medium remaining must stay under $1000")

        forgot_last = I - (fixed[0] + fixed[1]) - D  # drops the LAST fixed expense
        added_instead = I + F + D  # added instead of subtracted
        try:
            check_pairwise_distinct(answer, forgot_last, added_instead)
        except VerificationError:
            continue
        break
    else:
        raise ValueError("p10c_personal_budget: no collision-free medium draw found")

    _verify_budget_easy_medium(I, fixed, D, answer)

    text = (
        f"A person earns {_currency(I)} per week. Their fixed weekly expenses are "
        f"{_fixed_expenses_phrase(fixed)}. They also budget {_currency(disc_items[0])} for "
        f"{_DISCRETIONARY_LABELS[0]} and {_currency(disc_items[1])} for {_DISCRETIONARY_LABELS[1]}. "
        f"How much of their weekly income remains for savings after these fixed and "
        f"discretionary expenses?"
    )
    answer_text = _currency(answer)
    candidates = [_currency(forgot_last), _currency(added_instead)]
    distractors = pick_distractors(answer_text, candidates, n=2)
    check(len(distractors) == 2, "medium tier expects 2 distinct distractors")

    return dict(
        text=text,
        latex=text,
        answer=answer,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text, plain=True),
        working=[
            f"Fixed total = {' + '.join(_currency(v) for v in fixed)} = {_currency(F)}. "
            f"Discretionary total = {_currency(disc_items[0])} + {_currency(disc_items[1])} = "
            f"{_currency(D)}. Remaining = {_currency(I)} - {_currency(F)} - {_currency(D)} = "
            f"{answer_text}."
        ],
        distractors=distractors,
    )


def _verify_budget_hard(I: int, fixed: list[int], p: int, answer_cents: int):
    """Independent check, two routes:

    1. Recompute total spending as one combined figure via ``sum()`` over the
       fixed-item list (construction uses a running total) plus the
       discretionary figure, then re-subtract from income, all in exact
       cents via ``Fraction`` -- never float.
    2. Recompute the discretionary amount by an entirely separate symbolic
       route: convert ``p`` percent to ``Fraction(p, 100)`` and multiply by
       the income in cents, then assert the result equals ``I * p`` cents
       exactly (the identity the construction itself relies on, proved a
       second way).
    """
    F = sum(fixed)
    disc_cents_route1 = I * p
    total_spending_cents = F * 100 + disc_cents_route1
    check_equivalent(
        Fraction(I * 100 - total_spending_cents),
        Fraction(answer_cents),
        "hard remaining (cents) must equal income minus total spending",
    )

    I_cents = Fraction(I * 100)
    disc_cents_route2 = Fraction(p, 100) * I_cents
    check_equivalent(disc_cents_route2, Fraction(I * p), "discretionary cents identity must hold exactly")


def _budget_hard(rng):
    for _ in range(_RESAMPLE_ATTEMPTS):
        I = rng.randint(300, 900)
        num_fixed = rng.choice([2, 3])
        fixed = [rng.randint(20, 150) for _ in range(num_fixed)]
        running_total = 0
        for f in fixed:
            running_total += f
        F = running_total
        p = rng.choice([10, 15, 20, 25, 30])
        disc_cents = I * p  # exact identity: whole-dollar income x whole percent = whole cents
        answer_cents = I * 100 - F * 100 - disc_cents
        if answer_cents < 1000:  # < $10
            continue
        check(answer_cents < 100000, "hard remaining must stay under $1000")

        forgot_last_cents = answer_cents + fixed[-1] * 100
        added_instead_cents = I * 100 + F * 100 + disc_cents
        remainder_after_fixed = I - F
        disc3_cents = remainder_after_fixed * p  # percentage-of-remainder
        pct_of_remainder_cents = I * 100 - F * 100 - disc3_cents
        percent_as_dollars_cents = I * 100 - F * 100 - p * 100  # percent-read-as-dollars

        # Algebraic guards (state the reason, don't just assert-and-hope):
        # percentage-of-remainder differs from the correct discretionary base
        # by exactly F*p cents (I*p - (I-F)*p = F*p), always > 0 since F >= 20.
        check(
            pct_of_remainder_cents - answer_cents == F * p, "percentage-of-remainder must differ by exactly F*p cents"
        )
        # percent-read-as-dollars differs by exactly p*(I - 100) cents
        # (I*p - p*100 = p*(I-100)), always > 0 since I >= 300 > 100.
        check(
            percent_as_dollars_cents - answer_cents == p * (I - 100),
            "percent-read-as-dollars must differ by exactly p*(I-100) cents",
        )

        try:
            check_pairwise_distinct(
                answer_cents, forgot_last_cents, added_instead_cents, pct_of_remainder_cents, percent_as_dollars_cents
            )
        except VerificationError:
            continue
        break
    else:
        raise ValueError("p10c_personal_budget: no collision-free hard draw found")

    _verify_budget_hard(I, fixed, p, answer_cents)

    text = (
        f"A person earns {_currency(I)} per week. Their fixed weekly expenses are "
        f"{_fixed_expenses_phrase(fixed)}. They budget {p}% of their income for discretionary "
        f"spending. How much of their weekly income remains for savings after these fixed "
        f"expenses and their discretionary budget?"
    )
    answer_text = _cents_currency(answer_cents)
    candidates = [
        _cents_currency(forgot_last_cents),
        _cents_currency(added_instead_cents),
        _cents_currency(pct_of_remainder_cents),
        _cents_currency(percent_as_dollars_cents),
    ]
    distractors = pick_distractors(answer_text, candidates, n=4)
    check(len(distractors) == 4, "hard tier expects 4 distinct distractors")

    return dict(
        text=text,
        latex=text,
        answer=numeric_answer(Fraction(answer_cents, 100)),
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text, plain=True),
        working=[
            f"Fixed total = {_currency(F)}. Discretionary = {p}% of {_currency(I)} = "
            f"{_cents_currency(disc_cents)}. Remaining = {_currency(I)} - {_currency(F)} - "
            f"{_cents_currency(disc_cents)} = {answer_text}."
        ],
        distractors=distractors,
    )


@generator(
    "p10c_personal_budget",
    level=11,
    presets={
        "easy": dict(mode="easy"),
        "medium": dict(mode="medium"),
        "hard": dict(mode="hard"),
    },
)
def p10c_personal_budget(rng, mode="easy"):
    """The dollar amount of weekly income remaining after fixed and
    discretionary spending are accounted for. easy: two fixed expenses,
    subtract their sum. medium: three fixed expenses plus a two-item
    discretionary total, two separate category sums then subtract both.
    hard: discretionary spending stated as a percentage of income, folding a
    percentage-of-income calculation into the same structure."""
    if mode == "easy":
        return _budget_easy(rng)
    if mode == "medium":
        return _budget_medium(rng)
    if mode == "hard":
        return _budget_hard(rng)
    raise ValueError(f"unknown mode: {mode!r}")


# =============================================================================
# Skill 2: p10c_government_allowance (ACMGM003)
# =============================================================================


def _child_noun(n: int) -> str:
    return "dependent child" if n == 1 else "dependent children"


def _verify_allowance_easy_medium(B: int, C: int, n: int, W: int, answer: int):
    """Independent check: recompute the total in the OPPOSITE grouping from
    construction (which accumulates B, then += C*n, then += 2*W) -- here via
    Python's ``sum()`` over a list built in reverse order. For the medium
    tier (``W`` nonzero) also recompute the weekly-to-fortnightly conversion
    by repeated addition (``W + W``) rather than multiplication (``2 * W``),
    a genuinely different arithmetic route."""
    terms_reverse = [2 * W, C * n, B]
    check_equivalent(sum(terms_reverse), answer, "allowance total (reverse grouping) must match")
    if W:
        check_equivalent(W + W, 2 * W, "weekly-to-fortnightly conversion (repeated addition) must match")


def _allowance_easy(rng):
    for _ in range(_RESAMPLE_ATTEMPTS):
        B = rng.randint(300, 600)
        C = rng.randint(20, 60)
        n = rng.randint(1, 4)
        answer = B + C * n
        forgot_dependent = B  # drops the dependent-child term entirely
        try:
            check_distinct(answer, forgot_dependent)
        except VerificationError:
            continue
        break
    else:
        raise ValueError("p10c_government_allowance: no collision-free easy draw found")

    _verify_allowance_easy_medium(B, C, n, 0, answer)

    text = (
        f"A hypothetical government support payment scheme pays a base rate of {_currency(B)} per "
        f"fortnight, plus {_currency(C)} for each dependent child. Calculate the fortnightly payment "
        f"for a person with {n} {_child_noun(n)}."
    )
    answer_text = _currency(answer)
    candidates = [_currency(forgot_dependent)]
    distractors = pick_distractors(answer_text, candidates, n=1)
    check(
        len(distractors) == 1,
        "easy tier's construction genuinely supports only 1 distractor (forgot-dependent-supplement)",
    )

    return dict(
        text=text,
        latex=text,
        answer=answer,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text, plain=True),
        working=[
            f"Payment = base rate + (rate per child x children) = {_currency(B)} + "
            f"({_currency(C)} x {n}) = {answer_text}."
        ],
        distractors=distractors,
    )


def _allowance_medium(rng):
    for _ in range(_RESAMPLE_ATTEMPTS):
        B = rng.randint(300, 600)
        C = rng.randint(20, 60)
        n = rng.randint(1, 4)
        W = rng.randint(10, 40)
        if W == C * n:
            continue  # forgot-dependent-supplement (B+2W) would collide with forgot-unit-conversion (B+Cn+W)
        answer = B + C * n + 2 * W
        forgot_dependent = B + 2 * W  # drops the dependent-child term
        forgot_conversion = B + C * n + W  # uses the weekly figure directly, no x2
        try:
            check_pairwise_distinct(answer, forgot_dependent, forgot_conversion)
        except VerificationError:
            continue
        break
    else:
        raise ValueError("p10c_government_allowance: no collision-free medium draw found")

    _verify_allowance_easy_medium(B, C, n, W, answer)

    text = (
        f"A hypothetical government support payment scheme pays a base rate of {_currency(B)} per "
        f"fortnight, plus {_currency(C)} for each dependent child, plus a weekly energy supplement "
        f"of {_currency(W)}, which this person is eligible to receive. Calculate the FORTNIGHTLY "
        f"payment for a person with {n} {_child_noun(n)}."
    )
    answer_text = _currency(answer)
    candidates = [_currency(forgot_dependent), _currency(forgot_conversion)]
    distractors = pick_distractors(answer_text, candidates, n=2)
    check(len(distractors) == 2, "medium tier expects 2 distinct distractors")

    return dict(
        text=text,
        latex=text,
        answer=answer,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text, plain=True),
        working=[
            f"Fortnightly supplement = 2 x {_currency(W)} = {_currency(2 * W)}. Payment = "
            f"{_currency(B)} + ({_currency(C)} x {n}) + {_currency(2 * W)} = {answer_text}."
        ],
        distractors=distractors,
    )


def _verify_allowance_hard(E: int, T: int, s: int, r: int, reduction: int):
    """Independent check: recompute the reduction by COUNTING complete
    increments of `s` via repeated subtraction from (E - T) -- a genuinely
    different computational route from construction's floor-division-then-
    multiply -- and separately assert the exact-multiple guarantee directly."""
    check((E - T) % s == 0, "hard tier construction must land on an exact multiple of s")
    remaining = E - T
    count = 0
    while remaining - s >= 0:
        remaining -= s
        count += 1
    check_equivalent(count * r, reduction, "reduction must equal the repeated-subtraction count times r")


def _allowance_hard(rng):
    for _ in range(_RESAMPLE_ATTEMPTS):
        B = rng.randint(300, 600)
        C = rng.randint(20, 60)
        n = rng.randint(1, 4)
        T = rng.randint(200, 500)
        s = rng.randint(50, 100)
        r = rng.randint(10, 40)
        k = rng.randint(1, 4)  # small k -- keeps E under $1000 and the taper modest
        E = T + k * s
        if E >= 1000:
            continue

        base_payment = B + C * n
        reduction = k * r
        if not (0 <= reduction < base_payment):
            continue
        answer = base_payment - reduction

        forgot_dependent = B - reduction  # drops the dependent term
        if forgot_dependent < 0:
            continue  # a negative "payment" distractor is not a sensible currency figure
        added_instead = base_payment + reduction  # added instead of subtracted
        t_div_s = T // s
        reduction_full_income = (k + t_div_s) * r  # tapers the WHOLE income E
        # Algebraic guard: reduction_full_income - reduction == t_div_s * r,
        # and t_div_s >= 2 always (T >= 200, s <= 100), so this is never zero.
        check(
            reduction_full_income - reduction == t_div_s * r,
            "reduction-on-full-income must differ by exactly (T // s) * r",
        )
        check(t_div_s >= 2, "T // s must be at least 2 by construction (T >= 200, s <= 100)")
        reduction_on_full_income = base_payment - reduction_full_income
        if reduction_on_full_income < 0:
            continue  # same guard for the other subtractive distractor

        try:
            check_pairwise_distinct(answer, forgot_dependent, added_instead, reduction_on_full_income)
        except VerificationError:
            continue
        break
    else:
        raise ValueError("p10c_government_allowance: no collision-free hard draw found")

    _verify_allowance_hard(E, T, s, r, reduction)

    text = (
        f"A hypothetical government support payment scheme pays a base rate of {_currency(B)} per "
        f"fortnight, plus {_currency(C)} for each dependent child. The payment is reduced by "
        f"{_currency(r)} for every complete {_currency(s)} this person earns above a fortnightly "
        f"income-free threshold of {_currency(T)}. This person has {n} {_child_noun(n)} and earns "
        f"{_currency(E)} per fortnight from part-time work. Calculate their fortnightly payment."
    )
    answer_text = _currency(answer)
    candidates = [_currency(forgot_dependent), _currency(reduction_on_full_income), _currency(added_instead)]
    distractors = pick_distractors(answer_text, candidates, n=3)
    check(len(distractors) == 3, "hard tier expects 3 distinct distractors")

    return dict(
        text=text,
        latex=text,
        answer=answer,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text, plain=True),
        working=[
            f"Base payment = {_currency(B)} + ({_currency(C)} x {n}) = {_currency(base_payment)}. "
            f"Excess over threshold = {_currency(E)} - {_currency(T)} = {_currency(E - T)}, which "
            f"is {k} complete {_currency(s)} increments; reduction = {k} x {_currency(r)} = "
            f"{_currency(reduction)}. Payment = {_currency(base_payment)} - {_currency(reduction)} "
            f"= {answer_text}."
        ],
        distractors=distractors,
    )


@generator(
    "p10c_government_allowance",
    level=11,
    presets={
        "easy": dict(mode="easy"),
        "medium": dict(mode="medium"),
        "hard": dict(mode="hard"),
    },
)
def p10c_government_allowance(rng, mode="medium"):
    """A fortnightly payment computed from a stated, explicitly hypothetical
    support-payment scheme's rules. easy: base rate plus a per-dependent-
    child rate. medium: adds a supplement stated as a WEEKLY rate that must
    be converted to fortnightly. hard: adds an income test -- a taper
    reducing the payment above an income-free threshold, constructed so the
    excess is an exact multiple of the taper step."""
    if mode == "easy":
        return _allowance_easy(rng)
    if mode == "medium":
        return _allowance_medium(rng)
    if mode == "hard":
        return _allowance_hard(rng)
    raise ValueError(f"unknown mode: {mode!r}")
