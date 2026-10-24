"""Generates `rational_percentage_financial_model` and `ratio_rate_financial_model` questions: money amounts
split by fractions, percentages, ratios and rates.

Two F-10 financial-modelling skills.

- ``rational_percentage_financial_model`` (`f10-rational-percentage-financial-model`,
  AC9M6N09/AC9M7N09/AC9M8N05) -- the core skill is the CHOICE between a fraction
  operation and a percentage operation applied to a stated money amount, and the
  resulting compute. Every seed at every tier mixes both registers (never a
  percentage-only or fraction-only draw), which is what separates this skill from
  `percentage-of-quantity`/`percentage-discount`/`reverse-percentage` (always
  percentage-only) -- see this module's
  docstrings below for the per-tier reasoning escalation.

- ``ratio_rate_financial_model`` (`f10-ratio-rate-financial-model`, AC9M7M06/AC9M8M07)
  -- the core skill is the CHOICE between a ratio-split representation (a fixed
  total divided in a stated proportion) and a rate representation (a per-unit price
  applied to a quantity), in a MONEY context. Separated from `ratio-rate-proportion`
  (abstract "simplify the ratio", no money, no choice) and
  `f10-circle-and-prism-duration-rates` (a fixed time-zone rate, no ratio option, no
  money) by the CHOICE itself and the money framing.

Both skills work entirely in exact integer CENTS (`Fraction`), never floats, and
every displayed money value is formatted with
`mathgen.core.stated_precision_string(value_in_dollars, 2)` so the real grader's
2-decimal-place tolerance (`stated_precision_tolerance(2) == 0.005`) is always safe
against every distractor here (every distractor differs from the answer by at least
one whole cent by construction, checked with `check_distinct`/`check_pairwise_distinct`
unconditionally, every seed).
"""

from __future__ import annotations

from fractions import Fraction

from mathgen.core import (
    check,
    check_distinct,
    check_equivalent,
    check_pairwise_distinct,
    generator,
    latex_answer,
    money_cents_rounded_2dp,
    numeric_answer,
    pick_distractors,
    round_half_up,
    stated_precision_string,
    stated_precision_tolerance,
)
from mathgen.slots import NAMES, OBJECTS

_RESAMPLE_ATTEMPTS = 200
_TOL = stated_precision_tolerance(2)  # 0.005 dollars -- the real grader's 2dp tolerance


def _money(cents: Fraction | int) -> str:
    """Format an exact integer-cents value as a 2dp dollar string via the
    stated-precision helper (never a hand-rolled f-string): the canonical
    answer must always land here on an exact cent (`.denominator == 1` after
    dividing by 100), asserted by the caller before this is reached."""
    cents = Fraction(cents)
    check(cents.denominator == 1, f"{cents} is not an exact integer cents value")
    return stated_precision_string(cents / 100, 2)


def _money_rounded(cents: Fraction) -> str:
    """Like `_money`, but for a DISTRACTOR display value that may not land
    on an exact cent (e.g. a reverse-model misconception's own arithmetic) --
    rounds half-up to the cent, never raises. Never used for a canonical
    answer_text. Delegates to `mathgen.core.money_cents_rounded_2dp` (a
    helper factored out of this module)."""
    return money_cents_rounded_2dp(cents)


def _two_names(rng):
    a, b = rng.sample(NAMES, 2)
    return a, b


#: Plausible things to spend a share of an allowance, or of a household's monthly budget, on.
_ALLOWANCE_ITEMS = ("book", "movie ticket", "board game", "hoodie", "cap", "poster", "headset", "sketchbook")
_BUDGET_ITEMS = ("power bill", "water bill", "phone plan", "internet plan", "insurance premium", "gym membership")


def _pm_item(rng, context: str) -> str:
    """A spending item that makes sense for the context (allowance, earnings or household budget)."""
    return rng.choice(_BUDGET_ITEMS if context == "budget" else _ALLOWANCE_ITEMS)


def _a(item: str) -> str:
    """ "a"/"an" article agreement for an `OBJECTS` singular noun (only
    "apple" among the bank starts with a vowel sound today, but this is
    correct for any future addition too)."""
    return f"an {item}" if item[:1].lower() in "aeiou" else f"a {item}"


def _draw_multiple(rng, lo: int, hi: int, of: int) -> int:
    """A genuine multiple of `of` in `[lo, hi]` -- unlike
    `rng.randrange(lo, hi + 1, of)`, which steps by `of` FROM `lo` and so is
    only a multiple of `of` when `lo` itself already is (the bug this helper
    exists to avoid: a $100 total with 9 ratio parts stepping from 100 by 9
    gives 100, 109, 118, ... -- none of them divisible by 9)."""
    lo_k = -(-lo // of)  # ceiling division
    hi_k = hi // of
    check(lo_k <= hi_k, f"no multiple of {of} in [{lo}, {hi}]")
    return of * rng.randint(lo_k, hi_k)


# =============================================================================
# Skill 1: rational_percentage_financial_model
# =============================================================================

# Fraction registers: denominators restricted to {2, 4, 5, 10} by the
# own construction rule -- since 100 % d == 0 for every d in this set, a
# fraction step applied to ANY whole-cents money amount lands on an exact
# cent with no resampling needed at all (100 / d is always an integer, so
# cents * num / d == cents * num * (100 // d) / 100, an integer number of
# hundredths-of-a-cent... concretely: cents is already a multiple of 100 (a
# whole-dollar draw), so cents * num / d is an integer whenever d | cents,
# which a whole-dollar-multiple-of-100-cents base guarantees for every d
# here). Kept as a live Fraction computation (never shortcut arithmetic) so
# every step is provably exact by construction, not by argument.
_FRACTIONS: dict[str, Fraction] = {
    "1/2": Fraction(1, 2),
    "1/4": Fraction(1, 4),
    "3/4": Fraction(3, 4),
    "1/5": Fraction(1, 5),
    "2/5": Fraction(2, 5),
    "3/5": Fraction(3, 5),
    "4/5": Fraction(4, 5),
    "1/10": Fraction(1, 10),
    "3/10": Fraction(3, 10),
    "7/10": Fraction(7, 10),
    "9/10": Fraction(9, 10),
}
_PERCENTS = (5, 10, 15, 20, 25, 30, 40, 50, 60, 65, 75, 80)

# Three genuinely different sentence templates (different verbs/nouns, not
# just different numbers) so `prompt_shape` sees >= 3 distinct shapes per
# tier -- `mathgen.slots` NAMES/OBJECTS are masked by `prompt_shape`, so only
# the FIXED wording differences below are what create real shape variety.
_PM_CONTEXTS = ("allowance", "earnings", "budget")


def _pm_context_text(
    context: str, name: str, other: str, item: str, M: int, frac_text: str, pct_text: str, asked: str
) -> str:
    """Render one of the three fixed context templates. `asked` is
    "fraction" or "percentage" -- selects which clause's amount is asked
    for."""
    if context == "allowance":
        base = f"{name} has a ${M} allowance. {name} saves {frac_text} of it and spends {pct_text} of it on {_a(item)}."
        tail = f"How much does {name} save?" if asked == "fraction" else f"How much does {name} spend on the {item}?"
        return f"{base} {tail}"
    if context == "earnings":
        base = f"{name} earns ${M} from a weekend job. {name} donates {frac_text} of it to charity and pays {pct_text} of it to {other} as a loan repayment."
        tail = (
            f"How much does {name} donate to charity?"
            if asked == "fraction"
            else f"How much does {name} repay {other}?"
        )
        return f"{base} {tail}"
    if context == "budget":
        base = f"A household budgets ${M} for the month. They spend {frac_text} of it on rent and {pct_text} of it on {_a(item)}."
        tail = "How much do they spend on rent?" if asked == "fraction" else f"How much do they spend on the {item}?"
        return f"{base} {tail}"
    raise ValueError(f"unknown context: {context!r}")


def _pm_single_step_question(rng, *, m_lo: int, m_hi: int) -> dict:
    """easy tier: a fraction clause AND a percentage clause both stated over the same base amount M (mixed every
    seed, never percentage-only or fraction-only), one of the two asked for at random -- the
    fraction-vs-percentage CHOICE is real because both are present in the text and the learner must read which
    one the question asks about, then apply that single operation to M. This is a deliberate, disclosed
    departure from a literal "one operation stated" easy-tier wording: a single isolated percentage
    (or fraction) clause is not distinguishable, in isolation, from
    `percentage-of-quantity`/`percentage-discount`'s own questions -- stating both registers together, every
    seed, is what makes the choice-then-compute kernel exercisable at the easy tier at all."""
    for _ in range(_RESAMPLE_ATTEMPTS):
        M = rng.randrange(m_lo, m_hi + 1, 20)
        frac_key = rng.choice(list(_FRACTIONS))
        pct = rng.choice(_PERCENTS)
        frac_value = _FRACTIONS[frac_key]
        pct_value = Fraction(pct, 100)
        if frac_value == pct_value:
            continue  # op-swap would collide with the answer; resample
        if frac_value + pct_value > 1:
            continue  # both clauses take "of it" from the same M; past 100% is impossible
        M_cents = Fraction(M * 100)
        frac_cents = M_cents * frac_value
        pct_cents = M_cents * pct_value
        check(frac_cents.denominator == 1, "fraction step must land on an exact cent")
        check(pct_cents.denominator == 1, "percent step must land on an exact cent")
        asked = rng.choice(("fraction", "percentage"))
        answer_cents = frac_cents if asked == "fraction" else pct_cents
        swap_cents = pct_cents if asked == "fraction" else frac_cents
        complement_cents = M_cents - answer_cents
        shift_cents = answer_cents * 10
        try:
            check_pairwise_distinct(
                answer_cents,
                swap_cents,
                complement_cents,
                shift_cents,
                msg="easy-tier candidates must be pairwise distinct",
            )
        except Exception:
            continue

        context = rng.choice(_PM_CONTEXTS)
        name, other = _two_names(rng)
        item = _pm_item(rng, context)
        frac_text = frac_key
        pct_text = f"{pct}%"
        text = _pm_context_text(context, name, other, item, M, frac_text, pct_text, asked)

        # In-generator check: re-derive both clause amounts a second way,
        # via plain integer cents arithmetic (M cents * numerator * (100 //
        # denominator) for the fraction step, M * pct for the percent step
        # in cents), never reusing the Fraction-multiplication route above.
        num, den = frac_value.numerator, frac_value.denominator
        frac_cents_2 = Fraction(M * num * (100 // den))
        pct_cents_2 = Fraction(M * pct)
        check_equivalent(frac_cents_2, frac_cents, "fraction step re-derivation must match")
        check_equivalent(pct_cents_2, pct_cents, "percent step re-derivation must match")

        answer_text = _money(answer_cents)
        candidates = [_money(swap_cents), _money(complement_cents), _money(shift_cents)]
        distractors = pick_distractors(answer_text, candidates, n=3, tolerance=_TOL)
        check(len(distractors) >= 2, "expected at least 2 surviving easy-tier distractors")

        return dict(
            text=text,
            answer=numeric_answer(answer_cents / 100),
            answer_text=answer_text,
            answer_latex=latex_answer(answer_text),
            working=[
                f"{frac_text} of ${M} is {_money(frac_cents)}; {pct_text} of ${M} is {_money(pct_cents)}; the question asks for the {asked} amount, {answer_text}."
            ],
            distractors=distractors,
        )
    raise ValueError("rational_percentage_financial_model/easy: no valid draw found")


def _pm_chain_question(rng, *, m_lo: int, m_hi: int) -> dict:
    """medium tier: a genuine two-step chain, one fraction step and one
    percent step, in a randomly drawn order (fraction-then-percent or
    percent-then-fraction) -- the SECOND step always applies to the RUNNING
    total from the first step (never the original M), and the question asks
    for the final (second-step) amount. A learner must track state across
    two different operation kinds in sequence, genuinely different from
    either a percentage-only chain (`percentage-discount` hard) or a bare
    fraction chain."""
    for _ in range(_RESAMPLE_ATTEMPTS):
        M = rng.randrange(m_lo, m_hi + 1, 20)
        frac_key = rng.choice(list(_FRACTIONS))
        pct = rng.choice(_PERCENTS)
        frac_value = _FRACTIONS[frac_key]
        pct_value = Fraction(pct, 100)
        order = rng.choice(("fraction_first", "percent_first"))
        M_cents = Fraction(M * 100)
        if order == "fraction_first":
            step1_cents = M_cents * frac_value
        else:
            step1_cents = M_cents * pct_value
        if step1_cents.denominator != 1:
            continue
        if order == "fraction_first":
            step2_cents = step1_cents * pct_value
        else:
            step2_cents = step1_cents * frac_value
        if step2_cents.denominator != 1:
            continue
        answer_cents = step2_cents

        # missed-step: reports the first-step amount, forgetting to apply
        # the second operation at all.
        missed_step_cents = step1_cents
        # wrong-base: applies the second operation to the ORIGINAL M instead of the running total
        # from step 1.
        if order == "fraction_first":
            wrong_base_cents = M_cents * pct_value
        else:
            wrong_base_cents = M_cents * frac_value
        shift_cents = answer_cents * 10
        try:
            check_pairwise_distinct(
                answer_cents,
                missed_step_cents,
                wrong_base_cents,
                shift_cents,
                msg="medium-tier candidates must be pairwise distinct",
            )
        except Exception:
            continue

        context = rng.choice(_PM_CONTEXTS)
        name, other = _two_names(rng)
        item = _pm_item(rng, context)
        first_text = frac_key if order == "fraction_first" else f"{pct}%"
        second_text = f"{pct}%" if order == "fraction_first" else frac_key
        if context == "allowance":
            text = (
                f"{name} has a ${M} allowance. {name} saves {first_text} of it. "
                f"{name} then spends {second_text} of the savings on {_a(item)}. "
                f"How much does {name} spend on the {item}?"
            )
        elif context == "earnings":
            text = (
                f"{name} earns ${M} from a weekend job. {name} sets aside {first_text} of it. "
                f"{name} then gives {second_text} of that amount to {other}. "
                f"How much does {name} give {other}?"
            )
        else:
            text = (
                f"A household budgets ${M} for the month. They set aside {first_text} of it for groceries. "
                f"They then spend {second_text} of the grocery money on {_a(item)}. "
                f"How much do they spend on the {item}?"
            )

        # In-generator check: recompute the two-step chain in one combined
        # formula rather than the step-wise route above.
        num1, den1 = (frac_value.numerator, frac_value.denominator)
        if order == "fraction_first":
            combined = Fraction(M * 100) * frac_value * pct_value
        else:
            combined = Fraction(M * 100) * pct_value * frac_value
        check_equivalent(combined, answer_cents, "combined single-formula recompute must match the step-wise chain")

        answer_text = _money(answer_cents)
        candidates = [_money(missed_step_cents), _money(wrong_base_cents), _money(shift_cents)]
        distractors = pick_distractors(answer_text, candidates, n=3, tolerance=_TOL)
        check(len(distractors) >= 2, "expected at least 2 surviving medium-tier distractors")

        return dict(
            text=text,
            answer=numeric_answer(answer_cents / 100),
            answer_text=answer_text,
            answer_latex=latex_answer(answer_text),
            working=[
                f"Step 1: {first_text} of ${M} is {_money(step1_cents)}. Step 2: {second_text} of {_money(step1_cents)} is {answer_text}."
            ],
            distractors=distractors,
        )
    raise ValueError("rational_percentage_financial_model/medium: no valid draw found")


def _pm_reverse_forward_simulate(M_cents: Fraction, frac_value: Fraction, pct_value: Fraction, order: str) -> Fraction:
    """Forward-simulate the stated two-step REMAINDER model from a starting
    amount to the final amount -- `remaining = M * (1 - a) * (1 - b)` in the
    drawn order. Used both by construction's own independent check and
    mirrored (from parsed text) by the oracle test file."""
    if order == "fraction_first":
        after1 = M_cents * (1 - frac_value)
        after2 = after1 * (1 - pct_value)
    else:
        after1 = M_cents * (1 - pct_value)
        after2 = after1 * (1 - frac_value)
    return after2


def _pm_reverse_question(rng, *, f_lo: int, f_hi: int) -> dict:
    """hard tier: REVERSE direction. A mixed fraction-then-percent (or
    percent-then-fraction) forward "what remains" model is stated; the
    final remaining amount F is given and the learner solves for the
    starting amount M -- running a mixed two-operation-kind model
    BACKWARDS, genuinely different reasoning from medium's forward chain,
    and distinct from `reverse-percentage`'s own single-step, percentage-
    only reversal."""
    for _ in range(_RESAMPLE_ATTEMPTS):
        M = rng.randrange(40, 2000, 20)
        frac_key = rng.choice(list(_FRACTIONS))
        pct = rng.choice(_PERCENTS)
        frac_value = _FRACTIONS[frac_key]
        pct_value = Fraction(pct, 100)
        order = rng.choice(("fraction_first", "percent_first"))
        M_cents = Fraction(M * 100)
        F_cents = _pm_reverse_forward_simulate(M_cents, frac_value, pct_value, order)
        if F_cents.denominator != 1:
            continue
        F_dollars = F_cents / 100
        if not (f_lo <= F_dollars <= f_hi):
            continue

        # In-generator check: forward-simulate from the (already known)
        # starting amount M and confirm it lands exactly on F -- proving
        # construction's own displayed F the "other way", as the
        # required discipline.
        check_equivalent(
            _pm_reverse_forward_simulate(M_cents, frac_value, pct_value, order),
            F_cents,
            "forward simulation from the starting amount must reach the stated final amount",
        )

        answer_cents = M_cents
        # forward-applied: mistakes the direction, applying the forward
        # model TO F instead of inverting it.
        forward_applied_cents = _pm_reverse_forward_simulate(F_cents, frac_value, pct_value, order)
        # additive-reverse: "undoes" both steps by ADDING the two rates
        # back onto F instead of dividing by the two complements.
        additive_reverse_cents = F_cents * (1 + frac_value + pct_value)
        # wrong-base (missed-step): undoes only the second step, leaving
        # the first step un-inverted.
        if order == "fraction_first":
            wrong_base_cents = F_cents / (1 - pct_value)
        else:
            wrong_base_cents = F_cents / (1 - frac_value)
        try:
            check_pairwise_distinct(
                answer_cents,
                forward_applied_cents,
                additive_reverse_cents,
                wrong_base_cents,
                msg="hard-tier candidates must be pairwise distinct",
            )
        except Exception:
            continue

        context = rng.choice(_PM_CONTEXTS)
        name, other = _two_names(rng)
        first_text = frac_key if order == "fraction_first" else f"{pct}%"
        second_text = f"{pct}%" if order == "fraction_first" else frac_key
        F_text = _money(F_cents)
        if context == "allowance":
            text = (
                f"After spending {first_text} of {other}'s allowance, then a further {second_text} of what remained, "
                f"{other} has ${F_text} left. How much did {other} start with?"
            )
        elif context == "earnings":
            text = (
                f"{name} spent {first_text} of this week's earnings, then a further {second_text} of what remained. "
                f"{name} now has ${F_text} left. How much did {name} earn this week?"
            )
        else:
            text = (
                f"A household spent {first_text} of its monthly budget, then a further {second_text} of what remained. "
                f"${F_text} is left. How much was the monthly budget?"
            )

        answer_text = _money(answer_cents)
        candidates = [
            _money_rounded(forward_applied_cents),
            _money_rounded(additive_reverse_cents),
            _money_rounded(wrong_base_cents),
        ]
        distractors = pick_distractors(answer_text, candidates, n=3, tolerance=_TOL)
        check(len(distractors) >= 2, "expected at least 2 surviving hard-tier distractors")

        return dict(
            text=text,
            answer=numeric_answer(answer_cents / 100),
            answer_text=answer_text,
            answer_latex=latex_answer(answer_text),
            working=[f"Working backwards through both steps from the final ${F_text}: start amount is {answer_text}."],
            distractors=distractors,
        )
    raise ValueError("rational_percentage_financial_model/hard: no valid draw found")


def _pm_exam_forward_simulate(
    M_cents: Fraction, frac_value: Fraction, pct2_value: Fraction, pct3_value: Fraction
) -> Fraction:
    """Forward-simulate the exam tier's THREE-deduction model:
    step 1 (fraction) and the flat fee both apply to the ORIGINAL amount M;
    step 2 (percent) applies to what remains AFTER step 1 -- two different
    tracked bases in the same question (M itself, and the running
    remainder), which no lower tier ever requires the learner to
    distinguish. `remaining = M*(1-frac) - M*(1-frac)*pct2 - M*pct3`."""
    after1 = M_cents * (1 - frac_value)
    after2 = after1 * (1 - pct2_value)
    fee = M_cents * pct3_value
    return after2 - fee


def _pm_exam_question(rng, *, f_lo: int, f_hi: int) -> dict:
    """exam tier: a three-deduction reverse model that exceeds hard's two-step ceiling by adding a genuinely NEW
    decision, not merely one more inversion of the same kind -- two of the three deductions (the fraction
    step, and a percent step on what remains) cascade off the running remainder exactly as hard's model does,
    but the THIRD deduction (a flat percentage service fee) is stated to apply to the ORIGINAL amount, not the
    remainder. The learner must therefore decide, per deduction, which of two different bases (M itself, or
    the running remainder) it operates on -- a decision hard's single-base cascade never poses."""
    for _ in range(_RESAMPLE_ATTEMPTS):
        M = rng.randrange(200, 4000, 20)
        frac_key = rng.choice(list(_FRACTIONS))
        pct2 = rng.choice(_PERCENTS)
        pct3 = rng.choice((5, 10, 15, 20))
        frac_value = _FRACTIONS[frac_key]
        pct2_value = Fraction(pct2, 100)
        pct3_value = Fraction(pct3, 100)
        M_cents = Fraction(M * 100)
        coeff = (1 - frac_value) * (1 - pct2_value) - pct3_value
        if coeff <= Fraction(1, 20):  # keep the resulting M solvable and not absurdly sensitive
            continue
        F_cents = M_cents * coeff
        if F_cents.denominator != 1:
            continue
        F_dollars = F_cents / 100
        if not (f_lo <= F_dollars <= f_hi):
            continue

        check_equivalent(
            _pm_exam_forward_simulate(M_cents, frac_value, pct2_value, pct3_value),
            F_cents,
            "forward simulation from the starting amount must reach the stated final amount",
        )

        answer_cents = M_cents
        # forward-applied: applies the forward model to F instead of
        # inverting it.
        forward_applied_cents = _pm_exam_forward_simulate(F_cents, frac_value, pct2_value, pct3_value)
        # ignore-fee (missed-step): solves the two-step cascade only,
        # ignoring the flat fee entirely (this is hard tier's own reverse
        # formula, applied to a question that actually needs the fee term).
        two_step_coeff = (1 - frac_value) * (1 - pct2_value)
        ignore_fee_cents = F_cents / two_step_coeff
        # wrong-base-cascading: assumes (incorrectly) that the fee ALSO
        # cascades off the remainder rather than the original amount.
        cascading_coeff = (1 - frac_value) * (1 - pct2_value) * (1 - pct3_value)
        if cascading_coeff <= 0:
            continue
        wrong_base_cascading_cents = F_cents / cascading_coeff
        try:
            check_pairwise_distinct(
                answer_cents,
                forward_applied_cents,
                ignore_fee_cents,
                wrong_base_cascading_cents,
                msg="exam-tier candidates must be pairwise distinct",
            )
        except Exception:
            continue

        context = rng.choice(_PM_CONTEXTS)
        name, other = _two_names(rng)
        F_text = _money(F_cents)
        if context == "allowance":
            subject = other
        elif context == "earnings":
            subject = name
        else:
            subject = "the household"
        subject_cap = subject[0].upper() + subject[1:]
        text = (
            f"{subject_cap} spent {frac_key} of the ORIGINAL amount on one item, then a further {pct2}% of what THEN "
            f"remained on a second item. {subject_cap} also pays a flat {pct3}% service fee, calculated on the ORIGINAL "
            f"amount (not on what remains). After all three, {subject} has ${F_text} left. "
            f"How much did {subject} start with?"
        )

        answer_text = _money(answer_cents)
        candidates = [
            _money_rounded(forward_applied_cents),
            _money_rounded(ignore_fee_cents),
            _money_rounded(wrong_base_cascading_cents),
        ]
        distractors = pick_distractors(answer_text, candidates, n=3, tolerance=_TOL)
        check(len(distractors) >= 2, "expected at least 2 surviving exam-tier distractors")

        return dict(
            text=text,
            answer=numeric_answer(answer_cents / 100),
            answer_text=answer_text,
            answer_latex=latex_answer(answer_text),
            working=[
                f"Two deductions cascade off the running remainder; the {pct3}% fee is taken off the ORIGINAL amount "
                f"separately. Start amount: {answer_text}."
            ],
            distractors=distractors,
        )
    raise ValueError("rational_percentage_financial_model/exam: no valid draw found")


@generator(
    "rational_percentage_financial_model",
    level=8,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def rational_percentage_financial_model(rng, tier: str = "easy"):
    """Determine whether a fraction or a percentage operation correctly
    models a stated financial situation, and compute the result. easy: both
    a fraction clause and a percentage clause are stated over the same base
    amount, one is asked for. medium: a genuine two-step chain mixing a
    fraction step and a percent step, in a random order, the second step
    always applied to the running total from the first. hard: the REVERSE
    of a mixed two-step "what remains" model -- given the final remaining
    amount, find the start. exam: a three-deduction reverse model where two
    deductions cascade off the running remainder and a third (flat
    percentage fee) applies to the original amount instead -- a genuinely
    new base-tracking decision, not merely one more inversion of hard's own
    shape."""
    if tier == "easy":
        return _pm_single_step_question(rng, m_lo=20, m_hi=1200)
    if tier == "medium":
        return _pm_chain_question(rng, m_lo=40, m_hi=1600)
    if tier == "hard":
        return _pm_reverse_question(rng, f_lo=10, f_hi=1800)
    if tier == "exam":
        return _pm_exam_question(rng, f_lo=20, f_hi=3600)
    raise ValueError(f"unknown tier: {tier!r}")


# =============================================================================
# Skill 2: ratio_rate_financial_model
# =============================================================================

_RATIO_PAIRS = ((1, 3), (1, 4), (2, 3), (2, 5), (3, 4), (3, 5), (1, 2), (3, 7), (2, 7), (4, 5))
_RATIO_TRIPLES = ((1, 2, 3), (1, 1, 2), (2, 3, 5), (1, 2, 4), (1, 3, 4), (2, 2, 3))
_RATES_CENTS = (150, 180, 200, 250, 300, 350, 400, 450, 500, 120, 175, 225)
_RR_CONTEXTS = ("prize", "bill", "market")
_RR_UNIT_WORDS = (("kg", "kg"), ("litre", "litres"), ("hour", "hours"), ("metre", "metres"))

# (item_singular, item_plural, unit, unit_plural) -- a curated, hand-vetted set
# of genuinely sensible retail/trade pairings for the rate branch's flavour
# text (easy/medium/hard tiers only). Deliberately NOT `mathgen.slots.OBJECTS`:
# that generic flavour-noun list (apple, marble, sticker, pencil, cupcake,
# balloon, ribbon, ticket, coin, button, biscuit, chair, box, cookie, crayon,
# bead, shell, bottle) was never designed for per-unit-rate framing and drawing
# it independently of the unit produced nonsense like "crayons cost $2.00 per
# metre" or "chairs cost $1.80 per kg". Every entry below is something that is
# genuinely priced by that exact unit in ordinary English. `item_plural` is the
# real plural (never built by string concatenation, so no more "Boxs"); for
# uncountable/mass nouns (fuel, paint, ...) `item_plural == item_singular`,
# used as a signal below to pick the correct verb form.
_RR_ITEM_UNIT_PAIRS = (
    ("ribbon", "ribbons", "metre", "metres"),
    ("rope", "ropes", "metre", "metres"),
    ("cable", "cable", "metre", "metres"),
    ("fabric", "fabric", "metre", "metres"),
    ("fuel", "fuel", "litre", "litres"),
    ("paint", "paint", "litre", "litres"),
    ("juice", "juice", "litre", "litres"),
    ("milk", "milk", "litre", "litres"),
    ("apple", "apples", "kg", "kg"),
    ("flour", "flour", "kg", "kg"),
    ("rice", "rice", "kg", "kg"),
    ("cheese", "cheese", "kg", "kg"),
    ("parking", "parking", "hour", "hours"),
    ("car hire", "car hire", "hour", "hours"),
    ("tutoring", "tutoring", "hour", "hours"),
)


def _rr_plural_verb(item_plural: str, item_singular: str, *, plural: str, singular: str) -> str:
    """Pick the grammatically correct verb form for `item_plural` as a
    sentence subject: a genuine plural (differs from the singular, e.g.
    "ribbons") takes `plural` ("cost"/"are"); an uncountable/mass noun used
    as-is (`item_plural == item_singular`, e.g. "fuel") takes `singular`
    ("costs"/"is")."""
    return plural if item_plural != item_singular else singular


def _rr_single_step_question(rng, *, t_lo: int, t_hi: int, qty_lo: int, qty_hi: int) -> dict:
    """easy tier: a single-step choice between a RATIO split (a fixed total
    divided in a stated proportion, find one share) and a RATE application
    (a stated per-unit price times a stated quantity, find the total
    cost) -- randomised which representation applies, every seed."""
    for _ in range(_RESAMPLE_ATTEMPTS):
        branch = rng.choice(("ratio", "rate"))
        context = rng.choice(_RR_CONTEXTS)
        name1, name2 = _two_names(rng)
        if branch == "ratio":
            p, q = rng.choice(_RATIO_PAIRS)
            total_parts = p + q
            T = _draw_multiple(rng, t_lo, t_hi, total_parts)
            T_cents = Fraction(T * 100)
            share_p_cents = T_cents * p / total_parts
            share_q_cents = T_cents * q / total_parts
            check(
                share_p_cents.denominator == 1 and share_q_cents.denominator == 1,
                "ratio shares must land on an exact cent",
            )
            asked_p = rng.choice((True, False))
            answer_cents = share_p_cents if asked_p else share_q_cents
            other_share_cents = share_q_cents if asked_p else share_p_cents
            # representation-swap: reads "p:q" as a RATE (p per q) instead
            # of a ratio split -- computes T * p / q.
            swap_cents = T_cents * p / q
            shift_cents = answer_cents * 10
            try:
                check_pairwise_distinct(
                    answer_cents,
                    other_share_cents,
                    swap_cents,
                    shift_cents,
                    msg="ratio easy-tier candidates must be pairwise distinct",
                )
            except Exception:
                continue
            who = name1 if asked_p else name2
            if context == "prize":
                text = f"A ${T} prize is split between {name1} and {name2} in the ratio {p}:{q}. How much does {who} receive?"
            elif context == "bill":
                text = (
                    f"A ${T} bill is split between {name1} and {name2} in the ratio {p}:{q}. How much does {who} pay?"
                )
            else:
                text = f"A ${T} market stall's takings are split between {name1} and {name2} in the ratio {p}:{q}. How much does {who} keep?"
            check_equivalent(
                Fraction(T * (p if asked_p else q), total_parts) * 100,
                answer_cents,
                "ratio-share re-derivation must match",
            )
            candidates = [_money(other_share_cents), _money_rounded(swap_cents), _money(shift_cents)]
            working_line = (
                f"The ratio {p}:{q} makes {total_parts} equal parts, so {who} gets "
                f"{p if asked_p else q}/{total_parts} of ${T}, which is ${_money(answer_cents)}."
            )
        else:
            item, item_plural, unit, unit_plural = rng.choice(_RR_ITEM_UNIT_PAIRS)
            rate_cents = rng.choice(_RATES_CENTS)
            qty = rng.randrange(qty_lo, qty_hi + 1)
            total_cents = Fraction(rate_cents * qty)
            answer_cents = total_cents
            # ignores-quantity: treats the stated rate as the whole cost.
            ignore_qty_cents = Fraction(rate_cents)
            # rate-inversion: divides instead of multiplies.
            inversion_cents = Fraction(rate_cents, qty) if qty else Fraction(rate_cents)
            shift_cents = answer_cents * 10
            if qty <= 1:
                continue
            try:
                check_pairwise_distinct(
                    answer_cents,
                    ignore_qty_cents,
                    inversion_cents,
                    shift_cents,
                    msg="rate easy-tier candidates must be pairwise distinct",
                )
            except Exception:
                continue
            if context == "prize":
                text = f"{name1} sells {item_plural} at a rate of ${_money(rate_cents)} per {unit}. {name2} buys {qty} {unit_plural}. What is the total cost?"
            elif context == "bill":
                text = f"A tradesperson charges ${_money(rate_cents)} per {unit} of work. {name1} is billed for {qty} {unit_plural}. What is the total cost?"
            else:
                verb = _rr_plural_verb(item_plural, item, plural="cost", singular="costs")
                text = f"At the market, {item_plural} {verb} ${_money(rate_cents)} per {unit}. {name1} buys {qty} {unit_plural}. What is the total cost?"
            check_equivalent(Fraction(rate_cents) * qty, answer_cents, "rate re-derivation must match")
            candidates = [_money(ignore_qty_cents), _money_rounded(inversion_cents), _money(shift_cents)]
            working_line = f"{qty} × ${_money(rate_cents)} = ${_money(answer_cents)}."

        answer_text = _money(answer_cents)
        distractors = pick_distractors(answer_text, candidates, n=3, tolerance=_TOL)
        check(len(distractors) >= 2, "expected at least 2 surviving ratio/rate easy-tier distractors")
        return dict(
            text=text,
            answer=numeric_answer(answer_cents / 100),
            answer_text=answer_text,
            answer_latex=latex_answer(answer_text),
            working=[working_line],
            distractors=distractors,
        )
    raise ValueError("ratio_rate_financial_model/easy: no valid draw found")


def _rr_composed_question(rng, *, t_lo: int, t_hi: int, qty_lo: int, qty_hi: int) -> dict:
    """medium tier: a two-step composition WITHIN one representation kind.
    ratio: a three-way split, then two of the three shares are combined.
    rate: a total cost from a rate x quantity, then a further flat
    adjustment is applied. A genuine second reasoning step, not merely
    wider numbers."""
    for _ in range(_RESAMPLE_ATTEMPTS):
        branch = rng.choice(("ratio", "rate"))
        context = rng.choice(_RR_CONTEXTS)
        name1, name2, name3 = rng.sample(NAMES, 3)
        if branch == "ratio":
            p, q, r = rng.choice(_RATIO_TRIPLES)
            total_parts = p + q + r
            T = _draw_multiple(rng, t_lo, t_hi, total_parts)
            T_cents = Fraction(T * 100)
            share_p = T_cents * p / total_parts
            share_q = T_cents * q / total_parts
            share_r = T_cents * r / total_parts
            if any(s.denominator != 1 for s in (share_p, share_q, share_r)):
                continue
            combined_cents = share_p + share_q  # the two pooled shares
            odd_one_out_cents = share_r
            # misapplies-total-parts: treats p and q as if they were the
            # ONLY two parts in the ratio (a two-way split of the whole
            # total), ignoring that a third share (r) also exists and so
            # p + q is not the true total-parts denominator.
            wrong_denominator_cents = T_cents * Fraction(p, p + q)
            shift_cents = combined_cents * 10
            try:
                check_pairwise_distinct(
                    combined_cents,
                    odd_one_out_cents,
                    wrong_denominator_cents,
                    shift_cents,
                    msg="ratio medium-tier candidates must be pairwise distinct",
                )
            except Exception:
                continue
            answer_cents = combined_cents
            if context == "prize":
                text = (
                    f"A ${T} prize is shared between {name1}, {name2} and {name3} in the ratio {p}:{q}:{r}. "
                    f"{name1} and {name2} pool their shares together. How much do they have combined?"
                )
            elif context == "bill":
                text = (
                    f"A ${T} bill is split between {name1}, {name2} and {name3} in the ratio {p}:{q}:{r}. "
                    f"{name1} and {name2} then combine their two portions to pay together. How much do they pay combined?"
                )
            else:
                text = (
                    f"${T} in market takings is shared between {name1}, {name2} and {name3} in the ratio {p}:{q}:{r}. "
                    f"{name1} and {name2} pool their shares. How much do they have combined?"
                )
            check_equivalent(
                T_cents * Fraction(p + q, total_parts), answer_cents, "combined-share re-derivation must match"
            )
            candidates = [_money(odd_one_out_cents), _money_rounded(wrong_denominator_cents), _money(shift_cents)]
            working_line = (
                f"The ratio {p}:{q}:{r} makes {total_parts} equal parts, so {name1} and {name2} together get "
                f"{p + q}/{total_parts} of ${T}, which is ${_money(answer_cents)}."
            )
        else:
            item, item_plural, unit, unit_plural = rng.choice(_RR_ITEM_UNIT_PAIRS)
            rate_cents = rng.choice(_RATES_CENTS)
            qty = rng.randrange(qty_lo, qty_hi + 1)
            if qty <= 1:
                continue
            base_cents = Fraction(rate_cents * qty)
            adjust_cents = Fraction(rng.choice((500, 1000, 1500, 2000, 250, 750)))
            direction = rng.choice(("discount", "surcharge"))
            answer_cents = base_cents - adjust_cents if direction == "discount" else base_cents + adjust_cents
            if answer_cents <= 0:
                continue
            forgets_adjustment_cents = base_cents
            wrong_sign_cents = base_cents + adjust_cents if direction == "discount" else base_cents - adjust_cents
            shift_cents = answer_cents * 10
            try:
                check_pairwise_distinct(
                    answer_cents,
                    forgets_adjustment_cents,
                    wrong_sign_cents,
                    shift_cents,
                    msg="rate medium-tier candidates must be pairwise distinct",
                )
            except Exception:
                continue
            verb = "flat discount" if direction == "discount" else "flat surcharge"
            if context == "prize":
                text = (
                    f"{name1} sells {item_plural} at a rate of ${_money(rate_cents)} per {unit}. {name2} buys {qty} {unit_plural}, "
                    f"then receives a ${_money(adjust_cents)} {verb} on the total. What is the final cost?"
                )
            elif context == "bill":
                text = (
                    f"A tradesperson charges ${_money(rate_cents)} per {unit} of work. After {qty} {unit_plural}, "
                    f"a ${_money(adjust_cents)} {verb} is applied to the bill. What is the final cost?"
                )
            else:
                cost_verb = _rr_plural_verb(item_plural, item, plural="cost", singular="costs")
                text = (
                    f"At the market, {item_plural} {cost_verb} ${_money(rate_cents)} per {unit}. {name1} buys {qty} {unit_plural}, "
                    f"then a ${_money(adjust_cents)} {verb} is applied. What is the final cost?"
                )
            check_equivalent(
                Fraction(rate_cents) * qty + (adjust_cents if direction == "surcharge" else -adjust_cents),
                answer_cents,
                "adjusted-total re-derivation must match",
            )
            candidates = [_money(forgets_adjustment_cents), _money(wrong_sign_cents), _money(shift_cents)]
            working_line = (
                f"{qty} × ${_money(rate_cents)} = ${_money(base_cents)}, then "
                f"{'subtract' if direction == 'discount' else 'add'} ${_money(adjust_cents)} to get ${_money(answer_cents)}."
            )

        answer_text = _money(answer_cents)
        distractors = pick_distractors(answer_text, candidates, n=3, tolerance=_TOL)
        check(len(distractors) >= 2, "expected at least 2 surviving ratio/rate medium-tier distractors")
        return dict(
            text=text,
            answer=numeric_answer(answer_cents / 100),
            answer_text=answer_text,
            answer_latex=latex_answer(answer_text),
            working=[working_line],
            distractors=distractors,
        )
    raise ValueError("ratio_rate_financial_model/medium: no valid draw found")


def _rr_trap_question(rng, *, t_lo: int, t_hi: int) -> dict:
    """hard tier: the CHOICE itself becomes the trap. Two scenario types,
    randomised: (a) surface wording is ratio-flavoured ("split in the
    ratio") but a flat, non-proportional fee must be removed FIRST, so the
    ratio only governs the remainder, not the full stated total; (b)
    surface wording is a plain per-unit rate, but a stated number of
    free/promotional units must be subtracted from the quantity first, so
    the rate only applies to the PAID units, not the full stated quantity.
    Both require the learner to recognise which part of the stated figure
    the named representation actually governs, rather than applying it to
    the headline number directly."""
    for _ in range(_RESAMPLE_ATTEMPTS):
        scenario = rng.choice(("ratio_with_fee", "rate_with_free_units"))
        name1, name2 = _two_names(rng)
        if scenario == "ratio_with_fee":
            p, q = rng.choice(_RATIO_PAIRS)
            total_parts = p + q
            flat = rng.choice((500, 1000, 1500, 2000, 2500))
            T = _draw_multiple(rng, t_lo, t_hi, total_parts)
            T_cents = Fraction(T * 100)
            remainder_cents = T_cents - flat
            if remainder_cents <= 0:
                continue
            share_cents = remainder_cents * p / total_parts
            if share_cents.denominator != 1:
                continue
            answer_cents = share_cents
            # naive-full-split: applies the ratio to the WHOLE bill,
            # ignoring that the flat fee must be removed first.
            naive_full_split_cents = T_cents * p / total_parts
            shift_cents = answer_cents * 10
            try:
                check_pairwise_distinct(
                    answer_cents,
                    naive_full_split_cents,
                    shift_cents,
                    msg="ratio-with-fee hard-tier candidates must be pairwise distinct",
                )
            except Exception:
                continue
            text = (
                f"A ${T} job payment is described as being split in the ratio {p}:{q} between {name1} and {name2}. "
                f"In fact, a flat ${_money(flat)} materials fee is taken off the top FIRST, and only the remainder "
                f"is split in that ratio. How much does {name1} actually receive?"
            )
            check_equivalent(
                (T_cents - flat) * Fraction(p, total_parts), answer_cents, "fee-then-split re-derivation must match"
            )
            candidates = [_money(naive_full_split_cents), _money(shift_cents)]
            working_line = (
                f"Take off the fee first: ${T} - ${_money(flat)} = ${_money(remainder_cents)}. "
                f"Then {p}/{total_parts} of that is ${_money(answer_cents)}."
            )
        else:
            item, item_plural, unit, unit_plural = rng.choice(_RR_ITEM_UNIT_PAIRS)
            rate_cents = rng.choice(_RATES_CENTS)
            free_qty = rng.randint(2, 5)
            qty = rng.randint(free_qty + 3, free_qty + 20)
            paid_qty = qty - free_qty
            answer_cents = Fraction(rate_cents * paid_qty)
            # naive-full-rate: charges for every unit, ignoring the free
            # promotional units.
            naive_full_rate_cents = Fraction(rate_cents * qty)
            shift_cents = answer_cents * 10
            try:
                check_pairwise_distinct(
                    answer_cents,
                    naive_full_rate_cents,
                    shift_cents,
                    msg="rate-with-free-units hard-tier candidates must be pairwise distinct",
                )
            except Exception:
                continue
            sold_verb = _rr_plural_verb(item_plural, item, plural="are", singular="is")
            text = (
                f"{item_plural.capitalize()} {sold_verb} sold at a rate of ${_money(rate_cents)} per {unit}. As a promotion, the first "
                f"{free_qty} {unit_plural} are free. {name1} buys {qty} {unit_plural} in total. How much does {name1} pay?"
            )
            check_equivalent(
                Fraction(rate_cents) * (qty - free_qty), answer_cents, "paid-units re-derivation must match"
            )
            candidates = [_money(naive_full_rate_cents), _money(shift_cents)]
            working_line = (
                f"Only {paid_qty} {unit_plural if paid_qty != 1 else unit} are paid for ({qty} - {free_qty}), "
                f"so {paid_qty} × ${_money(rate_cents)} = ${_money(answer_cents)}."
            )

        answer_text = _money(answer_cents)
        distractors = pick_distractors(answer_text, candidates, n=len(candidates), tolerance=_TOL)
        check(len(distractors) >= 1, "expected at least 1 surviving hard-tier distractor")
        return dict(
            text=text,
            answer=numeric_answer(answer_cents / 100),
            answer_text=answer_text,
            answer_latex=latex_answer(answer_text),
            working=[working_line],
            distractors=distractors,
        )
    raise ValueError("ratio_rate_financial_model/hard: no valid draw found")


def _rr_exam_question(rng, *, t_lo: int, t_hi: int) -> dict:
    """exam tier: a three-quantity chained model requiring BOTH a ratio
    split AND a rate application in the same question -- a bill is split
    in a stated ratio between two people, and the larger share is then
    spent entirely on a per-unit rate purchase; the learner solves for the
    resulting QUANTITY (litres/kg/hours), not a dollar amount, genuinely
    exceeding hard's single-representation ceiling."""
    for _ in range(_RESAMPLE_ATTEMPTS):
        p, q = rng.choice(_RATIO_PAIRS)
        if p == q:
            continue
        total_parts = p + q
        T = _draw_multiple(rng, t_lo, t_hi, total_parts)
        T_cents = Fraction(T * 100)
        larger_parts = max(p, q)
        smaller_parts = min(p, q)
        larger_share_cents = T_cents * larger_parts / total_parts
        smaller_share_cents = T_cents * smaller_parts / total_parts
        if larger_share_cents.denominator != 1:
            continue
        rate_cents = rng.choice(_RATES_CENTS)
        quantity = larger_share_cents / rate_cents
        scaled = quantity * 100
        if scaled.denominator != 1:
            continue  # reject-resample: must land on an exact 2dp quantity

        name1, name2 = _two_names(rng)
        # Fixed to litre: the item word is the literal "fuel" in every one of
        # this tier's three context templates below, and fuel is never sold
        # by the kilogram in ordinary retail/trade usage ("fuel at $1.75 per kg"
        # was a nonsense item/unit pairing; "kg" is dropped rather than
        # paired with a second item word, to keep this tier's own draw space
        # unchanged in every other respect).
        unit, unit_plural = "litre", "litres"
        answer_value = quantity
        # naive-whole-bill: uses the whole bill instead of the larger
        # share.
        naive_whole_bill = T_cents / rate_cents
        # smaller-share-instead: uses the smaller share instead of the
        # larger one.
        smaller_share_value = smaller_share_cents / rate_cents
        shift_value = answer_value * 10
        try:
            check_pairwise_distinct(
                answer_value,
                naive_whole_bill,
                smaller_share_value,
                shift_value,
                msg="exam-tier candidates must be pairwise distinct",
            )
        except Exception:
            continue
        who = name1 if p >= q else name2
        context = rng.choice(_RR_CONTEXTS)
        if context == "prize":
            text = (
                f"A ${T} prize is split between {name1} and {name2} in the ratio {p}:{q}. {who}'s larger share is then "
                f"spent entirely on fuel at ${_money(rate_cents)} per {unit}. How many {unit_plural} does {who} buy?"
            )
        elif context == "bill":
            text = (
                f"A ${T} bill is split between {name1} and {name2} in the ratio {p}:{q}. {who}'s larger share is then "
                f"spent entirely on fuel at ${_money(rate_cents)} per {unit}. How many {unit_plural} does {who} buy?"
            )
        else:
            text = (
                f"${T} in market takings is split between {name1} and {name2} in the ratio {p}:{q}. {who}'s larger "
                f"share is then spent entirely on fuel at ${_money(rate_cents)} per {unit}. How many {unit_plural} does {who} buy?"
            )
        check_equivalent(larger_share_cents / rate_cents, answer_value, "ratio-then-rate re-derivation must match")

        answer_text = stated_precision_string(answer_value, 2)
        candidates = [
            stated_precision_string(round_half_up(naive_whole_bill, 2), 2),
            stated_precision_string(round_half_up(smaller_share_value, 2), 2),
            stated_precision_string(round_half_up(shift_value, 2), 2),
        ]
        distractors = pick_distractors(answer_text, candidates, n=3, tolerance=_TOL)
        check(len(distractors) >= 2, "expected at least 2 surviving exam-tier distractors")
        return dict(
            text=text,
            answer=numeric_answer(answer_value),
            answer_text=answer_text,
            answer_latex=latex_answer(answer_text),
            working=[
                f"Larger share: ${_money(larger_share_cents)}; at ${_money(rate_cents)} per {unit} that buys {answer_text} {unit_plural}."
            ],
            distractors=distractors,
        )
    raise ValueError("ratio_rate_financial_model/exam: no valid draw found")


@generator(
    "ratio_rate_financial_model",
    level=8,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def ratio_rate_financial_model(rng, tier: str = "easy"):
    """Determine whether a ratio-split or a rate representation correctly
    models a stated money situation, and solve for the resulting value.
    easy: a single-step choice between a ratio split and a rate
    application. medium: a two-step composition within one representation
    kind (a three-way ratio split combined, or a rate total with a flat
    adjustment). hard: the choice itself is the trap -- surface wording
    suggests one representation while the correct reading needs a
    correction first. exam: a chained ratio-THEN-rate model, solving for a
    quantity rather than a dollar amount."""
    if tier == "easy":
        return _rr_single_step_question(rng, t_lo=20, t_hi=600, qty_lo=2, qty_hi=40)
    if tier == "medium":
        return _rr_composed_question(rng, t_lo=60, t_hi=1200, qty_lo=2, qty_hi=40)
    if tier == "hard":
        return _rr_trap_question(rng, t_lo=100, t_hi=2000)
    if tier == "exam":
        return _rr_exam_question(rng, t_lo=100, t_hi=3000)
    raise ValueError(f"unknown tier: {tier!r}")
