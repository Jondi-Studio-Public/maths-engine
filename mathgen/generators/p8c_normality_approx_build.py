"""Examine the approximate normality of the distribution of p-hat.

- ``p8c_normality_approx_phat``
  (``senior-examine-the-approximate-normality-of-the-distribution-of``,
  ACMMM175, domain statistics-probability, answer: mcq)
  -- a 4-way classification of whether the normal approximation to p-hat's
  sampling distribution is valid, using the standard bounded rule (both
  n*phat >= 10 and n*(1-phat) >= 10), stated explicitly in every question's
  own text -- "unstated assumptions" is a raw exclusion, so the threshold is
  never assumed memorised.

This is a diagnostic/classification skill, never an interval computation:
it never produces a ``(lower, upper)`` pair -- that shape belongs
exclusively to the co-located sibling
``senior-use-the-approximate-confidence-interval-left-widehat-p``
(``p8c_confidence_interval_build.py``, ACMMM178), which computes
the interval once the approximation already holds.
note: this generator never presents any interval bounds and never asks the
learner to compute one.

Two variants present the inputs: ``direct`` (phat given as a decimal) and
``from_counts``/``borderline`` (x successes out of n stated, phat = x/n
derived). Using counts makes both n*phat and n*(1-phat) EXACT integers
(x and n-x respectively) -- so the ``hard`` tier's deliberately borderline
draws never depend on a rounding boundary the way a decimal-phat draw
would; the classification is exact arithmetic on integers, by construction.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import check, check_equivalent, generator, latex_answer, pick_distractors

CATEGORY_VALID = "Valid -- both n×p-hat and n×(1-p-hat) are at least 10"
CATEGORY_P_FAILS = "Not valid -- n×p-hat is less than 10"
CATEGORY_Q_FAILS = "Not valid -- n×(1-p-hat) is less than 10"
CATEGORY_BOTH_FAIL = "Not valid -- both n×p-hat and n×(1-p-hat) are less than 10"

ALL_CATEGORIES = (CATEGORY_VALID, CATEGORY_P_FAILS, CATEGORY_Q_FAILS, CATEGORY_BOTH_FAIL)

RULE_SENTENCE = (
    "The normal approximation to the distribution of p-hat is considered "
    "valid only when both n×p-hat and n×(1-p-hat) are at least 10."
)
_ASK = "Classify whether the normal approximation to the distribution of p-hat is valid for this sample."


def _classify(prod_p: Fraction, prod_q: Fraction) -> str:
    """Construction's own classification route."""
    p_ok = prod_p >= 10
    q_ok = prod_q >= 10
    if p_ok and q_ok:
        return CATEGORY_VALID
    if p_ok and not q_ok:
        return CATEGORY_Q_FAILS
    if q_ok and not p_ok:
        return CATEGORY_P_FAILS
    return CATEGORY_BOTH_FAIL


def _verify_classify(prod_p: Fraction, prod_q: Fraction) -> str:
    """Independently written if/elif chain for the verification routine -- a different code object
    from:func:`_classify`, never called by it and never calling it, even though the logic necessarily mirrors
    it."""
    if prod_p >= 10:
        if prod_q >= 10:
            return CATEGORY_VALID
        else:
            return CATEGORY_Q_FAILS
    else:
        if prod_q >= 10:
            return CATEGORY_P_FAILS
        else:
            return CATEGORY_BOTH_FAIL


def _success_word(x: int) -> str:
    return "success" if x == 1 else "successes"


def _fmt_product(value: Fraction) -> str:
    """Display-only formatting for the ``working`` lines (never graded)."""
    if value.denominator == 1:
        return str(value.numerator)
    return f"{float(value):.2f}"


def _independent_check_direct(n: int, phat: Fraction, phat_display: str, category: str) -> None:
    """Reparses phat from its own DISPLAYED decimal string -- a different
    route than construction's ``Fraction(hundredths, 100)`` build -- then
    re-derives the category via :func:`_verify_classify`, never
    :func:`_classify`."""
    phat_reparsed = Fraction(phat_display)
    check_equivalent(phat_reparsed, phat, "direct: displayed phat does not reparse to the constructed value")
    prod_p = n * phat_reparsed
    prod_q = n * (1 - phat_reparsed)
    check(_verify_classify(prod_p, prod_q) == category, "direct: independent check disagrees with construction")


def _independent_check_counts(n: int, x: int, category: str) -> None:
    """Derives n*phat and n*(1-phat) as the bare integers x and n-x -- a
    different expression than a route going through ``n * Fraction(x, n)``."""
    prod_p = Fraction(x)
    prod_q = Fraction(n - x)
    check(_verify_classify(prod_p, prod_q) == category, "from_counts: independent check disagrees with construction")


def _independent_check_borderline(n: int, x: int, category: str) -> None:
    """Derives n*phat and n*(1-phat) via ``n * Fraction(x, n)`` -- the
    opposite route from :func:`_independent_check_counts`'s bare
    subtraction, since construction here builds the products as bare
    integers directly (see :func:`_borderline`)."""
    phat = Fraction(x, n)
    prod_p = n * phat
    prod_q = n * (1 - phat)
    check(_verify_classify(prod_p, prod_q) == category, "borderline: independent check disagrees with construction")


def _direct(rng):
    """Easy: phat given directly (1 or 2 dp), n given directly. Both
    products are resampled until confidently clear of the threshold (margin
    > 3) in whichever direction is true -- a confident, non-borderline
    classification either way."""
    margin = Fraction(3)
    for _attempt in range(500):
        dp = rng.choice([1, 2])
        n = rng.randint(5, 60)
        if dp == 1:
            hundredths = rng.randint(1, 9) * 10
        else:
            hundredths = rng.randint(5, 95)
        phat = Fraction(hundredths, 100)
        prod_p = n * phat
        prod_q = n * (1 - phat)
        if abs(prod_p - 10) > margin and abs(prod_q - 10) > margin:
            break
    else:
        raise RuntimeError("p8c_normality_approx_phat/direct: no draw cleared the confident margin in 500 attempts")

    category = _classify(prod_p, prod_q)
    check(category in ALL_CATEGORIES, "direct: category not recognised")
    phat_display = f"{float(phat):.{dp}f}"
    _independent_check_direct(n, phat, phat_display, category)

    distractors = pick_distractors(category, [c for c in ALL_CATEGORIES if c != category])
    text = f"A sample of size n = {n} has a sample proportion of p-hat = {phat_display}. {RULE_SENTENCE} {_ASK}"
    working = [
        f"n×p-hat = {n} × {phat_display} = {_fmt_product(prod_p)} (compare to 10).",
        f"n×(1-p-hat) = {n} × {1 - phat} = {_fmt_product(prod_q)} (compare to 10).",
    ]
    return dict(
        text=text,
        latex=text,
        answer=category,
        answer_text=category,
        answer_latex=latex_answer(category, plain=True),
        working=working,
        distractors=distractors,
    )


def _from_counts(rng):
    """Medium: x successes out of n stated; phat = x/n must be derived
    first. Resampled with the same confident margin as the direct variant so
    medium tests the same rule, plus one extra derivation step, never an
    accidental borderline draw."""
    margin = 3
    for _attempt in range(500):
        n = rng.randint(8, 80)
        # x >= 2: text_quality.py's singular-count-plural check has no
        # carve-out for a singular noun that itself ends in "s" ("1
        # success"), so it is side-stepped by
        # construction instead of by wording.
        x = rng.randint(2, n - 1)
        prod_p = x
        prod_q = n - x
        if abs(prod_p - 10) > margin and abs(prod_q - 10) > margin:
            break
    else:
        raise RuntimeError(
            "p8c_normality_approx_phat/from_counts: no draw cleared the confident margin in 500 attempts"
        )

    prod_p_f, prod_q_f = Fraction(prod_p), Fraction(prod_q)
    category = _classify(prod_p_f, prod_q_f)
    check(category in ALL_CATEGORIES, "from_counts: category not recognised")
    _independent_check_counts(n, x, category)

    distractors = pick_distractors(category, [c for c in ALL_CATEGORIES if c != category])
    text = f"A sample of size n = {n} finds x = {x} {_success_word(x)}, so p-hat = x / n. {RULE_SENTENCE} {_ASK}"
    working = [
        f"n×p-hat = x = {prod_p} (compare to 10).",
        f"n×(1-p-hat) = n - x = {prod_q} (compare to 10).",
    ]
    return dict(
        text=text,
        latex=text,
        answer=category,
        answer_text=category,
        answer_latex=latex_answer(category, plain=True),
        working=working,
        distractors=distractors,
    )


def _borderline(rng):
    """Hard: construct n and x (counts) so that at least one of
    n*phat=x / n*(1-phat)=n-x lands within 1 or 2 of the threshold of 10,
    never exactly on it -- forcing genuine arithmetic care rather than
    eyeballing. The OTHER product is drawn independently across a wide
    range so all three "not valid" outcomes and "valid" are all reachable,
    not only one shape."""
    for _attempt in range(200):
        target = rng.choice(["p", "q"])
        diff = rng.choice([1, 2])
        sign = rng.choice([-1, 1])
        near_value = 10 + sign * diff
        # other_value >= 2: same "1 success" side-step as _from_counts above.
        other_value = rng.randint(2, 30)
        if other_value == 10:
            continue
        if target == "p":
            x, other = near_value, other_value
        else:
            other, x = other_value, near_value
        n = x + other
        prod_p, prod_q = x, n - x
        if prod_p != 10 and prod_q != 10 and min(abs(prod_p - 10), abs(prod_q - 10)) in (1, 2):
            break
    else:
        raise RuntimeError(
            "p8c_normality_approx_phat/borderline: no draw produced a genuine near-threshold pair in 200 attempts"
        )

    prod_p_f, prod_q_f = Fraction(prod_p), Fraction(prod_q)
    category = _classify(prod_p_f, prod_q_f)
    check(category in ALL_CATEGORIES, "borderline: category not recognised")
    _independent_check_borderline(n, x, category)

    distractors = pick_distractors(category, [c for c in ALL_CATEGORIES if c != category])
    text = f"A sample of size n = {n} finds x = {x} {_success_word(x)}, so p-hat = x / n. {RULE_SENTENCE} {_ASK}"
    working = [
        f"n×p-hat = x = {prod_p} (compare to 10).",
        f"n×(1-p-hat) = n - x = {prod_q} (compare to 10).",
    ]
    return dict(
        text=text,
        latex=text,
        answer=category,
        answer_text=category,
        answer_latex=latex_answer(category, plain=True),
        working=working,
        distractors=distractors,
    )


def _independent_check_failures(n: int, f: int, category: str) -> None:
    """Re-derives the products from the stated failure count by the opposite
    route: n*phat via ``n * Fraction(n - f, n)`` and n*(1-phat) via
    ``n * (1 - Fraction(n - f, n))``, then classifies with
    :func:`_verify_classify`."""
    phat = Fraction(n - f, n)
    prod_p = n * phat
    prod_q = n * (1 - phat)
    check(prod_q == f, "from_failures: n*(1-phat) is not the stated failure count")
    check(_verify_classify(prod_p, prod_q) == category, "from_failures: independent check disagrees with construction")


def _from_failures(rng):
    """Hard (new): the sample is stated as n and x FAILURES while p-hat is
    the proportion of SUCCESSES, so n*p-hat = n - x and n*(1-p-hat) = x. A
    learner who treats x as successes swaps the two 'not valid' categories.
    Exactly one product is below 10 in every draw (so the swap trap is
    always live); VALID and BOTH_FAIL are therefore never produced here."""
    for _attempt in range(200):
        low_is_failures = rng.choice([True, False])
        low = rng.randint(2, 9)
        high = rng.randint(10, 30)
        f, s = (low, high) if low_is_failures else (high, low)
        n = s + f
        if f >= 2 and s >= 2 and (s < 10) != (f < 10):
            break
    else:
        raise RuntimeError("p8c_normality_approx_phat/from_failures: no draw produced an asymmetric pair")

    prod_p, prod_q = Fraction(n - f), Fraction(f)
    category = _classify(prod_p, prod_q)
    check(category in (CATEGORY_P_FAILS, CATEGORY_Q_FAILS), "from_failures: category must be a single-product failure")
    _independent_check_failures(n, f, category)

    distractors = pick_distractors(category, [c for c in ALL_CATEGORIES if c != category])
    text = f"A sample of size n = {n} finds x = {f} failures, and p-hat is the proportion of successes. {RULE_SENTENCE} {_ASK}"
    working = [
        f"n×p-hat = n - x = {n} - {f} = {n - f} (compare to 10).",
        f"n×(1-p-hat) = x = {f} (compare to 10).",
    ]
    return dict(
        text=text,
        latex=text,
        answer=category,
        answer_text=category,
        answer_latex=latex_answer(category, plain=True),
        working=working,
        distractors=distractors,
    )


@generator(
    "p8c_normality_approx_phat",
    level=12,
    presets={
        "easy": dict(variant="direct"),
        "medium": dict(variant="from_counts"),
        "hard": dict(variant="borderline"),
    },
)
def p8c_normality_approx_phat(rng, variant="direct"):
    """Classify whether the normal approximation to the distribution of
    p-hat is valid, given bounded, fully specified n and phat (or counts)."""
    if variant == "direct":
        return _direct(rng)
    if variant == "from_counts":
        return _from_counts(rng)
    if variant == "borderline":
        return _borderline(rng)
    if variant == "from_failures":
        return _from_failures(rng)
    raise ValueError(f"p8c_normality_approx_phat: unknown variant {variant!r}")
