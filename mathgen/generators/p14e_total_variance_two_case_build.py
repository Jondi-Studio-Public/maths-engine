"""Generates `p14e_total_variance_two_case` questions (`ucv1-total-variance-finite`): total variance over a two-
case split, answered as a fraction.

``ucv1-total-variance-finite``, built by this module (topic ``p14e_total_variance_two_case``, answer:
fraction).

Kernel, held strictly to "two cases" (the skill's own words -- never
extended to three or more at any tier, including exam): ``X`` takes exactly
two values with probabilities ``p1`` and ``p2 = 1 - p1``. Given the
conditional means ``mu1, mu2`` and conditional variances ``sigma1^2,
sigma2^2`` for each case, compute the total variance via the law of total
variance::

    E[Var(Y|X)] = p1*sigma1^2 + p2*sigma2^2
    E[Y]        = p1*mu1 + p2*mu2
    Var(E[Y|X]) = p1*(mu1 - E[Y])^2 + p2*(mu2 - E[Y])^2
    Var(Y)      = E[Var(Y|X)] + Var(E[Y|X])

Every quantity is an exact `Fraction`.

Boundary against `finite-distribution-variance`: that skill computes the variance of ONE unconditional finite
distribution directly from its own outcomes and weights -- no conditioning, no decomposition. This skill is
explicitly the two-term DECOMPOSITION `Var(Y) = E[Var(Y|X)] + Var(E[Y|X])` from SUPPLIED conditional
summaries, never from a raw outcome table -- a learner could not mistake one for the other, since this prompt
never lists individual outcome values or their probabilities, only per-case means and variances.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import check, check_equivalent, generator, paren_negative, phrase, rational_text

# --- shared law-of-total-variance mechanics --------------------------------


def _e_var_within(p1: Fraction, p2: Fraction, s1: Fraction, s2: Fraction) -> Fraction:
    return p1 * s1 + p2 * s2


def _e_y(p1: Fraction, p2: Fraction, mu1: Fraction, mu2: Fraction) -> Fraction:
    return p1 * mu1 + p2 * mu2


def _var_between(p1: Fraction, p2: Fraction, mu1: Fraction, mu2: Fraction) -> Fraction:
    ey = _e_y(p1, p2, mu1, mu2)
    return p1 * (mu1 - ey) ** 2 + p2 * (mu2 - ey) ** 2


def _var_total(p1, p2, mu1, mu2, s1, s2) -> Fraction:
    return _e_var_within(p1, p2, s1, s2) + _var_between(p1, p2, mu1, mu2)


def _e_y2(p1, p2, mu1, mu2, s1, s2) -> Fraction:
    """E[Y^2] via the law of total expectation applied to Y^2 -- the
    independent-check route, genuinely different from the two-term
    decomposition used to construct the answer."""
    return p1 * (s1 + mu1 * mu1) + p2 * (s2 + mu2 * mu2)


def _fraction_payload(value: Fraction) -> tuple[dict, str]:
    value = Fraction(value)
    return {"p": value.numerator, "q": value.denominator}, rational_text(value)


def _pairwise_distinct_fractions(answer: Fraction, *candidates: Fraction) -> list[Fraction]:
    """Drop any candidate equal in VALUE to the answer or to an
    already-kept candidate -- `grade_fraction` compares by exact value, so dedup here
    must be by value too, never by string."""
    kept: list[Fraction] = [answer]
    out: list[Fraction] = []
    for c in candidates:
        if c in kept:
            continue
        kept.append(c)
        out.append(c)
        if len(out) == 4:
            break
    return out


# --- easy: p1 = 1/2 exactly, small integer conditional summaries ---------

_EASY_TEMPLATES = (
    "A random variable X takes 2 values with equal probability, case 1 (p1 = "
    "{p1}) and case 2 (p2 = {p2}). Given Y has conditional mean {mu1} and "
    "conditional variance {s1} in case 1, and conditional mean {mu2} and "
    "conditional variance {s2} in case 2, find the total variance Var(Y) "
    "using the law of total variance.",
    "For a two-case conditional model, case 1 has probability {p1} "
    "and case 2 has probability {p2}. In case 1, Y has conditional mean "
    "{mu1} and conditional variance {s1}; in case 2, Y has conditional mean "
    "{mu2} and conditional variance {s2}. Apply the law of total variance to "
    "find Var(Y).",
    "X is a two-valued random variable, taking case 1 or case 2 with equal "
    "probability ({p1} each). Conditional on case 1, Y has mean {mu1} and "
    "variance {s1}; conditional on case 2, Y has mean {mu2} and variance "
    "{s2}. Use the law of total variance to find Var(Y).",
)


def _easy(rng) -> dict:
    p1 = Fraction(1, 2)
    p2 = Fraction(1, 2)
    mu1 = rng.randint(-8, 8)
    mu2 = rng.choice([v for v in range(-8, 9) if v != mu1])
    s1 = rng.randint(1, 10)
    s2 = rng.randint(1, 10)
    return _build_evaluate(rng, p1, p2, mu1, mu2, s1, s2, _EASY_TEMPLATES, counts=None)


# --- medium: p1 a general fraction (never 1/2) ---------------------------

_MEDIUM_TEMPLATES = (
    "A random variable X takes 2 values: case 1 with probability {p1}, and "
    "case 2 with probability {p2}. Given Y has conditional mean {mu1} and "
    "conditional variance {s1} in case 1, and conditional mean {mu2} and "
    "conditional variance {s2} in case 2, find the total variance Var(Y) "
    "using the law of total variance.",
    "For a two-case conditional model, case 1 has probability {p1} "
    "and case 2 has probability {p2}. In case 1, Y has conditional mean "
    "{mu1} and conditional variance {s1}; in case 2, Y has conditional mean "
    "{mu2} and conditional variance {s2}. Apply the law of total variance to "
    "find Var(Y).",
)


def _medium(rng) -> dict:
    denom = rng.choice([3, 4, 5])
    numer = rng.choice([v for v in range(1, denom) if 2 * v != denom])
    p1 = Fraction(numer, denom)
    p2 = 1 - p1
    mu1 = rng.randint(-10, 10)
    mu2 = rng.choice([v for v in range(-10, 11) if v != mu1])
    s1 = rng.randint(1, 12)
    s2 = rng.randint(1, 12)
    return _build_evaluate(rng, p1, p2, mu1, mu2, s1, s2, _MEDIUM_TEMPLATES, counts=None)


# --- hard: population split given as counts n1 : n2 out of a total N -----

_HARD_TEMPLATES = (
    "A population of N = {N} splits into case 1 and case 2 in the ratio "
    "{n1} : {n2}. Given Y has conditional mean {mu1} and conditional "
    "variance {s1} in case 1, and conditional mean {mu2} and conditional "
    "variance {s2} in case 2, find the total variance Var(Y) using the law "
    "of total variance.",
    "Out of a population of N = {N}, {n1} individuals belong to case 1 and "
    "{n2} to case 2. In case 1, Y has conditional mean {mu1} and conditional "
    "variance {s1}; in case 2, Y has conditional mean {mu2} and conditional "
    "variance {s2}. Apply the law of total variance to find Var(Y).",
    "A population of size N = {N} is split {n1} : {n2} between case 1 and "
    "case 2. Conditional on case 1, Y has mean {mu1} and variance {s1}; "
    "conditional on case 2, Y has mean {mu2} and variance {s2}. Use the law "
    "of total variance to find Var(Y).",
)


def _draw_counts(rng) -> tuple[int, int, int]:
    total = rng.randint(10, 40)
    # n1, n2 both >= 2: a count of exactly 1 trips a "1 individuals" singular/
    # plural text-quality defect in the "{n} individuals belong to..." template.
    n1 = rng.randint(2, total - 2)
    n2 = total - n1
    return n1, n2, total


def _hard(rng) -> dict:
    n1, n2, N = _draw_counts(rng)
    p1 = Fraction(n1, N)
    p2 = Fraction(n2, N)
    mu1 = rng.randint(-10, 12)
    mu2 = rng.choice([v for v in range(-10, 13) if v != mu1])
    s1 = rng.randint(1, 14)
    s2 = rng.randint(1, 14)
    return _build_evaluate(rng, p1, p2, mu1, mu2, s1, s2, _HARD_TEMPLATES, counts=(n1, n2, N))


def _build_evaluate(rng, p1, p2, mu1, mu2, s1, s2, templates, counts):
    p1, p2 = Fraction(p1), Fraction(p2)
    mu1, mu2, s1, s2 = Fraction(mu1), Fraction(mu2), Fraction(s1), Fraction(s2)
    check(p1 + p2 == 1, "conditional probabilities do not sum to 1")
    check(s1 > 0 and s2 > 0, "conditional variances must be positive")
    check(mu1 != mu2, "conditional means must differ")

    e_within = _e_var_within(p1, p2, s1, s2)
    e_between = _var_between(p1, p2, mu1, mu2)
    answer = e_within + e_between

    # An answer that lands on a bare integer is the sole integer-shaped option against otherwise
    # fraction-shaped distractors -- a shape giveaway. Nudge s2 (bounded) until the total variance
    # is a genuine non-integer fraction; s2 stays positive throughout since it only ever increases.
    attempts = 0
    while answer.denominator == 1 and attempts < 8:
        s2 = s2 + 1
        e_within = _e_var_within(p1, p2, s1, s2)
        answer = e_within + e_between
        attempts += 1

    # Independent check: E[Y^2] route, genuinely different from the two-term
    # decomposition used above.
    e_y2 = _e_y2(p1, p2, mu1, mu2, s1, s2)
    e_y = _e_y(p1, p2, mu1, mu2)
    check_equivalent(e_y2 - e_y * e_y, answer, "E[Y^2] route disagrees with the two-term decomposition")

    # Misconceptions, as raw Fraction values.
    drops_within = e_between  # "total variance = variance of the conditional means"
    drops_between = e_within  # "total variance = average conditional variance" (the more common slip)

    ey_uw = (mu1 + mu2) / 2
    var_between_uw = p1 * (mu1 - ey_uw) ** 2 + p2 * (mu2 - ey_uw) ** 2
    unweighted_ey = e_within + var_between_uw

    ey_swap = p2 * mu1 + p1 * mu2
    var_between_swap = p2 * (mu1 - ey_swap) ** 2 + p1 * (mu2 - ey_swap) ** 2
    e_within_swap = p2 * s1 + p1 * s2
    swapped_weights = e_within_swap + var_between_swap

    candidates = [drops_within, drops_between, unweighted_ey, swapped_weights]

    if counts is not None:
        n1, n2, N = counts
        e_within_raw = n1 * s1 + n2 * s2
        ey_raw = n1 * mu1 + n2 * mu2
        var_between_raw = n1 * (mu1 - ey_raw) ** 2 + n2 * (mu2 - ey_raw) ** 2
        counts_not_divided = e_within_raw + var_between_raw
        candidates.append(counts_not_divided)

    distinct_candidates = _pairwise_distinct_fractions(answer, *candidates)

    answer_payload, answer_text = _fraction_payload(answer)
    distractor_texts = [_fraction_payload(c)[1] for c in distinct_candidates]

    slots = dict(p1=rational_text(p1), p2=rational_text(p2), mu1=mu1, mu2=mu2, s1=s1, s2=s2)
    if counts is not None:
        n1, n2, N = counts
        slots.update(n1=n1, n2=n2, N=N)
    text = phrase(rng, templates, **slots)

    working = [
        f"E[Var(Y|X)] = {rational_text(p1)}×{s1} + {rational_text(p2)}×{s2} = {rational_text(e_within)}.",
        f"E[Y] = {rational_text(p1)}×{mu1} + {rational_text(p2)}×{mu2} = {rational_text(e_y)}.",
        f"Var(E[Y|X]) = {rational_text(p1)}×({paren_negative(mu1)} - {paren_negative(e_y)})^2 + "
        f"{rational_text(p2)}×({paren_negative(mu2)} - {paren_negative(e_y)})^2 = {rational_text(e_between)}.",
        f"Var(Y) = {rational_text(e_within)} + {rational_text(e_between)} = {answer_text}.",
    ]

    return dict(
        text=text,
        answer=answer_payload,
        answer_text=answer_text,
        working=working,
        distractors=distractor_texts[:4],
    )


# --- exam: inversion -- Var(Y) stated, sigma2^2 withheld ------------------

_EXAM_TEMPLATES_PROB = (
    "A random variable X takes 2 values: case 1 with probability {p1}, and "
    "case 2 with probability {p2}. In case 1, Y has conditional mean {mu1} "
    "and conditional variance {s1}. In case 2, Y has conditional mean {mu2}, "
    "and its total variance is Var(Y) = {var_total}. Find the conditional "
    "variance of Y in case 2.",
    "For a two-case conditional model, case 1 has probability {p1} "
    "and case 2 has probability {p2}. Case 1 has conditional mean {mu1} and "
    "conditional variance {s1}; case 2 has conditional mean {mu2}. The total "
    "variance is Var(Y) = {var_total}. Determine the conditional variance of "
    "Y in case 2.",
)

_EXAM_TEMPLATES_COUNTS = (
    "A population of N = {N} splits into case 1 and case 2 in the ratio "
    "{n1} : {n2}. In case 1, Y has conditional mean {mu1} and conditional "
    "variance {s1}. In case 2, Y has conditional mean {mu2}, and its total "
    "variance is Var(Y) = {var_total}. Find the conditional variance of Y in "
    "case 2.",
    "Out of a population of N = {N}, {n1} individuals belong to case 1 and "
    "{n2} to case 2. Case 1 has conditional mean {mu1} and conditional "
    "variance {s1}; case 2 has conditional mean {mu2}. The total variance is "
    "Var(Y) = {var_total}. Determine the conditional variance of Y in case 2.",
)


def _solve_s2(target: Fraction, p1: Fraction, p2: Fraction, mu1: Fraction, mu2: Fraction, s1: Fraction) -> Fraction:
    """Solve `target = p1*s1 + p2*s2 + Var(E[Y|X])` for s2 (linear in s2)."""
    between = _var_between(p1, p2, mu1, mu2)
    return (target - between - p1 * s1) / p2


def _exam(rng) -> dict:
    use_counts = rng.choice([True, False])
    if use_counts:
        n1, n2, N = _draw_counts(rng)
        p1 = Fraction(n1, N)
        p2 = Fraction(n2, N)
    else:
        denom = rng.choice([3, 4, 5])
        numer = rng.choice([v for v in range(1, denom) if 2 * v != denom])
        p1 = Fraction(numer, denom)
        p2 = 1 - p1

    mu1 = rng.randint(-10, 10)
    mu2 = rng.choice([v for v in range(-10, 11) if v != mu1])
    s1 = rng.randint(1, 12)
    # s2 is drawn as a GENUINE (non-integer) fraction, never a bare integer: otherwise the withheld
    # answer is always the sole integer-shaped option against fraction-shaped distractors, a shape
    # giveaway.
    s2_denom = rng.choice([2, 3, 4])
    s2_numer = rng.choice([v for v in range(1, 12 * s2_denom) if v % s2_denom != 0])
    s2_true = Fraction(s2_numer, s2_denom)
    mu1, mu2, s1 = Fraction(mu1), Fraction(mu2), Fraction(s1)

    check(p1 + p2 == 1, "exam probabilities do not sum to 1")
    check(mu1 != mu2, "exam conditional means must differ")

    var_total_true = _var_total(p1, p2, mu1, mu2, s1, s2_true)

    # Independent check: solve for s2 via the isolate-and-divide formula, then
    # substitute the SOLVED value back into the forward two-term formula and
    # confirm it reproduces the stated target Var(Y) exactly -- never reusing
    # the isolate-and-divide arithmetic itself for this confirmation.
    solved_s2 = _solve_s2(var_total_true, p1, p2, mu1, mu2, s1)
    check_equivalent(solved_s2, s2_true, "exam construction: solved s2 does not match the drawn s2")
    reforward = _var_total(p1, p2, mu1, mu2, s1, solved_s2)
    check_equivalent(
        reforward, var_total_true, "exam independent check: forward substitution of solved s2 did not reproduce Var(Y)"
    )

    answer = s2_true

    # Misconceptions: each is the s2 a learner would solve for under a wrong equation for Var(Y),
    # given the SAME stated target and other values. drops-the-within-term has no sigma2 dependence
    # under its own wrong equation (Var(Y) = Var(E[Y|X]) only) and so is not invertible here --
    # honestly omitted at this tier.
    between = _var_between(p1, p2, mu1, mu2)

    # drops-the-between-term: wrong equation Var(Y) = p1*s1 + p2*s2 (no
    # between term at all) -> s2 = (V - p1*s1) / p2.
    drops_between_s2 = (var_total_true - p1 * s1) / p2

    # unweighted-EY: the between term used in solving is computed with the
    # unweighted mean (mu1+mu2)/2 instead of the true E[Y].
    ey_uw = (mu1 + mu2) / 2
    between_uw = p1 * (mu1 - ey_uw) ** 2 + p2 * (mu2 - ey_uw) ** 2
    unweighted_ey_s2 = (var_total_true - between_uw - p1 * s1) / p2

    # swapped-weights: p1/p2 transposed throughout the wrong equation.
    ey_swap = p2 * mu1 + p1 * mu2
    between_swap = p2 * (mu1 - ey_swap) ** 2 + p1 * (mu2 - ey_swap) ** 2
    swapped_weights_s2 = (var_total_true - between_swap - p2 * s1) / p1

    candidates = [drops_between_s2, unweighted_ey_s2, swapped_weights_s2]

    if use_counts:
        # counts-not-divided: solves using the raw counts n1, n2 in place of
        # p1, p2 throughout (including the between term).
        ey_raw = n1 * mu1 + n2 * mu2
        between_raw = n1 * (mu1 - ey_raw) ** 2 + n2 * (mu2 - ey_raw) ** 2
        counts_not_divided_s2 = (var_total_true - between_raw - n1 * s1) / n2
        candidates.append(counts_not_divided_s2)

    distinct_candidates = _pairwise_distinct_fractions(answer, *candidates)

    answer_payload, answer_text = _fraction_payload(answer)
    distractor_texts = [_fraction_payload(c)[1] for c in distinct_candidates]

    slots = dict(
        p1=rational_text(p1), p2=rational_text(p2), mu1=mu1, mu2=mu2, s1=s1, var_total=rational_text(var_total_true)
    )
    templates = _EXAM_TEMPLATES_COUNTS if use_counts else _EXAM_TEMPLATES_PROB
    if use_counts:
        slots.update(n1=n1, n2=n2, N=N)
    text = phrase(rng, templates, **slots)

    working = [
        f"Var(E[Y|X]) = {rational_text(p1)}×({mu1} - E[Y])^2 + "
        f"{rational_text(p2)}×({mu2} - E[Y])^2 = {rational_text(between)}.",
        f"{rational_text(var_total_true)} = {rational_text(p1)}×{s1} + "
        f"{rational_text(p2)}×sigma2^2 + {rational_text(between)}.",
        f"Solving: sigma2^2 = ({rational_text(var_total_true)} - {rational_text(between)} - "
        f"{rational_text(p1)}×{s1}) / {rational_text(p2)} = {answer_text}.",
    ]

    return dict(
        text=text,
        answer=answer_payload,
        answer_text=answer_text,
        working=working,
        distractors=distractor_texts[:4],
    )


_TIER_BUILDERS = {
    "easy": _easy,
    "medium": _medium,
    "hard": _hard,
    "exam": _exam,
}


@generator(
    "p14e_total_variance_two_case",
    level=15,
    presets={
        "easy": dict(kind="easy"),
        "medium": dict(kind="medium"),
        "hard": dict(kind="hard"),
        "exam": dict(kind="exam"),
    },
)
def p14e_total_variance_two_case(rng, kind: str = "easy"):
    """Apply the law of total variance to a two-case conditional model,
    computing (or, at the exam tier, inverting for) the total variance
    Var(Y) = E[Var(Y|X)] + Var(E[Y|X]). See the module docstring for the
    kernel and the boundary against `finite-distribution-variance`."""
    return _TIER_BUILDERS[kind](rng)
