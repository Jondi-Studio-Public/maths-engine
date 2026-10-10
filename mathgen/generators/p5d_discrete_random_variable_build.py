"""Senior non-uniform discrete random variable.

Generate questions about computing missing probabilities and expected values
in non-uniform discrete random variable distributions.
"""

from fractions import Fraction
import random

from mathgen.core import generator, VerificationError, pick_distractors, rational_text


def _sample_distribution(rng, k=None, D=None):
    """Sample a non-uniform discrete random variable distribution.

    Returns (xs, numerators, D) where:
    - xs: sorted list of k distinct small integer outcomes
    - numerators: list of k positive integer numerators (sum to D)
    - D: common denominator

    Resamples up to 30 attempts if any numerator is 0 or if all numerators
    are equal (all-uniform check to ensure non-uniformity).
    """
    if k is None:
        k = rng.choice([3, 4])
    if D is None:
        D = rng.choice([8, 10, 12])

    for attempt in range(30):
        # Sample x values: k distinct integers from range(0, 5)
        xs = sorted(rng.sample(range(5), k))

        # Sample numerators via stars-and-bars: choose k-1 cut points in 1..D-1
        if k == 1:
            numerators = [D]
        else:
            cut_points = sorted(rng.sample(range(1, D), k - 1))
            # Consecutive differences
            numerators = (
                [cut_points[0]]
                + [cut_points[i] - cut_points[i - 1] for i in range(1, len(cut_points))]
                + [D - cut_points[-1]]
            )

        # Check: no numerator is 0
        if any(n == 0 for n in numerators):
            continue

        # Check: not all numerators equal (non-uniform requirement)
        if len(set(numerators)) == 1:
            continue

        return xs, numerators, D

    # Fallback (should rarely happen)
    raise RuntimeError("Could not sample non-uniform distribution after 30 attempts")


def _check_missing_probability(xs, numerators, D, missing_idx):
    """Independent check for missing probability via pure integer arithmetic.

    Returns the missing numerator n_m computed as D - sum(other numerators).
    """
    sum_other = sum(n for i, n in enumerate(numerators) if i != missing_idx)
    n_m = D - sum_other

    # Verify the sum of all numerators equals D
    assert sum(numerators) == D, f"numerators {numerators} do not sum to {D}"

    return n_m


def _check_expectation(xs, numerators, D):
    """Independent check for expectation via integer arithmetic first, then division.

    Returns E(X) as a Fraction computed from sum(x_i * n_i) / D.
    """
    # Pure integer computation first
    numerator_E = sum(xs[i] * numerators[i] for i in range(len(xs)))

    # Verify the sum of all numerators equals D
    assert sum(numerators) == D, f"numerators {numerators} do not sum to {D}"

    return Fraction(numerator_E, D)


@generator(
    "non_uniform_discrete_random_variable",
    level=11,
    presets={
        "easy": {"tier": "easy"},
        "medium": {"tier": "medium"},
        "hard": {"tier": "hard"},
    },
)
def non_uniform_discrete_random_variable(rng, tier="easy"):
    """Generate a non-uniform discrete random variable question.

    Easy: missing probability (one ? in table), question: P(X = x_m)?
    Medium: all probabilities shown, question: E(X)?
    Hard: missing probability AND question: E(X)?
    """
    if tier == "easy":
        return _generate_easy(rng)
    elif tier == "medium":
        return _generate_medium(rng)
    elif tier == "hard":
        return _generate_hard(rng)
    else:
        raise ValueError(f"unknown tier: {tier}")


def _generate_easy(rng):
    """Easy tier: missing probability. Answer: missing numerator over D."""
    xs, numerators, D = _sample_distribution(rng, k=rng.choice([3, 4]), D=rng.choice([8, 10, 12]))

    # Choose which probability to hide
    missing_idx = rng.randrange(len(xs))
    x_m = xs[missing_idx]
    n_m = numerators[missing_idx]

    # Build distribution text as outcome-probability pairs
    pairs_list = []
    for i, x in enumerate(xs):
        if i == missing_idx:
            pairs_list.append(f"P(X = {x}) = ?")
        else:
            pairs_list.append(f"P(X = {x}) = {numerators[i]}/{D}")
    pairs_str = ", ".join(pairs_list)

    question_text = (
        f"A discrete random variable X has this probability distribution: {pairs_str}. "
        f"The probabilities sum to 1. "
        f"What is P(X = {x_m})?"
    )

    # Independent check: compute n_m from other numerators
    check_n_m = _check_missing_probability(xs, numerators, D, missing_idx)
    assert check_n_m == n_m, f"check mismatch: {check_n_m} != {n_m}"

    # Ensure answer is reduced
    frac = Fraction(n_m, D)
    answer = {"p": frac.numerator, "q": frac.denominator}
    answer_text = rational_text(frac)

    # Working: show sum-to-1 step
    other_fracs = [f"{numerators[i]}/{D}" for i in range(len(xs)) if i != missing_idx]
    working = [f"The probabilities must sum to 1, so P(X = {x_m}) = 1 - ({' + '.join(other_fracs)}) = {answer_text}."]

    # Distractors
    # 1. assumes-uniform: 1/k as if all outcomes equally likely
    assumes_uniform = Fraction(1, len(xs))

    # 2. reports-known-sum-not-complement: D - n_m (sum of other numerators)
    sum_other = D - n_m
    reports_known_sum = Fraction(sum_other, D)

    # 3. off-by-one-numerator: n_m +/- 1
    if n_m > 1:
        off_by_one = Fraction(n_m - 1, D)
    else:
        off_by_one = Fraction(n_m + 1, D)

    candidate_distractors = [
        rational_text(assumes_uniform),
        rational_text(reports_known_sum),
        rational_text(off_by_one),
    ]
    distractors = pick_distractors(answer_text, candidate_distractors, n=3)

    # Verify distractors are distinct Fractions
    answer_frac = Fraction(n_m, D)
    for dist in distractors:
        dist_frac = Fraction(dist)
        assert dist_frac != answer_frac, f"distractor {dist} equals answer {answer_text}"

    return dict(
        text=question_text,
        latex=f"plain:{question_text}",
        answer=answer,
        answer_text=answer_text,
        working=working,
        distractors=distractors,
    )


def _generate_medium(rng):
    """Medium tier: all probabilities shown. Answer: E(X) = sum(x_i * p_i)."""
    xs, numerators, D = _sample_distribution(rng, k=rng.choice([3, 4]), D=rng.choice([8, 10, 12]))

    # Build distribution text as outcome-probability pairs
    pairs_list = [f"P(X = {x}) = {n}/{D}" for x, n in zip(xs, numerators)]
    pairs_str = ", ".join(pairs_list)

    question_text = (
        f"A discrete random variable X has this probability distribution: {pairs_str}. "
        "What is E(X), the expected value of X?"
    )

    # Compute E(X) = sum(x_i * p_i) via weighted sum approach (fractions accumulate)
    E_frac = Fraction(0)
    for x, n in zip(xs, numerators):
        E_frac += x * Fraction(n, D)

    # Independent check: use pure integer first, then divide
    check_E = _check_expectation(xs, numerators, D)
    assert check_E == E_frac, f"check mismatch: {check_E} != {E_frac}"

    # Ensure answer is reduced (Fraction auto-reduces)
    E_frac_reduced = Fraction(E_frac.numerator, E_frac.denominator)
    answer = {"p": E_frac_reduced.numerator, "q": E_frac_reduced.denominator}
    answer_text = rational_text(E_frac_reduced)

    # Working: show weighted-sum calculation
    terms = []
    for x, n in zip(xs, numerators):
        terms.append(f"{x} times {n}/{D}")
    working = [f"E(X) = {' plus '.join(terms)} = {answer_text}."]

    # Distractors for expectation
    # 1. unweighted-mean: sum(xs) / k (ignore probabilities)
    unweighted = Fraction(sum(xs), len(xs))

    # 2. mode-instead-of-mean: x_i with largest numerator (use smallest x as tiebreaker)
    max_n = max(numerators)
    mode_idx = min(i for i, n in enumerate(numerators) if n == max_n)
    mode_value = Fraction(xs[mode_idx], 1)

    # 3. off-by-one-numerator: E_num +/- 1 over D
    E_num = E_frac.numerator
    if E_num > 1:
        off_by_one = Fraction(E_num - 1, D)
    else:
        off_by_one = Fraction(E_num + 1, D)

    candidate_distractors = [
        rational_text(unweighted),
        rational_text(mode_value),
        rational_text(off_by_one),
    ]
    distractors = pick_distractors(answer_text, candidate_distractors, n=3)

    # Verify distractors are distinct
    for dist in distractors:
        dist_frac = Fraction(dist)
        assert dist_frac != E_frac, f"distractor {dist} equals answer {answer_text}"

    return dict(
        text=question_text,
        latex=f"plain:{question_text}",
        answer=answer,
        answer_text=answer_text,
        working=working,
        distractors=distractors,
    )


def _generate_hard(rng):
    """Hard tier: missing probability AND compute E(X).

    Learner must first recover the missing probability, then use it to
    compute the expectation. Uses SAME distractor misconceptions as medium
    but computed with the recovered numerator.
    """
    xs, numerators, D = _sample_distribution(rng, k=rng.choice([3, 4]), D=rng.choice([8, 10, 12]))

    # Choose which probability to hide
    missing_idx = rng.randrange(len(xs))
    x_m = xs[missing_idx]
    n_m = numerators[missing_idx]

    # Build distribution text as outcome-probability pairs with one missing
    pairs_list = []
    for i, x in enumerate(xs):
        if i == missing_idx:
            pairs_list.append(f"P(X = {x}) = ?")
        else:
            pairs_list.append(f"P(X = {x}) = {numerators[i]}/{D}")
    pairs_str = ", ".join(pairs_list)

    question_text = (
        f"A discrete random variable X has this probability distribution: {pairs_str}. "
        "What is E(X), the expected value of X?"
    )

    # Compute correct answer using the full distribution
    E_frac = Fraction(0)
    for x, n in zip(xs, numerators):
        E_frac += x * Fraction(n, D)

    # Independent check: recover the missing numerator first, then compute E
    check_n_m = _check_missing_probability(xs, numerators, D, missing_idx)
    assert check_n_m == n_m, f"check mismatch on n_m: {check_n_m} != {n_m}"

    # Then compute E with recovered numerator
    check_numerators = numerators.copy()
    check_E = _check_expectation(xs, check_numerators, D)
    assert check_E == E_frac, f"check mismatch on E(X): {check_E} != {E_frac}"

    # Ensure answer is reduced (Fraction auto-reduces)
    E_frac_reduced = Fraction(E_frac.numerator, E_frac.denominator)
    answer = {"p": E_frac_reduced.numerator, "q": E_frac_reduced.denominator}
    answer_text = rational_text(E_frac_reduced)

    # Working: show both steps
    other_fracs = [f"{numerators[i]}/{D}" for i in range(len(xs)) if i != missing_idx]
    step1 = f"First, find the missing probability: P(X = {x_m}) = 1 - ({' + '.join(other_fracs)}) = {rational_text(Fraction(n_m, D))}."

    terms = []
    for x, n in zip(xs, numerators):
        terms.append(f"{x} times {n}/{D}")
    step2 = f"Then compute E(X) = {' plus '.join(terms)} = {answer_text}."

    working = [step1, step2]

    # Distractors: use the SAME formulas as medium tier but all are different
    # numeric values because the recovered n_m differs from any medium sample
    # 1. unweighted-mean: sum(xs) / k
    unweighted = Fraction(sum(xs), len(xs))

    # 2. mode-instead-of-mean: x_i with largest numerator (including recovered one)
    max_n = max(numerators)
    mode_idx = min(i for i, n in enumerate(numerators) if n == max_n)
    mode_value = Fraction(xs[mode_idx], 1)

    # 3. off-by-one-numerator: E_num +/- 1 over D
    E_num = E_frac.numerator
    if E_num > 1:
        off_by_one = Fraction(E_num - 1, D)
    else:
        off_by_one = Fraction(E_num + 1, D)

    candidate_distractors = [
        rational_text(unweighted),
        rational_text(mode_value),
        rational_text(off_by_one),
    ]
    distractors = pick_distractors(answer_text, candidate_distractors, n=3)

    # Verify distractors are distinct
    for dist in distractors:
        dist_frac = Fraction(dist)
        assert dist_frac != E_frac, f"distractor {dist} equals answer {answer_text}"

    return dict(
        text=question_text,
        latex=f"plain:{question_text}",
        answer=answer,
        answer_text=answer_text,
        working=working,
        distractors=distractors,
    )
