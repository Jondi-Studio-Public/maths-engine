"""Small, exact generators for a statistics and statistical-learning pilot.

This module intentionally covers only finite, closed-form slices.  It does
not claim to assess measure-theoretic probability, asymptotic statistics,
general regression diagnostics, or a full optimisation course.  Every topic
uses exact integer/rational arithmetic and the existing answer contracts.
"""

from __future__ import annotations

from fractions import Fraction
from math import comb

from ..core import check, generator, pick_distractors, paren_negative


def _fraction_payload(value: Fraction) -> tuple[dict[str, int], str]:
    value = Fraction(value)
    text = str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"
    return {"p": value.numerator, "q": value.denominator}, text


@generator(
    "hypergeometric_probability",
    level=14,
    presets={
        "easy": dict(population_hi=12),
        "medium": dict(population_hi=24),
        "hard": dict(population_hi=45),
    },
)
def hypergeometric_probability(rng, population_hi=24):
    """Find an exact without-replacement success probability."""
    population = rng.randint(6, population_hi)
    successes = rng.randint(1, population - 1)
    draws = rng.randint(2, population - 2)
    lower = max(0, draws - (population - successes))
    upper = min(draws, successes)
    target = rng.randint(lower, upper)

    def probability(k: int) -> Fraction:
        if not lower <= k <= upper:
            return Fraction(0)
        return Fraction(comb(successes, k) * comb(population - successes, draws - k), comb(population, draws))

    answer_value = probability(target)
    exhaustive_total = sum(probability(k) for k in range(lower, upper + 1))
    direct_count = comb(successes, target) * comb(population - successes, draws - target)
    check(
        answer_value == Fraction(direct_count, comb(population, draws)),
        "hypergeometric combination calculation mismatch",
    )
    check(exhaustive_total == 1, "hypergeometric probabilities do not sum to one")
    check(0 <= answer_value <= 1, "invalid hypergeometric probability")

    answer, answer_text = _fraction_payload(answer_value)
    with_replacement = Fraction(
        comb(draws, target) * successes**target * (population - successes) ** (draws - target), population**draws
    )
    alternatives = [
        _fraction_payload(probability(target - 1))[1],
        _fraction_payload(probability(target + 1))[1],
        _fraction_payload(with_replacement)[1],
        _fraction_payload(1 - answer_value)[1],
        _fraction_payload(answer_value + Fraction(1, comb(population, draws)))[1],
    ]
    return dict(
        text=(
            f"A population has {population} objects, {successes} of them successes. "
            f"If {draws} objects are drawn without replacement, find P(X={target})."
        ),
        latex=rf"P(X={target})=\frac{{\binom{{{successes}}}{{{target}}}\binom{{{population - successes}}}{{{draws - target}}}}}{{\binom{{{population}}}{{{draws}}}}}",
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"Favourable selections = C({successes},{target})·C({population - successes},{draws - target}) = {direct_count}.",
            f"All selections = C({population},{draws}); therefore P(X={target}) = {answer_text}.",
        ],
        distractors=pick_distractors(answer_text, alternatives),
    )


@generator(
    "finite_distribution_variance",
    level=14,
    presets={
        "easy": dict(outcome_hi=5, weight_hi=3, outcomes=3),
        "medium": dict(outcome_hi=10, weight_hi=6, outcomes=4),
        "hard": dict(outcome_hi=18, weight_hi=10, outcomes=5),
    },
)
def finite_distribution_variance(rng, outcome_hi=10, weight_hi=6, outcomes=4):
    """Compute the exact variance of a finite discrete distribution."""
    values = sorted(rng.sample(range(-outcome_hi, outcome_hi + 1), outcomes))
    weights = [rng.randint(1, weight_hi) for _ in values]
    total = sum(weights)
    mean = Fraction(sum(x * w for x, w in zip(values, weights)), total)
    second_moment = Fraction(sum(x * x * w for x, w in zip(values, weights)), total)
    variance = second_moment - mean * mean
    centred = Fraction(sum((x - mean) ** 2 * w for x, w in zip(values, weights)), total)
    check(variance == centred and variance >= 0, "finite variance calculation mismatch")
    check(sum(weights) == total and total > 0, "invalid distribution weights")

    answer, answer_text = _fraction_payload(variance)
    alternatives = [
        _fraction_payload(mean)[1],
        _fraction_payload(second_moment)[1],
        _fraction_payload(mean * mean)[1],
        _fraction_payload(variance + 1)[1],
        _fraction_payload(-variance)[1],
    ]
    distribution = ", ".join(f"P(X={x})={w}/{total}" for x, w in zip(values, weights))
    return dict(
        text=f"A finite random variable has {distribution}. Find Var(X).",
        latex=rf"\operatorname{{Var}}(X)=E[X^2]-E[X]^2",
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[f"E[X] = {mean}; E[X²] = {second_moment}.", f"Var(X) = {second_moment} − ({mean})² = {answer_text}."],
        distractors=pick_distractors(answer_text, alternatives),
    )


@generator(
    "bernoulli_mle",
    level=14,
    presets={
        "easy": dict(n_hi=10),
        "medium": dict(n_hi=25),
        "hard": dict(n_hi=60),
    },
)
def bernoulli_mle(rng, n_hi=25):
    """Find the Bernoulli maximum-likelihood estimate p-hat = k/n."""
    n = rng.randint(4, n_hi)
    k = rng.randint(1, n - 1)
    estimate = Fraction(k, n)
    score_numerator = k * (1 - estimate) - (n - k) * estimate
    sample_mean = Fraction(k, n)
    check(score_numerator == 0, "Bernoulli likelihood score does not vanish at estimate")
    check(estimate == sample_mean and 0 < estimate < 1, "Bernoulli MLE mismatch")

    answer, answer_text = _fraction_payload(estimate)
    alternatives = [
        _fraction_payload(Fraction(n - k, n))[1],
        _fraction_payload(Fraction(k, n + 1))[1],
        _fraction_payload(Fraction(k + 1, n))[1],
        _fraction_payload(Fraction(k, n - 1))[1],
        str(round(k / n, 2)),
    ]
    return dict(
        text=f"In {n} independent Bernoulli trials there are {k} successes. Find the MLE of p.",
        latex=rf"\hat p=\arg\max_{{0<p<1}}p^{{{k}}}(1-p)^{{{n - k}}}",
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"The log-likelihood score is k/p − (n−k)/(1−p).",
            f"Setting it to zero gives p-hat = k/n = {k}/{n} = {answer_text}.",
        ],
        distractors=pick_distractors(answer_text, alternatives),
    )


@generator(
    "least_squares_line",
    level=15,
    presets={
        "easy": dict(coefficient_hi=4, intercept_hi=6),
        "medium": dict(coefficient_hi=8, intercept_hi=12),
        "hard": dict(coefficient_hi=14, intercept_hi=20),
    },
)
def least_squares_line(rng, coefficient_hi=8, intercept_hi=12):
    """Recover the least-squares slope and intercept from a small data set."""
    slope = rng.randint(-coefficient_hi, coefficient_hi)
    intercept = rng.randint(-intercept_hi, intercept_hi)
    while slope == intercept:
        intercept = rng.randint(-intercept_hi, intercept_hi)
    x_values = [0, 1, 2, 3]
    # These centred residuals make a genuinely noisy data set while preserving
    # the constructed least-squares line exactly: sum(e)=sum(xe)=0.
    residuals = [1, -1, -1, 1]
    y_values = [slope * x + intercept + e for x, e in zip(x_values, residuals)]
    x_bar = Fraction(sum(x_values), len(x_values))
    y_bar = Fraction(sum(y_values), len(y_values))
    s_xx = sum((x - x_bar) ** 2 for x in x_values)
    s_xy = sum((x - x_bar) * (y - y_bar) for x, y in zip(x_values, y_values))
    fitted_slope = Fraction(s_xy, s_xx)
    fitted_intercept = y_bar - fitted_slope * x_bar
    check(fitted_slope == slope and fitted_intercept == intercept, "least-squares fit mismatch")
    check(
        sum(residuals) == 0 and sum(x * e for x, e in zip(x_values, residuals)) == 0,
        "constructed residuals do not preserve fit",
    )

    answer = {"m": slope, "b": intercept}
    answer_text = f"m = {slope}, b = {intercept}"
    alternatives = [
        f"m = {intercept}, b = {slope}",
        f"m = {-slope}, b = {intercept}",
        f"m = {slope}, b = {-intercept}",
        f"m = {slope}, b = {y_values[0]}",
        f"m = {Fraction(sum(y_values), len(y_values))}, b = {intercept}",
    ]
    points = ", ".join(f"({x}, {y})" for x, y in zip(x_values, y_values))
    return dict(
        text=f"For the data points {points}, find the least-squares line y = mx + b.",
        latex=rf"\hat y=\hat m x+\hat b,\quad \text{{data}}=\{{{points}\}}",
        answer=answer,
        answer_text=answer_text,
        answer_latex=rf"m={slope},\;b={intercept}",
        working=[
            f"x-bar = {x_bar}, y-bar = {y_bar}; Sxx = {s_xx}, Sxy = {s_xy}.",
            f"m-hat = Sxy/Sxx = {slope}, b-hat = y-bar − m-hat·x-bar = {intercept}.",
        ],
        distractors=pick_distractors(answer_text, alternatives),
    )


@generator(
    "gradient_descent_quadratic",
    level=15,
    presets={
        "easy": dict(target_hi=5, eta_index=0),
        "medium": dict(target_hi=10, eta_index=1),
        "hard": dict(target_hi=18, eta_index=2),
    },
)
def gradient_descent_quadratic(rng, target_hi=10, eta_index=1):
    """Take one exact gradient-descent step on L(theta)=1/2(theta-t)^2."""
    target = rng.randint(-target_hi, target_hi)
    theta = rng.randint(-target_hi, target_hi)
    while theta == target:
        theta = rng.randint(-target_hi, target_hi)
    eta = [Fraction(1, 2), Fraction(1, 4), Fraction(3, 4)][eta_index % 3]
    gradient = theta - target
    theta_new = Fraction(theta) - eta * gradient
    direct = Fraction(theta) - eta * (theta - target)
    check(theta_new == direct, "gradient-descent update mismatch")
    check(gradient == theta - target, "quadratic loss gradient mismatch")
    check(
        Fraction(1, 2) * (theta_new - target) ** 2 <= Fraction(1, 2) * (theta - target) ** 2,
        "chosen step did not reduce the quadratic loss",
    )

    answer, answer_text = _fraction_payload(theta_new)
    alternatives = [
        str(theta),
        str(target),
        _fraction_payload(Fraction(theta) + eta * gradient)[1],
        _fraction_payload(Fraction(theta) - gradient)[1],
        _fraction_payload(Fraction(gradient))[1],
    ]
    return dict(
        text=(
            f"For L(theta) = 1/2(theta − {paren_negative(target)})², start at theta = {theta} "
            f"and use learning rate eta = {eta}. Take one gradient-descent step."
        ),
        latex=rf"L(\theta)=\frac12(\theta-{target})^2,\quad \theta_{{new}}=\theta-\eta\nabla L(\theta)",
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"Gradient: dL/dtheta = theta − {paren_negative(target)} = {gradient}.",
            f"theta-new = {theta} − ({eta})({gradient}) = {answer_text}.",
        ],
        distractors=pick_distractors(answer_text, alternatives),
    )
