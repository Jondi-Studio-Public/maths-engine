"""Small, exact generators for a statistics and statistical-learning pilot.

This module intentionally covers only finite, closed-form slices.  It does
not claim to assess measure-theoretic probability, asymptotic statistics,
general regression diagnostics, or a full optimisation course.  Every topic
uses exact integer/rational arithmetic and the existing answer contracts.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations, product
from math import comb

from ..core import GRADER_TOLERANCE, check, draw_until, equals_chain, generator, pick_distractors, paren_negative


def _fraction_payload(value: Fraction) -> tuple[dict[str, int], str]:
    value = Fraction(value)
    text = str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"
    return {"p": value.numerator, "q": value.denominator}, text


def _hypergeometric_at_least(rng, population_hi):
    """P(X>=1) by the complement 1 - C(N-K,n)/C(N,n); private helper."""
    population = rng.randint(10, max(10, population_hi))
    # Few successes keep the answer well away from 1 (many successes make it N/(N+1)-like).
    successes = rng.randint(2, min(population - 3, 6))
    draws = rng.randint(2, min(population - successes, 8))
    total = comb(population, draws)
    none_value = Fraction(comb(population - successes, draws), total)
    answer_value = 1 - none_value
    summed = sum(
        Fraction(comb(successes, x) * comb(population - successes, draws - x), total)
        for x in range(1, min(draws, successes) + 1)
    )
    check(summed == answer_value, "hypergeometric at-least-one complement mismatch")
    check(0 < none_value < 1 and 0 < answer_value < 1, "invalid at-least-one probability")

    one_value = Fraction(comb(successes, 1) * comb(population - successes, draws - 1), total)
    two_up = answer_value - one_value
    answer, answer_text = _fraction_payload(answer_value)
    with_replacement = 1 - (Fraction(population - successes, population)) ** draws
    alternatives = [
        _fraction_payload(none_value)[1],
        _fraction_payload(one_value)[1],
        _fraction_payload(1 - one_value)[1],
        _fraction_payload(with_replacement)[1],
        _fraction_payload(two_up)[1],
    ]
    return dict(
        text=(
            f"A population has {population} objects, {successes} of them successes. "
            f"If {draws} objects are drawn without replacement, find the probability "
            f"that at least one success is drawn, P(X>=1)."
        ),
        latex=rf"P(X\ge 1)=1-\frac{{\binom{{{population - successes}}}{{{draws}}}}}{{\binom{{{population}}}{{{draws}}}}}",
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"P(X>=1) = 1 - P(X=0) = 1 - C({population - successes},{draws})/C({population},{draws}).",
            f"C({population - successes},{draws}) = {comb(population - successes, draws)} and C({population},{draws}) = {total}; "
            f"therefore P(X>=1) = {answer_text}.",
        ],
        distractors=pick_distractors(answer_text, alternatives),
    )


@generator(
    "hypergeometric_probability",
    level=14,
    presets={
        "easy": dict(population_hi=12),
        "medium": dict(population_hi=24),
        "hard": dict(population_hi=45),
    },
)
def hypergeometric_probability(rng, population_hi=24, kind=None):
    """Find an exact without-replacement success probability.

    ``kind="at_least"`` asks P(X>=1) and needs the complement rule.
    """
    if kind == "at_least":
        return _hypergeometric_at_least(rng, population_hi)
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


def _bernoulli_mle_pooled(rng, n_hi):
    """Pooled MLE of p from two independent batches: (k1+k2)/(n1+n2)."""
    while True:
        n1 = rng.randint(4, n_hi)
        n2 = rng.randint(4, n_hi)
        k1 = rng.randint(1, n1 - 1)
        k2 = rng.randint(1, n2 - 1)
        total_n, total_k = n1 + n2, k1 + k2
        estimate = Fraction(total_k, total_n)
        candidates = [
            Fraction(k1, n1) / 2 + Fraction(k2, n2) / 2,
            Fraction(total_k, n1),
            Fraction(total_k, n2),
            Fraction(total_n - total_k, total_n),
            Fraction(total_k, total_n + 1),
        ]
        if estimate in (Fraction(k1, n1), Fraction(k2, n2)):
            continue
        # Every distractor must differ from the answer and from the others by value.
        if len(set(candidates + [estimate])) != len(candidates) + 1:
            continue
        break
    # Second route: the pooled log-likelihood score vanishes at the estimate.
    score = total_k * (1 - estimate) - (total_n - total_k) * estimate
    check(score == 0 and 0 < estimate < 1, "pooled Bernoulli MLE score does not vanish")
    check(
        estimate == Fraction(k1 + k2, n1 + n2) and estimate != candidates[0],
        "pooled Bernoulli MLE mismatch",
    )
    answer, answer_text = _fraction_payload(estimate)
    alternatives = [_fraction_payload(c)[1] for c in candidates]
    return dict(
        text=(
            f"Batch 1 has {k1} {'success' if k1 == 1 else 'successes'} in {n1} independent Bernoulli trials "
            f"and batch 2 has {k2} {'success' if k2 == 1 else 'successes'} in {n2} independent trials, "
            "both with the same success probability p. "
            "Find the MLE of p from the combined data."
        ),
        latex=rf"\hat p=\arg\max_{{0<p<1}}p^{{{total_k}}}(1-p)^{{{total_n - total_k}}}",
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"With a common p the likelihood is p^({k1}+{k2})(1−p)^({n1 - k1}+{n2 - k2}), so pool the counts.",
            f"p-hat = (k1+k2)/(n1+n2) = {total_k}/{total_n} = {answer_text}.",
        ],
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
def bernoulli_mle(rng, n_hi=25, kind=None):
    """Find the Bernoulli maximum-likelihood estimate p-hat = k/n.

    ``kind="pooled"`` pools two independent batches that share one p.
    """
    if kind == "pooled":
        return _bernoulli_mle_pooled(rng, n_hi)
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
        text=f"In {n} independent Bernoulli trials there {'is' if k == 1 else 'are'} {k} {'success' if k == 1 else 'successes'}. Find the MLE of p.",
        latex=rf"\hat p=\arg\max_{{0<p<1}}p^{{{k}}}(1-p)^{{{n - k}}}",
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"The log-likelihood score is k/p − (n−k)/(1−p).",
            f"Setting it to zero gives p-hat = k/n = {equals_chain(f'{k}/{n}', answer_text)}.",
        ],
        distractors=pick_distractors(answer_text, alternatives),
    )


_LSQ_CASES: list | None = None


def _lsq_cases():
    """Every (x-set, centred residual vector) with sum(e)=sum(xe)=0, e in -2..2, not all zero."""
    global _LSQ_CASES
    if _LSQ_CASES is None:
        cases = []
        for n in (4, 5):
            for xs in combinations(range(9), n):
                residual_sets = [
                    es
                    for es in product(range(-2, 3), repeat=n)
                    if any(es) and sum(es) == 0 and sum(x * e for x, e in zip(xs, es)) == 0
                ]
                if residual_sets:
                    cases.append((xs, residual_sets))
        _LSQ_CASES = cases
    return _LSQ_CASES


def _fit_from_sums(n, s_x, s_y, s_xy, s_xx):
    """Normal equations m=(n Sxy - Sx Sy)/(n Sxx - Sx^2), b=(Sy - m Sx)/n."""
    m = Fraction(n * s_xy - s_x * s_y, n * s_xx - s_x**2)
    return m, (s_y - m * s_x) / n


def _least_squares_summary(rng, slope, intercept):
    """Summary-statistics variant: integer fit, points hidden; private helper."""
    n_points = rng.choice((4, 5))
    xs, residual_sets = rng.choice([case for case in _lsq_cases() if len(case[0]) == n_points])
    residuals = rng.choice(residual_sets)
    x_values = list(xs)
    y_values = [slope * x + intercept + e for x, e in zip(x_values, residuals)]
    n = len(x_values)
    s_x, s_y = sum(x_values), sum(y_values)
    s_xy = sum(x * y for x, y in zip(x_values, y_values))
    s_xx = sum(x * x for x in x_values)
    from_sums = _fit_from_sums(n, s_x, s_y, s_xy, s_xx)
    x_bar, y_bar = Fraction(s_x, n), Fraction(s_y, n)
    centred_xx = sum((x - x_bar) ** 2 for x in x_values)
    fitted_slope = sum((x - x_bar) * (y - y_bar) for x, y in zip(x_values, y_values)) / centred_xx
    from_points = (fitted_slope, y_bar - fitted_slope * x_bar)
    check(from_points == from_sums == (slope, intercept), "least-squares summary fit mismatch")
    check(
        any(residuals) and sum(residuals) == 0 and sum(x * e for x, e in zip(x_values, residuals)) == 0,
        "constructed residuals do not preserve fit",
    )

    answer = {"m": slope, "b": intercept}
    answer_text = f"m = {slope}, b = {intercept}"
    candidates = [
        (intercept, slope),
        (-slope, intercept),
        (slope, n * intercept),  # forgot to divide by n in b = (Sy - m Sx)/n
        (slope + 1, intercept),
        (slope, intercept + 1),
        (slope - 1, intercept),
        (slope, intercept - 1),
        (slope, -intercept),
    ]
    alternatives = [f"m = {m}, b = {b}" for m, b in candidates if abs(m - slope) >= 1 or abs(b - intercept) >= 1]
    distractors = pick_distractors(answer_text, alternatives, tolerance=GRADER_TOLERANCE)
    check(len(distractors) == 3, "least-squares summary needs three distractors")
    return dict(
        text=(
            f"A data set of {n} points has sum x = {s_x}, sum y = {s_y}, sum xy = {s_xy}, "
            f"sum x^2 = {s_xx}. Find the least-squares line y = mx + b."
        ),
        latex=(
            rf"n={n},\ \sum x={s_x},\ \sum y={s_y},\ \sum xy={s_xy},\ \sum x^2={s_xx};"
            rf"\quad \hat y=\hat m x+\hat b"
        ),
        answer=answer,
        answer_text=answer_text,
        answer_latex=rf"m={slope},\;b={intercept}",
        working=[
            f"m-hat = (n·Sxy − Sx·Sy)/(n·Sxx − Sx²) = ({n}·{s_xy} − {s_x}·{s_y})/({n}·{s_xx} − {s_x}²) = {slope}.",
            f"b-hat = (Sy − m-hat·Sx)/n = ({s_y} − {paren_negative(slope)}·{s_x})/{n} = {intercept}.",
        ],
        distractors=distractors,
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
def least_squares_line(rng, coefficient_hi=8, intercept_hi=12, kind=None):
    """Recover the least-squares slope and intercept from a small data set.

    ``kind="summary"`` gives only n, sum x, sum y, sum xy, sum x^2 and asks
    for the fit via the normal equations (no points to read)."""
    slope = rng.randint(-coefficient_hi, coefficient_hi)
    intercept = rng.randint(-intercept_hi, intercept_hi)
    while slope == intercept:
        intercept = rng.randint(-intercept_hi, intercept_hi)
    if kind == "summary":
        return _least_squares_summary(rng, slope, intercept)
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
    if target_hi < 1:
        raise ValueError("gradient_descent_quadratic needs target_hi >= 1: the start must differ from the target")
    target = rng.randint(-target_hi, target_hi)
    theta = draw_until(
        rng,
        lambda r: r.randint(-target_hi, target_hi),
        lambda start: start != target,
        what="a start different from the target",
    )
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
