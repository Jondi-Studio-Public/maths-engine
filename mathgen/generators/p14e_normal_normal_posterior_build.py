"""A conjugate normal-normal Bayesian posterior update (``ucv1-bayes-normal-posterior``,
``statistical-learning``, topic ``p14e_normal_normal_posterior``, ``answer: pair``).

**Derivation (checked by the independent oracle and by a second, calculus
route inside the generator, never just trusted).** Given a prior
``N(mu0, tau0^2)`` on an unknown mean, a known data variance ``sigma^2``, a
sample size ``n`` and a sample mean ``x-bar``, the conjugate normal-normal
posterior is

    precision1 = 1/tau0^2 + n/sigma^2
    tau1^2     = 1 / precision1
    mu1        = tau1^2 * (mu0/tau0^2 + n*x-bar/sigma^2)

Every quantity is an exact ``Fraction`` -- no square roots, no floats.

**Independent in-generator check -- "complete the square via calculus", a
genuinely different derivation route from the direct weighted-average
formula above.** The (unnormalised) negative log-posterior is a quadratic in
a ``sympy.Symbol``:

    neg_log_posterior = (x - mu0)**2 / (2*tau0^2) + n*(x - x-bar)**2 / (2*sigma^2)

``mu1`` is the critical point (``solve(diff(neg_log_posterior, x), x)``);
``tau1^2`` is the reciprocal of the second derivative (the posterior
precision). ``check_equivalent`` both against the values the weighted-average
formula constructed.

**Boundary against the neighbour** ``ucv1-bayes-finite-counts``
(``mathgen/generators/university_stat_learning.py``'s sibling module family):
that skill is a DISCRETE Bayes update over a finite count table -- two
groups, raw item counts, a per-group rate for an evidence property -- and
answers a single ``fraction`` probability that an item came from one group.
This skill is the CONTINUOUS conjugate-normal update: a prior *distribution*
over a continuous unknown mean, updated by a known-variance sample, answering
a *pair* (posterior mean, posterior variance) -- never a probability, and no
finite table anywhere in the prompt. A learner could never swap one
question into the other's skill.
"""

from __future__ import annotations

from fractions import Fraction

import sympy

from ..core import (
    GRADER_TOLERANCE,
    VerificationError,
    check,
    check_equivalent,
    check_pairwise_distinct,
    generator,
    latex_answer,
    phrase,
    pick_distractors,
    rational_text,
)

_MAX_DRAW_ATTEMPTS = 500

_EASY_TEMPLATES = (
    "A prior for an unknown mean is N({mu0}, {tau0_sq}) (that is, prior mean "
    "mu0 = {mu0} and prior variance tau0^2 = {tau0_sq}). Data with known "
    "variance sigma^2 = {sigma_sq} give a sample of size n = {n} with sample "
    "mean x-bar = {xbar}. Find the posterior mean mu1 and posterior variance "
    "tau1^2 of the conjugate normal-normal update.",
    "The prior belief about an unknown mean is N({mu0}, {tau0_sq}): prior "
    "mean mu0 = {mu0}, prior variance tau0^2 = {tau0_sq}. A sample of size "
    "n = {n} with known data variance sigma^2 = {sigma_sq} has sample mean "
    "x-bar = {xbar}. Compute the posterior mean mu1 and posterior variance "
    "tau1^2.",
    "Given a prior N({mu0}, {tau0_sq}) on an unknown mean (mu0 = {mu0}, "
    "tau0^2 = {tau0_sq}), and a sample of size n = {n} with known variance "
    "sigma^2 = {sigma_sq} and sample mean x-bar = {xbar}, find the "
    "conjugate normal-normal posterior mean mu1 and variance tau1^2.",
)

_MEDIUM_TEMPLATES = (
    "A prior for an unknown mean is N({mu0}, {tau0_sq}) (prior mean mu0 = "
    "{mu0}, prior variance tau0^2 = {tau0_sq}). Data with known variance "
    "sigma^2 = {sigma_sq} give a sample of size n = {n} with sample mean "
    "x-bar = {xbar}. Find the posterior mean mu1 and posterior variance "
    "tau1^2 of the conjugate normal-normal update.",
    "The prior belief about an unknown mean is N({mu0}, {tau0_sq}): prior "
    "mean mu0 = {mu0}, prior variance tau0^2 = {tau0_sq}. A sample of size "
    "n = {n} with known data variance sigma^2 = {sigma_sq} has sample mean "
    "x-bar = {xbar}. Compute the posterior mean mu1 and posterior variance "
    "tau1^2.",
    "Given a prior N({mu0}, {tau0_sq}) on an unknown mean (mu0 = {mu0}, "
    "tau0^2 = {tau0_sq}), and a sample of size n = {n} with known variance "
    "sigma^2 = {sigma_sq} and sample mean x-bar = {xbar}, find the "
    "conjugate normal-normal posterior mean mu1 and variance tau1^2.",
)

_HARD_TEMPLATES = (
    "A prior for an unknown mean has mean mu0 = {mu0} and is stated as a "
    "precision of lambda0 = {lambda0} (rather than a variance). Data with "
    "known variance sigma^2 = {sigma_sq} give a sample of size n = {n} with "
    "sample mean x-bar = {xbar}. Find the posterior mean mu1 and posterior "
    "variance tau1^2 of the conjugate normal-normal update.",
    "The prior on an unknown mean has mean mu0 = {mu0} and a stated "
    "precision lambda0 = {lambda0} (the reciprocal of the prior variance). "
    "A sample of size n = {n} with known data variance sigma^2 = {sigma_sq} "
    "has sample mean x-bar = {xbar}. Compute the posterior mean mu1 and "
    "posterior variance tau1^2.",
    "An unknown mean has prior mean mu0 = {mu0} and prior PRECISION "
    "lambda0 = {lambda0} (not a variance). Given a sample of size n = {n} "
    "with known variance sigma^2 = {sigma_sq} and sample mean x-bar = "
    "{xbar}, find the conjugate normal-normal posterior mean mu1 and "
    "variance tau1^2.",
)

_EXAM_TEMPLATES = (
    "A prior for an unknown mean is N({mu0}, {tau0_sq}) (mu0 = {mu0}, "
    "tau0^2 = {tau0_sq}). It is updated TWICE with independent samples of "
    "known variance sigma^2 = {sigma_sq}: first a sample of size n1 = {n1} "
    "with sample mean x-bar_1 = {xbar1}, giving an intermediate posterior; "
    "then a second sample of size n2 = {n2} with sample mean x-bar_2 = "
    "{xbar2}, updating that intermediate posterior again. Find the FINAL "
    "posterior mean mu1 and variance tau1^2 after both updates.",
    "An unknown mean starts with prior N({mu0}, {tau0_sq}) (mu0 = {mu0}, "
    "tau0^2 = {tau0_sq}). Two independent samples arrive in sequence, both "
    "with known data variance sigma^2 = {sigma_sq}: sample 1 has size n1 = "
    "{n1} and mean x-bar_1 = {xbar1}; sample 2 has size n2 = {n2} and mean "
    "x-bar_2 = {xbar2}. Update the prior with sample 1 to get an "
    "intermediate posterior, then update THAT posterior with sample 2. "
    "Find the final posterior mean mu1 and variance tau1^2.",
    "The prior belief about an unknown mean is N({mu0}, {tau0_sq}). Two "
    "rounds of data collection follow, each with known variance sigma^2 = "
    "{sigma_sq}: round one gives n1 = {n1} observations with mean x-bar_1 = "
    "{xbar1}; round two gives n2 = {n2} observations with mean x-bar_2 = "
    "{xbar2}. Apply the conjugate normal-normal update twice, in sequence, "
    "and report the final posterior mean mu1 and variance tau1^2.",
)


def _pair_payload(mu, tau_sq) -> tuple[dict, str]:
    mu, tau_sq = Fraction(mu), Fraction(tau_sq)
    mu_text, tau_text = rational_text(mu), rational_text(tau_sq)
    return {"mean": mu, "variance": tau_sq}, f"mean = {mu_text}, variance = {tau_text}"


def _format_pair(mu, tau_sq) -> str:
    return f"mean = {rational_text(Fraction(mu))}, variance = {rational_text(Fraction(tau_sq))}"


def _pair_latex(mu, tau_sq) -> str:
    """Real LaTeX for the (mean, variance) pair -- `answer_text`'s own
    `"mean = X, variance = Y"` shape fed through `latex_answer` unmodified
    mangles. Both words are wrapped in `\\text{}`
    so they render upright instead, and each numeric value still goes
    through `latex_answer` for a real `\\frac{}{}`."""
    mu_l = latex_answer(str(Fraction(mu)))
    tau_l = latex_answer(str(Fraction(tau_sq)))
    return rf"\text{{mean}} = {mu_l},\; \text{{variance}} = {tau_l}"


def _posterior(mu0, tau0_sq, sigma_sq, n, xbar):
    """The direct weighted-average closed form."""
    mu0, tau0_sq, sigma_sq, xbar = Fraction(mu0), Fraction(tau0_sq), Fraction(sigma_sq), Fraction(xbar)
    precision1 = Fraction(1, 1) / tau0_sq + Fraction(n) / sigma_sq
    tau1_sq = Fraction(1, 1) / precision1
    mu1 = tau1_sq * (mu0 / tau0_sq + Fraction(n) * xbar / sigma_sq)
    return mu1, tau1_sq, precision1


def _sympy_check(mu0, tau0_sq, sigma_sq, n, xbar, mu1, tau1_sq):
    """Independent route: complete the square via calculus on the negative
    log-posterior, a genuinely different derivation from the weighted-average
    construction in `_posterior`."""
    mu0, tau0_sq, sigma_sq, xbar = Fraction(mu0), Fraction(tau0_sq), Fraction(sigma_sq), Fraction(xbar)
    x = sympy.Symbol("x")
    mu0_s = sympy.Rational(mu0.numerator, mu0.denominator)
    tau0_sq_s = sympy.Rational(tau0_sq.numerator, tau0_sq.denominator)
    sigma_sq_s = sympy.Rational(sigma_sq.numerator, sigma_sq.denominator)
    xbar_s = sympy.Rational(xbar.numerator, xbar.denominator)
    neg_log_posterior = (x - mu0_s) ** 2 / (2 * tau0_sq_s) + n * (x - xbar_s) ** 2 / (2 * sigma_sq_s)
    critical_points = sympy.solve(sympy.diff(neg_log_posterior, x), x)
    check(len(critical_points) == 1, "posterior critical point is not unique")
    mu1_check = critical_points[0]
    tau1_sq_check = 1 / sympy.diff(neg_log_posterior, x, 2)
    check_equivalent(mu1_check, mu1, "calculus route: posterior mean mismatch")
    check_equivalent(tau1_sq_check, tau1_sq, "calculus route: posterior variance mismatch")


def _clears_pair_tolerance(correct_pair, candidate_pair) -> bool:
    """Whether at least one component differs from the correct value by more
    than the real pair grader's own tolerance (`mathgen/grading/typed.py::grade_pair`,
    `_PAIR_TOLERANCE = 0.051`, matching `core.GRADER_TOLERANCE`) -- so the
    real grader is guaranteed to reject the candidate."""
    return any(abs(Fraction(c) - Fraction(a)) > GRADER_TOLERANCE for c, a in zip(candidate_pair, correct_pair))


def _misconceptions(mu0, tau0_sq, sigma_sq, n, xbar, mu1, tau1_sq, precision1):
    mu0, tau0_sq, sigma_sq, xbar = Fraction(mu0), Fraction(tau0_sq), Fraction(sigma_sq), Fraction(xbar)

    # forgot-n: treats the sample as a single observation (n omitted from
    # both the precision AND the weighting term it feeds).
    precision_forgot_n = Fraction(1, 1) / tau0_sq + Fraction(1, 1) / sigma_sq
    tau_forgot_n = Fraction(1, 1) / precision_forgot_n
    mu_forgot_n = tau_forgot_n * (mu0 / tau0_sq + xbar / sigma_sq)

    # swapped-precision-weights: uses tau0^2/n where n/sigma^2 belongs (mixes
    # up which precision goes with which term).
    precision_swapped = Fraction(1, 1) / tau0_sq + tau0_sq / Fraction(n)
    tau_swapped = Fraction(1, 1) / precision_swapped
    mu_swapped = tau_swapped * (mu0 / tau0_sq + xbar * tau0_sq / Fraction(n))

    return {
        "doesnt-shrink": (mu1, tau0_sq),
        "forgot-n": (mu_forgot_n, tau_forgot_n),
        "swapped-precision-weights": (mu_swapped, tau_swapped),
        "reports-precision-not-variance": (mu1, precision1),
        "unweighted-average": ((mu0 + xbar) / 2, tau1_sq),
    }


def _finish(mu0, tau0_sq, sigma_sq, n, xbar, mu1, tau1_sq, precision1, text):
    misconceptions = _misconceptions(mu0, tau0_sq, sigma_sq, n, xbar, mu1, tau1_sq, precision1)
    answer, answer_text = _pair_payload(mu1, tau1_sq)
    values_to_check = [(mu1, tau1_sq), *misconceptions.values()]
    check_pairwise_distinct(
        *values_to_check,
        msg=f"posterior collision: mu0={mu0}, tau0_sq={tau0_sq}, sigma_sq={sigma_sq}, n={n}, xbar={xbar}",
    )
    for name, candidate in misconceptions.items():
        check(
            _clears_pair_tolerance((mu1, tau1_sq), candidate),
            f"{name} distractor within the real pair grader's tolerance",
        )
    distractor_strs = [_format_pair(*cand) for cand in misconceptions.values()]
    return answer, answer_text, distractor_strs


def _build_easy(rng) -> dict:
    for _attempt in range(_MAX_DRAW_ATTEMPTS):
        mu0 = rng.randint(-8, 8)
        tau0_sq = rng.randint(1, 6)
        sigma_sq = rng.randint(1, 8)
        n = rng.randint(2, 8)
        xbar = rng.randint(-8, 8)
        mu1, tau1_sq, precision1 = _posterior(mu0, tau0_sq, sigma_sq, n, xbar)
        _sympy_check(mu0, tau0_sq, sigma_sq, n, xbar, mu1, tau1_sq)
        try:
            answer, answer_text, distractor_strs = _finish(
                mu0,
                tau0_sq,
                sigma_sq,
                n,
                xbar,
                mu1,
                tau1_sq,
                precision1,
                None,
            )
        except VerificationError:
            continue
        text = phrase(rng, _EASY_TEMPLATES, mu0=mu0, tau0_sq=tau0_sq, sigma_sq=sigma_sq, n=n, xbar=xbar)
        return dict(
            text=text,
            latex=text,
            answer=answer,
            answer_text=answer_text,
            answer_latex=_pair_latex(mu1, tau1_sq),
            working=[
                f"precision1 = 1/tau0^2 + n/sigma^2 = 1/{tau0_sq} + {n}/{sigma_sq} = {rational_text(precision1)}.",
                f"tau1^2 = 1/precision1 = {rational_text(tau1_sq)}; "
                f"mu1 = tau1^2 × (mu0/tau0^2 + n × x-bar/sigma^2) = {answer_text}.",
            ],
            distractors=pick_distractors(answer_text, distractor_strs, n=5),
        )
    raise VerificationError("could not draw a collision-free easy-tier posterior question")


def _build_medium(rng) -> dict:
    for _attempt in range(_MAX_DRAW_ATTEMPTS):
        mu0 = rng.randint(-10, 10)
        tau0_sq = rng.randint(1, 8)
        sigma_sq = rng.randint(1, 10)
        n = rng.randint(2, 9)
        denom = rng.choice((2, 3, 4))
        numer = rng.randint(-10 * denom, 10 * denom)
        while numer % denom == 0:
            numer = rng.randint(-10 * denom, 10 * denom)
        xbar = Fraction(numer, denom)
        mu1, tau1_sq, precision1 = _posterior(mu0, tau0_sq, sigma_sq, n, xbar)
        _sympy_check(mu0, tau0_sq, sigma_sq, n, xbar, mu1, tau1_sq)
        try:
            answer, answer_text, distractor_strs = _finish(
                mu0,
                tau0_sq,
                sigma_sq,
                n,
                xbar,
                mu1,
                tau1_sq,
                precision1,
                None,
            )
        except VerificationError:
            continue
        xbar_text = rational_text(xbar)
        text = phrase(rng, _MEDIUM_TEMPLATES, mu0=mu0, tau0_sq=tau0_sq, sigma_sq=sigma_sq, n=n, xbar=xbar_text)
        return dict(
            text=text,
            latex=text,
            answer=answer,
            answer_text=answer_text,
            answer_latex=_pair_latex(mu1, tau1_sq),
            working=[
                f"precision1 = 1/tau0^2 + n/sigma^2 = 1/{tau0_sq} + {n}/{sigma_sq} = {rational_text(precision1)}.",
                f"tau1^2 = 1/precision1 = {rational_text(tau1_sq)}; "
                f"mu1 = tau1^2 × (mu0/tau0^2 + n × x-bar/sigma^2) = {answer_text}.",
            ],
            distractors=pick_distractors(answer_text, distractor_strs, n=5),
        )
    raise VerificationError("could not draw a collision-free medium-tier posterior question")


def _build_hard(rng) -> dict:
    for _attempt in range(_MAX_DRAW_ATTEMPTS):
        mu0 = rng.randint(-8, 8)
        lambda0 = rng.randint(1, 8)
        tau0_sq = Fraction(1, lambda0)
        sigma_sq = rng.randint(1, 8)
        n = rng.randint(2, 8)
        xbar = rng.randint(-8, 8)
        mu1, tau1_sq, precision1 = _posterior(mu0, tau0_sq, sigma_sq, n, xbar)
        _sympy_check(mu0, tau0_sq, sigma_sq, n, xbar, mu1, tau1_sq)
        try:
            answer, answer_text, distractor_strs = _finish(
                mu0,
                tau0_sq,
                sigma_sq,
                n,
                xbar,
                mu1,
                tau1_sq,
                precision1,
                None,
            )
        except VerificationError:
            continue
        text = phrase(rng, _HARD_TEMPLATES, mu0=mu0, lambda0=lambda0, sigma_sq=sigma_sq, n=n, xbar=xbar)
        return dict(
            text=text,
            latex=text,
            answer=answer,
            answer_text=answer_text,
            answer_latex=_pair_latex(mu1, tau1_sq),
            working=[
                f"tau0^2 = 1/lambda0 = 1/{lambda0} = {rational_text(tau0_sq)}.",
                f"precision1 = lambda0 + n/sigma^2 = {lambda0} + {n}/{sigma_sq} = {rational_text(precision1)}.",
                f"tau1^2 = 1/precision1 = {rational_text(tau1_sq)}; "
                f"mu1 = tau1^2 × (mu0/tau0^2 + n × x-bar/sigma^2) = {answer_text}.",
            ],
            distractors=pick_distractors(answer_text, distractor_strs, n=5),
        )
    raise VerificationError("could not draw a collision-free hard-tier posterior question")


def _build_exam(rng) -> dict:
    for _attempt in range(_MAX_DRAW_ATTEMPTS):
        mu0 = rng.randint(-8, 8)
        tau0_sq = rng.randint(1, 6)
        sigma_sq = rng.randint(1, 8)
        n1 = rng.randint(2, 6)
        xbar1 = rng.randint(-8, 8)
        n2 = rng.randint(2, 6)
        xbar2 = rng.randint(-8, 8)

        mu_a, tau_a_sq, _precision_a = _posterior(mu0, tau0_sq, sigma_sq, n1, xbar1)
        _sympy_check(mu0, tau0_sq, sigma_sq, n1, xbar1, mu_a, tau_a_sq)

        mu1, tau1_sq, precision1 = _posterior(mu_a, tau_a_sq, sigma_sq, n2, xbar2)
        _sympy_check(mu_a, tau_a_sq, sigma_sq, n2, xbar2, mu1, tau1_sq)

        try:
            answer, answer_text, distractor_strs = _finish(
                mu_a,
                tau_a_sq,
                sigma_sq,
                n2,
                xbar2,
                mu1,
                tau1_sq,
                precision1,
                None,
            )
        except VerificationError:
            continue
        text = phrase(
            rng,
            _EXAM_TEMPLATES,
            mu0=mu0,
            tau0_sq=tau0_sq,
            sigma_sq=sigma_sq,
            n1=n1,
            xbar1=xbar1,
            n2=n2,
            xbar2=xbar2,
        )
        return dict(
            text=text,
            latex=text,
            answer=answer,
            answer_text=answer_text,
            answer_latex=_pair_latex(mu1, tau1_sq),
            working=[
                f"Stage 1: precision_a = 1/tau0^2 + n1/sigma^2 = {rational_text(Fraction(1, 1) / Fraction(tau0_sq) + Fraction(n1) / Fraction(sigma_sq))}; "
                f"intermediate posterior mu_a = {rational_text(mu_a)}, tau_a^2 = {rational_text(tau_a_sq)}.",
                f"Stage 2: precision1 = 1/tau_a^2 + n2/sigma^2 = {rational_text(precision1)}; "
                f"tau1^2 = 1/precision1 = {rational_text(tau1_sq)}; mu1 = {answer_text}.",
            ],
            distractors=pick_distractors(answer_text, distractor_strs, n=5),
        )
    raise VerificationError("could not draw a collision-free exam-tier posterior question")


@generator(
    "p14e_normal_normal_posterior",
    level=16,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def normal_normal_posterior(rng, tier="easy"):
    """Compute the posterior mean and variance for a supplied normal-normal
    conjugate update (the skill's stated contract).

    easy: mu0, tau0^2, sigma^2, n, x-bar all small integers -- single
    substitution.
    medium: x-bar is a genuine small fraction -- the arithmetic itself
    requires fraction bookkeeping throughout.
    hard: the prior is stated as a PRECISION lambda0 = 1/tau0^2 instead of a
    variance -- a different quantity is given vs. derived.
    exam: two sequential updates -- the same single-update kernel applied
    twice, chaining the first posterior into the second prior.
    """
    if tier == "easy":
        return _build_easy(rng)
    if tier == "medium":
        return _build_medium(rng)
    if tier == "hard":
        return _build_hard(rng)
    if tier == "exam":
        return _build_exam(rng)
    raise ValueError(f"unknown tier {tier!r}")
