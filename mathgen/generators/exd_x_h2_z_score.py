"""Exam-tier z-score questions: the mean and standard deviation are unknown, or two scales are compared.

The practice tier standardises one value from a stated mean and standard deviation. The exam tier turns the
relationship round and composes it:

* ``find_mean``: two scores with their z-scores are given, and the learner solves the pair of equations
  ``x = mean + z * sd`` for the mean.
* ``find_sd``: the same information, solved for the standard deviation.
* ``equivalent``: a score on one test is moved to a second test with a different mean and spread, by keeping
  the z-score fixed (standardise on the first scale, then un-standardise on the second).

Every answer is a whole number by construction (an even standard deviation with z-scores in halves), re-derived
by solving the linear system in sympy before the question is returned.
"""

from __future__ import annotations

import random
from fractions import Fraction

import sympy

from ..core import GRADER_TOLERANCE, check, generator, phrase, pick_distractors

_SD_CHOICES = (4, 6, 8, 10, 12)
_Z_CHOICES = tuple(Fraction(k, 2) for k in range(-5, 6) if k != 0)  # -2.5 .. 2.5 in halves, never 0


def _z_text(z: Fraction) -> str:
    return f"{float(z):g}"


def _z_paren(z: Fraction) -> str:
    """A z-score for use inside an expression: a negative value gets its own brackets."""
    return f"({_z_text(z)})" if z < 0 else _z_text(z)


def _finish(text: str, answer: int, candidates: list, working: list[str]) -> dict:
    """Whole-number answer; only whole-number wrong values are offered (they share the answer's shape)."""
    wrong = []
    for c in candidates:
        c = Fraction(c)
        if c.denominator == 1 and c > 0:
            wrong.append(str(int(c)))
    distractors = pick_distractors(str(answer), wrong, tolerance=GRADER_TOLERANCE)
    return dict(
        text=text,
        latex=text,
        answer=answer,
        answer_text=str(answer),
        answer_latex=str(answer),
        working=working,
        distractors=distractors,
    )


def _two_scores(rng: random.Random) -> tuple[int, int, Fraction, Fraction, int, int]:
    """(mean, sd, z1, z2, x1, x2) with x_i = mean + z_i * sd, z1 != z2."""
    sd = rng.choice(_SD_CHOICES)
    mean = rng.randint(40, 90)
    z1, z2 = rng.sample(_Z_CHOICES, 2)
    x1, x2 = mean + z1 * sd, mean + z2 * sd
    check(x1.denominator == 1 and x2.denominator == 1, "scores must be whole numbers")
    return mean, sd, z1, z2, int(x1), int(x2)


def _solve_mean_sd(z1: Fraction, z2: Fraction, x1: int, x2: int) -> tuple[int, int]:
    """Independent route: solve x = mean + z * sd as a linear system."""
    mu, sigma = sympy.symbols("mu sigma")
    solution = sympy.solve(
        [
            mu + sympy.Rational(z1.numerator, z1.denominator) * sigma - x1,
            mu + sympy.Rational(z2.numerator, z2.denominator) * sigma - x2,
        ],
        [mu, sigma],
        dict=True,
    )
    check(len(solution) == 1, "the pair of equations must have exactly one solution")
    return int(solution[0][mu]), int(solution[0][sigma])


def _find_mean(rng: random.Random) -> dict:
    while True:
        mean, sd, z1, z2, x1, x2 = _two_scores(rng)
        answer = mean
        gap = x2 - x1
        cands = [
            Fraction(x1 + x2, 2),  # the midpoint of the two scores
            Fraction(x1) - z1 * gap,  # the raw score gap used as the standard deviation
            Fraction(x1) + z1 * sd,  # the z-score term added instead of subtracted
            Fraction(x1 - sd),  # assumes the first score is one standard deviation above the mean
        ]
        if len(_finish("", answer, cands, [])["distractors"]) >= 1:
            break
    check(_solve_mean_sd(z1, z2, x1, x2) == (mean, sd), "mean disagrees with the linear system")
    text = phrase(
        rng,
        (
            "In a normal distribution, a score of {x1} has a z-score of {z1} and a score of {x2} has a z-score of "
            "{z2}. Find the mean of the distribution.",
            "A test score of {x1} is {z1} standard deviations from the mean (z = {z1}), and a score of {x2} has "
            "z = {z2}. What is the mean score?",
        ),
        x1=x1,
        x2=x2,
        z1=_z_text(z1),
        z2=_z_text(z2),
    )
    working = [
        f"x = mean + z × sd. Subtracting the two equations: {x2} - {x1} = ({_z_paren(z2)} - {_z_paren(z1)}) × sd, "
        f"so sd = {sd}. Then mean = {x1} - {_z_paren(z1)} × {sd} = {mean}."
    ]
    return _finish(text, answer, cands, working)


def _find_sd(rng: random.Random) -> dict:
    while True:
        mean, sd, z1, z2, x1, x2 = _two_scores(rng)
        answer = sd
        gap = x2 - x1
        cands = [
            Fraction(abs(gap)),  # the score gap, not divided by the z-score gap
            Fraction(x1) / z1,  # the score divided by its z-score, with no centring
            Fraction(gap) / (z1 + z2) if z1 + z2 != 0 else Fraction(0),  # z-scores added instead of subtracted
            Fraction(abs(gap)) / 2,
        ]
        if len(_finish("", answer, cands, [])["distractors"]) >= 1:
            break
    check(_solve_mean_sd(z1, z2, x1, x2) == (mean, sd), "standard deviation disagrees with the linear system")
    text = phrase(
        rng,
        (
            "In a normal distribution, a score of {x1} has a z-score of {z1} and a score of {x2} has a z-score of "
            "{z2}. Find the standard deviation of the distribution.",
            "On a test, a mark of {x1} has z = {z1} and a mark of {x2} has z = {z2}. What is the standard deviation "
            "of the marks?",
        ),
        x1=x1,
        x2=x2,
        z1=_z_text(z1),
        z2=_z_text(z2),
    )
    working = [
        f"The two scores differ by {abs(x2 - x1)}, which is {_z_text(abs(z2 - z1))} standard deviations "
        f"(the gap between the z-scores {_z_text(z1)} and {_z_text(z2)}), so sd = {sd}."
    ]
    return _finish(text, answer, cands, working)


def _equivalent(rng: random.Random) -> dict:
    while True:
        sd_a, sd_b = rng.sample(_SD_CHOICES, 2)
        mean_a, mean_b = rng.randint(40, 90), rng.randint(40, 90)
        z = rng.choice(_Z_CHOICES)
        score_a = mean_a + z * sd_a
        score_b = mean_b + z * sd_b
        check(score_a.denominator == 1 and score_b.denominator == 1, "scores must be whole numbers")
        score_a, score_b = int(score_a), int(score_b)
        cands = [
            Fraction(score_a),  # the same raw score
            Fraction(mean_b + (score_a - mean_a)),  # the raw distance from the mean copied across
            Fraction(mean_b) + z * sd_a,  # the first test's spread used on the second test
            Fraction(mean_b) - z * sd_b,  # the sign of z lost
        ]
        if len(_finish("", score_b, cands, [])["distractors"]) >= 1:
            break
    # Independent route: equate (score_b - mean_b) / sd_b with (score_a - mean_a) / sd_a and solve for score_b.
    x = sympy.symbols("x")
    solved = sympy.solve(sympy.Rational(score_a - mean_a, sd_a) - (x - mean_b) / sympy.Integer(sd_b), x)
    check(solved == [score_b], "equivalent score disagrees with the equation")
    text = phrase(
        rng,
        (
            "Test A has mean {mean_a} and standard deviation {sd_a}. Test B has mean {mean_b} and standard deviation "
            "{sd_b}. A student scores {score_a} on Test A. What score on Test B would be equally far from the mean, "
            "measured in standard deviations?",
            "Two exams are marked on different scales: exam 1 has mean {mean_a} and standard deviation {sd_a}, and "
            "exam 2 has mean {mean_b} and standard deviation {sd_b}. A mark of {score_a} on exam 1 has the same "
            "z-score as which mark on exam 2?",
        ),
        mean_a=mean_a,
        sd_a=sd_a,
        mean_b=mean_b,
        sd_b=sd_b,
        score_a=score_a,
    )
    working = [
        f"z = ({score_a} - {mean_a}) / {sd_a} = {_z_text(z)}. On the second scale: {mean_b} + {_z_paren(z)} × {sd_b} "
        f"= {score_b}."
    ]
    return _finish(text, score_b, cands, working)


@generator(
    "exd_x_h2_z_score",
    level=12,
    presets={"exam": dict(tier="exam")},
)
def exd_x_h2_z_score(rng: random.Random, tier: str = "exam") -> dict:
    """Exam-tier z-score: recover the mean or spread from two scores, or move a score between two scales."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return rng.choice((_find_mean, _find_sd, _equivalent))(rng)
