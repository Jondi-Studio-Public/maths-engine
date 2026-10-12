"""Exam-tier source for `gradient-descent-quadratic`: two gradient-descent steps on a loss given as a polynomial.

The practice tiers give the loss in the shifted form 1/2 (theta - t)^2, whose gradient can be read off. Here the loss
is a general quadratic ``a theta^2 + b theta + c`` that the learner must differentiate, and the question is a chain
rather than one step:

(a) the gradient at the starting value                      fraction, 1 mark
(b) the value after the first step                          fraction, 2 marks, ``theta0 - eta * part_a``
(c) the value after a second step from the new value        fraction, 2 marks, the gradient recomputed at ``part_b``

Parts (b) and (c) carry ``follow_through``, so a learner whose gradient is wrong still earns the step that uses it
correctly, and a learner who stumbles on the first step still earns the second step taken from their own value.

Built backwards from integer ``a``, ``b``, ``c`` and the start, with a learning rate small enough that
``0 < 2 a eta < 1``: each step moves toward the minimiser without overshooting, as a sensible rate does.
"""

from __future__ import annotations

import random
from fractions import Fraction

from mathgen.answers import Distractor, exact_value
from mathgen.core import (
    attempts,
    check,
    generator,
    join_terms,
    latex_answer,
    paren_negative,
    phrase,
    pick_distractors,
    rational_text,
)

_RATES = (Fraction(1, 4), Fraction(1, 5), Fraction(1, 6), Fraction(1, 8), Fraction(1, 10))


def _fraction_part(prompt, value, marks, working, candidates, follow_through=None):
    """A `fraction` part whose answer is `value`; `candidates` are (misconception id or None, value) pairs."""
    text = rational_text(value)
    options = [Distractor(rational_text(v), tag, exact_value(v)) for tag, v in candidates if v != value]
    part = dict(
        prompt=prompt,
        answer_kind="fraction",
        answer=exact_value(value),
        answer_text=text,
        answer_latex=latex_answer(text),
        marks=marks,
        working=working,
        distractors=pick_distractors(text, options, 3),
    )
    if follow_through:
        part["follow_through"] = follow_through
    return part


@generator("exd_x_u5_gradient_descent", level=17, presets={"exam": dict(tier="exam")})
def exd_x_u5_gradient_descent(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: gradient, then two gradient-descent steps, on a quadratic loss written in expanded form."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(1000, "a question with three distinct distractors per part"):
        question = _draw(rng)
        if question is not None:
            return question


def _draw(rng: random.Random) -> dict | None:
    """One draw of the chain, or None when a part has fewer than three distinct distractors."""
    for _ in attempts(1000, "a loss and start with nonzero gradients"):
        a = rng.choice((1, 2, 3))
        b = rng.choice([n for n in range(-9, 10) if n])
        c = rng.randint(1, 9)
        theta0 = rng.choice([n for n in range(-8, 9) if n])
        eta = rng.choice(_RATES)
        if not 0 < 2 * a * eta < 1:
            continue
        g0 = 2 * a * theta0 + b
        theta1 = theta0 - eta * g0
        g1 = 2 * a * theta1 + b
        if g0 == 0 or g1 == 0:
            continue
        break

    def loss(t):
        return a * t * t + b * t + c

    # Independent route: the new value from the minimiser, theta* + (1 - 2 a eta) (theta - theta*).
    minimiser = Fraction(-b, 2 * a)
    shrink = 1 - 2 * a * eta
    theta2 = theta1 - eta * g1
    check(theta1 == minimiser + shrink * (theta0 - minimiser), "first step disagrees with the contraction formula")
    check(theta2 == minimiser + shrink**2 * (theta0 - minimiser), "second step disagrees with the contraction formula")
    check(loss(theta2) < loss(theta1) < loss(theta0), "a step did not reduce the loss")

    loss_text = join_terms([(a, "θ²"), (b, "θ"), (c, "")])
    eta_text = rational_text(eta)
    stem = phrase(
        rng,
        (
            "A model has loss L(θ) = {loss}. Gradient descent with learning rate η = {eta} starts at θ = {t0}.",
            "The loss of a model is L(θ) = {loss}. Gradient descent is run from θ = {t0} with learning rate η = {eta}.",
            "Gradient descent with learning rate η = {eta} is used to reduce the loss L(θ) = {loss}, starting from "
            "θ = {t0}.",
        ),
        loss=loss_text,
        eta=eta_text,
        t0=theta0,
    )

    part_a = _fraction_part(
        f"Find the gradient dL/dθ at θ = {theta0}.",
        Fraction(g0),
        1,
        [f"dL/dθ = {join_terms([(2 * a, 'θ'), (b, '')])}; at θ = {theta0} this is {g0}."],
        [
            ("dropped-factor-two-in-derivative", Fraction(a * theta0 + b)),
            ("evaluates-function-not-derivative", Fraction(loss(theta0))),
            (None, Fraction(2 * a * theta0)),
            (None, Fraction(2 * a * theta0 - b)),
        ],
    )
    part_b = _fraction_part(
        "Take one gradient-descent step. Find the new value of θ.",
        theta1,
        2,
        [f"θ₁ = {theta0} − {eta_text} × {paren_negative(g0)} = {rational_text(theta1)}."],
        [
            ("gradient-step-uphill", theta0 + eta * g0),
            ("reports-gradient-not-update", Fraction(g0)),
            ("forgot-learning-rate", theta0 - g0),
            ("dropped-factor-two-in-derivative", theta0 - eta * (a * theta0 + b)),
        ],
        follow_through=f"{theta0} - ({eta.numerator}/{eta.denominator})*part_a",
    )
    part_c = _fraction_part(
        "Take a second step from this new value, recomputing the gradient there. Find the next value of θ.",
        theta2,
        2,
        [
            f"At θ₁ = {rational_text(theta1)} the gradient is {rational_text(g1)}.",
            f"θ₂ = {rational_text(theta1)} − {eta_text} × {paren_negative(Fraction(g1))} = {rational_text(theta2)}.",
        ],
        [
            ("stale-gradient", theta1 - eta * g0),
            ("gradient-step-uphill", theta1 + eta * g1),
            ("reports-gradient-not-update", g1),
            ("forgot-learning-rate", theta1 - g1),
        ],
        follow_through=f"part_b - ({eta.numerator}/{eta.denominator})*(({2 * a})*part_b + ({b}))",
    )
    if any(len(part["distractors"]) < 3 for part in (part_a, part_b, part_c)):
        return None
    return dict(text=stem, parts=[part_a, part_b, part_c])
