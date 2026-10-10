"""Exam-tier questions for the two-state Markov chain skills.

Two topics live here, one per skill:

``exd_x_s2_markov_step``
    The chain is run for TWO steps rather than one. Either the initial law is
    given and ``P(X2=1)`` is asked (two forward substitutions that must be
    chained), or ``P(X2=1)`` is given and the learner must undo two steps to
    recover ``P(X0=1)``. With ``lam = 1 - p - q`` the one-step law is
    ``x1 = p + lam*x0``, so::

        x2 = p*(1 + lam) + lam**2 * x0

    The key is built from a drawn ``x0`` (never by inverting a drawn ``x2``), so
    it is exact, and ``lam`` is redrawn until it is non-zero so the backward
    question has a unique answer.

``exd_x_s2_markov_stationary``
    A long-run *rate* is asked instead of the stationary probability itself, so
    the learner must find the stationary law and then compose it with a
    transition probability. Writing ``pi1 = p/(p+q)`` and ``pi0 = q/(p+q)``:

    * ``up_move``      the long-run fraction of steps that go 0 -> 1: ``pi0*p``
    * ``change_state`` the long-run fraction of steps that change state:
      ``pi0*p + pi1*q``
    * ``from_pi``      ``p`` and ``pi1`` are given (not ``q``); the learner first
      recovers ``q`` from the balance equation, then asks for the fraction of
      steps that change state.

    Every answer is at most 1 (``2pq/(p+q)`` is a harmonic mean, bounded by the
    arithmetic mean ``(p+q)/2``).

Answers are exact fractions, graded exactly by ``grade_fraction``.
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..core import check, generator, pick_distractors, rational_text


def _payload(value: Fraction) -> tuple[dict, str]:
    value = Fraction(value)
    return {"p": value.numerator, "q": value.denominator}, rational_text(value)


def _dist(answer: Fraction, candidates: list[Fraction]) -> list[str]:
    answer_text = rational_text(answer)
    return pick_distractors(answer_text, [rational_text(c) for c in candidates if 0 <= c <= 1])


# --------------------------------------------------------------------------- step


def _two_step(p: Fraction, q: Fraction, x0: Fraction) -> Fraction:
    """P(X2=1) by pushing the law through the transition matrix twice."""
    x = x0
    for _ in range(2):
        x = (1 - x) * p + x * (1 - q)
    return x


def _wrong_row_step(p: Fraction, q: Fraction, x: Fraction) -> Fraction:
    """One step using P(1->0) as if it were the staying probability."""
    return (1 - x) * p + x * q


@generator(
    "exd_x_s2_markov_step",
    level=17,
    presets={
        "easy": dict(tier="exam"),
        "medium": dict(tier="exam"),
        "hard": dict(tier="exam"),
        "exam": dict(tier="exam"),
    },
)
def exd_x_s2_markov_step(rng: random.Random, tier: str = "exam") -> dict:
    """Two steps of a two-state chain, forward or backward."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    den = rng.randint(3, 9)
    while True:
        p = Fraction(rng.randint(1, den - 1), den)
        q = Fraction(rng.randint(1, den - 1), den)
        lam = 1 - p - q
        if lam != 0:
            break
    # x0 at the chain's stationary point would leave every step unchanged and make the question degenerate
    while True:
        x0 = Fraction(rng.randint(1, den - 1), den)
        if x0 != p / (p + q):
            break
    x1 = p + lam * x0
    x2 = p * (1 + lam) + lam * lam * x0
    check(x2 == _two_step(p, q, x0) and x1 == (1 - x0) * p + x0 * (1 - q), "two-step Markov mismatch")
    check(0 <= x1 <= 1 and 0 <= x2 <= 1, "Markov probability outside [0,1]")
    backward = rng.random() < 0.5
    if backward:
        recovered = (x2 - p * (1 + lam)) / (lam * lam)
        check(recovered == x0, "two-step backward recovery mismatch")
        answer = x0
        text = phrase_backward(rng, p, q, x2)
        candidates = [
            (x2 - p) / lam,  # inverts only one step
            x2,
            (x2 - p) / (lam * lam),  # forgets the constant carried by the first step
            x1,
            1 - x0,  # the other state's probability
            (x2 - p * (1 + lam)) / lam,  # divides by lam once instead of twice
            x2 - p,  # subtracts one step's constant only
        ]
        working = [
            f"Each step maps P(X=1) to {rational_text(p)} + ({rational_text(lam)})*P(X=1), where {rational_text(lam)} = 1 - p - q.",
            f"Two steps: P(X2=1) = p(1+λ) + λ²·P(X0=1) with λ = {rational_text(lam)}.",
            f"P(X0=1) = ({rational_text(x2)} - {rational_text(p * (1 + lam))}) / ({rational_text(lam * lam)}) = {rational_text(x0)}.",
        ]
    else:
        answer = x2
        text = phrase_forward(rng, p, q, x0)
        candidates = [
            x1,  # stops after one step
            _wrong_row_step(p, q, _wrong_row_step(p, q, x0)),  # wrong row, twice
            p + lam * lam * x0,  # forgets the carried constant
            x0,
            1 - x2,  # the other state's probability
            p * (1 + lam),  # drops the initial distribution
            lam * lam * x0,  # drops the constants gained on each step
        ]
        working = [
            f"P(X1=1) = (1 - {rational_text(x0)})*{rational_text(p)} + {rational_text(x0)}*{rational_text(1 - q)} = {rational_text(x1)}.",
            f"P(X2=1) = (1 - {rational_text(x1)})*{rational_text(p)} + {rational_text(x1)}*{rational_text(1 - q)} = {rational_text(x2)}.",
        ]
    payload, answer_text = _payload(answer)
    return dict(
        text=text,
        latex="plain:" + text,
        answer=payload,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=_dist(answer, candidates),
    )


def phrase_forward(rng: random.Random, p: Fraction, q: Fraction, x0: Fraction) -> str:
    templates = (
        "A two-state chain has P(0→1)={p} and P(1→0)={q}. Given P(X₀=1)={x}, find P(X₂=1).",
        "A machine is in state 0 (idle) or state 1 (running). Each step it moves from idle to running with probability {p} "
        "and from running to idle with probability {q}. If it is running at the start with probability {x}, "
        "what is the probability it is running after two steps?",
        "The transition probabilities of a two-state chain are P(0→1)={p} and P(1→0)={q}. "
        "The initial distribution satisfies P(X₀=1)={x}. Calculate P(X₂=1).",
    )
    return templates[rng.randrange(len(templates))].format(p=rational_text(p), q=rational_text(q), x=rational_text(x0))


def phrase_backward(rng: random.Random, p: Fraction, q: Fraction, x2: Fraction) -> str:
    templates = (
        "A two-state chain has P(0→1)={p} and P(1→0)={q}, and P(X₂=1)={x}. Find P(X₀=1).",
        "A machine is in state 0 (idle) or state 1 (running). Each step it moves from idle to running with probability {p} "
        "and from running to idle with probability {q}. After two steps it is running with probability {x}. "
        "What was the probability that it was running at the start?",
        "Two steps of a two-state chain with P(0→1)={p} and P(1→0)={q} leave P(X₂=1)={x}. "
        "Determine the initial probability P(X₀=1).",
    )
    return templates[rng.randrange(len(templates))].format(p=rational_text(p), q=rational_text(q), x=rational_text(x2))


# ------------------------------------------------------------------------ stationary


@generator(
    "exd_x_s2_markov_stationary",
    level=17,
    presets={
        "easy": dict(tier="exam"),
        "medium": dict(tier="exam"),
        "hard": dict(tier="exam"),
        "exam": dict(tier="exam"),
    },
)
def exd_x_s2_markov_stationary(rng: random.Random, tier: str = "exam") -> dict:
    """A long-run transition rate of a two-state chain."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    den = rng.randint(3, 9)
    p = Fraction(rng.randint(1, den - 1), den)
    q = Fraction(rng.randint(1, den - 1), den)
    pi1 = p / (p + q)
    pi0 = 1 - pi1
    check(pi0 * p == pi1 * q, "stationary balance failed")
    kind = rng.choice(("up_move", "change_state", "from_pi"))
    pt, qt = rational_text(p), rational_text(q)
    if kind == "up_move":
        answer = pi0 * p
        text = rng.choice(
            (
                f"A two-state chain has P(0→1)={pt} and P(1→0)={qt}. In the long run, what fraction of steps "
                "are moves from state 0 to state 1?",
                f"For a chain with P(0→1)={pt} and P(1→0)={qt}, find the long-run proportion of steps "
                "on which the chain goes from state 0 to state 1.",
            )
        )
        candidates = [pi0, pi1, p, 2 * answer, pi1 * p, p * q]
        working = [
            f"π₁ = p/(p+q) = {rational_text(pi1)}, so π₀ = {rational_text(pi0)}.",
            f"Fraction of steps 0→1 = π₀·p = {rational_text(pi0)}·{pt} = {rational_text(answer)}.",
        ]
    elif kind == "change_state":
        answer = pi0 * p + pi1 * q
        text = rng.choice(
            (
                f"A two-state chain has P(0→1)={pt} and P(1→0)={qt}. In the long run, what fraction of steps "
                "change the state of the chain?",
                f"For a chain with P(0→1)={pt} and P(1→0)={qt}, find the long-run proportion of steps "
                "on which the chain is in a different state from the step before.",
            )
        )
        candidates = [pi0 * p, pi1, pi0, p + q, p * q, 1 - answer]
        working = [
            f"π₁ = p/(p+q) = {rational_text(pi1)}, so π₀ = {rational_text(pi0)}.",
            f"Changes: π₀·p + π₁·q = {rational_text(pi0 * p)} + {rational_text(pi1 * q)} = {rational_text(answer)}.",
        ]
    else:
        answer = 2 * pi0 * p
        text = rng.choice(
            (
                f"A two-state chain has P(0→1)={pt}, and its stationary probability of state 1 is {rational_text(pi1)}. "
                "In the long run, what fraction of steps change the state of the chain?",
                f"A chain on two states has P(0→1)={pt} and long-run probability {rational_text(pi1)} of being in state 1. "
                "Find the long-run proportion of steps on which the state changes.",
            )
        )
        check(pi0 * p + pi1 * q == answer, "from_pi change rate mismatch")
        candidates = [pi0 * p, pi1, pi0, q, p + q, 1 - answer]
        working = [
            f"Balance: π₀·p = π₁·q gives q = {rational_text(pi0)}·{pt}/{rational_text(pi1)} = {qt}.",
            f"Changes: π₀·p + π₁·q = 2·{rational_text(pi0 * p)} = {rational_text(answer)}.",
        ]
    check(0 < answer <= 1, "long-run rate outside (0,1]")
    payload, answer_text = _payload(answer)
    return dict(
        text=text,
        latex="plain:" + text,
        answer=payload,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=_dist(answer, candidates),
    )
