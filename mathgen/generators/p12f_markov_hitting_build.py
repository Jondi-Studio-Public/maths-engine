"""A two-state Markov hitting probability (``ucv1-markov-hitting-probability``, ``stochastic-processes``, topic
``p12f_markov_hitting``, ``answer: fraction``).

The chain has states ``s`` (start) and ``t`` (target), transition
probabilities ``p = P(s->t)`` and ``q = P(t->s)`` (so ``P(s->s) = 1-p``,
``P(t->t) = 1-q``). **The chain always starts in ``s``, never in ``t`` and
never from a distribution over both** -- every question states this
explicitly.

Boundary against the direct prerequisite ``markov-two-state-step``
(``mathgen/generators/university_applied.py``): that skill asks for
``P(X1=1)`` after ONE transition from a stated INITIAL DISTRIBUTION over
both states -- a single-step distributional update. This skill asks for a
REACHABILITY event over TWO transitions (a union or first-passage event)
starting from a fixed, non-target state -- a multi-step reachability
question, never a distribution.

**Load-bearing fact, used throughout this module and re-proved by the
independent oracle via explicit path enumeration:** every tier's true
answer is a function of ``p`` ALONE. ``q`` never enters the correct value at
any tier, because "reach"/"not yet reached"/"first reached" are all events
about whether/when the chain FIRST leaves ``s`` for ``t`` -- what happens
after the chain arrives at ``t`` (governed by ``q``) is irrelevant to having
already reached it. This is also exactly why "uses q" is the primary,
most-exercisable misconception at every tier: a learner who treats
"reached by step 2" as "IN state t at step 2" computes a DIFFERENT quantity
that legitimately depends on ``q`` (``P(X2=t)``), and that wrong quantity is
this module's primary distractor.

This skill's kernel is a discrete-state, discrete-step 2-step Markov path probability over an explicit
0/1-style transition matrix -- finite path enumeration, never a count distribution, never an exponential
factor. No rendered question of either skill could be swapped into the other's contract without a learner
immediately noticing the different objects being reasoned about (a count of arrivals vs a state-reachability
event).
"""

from __future__ import annotations

from fractions import Fraction
from math import gcd

from ..core import (
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

# A wide pool of DISTINCT, already-reduced fractions in (0, 1), spanning denominators 3..14 -- built
# from every coprime (k, d) pair so no value is a hidden duplicate of a simpler fraction (e.g. 2/4
# is never generated, because it is not itself coprime; 1/2 only ever arises from d=2, which is
# excluded).
_POOL_P = tuple(Fraction(k, d) for d in range(3, 15) for k in range(1, d) if gcd(k, d) == 1)
assert len(_POOL_P) >= 40  # sanity: the pool really is wide, not a handful of values

_START_TEMPLATES = (
    "A two-state Markov chain has states s and t. The chain always starts in "
    "state s. Each step, P(s→t)={p} and P(t→s)={q}. {question}",
    "A two-state Markov chain has states s and t; the chain starts in state "
    "s (never t). Each step, P(s→t)={p} and P(t→s)={q}. {question}",
    "Consider a two-state Markov chain (states s and t) that always starts "
    "in s. Each step, P(s→t)={p} and P(t→s)={q}. {question}",
)

_EASY_QUESTIONS = (
    "Find the probability that the chain reaches t within the first two transitions.",
    "What is the probability that the chain has reached t by the second transition?",
    "Find P(the chain reaches t within 2 transitions).",
)

_MEDIUM_QUESTIONS = (
    "Find the probability that the chain has NOT reached t after 2 transitions.",
    "What is the probability that the chain is still in s after 2 transitions, having never visited t?",
    "Find P(the chain has not reached t within 2 transitions).",
)

_HARD_QUESTIONS = (
    "Find the probability that the chain reaches t for the first time exactly at transition 2.",
    "What is the probability that the chain is still in s after 1 transition but has reached t by transition 2?",
    "Find P(the chain first reaches t exactly at transition 2).",
)

_EXAM_TEMPLATES = (
    "A two-state Markov chain always starts in state s and has one-step "
    "probability p = P(s→t), for some unknown p strictly between 0 and 1. "
    "The probability that the chain reaches t within 2 transitions is {h}. "
    "Find p.",
    "For a two-state Markov chain starting in state s, with unknown "
    "one-step probability p = P(s→t) (0 < p < 1), the chain reaches t "
    "within 2 transitions with probability {h}. Find p.",
    "A two-state Markov chain starts in state s. Its one-step probability "
    "p = P(s→t) is unknown (0 < p < 1), but the probability it reaches t "
    "within 2 transitions is {h}. Find p.",
)


def _draw_p_q(rng) -> tuple[Fraction, Fraction]:
    p = rng.choice(_POOL_P)
    q = rng.choice(_POOL_P)
    return p, q


def _path_probabilities(p: Fraction, q: Fraction) -> dict[str, Fraction]:
    """The four 2-step trajectories from s, built by EXPLICIT transition-
    matrix multiplication (hand-written, never a library matrix-power call)
    -- this is the same route the independent oracle uses, kept here too so
    construction's own `check_equivalent` calls below compare against a
    second, structurally different derivation rather than re-deriving the
    same closed form twice."""
    m = [[1 - p, p], [q, 1 - q]]  # rows/cols indexed 0=s, 1=t
    m2 = [[sum(m[i][k] * m[k][j] for k in range(2)) for j in range(2)] for i in range(2)]
    check(m2[0][0] + m2[0][1] == 1, "two-step transition row from s does not sum to 1")
    paths = {
        "ss": (1 - p) * (1 - p),
        "st": (1 - p) * p,
        "ts": p * q,
        "tt": p * (1 - q),
    }
    check(sum(paths.values()) == 1, "the four 2-step path probabilities from s do not sum to 1")
    check_equivalent(paths["ss"] + paths["ts"], m2[0][0], "M^2[s,s] does not match the s->s->s / s->t->s path sum")
    check_equivalent(paths["st"] + paths["tt"], m2[0][1], "M^2[s,t] does not match the s->s->t / s->t->t path sum")
    return paths


def _fraction_payload(value: Fraction) -> tuple[dict, str]:
    value = Fraction(value)
    return {"p": value.numerator, "q": value.denominator}, rational_text(value)


def _build_easy(rng) -> dict:
    for _attempt in range(_MAX_DRAW_ATTEMPTS):
        p, q = _draw_p_q(rng)
        answer = p * (2 - p)

        # Independent route: 1 - P(never reaches t) via explicit path
        # enumeration, not the closed-form p*(2-p) algebra above.
        paths = _path_probabilities(p, q)
        check_equivalent(answer, 1 - paths["ss"], "easy-tier closed form does not match 1 - P(stays at s twice)")
        check(Fraction(0) < answer < Fraction(1), "easy-tier answer outside (0,1)")

        candidates = {
            "uses-q-in-place-of-or-alongside-p": p + (1 - p) * (1 - q),
            "multiplies-instead-of-unions": p * p,
            "swaps-p-and-q": q * (2 - q),
            "off-by-one-transition": p,
        }
        try:
            check_pairwise_distinct(
                answer,
                *candidates.values(),
                msg=f"easy-tier misconception collision: p={p}, q={q}",
            )
        except VerificationError:
            continue
        break
    else:  # pragma: no cover - defensive; the draw space clears this easily
        raise VerificationError("could not draw a collision-free easy-tier question")

    payload, answer_text = _fraction_payload(answer)
    question = phrase(rng, _EASY_QUESTIONS)
    text = phrase(rng, _START_TEMPLATES, p=rational_text(p), q=rational_text(q), question=question)
    distractor_strs = [rational_text(v) for v in candidates.values()]
    distractors = pick_distractors(answer_text, distractor_strs, n=4)
    return dict(
        text=text,
        latex=text,
        answer=payload,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        working=[
            f"P(reach t within 2) = 1 - P(s→s→s) = 1 - {rational_text((1 - p) * (1 - p))} = {answer_text}.",
        ],
        distractors=distractors,
    )


def _build_medium(rng) -> dict:
    for _attempt in range(_MAX_DRAW_ATTEMPTS):
        p, q = _draw_p_q(rng)
        answer = (1 - p) * (1 - p)

        paths = _path_probabilities(p, q)
        check_equivalent(answer, paths["ss"], "medium-tier closed form does not match P(s→s→s)")
        check(Fraction(0) < answer < Fraction(1), "medium-tier answer outside (0,1)")

        candidates = {
            "forgets-complement-in-never-reach": 1 - (1 - p) * (1 - p),
            "uses-q-in-place-of-or-alongside-p": (1 - p) * (1 - q),
            "multiplies-instead-of-unions": (1 - p) * p,
            "swaps-p-and-q": (1 - q) * (1 - q),
        }
        try:
            check_pairwise_distinct(
                answer,
                *candidates.values(),
                msg=f"medium-tier misconception collision: p={p}, q={q}",
            )
        except VerificationError:
            continue
        break
    else:  # pragma: no cover - defensive; the draw space clears this easily
        raise VerificationError("could not draw a collision-free medium-tier question")

    payload, answer_text = _fraction_payload(answer)
    question = phrase(rng, _MEDIUM_QUESTIONS)
    text = phrase(rng, _START_TEMPLATES, p=rational_text(p), q=rational_text(q), question=question)
    distractor_strs = [rational_text(v) for v in candidates.values()]
    distractors = pick_distractors(answer_text, distractor_strs, n=4)
    return dict(
        text=text,
        latex=text,
        answer=payload,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        working=[
            f"P(never reach t in 2) = P(s→s→s) = (1-{rational_text(p)})² = {answer_text}.",
        ],
        distractors=distractors,
    )


def _build_hard(rng) -> dict:
    for _attempt in range(_MAX_DRAW_ATTEMPTS):
        p, q = _draw_p_q(rng)
        answer = (1 - p) * p

        paths = _path_probabilities(p, q)
        check_equivalent(answer, paths["st"], "hard-tier closed form does not match P(s→s→t)")
        check(Fraction(0) < answer < Fraction(1), "hard-tier answer outside (0,1)")

        candidates = {
            "off-by-one-transition": p,
            "multiplies-instead-of-unions": p * p,
            "uses-q-in-place-of-or-alongside-p": (1 - p) * (1 - q),
            "swaps-p-and-q": (1 - q) * q,
        }
        try:
            check_pairwise_distinct(
                answer,
                *candidates.values(),
                msg=f"hard-tier misconception collision: p={p}, q={q}",
            )
        except VerificationError:
            continue
        break
    else:  # pragma: no cover - defensive; the draw space clears this easily
        raise VerificationError("could not draw a collision-free hard-tier question")

    payload, answer_text = _fraction_payload(answer)
    question = phrase(rng, _HARD_QUESTIONS)
    text = phrase(rng, _START_TEMPLATES, p=rational_text(p), q=rational_text(q), question=question)
    distractor_strs = [rational_text(v) for v in candidates.values()]
    distractors = pick_distractors(answer_text, distractor_strs, n=4)
    return dict(
        text=text,
        latex=text,
        answer=payload,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        working=[
            f"P(first reach t at 2) = P(s→s→t) = (1-{rational_text(p)})×{rational_text(p)} = {answer_text}.",
        ],
        distractors=distractors,
    )


def _build_exam(rng) -> dict:
    """Inverse tier: h = p*(2-p) is stated; the learner recovers p via
    ``p = 1 - sqrt(1 - h)``. Algebraically ``1 - h = 1 - (2p - p^2) =
    (1-p)^2`` EXACTLY for every p -- a perfect square of a rational by
    construction, never a search -- so the square root is always exact and
    rational, never irrational. Uniqueness: the other root of
    ``p*(2-p) = h`` is ``p = 1 + sqrt(1-h) = 1 + (1-p) = 2 - p``, which is
    ``> 1`` for every p in the pool (every pool value is `< 1` by
    construction, `k in 1..d-1` over `d`), so it is never a valid
    probability and the inverse is genuinely unique -- proved below,
    unconditionally, every seed, not merely asserted."""
    for _attempt in range(_MAX_DRAW_ATTEMPTS):
        p, _q = _draw_p_q(rng)
        h = p * (2 - p)

        # Independent check: the algebraic identity 1 - h = (1-p)^2 holds
        # exactly (a perfect square of a rational), proved via direct
        # Fraction arithmetic rather than trusting the closed form used to
        # construct h in the first place.
        one_minus_h = 1 - h
        check_equivalent(one_minus_h, (1 - p) * (1 - p), "1 - h is not exactly (1-p)^2")
        # sqrt(1-h) as an exact Fraction: since 1-h == (1-p)^2 and p < 1,
        # (1-p) > 0 is itself the exact rational square root.
        sqrt_one_minus_h = 1 - p
        check(sqrt_one_minus_h * sqrt_one_minus_h == one_minus_h, "chosen square root does not square back to 1-h")
        recovered_p = 1 - sqrt_one_minus_h
        check_equivalent(recovered_p, p, "recovered p does not match the drawn p")
        other_root = 1 + sqrt_one_minus_h
        check(other_root > 1, "the other algebraic root is not provably invalid as a probability")

        candidates = {
            "reports-the-other-root": other_root,
            "confuses-p-with-1-minus-p": 1 - p,
            "copies-h-directly": h,
            "halves-h": h / 2,
        }
        try:
            check_pairwise_distinct(
                p,
                *candidates.values(),
                msg=f"exam-tier misconception collision: p={p}, h={h}",
            )
        except VerificationError:
            continue
        break
    else:  # pragma: no cover - defensive; the draw space clears this easily
        raise VerificationError("could not draw a collision-free exam-tier question")

    answer_text = rational_text(p)
    h_str = rational_text(h)
    text = phrase(rng, _EXAM_TEMPLATES, h=h_str)
    distractor_strs = [rational_text(v) for v in candidates.values()]
    distractors = pick_distractors(answer_text, distractor_strs, n=4)
    payload = {"p": p.numerator, "q": p.denominator}
    return dict(
        text=text,
        latex=text,
        answer=payload,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        working=[
            f"1 - h = 1 - {h_str} = {rational_text(one_minus_h)} = (1-p)², so 1-p = "
            f"{rational_text(sqrt_one_minus_h)} and p = {answer_text} (the other algebraic root, "
            f"{rational_text(other_root)}, exceeds 1 and is not a valid probability).",
        ],
        distractors=distractors,
    )


@generator(
    "p12f_markov_hitting",
    level=17,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def markov_hitting(rng, tier="easy"):
    """Compute the probability of reaching a target state within two
    transitions for a supplied two-state transition matrix (the skill's
    own contract, verbatim).

    easy:   P(reach t within 2 transitions | X0 = s) = p*(2-p) -- a union
            (reach-by) event, q supplied but irrelevant to this event.
    medium: P(NOT reached t after 2 transitions | X0 = s) = (1-p)^2 -- the
            complementary event, a different setup (not merely 1 - easy).
    hard:   P(first reached t exactly at transition 2 | X0 = s) = (1-p)*p --
            a first-passage event, excluding "already there at step 1".
    exam:   inverse -- h = p*(2-p) is given, recover the unknown p via
            p = 1 - sqrt(1-h), proved unique every seed.
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
