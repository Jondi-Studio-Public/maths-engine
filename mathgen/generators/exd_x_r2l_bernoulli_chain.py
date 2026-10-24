"""Exam question: an ordered run of independent Bernoulli trials, built up in three connected parts.

The stem gives the probability of FAILURE q on each trial (as a fraction), so the learner must first turn it into
the probability of success (a). Part (b) asks for one stated ordered sequence of the next three or four trials (a
plain product of p and q factors in the stated order, never a count of unordered outcomes). Part (c) extends that
sequence by one more stated trial and asks for the probability of the whole run, which is (b) times one more factor.
Parts (b) and (c) carry `follow_through`, so a learner who slips on (a) still earns (b) and (c) for using their own
p correctly.

Built backwards from the stated q: every answer is an exact fraction computed from it, and the key of each part is
re-derived by an independent route (the sum over a probability tree built trial by trial).
"""

from __future__ import annotations

import random
from fractions import Fraction
from math import comb

from ..answers import Distractor, exact_value
from ..core import attempts, check, generator, latex_answer, phrase, pick_distractors, rational_text

# (stem pattern, singular unit, plural unit, success phrase, failure phrase). The verbs are third person singular
# because they follow "the first <unit>", "the second <unit>", ...
_CONTEXTS = (
    (
        "A goal-kicker misses a kick at goal with probability {q} each time, and the kicks are independent.",
        "kick",
        "kicks",
        "scores",
        "misses",
    ),
    (
        "Each parcel posted by a small business is delivered late with probability {q}, independently of every "
        "other parcel.",
        "parcel",
        "parcels",
        "arrives on time",
        "is late",
    ),
    (
        "A seed from a packet fails to germinate with probability {q}, and the seeds act independently.",
        "seed",
        "seeds",
        "germinates",
        "fails to germinate",
    ),
    (
        "On a suburban tram route, a tram runs late with probability {q} on any trip, independently of every other "
        "trip.",
        "trip",
        "trips",
        "runs on time",
        "runs late",
    ),
    (
        "A netball player misses a shot at goal with probability {q} each time, and her shots are independent.",
        "shot",
        "shots",
        "scores",
        "misses",
    ),
    (
        "A machine stamps a part that is rejected at inspection with probability {q}, independently of every "
        "other part.",
        "part",
        "parts",
        "passes inspection",
        "is rejected",
    ),
)
_ORDINALS = ("first", "second", "third", "fourth", "fifth")
_ORDINAL_NUMBERS = ("1st", "2nd", "3rd", "4th", "5th")

_STEM_TAIL = ("", " Treat each one as a Bernoulli trial.")

_PROMPT_A = (
    "Find the probability that a single {unit} {success}.",
    "Find the probability that one {unit} chosen at random {success}.",
)
_PROMPT_B = (
    "Find the probability that, of the next {n} {plural}, {sequence}.",
    "Considering the next {n} {plural} in order, find the probability that {sequence}.",
)
_PROMPT_C = (
    "Hence find the probability that the first {n} {plural} go as described in (b) and the {nth} {unit} {word}.",
    "Using your answer to (b), find the probability that the first {n} {plural} go as described in (b) and that "
    "the {nth} {unit} {word}.",
)


def _denominators_ok(q: Fraction) -> bool:
    return 4 <= q.denominator <= 10 and 0 < q < Fraction(1, 2)  # the slip or fault named is the rarer outcome


def _run_probability(p: Fraction, outcomes: str) -> Fraction:
    """P(this exact ordered outcome string), by walking a probability tree one trial at a time."""
    total = Fraction(1)
    for o in outcomes:
        total *= p if o == "S" else 1 - p
    return total


def _within_unit_interval(candidates: list[Distractor]) -> list[Distractor]:
    """The candidates that are possible probabilities: strictly between 0 and 1."""
    return [c for c in candidates if 0 < c.value < 1]


def _build(rng: random.Random) -> dict | None:
    for _ in attempts(1000, "a failure probability and pattern"):
        den = rng.randint(4, 10)
        num = rng.randint(1, den - 1)
        q = Fraction(num, den)
        if not _denominators_ok(q) or q.denominator != den:
            continue
        n = rng.choice((3, 3, 4))
        pattern = "".join(rng.choice("SF") for _ in range(n))
        if "S" not in pattern or "F" not in pattern:
            continue
        break
    p = 1 - q
    s, f = pattern.count("S"), pattern.count("F")
    stem_fmt, unit, plural, success, failure = rng.choice(_CONTEXTS)
    extra = rng.choice("SF")
    extra_word = success if extra == "S" else failure

    prob_b = p**s * q**f
    prob_c = prob_b * (p if extra == "S" else q)

    # Independent route: the probability tree, one trial at a time, and an enumeration over all outcomes.
    check(prob_b == _run_probability(p, pattern), "sequence probability disagrees with the tree walk")
    check(prob_c == _run_probability(p, pattern + extra), "extended sequence disagrees with the tree walk")
    check(
        sum(_run_probability(p, format(i, f"0{n}b").replace("0", "F").replace("1", "S")) for i in range(2**n)) == 1,
        "the outcomes of the sequence do not sum to one",
    )

    words = [success if o == "S" else failure for o in pattern]
    clauses = [f"the {_ORDINALS[i]} {unit} {w}" for i, w in enumerate(words)]
    sequence = ", ".join(clauses[:-1]) + " and " + clauses[-1]
    stem = phrase(rng, (stem_fmt,), q=rational_text(q)) + rng.choice(_STEM_TAIL)

    a_text, b_text, c_text = rational_text(p), rational_text(prob_b), rational_text(prob_c)

    # (a) the probability of success
    a_cands = [
        Distractor(rational_text(q), "swapped-success-failure", q),
        Distractor(rational_text(p / q), "odds-as-probability", p / q),
        Distractor(rational_text(1 / q), "reciprocal-of-probability", 1 / q),
        Distractor(rational_text(q / p), "odds-as-probability", q / p),
        # A total one out: the sample of `den` trials counted as `den - 1`, or as `den + 1`.
        Distractor(
            rational_text(Fraction(den - num, den - 1)),
            "wrong-denominator-for-proportion",
            Fraction(den - num, den - 1),
        ),
        Distractor(
            rational_text(Fraction(den - num, den + 1)),
            "wrong-denominator-for-proportion",
            Fraction(den - num, den + 1),
        ),
    ]
    # (b) the ordered sequence
    swapped = p**f * q**s
    b_cands = [
        Distractor(rational_text(swapped), "swapped-success-failure", swapped),
        Distractor(rational_text(comb(n, s) * prob_b), "counts-orderings-in-ordered-sequence", comb(n, s) * prob_b),
        Distractor(
            rational_text(prob_b / (p if pattern[-1] == "S" else q)),
            "drops-one-trial-from-sequence",
            prob_b / (p if pattern[-1] == "S" else q),
        ),
        Distractor(rational_text(p**s), "ignores-failures-in-sequence", p**s),
        Distractor(rational_text(s * p + f * q), "adds-instead-of-multiplies", s * p + f * q),
    ]
    # (c) the extended run
    factor = p if extra == "S" else q
    other = p if extra == "F" else q
    c_cands = [
        Distractor(b_text, "forgets-the-extra-trial", prob_b),
        Distractor(rational_text(prob_b + factor), "adds-instead-of-multiplies", prob_b + factor),
        Distractor(rational_text(prob_b * other), "swapped-success-failure", prob_b * other),
        Distractor(
            rational_text(comb(n + 1, s + (extra == "S")) * prob_c),
            "counts-orderings-in-ordered-sequence",
            comb(n + 1, s + (extra == "S")) * prob_c,
        ),
    ]
    # Every part asks for a probability, and an option above 1 (or below 0) is ruled out on sight.
    a_d = pick_distractors(a_text, _within_unit_interval(a_cands), 3)
    b_d = pick_distractors(b_text, _within_unit_interval(b_cands), 3)
    c_d = pick_distractors(c_text, _within_unit_interval(c_cands), 3)
    for d in a_d + b_d + c_d:
        check(isinstance(d, Distractor), "a distractor lost its tag")
    if not len(a_d) == len(b_d) == len(c_d) == 3:
        return None  # too few distinct options in range for this draw: draw again
    ft_b = f"part_a**{s}*(1 - part_a)**{f}"
    ft_c = f"part_b*{'part_a' if extra == 'S' else '(1 - part_a)'}"
    parts = [
        dict(
            prompt=phrase(rng, _PROMPT_A, unit=unit, success=success),
            answer_kind="fraction",
            answer=exact_value(p),
            answer_text=a_text,
            answer_latex=latex_answer(a_text),
            marks=1,
            sensible_range=(0, 1),
            working=[f"P(success) = 1 - P(failure) = 1 - {rational_text(q)} = {a_text}."],
            distractors=a_d,
        ),
        dict(
            prompt=phrase(rng, _PROMPT_B, n=n, plural=plural, sequence=sequence)
            + " Give your answer as a fraction in simplest form.",
            answer_kind="fraction",
            answer=exact_value(prob_b),
            answer_text=b_text,
            answer_latex=latex_answer(b_text),
            marks=2,
            sensible_range=(0, 1),
            follow_through=ft_b,
            working=[
                f"The trials are independent, so multiply along the sequence {pattern}: "
                f"{' × '.join(t if e == 1 else f'{t}^{e}' for t, e in (('p', s), ('q', f)) if e)} where p = {a_text} and q = {rational_text(q)}.",
                f"= {b_text}.",
            ],
            distractors=b_d,
        ),
        dict(
            prompt=phrase(
                rng,
                _PROMPT_C,
                n=n,
                plural=plural,
                unit=unit,
                nth=_ORDINAL_NUMBERS[n],
                word=extra_word,
            )
            + " Give your answer as a fraction in simplest form.",
            answer_kind="fraction",
            answer=exact_value(prob_c),
            answer_text=c_text,
            answer_latex=latex_answer(c_text),
            marks=2,
            sensible_range=(0, 1),
            follow_through=ft_c,
            working=[
                f"One more independent trial multiplies the probability by {'p' if extra == 'S' else 'q'} = "
                f"{rational_text(p if extra == 'S' else q)}.",
                f"{b_text} x {rational_text(p if extra == 'S' else q)} = {c_text}.",
            ],
            distractors=c_d,
        ),
    ]
    return dict(text=stem, parts=parts)


@generator("exd_x_r2l_bernoulli_chain", level=12, presets={"exam": dict(tier="exam")})
def exd_x_r2l_bernoulli_chain(rng: random.Random, tier: str = "exam") -> dict:
    """Failure probability to success probability, then an ordered run of trials, then the run extended by one."""
    check(tier == "exam", "exd_x_r2l_bernoulli_chain: only the exam tier exists")
    for _ in attempts(200, "a draw with three in-range options for every part"):
        built = _build(rng)
        if built is not None:
            return built
    raise AssertionError("unreachable")
