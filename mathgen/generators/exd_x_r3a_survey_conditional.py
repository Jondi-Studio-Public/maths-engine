"""Exam source of `conditional-probability-counts`: a two-set survey worked up to a conditional probability.

The practice tiers read P(A | B) off a table of counts. An exam question gives only the margins of a two-set survey
(how many do each, how many do neither) and builds the answer in stages:

(a) how many do both activities (inclusion and exclusion);
(b) how many do exactly one of the two;
(c) the probability that a student chosen from those who do exactly one does the first activity;
(d) the amount by which P(A and B) differs from P(A) x P(B), the standard check for independence.

Parts (b) to (d) follow through from the earlier parts.

## Construction

The four region counts (only A, only B, both, neither) are drawn first, so every key is an exact fraction. An
independent route builds the four regions from the stated margins with SymPy and recomputes (c) and (d).
"""

from __future__ import annotations

from fractions import Fraction

from sympy import Rational, Symbol, solve

from mathgen.answers import Distractor, exact_value
from mathgen.core import attempts, check, generator, latex_answer, pick_distractors, rational_text
from ._exd_r3_distinct import distinct_part_answers

_STUDENTS = [("students at a school", "student"), ("students in Year 10", "student")]
_CLUB = [("members of a sports club", "member"), ("people at a community centre", "person")]
_STREET = [("people in a street", "person"), ("households in a street", "household")]
# (plural verb phrase, singular verb phrase, plural verb phrase, singular verb phrase, who it is asked of)
_ACTIVITIES = [
    ("play netball", "plays netball", "play basketball", "plays basketball", _CLUB + _STUDENTS),
    ("study French", "studies French", "study Japanese", "studies Japanese", _STUDENTS),
    ("own a dog", "owns a dog", "own a cat", "owns a cat", _STREET[:1]),
    ("walk to school", "walks to school", "ride a bike to school", "rides a bike to school", _STUDENTS),
    ("play the violin", "plays the violin", "play the piano", "plays the piano", _STUDENTS),
    (
        "swim at the weekend",
        "swims at the weekend",
        "play cricket at the weekend",
        "plays cricket at the weekend",
        _CLUB,
    ),
]


def _t(value) -> str:
    return rational_text(value)


def _part(prompt, value, marks, distractors, working, follow_through=None, rng_range=(0, None)):
    part = dict(
        prompt=prompt,
        answer_kind="fraction",
        answer=exact_value(value),
        answer_text=_t(value),
        answer_latex=latex_answer(_t(value)),
        marks=marks,
        sensible_range=rng_range,
        working=working,
        distractors=distractors,
    )
    if follow_through:
        part["follow_through"] = follow_through
    return part


def _tag(pairs, answer, lo=None, hi=None):
    out = []
    for value, mid in pairs:
        value = Fraction(value)
        if value == answer or (lo is not None and value < lo) or (hi is not None and value > hi):
            continue
        out.append(Distractor(_t(value), mid, value))
    return pick_distractors(_t(answer), out, 3)


def _build(rng):
    for _ in attempts(3000, "survey counts with a non-trivial independence check"):
        only_a, only_b, both, neither = rng.randint(3, 22), rng.randint(3, 22), rng.randint(2, 15), rng.randint(2, 15)
        n = only_a + only_b + both + neither
        a, b = only_a + both, only_b + both
        exactly = only_a + only_b
        p_cond = Fraction(only_a, exactly)
        diff = Fraction(both, n) - Fraction(a, n) * Fraction(b, n)
        if diff == 0 or p_cond.denominator == 1:
            continue
        keys = {both, exactly}
        shown = {n, a, b, neither}
        if keys & shown or len({only_a, only_b, both}) < 3:
            continue
        return only_a, only_b, both, neither, n, a, b, exactly, p_cond, diff
    raise RuntimeError("no draw")


@generator("exd_x_r3a_survey_conditional", level=12, presets={"exam": {"tier": "exam"}})
@distinct_part_answers
def exd_x_r3a_survey_conditional(rng, tier: str = "exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(200, "enough distractors in every part"):
        only_a, only_b, both, neither, n, a, b, exactly, p_cond, diff = _build(rng)
        # Independent route: SymPy solves the regions from the margins.
        both_s = Symbol("both")
        (solved,) = solve(both_s + (a - both_s) + (b - both_s) + neither - n, both_s)
        check(solved == both, "SymPy finds a different overlap")
        check(
            Rational(a - solved, (a - solved) + (b - solved)) == Rational(p_cond.numerator, p_cond.denominator), "(c)"
        )
        check(
            Rational(solved, n) - Rational(a, n) * Rational(b, n) == Rational(diff.numerator, diff.denominator), "(d)"
        )
        union = n - neither
        pa, pb = Fraction(a, n), Fraction(b, n)
        a_d = _tag(
            [
                (a + b, "union-formula-without-overlap-subtracted"),
                (a + b - n, "neither-group-left-out-of-union"),
                (n - neither - a, "only-b-count-reported-as-both"),
                (n - a - b, "total-minus-counts-for-overlap"),
            ],
            Fraction(both),
            lo=1,
        )
        b_d = _tag(
            [
                (a + b, "overlap-counted-twice"),
                (union, "at-least-one-for-exactly-one"),
                (n - both, "neither-included-in-exactly-one"),
                (a + b - both, "at-least-one-for-exactly-one"),
            ],
            Fraction(exactly),
            lo=1,
        )
        c_d = _tag(
            [
                (Fraction(a, exactly), "overlap-included-in-numerator"),
                (Fraction(only_a, union), "wrong-denominator"),
                (Fraction(only_a, n), "wrong-denominator"),
                (Fraction(only_b, exactly), "complement-reported"),
                (Fraction(both, a), "reversed-conditioning"),
            ],
            p_cond,
            lo=0,
            hi=1,
        )
        d_d = _tag(
            [
                (Fraction(both, n), "independence-product-ignored"),
                (pa * pb, "joint-probability-ignored"),
                (-diff, "reversed-subtraction"),
                (Fraction(both, n) - pa, "single-probability-for-product"),
            ],
            diff,
            lo=-1,
            hi=1,
        )
        if min(len(a_d), len(b_d), len(c_d), len(d_d)) >= 2:
            break

    plural_a, single_a, plural_b, single_b, groups = rng.choice(_ACTIVITIES)
    group, one = rng.choice(groups)
    stem = (
        f"Of {n} {group}, {a} {plural_a}, {b} {plural_b} and {neither} do neither. "
        f"A {one} is chosen at random. Let A be the event that the {one} {single_a} and B the event that the {one} "
        f"{single_b}."
    )
    ft_b = f"{union} - part_a"
    ft_c = f"({a} - part_a)/part_b"
    ft_d = f"part_a/{n} - ({a}/{n})*({b}/{n})"
    return dict(
        text=stem,
        parts=[
            _part(
                f"How many of the {n} are in both A and B?",
                Fraction(both),
                2,
                a_d,
                [f"{a} + {b} + {neither} - {n} = {both} do both (those who do neither are outside both sets)."],
            ),
            _part(
                "Using your answer to (a), how many of the group are in exactly one of A and B?",
                Fraction(exactly),
                2,
                b_d,
                [f"At least one: {n} - {neither} = {union}. Exactly one: {union} - {both} = {exactly}."],
                ft_b,
            ),
            _part(
                f"A {one} who is in exactly one of A and B is chosen at random. Using your answers to (a) and (b), "
                "find the probability that this person is in A, as a fraction in lowest terms.",
                p_cond,
                3,
                c_d,
                [f"Only A: {a} - {both} = {only_a}. P = {only_a}/{exactly} = {_t(p_cond)}."],
                ft_c,
                rng_range=(0, 1),
            ),
            _part(
                "Using your answer to (a), find P(A ∩ B) - P(A) × P(B), as a fraction in lowest terms.",
                diff,
                3,
                d_d,
                [
                    f"{both}/{n} - ({a}/{n})({b}/{n}) = {_t(diff)}. This is not 0, so the activities are not independent."
                ],
                ft_d,
                rng_range=(-1, 1),
            ),
        ],
    )
