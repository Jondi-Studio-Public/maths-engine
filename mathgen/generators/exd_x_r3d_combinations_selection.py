"""Exam source of `combinations`: all selections, the selections meeting a condition, and the chance of one.

The practice tier counts the ways to choose r objects from n. An exam question sets a selection with a condition and
joins three answers:

(a) the number of selections with no condition, C(n, r);
(b) the number that meet the condition, found by splitting into groups, by a complement, or by fixing members;
(c) the probability, as a percentage to one decimal place, that a random selection meets the condition, which follows
    through from (a) and (b) as 100 b / a.

Four conditions are drawn: exactly j from one group; at least one from each of two groups (the complement of two
cases); two named members both included; and at most j from one group. Every count is checked by listing the
selections with `itertools.combinations` and testing the condition directly.
"""

from __future__ import annotations

import itertools
import math
import random
from fractions import Fraction

from mathgen.core import (
    GRADER_TOLERANCE,
    attempts,
    check,
    generator,
    latex_answer,
    round_half_up,
    stated_precision_string,
)

from .exd_x_r3d_common import tagged
from ._exd_r3_distinct import distinct_part_answers

# (selection noun, plural, first group, second group, word for one of the whole pool)
_CONTEXTS = (
    ("committee", "committees", ("teacher", "teachers"), ("student", "students"), "people"),
    ("team", "teams", ("boy", "boys"), ("girl", "girls"), "players"),
    ("jury", "juries", ("woman", "women"), ("man", "men"), "people"),
    ("tasting panel", "tasting panels", ("chef", "chefs"), ("diner", "diners"), "people"),
    ("working group", "working groups", ("engineer", "engineers"), ("designer", "designers"), "staff"),
    ("delegation", "delegations", ("junior", "juniors"), ("senior", "seniors"), "students"),
)
_PROMPT_A = (
    "How many different {nouns} can be chosen?",
    "Find the number of different {nouns} that can be chosen.",
)
_PROMPT_C = (
    "Hence find the probability, as a percentage correct to 1 decimal place, that a {noun} chosen at random {cond}.",
    "Using parts (a) and (b), find the percentage chance, correct to 1 decimal place, that a randomly chosen {noun} {cond}.",
)


def _c(n: int, r: int) -> int:
    return math.comb(n, r) if 0 <= r <= n else 0


def _pct(value: Fraction) -> str:
    return stated_precision_string(round_half_up(value, 1), 1)


@generator("exd_x_r3d_combinations_selection", level=13, presets={"exam": dict(tier="exam")})
@distinct_part_answers
def exd_x_r3d_combinations_selection(rng: random.Random, tier: str = "exam") -> dict:
    """Count all selections, those meeting a condition, and the probability of the condition."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(2000, "a selection problem with usable distractors"):
        noun, nouns, gx_forms, gy_forms, people = rng.choice(_CONTEXTS)
        gx, gy = gx_forms[1], gy_forms[1]
        kind = rng.choice(["exactly", "each", "both", "atmost"])
        x, y = rng.randint(4, 9), rng.randint(4, 9)
        r = rng.randint(3, 6)
        n = x + y
        if kind == "both":
            n = rng.randint(8, 14)
        if r >= n:
            continue
        total = _c(n, r)
        if kind == "exactly":
            j = rng.randint(1, r - 1)
            if j > y or r - j > x:
                continue
            fav = _c(y, j) * _c(x, r - j)
            cond_pl, cond_sg = f"have exactly {j} {gy_forms[j != 1]}", f"has exactly {j} {gy_forms[j != 1]}"
            wrong = [
                (_c(y, j) + _c(x, r - j), "counts-added-not-multiplied"),
                (_c(y, j) * _c(x, j), "second-group-size-wrong"),
                (_c(n, j), "only-one-group-chosen"),
            ]
            ok = lambda s: sum(1 for i in s if i >= x) == j  # noqa: E731
        elif kind == "each":
            if r > min(x, y):
                continue
            fav = total - _c(x, r) - _c(y, r)
            cond_pl = f"have at least one of the {gx} and at least one of the {gy}"
            cond_sg = f"has at least one of the {gx} and at least one of the {gy}"
            wrong = [
                (total - _c(x, r), "one-excluded-case-missed"),
                (total - 2, "complement-counted-as-two-cases"),
                (x * y * _c(n - 2, r - 2), "overcounts-by-choosing-required-members-first"),
            ]
            ok = lambda s: 0 < sum(1 for i in s if i >= x) < r  # noqa: E731
        elif kind == "both":
            fav = _c(n - 2, r - 2)
            cond_pl, cond_sg = "include both twins", "includes both twins"
            wrong = [
                (_c(n, r - 2), "required-members-not-removed-from-pool"),
                (_c(n - 2, r), "required-members-excluded"),
                (total - _c(n - 2, r), "at-least-one-for-both"),
            ]
            ok = lambda s: 0 in s and 1 in s  # noqa: E731
            x = y = 0
        else:
            j = rng.randint(1, r - 1)
            if j > y:
                continue
            fav = sum(_c(y, i) * _c(x, r - i) for i in range(j + 1))
            cond_pl, cond_sg = f"have at most {j} {gy_forms[j != 1]}", f"has at most {j} {gy_forms[j != 1]}"
            wrong = [
                (_c(y, j) * _c(x, r - j), "only-the-boundary-case-counted"),
                (sum(_c(y, i) * _c(x, r - i) for i in range(j)), "upper-case-excluded"),
                (total - fav, "complement-reported"),
            ]
            ok = lambda s: sum(1 for i in s if i >= x) <= j  # noqa: E731
        if fav <= 0 or fav >= total or total > 6000 or round_half_up(100 * Fraction(fav, total), 1) >= 100:
            continue
        # Independent route: list the selections and test the condition.
        counted = sum(1 for s in itertools.combinations(range(n), r) if ok(s))
        check(counted == fav, "listing the selections disagrees with the formula")
        pct = 100 * Fraction(fav, total)
        a_d = tagged(
            str(total),
            [
                (math.perm(n, r), "ordered-count-for-selection"),
                (_c(n + r - 1, r), "selection-with-repeats-assumed"),
                (n**r, "repeats-allowed-used-for-no-repeats"),
            ],
        )
        b_d = tagged(str(fav), wrong)
        key_c = _pct(pct)
        c_cands = [
            (round_half_up(100 * Fraction(fav, total - fav), 1), "odds-for-probability"),
            (round_half_up(100 - pct, 1), "complement-reported"),
            (round_half_up(100 * Fraction(fav, math.perm(n, r)), 1), "ordered-count-for-selection"),
        ]
        c_d = tagged(
            key_c,
            [c for c in c_cands if c[0] >= Fraction(1, 2)],
            fmt=lambda v: stated_precision_string(v, 1),
            tolerance=GRADER_TOLERANCE,
        )
        if len(a_d) >= 3 and len(b_d) >= 2 and len(c_d) >= 2:
            break

    if kind == "both":
        stem = f"A {noun} of {r} is chosen from {n} {people}. Two of the {people} are twins."
    else:
        stem = f"A {noun} of {r} is chosen from {x} {gx} and {y} {gy}."
    return dict(
        text=stem,
        parts=[
            dict(
                prompt=rng.choice(_PROMPT_A).format(nouns=nouns),
                answer_kind="number",
                answer=total,
                answer_text=str(total),
                answer_latex=latex_answer(total),
                marks=1,
                sensible_range=(0, None),
                working=[f"C({n}, {r}) = {total}."],
                distractors=a_d,
            ),
            dict(
                prompt=f"How many of these {nouns} {cond_pl}?",
                answer_kind="number",
                answer=fav,
                answer_text=str(fav),
                answer_latex=latex_answer(fav),
                marks=3,
                sensible_range=(0, None),
                working=[f"The number of {nouns} that {cond_pl.split(' ', 1)[0]} the condition is {fav}."],
                distractors=b_d,
            ),
            dict(
                prompt=rng.choice(_PROMPT_C).format(noun=noun, cond=cond_sg),
                answer_kind="number",
                answer=float(round_half_up(pct, 1)),
                answer_text=key_c,
                answer_latex=latex_answer(key_c),
                marks=2,
                follow_through="100*part_b/part_a",
                sensible_range=(0, 100),
                working=[f"{fav} ÷ {total} × 100 = {key_c}%."],
                distractors=c_d,
            ),
        ],
    )
