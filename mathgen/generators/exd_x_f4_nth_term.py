"""Exam-tier questions for using an nth-term rule.

Where the practice tiers hand the learner a rule and ask for one term (or the
position of one value), these questions make the learner work harder with the
rule. Three kinds, drawn with equal chance:

- ``from_two_terms``: two non-adjacent terms of an arithmetic sequence are
  given and the learner must recover the rule before evaluating a far term.
- ``first_exceeds``: a quadratic rule and a threshold; find the first term
  position whose value exceeds the threshold.
- ``count_in_range``: a linear rule and an interval; count the terms that lie
  in the interval (two inequalities, with the end-point cases to think about).

Every question is built backwards from the chosen answer or checked against a
brute-force scan of the sequence, so the key is exact for every seed.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import check, generator, join_terms, numeric_answer, phrase, pick_distractors

_KINDS = ("from_two_terms", "first_exceeds", "count_in_range")

_TWO_TERMS = (
    "The {p1} term of an arithmetic sequence is {t1} and the {p2} term is {t2}. Find the {k} term.",
    "An arithmetic sequence has {t1} as its {p1} term and {t2} as its {p2} term. What is its {k} term?",
    "In an arithmetic sequence, term number {p1n} equals {t1} and term number {p2n} equals {t2}. Work out term number {kn}.",
)
_EXCEEDS = (
    "The nth term of a sequence is T_n = {rule}. Find the smallest value of n for which T_n is greater than {m}.",
    "A sequence has nth term T_n = {rule}. What is the first term position n at which the term exceeds {m}?",
    "For the sequence with rule T_n = {rule}, find the least positive integer n such that T_n > {m}.",
)
_COUNT = (
    "The nth term of a sequence is T_n = {rule}. How many terms of the sequence lie between {lo} and {hi} inclusive?",
    "A sequence has rule T_n = {rule}, for n = 1, 2, 3, and so on. Count the terms whose value is at least {lo} and at most {hi}.",
    "For the sequence T_n = {rule}, how many of its terms are in the interval from {lo} to {hi}, including both ends?",
)


def _ordinal(n):
    """1st, 2nd, 3rd, 4th ... 11th, 12th, 13th, 21st ..."""
    if n % 100 in (11, 12, 13):
        return f"{n}th"
    return f"{n}" + {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def _two_terms(rng):
    for _ in range(100):
        a1 = rng.randint(-12, 25)
        d = rng.choice([x for x in range(-9, 10) if abs(x) >= 2])
        p1 = rng.randint(3, 6)
        p2 = p1 + rng.randint(2, 5)
        k = rng.randint(18, 40)
        t1 = a1 + d * (p1 - 1)
        t2 = a1 + d * (p2 - 1)
        key = a1 + d * (k - 1)
        gap = p2 - p1
        check((t2 - t1) % gap == 0 and (t2 - t1) // gap == d)
        cands = [
            t1 + (t2 - t1) * (k - p1),  # treats the gap between the given terms as one step
            key + d,  # counts from term 0
            key - d,  # counts one step short
            t2 + d * (k - p1),  # steps from the wrong given term
        ]
        distractors = pick_distractors(str(key), [str(c) for c in cands])
        if len(distractors) >= 3:
            text = phrase(
                rng, _TWO_TERMS, p1=_ordinal(p1), p2=_ordinal(p2), t1=t1, t2=t2, k=_ordinal(k), p1n=p1, p2n=p2, kn=k
            )
            return text, key, distractors
    check(False, "exd_x_f4_nth_term: no draw with three distinct distractors")


def _first_exceeds(rng):
    for _ in range(200):
        a = rng.randint(1, 3)
        b = rng.randint(-4, 6)
        c = rng.randint(-5, 10)
        n0 = rng.randint(6, 14)

        def term(n):
            return a * n * n + b * n + c

        if term(n0) - term(n0 - 1) < 3:
            continue
        # the threshold sits at or above the previous term and strictly below this one
        m = term(n0) - rng.randint(1, min(8, term(n0) - term(n0 - 1) - 1)) if term(n0) - term(n0 - 1) > 1 else None
        if m is None or m < term(n0 - 1):
            continue
        key = next(n for n in range(1, 200) if term(n) > m)
        if key != n0:
            continue
        rule = join_terms([(a, "n^2"), (b, "n"), (c, "")])
        cands = [
            n0 - 1,  # the last position that does not exceed the threshold
            n0 + 1,
            term(n0),  # the term value rather than its position
            term(n0 - 1),
        ]
        distractors = pick_distractors(str(n0), [str(x) for x in cands])
        if len(distractors) >= 3:
            return phrase(rng, _EXCEEDS, rule=rule, m=m), n0, distractors
    check(False, "exd_x_f4_nth_term: no first_exceeds draw")


def _count_in_range(rng):
    for _ in range(200):
        p = rng.randint(3, 9)
        q = rng.randint(-6, 12)
        n_lo = rng.randint(2, 8)
        n_hi = n_lo + rng.randint(4, 12)
        # interval ends are chosen to fall off the sequence, so the ends need real thought
        lo = p * n_lo + q - rng.randint(0, p - 1)
        hi = p * n_hi + q + rng.randint(0, p - 1)
        count = sum(1 for n in range(1, 400) if lo <= p * n + q <= hi)
        check(count == n_hi - n_lo + 1, "exd_x_f4_nth_term: count_in_range construction disagrees with the scan")
        if lo <= p + q:
            continue  # keep the first term below the interval, so the lower bound matters
        rule = join_terms([(p, "n"), (q, "")])
        cands = [
            (hi - lo) // p,  # fencepost: gaps instead of terms
            count + 1,
            (hi - q) // p,  # counts every term up to hi, ignoring the lower end
            -(-(hi - lo) // p) + 1,
        ]
        distractors = pick_distractors(str(count), [str(x) for x in cands])
        if len(distractors) >= 3:
            return phrase(rng, _COUNT, rule=rule, lo=lo, hi=hi), count, distractors
    check(False, "exd_x_f4_nth_term: no count_in_range draw")


_WORKING = {
    "from_two_terms": "Find the common difference from the two given terms, then the first term, then evaluate the rule: {key}.",
    "first_exceeds": "Find where the quadratic rule crosses the threshold, then check the terms either side: n = {key}.",
    "count_in_range": "Find the first and last positions whose terms lie in the interval, then count them: {key}.",
}
_BUILDERS = {"from_two_terms": _two_terms, "first_exceeds": _first_exceeds, "count_in_range": _count_in_range}


@generator("exd_x_f4_nth_term", level=10, presets={"exam": dict(tier="exam")})
def exd_x_f4_nth_term(rng, tier: str = "exam"):
    kind = rng.choice(_KINDS)
    text, key, distractors = _BUILDERS[kind](rng)
    check(isinstance(key, int))
    return dict(
        text=text,
        latex=text,
        answer=numeric_answer(Fraction(key)),
        answer_text=str(key),
        answer_latex=str(key),
        working=[_WORKING[kind].format(key=key)],
        distractors=distractors,
    )
