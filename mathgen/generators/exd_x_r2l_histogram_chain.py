"""Exam question: probabilities for a continuous variable from a histogram with one frequency missing.

The stem gives five equal-width classes with the grand total, but one class frequency is not shown. Part (a) finds
the missing frequency from the total. Part (b) asks for the probability that the variable lies between two cut
values that both fall INSIDE classes, so the learner must take a fraction of each end class (uniform spread) and
whole classes in between, one of which is the class found in (a). Part (c) conditions on the lower cut: the
probability of being below the upper cut given that the value is at least the lower cut. Parts (b) and (c) carry
`follow_through` from the earlier answers.

Built backwards from the table. The probability of part (b) is re-derived independently by integrating a density
that is constant on each class (the overlap of the interval with each class, times that class's frequency).
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..answers import Distractor, exact_value
from ..core import GRADER_TOLERANCE, attempts, check, generator, latex_answer, pick_distractors, rational_text

# (description with {n}, subject of the probability, unit, thing, widths, starts)
_CONTEXTS = (
    (
        "the waiting time, in minutes, of {n} patients at a medical centre",
        "the waiting time of a randomly chosen patient",
        "minutes",
        (10,),
        (0, 10),
    ),
    (
        "the mass, in grams, of {n} eggs sold by a farm",
        "the mass of a randomly chosen egg",
        "grams",
        (5, 10),
        (45, 50, 55),
    ),
    (
        "the time, in seconds, taken by {n} swimmers to complete 100 metres",
        "the time of a randomly chosen swimmer",
        "seconds",
        (10,),
        (60, 70, 80),
    ),
    (
        "the lifetime, in hours, of {n} torch batteries",
        "the lifetime of a randomly chosen battery",
        "hours",
        (20,),
        (0, 20, 100),
    ),
    (
        "the distance, in metres, of {n} javelin throws at a school carnival",
        "the distance of a randomly chosen throw",
        "metres",
        (5, 10),
        (10, 15, 20),
    ),
)
_K = 5  # number of classes


def _overlap_probability(freqs, start, width, lo, hi, n) -> Fraction:
    """P(lo <= X < hi) with the density constant on each class: sum of frequency x overlap / width / n."""
    total = Fraction(0)
    for i, f in enumerate(freqs):
        a, b = start + i * width, start + (i + 1) * width
        overlap = max(Fraction(0), min(Fraction(b), hi) - max(Fraction(a), lo))
        total += f * overlap / width
    return total / n


class _Redraw(Exception):
    """A draw that cannot be given enough distinct distractors; the generator draws again."""


def _build(rng: random.Random) -> dict:
    desc_fmt, subject, unit, widths, starts = rng.choice(_CONTEXTS)
    width = rng.choice(widths)
    start = rng.choice(starts)
    for _ in attempts(1000, "a table and two interior cut values"):
        i = rng.choice((0, 1))
        j = rng.choice((3, 4))
        missing = rng.randint(i + 1, j - 1)
        u = Fraction(rng.randint(1, 4), 5)  # cut position inside class i, as a fraction of the class
        v = Fraction(rng.randint(1, 4), 5)  # cut position inside class j
        freqs = [rng.randint(4, 30) for _ in range(_K)]
        n = sum(freqs)
        known_sum = n - freqs[missing]
        lo = Fraction(start + i * width) + u * width
        hi = Fraction(start + j * width) + v * width
        if lo.denominator != 1 or hi.denominator != 1 or n > 150:
            continue
        break
    lo_i, hi_i = int(lo), int(hi)

    prob_b = (freqs[i] * (1 - u) + sum(freqs[i + 1 : j]) + freqs[j] * v) / n
    tail_count = freqs[i] * (1 - u) + sum(freqs[i + 1 :])
    prob_c = prob_b * n / tail_count
    check(prob_b == _overlap_probability(freqs, start, width, lo, hi, n), "interval probability disagrees")
    check(
        prob_c
        == _overlap_probability(freqs, start, width, lo, hi, n)
        / _overlap_probability(freqs, start, width, lo, Fraction(start + _K * width), n),
        "conditional probability disagrees with the ratio of overlaps",
    )
    check(0 < prob_b < prob_c < 1, "probabilities out of order")

    shown = ", ".join(
        f"[{start + k * width}, {start + (k + 1) * width}): {'not shown' if k == missing else freqs[k]}"
        for k in range(_K)
    )
    stem = (
        f"A histogram summarises {desc_fmt.format(n=n)}. The class frequencies are {shown}. "
        f"Every observation lies in one of these classes. Assume that the values are spread uniformly within "
        f"each class."
    )
    miss_lo, miss_hi = start + missing * width, start + (missing + 1) * width

    a_text, b_text, c_text = str(freqs[missing]), rational_text(prob_b), rational_text(prob_c)

    # (a) the missing frequency
    others = [k for k in range(_K) if k != missing]
    a_cands = [
        Distractor(str(known_sum), "reports-known-sum-not-remainder", known_sum),
        Distractor(
            str(freqs[missing] + freqs[others[0]]), "left-a-class-out-of-the-sum", freqs[missing] + freqs[others[0]]
        ),
        Distractor(str(n), "reports-total-as-missing-value", n),
        Distractor(
            str(freqs[missing] - freqs[others[-1]])
            if freqs[missing] > freqs[others[-1]]
            else str(freqs[missing] + 1 + freqs[others[-1]]),
            "double-counted-a-class-in-the-sum",
            freqs[missing] - freqs[others[-1]]
            if freqs[missing] > freqs[others[-1]]
            else freqs[missing] + 1 + freqs[others[-1]],
        ),
    ]
    # (b) the interval probability
    whole = Fraction(freqs[i] + sum(freqs[i + 1 : j]) + freqs[j], n)
    inverted = (freqs[i] * u + sum(freqs[i + 1 : j]) + freqs[j] * (1 - v)) / n
    no_ends = Fraction(sum(freqs[i + 1 : j]), n)
    in_range = freqs[i] * (1 - u) + sum(freqs[i + 1 : j]) + freqs[j] * v
    own_denominator = in_range / (freqs[i] + sum(freqs[i + 1 : j]) + freqs[j])
    b_cands = [
        Distractor(rational_text(whole), "counts-partial-classes-whole", whole),
        Distractor(rational_text(inverted), "partial-class-fraction-inverted", inverted),
        Distractor(rational_text(no_ends), "drops-partial-classes", no_ends),
        Distractor(rational_text(own_denominator), "divides-by-classes-in-range-not-total", own_denominator),
    ]
    # (c) the conditional probability
    tail = tail_count / n
    below_hi = _overlap_probability(freqs, start, width, Fraction(start), hi, n)
    wrong_event = prob_b / below_hi
    complement = 1 - prob_c
    c_cands = [
        Distractor(b_text, "conditional-as-joint", prob_b),
        Distractor(rational_text(wrong_event), "conditioned-on-the-wrong-event", wrong_event),
        Distractor(rational_text(complement), "gave-complement-for-event", complement),
        Distractor(rational_text(tail), "reports-the-conditioning-event", tail),
    ]
    a_d = pick_distractors(a_text, a_cands, 3, tolerance=GRADER_TOLERANCE)
    b_d = pick_distractors(b_text, b_cands, 3)
    c_d = pick_distractors(c_text, c_cands, 3)
    if not (len(a_d) == len(b_d) == len(c_d) == 3):
        raise _Redraw("too few distinct distractors")
    for d in a_d + b_d + c_d:
        check(isinstance(d, Distractor), "a distractor lost its tag")

    def frac(x: Fraction) -> str:
        return f"{x.numerator}/{x.denominator}"

    mid_known = sum(freqs[i + 1 : j]) - freqs[missing]
    ft_b = f"({freqs[i]}*(1 - {frac(u)}) + part_a + {mid_known} + {freqs[j]}*{frac(v)})/{n}"
    rest_known = sum(freqs[i + 1 :]) - freqs[missing]
    ft_c = f"part_b*{n}/({freqs[i]}*(1 - {frac(u)}) + part_a + {rest_known})"

    parts = [
        dict(
            prompt=f"Find the frequency of the class [{miss_lo}, {miss_hi}).",
            answer_kind="number",
            answer=freqs[missing],
            answer_text=a_text,
            answer_latex=latex_answer(a_text),
            marks=1,
            sensible_range=(0, None),
            working=[f"Frequency = {n} - {known_sum} = {freqs[missing]}."],
            distractors=a_d,
        ),
        dict(
            prompt=(
                f"Estimate the probability that {subject} is at least {lo_i} {unit} and less than {hi_i} {unit}. "
                "Give your answer as a fraction in simplest form."
            ),
            answer_kind="fraction",
            answer=exact_value(prob_b),
            answer_text=b_text,
            answer_latex=latex_answer(b_text),
            marks=3,
            sensible_range=(0, 1),
            follow_through=ft_b,
            working=[
                f"From {lo_i} to {start + (i + 1) * width} is {frac(1 - u)} of the class with frequency {freqs[i]}.",
                f"From {start + j * width} to {hi_i} is {frac(v)} of the class with frequency {freqs[j]}.",
                f"Count = {freqs[i]} x {frac(1 - u)} + (whole classes in between) + {freqs[j]} x {frac(v)} "
                f"= {rational_text(prob_b * n)}, so the probability is {_over(rational_text(prob_b * n), n)} = {b_text}.",
            ],
            distractors=b_d,
        ),
        dict(
            prompt=(
                f"Given that {subject} is at least {lo_i} {unit}, estimate the probability that it is less than "
                f"{hi_i} {unit}. Give your answer as a fraction in simplest form."
            ),
            answer_kind="fraction",
            answer=exact_value(prob_c),
            answer_text=c_text,
            answer_latex=latex_answer(c_text),
            marks=2,
            sensible_range=(0, 1),
            follow_through=ft_c,
            working=[
                f"P(at least {lo_i}) = {_over(rational_text(tail_count), n)}.",
                f"P(less than {hi_i} | at least {lo_i}) = P({lo_i} to {hi_i}) / P(at least {lo_i}) "
                f"= {_over(rational_text(prob_b * n), rational_text(tail_count))} = {c_text}.",
            ],
            distractors=c_d,
        ),
    ]
    return dict(text=stem, parts=parts)


def _over(top, bottom) -> str:
    """top/bottom, bracketing either side that is itself a fraction so the line never reads 394/5/94."""
    top, bottom = str(top), str(bottom)
    return f"{'(' + top + ')' if '/' in top else top}/{'(' + bottom + ')' if '/' in bottom else bottom}"


@generator("exd_x_r2l_histogram_chain", level=12, presets={"exam": dict(tier="exam")})
def exd_x_r2l_histogram_chain(rng: random.Random, tier: str = "exam") -> dict:
    """A histogram with one frequency missing: find it, then an interpolated interval probability, then a conditional."""
    check(tier == "exam", "exd_x_r2l_histogram_chain: only the exam tier exists")
    for _ in attempts(50, "a draw with enough distinct distractors"):
        try:
            return _build(rng)
        except _Redraw:
            continue
    raise AssertionError("unreachable")
