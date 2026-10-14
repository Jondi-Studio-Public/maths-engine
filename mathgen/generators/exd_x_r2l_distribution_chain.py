"""Exam question: an estimated probability distribution from a frequency table with two unknown frequencies.

The table has five values of a discrete random variable X. Two of the frequencies are unknown (m and k); the stem
gives the total number of observations and the sample mean. Part (a) uses the total to find m + k. Part (b) uses the
mean as a second equation, together with (a), to find m. Part (c) finds k. Part (d) writes down the estimated
distribution, the list of P(X = x) in increasing order of x.

Parts (b) and (c) carry `follow_through`, so a slip in (a) or (b) is not paid for twice. Part (d) is a list, which a
follow-through expression cannot reproduce, so it is marked on its own.

Built backwards: the full table is drawn first, the total and the mean are read off it, and the mean is kept to at
most two decimal places so it can be stated exactly. The unknowns are re-derived independently by solving the
two-equation system with SymPy.
"""

from __future__ import annotations

import random
from fractions import Fraction

import sympy

from ..answers import Distractor, exact_value
from ..core import GRADER_TOLERANCE, attempts, check, generator, latex_answer, pick_distractors, rational_text

# (what X counts, what is surveyed with {n})
_CONTEXTS = (
    ("the number of children in a household", "{n} households were surveyed"),
    ("the number of passengers in a car", "{n} cars were observed at a toll gate"),
    ("the number of faults found in a vehicle inspection", "{n} vehicle inspections were reviewed"),
    ("the number of goals scored by a team in a match", "{n} matches were recorded"),
    ("the number of customers waiting at a counter", "{n} counter checks were made"),
)
_VALUES = 5


def _decimal_text(value: Fraction) -> str:
    check((value * 100).denominator == 1, "mean is not a terminating decimal of at most two places")
    text = f"{float(value):.2f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


class _Redraw(Exception):
    """A draw that cannot be given enough distinct distractors; the generator draws again."""


def _build(rng: random.Random) -> dict:
    what, survey = rng.choice(_CONTEXTS)
    start = rng.choice((0, 1))
    xs = list(range(start, start + _VALUES))
    for _ in attempts(5000, "a table with a total and mean that state exactly"):
        i, j = sorted(rng.sample(range(_VALUES), 2))
        freqs = [rng.randint(2, 20) for _ in range(_VALUES)]
        n = sum(freqs)
        mean = Fraction(sum(x * f for x, f in zip(xs, freqs)), n)
        if n > 90 or (mean * 100).denominator != 1 or freqs[i] == freqs[j]:
            continue
        break
    m, k = freqs[i], freqs[j]
    known = [t for t in range(_VALUES) if t not in (i, j)]
    known_sum = sum(freqs[t] for t in known)
    known_weighted = sum(xs[t] * freqs[t] for t in known)
    s_total = n - known_sum  # m + k
    r = n * mean - known_weighted  # xi*m + xj*k

    # Independent route: solve the two equations.
    sym_m, sym_k = sympy.symbols("m k")
    solved = sympy.solve(
        [
            sympy.Eq(sym_m + sym_k, s_total),
            sympy.Eq(xs[i] * sym_m + xs[j] * sym_k, sympy.Rational(r.numerator, r.denominator)),
        ],
        [sym_m, sym_k],
        dict=True,
    )[0]
    check(solved[sym_m] == m and solved[sym_k] == k, "the two equations do not give the drawn frequencies")
    probs = [Fraction(f, n) for f in freqs]
    check(sum(probs) == 1 and sum(x * pr for x, pr in zip(xs, probs)) == mean, "distribution does not match the mean")

    shown = ", ".join(f"x = {x}: {'m' if t == i else 'k' if t == j else freqs[t]}" for t, x in enumerate(xs))
    stem = (
        f"A random variable X is {what}. In a study, {survey.format(n=n)}, giving the frequency table {shown}. "
        f"The frequencies m and k are not shown. The mean of the {n} recorded values of X is {_decimal_text(mean)}."
    )
    xs_text = ", ".join(str(x) for x in xs)

    a_text, b_text, c_text = str(s_total), str(m), str(k)
    d_text = ", ".join(rational_text(pr) for pr in probs)

    def ints(*cands):
        return [c for c in cands if c.value is not None and c.value > 0 and Fraction(c.value).denominator == 1]

    a_cands = [
        Distractor(str(known_sum), "reports-known-sum-not-remainder", known_sum),
        Distractor(str(n), "reports-total-as-missing-value", n),
        Distractor(str(s_total + freqs[known[0]]), "left-a-class-out-of-the-sum", s_total + freqs[known[0]]),
        Distractor(str(s_total - 1), None, s_total - 1),
    ]
    ignore_known = Fraction(xs[j] * s_total - n * mean, xs[j] - xs[i])
    wrong_difference = Fraction(xs[j] * s_total - r, xs[j]) if xs[j] else None
    b_cands = ints(
        Distractor(str(k), "swapped-the-two-unknowns", k),
        *(
            [Distractor(str(int(ignore_known)), "ignores-known-values-in-mean-equation", ignore_known)]
            if ignore_known.denominator == 1 and ignore_known > 0
            else []
        ),
        *(
            [Distractor(str(int(wrong_difference)), "divides-by-wrong-difference", wrong_difference)]
            if wrong_difference is not None and wrong_difference.denominator == 1 and wrong_difference > 0
            else []
        ),
        Distractor(
            str(s_total - m + 1) if s_total - m + 1 != m else str(m + 2),
            None,
            s_total - m + 1 if s_total - m + 1 != m else m + 2,
        ),
        Distractor(str(m + 1), None, m + 1),
        Distractor(str(m + 2 if m > 1 else m + 3), None, m + 2 if m > 1 else m + 3),
    )
    c_cands = ints(
        Distractor(str(m), "swapped-the-two-unknowns", m),
        Distractor(str(s_total), "forgot-to-subtract-first-unknown", s_total),
        Distractor(str(s_total + m), "added-instead-of-subtracted", s_total + m),
        Distractor(str(k + 1), None, k + 1),
        Distractor(str(k + 2), None, k + 2),
    )
    swapped = list(probs)
    swapped[i], swapped[j] = probs[j], probs[i]
    shifted = probs[1:] + probs[:1]
    wrong_total = [Fraction(f, n + 1) for f in freqs]
    d_cands = [
        Distractor(", ".join(rational_text(pr) for pr in swapped), "swapped-the-two-unknowns", list(swapped)),
        Distractor(
            ", ".join(rational_text(pr) for pr in reversed(probs)),
            "distribution-listed-in-reverse",
            list(reversed(probs)),
        ),
        Distractor(", ".join(rational_text(pr) for pr in shifted), "distribution-shifted-by-one-value", list(shifted)),
        Distractor(", ".join(rational_text(pr) for pr in wrong_total), "wrong-denominator-for-proportion", wrong_total),
    ]
    a_d = pick_distractors(a_text, a_cands, 3, tolerance=GRADER_TOLERANCE)
    b_d = pick_distractors(b_text, b_cands, 3, tolerance=GRADER_TOLERANCE)
    c_d = pick_distractors(c_text, c_cands, 3, tolerance=GRADER_TOLERANCE)
    d_d = pick_distractors(d_text, d_cands, 3, unordered=False)
    if not (len(a_d) == len(b_d) == len(c_d) == len(d_d) == 3):
        raise _Redraw("too few distinct distractors")

    ft_b = (
        f"({xs[j]}*part_a - {r.numerator}/{r.denominator})/{xs[j] - xs[i]}"
        if r.denominator != 1
        else f"({xs[j]}*part_a - {int(r)})/{xs[j] - xs[i]}"
    )
    # Each frequency over the total; the two unknowns are the learner's own (b) and (c), so a slip there carries.
    ft_d = (
        "("
        + ", ".join(f"part_b/{n}" if t == i else f"part_c/{n}" if t == j else f"{freqs[t]}/{n}" for t in range(_VALUES))
        + ")"
    )
    parts = [
        dict(
            prompt="Use the total number of observations to find m + k.",
            answer_kind="number",
            answer=s_total,
            answer_text=a_text,
            answer_latex=latex_answer(a_text),
            marks=1,
            sensible_range=(0, None),
            working=[f"m + k = {n} - {known_sum} = {s_total}."],
            distractors=a_d,
        ),
        dict(
            prompt="The mean gives a second equation in m and k. Use it, with your answer to (a), to find m.",
            answer_kind="number",
            answer=m,
            answer_text=b_text,
            answer_latex=latex_answer(b_text),
            marks=2,
            sensible_range=(0, None),
            follow_through=ft_b,
            working=[
                f"Sum of x f = {n} x {_decimal_text(mean)} = {int(n * mean)}, so {xs[i]}m + {xs[j]}k = {int(r)}.",
                f"With k = {s_total} - m: {xs[i]}m + {xs[j]}({s_total} - m) = {int(r)}, so m = {m}.",
            ],
            distractors=b_d,
        ),
        dict(
            prompt="Hence find k.",
            answer_kind="number",
            answer=k,
            answer_text=c_text,
            answer_latex=latex_answer(c_text),
            marks=1,
            sensible_range=(0, None),
            follow_through="part_a - part_b",
            working=[f"k = {s_total} - {m} = {k}."],
            distractors=c_d,
        ),
        dict(
            prompt=(
                "Estimate the probability distribution of X from the completed table: list P(X = x) for each value "
                f"of x, IN INCREASING ORDER OF x (x = {xs_text}), each as a fraction in lowest terms."
            ),
            answer_kind="ordered_list",
            follow_through=ft_d,
            sensible_range=(0, 1),
            answer=list(probs),
            answer_text=d_text,
            answer_latex=d_text,
            marks=2,
            working=[f"P(X = {x}) = {f}/{n} = {rational_text(pr)}." for x, f, pr in zip(xs, freqs, probs)],
            distractors=d_d,
        ),
    ]
    return dict(text=stem, parts=parts)


@generator("exd_x_r2l_distribution_chain", level=12, presets={"exam": dict(tier="exam")})
def exd_x_r2l_distribution_chain(rng: random.Random, tier: str = "exam") -> dict:
    """Two unknown frequencies from a total and a mean, then the estimated distribution."""
    check(tier == "exam", "exd_x_r2l_distribution_chain: only the exam tier exists")
    for _ in attempts(50, "a draw with enough distinct distractors"):
        try:
            return _build(rng)
        except _Redraw:
            continue
    raise AssertionError("unreachable")
