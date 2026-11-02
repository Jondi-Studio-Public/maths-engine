"""Recipe subject `exact_binomial`: R ~ Bin(n, p) with n in 2..5 and p a fraction (n = 5 as well as 2..4: only 89 distinct conditional answers exist for n up to 4, fewer than the 100 per 300 seeds a recipe must show), exact answers only.

The asks are registered per subject (the closed vocabulary has no discrete property yet): a point probability, an exact standard deviation
(a simplified surd) and a conditional tail probability P(R >= a | R >= b) are all worked out from the subject's
`pmf` and `support` alone. The subject supplies the integers the question asks about and, for the point probability,
the wrong answers its binomial working invites (`point_slips`).
"""

from __future__ import annotations

import math
from fractions import Fraction

import sympy

from ..core import attempts, rational_text
from ..recipes import Subject, ask, subject
from ._r3f_kit import fraction_part
from ._recipe_kit import expression_part

_P = tuple(Fraction(a, b) for b in range(2, 7) for a in range(1, b) if math.gcd(a, b) == 1)

_SETUPS = (
    (
        "Each time a biased die is rolled, a six appears with probability {p}. The die is rolled {w} times",
        "the number of sixes",
    ),
    (
        "A factory packs items independently, and each item is faulty with probability {p}. {N} items are packed",
        "the number of faulty items",
    ),
    (
        "Each seed in a packet germinates independently with probability {p}. {N} seeds are planted",
        "the number of seeds that germinate",
    ),
    (
        "A basketball player scores each free throw independently with probability {p}. She takes {w} free throws",
        "the number of throws she scores",
    ),
)
_NUMBER_WORD = {2: "Two", 3: "Three", 4: "Four", 5: "Five"}


def _tail(s, j) -> Fraction:
    return sum((s.pmf(i) for i in s.support if i >= j), Fraction(0))


@subject("exact_binomial")
class ExactBinomial(Subject):
    families = frozenset({"discrete_pmf"})
    rv = "R"

    def __init__(self, n, p, setup, k, a, b):
        self.n, self.p, self.setup, self.point_k, self.cond_a, self.cond_b = n, p, setup, k, a, b
        self.support = tuple(range(n + 1))

    @classmethod
    def draw(cls, rng, technology_active):
        if technology_active:
            raise ValueError("exact_binomial asks exact answers only; it has no technology (decimal) form")
        for _ in attempts(500, "a binomial with a surd standard deviation and clean prompts"):
            n, p = rng.choice((2, 3, 4, 5)), rng.choice(_P)
            variance = n * p * (1 - p)
            root = sympy.sqrt(sympy.Rational(variance.numerator, variance.denominator))
            if root.is_Rational:
                continue
            k = rng.randint(1, n - 1)
            b = rng.randint(1, n - 1)
            a = rng.randint(b + 1, n)
            s = cls(n, p, rng.choice(_SETUPS), k, a, b)
            shown = {rational_text(p), rational_text(1 - p)}
            answers = {rational_text(s.pmf(k)), rational_text(_tail(s, a) / _tail(s, b))}
            if answers & shown:
                continue
            return s
        raise AssertionError("unreachable")

    def pmf(self, j) -> Fraction:
        return math.comb(self.n, j) * self.p**j * (1 - self.p) ** (self.n - j)

    def stem(self) -> str:
        lead, count = self.setup
        lead = lead.format(p=rational_text(self.p), w=_NUMBER_WORD[self.n].lower(), N=_NUMBER_WORD[self.n])
        return f"{lead}. Let R be {count}, so R is binomial with n = {self.n} and p = {rational_text(self.p)}."

    # -- the wrong answers a binomial point probability invites ---------------------------------------------

    def point_slips(self, k):
        n, p, q = self.n, self.p, 1 - self.p
        return [
            (p**k * q ** (n - k), "ebin-arrangements-forgotten"),
            (math.comb(n, k) * p ** (n - k) * q**k, "ebin-p-and-q-swapped"),
            (_tail(self, k), "exactly-read-as-at-least"),
            (math.comb(n, k) * p**k, "ebin-failures-ignored"),
        ]


# -- generic asks ---------------------------------------------------------------------------------------------


@ask("exact_binomial", "point_probability", default_marks=1)
def point_probability(s, rng, ctx):
    k = s.point_k
    value = s.pmf(k)
    return fraction_part(
        f"Find Pr({s.rv} = {k}).",
        value,
        1,
        [f"Pr({s.rv} = {k}) = {rational_text(value)}."],
        s.point_slips(k),
        sensible_range=(0, 1),
    )


@ask("exact_binomial", "standard_deviation", default_marks=2)
def standard_deviation(s, rng, ctx):
    mean = sum((j * s.pmf(j) for j in s.support), Fraction(0))
    var = sum(((j - mean) ** 2 * s.pmf(j) for j in s.support), Fraction(0))
    root = sympy.sqrt(sympy.Rational(var.numerator, var.denominator))
    text = str(root)
    mean_r = sympy.Rational(mean.numerator, mean.denominator)
    var_r = sympy.Rational(var.numerator, var.denominator)
    slips = [
        (var_r, "dd-sd-gives-the-variance", str(var_r)),
        (mean_r, "dd-sd-gives-the-mean", str(mean_r)),
        (sympy.sqrt(mean_r), "dd-sd-root-of-the-mean", str(sympy.sqrt(mean_r))),
    ]
    return expression_part(
        f"Find the standard deviation of {s.rv}, as an exact value in simplest form.",
        root,
        text,
        2,
        [f"Var({s.rv}) = {rational_text(var)}, so the standard deviation is {text}."],
        slips,
        answer_form="simplified_surd",
    )


@ask("exact_binomial", "conditional_at_least", default_marks=3)
def conditional_at_least(s, rng, ctx):
    a, b = s.cond_a, s.cond_b
    top, bottom = _tail(s, a), _tail(s, b)
    value = top / bottom
    slips = [
        (top, "conditional-ignored"),
        (bottom / top, "conditional-ratio-inverted"),  # above 1, so dropped below as no probability
        (top / (1 - bottom), "r3a-conditional-divides-by-the-complement"),
        (1 - value, "gave-complement-probability"),
        (top / sum((s.pmf(i) for i in s.support if i <= b), Fraction(0)), "r3a-conditional-divides-by-the-wrong-tail"),
    ]
    slips = [(v, tag) for v, tag in slips if v < 1]  # a probability never exceeds 1
    return fraction_part(
        f"Find Pr({s.rv} ≥ {a} | {s.rv} ≥ {b}).",
        value,
        3,
        [
            f"Pr({s.rv} ≥ {a} | {s.rv} ≥ {b}) = Pr({s.rv} ≥ {a}) / Pr({s.rv} ≥ {b}) = {rational_text(top)} / {rational_text(bottom)}.",
            f"= {rational_text(value)}.",
        ],
        slips,
        sensible_range=(0, 1),
    )
