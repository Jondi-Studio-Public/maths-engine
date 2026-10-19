"""Exam chain for integration with partial fractions: an improper rational function, split and integrated.

The integrand has a quadratic numerator over a quadratic denominator, so it is improper and a whole-number part
must come out before the fractions are split. Two shapes, both written ``k + A/(x - p) + ...``:

* two different linear factors:  ``f(x) = k + A/(x - p) + B/(x - q)``
* a repeated linear factor:      ``f(x) = k + A/(x - p) + B/(x - p)^2``

and the question is set the way a senior exam sets it:

(a) the whole-number part k                                         number, 1 mark
(b) the constant A                                                  number, 2 marks
(c) the constant B, using the learner's k (and A) to match terms     number, 2 marks, follow-through
(d) the exact value of the definite integral from lo to hi          expression, 3 marks, follow-through (a)-(c)

## Construction: choose k, A, B first

The integrand is built from the chosen ``k, A, B, p, q`` and expanded, so the printed fraction is exact and the
decomposition is unique. The limits lie to the right of every pole so that each logarithm has a positive
argument. (b) is the cover-up value ``N(p) / (p - q)`` for two factors; for a repeated factor it is read from the
coefficient of x. (c) matches the x terms (two factors) or the constant terms (repeated factor), which is why
it follows through from the earlier parts. The integral is
``k (hi - lo) + A ln((hi - p)/(lo - p)) + B ln((hi - q)/(lo - q))``, or with ``B (1/(lo - p) - 1/(hi - p))`` for the
repeated factor. The key is re-derived by integrating the printed fraction with SymPy.

## Distractors

* (a): the x-coefficient or the ratio of constant terms in place of the leading ratio.
* (b), (c): the residues swapped, the sign of the cover-up flipped, the cover-up value left undivided.
* (d): the whole-number part dropped, residues swapped, logarithms taken at the upper limit only; for a repeated
  factor the squared term integrated as a logarithm, or its sign wrong.
"""

from __future__ import annotations

from fractions import Fraction

from mpmath import mp
from sympy import Integer, Rational, log

from mathgen.answers import Distractor, exact_value
from mathgen.core import (
    check,
    generator,
    join_terms,
    latex_answer,
    paren_negative,
    phrase,
    pick_distractors,
    rational_text,
)

_STEMS = (
    "Let f(x) = ({num})/({den}). The function can be written as {form}, where k, A and B are constants.",
    "The rational function f(x) = ({num})/({den}) is to be integrated. It can be written as {form}, "
    "where k, A and B are constants.",
    "Consider f(x) = ({num})/({den}), which can be expressed as {form}, where k, A and B are constants.",
)
_TAILS = (
    "Hence find the exact value of the integral of f(x) from x = {lo} to x = {hi}.",
    "Hence evaluate the integral of f(x) from x = {lo} to x = {hi}, giving an exact answer.",
)


def _typed(expr) -> str:
    return str(expr).replace("**", "^").replace("log(", "ln(")


def _linear(r: int) -> str:
    return "x" if r == 0 else f"(x {'-' if r > 0 else '+'} {abs(r)})"


def _number_part(prompt, value, marks, working, wrong, follow=None):
    """A part whose answer is the exact rational `value`; `wrong` lists (value, misconception id)."""
    value = Fraction(value)
    text = rational_text(Rational(value.numerator, value.denominator))
    cands = [
        Distractor(
            rational_text(Rational(Fraction(v).numerator, Fraction(v).denominator)), tag, exact_value(Fraction(v))
        )
        for v, tag in wrong
        if v is not None
    ]
    part = dict(
        prompt=prompt,
        answer_kind="number",
        answer=int(value) if value.denominator == 1 else float(value),
        answer_text=text,
        answer_latex=latex_answer(text),
        marks=marks,
        working=working,
        distractors=pick_distractors(text, cands),
    )
    if follow:
        part["follow_through"] = follow
    return part


@generator("exd_x_r2c_partial_fractions", level=17, presets={"exam": {"tier": "exam"}})
def exd_x_r2c_partial_fractions(rng, tier: str = "exam"):
    """k + A/(x - p) + B/(x - q) or k + A/(x - p) + B/(x - p)^2: decompose, then integrate between limits."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    repeated = rng.random() < 0.4
    k = rng.choice([-3, -2, -1, 1, 2, 3, 4])
    a_c = rng.choice([-4, -3, -2, -1, 1, 2, 3, 4])
    b_c = rng.choice([v for v in (-5, -4, -3, -2, -1, 1, 2, 3, 4, 5) if abs(v) != abs(a_c)])  # A, B distinct in size
    p = rng.choice([-3, -2, -1, 1, 2, 3]) if repeated else rng.randint(-3, 3)
    q = p if repeated else rng.choice([v for v in range(-3, 4) if v != p])
    lo = max(p, q) + rng.randint(1, 2)
    hi = lo + rng.randint(1, 4)

    # Numerator coefficients n2 x^2 + n1 x + n0 of the integrand (n2 = k), from the decomposition.
    if repeated:
        n1, n0 = -2 * k * p + a_c, k * p * p - a_c * p + b_c
        d1, d0 = -2 * p, p * p
        # Independent check: N(p) = B, and A is the x coefficient shifted back by 2kp.
        check(k * p * p + n1 * p + n0 == b_c and n1 + 2 * k * p == a_c, "the repeated-factor terms disagree")
    else:
        n1, n0 = -k * (p + q) + a_c + b_c, k * p * q - a_c * q - b_c * p
        d1, d0 = -(p + q), p * q

        def n_at(r):
            return k * r * r + n1 * r + n0

        # Independent check: the cover-up values recover A and B.
        check(Fraction(n_at(p), p - q) == a_c and Fraction(n_at(q), q - p) == b_c, "cover-up disagrees")

    num_text = join_terms([(k, "x^2"), (n1, "x"), (n0, "")])
    den_text = f"{_linear(p)}^2" if repeated else f"{_linear(p)}{_linear(q)}"
    second = f"{_linear(p)}^2" if repeated else _linear(q)
    form = f"k + A/{_linear(p)} + B/{second}"
    stem = phrase(rng, _STEMS, num=num_text, den=den_text, form=form)
    tail = phrase(rng, _TAILS, lo=lo, hi=hi)

    # ---- the exact integral and its wrong versions --------------------------------------------------------
    def total(kk, aa, bb, *, swap=False, upper_only=False, log_square=False, flip=False):
        r1 = Rational(hi - p, lo - p)
        r2 = Rational(hi - q, lo - q)
        first, second = (bb, aa) if swap else (aa, bb)
        if upper_only:
            return (
                kk * (hi - lo) + first * log(Integer(hi - p)) + (second * log(Integer(hi - q)) if not repeated else 0)
            )
        if repeated:
            tail_term = bb * (Rational(1, lo - p) - Rational(1, hi - p))
            if log_square:
                tail_term = bb * log(r1)
            if flip:
                tail_term = -tail_term
            return kk * (hi - lo) + aa * log(r1) + tail_term
        return kk * (hi - lo) + first * log(r1) + second * log(r2)

    value = total(k, a_c, b_c)
    with mp.workdps(30):  # local, so make() leaves mpmath's global precision as it found it
        quadrature = mp.quad(lambda t: (k * t * t + n1 * t + n0) / ((t - p) * (t - q)), [lo, hi])
        check(
            abs(quadrature - mp.mpf(str(value.evalf(30)))) < mp.mpf("1e-20"), "the integral disagrees with the fraction"
        )
    value_text = _typed(value)

    # ---- parts ----------------------------------------------------------------------------------------------
    part_a = _number_part(
        "Find the value of k.",
        k,
        1,
        [f"The numerator and denominator have the same degree, so k = {k} (the ratio of the x² coefficients)."],
        [
            (Fraction(n0, d0) if d0 else None, "quotient-from-wrong-coefficients"),
            (Fraction(n1, d1) if d1 else None, "quotient-from-wrong-coefficients"),
            (n1, "quotient-from-wrong-coefficients"),
            (-k, "sign-error-moving-term"),
        ],
    )
    if repeated:
        a_follow = f"{n1} + 2*part_a*({p})"
        part_b = _number_part(
            "Find the value of A.",
            a_c,
            2,
            [f"Matching the x terms of the numerator: {n1} = -2·{paren_negative(k)}·({p}) + A, so A = {a_c}."],
            [
                (b_c, "swapped-partial-fraction-coefficients"),
                (n1, "read-coefficient-without-expanding"),
                (-a_c, "sign-error-moving-term"),
                (-b_c, "swapped-partial-fraction-coefficients"),
            ],
            follow=a_follow,
        )
        part_c = _number_part(
            "Find the value of B.",
            b_c,
            2,
            [f"Matching the constant terms: B = {n0} - {paren_negative(k)}·({p})² + A·({p}) = {b_c}."],
            [
                (a_c, "swapped-partial-fraction-coefficients"),
                (n0, "read-coefficient-without-expanding"),
                (n0 - k * p * p - a_c * p, "sign-error-moving-term"),
                (-a_c, "swapped-partial-fraction-coefficients"),
            ],
            follow=f"{n0} - part_a*({p})**2 + part_b*({p})",
        )
        d_follow = f"part_a*({hi} - {lo}) + part_b*log(({hi - p})/({lo - p})) + part_c*(1/({lo - p}) - 1/({hi - p}))"
        d_wrong = [
            ("ignored-improper-quotient", total(0, a_c, b_c)),
            ("repeated-factor-integrated-as-log", total(k, a_c, b_c, log_square=True)),
            ("sign-reversed-result", total(k, a_c, b_c, flip=True)),
            ("upper-limit-only", total(k, a_c, b_c, upper_only=True)),
        ]
    else:
        part_b = _number_part(
            "Find the value of A.",
            a_c,
            2,
            [f"Cover-up: A = N({p}) / ({p} - ({q})) = {a_c}."],
            [
                (b_c, "swapped-partial-fraction-coefficients"),
                (-a_c, "sign-reversed-result"),
                (k * p * p + n1 * p + n0, "forgot-to-divide-by-coefficient"),
                (-b_c, "swapped-partial-fraction-coefficients"),
            ],
        )
        part_c = _number_part(
            "Find the value of B.",
            b_c,
            2,
            [f"Matching the x terms: {n1} = -{paren_negative(k)}·({p} + ({q})) + A + B, so B = {b_c}."],
            [
                (a_c, "swapped-partial-fraction-coefficients"),
                (-b_c, "sign-reversed-result"),
                (n1 - k * (p + q) - a_c, "sign-error-moving-term"),
                (-a_c, "swapped-partial-fraction-coefficients"),
            ],
            follow=f"{n1} + part_a*({p + q}) - part_b",
        )
        d_follow = f"part_a*({hi} - {lo}) + part_b*log(({hi - p})/({lo - p})) + part_c*log(({hi - q})/({lo - q}))"
        d_wrong = [
            ("ignored-improper-quotient", total(0, a_c, b_c)),
            ("swapped-partial-fraction-coefficients", total(k, a_c, b_c, swap=True)),
            ("upper-limit-only", total(k, a_c, b_c, upper_only=True)),
            ("sign-error-moving-term", total(-k, a_c, b_c)),
        ]
    d_cands = [Distractor(_typed(e), tag, _typed(e)) for tag, e in d_wrong if abs(float(e) - float(value)) > 1e-9]
    part_d = dict(
        prompt=tail,
        answer_kind="expression",
        answer=str(value),
        answer_text=value_text,
        answer_latex=latex_answer(value),
        marks=3,
        working=[
            f"f(x) = {k} + ({a_c})/(x - ({p})) + ({b_c})/(…); integrate each term and substitute the limits.",
            f"The integral is {value_text}.",
        ],
        distractors=pick_distractors(value_text, d_cands),
        follow_through=d_follow,
    )
    return dict(text=stem, parts=[part_a, part_b, part_c, part_d])
