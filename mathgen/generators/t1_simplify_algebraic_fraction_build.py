"""Generates `simplify-algebraic-fractions` questions: simplify an algebraic fraction by cancelling common
factors.

Scope: every draw is confined so the reduced result is a bare polynomial (or monomial) -- never a fraction
that still contains a "/". The trinomial numerator always has exactly the denominator as one of its two linear
factors, so the reduction is exact polynomial division with zero remainder and the quotient is the OTHER
linear factor. This is what makes ``answer_form="expanded"`` work here unmodified: the canonical answer never
contains "(", so grading it against itself is guaranteed to pass the "(" not in stripped check, while the
original unreduced fraction text always contains "(" and is guaranteed to fail it.

Three tiers, genuine reasoning escalation:
  easy   -- difference of squares over one of its own linear factors.
  medium -- monic trinomial over one of its two linear factors.
  hard   -- non-monic trinomial (leading coefficient != 1) over one of its
            two linear factors -- an extra factor-pair search step the monic
            case does not need.
"""

from __future__ import annotations

import math

from sympy import Poly, div, expand, symbols

from ..core import check, check_distinct, check_equivalent, generator, join_terms, pick_distractors

x = symbols("x")


def _linear_text(coef, const) -> str:
    """'{coef}x + {const}' / '{coef}x - {const}' via join_terms -- never a
    literal '+ -', never a dropped non-unit coefficient."""
    return join_terms([(coef, "x"), (const, "")])


def _verify_reduction(numerator_expr, denominator_expr, answer_expr, msg: str):
    """Independent check, second route: polynomial DIVISION (never the
    multiplication used to build numerator_expr). Confirms the remainder is
    zero and the quotient equals the claimed answer."""
    quotient, remainder = div(Poly(numerator_expr, x), Poly(denominator_expr, x))
    check(remainder.is_zero, f"{msg}: nonzero remainder")
    check_equivalent(quotient.as_expr(), answer_expr, f"{msg}: quotient does not match answer")


@generator(
    "t1_simplify_algebraic_fraction",
    level=9,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def t1_simplify_algebraic_fraction(rng, tier: str = "medium"):
    if tier == "easy":
        # Difference of squares over one of its own linear factors:
        # (x^2 - k^2) / (x +/- k) -> x -/+ k (the OTHER root's factor).
        k = rng.randint(2, 15)
        sign = rng.choice([1, -1])

        numerator_expr = expand((x + k) * (x - k))
        denominator_expr = x + sign * k
        answer_expr = x - sign * k
        _verify_reduction(numerator_expr, denominator_expr, answer_expr, "easy: (x^2 - k^2) / (x +/- k)")

        numerator_text = join_terms([(1, "x^2"), (-(k * k), "")])
        denom_text = _linear_text(1, sign * k)
        answer_text = _linear_text(1, -sign * k)

        # sign-error: states the same-sign factor back (the denominator's own
        # factor) -- there are only two linear factors total in a difference
        # of squares, so this is a genuinely different value whenever k != 0
        # (guaranteed, k in [2..15]).
        sign_error_expr = denominator_expr
        sign_error_text = denom_text
        # constant-off-by-one: same sign as the correct answer, wrong constant.
        const_off_expr = x + (-sign * (k + 1))
        const_off_text = _linear_text(1, -sign * (k + 1))
        check_distinct(
            answer_expr, sign_error_expr, const_off_expr, msg="easy: distractor must be a genuinely different value"
        )

        cands = [sign_error_text, const_off_text]
        text = f"Simplify (x^2 - {k * k}) / ({denom_text})."

    elif tier == "medium":
        # Monic trinomial over one of its two linear factors:
        # (x + p)(x + q) / (x + p or x + q) -> the other factor.
        while True:
            p = rng.randint(-9, 9)
            q = rng.randint(-9, 9)
            if p == 0 or q == 0 or p == q:
                continue
            if p == -q:
                # Degenerates to the easy tier's difference-of-squares shape.
                continue
            break

        numerator_expr = expand((x + p) * (x + q))
        denom_is_p = rng.choice([True, False])
        denom_const, answer_const = (p, q) if denom_is_p else (q, p)
        denominator_expr = x + denom_const
        answer_expr = x + answer_const
        _verify_reduction(numerator_expr, denominator_expr, answer_expr, "medium: (x+p)(x+q) / (x+p or x+q)")

        numerator_text = join_terms([(1, "x^2"), (p + q, "x"), (p * q, "")])
        denom_text = _linear_text(1, denom_const)
        answer_text = _linear_text(1, answer_const)

        # copied-denominator: states the denominator's own factor as the
        # answer (a "cancelled the whole fraction to 1" confusion).
        copied_denominator_expr = denominator_expr
        copied_denominator_text = denom_text
        # sign-error-in-constant: flips the sign of the answer's own constant.
        sign_error_expr = x - answer_const
        sign_error_text = _linear_text(1, -answer_const)
        check_distinct(
            answer_expr,
            copied_denominator_expr,
            sign_error_expr,
            msg="medium: distractor must be a genuinely different value",
        )

        cands = [copied_denominator_text, sign_error_text]
        text = f"Simplify ({numerator_text}) / ({denom_text})."

    elif tier == "hard":
        # Non-monic trinomial over one of its two linear factors:
        # (mx + p)(nx + q) / (mx+p or nx+q) -> the other factor.
        while True:
            m = rng.randint(2, 5)
            n = rng.randint(1, 4)
            p = rng.choice([v for v in range(-9, 10) if v != 0])
            q = rng.choice([v for v in range(-9, 10) if v != 0])
            if math.gcd(m, p) != 1 or math.gcd(n, q) != 1:
                continue
            if (m, p) == (n, q):
                continue
            break

        numerator_expr = expand((m * x + p) * (n * x + q))
        # When n == 1, the n-factor's own coefficient is already 1, so the
        # dropped-leading-coefficient misconception would coincide with the
        # answer if the n-factor were chosen as the answer. Bias the
        # denominator choice in that case so the answer's own leading
        # coefficient is always > 1 (guaranteed for m, since m >= 2) --
        # this is what keeps the misconception "always exercisable" rather
        # than exercisable only sometimes.
        if n == 1:
            denom_is_m = False  # denominator = n*x+q = x+q; answer = m*x+p
        else:
            denom_is_m = rng.choice([True, False])
        if denom_is_m:
            denom_coef, denom_const = m, p
            answer_coef, answer_const = n, q
        else:
            denom_coef, denom_const = n, q
            answer_coef, answer_const = m, p
        denominator_expr = denom_coef * x + denom_const
        answer_expr = answer_coef * x + answer_const
        _verify_reduction(numerator_expr, denominator_expr, answer_expr, "hard: (mx+p)(nx+q) / (mx+p or nx+q)")
        check(
            answer_coef > 1,
            "hard: answer's own leading coefficient must be > 1 for dropped-leading-coefficient to be exercisable",
        )

        numerator_text = join_terms([(m * n, "x^2"), (m * q + n * p, "x"), (p * q, "")])
        denom_text = _linear_text(denom_coef, denom_const)
        answer_text = _linear_text(answer_coef, answer_const)

        # dropped-leading-coefficient: correct constant, leading coefficient
        # forced to 1 instead of the true answer_coef.
        dropped_leading_expr = x + answer_const
        dropped_leading_text = _linear_text(1, answer_const)
        # sign-error-in-constant: same as medium.
        sign_error_expr = answer_coef * x - answer_const
        sign_error_text = _linear_text(answer_coef, -answer_const)
        check_distinct(
            answer_expr,
            dropped_leading_expr,
            sign_error_expr,
            msg="hard: distractor must be a genuinely different value",
        )

        cands = [dropped_leading_text, sign_error_text]
        text = f"Simplify ({numerator_text}) / ({denom_text})."

    else:
        raise ValueError(f"unknown tier {tier!r}")

    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer_expr,
        answer_text=answer_text,
        answer_latex=answer_text,
        distractors=pick_distractors(answer_text, cands),
        answer_form="expanded",
    )
