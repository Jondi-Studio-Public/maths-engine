"""Senior polynomial-solve and cubic-factorisation generators.

Skill 1 (ACMSM091, senior-solve-simple-polynomial-equations): always a monic
quadratic in z with a non-real complex-conjugate root pair; the learner
states the root with positive imaginary part, a typed complex value ``p +
mi``. Distinct from ACMSM090 (senior-consider-conjugate-roots-for-...),
which only ever asks for a real-valued sum/product/coefficient built from a
stated root pair and never a typed complex answer -- this skill is the one
that requires the learner to type a complex value.

Skill 2 (ACMMM018, senior-factorise-cubic-polynomials-in-cases-where-a-linear):
canonical answer is the factored ``(x - p)(x^2 + qx + r)`` (monic tiers) or
``a(x - p)(x^2 + qx + r)`` (hard, non-monic) form -- never a list of roots
and never the solved equation. Every candidate offered (answer and every
distractor) is a bracketed factored expression: `mathgen.grading.grade_expression`
requires a bracket in the learner's input whenever the canonical answer
itself contains one, which is also the mechanism that rejects a bracket-free
roots-only answer for this skill.

Neither generator calls `sympy.solve`/`sympy.simplify` in the `make()` hot
path; each independent check is a genuinely different computational route
from construction (native `complex`/`cmath`-style residual arithmetic for
skill 1, `sympy.div` on the *displayed* expanded cubic for skill 2).
"""

from __future__ import annotations

from math import isqrt

from sympy import Poly, Symbol, expand

from ..core import check, check_distinct, fmt_signed, generator, join_terms, pick_distractors

x = Symbol("x")
i_sym = Symbol("i")  # a FORMAL symbol, never sympy.I:
# grade_expression parses learner input with sympy.parse_expr, and lowercase
# "i" parses as Symbol('i'), never the true imaginary unit. Every canonical
# answer_text/distractor here is built and graded consistently against that
# same formal symbol.

# --- Skill 1: simple_polynomial_equation (ACMSM091) -------------------------

# k = m^2 for m in 1..20 gives 20 distinct perfect-square k values.
M_CHOICES_EASY = list(range(1, 21))
M_CHOICES = [1, 2, 3, 4, 5, 6]
P_CHOICES = [p for p in range(-9, 10) if p != 0]


def _complex_text(p: int, m: int) -> str:
    """``p + mi`` the way a student writes it -- never ``p + -mi``."""
    return join_terms([(p, ""), (m, "i")])


@generator(
    "simple_polynomial_equation",
    level=12,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def simple_polynomial_equation(rng, tier="medium"):
    """Solve a monic quadratic in z with non-real conjugate roots; state the
    root with positive imaginary part."""

    if tier == "easy":
        m = rng.choice(M_CHOICES_EASY)
        k = m * m

        # Independent check: square the claimed root with native `complex`
        # arithmetic and confirm it reproduces -k, a route with no
        # dependence on how the root was constructed from m.
        root = complex(0, m)
        check(abs(root**2 + k) < 1e-9, f"root check failed for k={k}, m={m}")

        answer_text = fmt_signed(m, "i")
        answer_sym = m * i_sym

        m_plus = m + 1
        m_minus = m - 1 if m > 1 else m + 2
        conjugate_sym = -m * i_sym
        plus_sym = m_plus * i_sym
        minus_sym = m_minus * i_sym
        check_distinct(answer_sym, conjugate_sym, plus_sym, minus_sym)

        candidates = [fmt_signed(-m, "i"), fmt_signed(m_plus, "i"), fmt_signed(m_minus, "i")]
        text = f"Solve z² + {k} = 0 for z. State the root with positive imaginary part, in the form a + bi."
        latex = rf"z^2 + {k} = 0 \ \text{{State the root with positive imaginary part, in the form }} a + bi"
        working = [
            f"z² = -{k}",
            f"z = ±√(-{k}) = ±{m}i",
            f"The root with positive imaginary part is {answer_text}.",
        ]

    elif tier in ("medium", "hard"):
        p = rng.choice(P_CHOICES)
        m = rng.choice(M_CHOICES)
        k = m * m

        # Independent check: plug the claimed root into the EXPANDED
        # equation z^2 - 2pz + (p^2+m^2) = 0 with native `complex`
        # arithmetic -- a genuinely different route from constructing the
        # root directly as complex(p, m) from the chosen p, m.
        root = complex(p, m)
        residual = root**2 - 2 * p * root + (p**2 + k)
        check(abs(residual) < 1e-9, f"residual check failed for p={p}, m={m}")

        answer_text = _complex_text(p, m)
        answer_sym = p + m * i_sym

        m_plus = m + 1
        conjugate_sym = p - m * i_sym
        sign_error_sym = -p + m * i_sym
        missing_shift_sym = m * i_sym  # p != 0 here, so this is a genuine wrong answer
        offbyone_sym = p + m_plus * i_sym
        check_distinct(answer_sym, conjugate_sym, sign_error_sym, missing_shift_sym, offbyone_sym)

        candidates = [
            _complex_text(p, -m),
            _complex_text(-p, m),
            fmt_signed(m, "i"),
            _complex_text(p, m_plus),
        ]

        lin = join_terms([(1, "z"), (-p, "")])  # "z - p" with the sign always correct

        if tier == "medium":
            text = f"Solve ({lin})² + {k} = 0 for z. State the root with positive imaginary part, in the form a + bi."
            latex = rf"({lin})^2 + {k} = 0 \ \text{{State the root with positive imaginary part, in the form }} a + bi"
            working = [
                f"({lin})² = -{k}",
                f"{lin} = ±{m}i",
                f"z = {p} ± {m}i",
                f"The root with positive imaginary part is {answer_text}.",
            ]
        else:  # hard: same p, m pair, equation already expanded
            expanded = join_terms([(1, "z²"), (-2 * p, "z"), (p**2 + k, "")])
            text = f"Solve {expanded} = 0 for z. State the root with positive imaginary part, in the form a + bi."
            latex = rf"{expanded} = 0 \ \text{{State the root with positive imaginary part, in the form }} a + bi"
            working = [
                f"Complete the square: {expanded} = ({lin})² + {k}",
                f"({lin})² = -{k}",
                f"z = {p} ± {m}i",
                f"The root with positive imaginary part is {answer_text}.",
            ]
    else:
        raise ValueError(f"Unknown tier: {tier}")

    return dict(
        text=text,
        latex=latex,
        answer=answer_text,
        answer_text=answer_text,
        working=working,
        distractors=pick_distractors(answer_text, candidates, n=4),
    )


# --- Skill 2: factorise_cubic_linear_factor (ACMMM018) ----------------------

P_EASY = [-2, -1, 1, 2]
P_MEDIUM = [p for p in range(-6, 7) if p != 0]
P_HARD = [-4, -3, -2, -1, 1, 2, 3, 4]
A_HARD = [2, 3]


def _splits_over_integers(q, r):
    """True if x^2 + qx + r has integer roots (non-negative perfect-square discriminant)."""
    d = q * q - 4 * r
    return d >= 0 and isqrt(d) ** 2 == d


@generator(
    "factorise_cubic_linear_factor",
    level=12,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def factorise_cubic_linear_factor(rng, tier="medium"):
    """Factorise a cubic as (linear)(quadratic) -- never roots, never the
    solved equation. Non-monic (hard) tier requires the scalar `a` too."""

    if tier == "easy":
        p = rng.choice(P_EASY)
        q = rng.randint(-5, 5)
        r = rng.randint(-5, 5)
        while q == r:
            r = rng.randint(-5, 5)
        a = 1
    elif tier == "medium":
        p = rng.choice(P_MEDIUM)
        q = rng.randint(-8, 8)
        r = rng.randint(-8, 8)
        while q == r:
            r = rng.randint(-8, 8)
        a = 1
    elif tier == "hard":
        p = rng.choice(P_HARD)
        a = rng.choice(A_HARD)
        q = rng.randint(-6, 6)
        r = rng.randint(-6, 6)
        # "Completely" means the quadratic factor must not split over the integers either (discriminant a
        # perfect square), or the stated product is not fully factorised.
        while q == r or _splits_over_integers(q, r):
            q = rng.randint(-6, 6)
            r = rng.randint(-6, 6)
    else:
        raise ValueError(f"Unknown tier: {tier}")

    factored_expr = a * (x - p) * (x**2 + q * x + r)
    cubic = expand(factored_expr)
    poly = Poly(cubic, x)
    coeffs = poly.all_coeffs()
    check(len(coeffs) == 4, f"expected a degree-3 polynomial, got coeffs={coeffs}")
    A3, B2, C1, D0 = coeffs
    check(A3 == a, f"leading coefficient {A3} != a={a}")

    # Independent check: divide the DISPLAYED expanded cubic back apart by
    # (x - p) with sympy.div -- a genuinely different route from the
    # multiply-and-expand construction above.
    quotient, remainder = Poly(cubic, x).div(Poly(x - p, x))
    check(remainder.is_zero, f"remainder {remainder} != 0 dividing by (x - {p})")
    expected_quotient = expand(a * x**2 + a * q * x + a * r)
    check(
        expand(quotient.as_expr() - expected_quotient) == 0,
        f"quotient {quotient.as_expr()} != expected {expected_quotient}",
    )

    # "x^2"/"x^3" (caret), never the unicode superscript "x²"/"x³", in every
    # GRADED field (answer_text, distractors) and, for consistency, the
    # question text too. Confirmed directly against the real grader: a
    # canonical built with a literal "²" character cannot be parsed back by
    # `mathgen.grading._parse_expression` at all -- `parse_expr` raises on the
    # character and the spoken-transcript fallback it falls through to
    # silently drops both "²" and every "-" not immediately adjacent to a
    # digit, corrupting the expression -- so a learner typing the canonical
    # back verbatim would be marked wrong. Caret notation has no such defect
    # (`_parse_expression` replaces "^" with "**" before parsing) and is
    # exactly what the "caret-typed input" required test is really checking.
    cubic_text = join_terms([(A3, "x^3"), (B2, "x^2"), (C1, "x"), (D0, "")])
    cubic_latex = join_terms([(A3, "x^3"), (B2, "x^2"), (C1, "x"), (D0, "")])

    linear_text = join_terms([(1, "x"), (-p, "")])
    quad_disp = join_terms([(1, "x^2"), (q, "x"), (r, "")])

    if a == 1:
        answer_text = f"({linear_text})({quad_disp})"
    else:
        answer_text = f"{a}({linear_text})({quad_disp})"

    answer_expr = a * (x - p) * (x**2 + q * x + r)
    # `question.answer` (the internal field `_expected_expression` falls back
    # to) must round-trip through plain `sympify`, which -- unlike
    # `parse_expr` -- does NOT support implicit multiplication ("4x" fails);
    # sympy's own `str()` always uses explicit "*"/"**", so it is the safe
    # way to build this field.
    answer_internal = str(answer_expr)

    # Distractors, all bracketed (Boundary 1: never a roots-only string).
    linear_wrong_sign = join_terms([(1, "x"), (p, "")])  # (x + p) instead of (x - p)
    quad_swapped = join_terms([(1, "x^2"), (r, "x"), (q, "")])
    quad_wrong_const = join_terms([(1, "x^2"), (q, "x"), (r + 1, "")])

    dist_sign_expr = a * (x + p) * (x**2 + q * x + r)
    dist_swap_expr = a * (x - p) * (x**2 + r * x + q)
    dist_const_expr = a * (x - p) * (x**2 + q * x + (r + 1))

    if a == 1:
        candidates_text = [
            f"({linear_wrong_sign})({quad_disp})",
            f"({linear_text})({quad_swapped})",
            f"({linear_text})({quad_wrong_const})",
        ]
        check_distinct(answer_expr, dist_sign_expr, dist_swap_expr, dist_const_expr)
    else:
        # hard only: the fourth misconception is dropping the scalar `a`
        # entirely, a genuinely different value since a != 1.
        dist_a_omitted_expr = 1 * (x - p) * (x**2 + q * x + r)
        check_distinct(answer_expr, dist_sign_expr, dist_swap_expr, dist_const_expr, dist_a_omitted_expr)
        candidates_text = [
            f"{a}({linear_wrong_sign})({quad_disp})",
            f"{a}({linear_text})({quad_swapped})",
            f"{a}({linear_text})({quad_wrong_const})",
            f"({linear_text})({quad_disp})",
        ]

    text = f"Factorise {cubic_text} completely as a product of a linear factor and a quadratic factor."
    latex = rf"{cubic_latex}"
    working = [
        f"By the rational root theorem, x = {p} is a root: substituting confirms the cubic is zero there.",
        f"Dividing {cubic_text} by ({linear_text}) gives the quadratic factor {quad_disp}.",
        f"Answer: {answer_text}",
    ]

    return dict(
        text=text,
        latex=latex,
        answer=answer_internal,
        answer_text=answer_text,
        working=working,
        distractors=pick_distractors(answer_text, candidates_text, n=4),
        answer_form="factored",
    )
