"""Generates `expand_binomial_product_monic` and `quadratic_complex_conjugate_solutions` questions: expanding a
binomial product and solving a quadratic with complex roots.

The quadratics cluster.

Two unrelated skills sharing the domain ``quadratics`` and this module file
for import-list efficiency only -- not a boundary pair, independent content.

- ``expand_binomial_product_monic`` (f10-quadratic-algebra-expansion,
  AC9M9A02) -- scope:
  "Expand a binomial product of two linear binomial factors into a trinomial
  in standard form." The raw name ("Expand binomials and factorise
  monic quadratics") is compound-verb; the scope drops
  factorisation entirely -- ``skills/quadratics/solve-quadratics-factorable.yaml``
  already performs the identical factoring method as its own literal first
  solving step, so a standalone "output the factored form" rung would test
  the same computation a different way, not new content. This generator
  never factorises anything; it only expands two monic linear binomial
  factors into a trinomial.

- ``quadratic_complex_conjugate_solutions``
  (senior-determine-complex-conjugate-solutions-of-real-quadratic-equations,
  ACMSM075) -- no narrowing; the contract is: "Given bounded, fully specified inputs, determine complex
  conjugate solutions of real quadratic equations. Return the resulting
  value, expression, relation, or classification."

  Boundary against the shipping
  ``senior-consider-conjugate-roots-for-polynomials-with-real-coefficients``
  (domain ``functions-sequences``, topic ``conjugate_roots_real_coefficients``
  in ``p5c_one_to_one_conjugate_build.py``): that skill's own description
  states "All answers are real-valued integers; learners never type or
  manipulate complex numbers directly" -- it only asks for the sum or
  product of the (unstated) complex roots via Vieta's formulas, both of
  which are always real. This skill is the complement: the easy and medium
  tiers ask the learner to state a complex-valued solution in ``p + qi``
  form.

  The hard tier states the quadratic ``x^2 + bx + c = 0`` with ``b`` shown
  as a number and ``c`` withheld, gives one complex root ``p - qi``, and
  asks for the value of ``c``, via the conjugate-root theorem plus Vieta's
  product relation for a monic quadratic (the other root is ``p + qi``, so
  ``c = (p - qi)(p + qi) = p^2 + q^2``). This is the REVERSE direction from
  the sibling skill above: that skill always gives BOTH coefficients and
  asks for the sum or product of the (unstated) roots -- coefficients to
  sum/product. Here one coefficient is withheld and a root is given instead
  -- root to coefficient. That direction never appears in the sibling
  skill's own content, so despite both skills touching Vieta's formulas for
  a conjugate pair, they do not collide; an earlier draft of this generator
  avoided asking for ``b`` here on the mistaken belief that any Vieta-flavoured
  hard tier would duplicate the sibling skill, which produced a defective
  question whose answer could be read straight off the given root without
  section). Stating the reverse direction plainly, as above, is the fix.

  ``i`` is a free sympy Symbol under the grader's parse
  (``mathgen/grading/typed.py::_parse_expression`` uses implicit multiplication with
  no special-casing of a lowercase ``i``; only capital ``I`` maps to
  ``sympy.I``, confirmed by direct read and REPL check). Every ``answer``
  and ``answer_text`` here is built with ``Symbol("i")``, never
  ``sympy.I``/``complex()`` -- those belong only to the independent
  verification step, which is the one place actual complex arithmetic is
  used, on a deliberately different route from construction.
"""

from __future__ import annotations

from fractions import Fraction

import sympy
from sympy import Rational, Symbol, expand, symbols

from ..core import (
    check,
    check_distinct,
    check_equivalent,
    generator,
    join_terms,
    paren_negative,
    pick_distractors,
    rational_text,
)

x = symbols("x")
i_sym = Symbol("i")


def _poly_display(expr) -> str:
    """Format a polynomial expression for display: replace ** with ^ and remove *."""
    return str(expr).replace("**", "^").replace("*", "")


def _frac_text(value) -> str:
    """Render an int/Fraction/Rational the way a student would write it."""
    if isinstance(value, Fraction):
        value = Rational(value.numerator, value.denominator)
    return rational_text(value)


def _complex_text(p, q) -> str:
    """Render p + q*i in 'p + qi' form, q always the positive-imaginary-part
    magnitude (q > 0 by construction at every call site). Fractional parts
    are written bracket-free (e.g. '3/2i', never '(3/2)i'): confirmed by
    direct parse (`mathgen/grading/typed.py`'s own transformations,
    `standard_transformations + (implicit_multiplication_application,)`)
    that '3/2i' parses left-to-right as (3/2)*i, the intended value, not
    3/(2i) -- so no bracket is needed for correctness. A bracket in the
    canonical is actively harmful here: `grade_expression` rejects any
    learner answer that omits a bracket the canonical itself contains
    (`mathgen/grading/typed.py`, "if '(' in canonical and '(' not in stripped"), and
    a learner asked for 'p + qi' form would naturally type '3/2i' without
    parentheses -- a confirmed live false-reject pattern on an
    already-shipped indices skill, avoided here by construction."""
    p_str = _frac_text(p)
    q_abs_str = _frac_text(abs(q))
    imag = "i" if abs(q) == 1 else f"{q_abs_str}i"
    sign = "+" if q >= 0 else "-"
    return f"{p_str} {sign} {imag}"


def _complex_expr(p, q):
    return Rational(p.numerator, p.denominator) + Rational(q.numerator, q.denominator) * i_sym


# ============================================================================
# SKILL 1: f10-quadratic-algebra-expansion
# ============================================================================


def _factor_text(sign: int, val: int) -> str:
    return f"x + {val}" if sign > 0 else f"x - {val}"


def _term_x(coef: int) -> str:
    """'{coef}x', never rendering a unit coefficient as '1x'."""
    return "x" if coef == 1 else f"{coef}x"


def _expand_easy(rng):
    # (x + a)(x + b), a, b > 0, a != b (a == b would collide with the
    # perfect-square hard-tier pattern), ab != a+b (guards the
    # added-not-multiplied / multiplied-not-added distractors staying
    # distinct from the answer).
    while True:
        a = rng.randint(1, 12)
        b = rng.randint(1, 12)
        if a == b:
            continue
        if a * b == a + b:
            continue
        break

    factor1, factor2 = x + a, x + b
    trinomial = expand(factor1 * factor2)

    # Independent check, second route: numeric substitution at 3 x-values.
    for probe in (Rational(2, 1), Rational(-3, 1), Rational(5, 1)):
        check_equivalent(
            (probe + a) * (probe + b),
            trinomial.subs(x, probe),
            "numeric substitution must agree with the constructed trinomial",
        )

    answer_text = _poly_display(trinomial)

    # Misconceptions. "Sign error on one factor" has nothing to flip here -- both factors are
    # positive at easy -- so it is not exercisable at this tier.
    added_not_multiplied = expand((x + a) * (x + b) - a * b + (a + b))  # constant term becomes a+b
    multiplied_not_added = x**2 + (a * b) * x + a * b  # linear coeff becomes a*b instead of a+b
    dropped_middle = x**2 + a * b  # FOIL cross-terms omitted entirely
    check_distinct(
        trinomial,
        added_not_multiplied,
        multiplied_not_added,
        dropped_middle,
        msg="easy-tier distractor must be a genuinely different expanded expression",
    )

    text = f"Expand ({_factor_text(1, a)})({_factor_text(1, b)}) into a trinomial in standard form."
    distractors = pick_distractors(
        answer_text,
        [
            _poly_display(added_not_multiplied),
            _poly_display(multiplied_not_added),
            _poly_display(dropped_middle),
        ],
    )
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=trinomial,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"({_factor_text(1, a)})({_factor_text(1, b)}) = x^2 + {_term_x(a)} + {_term_x(b)} + {a * b}",
            f"= {answer_text}",
        ],
        distractors=distractors,
        answer_form="expanded",
    )


def _expand_medium(rng):
    # (x + a)(x - b), mixed signs, a != b (a == b would be the hard-tier
    # difference-of-squares pattern).
    while True:
        a = rng.randint(1, 12)
        b = rng.randint(1, 12)
        if a == b:
            continue
        break

    factor1, factor2 = x + a, x - b
    trinomial = expand(factor1 * factor2)

    for probe in (Rational(2, 1), Rational(-3, 1), Rational(5, 1)):
        check_equivalent(
            (probe + a) * (probe - b),
            trinomial.subs(x, probe),
            "numeric substitution must agree with the constructed trinomial",
        )

    answer_text = _poly_display(trinomial)

    # 1. sign-error-one-factor: treats the "-b" factor as "+b" (both same sign)
    sign_error_one_factor = x**2 + (a + b) * x + a * b
    # 2. added-not-multiplied: constant term becomes (a - b) instead of -a*b
    added_not_multiplied = x**2 + (a - b) * x + (a - b)
    # 3. dropped-middle: FOIL cross-terms omitted
    dropped_middle = x**2 - a * b
    check_distinct(
        trinomial,
        sign_error_one_factor,
        added_not_multiplied,
        dropped_middle,
        msg="medium-tier distractor must be a genuinely different expanded expression",
    )

    text = f"Expand ({_factor_text(1, a)})({_factor_text(-1, b)}) into a trinomial in standard form."
    distractors = pick_distractors(
        answer_text,
        [
            _poly_display(sign_error_one_factor),
            _poly_display(added_not_multiplied),
            _poly_display(dropped_middle),
        ],
    )
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=trinomial,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"({_factor_text(1, a)})({_factor_text(-1, b)}) = x^2 + {_term_x(a)} - {_term_x(b)} - {a * b}",
            f"= {answer_text}",
        ],
        distractors=distractors,
        answer_form="expanded",
    )


def _expand_hard_difference_of_squares(rng):
    a = rng.randint(1, 12)
    factor1, factor2 = x + a, x - a
    trinomial = expand(factor1 * factor2)  # x^2 - a^2, middle term vanishes

    for probe in (Rational(2, 1), Rational(-3, 1), Rational(5, 1)):
        check_equivalent(
            (probe + a) * (probe - a),
            trinomial.subs(x, probe),
            "numeric substitution must agree with the constructed trinomial",
        )

    answer_text = _poly_display(trinomial)

    # 1/2. assumed-nonzero-middle-term: treats it as a perfect square either direction
    assumed_same_sign_plus = x**2 + 2 * a * x + a**2
    assumed_same_sign_minus = x**2 - 2 * a * x + a**2
    # 3. sign-error-on-constant: dropped the negative on a^2
    sign_error_constant = x**2 + a**2
    check_distinct(
        trinomial,
        assumed_same_sign_plus,
        assumed_same_sign_minus,
        sign_error_constant,
        msg="hard-tier (difference-of-squares) distractor must be a genuinely different expanded expression",
    )

    text = f"Expand ({_factor_text(1, a)})({_factor_text(-1, a)}) into a trinomial in standard form."
    distractors = pick_distractors(
        answer_text,
        [
            _poly_display(assumed_same_sign_plus),
            _poly_display(assumed_same_sign_minus),
            _poly_display(sign_error_constant),
        ],
    )
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=trinomial,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"({_factor_text(1, a)})({_factor_text(-1, a)}) = x^2 - {_term_x(a)} + {_term_x(a)} - {a * a}",
            "The middle terms cancel.",
            f"= {answer_text}",
        ],
        distractors=distractors,
        answer_form="expanded",
    )


def _expand_hard_perfect_square(rng):
    a = rng.randint(1, 12)
    sign = rng.choice([1, -1])
    factor = x + sign * a
    trinomial = expand(factor * factor)  # x^2 + 2*sign*a*x + a^2

    for probe in (Rational(2, 1), Rational(-3, 1), Rational(5, 1)):
        check_equivalent(
            (probe + sign * a) ** 2,
            trinomial.subs(x, probe),
            "numeric substitution must agree with the constructed trinomial",
        )

    answer_text = _poly_display(trinomial)

    # 1. dropped-the-doubling: wrote ax instead of 2ax
    forgot_double = x**2 + sign * a * x + a**2
    # 2. sign-error-on-constant: flipped the sign of a^2
    sign_error_constant = x**2 + 2 * sign * a * x - a**2
    # 3. dropped-middle: FOIL cross-terms omitted entirely
    dropped_middle = x**2 + a**2
    check_distinct(
        trinomial,
        forgot_double,
        sign_error_constant,
        dropped_middle,
        msg="hard-tier (perfect-square) distractor must be a genuinely different expanded expression",
    )

    ftext = _factor_text(sign, a)
    text = f"Expand ({ftext})({ftext}) into a trinomial in standard form."
    distractors = pick_distractors(
        answer_text,
        [
            _poly_display(forgot_double),
            _poly_display(sign_error_constant),
            _poly_display(dropped_middle),
        ],
    )
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=trinomial,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"({ftext})({ftext}) = x^2 {'+' if sign > 0 else '-'} {_term_x(a)} "
            f"{'+' if sign > 0 else '-'} {_term_x(a)} + {a * a}",
            f"= {answer_text}",
        ],
        distractors=distractors,
        answer_form="expanded",
    )


def _expand_hard(rng):
    # Two sub-cases, both exercised across seeds: difference of squares and
    # perfect square (either sign).
    if rng.choice([True, False]):
        return _expand_hard_difference_of_squares(rng)
    return _expand_hard_perfect_square(rng)


@generator(
    "expand_binomial_product_monic",
    level=9,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def expand_binomial_product_monic(rng, tier="easy"):
    """Expand a binomial product of two linear monic binomial factors into a
    trinomial in standard form. Never factorises anything -- see module
    docstring."""
    if tier == "easy":
        return _expand_easy(rng)
    if tier == "medium":
        return _expand_medium(rng)
    if tier == "hard":
        return _expand_hard(rng)
    raise ValueError(f"unknown tier {tier!r}")


# ============================================================================
# SKILL 2: senior-determine-complex-conjugate-solutions-of-real-quadratic-equations
# ============================================================================


def _complex_easy(rng):
    # Monic x^2 + bx + c = 0. b = 2*b1, c = b1^2 + m1^2 so p = -b1, q = m1
    # are both clean integers.
    while True:
        b1 = rng.choice([v for v in range(-4, 5) if v != 0])
        m1 = rng.randint(1, 4)
        break
    b = 2 * b1
    c = b1 * b1 + m1 * m1
    p = Fraction(-b1, 1)
    q = Fraction(m1, 1)

    # Independent check: substitute p + q*I (real sympy.I, verification only)
    # into x^2 + bx + c and confirm it simplifies to 0 -- a different route
    # from the p/q derivation above.
    root = p + q * sympy.I
    check(sympy.simplify(root**2 + b * root + c) == 0, "the constructed root must satisfy x^2 + bx + c = 0")

    answer_expr = _complex_expr(p, q)
    answer_text = _complex_text(p, q)

    sign_error_real = _complex_expr(-p, q)
    sign_error_imag = _complex_expr(p, -q)
    swapped = _complex_expr(q, p) if p != q else _complex_expr(q, p + 1)
    check_distinct(
        answer_expr,
        sign_error_real,
        sign_error_imag,
        swapped,
        msg="easy-tier complex-conjugate distractor must be a genuinely different value",
    )

    text = (
        f"The equation x^2 {'+' if b >= 0 else '-'} {abs(b)}x + {c} = 0 has complex roots. "
        f"State the solution with the positive imaginary part, in the form p + qi."
    )
    distractors = pick_distractors(
        answer_text,
        [
            _complex_text(-p, q),
            _complex_text(p, -q),
            _complex_text(q, p) if p != q else _complex_text(q, p + 1),
        ],
    )
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer_expr,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[f"x = (-b +/- sqrt(b^2 - 4c)) / 2 = ({-b} +/- sqrt({b * b - 4 * c})) / 2", f"= {answer_text}"],
        distractors=distractors,
    )


def _complex_medium(rng):
    # Non-monic a*x^2 + bx + c = 0, a in {2, 3}. p = -b/(2a), q = sqrt(4ac-b^2)/(2a).
    # Constrain 4ac - b^2 to a perfect square so no surds ever appear; allow
    # q to land as a genuine fraction (the tier's actual escalation).
    while True:
        a = rng.choice([2, 3])
        b = rng.choice([v for v in range(-9, 10) if v != 0])
        m = rng.randint(1, 12)
        numerator = b * b + m * m
        if numerator % (4 * a) != 0:
            continue
        c = numerator // (4 * a)
        if c <= 0:
            continue
        break

    p = Fraction(-b, 2 * a)
    q = Fraction(m, 2 * a)

    root = Rational(p.numerator, p.denominator) + Rational(q.numerator, q.denominator) * sympy.I
    check(sympy.simplify(a * root**2 + b * root + c) == 0, "the constructed root must satisfy a*x^2 + bx + c = 0")

    answer_expr = _complex_expr(p, q)
    answer_text = _complex_text(p, q)

    sign_error_real = _complex_expr(-p, q)
    sign_error_imag = _complex_expr(p, -q)
    # forgot-to-divide-by-2a: divided by 2 instead of 2a (a != 1 here, so this
    # is genuinely a different value from the correct answer).
    forgot_2a = _complex_expr(Fraction(-b, 2), Fraction(m, 2))
    check_distinct(
        answer_expr,
        sign_error_real,
        sign_error_imag,
        forgot_2a,
        msg="medium-tier complex-conjugate distractor must be a genuinely different value",
    )

    quad_text = join_terms([(a, "x^2"), (b, "x"), (c, "")])
    text = (
        f"The equation {quad_text} = 0 has complex roots. "
        f"State the solution with the positive imaginary part, in the form p + qi."
    )
    distractors = pick_distractors(
        answer_text,
        [
            _complex_text(-p, q),
            _complex_text(p, -q),
            _complex_text(Fraction(-b, 2), Fraction(m, 2)),
        ],
    )
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer_expr,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[f"x = (-b +/- sqrt(b^2 - 4ac)) / (2a) = ({-b} +/- sqrt(-{m * m})) / {2 * a}", f"= {answer_text}"],
        distractors=distractors,
    )


def _complex_hard(rng):
    # One complex root is GIVEN: p - qi (q > 0), one solution of a real monic
    # quadratic x^2 + bx + c = 0. b is STATED as a number; c is withheld.
    # The learner must find c -- via the conjugate-root theorem (the other
    # root is p + qi) and Vieta's product relation for a monic quadratic
    # (c = (p - qi)(p + qi) = p^2 + q^2) -- so the equation's own coefficient
    # is load-bearing, not decoration: b confirms which quadratic this is,
    # and c is the thing actually being computed from the given root. See
    # the module docstring for why this does not collide with the sibling
    # skill's coefficients-to-sum/product content.
    #
    # Retry until the true answer and all four distractors are pairwise
    # distinct for these p, q -- squaring erases plain sign flips on p or q,
    # so a handful of small (p, q) pairs would otherwise produce a collision;
    # check_distinct below is the unconditional safety net, this loop is what
    # keeps it from ever firing.
    while True:
        p = rng.choice([v for v in range(-9, 10) if v != 0])
        q = rng.randint(1, 9)
        true_c = p * p + q * q
        diff_of_squares = p * p - q * q  # forgot i^2 = -1
        real_part_only = p * p  # forgot the imaginary contribution
        imag_part_only = q * q  # forgot the real contribution
        confused_with_b = -2 * p  # reported b instead of computing c
        if len({true_c, diff_of_squares, real_part_only, imag_part_only, confused_with_b}) == 5:
            break
    b = -2 * p
    c = true_c

    # Independent check, first route: the GIVEN root (p - q*I, real sympy.I)
    # must satisfy x^2 + bx + c = 0 with the constructed b and c together --
    # confirms b and c are mutually consistent for this root, not just c in
    # isolation.
    given_root = complex(p, -q)
    poly_at_given = given_root**2 + b * given_root + c
    check(abs(poly_at_given) < 1e-9, "the given root must satisfy x^2 + bx + c = 0")
    # Second, independent route to c alone: the other root is the complex
    # conjugate (conjugate-root theorem), and c is the product of the two
    # roots for a monic quadratic -- computed via Python's complex
    # multiplication, not the p*p + q*q formula used above.
    other_root = given_root.conjugate()
    product = given_root * other_root
    check(
        abs(product.imag) < 1e-9 and abs(product.real - c) < 1e-9, "the product of the two conjugate roots must equal c"
    )

    given_text = _complex_text(Fraction(p, 1), -Fraction(q, 1))
    answer_text = str(c)
    answer_expr = sympy.Integer(c)

    distractor_texts = [str(diff_of_squares), str(real_part_only), str(imag_part_only), str(confused_with_b)]
    check_distinct(
        answer_expr,
        sympy.Integer(diff_of_squares),
        sympy.Integer(real_part_only),
        sympy.Integer(imag_part_only),
        sympy.Integer(confused_with_b),
        msg="hard-tier c-value distractor must be a genuinely different value",
    )

    quad_text = f"x^2 {'+' if b >= 0 else '-'} {abs(b)}x + c"
    text = f"{given_text} is one solution of {quad_text} = 0, where the coefficients are real. Find the value of c."
    distractors = pick_distractors(answer_text, distractor_texts)
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer_expr,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            "Complex roots of a real-coefficient quadratic occur in conjugate pairs, "
            f"so the other root is {_complex_text(Fraction(p, 1), Fraction(q, 1))}.",
            "For a monic quadratic, c is the product of the two roots:",
            f"c = ({given_text})({_complex_text(Fraction(p, 1), Fraction(q, 1))}) = {paren_negative(p)}^2 + {paren_negative(q)}^2 = {answer_text}",
        ],
        distractors=distractors,
    )


@generator(
    "quadratic_complex_conjugate_solutions",
    level=11,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def quadratic_complex_conjugate_solutions(rng, tier="easy"):
    """Determine complex conjugate solutions of real quadratic equations,
    stated in p + qi form. See module docstring for the boundary against
    senior-consider-conjugate-roots-for-polynomials-with-real-coefficients."""
    if tier == "easy":
        return _complex_easy(rng)
    if tier == "medium":
        return _complex_medium(rng)
    if tier == "hard":
        return _complex_hard(rng)
    raise ValueError(f"unknown tier {tier!r}")
