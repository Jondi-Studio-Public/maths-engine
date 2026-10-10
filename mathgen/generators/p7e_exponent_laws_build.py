"""Generates `exponent_laws_integer_variable` questions: simplify products, quotients and powers of a variable
with integer exponents, three tiers.

One generator, ``exponent_laws_integer_variable``, three tiers:

- **easy**: product law only, ``x**a * x**b -> x**(a+b)``, with at least one of
  ``a``, ``b`` drawn negative on every seed. The prerequisite skill
  ``f10-exponent-laws-positive-zero`` only ever exercises non-negative exponents,
  so a negative operand on every seed is this tier's whole point.
- **medium**: quotient law only, ``x**a / x**b -> x**(a-b)``, ``a`` and ``b`` both
  positive, biased (``b > a`` on ~75% of draws) so the final exponent lands
  negative on a clear majority of seeds -- the added reasoning step is correctly
  subtracting into negative territory, not just applying the law.
- **hard**: power-of-a-power chained with product/quotient,
  ``(x**a)**b * x**c / x**d -> x**(a*b + c - d)``, with ``a``, ``c``, ``d`` each
  independently signed so the final exponent's sign cannot be read off any single
  input in isolation; ``b`` stays a small positive integer (the outer power),
  matching the senior precedent's own power-of-a-power tier.

Every canonical answer is a single power of ``x``, rendered by this module's own
``_display()`` -- deliberately never sympy's own ``str()``. ``str(x**-1)`` is
``"1/x"`` and every other negative exponent is wrapped in parentheses,
``str(x**-5) == "x**(-5)"``; ``mathgen.grading.grade_expression`` rejects any learner
answer that lacks a "(" whenever the canonical ``answer_text`` contains one (its
factored-form guard, meant for skills such as "3(2x + 3)"), so a canonical
``"x^(-5)"`` would mark a learner who typed the natural ``"x^-5"`` wrong.
``_display()`` always renders ``"x^<int>"``, parenthesis-free, so that guard never
fires and every tier's answer/distractors share one format.
"""

from __future__ import annotations

from sympy import symbols

from ..core import check_distinct, check_equivalent, generator, pick_distractors

x = symbols("x")


def _display(exp: int) -> str:
    """Learner-facing rendering of ``x**exp``: always ``"x^<int>"``, never
    sympy's ``str()`` (which collapses ``x**1`` to ``"x"``, ``x**0`` to ``"1"``,
    renders ``x**-1`` as ``"1/x"`` and wraps every other negative exponent in
    parentheses) -- so every option in a question looks alike, and no canonical
    answer ever contains a "(" that would trip the grader's factored-form guard.
    """
    return f"x^{exp}"


def _term_display(exp: int) -> str:
    """Question-text rendering of one ``x**exp`` input term: parenthesises a
    negative exponent for legibility. This text is only ever shown, never
    parsed by the grader, so the parenthesis-avoidance rule above does not
    apply here."""
    return f"x^{exp}" if exp >= 0 else f"x^({exp})"


def _verify(original_expr, final_exponent) -> None:
    """Independent check: sympy's own simplification of the unsimplified
    original expression against the hand-computed final exponent -- a
    genuinely different route from the add/subtract/multiply arithmetic used
    to derive ``final_exponent``."""
    check_equivalent(
        original_expr,
        x**final_exponent,
        "hand-computed exponent arithmetic disagrees with sympy's own "
        "simplification of the original unsimplified expression",
    )


def _build_easy(rng) -> dict:
    """Product law only: x**a * x**b -> x**(a+b). At least one of a, b is
    negative on every seed."""
    for _ in range(50):
        a = rng.randint(1, 9) * rng.choice([1, -1])
        b = rng.randint(1, 9) * rng.choice([1, -1])
        if a > 0 and b > 0:
            if rng.random() < 0.5:
                a = -a
            else:
                b = -b
        final_exponent = a + b

        wrong_multiply = a * b  # multiplies-instead-of-adds
        wrong_subtract = a - b  # wrong law (quotient instead of product)
        wrong_drop_sign = abs(a) + abs(b)  # drops-the-negative-sign

        if final_exponent not in (wrong_multiply, wrong_subtract, wrong_drop_sign):
            break
    else:
        raise RuntimeError("could not draw a collision-free easy exponent set")

    original_expr = x**a * x**b
    _verify(original_expr, final_exponent)

    answer_expr = x**final_exponent
    check_distinct(
        answer_expr,
        x**wrong_multiply,
        x**wrong_subtract,
        x**wrong_drop_sign,
        msg="an easy-tier distractor equals the answer",
    )

    answer_text = _display(final_exponent)
    text = f"Simplify {_term_display(a)} × {_term_display(b)}, giving your answer as a single power of x."
    latex_text = rf"x^{{{a}}} \cdot x^{{{b}}}"
    candidates = [_display(wrong_multiply), _display(wrong_subtract), _display(wrong_drop_sign)]

    return dict(
        text=text,
        latex=latex_text,
        answer=str(answer_expr),
        answer_text=answer_text,
        answer_latex=answer_text,
        distractors=pick_distractors(answer_text, candidates, n=3),
    )


def _build_medium(rng) -> dict:
    """Quotient law only: x**a / x**b -> x**(a-b), a and b positive, biased so
    the final exponent is negative on a clear majority of seeds."""
    for _ in range(50):
        a = rng.randint(1, 9)
        gap = rng.randint(1, 9)
        if rng.random() < 0.75:
            b = a + gap  # final exponent negative
        else:
            b = max(1, a - gap)
            if b == a:
                b = a + gap  # avoid a zero final exponent; falls back to negative
        final_exponent = a - b
        if final_exponent == 0:
            continue

        wrong_add = a + b  # adds-instead-of-subtracts (wrong law)
        wrong_drop_sign = b - a  # drops-the-negative-sign: reports the magnitude
        # with the wrong (positive) sign

        if final_exponent not in (wrong_add, wrong_drop_sign):
            break
    else:
        raise RuntimeError("could not draw a collision-free medium exponent set")

    original_expr = x**a / x**b
    _verify(original_expr, final_exponent)

    answer_expr = x**final_exponent
    check_distinct(
        answer_expr,
        x**wrong_add,
        x**wrong_drop_sign,
        msg="a medium-tier distractor equals the answer",
    )

    answer_text = _display(final_exponent)
    text = f"Simplify {_term_display(a)} ÷ {_term_display(b)}, giving your answer as a single power of x."
    latex_text = rf"x^{{{a}}} / x^{{{b}}}"
    candidates = [_display(wrong_add), _display(wrong_drop_sign)]

    return dict(
        text=text,
        latex=latex_text,
        answer=str(answer_expr),
        answer_text=answer_text,
        answer_latex=answer_text,
        distractors=pick_distractors(answer_text, candidates, n=3),
    )


def _build_hard(rng) -> dict:
    """Power-of-a-power chained with product/quotient:
    (x**a)**b * x**c / x**d -> x**(a*b + c - d). a, c, d are each
    independently signed so the final exponent's sign cannot be read off any
    single input; b (the outer power) is a small positive integer."""
    for _ in range(50):
        a = rng.randint(1, 6) * rng.choice([1, -1])
        b = rng.randint(2, 4)
        c = rng.randint(1, 9) * rng.choice([1, -1])
        d = rng.randint(1, 9) * rng.choice([1, -1])
        final_exponent = a * b + c - d
        if final_exponent == 0:
            continue

        wrong_power_add = (a + b) + c - d  # power-of-power adds instead of multiplies
        wrong_ignore_division = a * b + c + d  # adds instead of subtracts (wrong law)
        wrong_drop_sign = a * b + abs(c) - abs(d)  # drops-the-negative-sign on c, d

        if final_exponent not in (wrong_power_add, wrong_ignore_division, wrong_drop_sign):
            break
    else:
        raise RuntimeError("could not draw a collision-free hard exponent set")

    original_expr = (x**a) ** b * x**c / x**d
    _verify(original_expr, final_exponent)

    answer_expr = x**final_exponent
    check_distinct(
        answer_expr,
        x**wrong_power_add,
        x**wrong_ignore_division,
        x**wrong_drop_sign,
        msg="a hard-tier distractor equals the answer",
    )

    answer_text = _display(final_exponent)
    text = (
        f"Simplify ({_display(a)})^{b} × {_term_display(c)} ÷ {_term_display(d)}, "
        "giving your answer as a single power of x."
    )
    latex_text = rf"(x^{{{a}}})^{{{b}}} \cdot x^{{{c}}} / x^{{{d}}}"
    candidates = [
        _display(wrong_power_add),
        _display(wrong_ignore_division),
        _display(wrong_drop_sign),
    ]

    return dict(
        text=text,
        latex=latex_text,
        answer=str(answer_expr),
        answer_text=answer_text,
        answer_latex=answer_text,
        distractors=pick_distractors(answer_text, candidates, n=3),
    )


@generator(
    "exponent_laws_integer_variable",
    level=9,
    presets={
        "easy": dict(variant="product"),
        "medium": dict(variant="quotient"),
        "hard": dict(variant="power_chain"),
    },
)
def exponent_laws_integer_variable(rng, variant="product"):
    """Apply the product, quotient and power-of-a-power integer exponent laws
    to a symbolic variable base x, including negative exponents. The
    canonical answer is always a single power of x."""
    if variant == "product":
        return _build_easy(rng)
    if variant == "quotient":
        return _build_medium(rng)
    if variant == "power_chain":
        return _build_hard(rng)
    raise ValueError(f"unknown variant: {variant!r}")
