"""Generates `difference_of_squares` questions: expand or factorise a^2 - b^2, (a + b)^2 and (a - b)^2 by pattern
recognition.

Difference-of-squares.

Scope: the RECOGNISED special-product forms only -- a^2 - b^2, (a + b)^2,
(a - b)^2 -- answered by pattern recognition, both expanding and factorising.
This never multiplies out a general binomial product term by term (that is
``expand_bracket``/``expand_single_brackets``) and never factorises a general
(non-special) trinomial (that is ``f10-quadratic-algebra-expansion``'s boundary,
which itself only ever EXPANDS, never factorises).

Every Question this generator returns sets ``answer_form`` explicitly: "expanded" for an "Expand..." prompt,
"factored" for a "Factorise..." prompt. See mathgen/grading/typed.py:grade_expression.
"""

from __future__ import annotations

import math

from sympy import expand, symbols

from ..core import check_distinct, check_equivalent, fmt_signed, generator, pick_distractors

x, y = symbols("x y")


def _factor_text(a: int, sign: int, b: int) -> str:
    """'{a}x + {b}' / '{a}x - {b}', never a unit '1x' (fmt_signed)."""
    op = "+" if sign > 0 else "-"
    return f"{fmt_signed(a, 'x')} {op} {b}"


def _factor_text_xy(a: int, sign: int, b: int) -> str:
    """Like `_factor_text` but the second term is `b*y`, not a bare constant."""
    op = "+" if sign > 0 else "-"
    return f"{fmt_signed(a, 'x')} {op} {fmt_signed(b, 'y')}"


def _bracket_pair(f1: str, f2: str) -> str:
    return f"({f1})({f2})"


def _poly_text(expr) -> str:
    """Display a sympy polynomial the way a student would write it: no '*', '^' for powers."""
    return str(expr).replace("**", "^").replace("*", "")


def _dos(a: int, b: int):
    """a^2x^2 - b^2, closed-form difference-of-squares formula (route 1)."""
    return expand((a * x) ** 2 - b**2)


def _dos_two_var(a: int, b: int):
    """a^2x^2 - b^2y^2, closed-form two-variable difference-of-squares."""
    return expand((a * x) ** 2 - (b * y) ** 2)


def _ps_plus(a: int, b: int):
    """a^2x^2 + 2abx + b^2, closed-form perfect-square formula (route 1)."""
    return expand((a * x) ** 2 + 2 * a * b * x + b**2)


def _ps_minus(a: int, b: int):
    """a^2x^2 - 2abx + b^2, closed-form perfect-square formula (route 1)."""
    return expand((a * x) ** 2 - 2 * a * b * x + b**2)


@generator(
    "difference_of_squares",
    level=9,
    presets={
        "easy": dict(a_lo=1, a_hi=1, b_lo=2, b_hi=9),
        "medium": dict(a_lo=2, a_hi=6, b_lo=2, b_hi=9),
        "hard": dict(a_lo=1, a_hi=9, b_lo=10, b_hi=25),
    },
)
def difference_of_squares(rng, a_lo: int = 1, a_hi: int = 1, b_lo: int = 2, b_hi: int = 9, kind: str = "single_var"):
    if kind == "two_var":
        return _difference_of_squares_two_var(rng, a_lo, a_hi, b_lo, b_hi)
    # Coprime a/b so the resulting bracket never carries an extractable common
    # numerical factor (e.g. a=4, b=4 -> (4x + 4)^2, whose "cleanest" form is
    # really 16(x + 1)^2) -- that extraction is factor-common-numerical-
    # factor's own content, a listed prerequisite, and this skill's item
    # should test recognition of the special-product pattern alone.
    while True:
        a = rng.randint(a_lo, a_hi)
        b = rng.randint(b_lo, b_hi)
        if math.gcd(a, b) == 1:
            break
    form = rng.choice(["dos", "ps_plus", "ps_minus"])
    direction = rng.choice(["expand", "factorise"])

    dos_val = _dos(a, b)
    ps_plus_val = _ps_plus(a, b)
    ps_minus_val = _ps_minus(a, b)
    # Independent check, second route: a literal multiplication of the two
    # linear factors (dos) / the squared bracket (ps_plus, ps_minus) -- never
    # the closed-form arithmetic construction above.
    check_equivalent(
        dos_val, expand((a * x + b) * (a * x - b)), "dos construction must agree with the literal-multiplication route"
    )
    check_equivalent(
        ps_plus_val, expand((a * x + b) ** 2), "ps_plus construction must agree with the literal-multiplication route"
    )
    check_equivalent(
        ps_minus_val, expand((a * x - b) ** 2), "ps_minus construction must agree with the literal-multiplication route"
    )

    values = {"dos": dos_val, "ps_plus": ps_plus_val, "ps_minus": ps_minus_val}
    brackets = {
        "dos": _bracket_pair(_factor_text(a, 1, b), _factor_text(a, -1, b)),
        "ps_plus": _bracket_pair(_factor_text(a, 1, b), _factor_text(a, 1, b)),
        "ps_minus": _bracket_pair(_factor_text(a, -1, b), _factor_text(a, -1, b)),
    }
    trinomial = values[form]
    trinomial_text = _poly_text(trinomial)
    bracket_text = brackets[form]

    if direction == "expand":
        prompt_bracket = {
            "dos": f"({_factor_text(a, 1, b)})({_factor_text(a, -1, b)})",
            "ps_plus": f"({_factor_text(a, 1, b)})^2",
            "ps_minus": f"({_factor_text(a, -1, b)})^2",
        }[form]
        text = f"Expand {prompt_bracket}."

        if form == "dos":
            # The three named "expand" misconceptions (dropped-middle-term,
            # sign-error-middle-term, forgot-to-double) are all defined "for a
            # perfect-square product" and are not exercisable when the actual
            # question IS a genuine difference of squares (there is no middle
            # term to drop the sign of, or to forget to double). The disclosed
            # complement here: assumed-perfect-square (wrongly includes a
            # nonzero middle term, either sign) and sign-error-constant
            # (flips the sign of the constant term instead).
            sign_error_constant = expand((a * x) ** 2 + b**2)
            check_distinct(
                trinomial,
                values["ps_plus"],
                values["ps_minus"],
                sign_error_constant,
                msg="dos expand distractor must be a genuinely different expanded expression",
            )
            cands = [_poly_text(values["ps_plus"]), _poly_text(values["ps_minus"]), _poly_text(sign_error_constant)]
        elif form == "ps_plus":
            forgot_double = expand((a * x) ** 2 + a * b * x + b**2)
            check_distinct(
                trinomial,
                values["dos"],
                values["ps_minus"],
                forgot_double,
                msg="ps_plus expand distractor must be a genuinely different expanded expression",
            )
            cands = [_poly_text(values["dos"]), _poly_text(values["ps_minus"]), _poly_text(forgot_double)]
        else:  # ps_minus
            forgot_double = expand((a * x) ** 2 - a * b * x + b**2)
            check_distinct(
                trinomial,
                values["dos"],
                values["ps_plus"],
                forgot_double,
                msg="ps_minus expand distractor must be a genuinely different expanded expression",
            )
            cands = [_poly_text(values["dos"]), _poly_text(values["ps_plus"]), _poly_text(forgot_double)]

        answer, answer_text, answer_form = trinomial, trinomial_text, "expanded"
    else:
        text = f"Factorise {trinomial_text} using a recognised special product."
        b2 = b + 1  # wrong-constant-value: same sign pattern, constant off by one.
        wrong_const_value = {"dos": _dos, "ps_plus": _ps_plus, "ps_minus": _ps_minus}[form](a, b2)
        wrong_const_bracket = {
            "dos": _bracket_pair(_factor_text(a, 1, b2), _factor_text(a, -1, b2)),
            "ps_plus": _bracket_pair(_factor_text(a, 1, b2), _factor_text(a, 1, b2)),
            "ps_minus": _bracket_pair(_factor_text(a, -1, b2), _factor_text(a, -1, b2)),
        }[form]

        if form == "dos":
            # wrong-sign-in-factor: (a - b)(a - b) instead of (a + b)(a - b).
            # treated-sum-as-factorable: (a + b)(a + b), a wrong FORM (still a
            # syntactically valid bracketed product), not a nonsensical one.
            check_distinct(
                trinomial,
                values["ps_plus"],
                values["ps_minus"],
                wrong_const_value,
                msg="dos factorise distractor must be a genuinely different value",
            )
            cands = [brackets["ps_minus"], brackets["ps_plus"], wrong_const_bracket]
        elif form == "ps_plus":
            # wrong-sign-in-factor ("vice versa" reading): flips ONE
            # sign, landing on the dos form. sign-error-both-factors
            # (disclosed complement): flips BOTH signs, landing on ps_minus.
            check_distinct(
                trinomial,
                values["dos"],
                values["ps_minus"],
                wrong_const_value,
                msg="ps_plus factorise distractor must be a genuinely different value",
            )
            cands = [brackets["dos"], brackets["ps_minus"], wrong_const_bracket]
        else:  # ps_minus
            check_distinct(
                trinomial,
                values["dos"],
                values["ps_plus"],
                wrong_const_value,
                msg="ps_minus factorise distractor must be a genuinely different value",
            )
            cands = [brackets["dos"], brackets["ps_plus"], wrong_const_bracket]

        answer, answer_text, answer_form = bracket_text, bracket_text, "factored"

    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        distractors=pick_distractors(answer_text, cands),
        answer_form=answer_form,
    )


def _difference_of_squares_two_var(rng, a_lo: int = 1, a_hi: int = 6, b_lo: int = 1, b_hi: int = 6):
    """A SECOND variable, not merely bigger coefficients of the same one-variable pattern the easy/medium tiers
    use."""
    while True:
        a = rng.randint(a_lo, a_hi)
        b = rng.randint(b_lo, b_hi)
        if math.gcd(a, b) == 1:
            break
    direction = rng.choice(["expand", "factorise"])
    trinomial = _dos_two_var(a, b)
    check_equivalent(
        trinomial,
        expand((a * x + b * y) * (a * x - b * y)),
        "two-variable dos must agree with the literal-multiplication route",
    )
    trinomial_text = _poly_text(trinomial)
    plus_factor = _factor_text_xy(a, 1, b)
    minus_factor = _factor_text_xy(a, -1, b)
    bracket_text = _bracket_pair(plus_factor, minus_factor)
    sum_squares = expand((a * x) ** 2 + (b * y) ** 2)
    both_minus_sq = expand((a * x - b * y) ** 2)
    both_plus_sq = expand((a * x + b * y) ** 2)

    if direction == "expand":
        text = f"Expand ({plus_factor})({minus_factor})."
        check_distinct(
            trinomial,
            sum_squares,
            both_minus_sq,
            both_plus_sq,
            msg="two-var dos expand distractor must be a genuinely different expanded expression",
        )
        cands = [_poly_text(sum_squares), _poly_text(both_minus_sq), _poly_text(both_plus_sq)]
        answer, answer_text, answer_form = trinomial, trinomial_text, "expanded"
    else:
        text = f"Factorise {trinomial_text} using a recognised special product."
        b2 = b + 1
        wrong_const = expand((a * x) ** 2 - (b2 * y) ** 2)
        wrong_const_bracket = _bracket_pair(_factor_text_xy(a, 1, b2), _factor_text_xy(a, -1, b2))
        both_plus_bracket = _bracket_pair(plus_factor, plus_factor)
        both_minus_bracket = _bracket_pair(minus_factor, minus_factor)
        check_distinct(
            trinomial,
            both_plus_sq,
            both_minus_sq,
            wrong_const,
            msg="two-var dos factorise distractor must be a genuinely different value",
        )
        cands = [both_plus_bracket, both_minus_bracket, wrong_const_bracket]
        answer, answer_text, answer_form = bracket_text, bracket_text, "factored"

    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        distractors=pick_distractors(answer_text, cands),
        answer_form=answer_form,
    )
