"""Generates inverse-function questions (ACMSM095): derive the inverse of a linear or reciprocal one-to-one
function.

Senior inverse-function generators (ACMSM095): derive the inverse of
one-to-one functions (linear and reciprocal).

Three difficulty tiers: easy (linear f(x) = ax + b), medium (unexpanded
linear f(x) = a(x + c) + b), and hard (reciprocal f(x) = 1/(x - h) + k).
Every function and every tier's answer is real-valued (no imaginary i, no
fractional powers). Independent check via symbolic composition f(g(x)) = x,
a different route from algebraic rearrangement.
"""

from __future__ import annotations

from sympy import Rational, Symbol, sympify

from ..core import check, check_distinct, check_equivalent, generator, join_terms, pick_distractors

x = Symbol("x")


def _verify_inverse(f_expr, g_expr):
    """Independent check: compose the candidate inverse g back into f and
    verify f(g(x)) simplifies to x. This is the textbook definition of an
    inverse (a different route from algebraic rearrangement of y = f(x) for x).
    """
    composed = f_expr.subs(x, g_expr)
    check_equivalent(composed, x, f"f(g(x)) = {composed} does not simplify to x")


def _verify_answer_text(answer_text, g_expr):
    """The learner is graded against ``answer_text``, not against ``g_expr``.
    Parse the rendered string back and require it to be the verified inverse.

    Without this, :func:`_verify_inverse` validated an expression the learner
    never sees, while a separate hand-rolled formatter produced the string they
    are actually marked against. 197 of 600 questions shipped an ``answer_text``
    that was not the inverse, with the correct expression offered as a
    distractor, so a learner who answered correctly was marked wrong.
    """
    parsed = sympify(answer_text.replace("^", "**"))
    check_equivalent(
        parsed, g_expr, f"answer_text {answer_text!r} parses to {parsed}, which is not the verified inverse {g_expr}"
    )


def _format_rational(num, denom):
    """Format a rational expression (x - num) / denom as a string, using ASCII
    minus sign `-`, without bare * or **.

    Examples:
        (2, 1) -> "x - 2"
        (-2, 1) -> "x + 2"
        (3, 2) -> "(x - 3)/2"
        (3, -2) -> "-(x - 3)/2"
        (-1, 2) -> "(x + 1)/2"
    """
    # Handle negative denominators. (x - num)/(-d) is -((x - num)/d): the sign
    # moves outside and the numerator is unchanged. Negating num here as well
    # rendered -(x + num)/d, which is a different function -- see the docstring
    # example (3, -2) -> "-(x - 3)/2", which this code did not produce.
    if denom < 0:
        denom = -denom
        negate_result = True
    else:
        negate_result = False

    if num >= 0:
        numerator_str = f"x - {num}"
    else:
        numerator_str = f"x + {-num}"

    if denom == 1:
        result = numerator_str
    else:
        result = f"({numerator_str})/{denom}"

    if negate_result:
        result = f"-{result}"

    return result


def _format_answer_with_shift(frac_str, shift):
    """Format an answer expression with a shift: frac_str - shift or frac_str + shift.
    Uses ASCII minus `-` for parseable expressions.
    """
    if shift >= 0:
        return f"{frac_str} - {shift}"
    else:
        return f"{frac_str} + {-shift}"


@generator(
    "inverse_one_to_one_function",
    level=11,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def inverse_one_to_one_function(rng, tier="medium"):
    """Derive the inverse of a bounded, fully specified one-to-one function."""

    if tier == "easy":
        # f(x) = a*x + b, where a in {-4,-3,-2,2,3,4}, b in [-6, 6]
        # f^-1(x) = (x - b) / a
        a = rng.choice([i for i in range(-4, 5) if i not in [-1, 0, 1]])
        b = rng.randint(-6, 6)
        # Exclude b = 0 to prevent sign-error distractor collision
        while b == 0:
            b = rng.randint(-6, 6)

        # Build original function and inverse symbolically
        f_expr = a * x + b
        g_expr = (x - b) / a

        # Verify independent check
        _verify_inverse(f_expr, g_expr)

        # Format answer_text
        answer_text = _format_rational(b, a)
        _verify_answer_text(answer_text, g_expr)

        # Format working steps (using × for products, − for subtraction)
        if b >= 0:
            b_sign = "−"
            b_val = b
        else:
            b_sign = "+"
            b_val = -b

        # b_sign/b_val are the INVERSE step (subtract b); f itself carries the opposite sign.
        f_sign, f_val = ("+", b) if b >= 0 else ("−", -b)
        working = [
            f"Let y = {a}×x {f_sign} {f_val}. To find f⁻¹(x), solve for x:",
            f"y = {a}×x {f_sign} {f_val}",
            f"y {b_sign} {b_val} = {a}×x",
            f"(y {b_sign} {b_val})/{a} = x",
            f"f⁻¹(x) = {answer_text}",
        ]

        # Misconception distractors (use * for multiplication in expressions)
        # unchanged-function: states f(x) itself unchanged
        if b >= 0:
            unchanged = f"{a}*x - {b}"
        else:
            unchanged = f"{a}*x + {-b}"

        # sign-error-on-shift: (x + b) / a instead of (x - b) / a
        sign_error = _format_rational(-b, a)

        # forgot-the-coefficient: x - b instead of (x - b) / a
        if b >= 0:
            forgot_coef = f"x - {b}"
        else:
            forgot_coef = f"x + {-b}"

        candidates = [unchanged, sign_error, forgot_coef]

        # Verify all misconceptions are NOT sympy-equivalent to true answer
        g_sympified = sympify(answer_text)
        for cand_text in candidates:
            cand_sympified = sympify(cand_text)
            check_distinct(
                cand_sympified, g_sympified, msg=f"Misconception {cand_text} is sympy-equivalent to true inverse"
            )

        # Format prompt text: avoid awkward "+ -N" by using "- N"
        if b >= 0:
            prompt_b = f"+ {b}"
        else:
            prompt_b = f"- {-b}"

        # Apply display transformation to replace * with × in distractors
        distractors_raw = pick_distractors(answer_text, candidates)
        distractors = [d.replace("*", "×") if isinstance(d, str) else d for d in distractors_raw]

        return dict(
            text=f"Find the inverse function f⁻¹(x) for f(x) = {a}×x {prompt_b}.",
            latex=rf"f(x) = {a}x {f_sign} {f_val}",
            answer=answer_text,
            answer_text=answer_text,
            answer_latex=rf"f^{{-1}}(x) = {answer_text}",
            working=working,
            distractors=distractors,
        )

    elif tier == "medium":
        # f(x) = a*(x + c) + b, where a in {-4,-3,-2,2,3,4}, b,c in [-6, 6]
        # But we must ensure the unexpanded form is genuinely needed
        a = rng.choice([i for i in range(-4, 5) if i not in [-1, 0, 1]])
        b = rng.randint(-6, 6)
        c = rng.randint(-6, 6)

        # Exclude b = 0 and c = 0 to prevent collisions and ensure escalation
        while b == 0:
            b = rng.randint(-6, 6)
        while c == 0:
            c = rng.randint(-6, 6)

        # f^-1(x) = (x - b) / a - c
        f_expr = a * (x + c) + b
        g_expr = (x - b) / a - c

        # Verify independent check
        _verify_inverse(f_expr, g_expr)

        # Format inverse answer: (x - b) / a - c
        frac_str = _format_rational(b, a)
        answer_text = _format_answer_with_shift(frac_str, c)
        _verify_answer_text(answer_text, g_expr)

        # Format working (using × and −)
        if b >= 0:
            b_sign, b_val = "−", b
        else:
            b_sign, b_val = "+", -b

        if c >= 0:
            c_sign, c_val = "+", c
        else:
            c_sign, c_val = "−", -c

        # b_sign/b_val are the INVERSE step (subtract b); f itself carries the opposite sign.
        f_sign, f_val = ("+", b) if b >= 0 else ("−", -b)
        working = [
            f"Let y = {a}×(x {c_sign} {c_val}) {f_sign} {f_val}. To find f⁻¹(x), solve for x:",
            f"y = {a}×(x {c_sign} {c_val}) {f_sign} {f_val}",
            f"y {b_sign} {b_val} = {a}×(x {c_sign} {c_val})",
            f"(y {b_sign} {b_val})/{a} = x {c_sign} {c_val}",
            f"x = (y {b_sign} {b_val})/{a} {f'−' if c >= 0 else '+'} {c if c >= 0 else -c}",
            f"f⁻¹(x) = {answer_text}",
        ]

        # Misconception distractors
        # unchanged-function: states f(x) itself unchanged
        # Use sign/value pattern to avoid + - pattern
        if c >= 0 and b >= 0:
            unchanged = f"{a}*(x + {c}) - {b}"
        elif c >= 0 and b < 0:
            unchanged = f"{a}*(x + {c}) + {-b}"
        elif c < 0 and b >= 0:
            unchanged = f"{a}*(x - {-c}) - {b}"
        else:
            unchanged = f"{a}*(x - {-c}) + {-b}"

        # sign-error-on-shift: (x + b) / a - c
        sign_err_frac = _format_rational(-b, a)
        sign_error = _format_answer_with_shift(sign_err_frac, c)

        # forgot-the-coefficient: x - b - c
        if b >= 0 and c >= 0:
            forgot_coef = f"x - {b} - {c}"
        elif b >= 0 and c < 0:
            forgot_coef = f"x - {b} + {-c}"
        elif b < 0 and c >= 0:
            forgot_coef = f"x + {-b} - {c}"
        else:
            forgot_coef = f"x + {-b} + {-c}"

        candidates = [unchanged, sign_error, forgot_coef]

        # Verify all misconceptions
        g_sympified = sympify(answer_text)
        for cand_text in candidates:
            cand_sympified = sympify(cand_text)
            check_distinct(
                cand_sympified, g_sympified, msg=f"Misconception {cand_text} is sympy-equivalent to true inverse"
            )

        # Format prompt text: use ASCII characters
        if c >= 0:
            prompt_c = f"+ {c}"
        else:
            prompt_c = f"- {-c}"
        if b >= 0:
            prompt_b = f"+ {b}"
        else:
            prompt_b = f"- {-b}"

        # Apply display transformation to replace * with × in distractors
        distractors_raw = pick_distractors(answer_text, candidates)
        distractors = [d.replace("*", "×") if isinstance(d, str) else d for d in distractors_raw]

        return dict(
            text=f"Find the inverse function f⁻¹(x) for f(x) = {a}×(x {prompt_c}) {prompt_b}.",
            latex=rf"f(x) = {a}(x {c_sign} {c_val}) {f_sign} {f_val}",
            answer=answer_text,
            answer_text=answer_text,
            answer_latex=rf"f^{{-1}}(x) = {answer_text}",
            working=working,
            distractors=distractors,
        )

    elif tier == "hard":
        # f(x) = 1/(x - h) + k, where h,k in [-6, 6]
        # Domain is x != h (naturally one-to-one on reals != 0)
        # f^-1(x) = h + 1/(x - k)

        # Exclude h = 0 to prevent forgot-the-shift collision
        # Exclude h = k to prevent self-inverse collision
        # Exclude k = 0 to avoid awkward "+ 0" in prompts
        valid_h = [i for i in range(-6, 7) if i != 0]
        valid_k = [i for i in range(-6, 7) if i != 0]

        h = rng.choice(valid_h)
        k = rng.choice(valid_k)
        # Ensure h != k
        while k == h:
            k = rng.choice(valid_k)

        # Build expressions
        f_expr = 1 / (x - h) + k
        g_expr = h + 1 / (x - k)

        # Verify independent check
        _verify_inverse(f_expr, g_expr)

        # Format inverse answer: h + 1/(x - k) using ASCII characters for parsing
        if k >= 0:
            x_k_str_ascii = f"x - {k}"  # ASCII for parsing
        else:
            x_k_str_ascii = f"x + {-k}"

        answer_text = f"{h} + 1/({x_k_str_ascii})"
        _verify_answer_text(answer_text, g_expr)

        # Format working (using − for subtraction, ÷ for division in prose)
        if h >= 0:
            h_sign, h_val = "−", h
        else:
            h_sign, h_val = "+", -h

        if k >= 0:
            k_sign, k_val = "−", k
        else:
            k_sign, k_val = "+", -k

        working = [
            f"Let y = 1/(x {h_sign} {h_val}) {k_sign} {k_val}. To find f⁻¹(x), solve for x:",
            f"y = 1/(x {h_sign} {h_val}) {k_sign} {k_val}",
            f"y {f'−' if k >= 0 else '+'} {k if k >= 0 else -k} = 1/(x {h_sign} {h_val})",
            f"x {h_sign} {h_val} = 1/(y {f'−' if k >= 0 else '+'} {k if k >= 0 else -k})",
            f"x = {h} + 1/(y {f'−' if k >= 0 else '+'} {k if k >= 0 else -k})",
            f"f⁻¹(x) = {answer_text}",
        ]

        # Misconception distractors for hard tier (use ASCII for parsing)
        # unchanged-function: states f(x) itself unchanged
        if h >= 0:
            h_unchanged = f"x - {h}"
        else:
            h_unchanged = f"x + {-h}"
        # Use sign/value pattern to avoid + - pattern
        if k >= 0:
            unchanged = f"1/({h_unchanged}) - {k}"
        else:
            unchanged = f"1/({h_unchanged}) + {-k}"

        # sign-error-on-shift: h - 1/(x - k)
        sign_error = f"{h} - 1/({x_k_str_ascii})"

        # forgot-the-shift: 1/(x - k)
        forgot_shift = f"1/({x_k_str_ascii})"

        candidates = [unchanged, sign_error, forgot_shift]

        # Verify all misconceptions
        g_sympified = sympify(answer_text)
        for cand_text in candidates:
            cand_sympified = sympify(cand_text)
            check_distinct(
                cand_sympified, g_sympified, msg=f"Misconception {cand_text} is sympy-equivalent to true inverse"
            )

        # Format prompt text: use ASCII characters
        if h >= 0:
            prompt_h = f"- {h}"
        else:
            prompt_h = f"+ {-h}"
        if k >= 0:
            prompt_k = f"+ {k}"
        else:
            prompt_k = f"- {-k}"

        return dict(
            text=f"Find the inverse function f⁻¹(x) for f(x) = 1/(x {prompt_h}) {prompt_k}.",
            latex=rf"f(x) = \frac{{1}}{{x {h_sign} {h_val}}} {k_sign} {k_val}",
            answer=answer_text,
            answer_text=answer_text,
            answer_latex=rf"f^{{-1}}(x) = {h} + \frac{{1}}{{x {k_sign} {k_val}}}",
            working=working,
            distractors=pick_distractors(answer_text, candidates),
        )

    else:
        raise ValueError(f"Unknown tier: {tier}")
