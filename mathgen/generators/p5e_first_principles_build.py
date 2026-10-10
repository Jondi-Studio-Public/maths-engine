"""Generates `define_the_derivative_f_left_x_right_as` questions: the derivative of a function from first
principles.

The derivative from first principles.

- ``senior_define_the_derivative_f_left_x_right_as``
  (``senior-define-the-derivative-f-left-x-right-as``,
  ACMMM082, domain differentiation, answer: expression) -- given bounded, fully
  specified inputs, define the derivative f'(x) as lim_{h->0} (f(x+h)-f(x))/h.
  Return the resulting value, expression, relation, or classification.
"""

from __future__ import annotations

from fractions import Fraction
from sympy import symbols, expand, diff, simplify, sympify, Rational

from ..core import check, check_distinct, check_equivalent, generator, pick_distractors


# Display helpers (^/* convention): convert ** to ^ for typed, ** to ^ and * to · for prose
def _typed(expr) -> str:
    """Convert expression to learner-facing typed form: ** -> ^."""
    return str(expr).replace("**", "^")


def _prose(expr) -> str:
    """Convert expression to prose form: ** -> ^, * -> ·."""
    typed_str = str(expr).replace("**", "^")
    return typed_str.replace("*", "·")


# Module-scope symbols for easy/medium tiers (both use x and h)
x, h = symbols("x h")


@generator(
    "define_the_derivative_f_left_x_right_as",
    level=13,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def define_the_derivative_f_left_x_right_as(rng, tier="easy"):
    """
    Form and simplify the difference quotient, or apply the defining limit.

    easy: f(x) = a*x^2 + c (b=0); form and simplify (f(x+h)-f(x))/h
    medium: f(x) = a*x^2 + b*x + c (b≠0); form and simplify (f(x+h)-f(x))/h
    hard: f(x) = a*x^2 + b*x + c (b≠0); find f'(x) = lim_{h->0}(f(x+h)-f(x))/h
    """

    # Sample parameters
    a = rng.randrange(2, 7)  # [2, 6]
    c = rng.randrange(-9, 10)  # [-9, 9]

    if tier == "easy":
        # b is fixed at 0 for easy tier
        b = 0
        f = a * x**2 + c
        hand_answer = 2 * a * x + a * h

        # Question text
        question_text = f"f(x) = {_prose(f)}. Form and fully simplify the difference quotient (f(x+h) - f(x)) / h."

        # Construction check: expand the difference quotient and verify it matches
        dq = expand((f.subs(x, x + h) - f) / h)
        check_equivalent(dq, hand_answer, f"easy: construction check failed for a={a}, c={c}")

        # Independent check: numeric substitution at x0=3, h0=1/1000
        x0 = 3
        h0 = Rational(1, 1000)
        exact = simplify((f.subs(x, x0 + h0) - f.subs(x, x0)) / h0)
        handval = hand_answer.subs({x: x0, h: h0})
        check_equivalent(exact, handval, f"easy: numeric check failed for a={a}, c={c}")

        # Distractors
        not_divided = expand(f.subs(x, x + h) - f)
        dropped_h_term = 2 * a * x
        sign_err = -hand_answer

        # Verify distinctness
        check_distinct(hand_answer, not_divided, msg=f"easy: not_divided equals answer for a={a}, c={c}")
        check_distinct(hand_answer, dropped_h_term, msg=f"easy: dropped_h_term equals answer for a={a}, c={c}")
        check_distinct(hand_answer, sign_err, msg=f"easy: sign_err equals answer for a={a}, c={c}")

        candidates = [
            _typed(not_divided),
            _typed(dropped_h_term),
            _typed(sign_err),
        ]
        distractors = pick_distractors(_typed(hand_answer), candidates)

        return {
            "text": question_text,
            "latex": question_text,
            "answer": str(hand_answer),  # Bare str with ** allowed for grader
            "answer_text": _typed(hand_answer),
            "answer_latex": _typed(hand_answer),
            "distractors": distractors,
        }

    elif tier == "medium":
        # b is nonzero for medium tier
        b_pool = [v for v in range(-9, 10) if v != 0]
        b = rng.choice(b_pool)
        f = a * x**2 + b * x + c
        hand_answer = 2 * a * x + a * h + b

        # Question text
        question_text = f"f(x) = {_prose(f)}. Form and fully simplify the difference quotient (f(x+h) - f(x)) / h."

        # Construction check
        dq = expand((f.subs(x, x + h) - f) / h)
        check_equivalent(dq, hand_answer, f"medium: construction check failed for a={a}, b={b}, c={c}")

        # Independent check: numeric substitution at x0=3, h0=1/1000
        x0 = 3
        h0 = Rational(1, 1000)
        exact = simplify((f.subs(x, x0 + h0) - f.subs(x, x0)) / h0)
        handval = hand_answer.subs({x: x0, h: h0})
        check_equivalent(exact, handval, f"medium: numeric check failed for a={a}, b={b}, c={c}")

        # Distractors
        not_divided = expand(f.subs(x, x + h) - f)
        dropped_h_term = 2 * a * x + b
        sign_err = -hand_answer

        # Verify distinctness
        check_distinct(hand_answer, not_divided, msg=f"medium: not_divided equals answer for a={a}, b={b}, c={c}")
        check_distinct(hand_answer, dropped_h_term, msg=f"medium: dropped_h_term equals answer for a={a}, b={b}, c={c}")
        check_distinct(hand_answer, sign_err, msg=f"medium: sign_err equals answer for a={a}, b={b}, c={c}")

        candidates = [
            _typed(not_divided),
            _typed(dropped_h_term),
            _typed(sign_err),
        ]
        distractors = pick_distractors(_typed(hand_answer), candidates)

        return {
            "text": question_text,
            "latex": question_text,
            "answer": str(hand_answer),  # Bare str with ** allowed for grader
            "answer_text": _typed(hand_answer),
            "answer_latex": _typed(hand_answer),
            "distractors": distractors,
        }

    elif tier == "hard":
        # b is nonzero for hard tier
        b_pool = [v for v in range(-9, 10) if v != 0]
        b = rng.choice(b_pool)
        f = a * x**2 + b * x + c
        hand_answer = 2 * a * x + b  # No h term: the limit has been taken

        # Question text
        question_text = f"f(x) = {_prose(f)}. Using the definition f'(x) = lim_(h->0) (f(x+h) - f(x)) / h, find f'(x)."

        # Independent check: compare to sympy's own differentiation
        check_equivalent(diff(f, x), hand_answer, f"hard: diff check failed for a={a}, b={b}, c={c}")

        # Distractors
        keep_h_term = 2 * a * x + b + a * h
        b_dropped = 2 * a * x
        power_rule_err = a * x + b

        # Verify distinctness
        check(
            h in keep_h_term.free_symbols and h not in hand_answer.free_symbols,
            f"hard: keep_h_term should have h, hand_answer should not",
        )
        check_distinct(hand_answer, b_dropped, msg=f"hard: b_dropped equals answer for a={a}, b={b}, c={c}")
        check_distinct(hand_answer, power_rule_err, msg=f"hard: power_rule_err equals answer for a={a}, b={b}, c={c}")

        candidates = [
            _typed(keep_h_term),
            _typed(b_dropped),
            _typed(power_rule_err),
        ]
        distractors = pick_distractors(_typed(hand_answer), candidates)

        return {
            "text": question_text,
            "latex": question_text,
            "answer": str(hand_answer),  # Bare str with ** allowed for grader
            "answer_text": _typed(hand_answer),
            "answer_latex": _typed(hand_answer),
            "distractors": distractors,
        }
