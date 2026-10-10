"""Derivative of the inverse trigonometric functions.

- ``find_and_use_the_derivative_of_the_inverse``
  (``senior-find-and-use-the-derivative-of-the-inverse``, ACMSM120, domain differentiation,
  answer: expression) -- given bounded, fully specified inputs, find and use the derivative
  of the inverse trigonometric functions: arcsine, arccosine and arctangent. Return the
  resulting value, expression, relation, or classification.
"""

from __future__ import annotations

from sympy import sqrt, symbols, asin, acos, atan, diff, sec, cos, sin

from ..core import check, check_distinct, check_equivalent, generator, latex_answer, pick_distractors


# Display helpers (^/* convention): convert ** to ^ for typed, ** to ^ and * to · for prose
def _typed(expr) -> str:
    """Convert expression to learner-facing typed form: ** -> ^."""
    return str(expr).replace("**", "^")


def _prose(expr) -> str:
    """Convert expression to prose form: ** -> ^, * to ·."""
    typed_str = str(expr).replace("**", "^")
    return typed_str.replace("*", "·")


@generator(
    "find_and_use_the_derivative_of_the_inverse",
    level=10,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def find_and_use_the_derivative_of_the_inverse(rng, tier="easy"):
    """
    Differentiate inverse trigonometric functions using the chain rule.

    easy: differentiate bare inverse trig functions (asin(x), acos(x), atan(x))
    medium: chain rule with a pure scaling inner argument (k*x)
    hard: chain rule with a full affine inner argument (a*x+b)
    """
    x = symbols("x")

    if tier == "easy":
        # Easy: bare inverse trig function, no chain rule
        kind = rng.choice(["asin", "acos", "atan"])
        f_map = {"asin": asin(x), "acos": acos(x), "atan": atan(x)}
        f = f_map[kind]
        hand = {"asin": 1 / sqrt(1 - x**2), "acos": -1 / sqrt(1 - x**2), "atan": 1 / (1 + x**2)}[kind]

        # Independent check: sympy's own differentiation
        ans = diff(f, x)
        check_equivalent(ans, hand, f"Easy {kind}: symbolic check failed")

        # Distractors per kind
        if kind == "asin":
            sign_flip = -hand  # -1/sqrt(1-x**2) (acos formula)
            no_sqrt = 1 / (1 - x**2)
            confused_with_sin = cos(x)  # cos(x) is derivative of sin(x)
            candidates = [_typed(sign_flip), _typed(no_sqrt), _typed(confused_with_sin)]
        elif kind == "acos":
            sign_flip = -hand  # 1/sqrt(1-x**2) (asin formula)
            no_sqrt = -1 / (1 - x**2)
            confused_with_cos = -sin(x)  # derivative of cos(x)
            candidates = [_typed(sign_flip), _typed(no_sqrt), _typed(confused_with_cos)]
        elif kind == "atan":
            sign_flip = -hand  # -1/(1+x**2)
            confused_with_tan = sec(x) ** 2  # sec(x)^2 is the derivative of tan(x)
            candidates = [_typed(sign_flip), _typed(confused_with_tan)]

        # Verify all candidates are distinct from hand
        for candidate in candidates:
            candidate_expr = candidate.replace("^", "**")
            from sympy import sympify

            check_distinct(sympify(candidate_expr), hand, msg=f"Easy {kind}: candidate {candidate} equals hand")

        distractors = pick_distractors(_typed(hand), candidates)

        learner_name = {"asin": "arcsin(x)", "acos": "arccos(x)", "atan": "arctan(x)"}[kind]
        question_text = f"Differentiate {learner_name} with respect to x."

        return {
            "text": question_text,
            "latex": question_text,
            "answer": str(hand),
            "answer_text": _typed(hand),
            "answer_latex": latex_answer(hand),
            "distractors": distractors,
        }

    elif tier == "medium":
        # Medium: chain rule with pure scaling inner argument (k*x)
        kind = rng.choice(["asin", "acos", "atan"])
        k = rng.choice(range(2, 6))
        u = k * x

        f_map = {"asin": asin(u), "acos": acos(u), "atan": atan(u)}
        f = f_map[kind]
        hand = {"asin": k / sqrt(1 - u**2), "acos": -k / sqrt(1 - u**2), "atan": k / (1 + u**2)}[kind]

        # Independent check
        ans = diff(f, x)
        check_equivalent(ans, hand, f"Medium {kind} k={k}: symbolic check failed")

        # Forgot chain factor distractor (bare formula with u substituted)
        forgot_chain_factor = {"asin": 1 / sqrt(1 - u**2), "acos": -1 / sqrt(1 - u**2), "atan": 1 / (1 + u**2)}[kind]

        # Verify it's distinct from hand (k != 1, so always different)
        check(k != 1, "k must not be 1")
        check_distinct(forgot_chain_factor, hand, msg=f"Medium {kind}: forgot_chain_factor equals hand")

        # Sign flip distractor
        sign_flip = -hand
        check_distinct(sign_flip, hand, msg=f"Medium {kind}: sign_flip equals hand")

        candidates = [_typed(forgot_chain_factor), _typed(sign_flip)]
        distractors = pick_distractors(_typed(hand), candidates)

        learner_name = {"asin": "arcsin", "acos": "arccos", "atan": "arctan"}[kind]
        u_prose = _prose(u)
        question_text = f"Differentiate {learner_name}({u_prose}) with respect to x."

        return {
            "text": question_text,
            "latex": question_text,
            "answer": str(hand),
            "answer_text": _typed(hand),
            "answer_latex": latex_answer(hand),
            "distractors": distractors,
        }

    elif tier == "hard":
        # Hard: chain rule with full affine inner argument (a*x+b)
        kind = rng.choice(["asin", "acos", "atan"])
        a = rng.choice(range(2, 5))
        b = rng.choice([-3, -1, 1, 3])
        u = a * x + b

        f_map = {"asin": asin(u), "acos": acos(u), "atan": atan(u)}
        f = f_map[kind]
        hand = {"asin": a / sqrt(1 - u**2), "acos": -a / sqrt(1 - u**2), "atan": a / (1 + u**2)}[kind]

        # Independent check
        ans = diff(f, x)
        check_equivalent(ans, hand, f"Hard {kind} a={a} b={b}: symbolic check failed")

        # Forgot chain factor (bare formula with u substituted, but missing the a factor)
        forgot_chain_factor = {"asin": 1 / sqrt(1 - u**2), "acos": -1 / sqrt(1 - u**2), "atan": 1 / (1 + u**2)}[kind]
        check_distinct(forgot_chain_factor, hand, msg=f"Hard {kind}: forgot_chain_factor equals hand")

        # Sign flip
        sign_flip = -hand
        check_distinct(sign_flip, hand, msg=f"Hard {kind}: sign_flip equals hand")

        # Dropped intercept (used a*x instead of a*x+b in the substitution)
        dropped_intercept = {
            "asin": a / sqrt(1 - (a * x) ** 2),
            "acos": -a / sqrt(1 - (a * x) ** 2),
            "atan": a / (1 + (a * x) ** 2),
        }[kind]
        check_distinct(dropped_intercept, hand, msg=f"Hard {kind}: dropped_intercept equals hand")

        candidates = [_typed(forgot_chain_factor), _typed(sign_flip), _typed(dropped_intercept)]
        distractors = pick_distractors(_typed(hand), candidates)

        learner_name = {"asin": "arcsin", "acos": "arccos", "atan": "arctan"}[kind]
        u_prose = _prose(u)
        question_text = f"Differentiate {learner_name}({u_prose}) with respect to x."

        return {
            "text": question_text,
            "latex": question_text,
            "answer": str(hand),
            "answer_text": _typed(hand),
            "answer_latex": latex_answer(hand),
            "distractors": distractors,
        }
