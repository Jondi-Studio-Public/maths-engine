"""Evaluate and use inverse trigonometric functions.

- ``find_and_use_the_inverse_trigonometric_functions_arcsine``
  (``senior-find-and-use-the-inverse-trigonometric-functions-arcsine``,
  ACMSM119, domain differentiation, answer: expression) -- given bounded, fully
  specified inputs, find and use the inverse trigonometric functions: arcsine,
  arccosine and arctangent. Return the resulting value, expression, relation, or
  classification.
"""

from __future__ import annotations

from sympy import sqrt, Rational, pi, asin, acos, atan, sin, cos, tan, sympify

from ..core import check, check_distinct, check_equivalent, generator, latex_answer, pick_distractors


# Display helpers (^/* convention): convert ** to ^ for typed, ** to ^ and * to · for prose
def _typed(expr) -> str:
    """Convert expression to learner-facing typed form: ** -> ^."""
    return str(expr).replace("**", "^")


def _prose(expr) -> str:
    """Convert expression to prose form: ** -> ^, * to ·."""
    typed_str = str(expr).replace("**", "^")
    return typed_str.replace("*", "·")


# Fixed lookup tables (exact values)
_EASY_ROWS = [
    ("asin", "1/2", Rational(1, 2), "pi/6", ["pi/3", "pi/4", "-pi/6"]),
    ("asin", "sqrt(2)/2", sqrt(2) / 2, "pi/4", ["pi/2", "pi/3", "pi/8"]),
    ("asin", "sqrt(3)/2", sqrt(3) / 2, "pi/3", ["pi/6", "pi/2", "2*pi/3"]),
    ("asin", "1", Rational(1), "pi/2", ["pi", "0", "pi/4"]),
    ("acos", "1/2", Rational(1, 2), "pi/3", ["pi/6", "2*pi/3", "pi/2"]),
    ("acos", "sqrt(2)/2", sqrt(2) / 2, "pi/4", ["pi/2", "3*pi/4", "pi/8"]),
    ("acos", "sqrt(3)/2", sqrt(3) / 2, "pi/6", ["pi/3", "5*pi/6", "pi/2"]),
    ("acos", "1", Rational(1), "0", ["pi/2", "pi", "-pi/2"]),
    ("atan", "1", Rational(1), "pi/4", ["pi/2", "pi/3", "pi/8"]),
    ("atan", "sqrt(3)", sqrt(3), "pi/3", ["pi/6", "2*pi/3", "pi/2"]),
    ("atan", "sqrt(3)/3", sqrt(3) / 3, "pi/6", ["pi/3", "pi/2", "pi/12"]),
    ("atan", "0", Rational(0), "0", ["pi/2", "pi", "-pi/4"]),
]

_MEDIUM_ROWS = [
    ("asin", "-1/2", Rational(-1, 2), "-pi/6", ["pi/6", "pi/3", "-pi/3"]),
    ("asin", "-sqrt(2)/2", -sqrt(2) / 2, "-pi/4", ["pi/4", "pi/2", "-pi/2"]),
    ("asin", "-sqrt(3)/2", -sqrt(3) / 2, "-pi/3", ["pi/3", "pi/6", "-pi/6"]),
    ("asin", "-1", Rational(-1), "-pi/2", ["pi/2", "pi", "0"]),
    ("acos", "-1/2", Rational(-1, 2), "2*pi/3", ["-pi/3", "pi/3", "pi/2"]),
    ("acos", "-sqrt(2)/2", -sqrt(2) / 2, "3*pi/4", ["-pi/4", "pi/4", "pi/2"]),
    ("acos", "-sqrt(3)/2", -sqrt(3) / 2, "5*pi/6", ["-pi/6", "pi/6", "pi/2"]),
    ("acos", "-1", Rational(-1), "pi", ["-pi/2", "0", "pi/2"]),
    ("atan", "-1", Rational(-1), "-pi/4", ["pi/4", "pi/2", "-pi/2"]),
    ("atan", "-sqrt(3)", -sqrt(3), "-pi/3", ["pi/3", "pi/6", "-pi/6"]),
    ("atan", "-sqrt(3)/3", -sqrt(3) / 3, "-pi/6", ["pi/6", "pi/3", "-pi/3"]),
]

_TRIPLES = [(3, 4, 5), (5, 12, 13), (8, 15, 17), (7, 24, 25), (20, 21, 29)]
_HARD_KINDS = ["sin_acos", "cos_asin", "tan_acos", "tan_asin"]


def _lookup_question(fn, display, input_expr, answer_str, distractors_list):
    """Build a lookup question with exact-value evaluation.

    Args:
        fn: function name string ("asin", "acos", "atan")
        display: display string for input (e.g. "1/2")
        input_expr: sympy expression for the input
        answer_str: answer as a string (e.g. "pi/6")
        distractors_list: list of distractor strings
    """
    # Map function names to sympy functions and display names
    fn_map = {"asin": asin, "acos": acos, "atan": atan}
    display_fn = {"asin": "arcsin", "acos": "arccos", "atan": "arctan"}[fn]

    # Question text (prose: learner-facing name)
    question_text = f"Evaluate {display_fn}({display})."

    # Verify answer symbolically: the answer string, when parsed, equals the function applied
    answer_expr = fn_map[fn](input_expr)
    # Parse the answer string as a sympy expression
    answer_parsed = sympify(answer_str)
    check_equivalent(answer_parsed, answer_expr, f"{fn}: symbolic answer check failed for {display}")

    # Independent numeric check: evaluate both numerically
    numeric_answer = answer_expr.evalf()
    numeric_parsed = answer_parsed.evalf()
    check(abs(float(numeric_answer) - float(numeric_parsed)) < 1e-9, f"{fn}: numeric answer check failed for {display}")

    # Verify distractors are distinct from answer by value
    for distractor_str in distractors_list:
        distractor_parsed = sympify(distractor_str)
        check_distinct(
            answer_parsed, distractor_parsed, msg=f"{fn}: distractor {distractor_str} equals answer for {display}"
        )

    # Use _typed for answer and distractors (use asin, acos, atan format)
    candidates = distractors_list  # Already in correct format (asin/acos/atan)
    distractors = pick_distractors(answer_str, candidates)

    return {
        "text": question_text,
        "latex": question_text,
        "answer": answer_str,
        "answer_text": _typed(answer_parsed),
        "answer_latex": latex_answer(answer_parsed),
        "distractors": [_typed(sympify(d)) for d in distractors],
    }


def _composition_question(a, b, c, kind):
    """Build a Pythagorean-triple composition question.

    Args:
        a, b, c: legs a, b and hypotenuse c (a^2 + b^2 = c^2)
        kind: one of "sin_acos", "cos_asin", "tan_acos", "tan_asin"
    """
    # Verify the triple
    check(a**2 + b**2 == c**2, f"Invalid triple: {a}^2 + {b}^2 != {c}^2")

    # Determine the question and answer based on kind
    if kind == "sin_acos":
        # sin(acos(a/c)) = b/c
        question_text = f"Given cos(theta) = {a}/{c}, with theta in the range of arccos, find sin(theta)."
        answer = Rational(b, c)
        # Verify via sympy
        sympy_answer = sin(acos(Rational(a, c)))
        check_equivalent(sympy_answer, answer, f"sin_acos: sympy verification failed for triple ({a}, {b}, {c})")

    elif kind == "cos_asin":
        # cos(asin(a/c)) = b/c
        question_text = f"Given sin(theta) = {a}/{c}, with theta in the range of arcsin, find cos(theta)."
        answer = Rational(b, c)
        sympy_answer = cos(asin(Rational(a, c)))
        check_equivalent(sympy_answer, answer, f"cos_asin: sympy verification failed for triple ({a}, {b}, {c})")

    elif kind == "tan_acos":
        # tan(acos(a/c)) = b/a
        question_text = f"Given cos(theta) = {a}/{c}, with theta in the range of arccos, find tan(theta)."
        answer = Rational(b, a)
        sympy_answer = tan(acos(Rational(a, c)))
        check_equivalent(sympy_answer, answer, f"tan_acos: sympy verification failed for triple ({a}, {b}, {c})")

    elif kind == "tan_asin":
        # tan(asin(a/c)) = a/b
        question_text = f"Given sin(theta) = {a}/{c}, with theta in the range of arcsin, find tan(theta)."
        answer = Rational(a, b)
        sympy_answer = tan(asin(Rational(a, c)))
        check_equivalent(sympy_answer, answer, f"tan_asin: sympy verification failed for triple ({a}, {b}, {c})")

    # Build distractors based on kind
    if kind in ["sin_acos", "cos_asin"]:
        # Wrong candidates: a/c (original ratio), b/a (tan instead of sin/cos), a/b (reciprocal)
        candidates = [Rational(a, c), Rational(b, a), Rational(a, b)]
    elif kind == "tan_acos":
        # Wrong candidates: a/c, b/c, a/b
        candidates = [Rational(a, c), Rational(b, c), Rational(a, b)]
    elif kind == "tan_asin":
        # Wrong candidates: a/c, b/c, b/a
        candidates = [Rational(a, c), Rational(b, c), Rational(b, a)]

    # Verify all wrong candidates differ from answer
    for wrong in candidates:
        check(wrong != answer, f"{kind}: wrong candidate {wrong} equals answer for triple ({a}, {b}, {c})")

    # Convert to strings for pick_distractors
    answer_str = str(answer)
    candidates_str = [str(c) for c in candidates]
    distractors = pick_distractors(answer_str, candidates_str)

    return {
        "text": question_text,
        "latex": question_text,
        "answer": answer_str,
        "answer_text": _typed(answer),
        "answer_latex": latex_answer(answer),
        "distractors": [_typed(sympify(d)) for d in distractors],
    }


@generator(
    "find_and_use_the_inverse_trigonometric_functions_arcsine",
    level=11,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def find_and_use_the_inverse_trigonometric_functions_arcsine(rng, tier="easy"):
    """
    Evaluate and use inverse trigonometric functions via exact-value lookup or composition.

    easy: direct exact-value evaluation of asin/acos/atan for first-quadrant/non-negative inputs
    medium: exact-value evaluation for negative inputs, testing principal-branch awareness
    hard: composition via Pythagorean triples (trig-of-inverse-trig evaluation)
    """

    if tier == "easy":
        row = rng.choice(_EASY_ROWS)
        fn, display, input_expr, answer_str, distractors_list = row
        return _lookup_question(fn, display, input_expr, answer_str, distractors_list)

    elif tier == "medium":
        row = rng.choice(_MEDIUM_ROWS)
        fn, display, input_expr, answer_str, distractors_list = row
        return _lookup_question(fn, display, input_expr, answer_str, distractors_list)

    elif tier == "hard":
        triple = rng.choice(_TRIPLES)
        kind = rng.choice(_HARD_KINDS)
        a, b, c = triple
        return _composition_question(a, b, c, kind)
