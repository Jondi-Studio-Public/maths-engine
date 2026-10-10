"""Trigonometric equation solver: find all solutions of f(a(x-b))=c.

This is a single senior skill (ACMSM042) with three difficulty tiers:
- easy: direct reference-angle/CAST lookup (a=1, b=0)
- medium: adds substitution/shift (a=1, b=randint(10,80))
- hard: adds period-counting/multiple periods (a=2, b=0)

Uses the MCQ grader because a solution set (a collection of values) fits no other
scalar_expression grader kind naturally.
"""

from __future__ import annotations

import math
from fractions import Fraction

from ..core import check, generator, latex_answer, pick_distractors, VerificationError

# Reference angles and their exact string and float values
ANGLES = [30, 45, 60]
SIN_STR = {30: "1/2", 45: "sqrt(2)/2", 60: "sqrt(3)/2"}
COS_STR = {30: "sqrt(3)/2", 45: "sqrt(2)/2", 60: "1/2"}
TAN_STR = {30: "sqrt(3)/3", 45: "1", 60: "sqrt(3)"}

SIN_FLOAT = {30: 0.5, 45: math.sqrt(2) / 2, 60: math.sqrt(3) / 2}
COS_FLOAT = {30: math.sqrt(3) / 2, 45: math.sqrt(2) / 2, 60: 0.5}
TAN_FLOAT = {30: 1 / math.sqrt(3), 45: 1.0, 60: math.sqrt(3)}

FLOAT_FN = {"sin": math.sin, "cos": math.cos, "tan": math.tan}
STR_FN = {"sin": SIN_STR, "cos": COS_STR, "tan": TAN_STR}
VAL_FN = {"sin": SIN_FLOAT, "cos": COS_FLOAT, "tan": TAN_FLOAT}


def _base_set(func: str, theta: int, sign: int) -> set[Fraction]:
    """The (always exactly 2) solutions of func(x) = sign*func(theta) in
    degrees, for x in a 360-degree window. Uses the reference-angle/CAST method."""
    t = Fraction(theta)
    if func == "sin":
        return {t, 180 - t} if sign > 0 else {180 + t, 360 - t}
    if func == "cos":
        return {t, 360 - t} if sign > 0 else {180 - t, 180 + t}
    if func == "tan":
        return {t, 180 + t} if sign > 0 else {180 - t, 360 - t}
    raise ValueError(func)


def _fmt_deg(v: Fraction) -> str:
    """Format a degree value: integer if denominator is 1, else float to 1dp."""
    return str(v.numerator) if v.denominator == 1 else f"{float(v):.1f}"


def _format_solution_set(values: set[Fraction]) -> str:
    """Format a solution set as a readable string with degree symbols."""
    parts = [f"{_fmt_deg(v)}°" for v in sorted(values)]
    if len(parts) == 1:
        return f"x = {parts[0]}"
    if len(parts) == 2:
        return f"x = {parts[0]} and {parts[1]}"
    return f"x = {parts[0]}, {parts[1]}, {parts[2]} and {parts[3]}"


def _brute_force_solutions(func: str, a: int, b: int, c_float: float, tol: float = 1e-6) -> list[float]:
    """Independent completeness check: brute-force scan of all x in [0, 360)
    at 0.25° granularity, snapped to the 0.25 grid."""
    hits = []
    x = 0.0
    while x < 360.0:
        # Evaluate f(a(x - b))
        angle_rad = math.radians(a * (x - b))
        if abs(FLOAT_FN[func](angle_rad) - c_float) < tol:
            hits.append(round(x * 4) / 4)  # snap to 0.25 grid
        x += 0.25
    return hits


def _check_complete(func: str, a: int, b: int, c_float: float, solutions: set[Fraction]) -> None:
    """Verify that constructed solutions are complete and correct by comparing
    against brute-force scan. Both sides are on the 0.25 grid."""
    constructed = sorted(round(float(v) * 4) / 4 for v in solutions)
    brute_force = sorted(_brute_force_solutions(func, a, b, c_float))
    check(
        constructed == brute_force,
        f"Constructed {constructed} != brute-force {brute_force} for {func}({a}(x-{b}))={c_float}",
    )


@generator(
    "trig_equation_all_solutions",
    level=1,
    presets={
        "easy": dict(difficulty_tier="easy"),
        "medium": dict(difficulty_tier="medium"),
        "hard": dict(difficulty_tier="hard"),
    },
)
def trig_equation_all_solutions(rng, difficulty_tier="easy"):
    """Find all solutions of f(a(x-b)) = c where f in {sin, cos, tan},
    using the reference-angle/CAST method over a bounded 0 <= x < 360 degree domain."""

    # Pick random function, reference angle, and sign for all tiers
    func = rng.choice(["sin", "cos", "tan"])
    theta = rng.choice(ANGLES)
    sign = rng.choice([1, -1])

    # Build the RHS value string and float
    c_str = STR_FN[func][theta]
    if sign < 0:
        c_str = "-" + c_str
    c_float = VAL_FN[func][theta] * sign

    # Difficulty-specific construction
    if difficulty_tier == "easy":
        # Direct lookup: a=1, b=0
        a, b = 1, 0
        base = _base_set(func, theta, sign)
        solutions = base
        text = f"Find all solutions of {func}(x) = {c_str} for 0 <= x < 360, where x is in degrees."

    elif difficulty_tier == "medium":
        # Add substitution: a=1, b=randint(10,80)
        a = 1
        b = rng.randint(10, 80)
        base = _base_set(func, theta, sign)
        # Wrap each solution to [0, 360) by adding b, then reduce mod 360
        solutions = set()
        for v in base:
            v2 = v + b
            if v2 >= 360:
                v2 = v2 - 360
            solutions.add(v2)
        text = f"Find all solutions of {func}(x - {b}) = {c_str} for 0 <= x < 360, where x is in degrees."

    else:  # hard
        # Add period-counting: a=2, b=0
        a, b = 2, 0
        base = _base_set(func, theta, sign)
        # For a=2, the period is compressed, so we need solutions from two periods
        full = base | {v + 360 for v in base}  # 4 elements in u-space
        # Apply x = u/2 transformation
        solutions = {v / 2 for v in full}  # 4 elements (some may be x.5)
        text = f"Find all solutions of {func}(2x) = {c_str} for 0 <= x < 360, where x is in degrees."

    # Verify and format the correct answer
    _check_complete(func, a, b, c_float, solutions)
    answer_text = _format_solution_set(solutions)

    # Build distractors
    # 1. wrong_quadrant_cast: apply the *opposite* sign through the same tier's construction
    if difficulty_tier == "easy":
        wrong_base = _base_set(func, theta, -sign)
        wrong = wrong_base
    elif difficulty_tier == "medium":
        wrong_base = _base_set(func, theta, -sign)
        wrong = set()
        for v in wrong_base:
            v2 = v + b
            if v2 >= 360:
                v2 = v2 - 360
            wrong.add(v2)
    else:  # hard
        wrong_base = _base_set(func, theta, -sign)
        wrong_full = wrong_base | {v + 360 for v in wrong_base}
        wrong = {v / 2 for v in wrong_full}

    wrong_quadrant_cast = _format_solution_set(wrong)

    # 2. misses_solutions: depends on tier
    if difficulty_tier in ("easy", "medium"):
        # Only the minimum value (loses one solution)
        misses_solutions = _format_solution_set({min(solutions)})
    else:  # hard
        # The single-period set without the +360 extension (forgets a=2 doubles solutions)
        misses_solutions = _format_solution_set({v / 2 for v in base})

    # Ensure distractors are distinct from the answer and each other
    check(wrong_quadrant_cast != answer_text, "wrong_quadrant_cast equals answer_text")
    check(misses_solutions != answer_text, "misses_solutions equals answer_text")
    check(wrong_quadrant_cast != misses_solutions, "wrong_quadrant_cast equals misses_solutions")

    candidates = [wrong_quadrant_cast, misses_solutions]
    distractors = pick_distractors(answer_text, candidates, n=2)

    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer_text,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text, plain=True),
        working=[
            f"Using the reference-angle/CAST method to find all solutions of {func}({'' if a == 1 else f'{a}'}x{'-' + str(b) if b else ''}) = {c_str}.",
            f"Solutions: {answer_text}",
        ],
        distractors=distractors,
    )
