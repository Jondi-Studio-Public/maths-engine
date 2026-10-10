"""This test file does NOT import mathgen.generators.p6a_curves_position_build.
It derives the correct answer for each question from the rendered text alone.
"""

import re
import math
import pytest
import mathgen


def parse_vector_curve_easy_medium(text: str):
    """
    Parse 'easy' and 'medium' curve questions of the form:
    r(t) = (at, b/t) for t not equal to 0. Eliminating the parameter t
    gives a Cartesian equation of the form xy = k. Find k.

    Returns (a, b) coefficients.
    """
    # Pattern: (Nt, N/t) where N can be positive or negative
    match = re.search(r"r\(t\)\s*=\s*\(([^,]+),\s*([^)]+)\)", text)
    if not match:
        raise ValueError(f"Could not parse curve equation in: {text}")

    x_expr = match.group(1).strip()
    y_expr = match.group(2).strip()

    # x_expr should be like "7t" or "-8t" or "t" or "-t"
    x_match = re.match(r"(-?\d*)\s*\*?\s*t", x_expr)
    if not x_match:
        raise ValueError(f"Could not parse x coefficient from: {x_expr}")
    coef_str = x_match.group(1)
    if coef_str in ("", "+"):
        a = 1
    elif coef_str == "-":
        a = -1
    else:
        a = int(coef_str)

    # y_expr should be like "4/t" or "-6/t" or "-1/t"
    y_match = re.match(r"(-?\d*)\s*\*?\s*/\s*t", y_expr)
    if not y_match:
        raise ValueError(f"Could not parse y coefficient from: {y_expr}")
    coef_str = y_match.group(1)
    if coef_str in ("", "+"):
        b = 1
    elif coef_str == "-":
        b = -1
    else:
        b = int(coef_str)

    return a, b


def parse_vector_curve_hard(text: str):
    """
    Parse 'hard' curve questions of the form:
    r(t) = (at, b/t + c) for t not equal to 0. Find y when x = d.

    Returns (a, b, c, d) coefficients.
    """
    # Pattern: (Nt, N/t + N) and later "Find y when x = N"
    match = re.search(r"r\(t\)\s*=\s*\(([^,]+),\s*([^)]+)\)", text)
    if not match:
        raise ValueError(f"Could not parse curve equation in: {text}")

    x_expr = match.group(1).strip()
    y_expr = match.group(2).strip()

    # x_expr should be like "8t"
    x_match = re.match(r"(-?\d+)\s*\*?\s*t", x_expr)
    if not x_match:
        raise ValueError(f"Could not parse x coefficient from: {x_expr}")
    a = int(x_match.group(1))

    # y_expr should be like "8/t - 6" or "36/t + 5"
    # Match b/t (+ or -) c
    y_match = re.match(r"(-?\d+)\s*/\s*t\s*([-+])\s*(\d+)", y_expr)
    if not y_match:
        raise ValueError(f"Could not parse y expression from: {y_expr}")
    b = int(y_match.group(1))
    sign = y_match.group(2)
    c = int(y_match.group(3))
    if sign == "-":
        c = -c

    # Find "Find y when x = d"
    d_match = re.search(r"Find y when x = (-?\d+)", text)
    if not d_match:
        raise ValueError(f"Could not parse x value from: {text}")
    d = int(d_match.group(1))

    return a, b, c, d


def parse_position_vector_easy(text: str):
    """
    Parse 'easy' position vector questions of the form:
    r(t) = (at + b, ct + d), find position at t = N seconds.

    Returns (a, b, c, d, t_val) coefficients.
    """
    # Pattern: (at + b, ct + d) or (at - b, ct - d)
    match = re.search(r"r\(t\)\s*=\s*\(([^,]+),\s*([^)]+)\)", text)
    if not match:
        raise ValueError(f"Could not parse position vector in: {text}")

    x_expr = match.group(1).strip()
    y_expr = match.group(2).strip()

    # Parse x_expr: "at + b" or "at - b" or "at" or "t + b"
    # First try to match with constant
    x_match = re.match(r"(-?\d*)\s*\*?\s*t\s*([-+])\s*(\d+)", x_expr)
    if x_match:
        coef_str = x_match.group(1)
        if coef_str in ("", "+"):
            a = 1
        elif coef_str == "-":
            a = -1
        else:
            a = int(coef_str)
        x_sign = x_match.group(2)
        x_const = int(x_match.group(3))
        if x_sign == "-":
            x_const = -x_const
    else:
        # Try to match without constant (just "at" or "t")
        x_match = re.match(r"(-?\d*)\s*\*?\s*t", x_expr)
        if not x_match:
            raise ValueError(f"Could not parse x expression from: {x_expr}")
        coef_str = x_match.group(1)
        if coef_str in ("", "+"):
            a = 1
        elif coef_str == "-":
            a = -1
        else:
            a = int(coef_str)
        x_const = 0

    # Parse y_expr: "ct + d" or "ct - d" or "ct" or "t + d"
    y_match = re.match(r"(-?\d*)\s*\*?\s*t\s*([-+])\s*(\d+)", y_expr)
    if y_match:
        coef_str = y_match.group(1)
        if coef_str in ("", "+"):
            c = 1
        elif coef_str == "-":
            c = -1
        else:
            c = int(coef_str)
        y_sign = y_match.group(2)
        y_const = int(y_match.group(3))
        if y_sign == "-":
            y_const = -y_const
    else:
        # Try to match without constant
        y_match = re.match(r"(-?\d*)\s*\*?\s*t", y_expr)
        if not y_match:
            raise ValueError(f"Could not parse y expression from: {y_expr}")
        coef_str = y_match.group(1)
        if coef_str in ("", "+"):
            c = 1
        elif coef_str == "-":
            c = -1
        else:
            c = int(coef_str)
        y_const = 0

    # Find "at t = N seconds"
    t_match = re.search(r"t = (\d+)\s*seconds?", text)
    if not t_match:
        raise ValueError(f"Could not parse time value from: {text}")
    t_val = int(t_match.group(1))

    return a, x_const, c, y_const, t_val


def parse_position_vector_medium(text: str):
    """
    Parse 'medium' position vector questions of the form:
    r(t) = (at^2 + bt + c, dt^2 + et + f), find position at t = N seconds.

    Returns (a, b, c, d, e, f, t_val) coefficients.
    """
    match = re.search(r"r\(t\)\s*=\s*\(([^,]+),\s*([^)]+)\)", text)
    if not match:
        raise ValueError(f"Could not parse position vector in: {text}")

    x_expr = match.group(1).strip()
    y_expr = match.group(2).strip()

    # Parse x_expr: at^2 + bt + c (with various sign combinations)
    # This is tricky due to variable formatting
    a, b, c = parse_quadratic(x_expr)
    d, e, f = parse_quadratic(y_expr)

    # Find t value
    t_match = re.search(r"t = (\d+)\s*seconds?", text)
    if not t_match:
        raise ValueError(f"Could not parse time value from: {text}")
    t_val = int(t_match.group(1))

    return a, b, c, d, e, f, t_val


def parse_position_vector_hard(text: str):
    """
    Parse 'hard' position vector questions of the form:
    r(t) = (at^2 + bt + c, dt^2 + et + f), find displacement from t = t1 to t = t2.

    Returns (a, b, c, d, e, f, t1, t2) coefficients.
    """
    match = re.search(r"r\(t\)\s*=\s*\(([^,]+),\s*([^)]+)\)", text)
    if not match:
        raise ValueError(f"Could not parse position vector in: {text}")

    x_expr = match.group(1).strip()
    y_expr = match.group(2).strip()

    a, b, c = parse_quadratic(x_expr)
    d, e, f = parse_quadratic(y_expr)

    # Find "from t = t1 seconds to t = t2 seconds"
    times_match = re.search(r"from t = (\d+)\s*seconds? to t = (\d+)\s*seconds?", text)
    if not times_match:
        raise ValueError(f"Could not parse time values from: {text}")
    t1 = int(times_match.group(1))
    t2 = int(times_match.group(2))

    return a, b, c, d, e, f, t1, t2


def parse_quadratic(expr: str):
    """
    Parse a quadratic expression like "t^2 + 4t - 4" and extract (a, b, c).

    Handles various formatting: t^2, t**2, coefficients with/without *.
    Returns (a, b, c) where the quadratic is at^2 + bt + c.
    """
    # Remove spaces
    expr = expr.replace(" ", "")

    # Pattern to match t^2 or t**2 terms
    t2_match = re.search(r"([+-]?\d*)\*?t\^2", expr)
    if t2_match:
        coef_str = t2_match.group(1)
        if coef_str in ("", "+"):
            a = 1
        elif coef_str == "-":
            a = -1
        else:
            a = int(coef_str)
    else:
        a = 0

    # Pattern to match t (but not t^2) - use lookahead to avoid matching t^2
    # Need to find all t terms first, then remove t^2 matches
    expr_no_t2 = re.sub(r"[+-]?\d*\*?t\^2", "", expr)
    t_match = re.search(r"([+-]?\d*)\*?t(?!\^)", expr_no_t2)
    if t_match:
        coef_str = t_match.group(1)
        if coef_str in ("", "+"):
            b = 1
        elif coef_str == "-":
            b = -1
        else:
            b = int(coef_str)
    else:
        b = 0

    # Pattern to match constant (no t)
    # Remove all t^2 and t terms, then parse what's left
    const_expr = re.sub(r"[+-]?\d*\*?t\^2", "", expr)
    const_expr = re.sub(r"[+-]?\d*\*?t", "", const_expr)
    if const_expr:
        c = int(const_expr)
    else:
        c = 0

    return a, b, c


def parse_answer_number(answer_text: str):
    """Parse a numeric answer like '32' or '-42'."""
    match = re.search(r"(-?\d+)", answer_text)
    if not match:
        raise ValueError(f"Could not parse numeric answer from: {answer_text}")
    return int(match.group(1))


def parse_answer_pair(answer_text: str):
    """Parse a pair answer like 'x = 13, y = 15'."""
    x_match = re.search(r"x\s*=\s*(-?\d+)", answer_text)
    y_match = re.search(r"y\s*=\s*(-?\d+)", answer_text)
    if not x_match or not y_match:
        raise ValueError(f"Could not parse pair answer from: {answer_text}")
    return int(x_match.group(1)), int(y_match.group(1))


def compute_curve_easy_medium_answer(a: int, b: int) -> int:
    """
    For r(t) = (at, b/t), compute k where xy = k.
    From x = at and y = b/t, we get xy = at * (b/t) = ab.
    """
    return a * b


def compute_curve_hard_answer(a: int, b: int, c: int, d: int) -> int:
    """
    For r(t) = (at, b/t + c), find y when x = d.
    From x = at, we get t = d/a.
    Then y = b/(d/a) + c = b*a/d + c.
    """
    if d == 0 or a == 0:
        raise ValueError(f"Cannot compute with d={d} or a={a}")

    # t = d / a
    # y = b / t + c = b / (d/a) + c = b*a/d + c
    if (b * a) % d != 0:
        raise ValueError(f"Non-integer result: b*a={b * a}, d={d}")

    return (b * a) // d + c


def compute_position_easy_answer(a: int, b: int, c: int, d: int, t: int) -> tuple:
    """
    For r(t) = (at + b, ct + d), compute position at time t.
    x = at + b, y = ct + d.
    """
    x = a * t + b
    y = c * t + d
    return x, y


def compute_position_medium_answer(a: int, b: int, c: int, d: int, e: int, f: int, t: int) -> tuple:
    """
    For r(t) = (at^2 + bt + c, dt^2 + et + f), compute position at time t.
    """
    x = a * t * t + b * t + c
    y = d * t * t + e * t + f
    return x, y


def compute_position_hard_answer(a: int, b: int, c: int, d: int, e: int, f: int, t1: int, t2: int) -> tuple:
    """
    For r(t) = (at^2 + bt + c, dt^2 + et + f), compute displacement from t1 to t2.
    displacement = r(t2) - r(t1).
    """
    x1 = a * t1 * t1 + b * t1 + c
    y1 = d * t1 * t1 + e * t1 + f

    x2 = a * t2 * t2 + b * t2 + c
    y2 = d * t2 * t2 + e * t2 + f

    return x2 - x1, y2 - y1


# Coverage is asserted inside each oracle test, from a LOCAL counter.
#
# These counts used to live in two module-level dicts that `test_curves_oracle` and
# `test_position_oracle` mutated and `test_verify_totals` then read. Under pytest-xdist every worker
# is a separate PROCESS with its own import of this module, so that only worked while all three
# tests happened to land on the same worker. They did; then `test_verify_totals` read its own zeroed
# dicts and failed. The suite was green by luck.
#
# `xdist_group` is not the fix -- `--dist worksteal` ignores that marker. Cross-test state is the
# bug, so it is gone: each test now proves its own coverage, and the >= 360 total is checked as
# arithmetic over the constants below.
TIERS = ("easy", "medium", "hard")
SEEDS_PER_TIER = 70  # >= 60 required per tier
MIN_PER_TIER = 60
MIN_TOTAL = 360  # 60 per tier x 3 tiers x 2 topics


def test_curves_oracle():
    """Test curve questions independently."""
    checked = {tier: 0 for tier in TIERS}

    for tier in TIERS:
        for seed in range(SEEDS_PER_TIER):
            q = mathgen.make("use_vector_equations_of_curves_in_two_or", seed=seed, difficulty=tier)
            checked[tier] += 1

            if tier in ["easy", "medium"]:
                a, b = parse_vector_curve_easy_medium(q.text)
                computed_answer = compute_curve_easy_medium_answer(a, b)
            else:  # hard
                a, b, c, d = parse_vector_curve_hard(q.text)
                computed_answer = compute_curve_hard_answer(a, b, c, d)

            expected_answer = parse_answer_number(q.answer_text)

            assert computed_answer == expected_answer, (
                f"Seed {seed}, tier {tier}: computed {computed_answer}, expected {expected_answer}\nQuestion: {q.text}"
            )

    for tier in TIERS:
        assert checked[tier] >= MIN_PER_TIER, (
            f"Curves: only {checked[tier]} questions re-derived for tier {tier}, need >= {MIN_PER_TIER}"
        )


def test_position_oracle():
    """Test position vector questions independently."""
    checked = {tier: 0 for tier in TIERS}

    for tier in TIERS:
        for seed in range(SEEDS_PER_TIER):
            q = mathgen.make("consider_position_of_vectors_as_a_function_of", seed=seed, difficulty=tier)
            checked[tier] += 1

            if tier == "easy":
                a, b, c, d, t_val = parse_position_vector_easy(q.text)
                computed_answer = compute_position_easy_answer(a, b, c, d, t_val)
            elif tier == "medium":
                a, b, c, d, e, f, t_val = parse_position_vector_medium(q.text)
                computed_answer = compute_position_medium_answer(a, b, c, d, e, f, t_val)
            else:  # hard
                a, b, c, d, e, f, t1, t2 = parse_position_vector_hard(q.text)
                computed_answer = compute_position_hard_answer(a, b, c, d, e, f, t1, t2)

            expected_x, expected_y = parse_answer_pair(q.answer_text)
            computed_x, computed_y = computed_answer

            assert computed_x == expected_x and computed_y == expected_y, (
                f"Seed {seed}, tier {tier}: computed ({computed_x}, {computed_y}), expected ({expected_x}, {expected_y})\nQuestion: {q.text}"
            )

    for tier in TIERS:
        assert checked[tier] >= MIN_PER_TIER, (
            f"Position: only {checked[tier]} questions re-derived for tier {tier}, need >= {MIN_PER_TIER}"
        )


def test_verify_totals():
    """The >= 360 guarantee, as arithmetic over the constants the oracle tests use.

    This deliberately cannot read what those tests counted: under xdist they may run in
    a different PROCESS entirely, which is precisely the bug this file used to have.
    Each oracle test now asserts its own >= 60 per tier; this asserts the shared
    configuration still adds up, so SEEDS_PER_TIER cannot be lowered without failing.
    """
    assert SEEDS_PER_TIER >= MIN_PER_TIER, (
        f"SEEDS_PER_TIER is {SEEDS_PER_TIER}, below the {MIN_PER_TIER} required per tier"
    )
    total = SEEDS_PER_TIER * len(TIERS) * 2  # two topics: curves and position
    assert total >= MIN_TOTAL, f"Total questions re-derived {total} < {MIN_TOTAL}"
