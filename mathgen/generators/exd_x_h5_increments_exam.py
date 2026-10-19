"""Exam-tier sources for three first-principles / increment skills.

Topics registered here:

``exd_x_h5_increments_unknown_coefficient``
    ``y = k x^2 + b x`` with ``k`` unknown. One change in x and the resulting change in y are stated; the
    learner uses them to find ``k`` and then reports (δx, δy) for a second change.
``exd_x_h5_quotient_reciprocal_or_cubic``
    The difference quotient δy/δx at a stated point and step for ``f(x) = A/x + B`` or ``f(x) = A x^3 + B x`` --
    a function whose quotient does not come from the quadratic pattern.
``exd_x_h5_limit_two_points_unknown_coefficients``
    ``f(x) = a x^2 + b x`` with ``a`` and ``b`` unknown. The limiting value of the difference quotient is
    given at two points and asked for at a third.

Every question is built backwards from the chosen coefficients; each generator re-checks the key by an
independent route (a symbolic increment, or differentiating).
"""

from __future__ import annotations

from fractions import Fraction

from sympy import Rational, diff, expand, symbols

from ..core import (
    GRADER_TOLERANCE,
    check,
    check_distinct,
    check_equivalent,
    bracket_compound,
    generator,
    join_terms,
    paren_negative,
    latex_answer,
    phrase,
    pick_distractors,
    rational_text,
)

_x, _dx = symbols("x dx")


def _signed(value) -> str:
    return f" + {value}" if value > 0 else f" - {-value}"


# --------------------------------------------------------------------------------------------------
# (δx, δy) with an unknown coefficient.
# --------------------------------------------------------------------------------------------------

_INC_PROMPTS = (
    "A curve has equation y = kx^2 {lin}, where k is a constant. When x changes from {u1} to {u2}, "
    "y changes by δy = {first}. Find δx and δy when x changes from {w1} to {w2}.",
    "Given y = kx^2 {lin} with k constant: x moves from {u1} to {u2} and y changes by {first} (that is, δy = {first}). "
    "Now x moves from {w1} to {w2}. State δx and δy.",
    "The variables satisfy y = kx^2 {lin} for a constant k. When x goes from {u1} to {u2}, the increment "
    "in y is δy = {first}. What are δx and δy for a change in x from {w1} to {w2}?",
)


@generator("exd_x_h5_increments_unknown_coefficient", level=13, presets={"exam": dict(tier="exam")})
def exd_x_h5_increments_unknown_coefficient(rng, tier="exam"):
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _attempt in range(300):
        k = rng.choice([v for v in range(-5, 6) if v not in (0, 1)])
        b = rng.choice([v for v in range(-8, 9) if v != 0])
        u1, u2 = rng.sample(range(-4, 6), 2)
        w1, w2 = rng.sample(range(-5, 7), 2)
        if (u1 + u2) == 0 or (u1, u2) == (w1, w2):
            continue
        first = k * (u2**2 - u1**2) + b * (u2 - u1)
        dx = w2 - w1
        dy = k * (w2**2 - w1**2) + b * (w2 - w1)
        if dy == 0 or dy == dx or first == 0:
            continue

        def pair_text(x_inc, y_inc):
            return f"δx = {x_inc}, δy = {y_inc}"

        # Slips.
        k_nolin = Fraction(first, u2**2 - u1**2)  # forgets the linear term when finding k
        dy_nolin = k_nolin * (w2**2 - w1**2) + b * (w2 - w1)
        k_slope = Fraction(first, u2 - u1)  # reads the average rate of change as k
        dy_slope = k_slope * (w2**2 - w1**2) + b * (w2 - w1)
        cands = [
            (dx, -dy),
            (dy, dx),
            (-dx, dy),
        ]
        for wrong in (dy_nolin, dy_slope):
            if wrong.denominator == 1 and int(wrong) != dy:
                cands.append((dx, int(wrong)))
        texts = []
        for cx, cy in cands:
            if (cx, cy) != (dx, dy):
                texts.append(pair_text(cx, cy))
        if len(set(texts)) >= 3:
            break
    else:
        raise RuntimeError("could not draw an increments question with distinct distractors")

    general = expand((k * _x**2 + b * _x).subs(_x, _x + _dx) - (k * _x**2 + b * _x))
    check_equivalent(general.subs({_x: w1, _dx: dx}), dy, "independent increment check failed")
    check_equivalent(
        (k * _x**2 + b * _x).subs(_x, u2) - (k * _x**2 + b * _x).subs(_x, u1), first, "stated increment mismatch"
    )
    answer_text = pair_text(dx, dy)
    distractors = pick_distractors(answer_text, texts, n=3, tolerance=GRADER_TOLERANCE)
    check(len(distractors) >= 2, "expected at least two distractors")
    lin = f"{'+' if b > 0 else '-'} {abs(b) if abs(b) != 1 else ''}x"
    text = phrase(rng, _INC_PROMPTS, lin=lin, u1=u1, u2=u2, first=first, w1=w1, w2=w2)
    pn = paren_negative

    def sc(c, body):
        """A coefficient written before a bracket: 1 and -1 are not written out."""
        return f"({body})" if c == 1 else (f"-({body})" if c == -1 else f"{pn(c)}({body})")

    working = [
        f"From x = {u1} to x = {u2}: δy = k({pn(u2)}² - {pn(u1)}²) + {sc(b, f'{u2} - {pn(u1)}')} = {first}.",
        f"So {join_terms([(u2**2 - u1**2, 'k'), (b * (u2 - u1), '')])} = {first}, giving k = {k}.",
        f"δx = {w2} - {pn(w1)} = {dx}.",
        f"δy = {sc(k, f'{pn(w2)}² - {pn(w1)}²')} + {sc(b, f'{w2} - {pn(w1)}')} = {dy}.",
    ]
    return {
        "text": text,
        "latex": text,
        "answer": {"dx": dx, "dy": dy},
        "answer_text": answer_text,
        "answer_latex": rf"\delta x = {dx},\; \delta y = {dy}",
        "distractors": distractors,
        "working": working,
    }


# --------------------------------------------------------------------------------------------------
# δy/δx for a reciprocal or cubic function.
# --------------------------------------------------------------------------------------------------

_STEP_POOL = (Fraction(1, 2), Fraction(1, 3), Fraction(1, 4), Fraction(2), Fraction(-1, 2), Fraction(-1, 3))
_QUOT_PROMPTS = (
    "y = f(x) = {f}. At x = {p}, x changes by δx = {dx}. Find δy/δx for this change, giving your answer as a fraction.",
    "Let y = {f}. When x changes from {p} by δx = {dx}, what is the value of δy/δx? Give your answer as a fraction.",
    "For y = {f}, x changes from {p} by δx = {dx}. Calculate the difference quotient δy/δx as a fraction.",
)


@generator("exd_x_h5_quotient_reciprocal_or_cubic", level=13, presets={"exam": dict(tier="exam")})
def exd_x_h5_quotient_reciprocal_or_cubic(rng, tier="exam"):
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _attempt in range(300):
        kind = rng.choice(("reciprocal", "cubic"))
        p = rng.choice([v for v in range(-5, 6) if v != 0])
        dx = rng.choice(_STEP_POOL)
        a = rng.choice([v for v in range(-6, 7) if v not in (0, 1, -1)])
        b = rng.choice([v for v in range(-9, 10) if v != 0])
        if p + dx == 0:
            continue
        pf, q = Fraction(p), Fraction(p) + dx
        if kind == "reciprocal":

            def func(v):
                return Fraction(a) / v + b

            f_text = f"{a}/x{_signed(b)}"
            limit_value = Fraction(-a, p * p)
            f_sym = a / _x + b
        else:

            def func(v):
                return a * v**3 + b * v

            f_text = join_terms([(a, "x^3"), (b, "x")])
            limit_value = Fraction(3 * a * p * p + b)
            f_sym = a * _x**3 + b * _x
        delta_y = func(q) - func(pf)
        answer = delta_y / dx
        general = ((f_sym.subs(_x, _x + _dx) - f_sym) / _dx).subs({_x: p, _dx: Rational(dx.numerator, dx.denominator)})
        check_equivalent(general, Rational(answer.numerator, answer.denominator), "independent quotient check failed")
        if answer in (0, 1, -1):
            continue
        cands = [delta_y, 1 / answer, -answer, limit_value]
        if dx < 0:
            cands.append((func(pf - dx) - func(pf)) / (-dx))  # uses |δx| as the step
        keep = [c for c in cands if c != answer]
        if len({c for c in keep}) >= 3:
            break
    else:
        raise RuntimeError("could not draw a quotient question with distinct distractors")

    for c in keep:
        check_distinct(Rational(answer.numerator, answer.denominator), Rational(c.numerator, c.denominator))
    answer_text = rational_text(answer)
    distractors = pick_distractors(answer_text, [rational_text(c) for c in keep], n=3)
    check(len(distractors) >= 3, "expected three distractors")
    text = phrase(rng, _QUOT_PROMPTS, f=f_text, p=p, dx=rational_text(dx))
    new_x = f"{p} {'+' if dx > 0 else '-'} {rational_text(abs(dx))}"
    working = [
        f"f({p}) = {rational_text(func(pf))}.",
        f"f({new_x}) = f({rational_text(q)}) = {rational_text(func(q))}.",
        f"δy = {rational_text(func(q))} - {paren_negative(func(pf)) if func(pf).denominator == 1 else '(' + rational_text(func(pf)) + ')'} = {rational_text(delta_y)}.",
        f"δy/δx = {bracket_compound(rational_text(delta_y))} ÷ {bracket_compound(rational_text(dx))} = {answer_text}.",
    ]
    return {
        "text": text,
        "latex": text,
        "answer": {"p": answer.numerator, "q": answer.denominator},
        "answer_text": answer_text,
        "answer_latex": latex_answer(answer),
        "distractors": distractors,
        "working": working,
    }


# --------------------------------------------------------------------------------------------------
# Limit of the difference quotient from two points, asked at a third.
# --------------------------------------------------------------------------------------------------

_LIMIT_PROMPTS = (
    "For f(x) = ax^2 + bx, where a and b are constants, the difference quotient (f(x+h) - f(x))/h approaches "
    "{l1} as h approaches 0 when x = {p1}, and approaches {l2} when x = {p2}. "
    "What value does the difference quotient approach as h approaches 0 when x = {p3}?",
    "Two constants a and b define f(x) = ax^2 + bx. Its difference quotient (f(x+h) - f(x))/h has limit {l1} as h "
    "goes to 0 when x = {p1}, and limit {l2} when x = {p2}. What is the limit as h goes to 0 when x = {p3}?",
    "A function is f(x) = ax^2 + bx with constants a and b. As h tends to 0, (f(x+h) - f(x))/h tends to {l1} "
    "at x = {p1} and to {l2} at x = {p2}. Find the value it tends to at x = {p3}.",
)


@generator("exd_x_h5_limit_two_points_unknown_coefficients", level=13, presets={"exam": dict(tier="exam")})
def exd_x_h5_limit_two_points_unknown_coefficients(rng, tier="exam"):
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _attempt in range(300):
        a = rng.choice([v for v in range(-5, 6) if v != 0])
        b = rng.choice([v for v in range(-9, 10) if v != 0])
        p1, p2, p3 = rng.sample(range(-5, 6), 3)
        l1, l2 = 2 * a * p1 + b, 2 * a * p2 + b
        answer = 2 * a * p3 + b
        if answer == 0 or l1 == l2:
            continue
        cands = [
            -answer,
            2 * a * p3,  # drops b
            a * p3 + b,  # power rule without the factor 2
            2 * a * p3 - b,  # sign slip on b
            a * p3**2 + b * p3,  # the function value in place of the gradient
        ]
        keep = [c for c in cands if c != answer]
        if len(set(keep)) >= 3:
            break
    else:
        raise RuntimeError("could not draw a limit question with distinct distractors")

    f_sym = a * _x**2 + b * _x
    check_equivalent(diff(f_sym, _x).subs(_x, p1), l1, "stated limit at the first point is wrong")
    check_equivalent(diff(f_sym, _x).subs(_x, p2), l2, "stated limit at the second point is wrong")
    check_equivalent(diff(f_sym, _x).subs(_x, p3), answer, "key is wrong")
    for c in keep:
        check_distinct(answer, c)
    distractors = pick_distractors(str(answer), [str(c) for c in keep], n=3)
    check(len(distractors) == 3, "expected three distractors")
    text = phrase(rng, _LIMIT_PROMPTS, l1=l1, l2=l2, p1=p1, p2=p2, p3=p3)
    return {
        "text": text,
        "latex": text,
        "answer": answer,
        "answer_text": str(answer),
        "answer_latex": latex_answer(answer),
        "distractors": distractors,
    }
