"""One-parameter bifurcation threshold + parametric curve curvature.

Two skills, one module:

- ``p13e_bifurcation_sign_check`` (``ucv1-bifurcation-sign-check``, ``answer:
  number``) -- identify the parameter value ``r`` at which a one-parameter
  family of scalar equilibrium equations changes its number of real
  equilibria (a discriminant-zero threshold).
- ``p13e_parametric_curvature`` (``ucv1-parametric-curvature``, ``answer:
  number``) -- evaluate the curvature of a planar parametric curve at a
  stated parameter value.
"""

from __future__ import annotations

import math
from fractions import Fraction

from sympy import Abs as sympy_Abs, Eq, diff, nsimplify, simplify, solve, symbols

from ..core import (
    check,
    check_distinct,
    check_equivalent,
    fmt_signed,
    generator,
    join_terms,
    numeric_answer,
    phrase,
    pick_distractors,
    stated_precision_string,
)

_MAX_ATTEMPTS = 500

# ===========================================================================
# Skill 1: ucv1-bifurcation-sign-check
# ===========================================================================
#
# Family x' = a*x^2 + b*x + r (r the parameter, unless noted). Equilibria are
# the real roots of a*x^2 + b*x + r = 0; the count changes at the
# discriminant-zero threshold r* = b^2/(4a).

_EASY_A_POOL = (1, 2, 3, 4, 5)  # backward-constructed: a * n^2 stays an exact
# integer by construction.
_EASY_N_POOL = tuple(n for n in range(-6, 7) if n != 0)
_MEDIUM_A_POOL = (-5, -4, -2, -1, 1, 2, 4, 5)  # only prime factors 2, 5 -> 4a always terminates
_MEDIUM_B_POOL = tuple(b for b in range(-9, 10) if b != 0)
_MEDIUM_MAX_DP = 4
_HARD_S_POOL = tuple(range(2, 46))


def _divisors(n: int) -> list[int]:
    out = []
    for d in range(1, n + 1):
        if n % d == 0:
            out.append(d)
    return out


def _balanced_divisors(n: int, s: int) -> list[int]:
    """Divisors of ``n = s^2``, preferring ones close to ``s`` (balanced
    ``a, c``) once ``s`` is large enough that the extreme ``1``/``s^2`` split
    would render an unreasonably large coefficient. Below the threshold every
    divisor is offered, unchanged from before the pool-widening."""
    divs = _divisors(n)
    if s <= 15:
        return divs
    threshold = max(1, s // 4)
    balanced = [d for d in divs if threshold <= d <= n // threshold]
    return balanced or divs


def _independent_threshold(a, b) -> Fraction:
    """Re-derive r* from a, b ALONE via sympy.solve on the literal
    discriminant expression -- a symbolic solve, never the closed-form
    division construction reuses."""
    R = symbols("R")
    sol = solve(Eq(b**2 - 4 * a * R, 0), R)
    check(len(sol) == 1, f"expected exactly one threshold, got {sol}")
    v = sol[0]
    return Fraction(int(v.p), int(v.q))


def _independent_positive_root(a, c) -> int:
    """Re-derive the positive root of r^2 - 4*a*c = 0 from a, c ALONE via
    sympy.solve, filtered to the positive root."""
    R = symbols("R")
    sol = solve(Eq(R**2 - 4 * a * c, 0), R)
    positive = [s for s in sol if s.is_real and s > 0]
    check(len(positive) == 1, f"expected exactly one positive root, got {sol}")
    return int(positive[0])


def _on_grid(value: Fraction, dp: int) -> bool:
    return (value * (10**dp)).denominator == 1


def _terminating_dp(value: Fraction, max_dp: int) -> int | None:
    for d in range(0, max_dp + 1):
        if _on_grid(value, d):
            return d
    return None


def _offer(answer, value, candidates, label):
    """Append `value` (a Fraction) to `candidates` iff it differs from
    `answer` and from every candidate already offered."""
    if value == answer or any(value == existing for existing in candidates):
        return
    check_distinct(answer, value, msg=f"{label} collided with the answer")
    candidates.append(value)


def _family_expr(a, b) -> str:
    """ "a*x^2 + b*x" (numeric part only -- the parameter "+ r" is appended by
    the caller, since r's own coefficient is always +1 by the family's own
    definition)."""
    return join_terms([(a, "x^2"), (b, "x")])


_EASY_MEDIUM_PATTERNS = [
    "Consider the family x' = {expr} + r, where r is a parameter. Identify the value of r at "
    "which the number of real equilibria changes{precision_note}.",
    "For the family x' = {expr} + r (r a parameter), find the value of r at which the number of "
    "real equilibria of the equation changes{precision_note}.",
    "The equation x' = {expr} + r depends on the parameter r. Identify the value of r at which "
    "the number of real equilibria changes{precision_note}.",
]

_HARD_PATTERNS = [
    "Consider the family x' = {expr}, where r is a parameter (the coefficient of x). Identify "
    "the POSITIVE value of r at which the number of real equilibria changes.",
    "For the family x' = {expr} (r a parameter, the coefficient of x), find the POSITIVE value "
    "of r at which the discriminant of the equilibrium equation is zero.",
    "The equation x' = {expr}, where r is a parameter (the coefficient of x), has a real "
    "equilibrium count that depends on r. Identify the POSITIVE value of r at which that "
    "number changes.",
]

_BIF_EXAM_PATTERNS = [
    "For the family x' = {expr} + r, where r is a parameter, the bifurcation threshold r* "
    "satisfies the discriminant condition b^2 - 4ar = 0. Two formulas are proposed for r* in "
    "terms of a and b -- Formula A: r* = {formula_a}. Formula B: r* = {formula_b}. Exactly one "
    "of these two formulas is correct for every choice of a and b. For a = {a}, b = {b}, "
    "compute r*{precision_note}.",
    "Consider the family x' = {expr} + r, where r is a parameter; the bifurcation threshold r* "
    "satisfies b^2 - 4ar = 0. Two candidate formulas for r*, in terms of a and b, are proposed "
    "-- Formula A: r* = {formula_a}. Formula B: r* = {formula_b}. Only one formula holds for "
    "every choice of a and b. For a = {a}, b = {b}, compute "
    "r*{precision_note}.",
    "The family x' = {expr} + r depends on the parameter r, and its bifurcation threshold r* "
    "satisfies the discriminant condition b^2 - 4ar = 0. Two formulas for r*, in terms of a "
    "and b, are proposed -- Formula A: r* = {formula_a}. Formula B: r* = {formula_b}. Exactly "
    "one is correct for every a and b. For a = {a}, b = {b}, "
    "compute r*{precision_note}.",
]

_EXAM_CORRECT_FORMULA = "b^2 / (4a)"  # no literal "*" -- the project's own text
# quality gate (mathgen/text_quality.py's star_operator check) flags an
# asterisk used as a multiplication sign in learner-facing text; "4a" reads
# the way a student would write it, exactly like the family expression's own
# "4ar" in the surrounding sentence.
_EXAM_WRONG_FORMULAS = (
    ("b^2 / a", "drops-the-four"),
    ("-b^2 / (4a)", "sign-flip"),
    ("a^2 / (4b)", "swaps-coefficients"),
)


def _exam_formula_value(a: int, b: int, label: str) -> Fraction:
    if label == "drops-the-four":
        return Fraction(b * b, a)
    if label == "sign-flip":
        return -Fraction(b * b, 4 * a)
    if label == "swaps-coefficients":
        return Fraction(a * a, 4 * b)
    raise ValueError(f"unknown wrong-formula label {label!r}")


def _precision_note(dp: int) -> str:
    if dp == 0:
        return ""
    plural = "s" if dp != 1 else ""
    return f" (state your answer as a decimal, to {dp} decimal place{plural})"


def _p13e_bifurcation_easy(rng):
    for _attempt in range(_MAX_ATTEMPTS):
        a = rng.choice(_EASY_A_POOL)
        n = rng.choice(_EASY_N_POOL)
        b = 2 * a * n  # constructed backward so b^2/(4a) = a*n^2 exactly, by
        # construction -- no divisibility check or reject/redraw needed.
        answer = Fraction(b * b, 4 * a)
        check(answer.denominator == 1, "easy: threshold is not an exact integer")
        check(answer == a * n * n, "easy: threshold does not equal a*n^2 by construction")

        indep = _independent_threshold(a, b)
        check_equivalent(indep, answer, "easy: independent threshold check failed")

        candidates = []
        _offer(answer, Fraction(b * b, a), candidates, "drops-the-four")
        _offer(answer, -answer, candidates, "sign-flip")
        swaps = Fraction(a * a, 4 * b)
        if swaps.denominator == 1:
            _offer(answer, swaps, candidates, "swaps-coefficients")

        if not candidates:
            continue

        answer_int = int(answer)
        answer_text = str(answer_int)
        distractor_values = [str(int(c)) for c in candidates]
        distractors = pick_distractors(answer_text, distractor_values)
        if not distractors:
            continue

        expr = _family_expr(a, b)
        text = phrase(rng, _EASY_MEDIUM_PATTERNS, expr=expr, precision_note="")
        working = [
            f"Discriminant = b^2 - 4ar = ({b})^2 - 4({a})r; setting it to 0 gives "
            f"r* = ({b})^2 / (4 x {a}) = {answer_int}.",
        ]
        return dict(
            text=text,
            latex=text,
            answer=answer_int,
            answer_text=answer_text,
            working=working,
            distractors=distractors,
        )
    raise RuntimeError("easy: could not draw a usable question")


def _p13e_bifurcation_medium(rng):
    for _attempt in range(_MAX_ATTEMPTS):
        a = rng.choice(_MEDIUM_A_POOL)
        b = rng.choice(_MEDIUM_B_POOL)
        answer = Fraction(b * b, 4 * a)
        dp = _terminating_dp(answer, _MEDIUM_MAX_DP)
        if dp is None:
            continue

        indep = _independent_threshold(a, b)
        check_equivalent(indep, answer, "medium: independent threshold check failed")

        candidates = []
        for value, label in (
            (Fraction(b * b, a), "drops-the-four"),
            (-answer, "sign-flip"),
            (Fraction(a * a, 4 * b), "swaps-coefficients"),
        ):
            if _on_grid(value, dp):
                _offer(answer, value, candidates, label)

        if not candidates:
            continue

        answer_text = stated_precision_string(answer, dp)
        distractor_values = [stated_precision_string(c, dp) for c in candidates]
        distractors = pick_distractors(answer_text, distractor_values)
        if not distractors:
            continue

        expr = _family_expr(a, b)
        text = phrase(rng, _EASY_MEDIUM_PATTERNS, expr=expr, precision_note=_precision_note(dp))
        working = [
            f"Discriminant = b^2 - 4ar = ({b})^2 - 4({a})r; setting it to 0 gives "
            f"r* = ({b})^2 / (4 x {a}) = {answer_text}.",
        ]
        return dict(
            text=text,
            latex=text,
            answer=numeric_answer(answer),
            answer_text=answer_text,
            working=working,
            distractors=distractors,
        )
    raise RuntimeError("medium: could not draw a usable question")


def _p13e_bifurcation_hard(rng):
    for _attempt in range(_MAX_ATTEMPTS):
        s = rng.choice(_HARD_S_POOL)
        d = rng.choice(_balanced_divisors(s * s, s))
        sign = rng.choice((1, -1))
        a = sign * d
        c = sign * (s * s // d)
        check(a != 0 and c != 0, "hard: a or c drawn as zero")
        check(a * c == s * s, "hard: a*c does not equal s^2")

        answer = 2 * s
        indep = _independent_positive_root(a, c)
        check_equivalent(indep, answer, "hard: independent positive-root check failed")

        candidates = []
        _offer(answer, -2 * s, candidates, "wrong-root-sign")
        if not candidates:
            continue

        answer_text = str(answer)
        distractors = pick_distractors(answer_text, [str(v) for v in candidates])
        if not distractors:
            continue

        lead = fmt_signed(a, "x^2")
        sign_str = "+" if c >= 0 else "-"
        expr = f"{lead} + r*x {sign_str} {abs(c)}"
        text = phrase(rng, _HARD_PATTERNS, expr=expr)
        working = [
            f"Discriminant = r^2 - 4({a})({c}) = r^2 - 4({s * s}); setting it to 0 gives "
            f"r = +/- 2 x {s} = +/- {answer}. The positive value is r = {answer}.",
        ]
        return dict(
            text=text,
            latex=text,
            answer=answer,
            answer_text=answer_text,
            working=working,
            distractors=distractors,
        )
    raise RuntimeError("hard: could not draw a usable question")


def _p13e_bifurcation_exam(rng):
    """The learner's decision: of two general FORMULAS proposed for the
    bifurcation threshold r* (each a function of a and b alone, never a
    number), decide which one holds for every a and b, then apply the
    correct one to the drawn a, b -- the numeric answer never appears
    anywhere in the prompt."""
    a_sym, b_sym = symbols("a b")
    true_formula_expr = b_sym**2 / (4 * a_sym)
    wrong_formula_exprs = {
        "drops-the-four": b_sym**2 / a_sym,
        "sign-flip": -(b_sym**2) / (4 * a_sym),
        "swaps-coefficients": a_sym**2 / (4 * b_sym),
    }
    for _attempt in range(_MAX_ATTEMPTS):
        a = rng.choice(_MEDIUM_A_POOL)
        b = rng.choice(_MEDIUM_B_POOL)
        r_true = Fraction(b * b, 4 * a)
        dp = _terminating_dp(r_true, _MEDIUM_MAX_DP)
        if dp is None:
            continue

        indep = _independent_threshold(a, b)
        check_equivalent(indep, r_true, "exam: independent threshold check failed")

        wrong_order = list(_EXAM_WRONG_FORMULAS)
        rng.shuffle(wrong_order)
        chosen = None
        for formula_str, label in wrong_order:
            value = _exam_formula_value(a, b, label)
            if value != r_true and _on_grid(value, dp):
                chosen = (formula_str, label, value)
                break
        if chosen is None:
            continue  # every wrong formula either coincided with r_true, or fell
            # off the answer's decimal grid, for this a, b -- redraw
        wrong_formula_str, wrong_label, wrong_value = chosen

        # Prove the wrong formula is SYMBOLICALLY different in general, not
        # merely numerically different for this one draw.
        wrong_expr = wrong_formula_exprs[wrong_label]
        check(
            simplify(true_formula_expr - wrong_expr) != 0,
            f"exam: {wrong_label} is not symbolically distinct from the correct formula",
        )

        answer = r_true
        candidates = []
        _offer(answer, wrong_value, candidates, f"uses-{wrong_label}")
        for other_formula_str, other_label in _EXAM_WRONG_FORMULAS:
            if other_label == wrong_label:
                continue
            other_value = _exam_formula_value(a, b, other_label)
            if _on_grid(other_value, dp):
                _offer(answer, other_value, candidates, other_label)

        if not candidates:
            continue

        answer_text = stated_precision_string(answer, dp)
        distractor_values = [stated_precision_string(c, dp) for c in candidates]
        distractors = pick_distractors(answer_text, distractor_values)
        if not distractors:
            continue

        formula_a_str, formula_b_str = _EXAM_CORRECT_FORMULA, wrong_formula_str
        correct_slot = "A"
        if rng.random() < 0.5:
            formula_a_str, formula_b_str = formula_b_str, formula_a_str
            correct_slot = "B"

        expr = _family_expr(a, b)
        text = phrase(
            rng,
            _BIF_EXAM_PATTERNS,
            expr=expr,
            formula_a=formula_a_str,
            formula_b=formula_b_str,
            a=a,
            b=b,
            precision_note=_precision_note(dp),
        )
        working = [
            f"Formula A: r* = {formula_a_str}. Formula B: r* = {formula_b_str}.",
            f"The discriminant condition b^2 - 4ar = 0 gives r* = b^2 / (4a), so Formula "
            f"{correct_slot} is the correct one.",
            f"With a = {a}, b = {b}: r* = {answer_text}.",
        ]
        return dict(
            text=text,
            latex=text,
            answer=numeric_answer(answer),
            answer_text=answer_text,
            working=working,
            distractors=distractors,
        )
    raise RuntimeError("exam: could not draw a usable question")


@generator(
    "p13e_bifurcation_sign_check",
    level=17,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p13e_bifurcation_sign_check(rng, tier="easy"):
    if tier == "easy":
        return _p13e_bifurcation_easy(rng)
    if tier == "medium":
        return _p13e_bifurcation_medium(rng)
    if tier == "hard":
        return _p13e_bifurcation_hard(rng)
    if tier == "exam":
        return _p13e_bifurcation_exam(rng)
    raise ValueError(f"unknown tier {tier!r}")


# ===========================================================================
# Skill 2: ucv1-parametric-curvature
# ===========================================================================
#
# kappa(t0) = |x'(t0)y''(t0) - y'(t0)x''(t0)| / (x'(t0)^2 + y'(t0)^2)^(3/2).
# Every tier constructs x'(t0) = 3, y'(t0) = 4 exactly (a Pythagorean pair
# with m = 1), so the denominator is always exactly 5^3 = 125 and the answer
# is always an exact decimal at (at most) 3 d.p.

_EASY_PR_RANGE = tuple(range(-4, 5))  # kept for medium
_CURVATURE_EASY_PR_RANGE = tuple(range(-7, 8))  # Widened
_T0_POOL = (1, 2, 3, -1, -2, -3)
_HARD_COEF_POOL = tuple(v for v in range(-3, 4))
_HARD_LEAD_POOL = tuple(v for v in range(-3, 4) if v != 0)  # a3/b3, nonzero (genuinely cubic)


def _kappa_fraction(xp0: int, xpp0: int, yp0: int, ypp0: int) -> Fraction | None:
    """kappa magnitude from four derivative VALUES at a point, exact. Returns
    None if (xp0^2+yp0^2) is not a perfect square (no clean exact rational)."""
    num = abs(xp0 * ypp0 - yp0 * xpp0)
    s = xp0 * xp0 + yp0 * yp0
    if s == 0:
        return None
    m = math.isqrt(s)
    if m * m != s:
        return None
    return Fraction(num, m**3)


def _sympy_kappa(x_poly: dict, y_poly: dict, t0: int) -> Fraction:
    """Independent route: full symbolic differentiation via sympy.diff on the
    SAME polynomials (never the closed-form derivative-value construction
    above), substituting t0 and applying the curvature formula literally."""
    t = symbols("t")
    xt = sum(coef * t**power for power, coef in x_poly.items())
    yt = sum(coef * t**power for power, coef in y_poly.items())
    xp = diff(xt, t)
    xpp = diff(xp, t)
    yp = diff(yt, t)
    ypp = diff(yp, t)
    xp0 = xp.subs(t, t0)
    xpp0 = xpp.subs(t, t0)
    yp0 = yp.subs(t, t0)
    ypp0 = ypp.subs(t, t0)
    num = sympy_Abs(xp0 * ypp0 - yp0 * xpp0)
    denom = (xp0**2 + yp0**2) ** (Fraction(3, 2))
    value = nsimplify(num / denom)
    return Fraction(int(value.p), int(value.q))


def _offer_kappa(answer, value, candidates, label):
    if value is None or value == answer or any(value == existing for existing in candidates):
        return
    check_distinct(answer, value, msg=f"{label} collided with the answer")
    candidates.append(value)


_EASY_PATTERNS = [
    "A curve is given by x(t) = {x_expr}, y(t) = {y_expr}. Find the curvature kappa(0) of this "
    "curve at t = 0, to 3 decimal places.",
    "A planar curve satisfies x(t) = {x_expr} and y(t) = {y_expr}. Compute its curvature at "
    "t = 0, to 3 decimal places.",
    "For the parametric curve x(t) = {x_expr}, y(t) = {y_expr}, find kappa(0), to 3 decimal places.",
]

_MEDIUM_HARD_PATTERNS = [
    "A curve is given by x(t) = {x_expr}, y(t) = {y_expr}. Find the curvature kappa({t0}) of "
    "this curve at t = {t0}, to 3 decimal places.",
    "A planar curve satisfies x(t) = {x_expr} and y(t) = {y_expr}. Compute its curvature at "
    "t = {t0}, to 3 decimal places.",
    "For the parametric curve x(t) = {x_expr}, y(t) = {y_expr}, find kappa({t0}), to 3 decimal places.",
]

_EXAM_PATTERNS = [
    "A curve is given by x(t) = {x_expr}, y(t) = {y_expr}, where a is an unknown integer "
    "coefficient. At t = {t0}, the curvature is exactly {kappa_text}, and "
    "x'(t)y''(t) - y'(t)x''(t) evaluated at t = {t0} is {sign_word}. Find a.",
    "For the parametric curve x(t) = {x_expr} (with unknown coefficient a) and y(t) = "
    "{y_expr}, the curvature at t = {t0} is exactly {kappa_text}, and "
    "x'(t)y''(t) - y'(t)x''(t) is {sign_word} at t = {t0}. Determine a.",
    "A planar curve has x(t) = {x_expr} (a an unknown integer coefficient) and y(t) = "
    "{y_expr}. Its curvature at t = {t0} is exactly {kappa_text}, and "
    "x'(t)y''(t) - y'(t)x''(t) at t = {t0} is {sign_word}. Find a.",
]


def _p13e_curvature_easy(rng):
    for _attempt in range(_MAX_ATTEMPTS):
        p = rng.choice(_CURVATURE_EASY_PR_RANGE)
        r = rng.choice(_CURVATURE_EASY_PR_RANGE)
        if p == 0 and r == 0:
            continue  # trivial straight-line case -- allowed occasionally, not every draw

        xp0, xpp0 = 3, 2 * p
        yp0, ypp0 = 4, 2 * r
        answer = _kappa_fraction(xp0, xpp0, yp0, ypp0)
        check(answer is not None, "easy: denominator unexpectedly not a perfect square")
        check(answer.denominator in (1, 5, 25, 125), "easy: answer off the expected /125 grid")

        x_poly = {2: p, 1: 3}
        y_poly = {2: r, 1: 4}
        indep = _sympy_kappa(x_poly, y_poly, 0)
        check_equivalent(indep, answer, "easy: independent sympy curvature check failed")

        candidates = []
        signed_num = xp0 * ypp0 - yp0 * xpp0
        _offer_kappa(answer, Fraction(signed_num, 125), candidates, "drops-absolute-value")
        _offer_kappa(answer, Fraction(-signed_num, 125), candidates, "swaps-derivative-order")
        num_abs = abs(signed_num)
        for exp, label in ((1, "wrong-power-1"), (2, "wrong-power-2")):
            s_pow = 25**exp
            cand = Fraction(num_abs, s_pow)
            if _on_grid(cand, 3):
                _offer_kappa(answer, cand, candidates, label)

        if not candidates:
            continue

        answer_text = stated_precision_string(answer, 3)
        distractor_values = [stated_precision_string(c, 3) for c in candidates]
        distractors = pick_distractors(answer_text, distractor_values)
        if not distractors:
            continue

        x_expr = join_terms([(p, "t^2"), (3, "t")])
        y_expr = join_terms([(r, "t^2"), (4, "t")])
        text = phrase(rng, _EASY_PATTERNS, x_expr=x_expr, y_expr=y_expr)
        working = [
            f"x'(0) = 3, x''(0) = {xpp0}; y'(0) = 4, y''(0) = {ypp0}.",
            f"kappa(0) = |3 x ({ypp0}) - 4 x ({xpp0})| / 125 = {answer_text}.",
        ]
        return dict(
            text=text,
            latex=text,
            answer=numeric_answer(answer),
            answer_text=answer_text,
            working=working,
            distractors=distractors,
        )
    raise RuntimeError("easy: could not draw a usable question")


def _p13e_curvature_medium(rng):
    for _attempt in range(_MAX_ATTEMPTS):
        p = rng.choice(_EASY_PR_RANGE)
        r = rng.choice(_EASY_PR_RANGE)
        if p == 0 and r == 0:
            continue
        t0 = rng.choice(_T0_POOL)
        c1 = 3 - 2 * p * t0
        c2 = 4 - 2 * r * t0

        xp0, xpp0 = 3, 2 * p
        yp0, ypp0 = 4, 2 * r
        answer = _kappa_fraction(xp0, xpp0, yp0, ypp0)
        check(answer is not None, "medium: denominator unexpectedly not a perfect square")

        x_poly = {2: p, 1: c1}
        y_poly = {2: r, 1: c2}
        indep = _sympy_kappa(x_poly, y_poly, t0)
        check_equivalent(indep, answer, "medium: independent sympy curvature check failed")

        candidates = []
        signed_num = xp0 * ypp0 - yp0 * xpp0
        _offer_kappa(answer, Fraction(signed_num, 125), candidates, "drops-absolute-value")
        _offer_kappa(answer, Fraction(-signed_num, 125), candidates, "swaps-derivative-order")
        num_abs = abs(signed_num)
        for exp, label in ((1, "wrong-power-1"), (2, "wrong-power-2")):
            cand = Fraction(num_abs, 25**exp)
            if _on_grid(cand, 3):
                _offer_kappa(answer, cand, candidates, label)
        # evaluates-at-wrong-point: substitutes t=0 for x'(t0)/y'(t0), leaving
        # x''(t0)/y''(t0) unchanged (they are already t-independent constants
        # here). Only offered when the resulting value is on the same 3 d.p.
        # grid -- not always exercisable.
        wrong_point = _kappa_fraction(c1, xpp0, c2, ypp0)
        if wrong_point is not None and _on_grid(wrong_point, 3):
            _offer_kappa(answer, wrong_point, candidates, "evaluates-at-wrong-point")

        if not candidates:
            continue

        answer_text = stated_precision_string(answer, 3)
        distractor_values = [stated_precision_string(c, 3) for c in candidates]
        distractors = pick_distractors(answer_text, distractor_values)
        if not distractors:
            continue

        x_expr = join_terms([(p, "t^2"), (c1, "t")])
        y_expr = join_terms([(r, "t^2"), (c2, "t")])
        text = phrase(rng, _MEDIUM_HARD_PATTERNS, x_expr=x_expr, y_expr=y_expr, t0=t0)
        working = [
            f"x'(t) = {2 * p}t + ({c1}), so x'({t0}) = 3; y'(t) = {2 * r}t + ({c2}), so y'({t0}) = 4.",
            f"x''({t0}) = {xpp0}; y''({t0}) = {ypp0}.",
            f"kappa({t0}) = |3 x ({ypp0}) - 4 x ({xpp0})| / 125 = {answer_text}.",
        ]
        return dict(
            text=text,
            latex=text,
            answer=numeric_answer(answer),
            answer_text=answer_text,
            working=working,
            distractors=distractors,
        )
    raise RuntimeError("medium: could not draw a usable question")


def _p13e_curvature_hard(rng):
    for _attempt in range(_MAX_ATTEMPTS):
        a3 = rng.choice(_HARD_LEAD_POOL)
        a2 = rng.choice(_HARD_COEF_POOL)
        b3 = rng.choice(_HARD_LEAD_POOL)
        b2 = rng.choice(_HARD_COEF_POOL)
        t0 = rng.choice(_T0_POOL)
        a1 = 3 - 3 * a3 * t0 * t0 - 2 * a2 * t0
        b1 = 4 - 3 * b3 * t0 * t0 - 2 * b2 * t0

        xp0, xpp0 = 3, 6 * a3 * t0 + 2 * a2
        yp0, ypp0 = 4, 6 * b3 * t0 + 2 * b2
        answer = _kappa_fraction(xp0, xpp0, yp0, ypp0)
        check(answer is not None, "hard: denominator unexpectedly not a perfect square")

        x_poly = {3: a3, 2: a2, 1: a1}
        y_poly = {3: b3, 2: b2, 1: b1}
        indep = _sympy_kappa(x_poly, y_poly, t0)
        check_equivalent(indep, answer, "hard: independent sympy curvature check failed")

        candidates = []
        signed_num = xp0 * ypp0 - yp0 * xpp0
        _offer_kappa(answer, Fraction(signed_num, 125), candidates, "drops-absolute-value")
        _offer_kappa(answer, Fraction(-signed_num, 125), candidates, "swaps-derivative-order")
        num_abs = abs(signed_num)
        for exp, label in ((1, "wrong-power-1"), (2, "wrong-power-2")):
            cand = Fraction(num_abs, 25**exp)
            if _on_grid(cand, 3):
                _offer_kappa(answer, cand, candidates, label)
        # evaluates-at-wrong-point: substitutes t=0 everywhere (both x''/y''
        # AND x'/y', which here genuinely differ from the t0 values).
        wrong_xpp0 = 2 * a2
        wrong_ypp0 = 2 * b2
        wrong_point = _kappa_fraction(a1, wrong_xpp0, b1, wrong_ypp0)
        if wrong_point is not None and _on_grid(wrong_point, 3):
            _offer_kappa(answer, wrong_point, candidates, "evaluates-at-wrong-point")

        if not candidates:
            continue

        answer_text = stated_precision_string(answer, 3)
        distractor_values = [stated_precision_string(c, 3) for c in candidates]
        distractors = pick_distractors(answer_text, distractor_values)
        if not distractors:
            continue

        x_expr = join_terms([(a3, "t^3"), (a2, "t^2"), (a1, "t")])
        y_expr = join_terms([(b3, "t^3"), (b2, "t^2"), (b1, "t")])
        text = phrase(rng, _MEDIUM_HARD_PATTERNS, x_expr=x_expr, y_expr=y_expr, t0=t0)
        working = [
            f"x'({t0}) = 3, x''({t0}) = {xpp0}; y'({t0}) = 4, y''({t0}) = {ypp0}.",
            f"kappa({t0}) = |3 x ({ypp0}) - 4 x ({xpp0})| / 125 = {answer_text}.",
        ]
        return dict(
            text=text,
            latex=text,
            answer=numeric_answer(answer),
            answer_text=answer_text,
            working=working,
            distractors=distractors,
        )
    raise RuntimeError("hard: could not draw a usable question")


def _p13e_curvature_exam(rng):
    """Inverse direction: the full curve (x(t) with its linear coefficient
    blanked as "a"), t0, and the curvature value plus the explicit SIGN of
    x'(t0)y''(t0) - y'(t0)x''(t0) are given; the learner solves backward for
    the blanked coefficient. The sign is stated explicitly because kappa's
    own absolute value would otherwise leave the numerator's sign -- and so
    the unknown coefficient -- ambiguous (only one of +N/-N reconstructs a
    consistent a1 in general)."""
    for _attempt in range(_MAX_ATTEMPTS):
        a3 = rng.choice(_HARD_LEAD_POOL)
        a2 = rng.choice(_HARD_COEF_POOL)
        b3 = rng.choice(_HARD_LEAD_POOL)
        b2 = rng.choice(_HARD_COEF_POOL)
        t0 = rng.choice(_T0_POOL)
        a1 = 3 - 3 * a3 * t0 * t0 - 2 * a2 * t0
        b1 = 4 - 3 * b3 * t0 * t0 - 2 * b2 * t0

        xp0, xpp0 = 3, 6 * a3 * t0 + 2 * a2
        yp0, ypp0 = 4, 6 * b3 * t0 + 2 * b2
        if ypp0 == 0:
            continue  # y''(t0) = 0 -- the backward solve for a1 divides by this
        signed_num = xp0 * ypp0 - yp0 * xpp0
        if signed_num == 0:
            continue  # zero curvature -- sign statement would be meaningless

        answer_kappa = _kappa_fraction(xp0, xpp0, yp0, ypp0)
        check(answer_kappa is not None, "exam: denominator unexpectedly not a perfect square")

        x_poly = {3: a3, 2: a2, 1: a1}
        y_poly = {3: b3, 2: b2, 1: b1}
        indep = _sympy_kappa(x_poly, y_poly, t0)
        check_equivalent(indep, answer_kappa, "exam: independent sympy curvature check failed")

        # Independent re-derivation of a1 from ONLY the "public" facts
        # (curvature magnitude, the stated sign, t0, a2, a3, b1, b2, b3) --
        # never the closed-form construction's own a1 formula.
        known_ypp0 = ypp0  # does not involve a1
        known_xpp0 = xpp0  # does not involve a1
        n_reconstructed = signed_num  # sign is told exactly, magnitude from kappa*125
        base = 3 * a3 * t0 * t0 + 2 * a2 * t0  # x'(t0) - a1
        a1_reconstructed = Fraction(n_reconstructed - known_ypp0 * base + 4 * known_xpp0, known_ypp0)
        check(a1_reconstructed.denominator == 1, "exam: reconstructed a1 is not an integer")
        check_equivalent(int(a1_reconstructed), a1, "exam: reconstructed a1 does not match construction")

        answer = a1
        answer_text = str(answer)

        candidates = []
        # evaluates-at-wrong-point: assumes t0 = 0 when solving, i.e. treats
        # x'(0) = a1 directly (ignoring the true t0-dependent base term).
        wrong_a1_zero_point = 3
        _offer(answer, wrong_a1_zero_point, candidates, "evaluates-at-wrong-point")
        # wrong-numerator-sign: uses the OPPOSITE of the told sign.
        n_flipped = -signed_num
        a1_flipped = Fraction(n_flipped - known_ypp0 * base + 4 * known_xpp0, known_ypp0)
        if a1_flipped.denominator == 1:
            _offer(answer, int(a1_flipped), candidates, "wrong-numerator-sign")

        if not candidates:
            continue

        distractors = pick_distractors(answer_text, [str(c) for c in candidates])
        if not distractors:
            continue

        kappa_text = stated_precision_string(answer_kappa, 3)
        sign_word = "positive" if signed_num > 0 else "negative"
        x_expr = join_terms([(a3, "t^3"), (a2, "t^2")]) + " + a*t"
        y_expr = join_terms([(b3, "t^3"), (b2, "t^2"), (b1, "t")])
        text = phrase(
            rng,
            _EXAM_PATTERNS,
            x_expr=x_expr,
            y_expr=y_expr,
            t0=t0,
            kappa_text=kappa_text,
            sign_word=sign_word,
        )
        working = [
            f"x'({t0}) = 3({a3})({t0})^2 + 2({a2})({t0}) + a; x''({t0}) = {xpp0} (independent of a); "
            f"y'({t0}) = 4, y''({t0}) = {ypp0}.",
            f"x'({t0})y''({t0}) - y'({t0})x''({t0}) is {sign_word}, so it equals "
            f"{'+' if signed_num > 0 else '-'}({kappa_text} x 125) = {signed_num}.",
            f"Solving for a: a = {answer}.",
        ]
        return dict(
            text=text,
            latex=text,
            answer=answer,
            answer_text=answer_text,
            working=working,
            distractors=distractors,
        )
    raise RuntimeError("exam: could not draw a usable question")


@generator(
    "p13e_parametric_curvature",
    level=18,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p13e_parametric_curvature(rng, tier="easy"):
    if tier == "easy":
        return _p13e_curvature_easy(rng)
    if tier == "medium":
        return _p13e_curvature_medium(rng)
    if tier == "hard":
        return _p13e_curvature_hard(rng)
    if tier == "exam":
        return _p13e_curvature_exam(rng)
    raise ValueError(f"unknown tier {tier!r}")
