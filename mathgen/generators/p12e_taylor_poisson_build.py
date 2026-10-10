"""The Taylor linear-approximation + Poisson count-probability pair.

Two `university-core-v1` skills, one generator module, two structurally
unrelated operations:

- ``ucv1-taylor-linear-approximation`` (``single-variable-calculus``, topic
  ``p12e_taylor_linear_approx``): evaluate the first-order local (linear)
  approximation ``L(x) = f(a) + f'(a)*(x - a)`` of a supplied smooth function
  ``f`` near a stated point ``a``, at a stated ``x0``. This is a LOCAL,
  APPROXIMATE computation via differentiation -- never claimed exact for a
non-linear ``f`` -- distinct
  ``ucv1-ftc-polynomial`` ("Evaluate a definite polynomial integral by
  finding an antiderivative and applying endpoint subtraction"), which is a
  GLOBAL, EXACT computation over an interval via integration. No learner
  action overlaps: one integrates and subtracts two antiderivative values;
  the other evaluates a function and its derivative at one point and
  extrapolates linearly.

- ``ucv1-poisson-count-probability`` (``stochastic-processes``, topic
  ``p12e_poisson_count``): compute ``P(N(t)=k)`` for a Poisson process, kept
  inside ``answer: fraction`` (an exact rational grader) by supplying the
  irrational exponential factor ``e^{-lambda*t}`` as a GIVEN, stated exact
  rational value in the question text itself -- the learner is never asked
  to compute or know the exponential, only to assemble the rest of the
  Poisson pmf formula exactly. The hard tier is a ratio of two Poisson
  probabilities for the SAME lambda*t, which cancels the exponential factor
  exactly and needs no given value at all.
"""

from __future__ import annotations

from fractions import Fraction
from math import factorial

import sympy
from sympy import symbols

from ..core import (
    VerificationError,
    check,
    check_equivalent,
    check_pairwise_distinct,
    generator,
    latex_answer,
    paren_negative,
    phrase,
    pick_distractors,
    rational_text,
    round_half_up,
    rounds_unambiguously,
    stated_precision_string,
)

_x = symbols("x")

_MAX_DRAW_ATTEMPTS = 500


def _nonzero_randint(rng, lo: int, hi: int) -> int:
    return rng.choice([v for v in range(lo, hi + 1) if v != 0])


# ---------------------------------------------------------------------------
# ucv1-taylor-linear-approximation
#
# Shared closed-form value/derivative helpers for the two function families
# used (a quadratic polynomial for easy/medium/exam, k/x for hard). These are
# plain Fraction arithmetic -- the CONSTRUCTION route. The independent check
# route is `sympy.diff` on the symbolic expression, a structurally different
# code path (a real symbolic-differentiation engine vs hand power/quotient
# rule), exact, and fast (one `diff`+`subs` per question, not a slow
# `sympy.limit`, which p11e's own module docstring already measured as
# needlessly slow under the per-topic sweep load for a case with a closed
# form on hand).


def _quad_value(p: int, q: int, r: int, x) -> Fraction:
    x = Fraction(x)
    return p * x * x + q * x + r


def _quad_derivative_at(p: int, q: int, x) -> Fraction:
    x = Fraction(x)
    return 2 * p * x + q


def _quad_expr(p: int, q: int, r: int):
    return p * _x**2 + q * _x + r


def _rational_value(k: int, x) -> Fraction:
    x = Fraction(x)
    return Fraction(k, 1) / x


def _rational_derivative_at(k: int, x) -> Fraction:
    x = Fraction(x)
    return Fraction(-k, 1) / (x * x)


def _rational_expr(k: int):
    return k / _x


def _poly_display(expr) -> str:
    """Learner-facing rendering: ``**`` -> ``^``, bare ``*`` dropped (mirrors
    `p11e_calculus_build.py`'s own non-shared `_poly_display`; the two are
    two-line duplicates by design, same discipline as that module's own
    docstring documents for its own copy)."""
    return str(expr).replace("**", "^").replace("*", "")


_QUAD_P_CHOICES = (1, 2, 3, -1, -2, -3)
_QUAD_QR_LO, _QUAD_QR_HI = -6, 6
_QUAD_A_LO, _QUAD_A_HI = -4, 4

_EASY_TEMPLATES = (
    "f(x) = {poly}. Using the linear approximation L(x) = f(a) + f'(a)(x - a) at a = {a}, estimate f({x0}).",
    "f(x) = {poly}. Use the first-order local approximation L(x) = f(a) + f'(a)(x - a) at a = {a} to estimate f({x0}).",
    "f(x) = {poly}. Near a = {a}, use L(x) = f(a) + f'(a)(x - a) to estimate the value of f({x0}).",
)

_MEDIUM_TEMPLATES = (
    "f(x) = {poly}. Using the linear approximation L(x) = f(a) + f'(a)(x - a) "
    "at a = {a}, estimate f({x0}). Give your answer to 2 decimal places.",
    "f(x) = {poly}. Use the first-order local approximation L(x) = f(a) + "
    "f'(a)(x - a) at a = {a} to estimate f({x0}). Give your answer to 2 "
    "decimal places.",
    "f(x) = {poly}. Near a = {a}, use L(x) = f(a) + f'(a)(x - a) to estimate f({x0}) to 2 decimal places.",
)

_HARD_TEMPLATES = (
    "f(x) = {k}/x. Using the linear approximation L(x) = f(a) + f'(a)(x - a) "
    "at a = {a}, estimate f({x0}). Give your answer to 2 decimal places.",
    "f(x) = {k}/x. Use the first-order local approximation L(x) = f(a) + "
    "f'(a)(x - a) at a = {a} to estimate f({x0}). Give your answer to 2 "
    "decimal places.",
    "f(x) = {k}/x. Near a = {a}, use L(x) = f(a) + f'(a)(x - a) to estimate f({x0}) to 2 decimal places.",
)

_EXAM_TEMPLATES = (
    "A function f satisfies f({a}) = {fa} and its first-order local "
    "approximation near x = {a} gives f({x0}) approximately {Lx0}. Find f'({a}).",
    "Near x = {a}, a function f has f({a}) = {fa}. Its linear approximation estimates f({x0}) as {Lx0}. Find f'({a}).",
    "A function f has f({a}) = {fa}. Using the linear approximation L(x) = "
    "f(a) + f'(a)(x - a), f({x0}) is estimated as {Lx0}. Find f'({a}).",
)


def _display_int_or_2dp(value: Fraction, *, force_2dp: bool) -> str:
    """easy tier (integer offset): a plain integer, no trailing `.0` -- every displayed value (answer + every
    distractor) is confirmed integer by construction at that tier. medium/hard: every displayed value at a
    fixed 2 decimal places (`stated_precision_string`), so the answer and every distractor are formatted
    alike."""
    if not force_2dp:
        check(value.denominator == 1, f"expected an exact integer, got {value}")
        return str(int(value))
    return stated_precision_string(value, 2)


def _build_taylor_polynomial(rng, tier: str) -> dict:
    """easy: integer offset (x0 = a +/- 1), all values exact integers.
    medium: half-integer offset (x0 = a +/- 1/2), f degree 2 ONLY (not 2 or
    3 as the starting-point design allowed) -- found wrong: a degree-3
    f evaluated directly at x0 = a + 1/2 lands on EIGHTHS (denominator 8,
    e.g. 0.125) for the `evaluates-f-directly-at-x0` misconception, while the
    true answer L(x0) (linear in the offset) only ever needs a HALF
    (denominator <= 2) -- a precision mismatch that would force either an
    inconsistent per-option decimal count or a 3-dp display that misstates
    the medium tier's own stated precision. Restricting f to degree 2 keeps
    every value (answer and every distractor) on the quarter grid
    (denominator divides 4), so a single stated 2 dp is always EXACT (never
    a real rounding) for every option.
    """
    force_2dp = tier == "medium"
    for _attempt in range(_MAX_DRAW_ATTEMPTS):
        p = rng.choice(_QUAD_P_CHOICES)
        q = rng.randint(_QUAD_QR_LO, _QUAD_QR_HI)
        r = rng.randint(_QUAD_QR_LO, _QUAD_QR_HI)
        a = rng.randint(_QUAD_A_LO, _QUAD_A_HI)
        offset = rng.choice([1, -1]) if tier == "easy" else rng.choice([Fraction(1, 2), Fraction(-1, 2)])
        x0 = Fraction(a) + offset

        f_a = _quad_value(p, q, r, a)
        fprime_a = _quad_derivative_at(p, q, a)
        answer = f_a + fprime_a * offset

        # Independent check: sympy's own symbolic derivative, a structurally
        # different route from the hand power-rule arithmetic above.
        sym_fprime_a = sympy.diff(_quad_expr(p, q, r), _x).subs(_x, a)
        check_equivalent(fprime_a, sym_fprime_a, f"independent derivative check failed: p={p}, q={q}, a={a}")

        f_x0 = _quad_value(p, q, r, x0)
        fprime_x0 = _quad_derivative_at(p, q, x0)
        candidates = {
            "omits-derivative-term": f_a,
            "evaluates-f-directly-at-x0": f_x0,
            "sign-flip-on-offset": f_a - fprime_a * offset,
            "forgets-to-evaluate-derivative-at-a": f_a + fprime_x0 * offset,
        }
        try:
            check_pairwise_distinct(
                answer,
                *candidates.values(),
                msg=(f"misconception collision: tier={tier}, p={p}, q={q}, r={r}, a={a}, x0={x0}"),
            )
        except VerificationError:
            continue
        break
    else:  # pragma: no cover - defensive; the draw space clears this easily
        raise VerificationError("could not draw a collision-free taylor polynomial question")

    answer_text = _display_int_or_2dp(answer, force_2dp=force_2dp)
    poly_text = _poly_display(_quad_expr(p, q, r))
    x0_text = str(x0) if x0.denominator == 1 else f"{float(x0):g}"
    templates = _MEDIUM_TEMPLATES if tier == "medium" else _EASY_TEMPLATES
    question_text = phrase(rng, templates, poly=poly_text, a=a, x0=x0_text)
    distractor_strs = [_display_int_or_2dp(v, force_2dp=force_2dp) for v in candidates.values()]
    distractors = pick_distractors(answer_text, distractor_strs, n=4)
    return dict(
        text=question_text,
        latex=question_text,
        answer=int(answer) if answer.denominator == 1 and not force_2dp else float(answer),
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        working=[
            f"f({a}) = {f_a}, f'({a}) = {fprime_a}; L({x0_text}) = f({a}) + f'({a})({x0_text} - "
            f"{paren_negative(a)}) = {answer_text}.",
        ],
        distractors=distractors,
    )


_HARD_K_CHOICES = tuple(v for v in range(-15, 16) if v != 0)
_HARD_A_CHOICES = tuple(v for v in range(-6, 7) if v != 0)


def _build_taylor_rational(rng) -> dict:
    """hard tier: f(x) = k/x, a genuinely different function FAMILY, never a
    polynomial -- the learner must apply the SAME formula to an unfamiliar
    shape. `a` and `x0` are kept the SAME SIGN by construction (the offset
    always moves away from zero, so a and a + offset never straddle x = 0).

    Termination trap, resolved by stating the precision: `L(x0)` is generically a NON-terminating
    decimal for this family (e.g. a = 3 gives 25/18), so the question states "Give your answer to 2 decimal
    places" in its own text. The canonical AND every distractor are rounded to the same stated 2 dp via
    `round_half_up`, so no option is identifiable by being the only one that "looks exact" -- and
    `rounds_unambiguously` rejects any draw whose ANSWER sits on a rounding boundary."""
    for _attempt in range(_MAX_DRAW_ATTEMPTS):
        k = rng.choice(_HARD_K_CHOICES)
        a = rng.choice(_HARD_A_CHOICES)
        offset = 1 if a > 0 else -1  # away from zero -- never crosses x = 0
        x0 = a + offset

        f_a = _rational_value(k, a)
        fprime_a = _rational_derivative_at(k, a)
        answer = f_a + fprime_a * offset

        if not rounds_unambiguously(float(answer), 2):
            continue

        sym_fprime_a = sympy.diff(_rational_expr(k), _x).subs(_x, a)
        check_equivalent(fprime_a, sym_fprime_a, f"independent derivative check failed: k={k}, a={a}")

        f_x0 = _rational_value(k, x0)
        fprime_x0 = _rational_derivative_at(k, x0)
        candidates = {
            "omits-derivative-term": f_a,
            "evaluates-f-directly-at-x0": f_x0,
            "sign-flip-on-offset": f_a - fprime_a * offset,
            "forgets-to-evaluate-derivative-at-a": f_a + fprime_x0 * offset,
        }
        rounded_answer = round_half_up(answer, 2)
        rounded_candidates = {name: round_half_up(v, 2) for name, v in candidates.items()}
        try:
            check_pairwise_distinct(
                rounded_answer,
                *rounded_candidates.values(),
                msg=f"misconception collision after rounding: k={k}, a={a}, x0={x0}",
            )
        except VerificationError:
            continue
        break
    else:  # pragma: no cover - defensive; the draw space clears this easily
        raise VerificationError("could not draw a collision-free taylor rational question")

    answer_text = stated_precision_string(rounded_answer, 2)
    question_text = phrase(rng, _HARD_TEMPLATES, k=k, a=a, x0=x0)
    distractor_strs = [stated_precision_string(v, 2) for v in rounded_candidates.values()]
    distractors = pick_distractors(answer_text, distractor_strs, n=4)
    return dict(
        text=question_text,
        latex=question_text,
        answer=float(rounded_answer),
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        working=[
            f"f({a}) = {f_a}, f'({a}) = {fprime_a}; L({x0}) = f({a}) + f'({a})({x0} - "
            f"{paren_negative(a)}) = {answer} ≈ {answer_text} (2 d.p.).",
        ],
        distractors=distractors,
    )


def _build_taylor_exam(rng) -> dict:
    """exam tier: genuine escalation over hard -- the INVERSE direction.
    f(a), a, x0 and the already-computed L(x0) are stated; the learner finds
    f'(a) by rearranging L(x0) = f(a) + f'(a)(x0 - a), never differentiating
    a function directly. Reuses the degree-2 polynomial family (integer
    offset, so every value stays an exact integer -- no rounding question to
    settle twice)."""
    for _attempt in range(_MAX_DRAW_ATTEMPTS):
        p = rng.choice(_QUAD_P_CHOICES)
        q = rng.randint(_QUAD_QR_LO, _QUAD_QR_HI)
        r = rng.randint(_QUAD_QR_LO, _QUAD_QR_HI)
        a = rng.randint(_QUAD_A_LO, _QUAD_A_HI)
        offset = rng.choice([1, -1])
        x0 = a + offset

        f_a = _quad_value(p, q, r, a)
        fprime_a = _quad_derivative_at(p, q, a)
        answer = fprime_a
        l_x0 = f_a + fprime_a * offset

        sym_fprime_a = sympy.diff(_quad_expr(p, q, r), _x).subs(_x, a)
        check_equivalent(fprime_a, sym_fprime_a, f"independent derivative check failed: p={p}, q={q}, a={a}")
        # Independent check on the INVERSE arithmetic itself: recovering
        # f'(a) back out of L(x0) via (L(x0) - f(a)) / (x0 - a) must equal
        # the same fprime_a used to build L(x0) in the first place.
        recovered = (l_x0 - f_a) / Fraction(offset)
        check_equivalent(fprime_a, recovered, f"inverse recovery check failed: p={p}, q={q}, a={a}")

        candidates = {
            # (f(a) - L(x0)) / offset = -f'(a) exactly (an integer, since
            # L(x0) - f(a) = f'(a)*offset by construction) -- the true
            # derivative with the subtraction order swapped, never a raw
            # reciprocal (which would generally be a non-integer and break
            # "formatted alike" against the other, always-integer options).
            "swaps-subtraction-order": (f_a - l_x0) / Fraction(offset) if fprime_a != 0 else None,
            "ignores-offset-sign": (l_x0 - f_a) / abs(offset) if offset < 0 else None,
            "confuses-slope-with-target": l_x0,
        }
        candidates = {k2: v for k2, v in candidates.items() if v is not None}
        try:
            check_pairwise_distinct(
                answer,
                *candidates.values(),
                msg=f"misconception collision: p={p}, q={q}, r={r}, a={a}, x0={x0}",
            )
        except VerificationError:
            continue
        break
    else:  # pragma: no cover - defensive; the draw space clears this easily
        raise VerificationError("could not draw a collision-free taylor exam question")

    answer_text = _display_int_or_2dp(answer, force_2dp=False)
    question_text = phrase(rng, _EXAM_TEMPLATES, a=a, fa=int(f_a), x0=x0, Lx0=int(l_x0))
    distractor_strs = [_display_int_or_2dp(v, force_2dp=False) for v in candidates.values()]
    distractors = pick_distractors(answer_text, distractor_strs, n=4)
    return dict(
        text=question_text,
        latex=question_text,
        answer=int(answer),
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        working=[
            f"f'({a}) = (L({x0}) - f({a})) / ({x0} - {paren_negative(a)}) = ({int(l_x0)} - "
            f"{paren_negative(int(f_a))}) / {offset} = {answer_text}.",
        ],
        distractors=distractors,
    )


@generator(
    "p12e_taylor_linear_approx",
    level=19,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def taylor_linear_approx(rng, tier="easy"):
    """Approximate a supplied smooth function near a stated point using its
    value and derivative (the skill's own contract,
    verbatim).

    easy:   f a quadratic, x0 = a +/- 1 (integer offset) -- single
            substitution, all values exact integers.
    medium: f a quadratic, x0 = a +/- 1/2 (half-integer offset) -- carries a
            fraction through f'(a)*(x0-a), all values displayed at a
            (genuinely exact, never rounded) 2 d.p.
    hard:   f(x) = k/x (a stated non-zero rational k), a genuinely different
            function family from a polynomial; x0 = a +/- 1 kept the SAME
            SIGN as a. Non-terminating in general -- rounded to 2 d.p., the
            instruction stated in the question's own text.
    exam:   the INVERSE direction -- f(a), a, x0 and L(x0) are given; find
            f'(a). Genuine escalation over hard: a different unknown, never
            just a bigger number.
    """
    if tier in ("easy", "medium"):
        return _build_taylor_polynomial(rng, tier)
    if tier == "hard":
        return _build_taylor_rational(rng)
    if tier == "exam":
        return _build_taylor_exam(rng)
    raise ValueError(f"unknown tier {tier!r}")


# ---------------------------------------------------------------------------
# ucv1-poisson-count-probability
#
# `P(N(t)=k) = e^{-lambda*t} * (lambda*t)^k / k!` is irrational for
# essentially every real lambda*t, so it cannot be an exact `answer:
# fraction` on its own. The exponential factor is supplied as a GIVEN exact
# rational value in the question text itself (never computed by the
# learner), keyed by lambda*t -- a small, honestly-labelled set of
# APPROXIMATIONS (never claimed to be the true irrational value), phrased
# "take e^(-lambda t) to be ..." rather than a literal "e^-3 = ..." so the
# MEDIUM tier (which states lambda and t separately) does not leak lambda*t
# through the exponent's own digits.

_E_TABLE: dict[int, Fraction] = {
    1: Fraction(3, 8),
    2: Fraction(2, 15),
    3: Fraction(1, 20),
    4: Fraction(1, 55),
    5: Fraction(1, 150),
    6: Fraction(1, 400),
}

# medium tier: (lambda, t) pairs with t >= 2 (so lambda*t genuinely requires
# a multiplication step -- t = 1 would make lambda == lambda*t, collapsing
# the "state them separately" escalation back to the easy tier and also
# collapsing the uses-lambda-not-lambda-t misconception into the correct
# answer) and lambda*t landing in `_E_TABLE`'s domain.
_MEDIUM_LT_PAIRS = tuple((lam, t) for lam in range(1, 7) for t in range(2, 7) if lam * t in _E_TABLE)

_EASY_TEMPLATES_P = (
    "A Poisson process has lambda*t = {m}. Take e^(-lambda t) to be {e_str} for this question. Find P(N(t) = {k}).",
    "For a Poisson process, lambda*t = {m}. Using e^(-lambda t) = {e_str} "
    "(given, exact for this question), find P(N(t) = {k}).",
    "A Poisson process has lambda*t = {m}, and e^(-lambda t) is taken to be "
    "{e_str} for this question. Compute P(N(t) = {k}).",
)

_MEDIUM_TEMPLATES_P = (
    "A Poisson process has rate lambda = {lam} {event_word} per unit time. Over "
    "t = {t} units of time, take e^(-lambda t) to be {e_str} for this "
    "question. Find P(N(t) = {k}).",
    "For a Poisson process with rate lambda = {lam} and time t = {t}, using "
    "e^(-lambda t) = {e_str} (given, exact for this question), find "
    "P(N(t) = {k}).",
    "A Poisson process has rate lambda = {lam}. Over t = {t} units, e^(-lambda "
    "t) is taken to be {e_str} for this question. Compute P(N(t) = {k}).",
)

_HARD_TEMPLATES_P = (
    "A Poisson process has lambda*t = {m}. Find the exact value of P(N(t) = {k1}) divided by P(N(t) = {k2}).",
    "For a Poisson process with lambda*t = {m}, compute the ratio P(N(t) = {k1}) / P(N(t) = {k2}) exactly.",
    "A Poisson process has lambda*t = {m}. Find P(N(t) = {k1}) / "
    "P(N(t) = {k2}), the ratio of these two probabilities, exactly.",
)

_EXAM_TEMPLATES_P = (
    "A Poisson process has lambda*t = {m}. P(N(t) = {k1}) / P(N(t) = k2) = "
    "{ratio_str} exactly, for some whole number k2. Find k2.",
    "For a Poisson process with lambda*t = {m}, the ratio P(N(t) = {k1}) / "
    "P(N(t) = k2) equals {ratio_str} exactly, for some whole number k2. "
    "Find k2.",
    "A Poisson process has lambda*t = {m}. Given P(N(t) = {k1}) / "
    "P(N(t) = k2) = {ratio_str} exactly for some whole-number count k2, "
    "find k2.",
)


def _poisson_probability(e_val: Fraction, m: int, k: int) -> Fraction:
    return e_val * Fraction(m, 1) ** k / factorial(k)


def _build_poisson_easy_medium(rng, tier: str) -> dict:
    """easy: lambda*t stated directly. medium: lambda and t stated
    separately, requiring the learner to multiply them first -- a genuinely
    different first step, not a bigger easy-tier instance. Both drawn with
    k >= 1 (found wrong against the initially suggested `k in 0..4`:
    at k = 0, `omits-factorial` is numerically IDENTICAL to the correct
    answer (0! = 1), and the correct answer is a bare COPY of the given
    exponential value -- no assembly step at all. Restricted to k in 1..6
    (also widened from the initial 0..4, found wrong a second way: `k in
    1..4` with only `_E_TABLE`'s 6 `m` values gave just 16 distinct
    answers over 200 seeds, the pool-variety check's own top-answer share landing
    at 15% -- exactly its CONCENTRATED_TOP1 threshold; see the result
    file)."""
    for _attempt in range(_MAX_DRAW_ATTEMPTS):
        if tier == "easy":
            m = rng.choice(sorted(_E_TABLE))
            lam = t = None
        else:
            lam, t = rng.choice(_MEDIUM_LT_PAIRS)
            m = lam * t
        k = rng.randint(1, 6)
        e_val = _E_TABLE[m]
        answer = _poisson_probability(e_val, m, k)

        # Independent check: (lambda*t)**k and k! via a from-scratch loop,
        # never math.factorial/** -- a structurally different route from
        # the direct Fraction arithmetic used above.
        power = 1
        for _ in range(k):
            power *= m
        fact = 1
        for i in range(2, k + 1):
            fact *= i
        check_equivalent(answer, e_val * Fraction(power, fact), f"independent recompute failed: m={m}, k={k}")

        candidates = {}
        if k >= 2:
            candidates["omits-factorial"] = e_val * Fraction(m, 1) ** k
        off_by_one_k = k - 1 if k >= 2 else k + 1
        candidates["off-by-one-count"] = _poisson_probability(e_val, m, off_by_one_k)
        candidates["drops-exponential-factor"] = Fraction(m, 1) ** k / factorial(k)
        if tier == "medium":
            candidates["uses-lambda-not-lambda-t"] = _poisson_probability(e_val, lam, k)

        try:
            check_pairwise_distinct(
                answer,
                *candidates.values(),
                msg=f"misconception collision: tier={tier}, m={m}, k={k}",
            )
        except VerificationError:
            continue
        break
    else:  # pragma: no cover - defensive; the draw space clears this easily
        raise VerificationError("could not draw a collision-free poisson easy/medium question")

    answer_text = rational_text(answer)
    e_str = rational_text(e_val)
    if tier == "easy":
        question_text = phrase(rng, _EASY_TEMPLATES_P, m=m, e_str=e_str, k=k)
    else:
        event_word = "event" if lam == 1 else "events"
        question_text = phrase(
            rng,
            _MEDIUM_TEMPLATES_P,
            lam=lam,
            t=t,
            e_str=e_str,
            k=k,
            event_word=event_word,
        )
    distractor_strs = [rational_text(v) for v in candidates.values()]
    distractors = pick_distractors(answer_text, distractor_strs, n=4)
    return dict(
        text=question_text,
        latex=question_text,
        answer={"p": answer.numerator, "q": answer.denominator},
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        working=[
            f"P(N(t) = {k}) = e^(-lambda t) × (lambda t)^{k} / {k}! = {e_str} × {m}^{k} / {k}! = {answer_text}.",
        ],
        distractors=distractors,
    )


_HARD_M_CHOICES = tuple(range(1, 9))


def _build_poisson_hard(rng) -> dict:
    """Genuinely different reasoning from easy/medium: recognising which factor cancels, not repeating the
    substitution with different numbers."""
    for _attempt in range(_MAX_DRAW_ATTEMPTS):
        m = rng.choice(_HARD_M_CHOICES)
        k1 = rng.randint(0, 6)
        k2 = rng.choice([v for v in range(0, 7) if v != k1 and abs(v - k1) <= 4])
        answer = Fraction(m, 1) ** (k1 - k2) * Fraction(factorial(k2), factorial(k1))

        # Independent check: the DIRECT quotient of two independently
        # assembled P(N=k1)/P(N=k2) fractions (each built via the easy
        # tier's own formula, with e_val = 1 -- the SAME lambda*t so the
        # exponential is identical top and bottom and cancels in the
        # division regardless of what constant is used for it), rather
        # than the cancelled closed-form construction used above.
        p1 = _poisson_probability(Fraction(1), m, k1)
        p2 = _poisson_probability(Fraction(1), m, k2)
        check_equivalent(answer, p1 / p2, f"independent ratio recompute failed: m={m}, k1={k1}, k2={k2}")

        candidates = {"omits-factorial": Fraction(m, 1) ** (k1 - k2)}
        off = 1 if min(k1, k2) >= 1 else -1
        if k1 + off >= 0 and k2 + off >= 0:
            k1_off, k2_off = k1 + off, k2 + off
            candidates["off-by-one-count"] = Fraction(m, 1) ** (k1_off - k2_off) * Fraction(
                factorial(k2_off), factorial(k1_off)
            )
        candidates["inverted-ratio"] = 1 / answer

        try:
            check_pairwise_distinct(
                answer,
                *candidates.values(),
                msg=f"misconception collision: m={m}, k1={k1}, k2={k2}",
            )
        except VerificationError:
            continue
        # `rational_text` displays an integer-valued Fraction with no "/" at
        # all -- if the answer happens to be integer-valued while every
        # candidate is a genuine fraction (or vice versa), the answer is the
        # only option of its shape and pickable without doing the
        # arithmetic. Resample rather than ship it.
        answer_is_integer = answer.denominator == 1
        if not any((v.denominator == 1) == answer_is_integer for v in candidates.values()):
            continue
        break
    else:  # pragma: no cover - defensive; the draw space clears this easily
        raise VerificationError("could not draw a collision-free poisson hard question")

    answer_text = rational_text(answer)
    question_text = phrase(rng, _HARD_TEMPLATES_P, m=m, k1=k1, k2=k2)
    distractor_strs = [rational_text(v) for v in candidates.values()]
    distractors = pick_distractors(answer_text, distractor_strs, n=4)
    return dict(
        text=question_text,
        latex=question_text,
        answer={"p": answer.numerator, "q": answer.denominator},
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        working=[
            f"P(N(t) = {k1}) / P(N(t) = {k2}) = (lambda t)^({k1}-{k2}) × {k2}! / {k1}! = {answer_text} "
            "(the exponential factor is identical in numerator and denominator and cancels).",
        ],
        distractors=distractors,
    )


def _build_poisson_exam(rng) -> dict:
    """exam tier: a two-step composition of the medium tier's substitution idea and the hard tier's
    ratio-cancellation idea in one question -- lambda*t and k1 are stated along with the TARGET ratio
    P(N(t)=k1)/P(N(t)=k2), and the learner must recover the whole-number count k2 by inverting the same
    cancelled-exponential identity the hard tier uses forwards. Genuine escalation over hard: the unknown
    moves from the ratio itself to one of the counts inside it, so the learner must search/invert rather than
    substitute forwards."""
    search_range = range(0, 11)
    for _attempt in range(_MAX_DRAW_ATTEMPTS):
        m = rng.choice(_HARD_M_CHOICES)
        k1 = rng.randint(0, 7)
        k2 = rng.choice([v for v in range(0, 8) if v != k1])
        ratio = Fraction(m, 1) ** (k1 - k2) * Fraction(factorial(k2), factorial(k1))

        def _ratio_for(k2_candidate: int) -> Fraction:
            return Fraction(m, 1) ** (k1 - k2_candidate) * Fraction(factorial(k2_candidate), factorial(k1))

        matches = [v for v in search_range if _ratio_for(v) == ratio]
        if matches != [k2]:
            continue  # not unique over the search range -- resample

        candidates = {
            "confuses-k1-for-k2": Fraction(k1),
            "off-by-one-high": Fraction(k2 + 1) if k2 + 1 in search_range else None,
            "off-by-one-low": Fraction(k2 - 1) if k2 - 1 >= 0 else None,
        }
        candidates = {name: v for name, v in candidates.items() if v is not None}
        try:
            check_pairwise_distinct(
                Fraction(k2),
                *candidates.values(),
                msg=f"misconception collision: m={m}, k1={k1}, k2={k2}",
            )
        except VerificationError:
            continue
        break
    else:  # pragma: no cover - defensive; the draw space clears this easily
        raise VerificationError("could not draw a collision-free poisson exam question")

    answer_text = rational_text(Fraction(k2))
    ratio_str = rational_text(ratio)
    question_text = phrase(rng, _EXAM_TEMPLATES_P, m=m, k1=k1, ratio_str=ratio_str)
    distractor_strs = [rational_text(v) for v in candidates.values()]
    distractors = pick_distractors(answer_text, distractor_strs, n=4)
    return dict(
        text=question_text,
        latex=question_text,
        answer={"p": k2, "q": 1},
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        working=[
            f"P(N(t)={k1})/P(N(t)=k2) = (lambda t)^({k1}-k2) × k2!/{k1}! = {ratio_str} is solved by "
            f"trying whole-number k2 values; k2 = {answer_text} is the only one that matches over "
            "0 to 8.",
        ],
        distractors=distractors,
    )


@generator(
    "p12e_poisson_count",
    level=16,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def poisson_count(rng, tier="easy"):
    """Compute P(N(t)=k) for a supplied Poisson rate, time and count using an
    exact bounded parameter set (the skill's own contract).

    easy:   lambda*t and k (>= 1) stated directly, with the exact
            e^(-lambda t) given inline as a stated fraction.
    medium: lambda and t stated separately (t >= 2), requiring the learner
            to compute lambda*t themselves before substituting.
    hard:   a RATIO question -- P(N=k1)/P(N=k2) for the same lambda*t,
            which cancels the exponential factor exactly; no given
            exponential value at all.
    exam:   a two-step composition -- lambda*t, k1 and a TARGET ratio are
            given; the learner inverts the ratio identity to recover the
            whole-number count k2.
    """
    if tier in ("easy", "medium"):
        return _build_poisson_easy_medium(rng, tier)
    if tier == "hard":
        return _build_poisson_hard(rng)
    if tier == "exam":
        return _build_poisson_exam(rng)
    raise ValueError(f"unknown tier {tier!r}")
