"""Generates `p19f_difference_quotient_stated_h` questions: evaluate (f(x+h) - f(x))/h for a stated nonzero
integer h.

The stated-h difference quotient.

``senior-interpret-the-difference-quotient-frac-f-left-x`` (ACMMM077, topic
``p19f_difference_quotient_stated_h``) computes the difference quotient
(f(x+h)-f(x))/h from ONE fully stated, fixed, NONZERO integer h -- never a
sequence of shrinking h, never a limit. `h` is stated once in the prompt and
the prompt says explicitly that it is fixed and not taken to a limit.

**The one-sentence tests that separate this skill from its neighbours:**

- ``senior-examine-the-behaviour-of-the-difference-quotient-frac`` (ACMMM081,
  ``mathgen/generators/p9c_difference_quotient_build.py``, topic
  ``p9c_difference_quotient_behaviour``) shows a SEQUENCE of shrinking h and
  asks what value it approaches as h -> 0 -- a numeric TREND read, never a
  single stated h, never this skill's fixed-step computation.
- ``senior-define-the-derivative-f-left-x-right-as`` (ACMMM082,
  ``mathgen/generators/p5e_first_principles_build.py``) is ALGEBRAIC: it
  forms and simplifies the difference quotient as an EXPRESSION IN x AND h
  (``answer: expression``), with no specific point, and its hard tier takes
  the algebraic limit to give f'(x) as an expression in x alone. This skill
  never returns an expression in x/h -- always a plain number at one stated
  numeric point and one stated numeric h.
- ``senior-use-the-notation-frac-delta-y-delta-x`` (ACMMM079,
  ``mathgen/generators/p9c_difference_quotient_build.py``, topic
  ``p9c_delta_y_delta_x_notation``) is MECHANICALLY the same single-quotient-
  at-one-stated-point computation, but under Leibniz delta-notation (delta-x,
  delta-y) and an exact-fraction delta-x pool -- the ACMMM077 vs ACMMM079
  curriculum split is by NOTATION, not behaviour: this skill always says "h"
  and "the average rate of change of f over the interval [x, x+h]"; that
  skill always says "delta-x"/"delta-y" and never uses the word "h" at all.

The sharpest possible distractor for this skill is exactly ACMMM081's answer:
report 2*A*p+B (the h->0 limit) instead of 2*A*p+B+A*h (the actual stated-h
quotient) -- see ``used_h_as_limit`` below, always genuinely different
whenever h != 0.
"""

from __future__ import annotations

from sympy import diff, expand, symbols

from ..core import (
    check_distinct,
    check_equivalent,
    generator,
    latex_answer,
    pick_distractors,
)

_x, _h = symbols("x h")

_P_POOL = tuple(v for v in range(-6, 7) if v != 0)
_M_POOL = tuple(range(-4, 5))
_H_POSITIVE = (1, 2, 3)
_H_ANY_SIGN = (-3, -2, -1, 1, 2, 3)
_H_NEGATIVE = (-3, -2, -1)


def _prose(expr) -> str:
    """Learner-facing rendering: ``**`` -> ``^``, ``*`` -> the middle dot --
    matches `p9c_difference_quotient_build.py`'s own `_prose` helper exactly;
    not shared centrally because it is two lines and used by a handful of
    single-variable-polynomial generators only."""
    return str(expr).replace("**", "^").replace("*", "·")


@generator(
    "p19f_difference_quotient_stated_h",
    level=12,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p19f_difference_quotient_stated_h(rng, tier="easy"):
    """Compute (f(x+h)-f(x))/h for f(x) = A*x^2 + B*x + C at one stated
    integer point x and one stated, fixed, NONZERO integer h -- never a
    limit, never a sequence.

    easy:   B = 0, h drawn positive-only from {1, 2, 3}.
    medium: B nonzero, h drawn from either sign, {-3,-2,-1,1,2,3}.
    hard:   B nonzero, h drawn NEGATIVE-only from {-3,-2,-1}; the prompt
            states the interval explicitly as running from x = p to
            x = p+h, so the learner must recognise that a negative h makes
            the interval run leftward -- a genuine decision about direction,
            not a bigger number.
    exam:   same negative-h shape as hard, but the evaluation point itself
            needs a one-step derivation first: x is given as x = 2m - 1 for
            a stated integer m, so the learner must compute x before
            applying the quotient at all -- a shape neither easy, medium
            nor hard contains.

    Because (f(p+h)-f(p))/h = A*(2p+h) + B is an exact integer identity for
    integer A, B, p, h, the answer is a bare integer with no fraction or
    precision machinery needed.
    """
    a = rng.randrange(1, 6)
    c = rng.randrange(-9, 10)

    if tier == "exam":
        m = rng.choice(_M_POOL)
        p = 2 * m - 1
    else:
        p = rng.choice(_P_POOL)
        m = None

    if tier == "easy":
        h_pool = _H_POSITIVE
    elif tier == "medium":
        h_pool = _H_ANY_SIGN
    else:
        h_pool = _H_NEGATIVE

    b = 0
    h = h_pool[0]
    for _attempt in range(50):
        if tier != "easy":
            b_pool = [v for v in range(-9, 10) if v != 0]
            b = rng.choice(b_pool)
        h = rng.choice(h_pool)
        answer = a * (2 * p + h) + b
        # h == 1 makes the forgot-to-divide distractor (h * answer) equal
        # the answer itself; answer == 0 makes both that AND the sign-error
        # distractor (-answer) collide. Neither is a real constraint on the
        # skill -- just a construction detail to redraw past.
        if answer != 0 and h != 1:
            break
    answer = a * (2 * p + h) + b

    # Independent check: the general symbolic quotient expanded in x and h
    # (mirroring p9c_difference_quotient_build.py's own `dq` route), THEN
    # substituted with the stated numeric p and h directly -- never taking
    # h -> 0. A genuinely different route from the arithmetic identity above.
    f = a * _x**2 + b * _x + c
    dq_general = expand((f.subs(_x, _x + _h) - f) / _h)
    expected = dq_general.subs({_x: p, _h: h})
    check_equivalent(answer, expected, f"row1 mismatch: a={a}, b={b}, c={c}, p={p}, h={h}")

    f_prose = _prose(expand(f))
    if tier in ("easy", "medium"):
        question_text = (
            f"f(x) = {f_prose}. Using x = {p} and the fixed step h = {h} "
            f"(h is stated once and is never taken to a limit), compute "
            f"the difference quotient (f(x+h) - f(x))/h."
        )
    elif tier == "hard":
        question_text = (
            f"f(x) = {f_prose}. The interval runs from x = {p} to "
            f"x = {p + h} (h = {h} is negative, so the interval runs "
            f"leftward; h is fixed and never taken to a limit). Compute "
            f"the average rate of change of f over that interval using "
            f"the difference quotient (f(x+h) - f(x))/h at x = {p}."
        )
    else:  # exam
        question_text = (
            f"f(x) = {f_prose}. Let x be the value satisfying x = 2m - 1 "
            f"for m = {m}. The interval then runs from that x to x+h, "
            f"where h = {h} is negative and fixed, never taken to a limit. "
            f"Compute the average rate of change of f over that interval "
            f"using the difference quotient (f(x+h) - f(x))/h."
        )

    answer_text = str(answer)

    # Misconceptions.
    forgot_to_divide = h * answer
    sign_error = -answer
    used_h_as_limit = int(dq_general.subs({_x: p, _h: 0}))

    check_distinct(answer, forgot_to_divide, msg=f"forgot-to-divide-by-h collides: a={a},b={b},p={p},h={h}")
    check_distinct(answer, sign_error, msg=f"sign-error collides: a={a},b={b},p={p},h={h}")
    check_distinct(answer, used_h_as_limit, msg=f"used-h-as-if-limit-were-taken collides: a={a},b={b},p={p},h={h}")

    candidates_text = [str(forgot_to_divide), str(sign_error), str(used_h_as_limit)]
    distractors = pick_distractors(answer_text, candidates_text)

    return {
        "text": question_text,
        "latex": question_text,
        "answer": answer,
        "answer_text": answer_text,
        "answer_latex": latex_answer(answer),
        "distractors": distractors,
    }
