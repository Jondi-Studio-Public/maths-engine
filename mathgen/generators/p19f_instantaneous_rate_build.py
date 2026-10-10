"""Generates `p19f_instantaneous_rate` questions: the instantaneous rate of change f'(x) at a stated point.

The derivative as instantaneous rate.

``senior-interpret-the-derivative-as-the-instantaneous-rate-of`` (ACMMM084,
topic ``p19f_instantaneous_rate``) evaluates f'(x) at a stated numeric x --
the instantaneous rate of change of f at that point. Always a plain number,
never a symbolic expression, never a chord/secant slope.

**The one-sentence tests that separate this skill from its neighbours:**

- ``differentiate-polynomial`` produces a symbolic derivative EXPRESSION in
  x, never evaluated at a point. This skill always evaluates at a stated
  numeric x and always returns a number.
- ``second-derivative`` produces a symbolic f''(x), never evaluated at a
  point. This skill never differentiates twice.
- ``stationary-points`` solves f'(x) = 0 for x. This skill is never solved for
  x; x is always GIVEN and f'(x) is what is computed.

Difficulty is a genuine reasoning escalation, not a bigger polynomial:

- easy: a bare quadratic f, "find the instantaneous rate of change of f at
  x = p" -- one power-rule term beyond the constant/linear ones.
- medium: the same bare-question shape but a cubic f -- one more
  power-rule term to track, same reasoning task.
- hard: the SAME cubic, but wrapped as an applied quantity Q(t) in one
  context sentence (e.g. a tank's volume) and asked for "the instantaneous
  rate of change of Q when t = p" -- the learner must recognise that phrase
  means Q'(t) evaluated at the point, not just apply the power rule to a
  bare polynomial already labelled f. A genuinely different decision from
  medium, not a bigger cubic.
- exam: same applied wrapping, but the evaluation point itself needs a
  one-step derivation first -- "D minutes after t = p", i.e. evaluate at
  t = p + D -- a shape none of easy/medium/hard contains.
"""

from __future__ import annotations

from sympy import diff, symbols

from ..core import (
    check_distinct,
    check_equivalent,
    check_pairwise_distinct,
    generator,
    latex_answer,
    pick_distractors,
)

_x = symbols("x")

_P_POOL = tuple(v for v in range(-6, 7) if v != 0)
# Hard and exam wrap the function in a real-world context where t is time
# since a stated origin event (valve opens / launch / sampling); a negative
# time there is incoherent. Bare easy/medium keep the full pool.
_HARD_P_POOL = tuple(range(1, 7))  # hard: t >= 1
_EXAM_P_POOL = tuple(range(0, 7))  # exam: an exam start of t >= 0
_D_POOL = (2, 3, 4, 5)  # never 1: "1 minutes" would be a singular/plural defect

_CONTEXTS = (
    ("the volume, in litres, of water in a tank t minutes after a valve opens is Q(t) = {f_prose}", "minutes"),
    ("the height, in metres, of a drone t seconds after launch is Q(t) = {f_prose}", "seconds"),
    ("the number of bacteria, in thousands, in a culture t hours after sampling is Q(t) = {f_prose}", "hours"),
)


@generator(
    "p19f_instantaneous_rate",
    level=12,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p19f_instantaneous_rate(rng, tier="easy"):
    """f'(x) evaluated at a stated point -- see the module docstring for the
    per-tier reasoning escalation. All coefficients are integers, so f'(p)
    is always an exact integer."""
    offset = None

    for _attempt in range(50):
        if tier == "easy":
            p = rng.choice(_P_POOL)
            a = rng.randrange(1, 6)
            b = rng.randrange(-9, 10)
            c = rng.randrange(-9, 10)
            f_sym = a * _x**2 + b * _x + c
            point = p
            answer = 2 * a * point + b
            f_value = a * point**2 + b * point + c
            dropped = a * point + b
            wrong_point_variant = None
        elif tier == "medium":
            p = rng.choice(_P_POOL)
            a = rng.randrange(1, 6)
            b = rng.randrange(-9, 10)
            c = rng.randrange(-9, 10)
            d = rng.randrange(-9, 10)
            f_sym = a * _x**3 + b * _x**2 + c * _x + d
            point = p
            answer = 3 * a * point**2 + 2 * b * point + c
            f_value = a * point**3 + b * point**2 + c * point + d
            dropped = a * point**2 + 2 * b * point + c
            wrong_point_variant = None
        else:
            p = rng.choice(_HARD_P_POOL if tier == "hard" else _EXAM_P_POOL)
            a = rng.randrange(1, 6)
            b = rng.randrange(-9, 10)
            c = rng.randrange(-9, 10)
            d = rng.randrange(-9, 10)
            f_sym = a * _x**3 + b * _x**2 + c * _x + d
            if tier == "exam":
                offset = rng.choice(_D_POOL)
                point = p + offset
            else:
                point = p
            answer = 3 * a * point**2 + 2 * b * point + c
            f_value = a * point**3 + b * point**2 + c * point + d
            dropped = a * point**2 + 2 * b * point + c
            if tier == "exam":
                wrong_point_variant = 3 * a * p**2 + 2 * b * p + c
            else:
                wrong_point_variant = None

        candidates = [answer, f_value, dropped]
        if wrong_point_variant is not None:
            candidates.append(wrong_point_variant)
        candidates.append(-answer)  # sign-error is always a candidate slot
        if len(set(candidates)) == len(candidates):
            break

    # Independent check: sympy's own diff operator, at the same point the
    # prompt itself computes -- a route independent of the arithmetic
    # power-rule formula used to construct `answer` above.
    check_equivalent(answer, diff(f_sym, _x).subs(_x, point), f"row2 mismatch: tier={tier}, p={p}, point={point}")

    f_prose = str(f_sym.expand()).replace("**", "^").replace("*", "·")

    if tier == "easy":
        question_text = f"f(x) = {f_prose}. Find the instantaneous rate of change of f at x = {p}."
    elif tier == "medium":
        question_text = f"f(x) = {f_prose}. Find the instantaneous rate of change of f at x = {p}."
    elif tier == "hard":
        context_prose = f_prose.replace("x", "t")
        context_template, _unit = _CONTEXTS[rng.randrange(len(_CONTEXTS))]
        context = context_template.format(f_prose=context_prose)
        question_text = f"Suppose {context}. Find the instantaneous rate of change of Q when t = {p}."
    else:  # exam
        context_prose = f_prose.replace("x", "t")
        context_template, unit = _CONTEXTS[rng.randrange(len(_CONTEXTS))]
        context = context_template.format(f_prose=context_prose)
        # The offset must be stated in the SAME unit as t itself -- "5
        # minutes after t = 4" is incoherent when t is measured in hours or
        # seconds. Match the context's own stated unit exactly.
        question_text = f"Suppose {context}. Find the instantaneous rate of change of Q, {offset} {unit} after t = {p}."

    answer_text = str(answer)

    sign_error = -answer
    check_distinct(answer, f_value, msg=f"evaluated-function-value collides: tier={tier}, point={point}")
    check_distinct(answer, dropped, msg=f"dropped-power-rule-coefficient collides: tier={tier}, point={point}")
    check_distinct(answer, sign_error, msg=f"sign-error collides: tier={tier}, point={point}")
    candidates_text = [str(f_value), str(dropped)]
    if tier == "exam":
        check_distinct(answer, wrong_point_variant, msg=f"used-wrong-point collides: point={point}, p={p}")
        candidates_text.append(str(wrong_point_variant))
    candidates_text.append(str(sign_error))
    check_pairwise_distinct(
        answer,
        *[int(c) for c in candidates_text],
        msg=f"a distractor collides with another option: tier={tier}, point={point}",
    )

    distractors = pick_distractors(answer_text, candidates_text)

    return {
        "text": question_text,
        "latex": question_text,
        "answer": answer,
        "answer_text": answer_text,
        "answer_latex": latex_answer(answer),
        "distractors": distractors,
    }
