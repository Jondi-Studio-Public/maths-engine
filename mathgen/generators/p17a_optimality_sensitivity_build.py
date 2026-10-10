"""Generates `p17a_optimality_sensitivity` questions: pick the justified conclusion about how a convex quadratic
optimum changes under a parameter perturbation.

`ucv1-optimality-sensitivity`.

This build is narrowed to a FOUR-conclusion mcq -- "given a stated convex quadratic optimum and
a stated parameter perturbation, and four stated sensitivity conclusions about it, select the one that is
justified."

Every tier states the ORIGINAL optimum and the perturbation; it never states
the new optimum or the change -- the learner must compute it. All four
options on one question make a claim about the SAME quantity (never mixed
minimiser-claims with value-claims), built as a 2x2 of {correct, wrong}
direction x {correct, wrong} magnitude so no option is identifiable by its
written form (lesson 6, `midpoint_coordinate_scalar`).

Ladder (genuine reasoning escalation, not span/magnitude alone):
  easy:   perturb the LINEAR coefficient b -> b + d; judge the MINIMISER's
          move (x* = -b/(2a), moves by -d/(2a)). Decision: apply the chain
          factor 1/(2a), not just -d/a or the raw -b/a formula.
  medium: perturb the CURVATURE a -> a + d (kept convex); judge the MINIMUM
          VALUE's move (f* = c - b^2/(4a), a non-linear dependence on a).
          Decision: the change is NOT linear in d -- a first-order
          (linearised) estimate is a genuinely different, wrong number.
  hard:   two-variable SEPARABLE f(x, y) = a1 x^2 + b1 x + a2 y^2 + b2 y + c;
          perturb the CURVATURE of one named variable's part. Decision: which
          part of the separable sum the perturbation touches -- computing the
          medium-tier formula from the OTHER (untouched) variable's
          coefficients is the honest trap.
  exam:   single-variable, perturb the LINEAR coefficient b -> b + d (as
          easy) but ask a COMPOUND judgement: whether the minimiser crosses a
          stated threshold t AND by how much the minimum value f* changes.
          f*'s change is quadratic in d (contains a d^2 term), so a
          first-order estimate is again a genuinely wrong number, combined
          with an independent crossing/no-crossing verdict.

Local-vs-global overclaim: NOT used. `convex-quadratic-minimum` (the prerequisite) guarantees a>0/convexity on
every draw, so local and global minima coincide -- an "only a local minimum" conclusion would not be honestly
wrong here. The magnitude-error family (dropping the chain factor, or linearising a non-linear dependence)
replaces it.
"""

from __future__ import annotations

import random
from fractions import Fraction

from mathgen.core import (
    check,
    check_pairwise_distinct,
    generator,
    join_terms,
    latex_answer,
    paren_negative,
    phrase,
    rational_text,
)

_LABELS = ["A", "B", "C", "D"]


def _poly2(a, b, c) -> str:
    return join_terms([(a, "x^2"), (b, "x"), (c, "")])


def _rt(v) -> str:
    return rational_text(Fraction(v))


def _direction_word(delta: Fraction) -> str:
    return "increases" if delta > 0 else "decreases"


def _claim(quantity: str, delta: Fraction) -> str:
    check(delta != 0, "optimality-sensitivity: a claimed change must be nonzero")
    return f"{quantity} {_direction_word(delta)} by {_rt(abs(delta))}"


def _build_four(rng, quantity: str, key_delta: Fraction, mag_err_delta: Fraction):
    """Standard 2x2: key (correct dir, correct mag), sign-flip (wrong dir,
    correct mag), magnitude-error (correct dir, wrong mag), third (wrong
    both). Returns (option_texts_in_ABCD_order, answer_letter)."""
    sign_flip_delta = -key_delta
    third_delta = -mag_err_delta
    check_pairwise_distinct(
        key_delta,
        sign_flip_delta,
        mag_err_delta,
        third_delta,
        msg="optimality-sensitivity: the four option deltas must be pairwise distinct",
    )
    values = [key_delta, sign_flip_delta, mag_err_delta, third_delta]
    texts = [_claim(quantity, v) for v in values]
    check(len(set(texts)) == 4, "optimality-sensitivity: the four options must all display differently")
    order = list(range(4))
    rng.shuffle(order)
    shuffled_texts = [texts[i] for i in order]
    answer_idx = order.index(0)  # index 0 is always the key
    return shuffled_texts, _LABELS[answer_idx]


def _render(rng, tier: str, stem: str, quantity_desc: str, options: list[str]) -> str:
    opts_text = "   ".join(f"{_LABELS[i]}: {options[i]}" for i in range(4))
    return (
        f"{stem} Exactly one of the following four conclusions about {quantity_desc} is "
        f"justified. Which one? {opts_text}"
    )


_EASY_STEMS = [
    "Consider f(x) = {poly}, whose minimiser is x* = {xstar} with minimum value f(x*) = {fstar}. "
    "The coefficient of x is then changed from {b} to {bnew}.",
    "Let f(x) = {poly}, with minimiser x* = {xstar} and minimum value f(x*) = {fstar}. "
    "Suppose the linear coefficient b is perturbed from {b} to {bnew}.",
    "For f(x) = {poly} the minimiser is x* = {xstar} and the minimum value is f(x*) = {fstar}. "
    "The coefficient of x is now changed to {bnew} (from {b}).",
]

_MEDIUM_STEMS = [
    "Consider f(x) = {poly}, whose minimiser is x* = {xstar} with minimum value f(x*) = {fstar}. "
    "The coefficient of x^2 is then changed from {a} to {anew}.",
    "Let f(x) = {poly}, with minimiser x* = {xstar} and minimum value f(x*) = {fstar}. "
    "Suppose the curvature coefficient a is perturbed from {a} to {anew}.",
    "For f(x) = {poly} the minimiser is x* = {xstar} and the minimum value is f(x*) = {fstar}. "
    "The coefficient of x^2 is now changed to {anew} (from {a}).",
]

_HARD_STEMS = [
    "Consider f(x, y) = {poly}, a separable function with minimiser (x*, y*) = ({xstar}, {ystar}) "
    "and minimum value f(x*, y*) = {fstar}. The coefficient of {var}^2 is then changed from {a} to {anew}.",
    "Let f(x, y) = {poly}, with minimiser (x*, y*) = ({xstar}, {ystar}) and minimum value "
    "f(x*, y*) = {fstar}. Suppose the curvature coefficient of {var} is perturbed from {a} to {anew}.",
    "For f(x, y) = {poly} the minimiser is (x*, y*) = ({xstar}, {ystar}) and the minimum value is "
    "f(x*, y*) = {fstar}. The coefficient of {var}^2 is now changed to {anew} (from {a}).",
]

_EXAM_STEMS = [
    "Consider f(x) = {poly}, whose minimiser is x* = {xstar} with minimum value f(x*) = {fstar}. "
    "The coefficient of x is then changed from {b} to {bnew}. Let t = {t}.",
    "Let f(x) = {poly}, with minimiser x* = {xstar} and minimum value f(x*) = {fstar}. "
    "Suppose the linear coefficient b is perturbed from {b} to {bnew}. Let t = {t}.",
    "For f(x) = {poly} the minimiser is x* = {xstar} and the minimum value is f(x*) = {fstar}. "
    "The coefficient of x is now changed to {bnew} (from {b}). Let t = {t}.",
]


def _draw_ab(rng, a_hi, b_hi):
    a = rng.randint(1, a_hi)
    b = rng.randint(-b_hi, -1) if rng.random() < 0.5 else rng.randint(1, b_hi)
    # keep b nonzero -- a zero linear coefficient makes several distractor
    # deltas coincide with 0 or with each other.
    return a, b


@generator(
    "p17a_optimality_sensitivity",
    level=19,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p17a_optimality_sensitivity(rng: random.Random, tier: str = "easy"):
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    if tier == "hard":
        return _hard(rng)
    if tier == "exam":
        return _exam(rng)
    raise KeyError(f"unknown tier {tier!r}")


def _easy(rng):
    for _ in range(60):
        a, b = _draw_ab(rng, 6, 8)
        c = rng.randint(-8, 8)
        d = rng.randint(1, 6) if rng.random() < 0.5 else rng.randint(-6, -1)
        b_new = b + d
        x_star = Fraction(-b, 2 * a)
        f_star = Fraction(c) - Fraction(b * b, 4 * a)
        key_delta = Fraction(-d, 2 * a)  # x* moves by -d/(2a)
        variant = rng.choice(["over", "under"])
        mag_err_delta = Fraction(-d, a) if variant == "over" else Fraction(-d, 4 * a)
        try:
            options, answer_letter = _build_four(rng, "the minimiser x*", key_delta, mag_err_delta)
        except Exception:
            continue
        break
    else:
        raise AssertionError("optimality-sensitivity easy: could not build 4 distinct options")

    check(
        x_star + key_delta == Fraction(-b_new, 2 * a),
        "optimality-sensitivity easy: minimiser-change verification failed",
    )

    stem = phrase(rng, _EASY_STEMS, poly=_poly2(a, b, c), xstar=_rt(x_star), fstar=_rt(f_star), b=b, bnew=b_new)
    text = _render(rng, "easy", stem, "how the minimiser x* moves", options)
    working = [
        f"x* = -b/(2a) = {_rt(x_star)}.",
        f"New x* = -{paren_negative(b_new)}/(2·{a}) = {_rt(x_star + key_delta)}, so x* {_direction_word(key_delta)} by {_rt(abs(key_delta))}.",
        f"Only option {answer_letter} states this correctly.",
    ]
    return dict(
        text=text,
        answer=answer_letter,
        answer_text=answer_letter,
        answer_latex=latex_answer(answer_letter, plain=True),
        working=working,
        distractors=[l for l in _LABELS if l != answer_letter],
    )


def _medium(rng):
    for _ in range(60):
        a, b = _draw_ab(rng, 6, 8)
        c = rng.randint(-8, 8)
        sign = rng.choice([-1, 1])
        d = sign * rng.randint(1, min(a - 1, 5) if a > 1 else 1)
        if a + d <= 0:
            continue
        a_new = a + d
        x_star = Fraction(-b, 2 * a)
        f_star = Fraction(c) - Fraction(b * b, 4 * a)
        f_star_new = Fraction(c) - Fraction(b * b, 4 * a_new)
        key_delta = f_star_new - f_star  # = b^2 d / (4 a (a+d))
        mag_err_delta = Fraction(b * b * d, 4 * a * a)  # linearised (wrong: non-linear truth)
        try:
            options, answer_letter = _build_four(rng, "the minimum value f*", key_delta, mag_err_delta)
        except Exception:
            continue
        break
    else:
        raise AssertionError("optimality-sensitivity medium: could not build 4 distinct options")

    check(
        Fraction(b * b * d, 4 * a * a_new) == key_delta,
        "optimality-sensitivity medium: value-change verification failed",
    )

    stem = phrase(rng, _MEDIUM_STEMS, poly=_poly2(a, b, c), xstar=_rt(x_star), fstar=_rt(f_star), a=a, anew=a_new)
    text = _render(rng, "medium", stem, "how the minimum value f* moves", options)
    working = [
        f"f* = c - b^2/(4a) = {_rt(f_star)}.",
        f"New f* = {c} - {b * b}/(4·{a_new}) = {_rt(f_star_new)}, so f* {_direction_word(key_delta)} by {_rt(abs(key_delta))}.",
        "This change is not linear in d, so a first-order estimate is a different (wrong) number.",
        f"Only option {answer_letter} states the true change.",
    ]
    return dict(
        text=text,
        answer=answer_letter,
        answer_text=answer_letter,
        answer_latex=latex_answer(answer_letter, plain=True),
        working=working,
        distractors=[l for l in _LABELS if l != answer_letter],
    )


def _value_change(a_val, b_val, dd):
    return Fraction(b_val * b_val * dd, 4 * a_val * (a_val + dd))


def _hard(rng):
    for _ in range(80):
        a1, b1 = _draw_ab(rng, 6, 8)
        a2, b2 = _draw_ab(rng, 6, 8)
        c = rng.randint(-8, 8)
        touch_x = rng.random() < 0.5
        a_t, b_t = (a1, b1) if touch_x else (a2, b2)
        a_o, b_o = (a2, b2) if touch_x else (a1, b1)
        sign = rng.choice([-1, 1])
        d = sign * rng.randint(1, min(a_t - 1, 5) if a_t > 1 else 1)
        if a_t + d <= 0 or a_o + d == 0:
            continue
        a_t_new = a_t + d

        key_delta = _value_change(a_t, b_t, d)
        wrong_part_delta = _value_change(a_o, b_o, d)  # honest trap: uses the UNTOUCHED part's data
        try:
            options, answer_letter = _build_four(rng, "the minimum value f*", key_delta, wrong_part_delta)
        except Exception:
            continue
        break
    else:
        raise AssertionError("optimality-sensitivity hard: could not build 4 distinct options")

    x_star = Fraction(-b1, 2 * a1)
    y_star = Fraction(-b2, 2 * a2)
    f_star = Fraction(c) - Fraction(b1 * b1, 4 * a1) - Fraction(b2 * b2, 4 * a2)
    if touch_x:
        f_star_new_direct = Fraction(c) - Fraction(b1 * b1, 4 * a_t_new) - Fraction(b2 * b2, 4 * a2)
    else:
        f_star_new_direct = Fraction(c) - Fraction(b1 * b1, 4 * a1) - Fraction(b2 * b2, 4 * a_t_new)
    check(f_star_new_direct - f_star == key_delta, "optimality-sensitivity hard: value-change verification failed")

    poly = join_terms([(a1, "x^2"), (b1, "x"), (a2, "y^2"), (b2, "y"), (c, "")])
    var = "x" if touch_x else "y"
    stem = phrase(
        rng,
        _HARD_STEMS,
        poly=poly,
        xstar=_rt(x_star),
        ystar=_rt(y_star),
        fstar=_rt(f_star),
        var=var,
        a=a_t,
        anew=a_t_new,
    )
    text = _render(rng, "hard", stem, "how the minimum value f* moves", options)
    working = [
        f"f* = {_rt(f_star)} (separable: each variable contributes its own -b_i^2/(4 a_i) term).",
        f"Only the {var}-part is perturbed, contributing {_direction_word(key_delta)} of {_rt(abs(key_delta))} "
        f"to f*; the other part is unchanged.",
        f"Only option {answer_letter} attributes the change to the correct part.",
    ]
    return dict(
        text=text,
        answer=answer_letter,
        answer_text=answer_letter,
        answer_latex=latex_answer(answer_letter, plain=True),
        working=working,
        distractors=[l for l in _LABELS if l != answer_letter],
    )


def _exam(rng):
    for _ in range(80):
        a, b = _draw_ab(rng, 6, 8)
        c = rng.randint(-8, 8)
        d = rng.randint(2, 8) if rng.random() < 0.5 else rng.randint(-8, -2)
        b_new = b + d
        x_star = Fraction(-b, 2 * a)
        x_star_new = Fraction(-b_new, 2 * a)
        if x_star == x_star_new:
            continue
        lo, hi = sorted([x_star, x_star_new])
        # t strictly between OR strictly outside both -- never equal to either.
        if rng.random() < 0.5:
            # place t strictly between (crossing case candidate)
            t = (
                Fraction(lo.numerator + hi.numerator, lo.denominator + hi.denominator)
                if lo.denominator == hi.denominator
                else (lo + hi) / 2
            )
        else:
            t = hi + Fraction(rng.randint(1, 4))
        if t in (x_star, x_star_new):
            continue
        crosses = (x_star - t) * (x_star_new - t) < 0

        f_star = Fraction(c) - Fraction(b * b, 4 * a)
        f_star_new = Fraction(c) - Fraction(b_new * b_new, 4 * a)
        key_value_delta = f_star_new - f_star  # exact, contains a d^2 term
        mag_err_delta = Fraction(-(2 * b * d), 4 * a)  # linearised: drops the d^2 term
        if key_value_delta == 0 or mag_err_delta == 0:
            continue
        if abs(key_value_delta) == abs(mag_err_delta):
            # degenerate: the magnitude-error class would then share the
            # key's own displayed amount (only sign differing), leaving no
            # distinguishable "wrong magnitude" on this draw -- redraw.
            continue

        def _combo(cross_claim: bool, value_delta: Fraction) -> str:
            crossing_text = "the minimiser moves past t" if cross_claim else "the minimiser does not move past t"
            return f"{crossing_text}, and {_claim('the minimum value f*', value_delta)}"

        try:
            check_pairwise_distinct(
                key_value_delta, mag_err_delta, msg="optimality-sensitivity exam: value deltas must differ"
            )
            key_text = _combo(crosses, key_value_delta)
            sign_flip_text = _combo(not crosses, key_value_delta)
            mag_err_text = _combo(crosses, mag_err_delta)
            third_text = _combo(not crosses, mag_err_delta)
        except Exception:
            continue
        texts = [key_text, sign_flip_text, mag_err_text, third_text]
        if len(set(texts)) != 4:
            continue
        order = list(range(4))
        rng.shuffle(order)
        shuffled = [texts[i] for i in order]
        answer_letter = _LABELS[order.index(0)]
        break
    else:
        raise AssertionError("optimality-sensitivity exam: could not build 4 distinct options")

    check(f_star_new - f_star == key_value_delta, "optimality-sensitivity exam: value-change verification failed")
    check(
        (x_star_new - t) * (x_star - t) < 0 if crosses else (x_star_new - t) * (x_star - t) > 0,
        "optimality-sensitivity exam: crossing verification failed",
    )

    stem = phrase(
        rng, _EXAM_STEMS, poly=_poly2(a, b, c), xstar=_rt(x_star), fstar=_rt(f_star), b=b, bnew=b_new, t=_rt(t)
    )
    text = _render(rng, "exam", stem, "the perturbation's effect", shuffled)
    working = [
        f"x* = {_rt(x_star)}; new x* = {_rt(x_star_new)}; t = {_rt(t)}.",
        f"The minimiser {'does move' if crosses else 'does not move'} past t.",
        f"f* changes from {_rt(f_star)} to {_rt(f_star_new)}, a change of {_rt(key_value_delta)} "
        f"(exact -- contains a d^2 term, so a linear estimate of {_rt(mag_err_delta)} is wrong).",
        f"Only option {answer_letter} states both facts correctly.",
    ]
    return dict(
        text=text,
        answer=answer_letter,
        answer_text=answer_letter,
        answer_latex=latex_answer(answer_letter, plain=True),
        working=working,
        distractors=[l for l in _LABELS if l != answer_letter],
    )
