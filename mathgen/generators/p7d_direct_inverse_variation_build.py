"""Generates `p7d_examine_direct_proportion` and `p7d_examine_inverse_proportion` questions: find a new value
under direct or inverse variation.

Direct and inverse variation.

- ``examine_direct_proportion``
  (``senior-examine-examples-of-direct-proportion-and-linearly-related``,
  ACMMM002, domain functions-sequences, answer: number) -- given bounded,
  fully specified inputs, examine examples of direct proportion and linearly
  related variables: compute a second value, a second input, or (hard tier)
  the new value produced by a two-step composed scaling of the input, always
  from a stated y = kx formula and one data point.
- ``examine_inverse_proportion``
  (``senior-examine-examples-of-inverse-proportion``, ACMMM012, domain
  functions-sequences, answer: number) -- the same item shape for a stated
  y = k/x formula.

Shared convention: every question states its algebraic relation as a formula, visibly, in the question
text -- never a bare "varies directly"/"is proportional to" without the formula -- and every distractor is the
arithmetic a learner would get by mistakenly applying one of the *other two* relations
(direct/inverse/quadratic) to the same drawn numbers. No question here ever asks a learner to classify an
unlabelled table of values; the relation type and formula are always given.

The hard tier below instead composes TWO scale steps (`t1` then `t2`) and asks for the resulting NEW value,
never printing the combined factor `t1*t2` and never naming "multiplied"/"divided" in connection with the
answer -- the learner must infer which operation applies from the stated relation itself. Every candidate
value that is meant to look like the answer's own step (everything except the deliberate `no-change`
distractor, which is allowed to echo the printed "before" value) is checked against the numbers literally
printed in the question text, every seed, by resampling on collision -- see the `printed` set in each
hard-tier branch below.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import (
    VerificationError,
    check_equivalent,
    check_pairwise_distinct,
    generator,
    pick_distractors,
    rational_text,
)

_RESAMPLE_ATTEMPTS = 100


# ==============================================================================
# Independent checks -- a route that does not reuse the construction algebra
# ==============================================================================


def _verify_direct(x1: int, y1: int, x2: int, y2: int, msg: str) -> None:
    """Independent check for y = kx.

    Derives k = y1/x1 as an exact Fraction from the *stated* data (a
    division), then recomputes y2 as k * x2 (a multiplication) and compares
    against the value actually being returned -- the reverse arithmetic
    order from every construction path above, which either multiplies y1 by
    t directly (easy) or multiplies x1 by t directly (medium).
    """
    k = Fraction(int(y1), int(x1))
    check_equivalent(k * int(x2), Fraction(int(y2)), msg)


def _verify_direct_hard(x1: int, f1: int, t1: int, t2: int, f2: int, msg: str) -> None:
    """Independent check for the hard-tier two-step composed answer.

    Derives k = f1/x1 from the stated data point and applies it to the new
    extension x2 = x1*t1*t2 via the relation's own formula -- a route the
    construction never takes (the construction's "headline" route is the
    direct scale-factor arithmetic F1*t1*t2). Confirms the two routes agree
    and both equal the value being returned.
    """
    k = Fraction(int(f1), int(x1))
    x2 = int(x1) * int(t1) * int(t2)
    via_k = k * x2
    via_scale = Fraction(int(f1)) * int(t1) * int(t2)
    check_equivalent(via_k, via_scale, msg)
    check_equivalent(via_scale, Fraction(int(f2)), msg)


def _verify_inverse(x1: int, y1: int, x2: int, y2: int, msg: str) -> None:
    """Independent check for y = k/x.

    Derives k = x1 * y1 (a multiplication) from the stated data, then
    recomputes the other side as k / x2 (a division) -- the reverse
    arithmetic order from every construction path above, which either
    divides y1 by t directly (easy) or multiplies x1 by t directly (medium).
    """
    k = int(x1) * int(y1)
    check_equivalent(Fraction(k, int(x2)), Fraction(int(y2)), msg)


def _verify_inverse_hard(v1: int, p1: int, t1: int, t2: int, p2: int, msg: str) -> None:
    """Independent check for the hard-tier two-step composed answer.

    Derives k = v1*p1 from the stated data point and applies it to the new
    volume v2 = v1*t1*t2 via the relation's own formula -- a route the
    construction never takes (the construction's "headline" route is the
    direct scale-factor arithmetic P1/(t1*t2)). Confirms the two routes
    agree and both equal the value being returned.
    """
    k = int(v1) * int(p1)
    v2 = int(v1) * int(t1) * int(t2)
    via_k = Fraction(k, v2)
    via_scale = Fraction(int(p1)) / (int(t1) * int(t2))
    check_equivalent(via_k, via_scale, msg)
    check_equivalent(via_scale, Fraction(int(p2)), msg)


# ==============================================================================
# Skill 1: examine examples of direct proportion and linearly related variables
# ==============================================================================


@generator(
    "p7d_examine_direct_proportion",
    level=11,
    presets={
        "easy": dict(variant="find_y2"),
        "medium": dict(variant="find_x2"),
        "hard": dict(variant="scale_factor"),
    },
)
def examine_direct_proportion(rng, variant="find_y2"):
    """
    y is directly proportional to x (y = kx). Every candidate value shown to
    the learner is a positive integer, by construction.

    find_y2 (easy): given (x1, y1) and a new x2 = x1*t, find y2. One
    unknown, direct substitution.
    find_x2 (medium): given (x1, y1) and a new y2, find x2 -- the unknown
    moves to the other side of the relation, a genuinely different
    substitution from easy.
    scale_factor (hard): Hooke's Law framing. Given one concrete (x1, F1)
    pair, the extension is scaled TWICE in succession (by t1, then by a
    further t2); find the resulting new force. The combined factor t1*t2
    is never printed and the operation ("multiplied") is never named in
    connection with the answer -- the learner must infer multiply-vs-divide
    from the stated F = kx relation itself, not be told it.
    """
    if variant == "find_y2":
        a = rng.randint(2, 9)
        t = rng.randint(2, 5)
        s = rng.randint(1, 9)
        y1 = t * s
        x2 = a * t
        y2 = t * y1  # = t^2 * s

        inv_distractor = y1 // t  # "as if inverse": y1/t = s
        quad_distractor = y1 * t * t  # "as if quadratic": y1*t^2 = t^3*s
        no_change_distractor = y1  # "forgot the value changes": stays at y1

        _verify_direct(a, y1, x2, y2, f"find_y2: independent k-route disagrees for a={a}, t={t}, s={s}")
        check_pairwise_distinct(
            y2,
            inv_distractor,
            quad_distractor,
            no_change_distractor,
            msg=f"find_y2: a distractor collides with the answer for a={a}, t={t}, s={s}",
        )

        text = f"y is directly proportional to x, so y = kx. When x = {a}, y = {y1}. What is y when x = {x2}?"
        k_str = rational_text(Fraction(y1, a))
        working = [
            "y = kx",
            f"k = y / x = {y1} / {a} = {k_str}",
            f"y = k × x = {k_str} × {x2} = {y2}",
        ]
        candidates = [str(inv_distractor), str(quad_distractor), str(no_change_distractor)]
        distractors = pick_distractors(str(y2), candidates, n=3)
        return dict(
            text=text,
            latex=text,
            answer=y2,
            answer_text=str(y2),
            answer_latex=str(y2),
            working=working,
            distractors=distractors,
        )

    elif variant == "find_x2":
        s = rng.randint(1, 9)  # y1
        t = rng.randint(2, 5)
        y1 = s
        y2 = t * s
        a2 = rng.randint(1, 9)
        a = t * a2  # x1
        x2 = a * t  # = t^2 * a2

        inv_distractor = a // t  # "as if inverse": a/t = a2
        quad_distractor = a * t * t  # "as if quadratic": a*t^2 = t^3*a2
        no_change_distractor = a  # "forgot the value changes": stays at x1

        _verify_direct(a, y1, x2, y2, f"find_x2: independent k-route disagrees for a={a}, t={t}, y1={y1}, y2={y2}")
        check_pairwise_distinct(
            x2,
            inv_distractor,
            quad_distractor,
            no_change_distractor,
            msg=f"find_x2: a distractor collides with the answer for a={a}, t={t}, a2={a2}",
        )

        text = f"y is directly proportional to x, so y = kx. When x = {a}, y = {y1}. For what value of x does y = {y2}?"
        k_str = rational_text(Fraction(y1, a))
        k_paren = k_str if "/" not in k_str else f"({k_str})"
        working = [
            "y = kx",
            f"k = y / x = {y1} / {a} = {k_str}",
            f"x = y / k = {y2} / {k_paren} = {x2}",
        ]
        candidates = [str(inv_distractor), str(quad_distractor), str(no_change_distractor)]
        distractors = pick_distractors(str(x2), candidates, n=3)
        return dict(
            text=text,
            latex=text,
            answer=x2,
            answer_text=str(x2),
            answer_latex=str(x2),
            working=working,
            distractors=distractors,
        )

    elif variant == "scale_factor":
        for _ in range(_RESAMPLE_ATTEMPTS):
            x1 = rng.randint(2, 9)
            s = rng.randint(1, 9)
            t1 = rng.randint(2, 5)
            t2 = rng.randint(2, 5)
            f1 = t1 * t2 * s

            answer = f1 * t1 * t2  # = (t1*t2)^2 * s
            inverse_distractor = f1 // (t1 * t2)  # "as if inverse": F1/(t1*t2) = s
            quad_distractor = f1 * (t1 * t2) ** 2  # "as if quadratic": F1*(t1*t2)^2 = (t1*t2)^3*s
            no_change_distractor = f1  # "forgot the value changes": stays at F1 -- deliberately
            # equal to the printed F1; every OTHER candidate must clear the printed-number check.

            printed = {x1, f1, t1, t2}
            if {answer, inverse_distractor, quad_distractor} & printed:
                continue
            try:
                check_pairwise_distinct(answer, inverse_distractor, quad_distractor, no_change_distractor)
            except VerificationError:
                continue
            break
        else:
            raise VerificationError("scale_factor: no collision-free x1/s/t1/t2 draw found")

        _verify_direct_hard(
            x1,
            f1,
            t1,
            t2,
            answer,
            f"scale_factor: independent k-route disagrees for x1={x1}, f1={f1}, t1={t1}, t2={t2}",
        )

        text = (
            f"The force needed to stretch this spring is directly proportional to its "
            f"extension, so F = kx. When the extension is {x1} cm, the force is {f1} N. "
            f"The extension is first increased so it is multiplied by {t1}, then increased "
            f"again so it is multiplied by a further {t2}. Find the new force, in N."
        )
        working = [
            "F = kx",
            f"extension is multiplied by {t1}, then by a further {t2}: new extension = x × {t1} × {t2}",
            f"force scales the same way: F2 = {f1} × {t1} × {t2} = {answer}",
        ]
        candidates = [str(inverse_distractor), str(quad_distractor), str(no_change_distractor)]
        distractors = pick_distractors(str(answer), candidates, n=3)
        return dict(
            text=text,
            latex=text,
            answer=answer,
            answer_text=str(answer),
            answer_latex=str(answer),
            working=working,
            distractors=distractors,
        )

    else:
        raise ValueError(f"unknown variant: {variant!r}")


# ==============================================================================
# Skill 2: examine examples of inverse proportion
# ==============================================================================


@generator(
    "p7d_examine_inverse_proportion",
    level=11,
    presets={
        "easy": dict(variant="find_y2"),
        "medium": dict(variant="find_x2"),
        "hard": dict(variant="divide_factor"),
    },
)
def examine_inverse_proportion(rng, variant="find_y2"):
    """
    y is inversely proportional to x (y = k/x). Every candidate value shown
    to the learner is a positive integer, by construction.

    find_y2 (easy): given (x1, y1) and a new x2 = x1*t, find y2. One
    unknown, direct substitution.
    find_x2 (medium): given (x1, y1) and a new y2, find x2 -- the unknown
    moves to the other side of the relation, a genuinely different
    substitution from easy.
    divide_factor (hard): Boyle's Law framing. Given one concrete (V1, P1)
    pair, the volume is scaled TWICE in succession (by t1, then by a
    further t2); find the resulting new pressure. The combined factor
    t1*t2 is never printed and neither "multiplied" nor "divided" is named
    in connection with the answer -- the learner must infer that the
    pressure divides (not multiplies) from the stated P = k/V relation
    itself, not be told it.
    """
    if variant == "find_y2":
        a = rng.randint(2, 9)
        t = rng.randint(2, 5)
        s = rng.randint(1, 9)
        y1 = t * s
        x2 = a * t
        y2 = y1 // t  # true inverse: y1/t = s

        direct_distractor = y1 * t  # "as if direct": y1*t = t^2*s
        quad_distractor = y1 * t * t  # "as if quadratic": y1*t^2 = t^3*s
        no_change_distractor = y1  # "forgot the value changes": stays at y1

        _verify_inverse(a, y1, x2, y2, f"find_y2: independent k-route disagrees for a={a}, t={t}, s={s}")
        check_pairwise_distinct(
            y2,
            direct_distractor,
            quad_distractor,
            no_change_distractor,
            msg=f"find_y2: a distractor collides with the answer for a={a}, t={t}, s={s}",
        )

        text = f"y is inversely proportional to x, so y = k/x. When x = {a}, y = {y1}. What is y when x = {x2}?"
        k = a * y1
        working = [
            "y = k / x",
            f"k = x × y = {a} × {y1} = {k}",
            f"y = k / x = {k} / {x2} = {y2}",
        ]
        candidates = [str(direct_distractor), str(quad_distractor), str(no_change_distractor)]
        distractors = pick_distractors(str(y2), candidates, n=3)
        return dict(
            text=text,
            latex=text,
            answer=y2,
            answer_text=str(y2),
            answer_latex=str(y2),
            working=working,
            distractors=distractors,
        )

    elif variant == "find_x2":
        s2 = rng.randint(1, 9)
        t = rng.randint(2, 5)
        y1 = t * s2
        y2 = s2  # so y1 / y2 = t exactly
        a2 = rng.randint(1, 9)
        x1 = t * a2

        x2 = x1 * t  # true inverse: x1 * y1 / y2 = t * (t*a2) = t^2*a2

        direct_distractor = x1 // t  # "as if direct": x1*y2/y1 = a2
        quad_distractor = x1 * t * t  # "as if quadratic": x1*t^2 = t^3*a2
        no_change_distractor = x1  # "forgot the value changes": stays at x1

        _verify_inverse(x1, y1, x2, y2, f"find_x2: independent k-route disagrees for x1={x1}, t={t}, s2={s2}, a2={a2}")
        check_pairwise_distinct(
            x2,
            direct_distractor,
            quad_distractor,
            no_change_distractor,
            msg=f"find_x2: a distractor collides with the answer for x1={x1}, t={t}, a2={a2}",
        )

        text = (
            f"y is inversely proportional to x, so y = k/x. When x = {x1}, y = {y1}. For what value of x does y = {y2}?"
        )
        k = x1 * y1
        working = [
            "y = k / x",
            f"k = x × y = {x1} × {y1} = {k}",
            f"x = k / y = {k} / {y2} = {x2}",
        ]
        candidates = [str(direct_distractor), str(quad_distractor), str(no_change_distractor)]
        distractors = pick_distractors(str(x2), candidates, n=3)
        return dict(
            text=text,
            latex=text,
            answer=x2,
            answer_text=str(x2),
            answer_latex=str(x2),
            working=working,
            distractors=distractors,
        )

    elif variant == "divide_factor":
        for _ in range(_RESAMPLE_ATTEMPTS):
            v1 = rng.randint(2, 9)
            s = rng.randint(1, 9)
            t1 = rng.randint(2, 5)
            t2 = rng.randint(2, 5)
            p1 = t1 * t2 * s

            answer = s  # = P1 / (t1*t2)
            direct_distractor = p1 * t1 * t2  # "as if direct": P1*(t1*t2) = (t1*t2)^2*s
            quad_distractor = p1 * (t1 * t2) ** 2  # "as if quadratic": P1*(t1*t2)^2 = (t1*t2)^3*s
            no_change_distractor = p1  # "forgot the value changes": stays at P1 -- deliberately
            # equal to the printed P1; every OTHER candidate must clear the printed-number check.

            printed = {v1, p1, t1, t2}
            if {answer, direct_distractor, quad_distractor} & printed:
                continue
            try:
                check_pairwise_distinct(answer, direct_distractor, quad_distractor, no_change_distractor)
            except VerificationError:
                continue
            break
        else:
            raise VerificationError("divide_factor: no collision-free v1/s/t1/t2 draw found")

        _verify_inverse_hard(
            v1,
            p1,
            t1,
            t2,
            answer,
            f"divide_factor: independent k-route disagrees for v1={v1}, p1={p1}, t1={t1}, t2={t2}",
        )

        text = (
            f"Boyle's Law states that at constant temperature, the pressure of a gas is "
            f"inversely proportional to its volume, so P = k/V. When the volume is {v1} L, "
            f"the pressure is {p1} kPa. The volume is first increased so it is multiplied by "
            f"{t1}, then increased again so it is multiplied by a further {t2}. Find the new "
            f"pressure, in kPa."
        )
        working = [
            "P = k / V",
            f"volume is multiplied by {t1}, then by a further {t2}: new volume = V × {t1} × {t2}",
            f"pressure scales inversely: P2 = {p1} / ({t1} × {t2}) = {answer}",
        ]
        candidates = [str(direct_distractor), str(quad_distractor), str(no_change_distractor)]
        distractors = pick_distractors(str(answer), candidates, n=3)
        return dict(
            text=text,
            latex=text,
            answer=answer,
            answer_text=str(answer),
            answer_latex=str(answer),
            working=working,
            distractors=distractors,
        )

    else:
        raise ValueError(f"unknown variant: {variant!r}")
