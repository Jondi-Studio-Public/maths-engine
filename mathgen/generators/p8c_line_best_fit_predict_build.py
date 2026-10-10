"""Generates `p8c_line_best_fit_predict` questions: predict a value from a given line of best fit by
interpolation or extrapolation.

Senior line-of-best-fit prediction.

- ``p8c_line_best_fit_predict`` (``senior-use-the-line-of-best-fit-to-make``,
  ACMEM145, domain statistics-probability, answer: number) -- use an
  ALREADY-KNOWN line of best fit (given directly as ``y = mx + c``, integers
  ``m``/``c``) to make predictions, by interpolation and extrapolation. The
  line's equation and a stated valid ``x``-range for the data it was fitted
  to are always given inputs; deriving ``m``/``c`` from raw data points is
  the sibling skill's contract, never
  this one's.

Difficulty is a genuine reasoning-difficulty axis, independent of the
interpolation/extrapolation axis (every tier draws both, across seeds):

- easy: forward substitution -- given ``x``, predict ``y``.
- medium: inverse -- given a target ``y`` (constructed from a hidden integer
  ``x_target`` so the reverse division is exact by construction), find ``x``.
- hard: predicted change over an interval -- given ``x1``/``x2``, find the
  predicted change in ``y``, ``m * (x2 - x1)``, without ever computing the
  intercept ``c``.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import check_distinct, check_equivalent, generator, pick_distractors


# (x description, y description) -- realistic bivariate-data contexts, never
# bare abstract coordinates. Deliberately no explicit numeric bound in the
# phrasing itself (e.g. no "out of 100") since m/c/x are drawn independently
# of the context and could otherwise imply an impossible value.
CONTEXTS = [
    ("hours of tutoring given", "number of students enrolled"),
    ("weeks of practice", "typing speed in words per minute"),
    ("items produced in a shift", "total production cost in dollars"),
    ("kilometres a courier travels", "delivery fee in dollars"),
    ("months since a gym opened", "total membership count"),
]


def _draw_range(rng):
    """A stated valid x-range [x_min, x_max] for the data the line was fitted to."""
    x_min = rng.randint(0, 5)
    x_max = x_min + rng.randint(12, 20)
    return x_min, x_max


def _draw_x(rng, x_min, x_max, inside):
    """One x-value, strictly inside or strictly outside [x_min, x_max] --
    never equal to either boundary, so the interpolation/extrapolation
    framing is always unambiguous. Every context here is a naturally
    non-negative quantity (hours, weeks, kilometres, items, months), so an
    "outside, below x_min" draw is only offered when it cannot go negative;
    x_min == 0 (or a small x_min) always resolves to "outside, above x_max"
    instead, never to a fabricated negative count."""
    if inside:
        return rng.randint(x_min + 1, x_max - 1)
    can_go_below = x_min >= 1
    if can_go_below and rng.random() < 0.5:
        return x_min - rng.randint(1, min(5, x_min))
    return x_max + rng.randint(1, 5)


def _is_inside(x, x_min, x_max):
    return x_min < x < x_max


def _framing_single(x, x_min, x_max, *, reveal_x):
    """State plainly whether one x-value is interpolation or extrapolation.

    ``reveal_x=False`` is used where x is the unknown being solved for (the
    medium tier): the framing still names which kind of prediction this is,
    without stating the value that would give the answer away.
    """
    lead = f"x = {x}" if reveal_x else "This prediction"
    if _is_inside(x, x_min, x_max):
        return f"{lead} lies within the range the line was fitted to, so this is interpolation."
    return f"{lead} lies outside the range the line was fitted to, so this is extrapolation."


def _framing_pair(x1, x2, x_min, x_max):
    """State plainly which of x1/x2 is interpolation/extrapolation (at least
    one is always outside the range, by construction)."""
    in1, in2 = _is_inside(x1, x_min, x_max), _is_inside(x2, x_min, x_max)
    if in1 and not in2:
        return (
            f"x = {x1} lies within the range the line was fitted to (interpolation), "
            f"and x = {x2} lies outside it (extrapolation)."
        )
    if in2 and not in1:
        return (
            f"x = {x2} lies within the range the line was fitted to (interpolation), "
            f"and x = {x1} lies outside it (extrapolation)."
        )
    return (
        f"Both x = {x1} and x = {x2} lie outside the range the line was fitted to, "
        f"so both predictions are by extrapolation."
    )


@generator(
    "p8c_line_best_fit_predict",
    level=12,
    presets={
        "easy": dict(variant="easy"),
        "medium": dict(variant="medium"),
        "hard": dict(variant="hard"),
    },
)
def p8c_line_best_fit_predict(rng, variant="easy"):
    """Use a given line of best fit to predict a value, by interpolation or
    extrapolation. m/c/x are always exact integers; the answer is always a
    plain integer."""

    x_desc, y_desc = rng.choice(CONTEXTS)
    x_min, x_max = _draw_range(rng)
    intro = (
        f"A line of best fit relates {x_desc} (x) to {y_desc} (y), fitted to "
        f"data with x from {x_min} to {x_max}. The line is y = {{m}}x + {{c}}."
    )

    if variant == "easy":
        for _ in range(500):
            m = rng.randint(2, 12)
            c = rng.randint(1, 40)
            x = _draw_x(rng, x_min, x_max, rng.choice([True, False]))
            if x == 1 or m == c:
                continue
            break
        else:
            raise RuntimeError("p8c easy: could not sample avoiding x==1/m==c")

        y = m * x + c
        answer_text = str(y)
        framing = _framing_single(x, x_min, x_max, reveal_x=True)
        text = f"{intro.format(m=m, c=c)} {framing} What value does the line predict for y when x = {x}?"
        working = [
            f"Substitute x = {x} into y = {m}x + {c}.",
            f"y = {m} times {x} plus {c} = {answer_text}.",
        ]

        d_swap = c * x + m  # swapped-gradient-intercept: y = cx + m
        d_omit = m * x  # omits-intercept: y = mx
        d_sign = m * x - c  # sign-error-on-intercept: y = mx - c
        check_distinct(y, d_swap, d_omit, d_sign, msg="easy: a distractor equals the answer")
        distractors = pick_distractors(answer_text, [str(d_swap), str(d_omit), str(d_sign)], n=3)

        # Independent check: the REVERSE direction from how y was produced.
        x_check = Fraction(y - c, m)
        check_equivalent(x_check, Fraction(x), "easy: (y-c)/m must recover the original x")

        return dict(
            text=text,
            latex=f"plain:{text}",
            answer=y,
            answer_text=answer_text,
            answer_latex=answer_text,
            distractors=distractors,
            working=working,
        )

    if variant == "medium":
        for _ in range(500):
            m = rng.randint(2, 12)
            c = rng.randint(1, 40)
            x_target = _draw_x(rng, x_min, x_max, rng.choice([True, False]))
            if x_target == 1 or m == c:
                continue
            y_target = m * x_target + c

            d_swap = round(Fraction(y_target - m, c))  # swapped: x = (y-m)/c
            d_omit = round(Fraction(y_target, m))  # omits-intercept: x = y/m
            d_sign = round(Fraction(y_target + c, m))  # sign-error: x = (y+c)/m
            if len({x_target, d_swap, d_omit, d_sign}) == 4:
                break
        else:
            raise RuntimeError("p8c medium: could not sample avoiding distractor collisions")

        answer_text = str(x_target)
        framing = _framing_single(x_target, x_min, x_max, reveal_x=False)
        text = f"{intro.format(m=m, c=c)} {framing} What value of x does the line predict gives y = {y_target}?"
        working = [
            f"Rearrange y = {m}x + {c} to x = (y - {c}) / {m}.",
            f"x = ({y_target} - {c}) / {m} = {answer_text}.",
        ]

        check_distinct(x_target, d_swap, d_omit, d_sign, msg="medium: a distractor equals the answer")
        distractors = pick_distractors(answer_text, [str(d_swap), str(d_omit), str(d_sign)], n=3)

        # Independent check: substitute the answer back in -- the REVERSE
        # direction from how x_target was used to build y_target.
        check_equivalent(m * x_target + c, y_target, "medium: m*x + c must reproduce y_target")

        return dict(
            text=text,
            latex=f"plain:{text}",
            answer=x_target,
            answer_text=answer_text,
            answer_latex=answer_text,
            distractors=distractors,
            working=working,
        )

    # hard: predicted change over an interval
    for _ in range(500):
        m = rng.randint(2, 12)
        c = rng.randint(1, 40)
        in1 = rng.choice([True, False])
        in2 = rng.choice([True, False]) if not in1 else False  # force >=1 outside
        x1 = _draw_x(rng, x_min, x_max, in1)
        x2 = _draw_x(rng, x_min, x_max, in2)
        if x1 == x2:
            continue
        if x1 > x2:
            x1, x2 = x2, x1
        if x1 == 1 or x2 == 1 or m == c:
            continue

        dx = x2 - x1
        answer = m * dx
        d_swap = c * dx  # swapped-gradient-intercept: c * dx
        d_x2 = m * x2 + c  # uses x2 instead of dx
        d_x1 = m * x1 + c  # uses x1 instead of dx
        if len({answer, d_swap, d_x2, d_x1}) == 4:
            break
    else:
        raise RuntimeError("p8c hard: could not sample avoiding distractor collisions")

    answer_text = str(answer)
    framing = _framing_pair(x1, x2, x_min, x_max)
    text = f"{intro.format(m=m, c=c)} {framing} What is the predicted change in y as x increases from {x1} to {x2}?"
    working = [
        f"The change in y over an interval is the gradient times the change in x: {m} times ({x2} - {x1}).",
        f"= {m} times {dx} = {answer}.",
    ]

    check_distinct(answer, d_swap, d_x2, d_x1, msg="hard: a distractor equals the answer")
    distractors = pick_distractors(answer_text, [str(d_swap), str(d_x2), str(d_x1)], n=3)

    # Independent check: the "slow" route this tier is designed to make
    # unnecessary -- evaluate y1 and y2 separately and subtract.
    y1, y2 = m * x1 + c, m * x2 + c
    check_equivalent(y2 - y1, answer, "hard: y2 - y1 must equal m * (x2 - x1)")

    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        distractors=distractors,
        working=working,
    )
