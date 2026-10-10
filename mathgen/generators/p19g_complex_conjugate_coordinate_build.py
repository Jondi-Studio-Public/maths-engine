"""Scope:
    Given a complex number plotted as the point (x, y) in the complex plane, state
    the coordinates of the point representing its complex conjugate.

One topic: ``p19g_complex_conjugate_coordinate``. The answer is always the ordered
pair ``(x, -y)`` given the plotted point ``(x, y)`` -- reflection of a plotted point
across the real (x) axis.

**Boundary against the shipping neighbour ``complex-conjugate``:** that skill is an ``mcq`` over complex numbers
written in ``a + bi`` SYMBOLIC form ("Find the conjugate of 4 + 6i", options offered
as other ``a + bi`` strings). This skill NEVER writes a complex number in ``a + bi``
form anywhere in its prompt or answer -- every question states the complex number
purely as a plotted point ``(x, y)`` (directly, via a worded description, or via the
conjugate's own plotted point at the exam tier) and the answer is purely a TYPED
coordinate pair ``(x, -y)``, graded by ``answer: pair`` (``mathgen/grading/typed.py::grade_pair``),
never a choice among offered strings. The underlying arithmetic fact (negate the
imaginary part) is shared; the graded deliverable is not.

Both ``x`` and ``y`` are drawn non-zero at every tier and every variant:
- ``y != 0`` is required by the contract itself (a real number's conjugate reflects
  to the identical point, a degenerate case for a skill about *location*).
- ``x != 0`` is a construction discipline this generator adds so the three named
  misconceptions -- negate both ``(-x, -y)``, negate neither ``(x, y)``, negate x
  instead of y ``(-x, y)`` -- stay pairwise distinct from each other and from the
  answer on every draw (at ``x == 0``, "negate x instead of y" collapses onto
  "negate neither": both become ``(0, y)``). This is stricter than the contract
  requires, never looser.

Difficulty progression (genuine reasoning escalation, not magnitude):
- ``easy`` (``variant="direct"``): the point is given directly as ``(x, y)``,
  plain integers, ``x`` always positive (so "reads a stated point and flips the
  sign of y" is the whole task -- no sign-reading burden on x).
- ``medium`` (``variant="mixed_sign_fraction"``): the point is still given
  directly, but ``x`` is always negative and either coordinate may be a simple
  half-integer fraction (e.g. ``3/2``) -- the learner must still ONLY negate y,
  which now takes active care since x is already negative and/or one coordinate
  is a fraction (the "negate both" and "negate neither" distractors are real
  temptations here in a way they are not at ``easy``).
- ``hard`` (``variant="worded"``): the point is stated INDIRECTLY, as a worded
  description ("z is plotted 5 units to the right of the origin and 3 units below
  the real axis") -- genuinely different reasoning: translate words to (x, y)
  first, then reflect, rather than reading a bare coordinate pair.
- ``exam`` (``variant="backward"``): the CONJUGATE's own plotted point is given,
  and the question asks for the ORIGINAL number's coordinates -- the same
  negate-y operation run backwards. Since conjugation is its own inverse,
  reflecting a given point a second time recovers the original, so this is a
  real, differently-framed question (which point is the given one, which is the
  unknown), not a trick and not a bigger version of the forward question.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import (
    check,
    check_equivalent,
    check_pairwise_distinct,
    generator,
    latex_answer,
    numeric_answer,
    phrase,
    pick_distractors,
    rational_text,
)

# --- shared drawing helpers ---------------------------------------------

_FRACTION_NUMERATORS = [1, 3, 5, 7, 9, 11, 13, 15, 17]  # odd/2 -> a "simple fraction"


def _draw_component(rng, *, sign: int | None, allow_fraction: bool) -> Fraction:
    """One non-zero coordinate component: a plain integer 1..9, or (when
    ``allow_fraction``) a half-integer "simple fraction" about 40% of the time.
    ``sign`` forces the sign (+1/-1) when given, else it is drawn 50/50."""
    if allow_fraction and rng.random() < 0.4:
        magnitude = Fraction(rng.choice(_FRACTION_NUMERATORS), 2)
    else:
        magnitude = Fraction(rng.randint(1, 9))
    s = sign if sign is not None else rng.choice([1, -1])
    return s * magnitude


def _point_text(x: Fraction, y: Fraction) -> str:
    return f"({rational_text(x)}, {rational_text(y)})"


def _point_latex(x: Fraction, y: Fraction) -> str:
    return rf"\left({latex_answer(rational_text(x))}, {latex_answer(rational_text(y))}\right)"


def _verify_and_build(gx: Fraction, gy: Fraction, extra_text: str, extra_latex: str = "") -> dict:
    """Shared answer/distractor/verification block, given the coordinates
    ``(gx, gy)`` of whichever point the question presents as the KNOWN one
    (the plotted z for the forward tiers, the plotted conjugate w for the
    backward exam tier). The answer is always ``(gx, -gy)``.

    ``extra_text``/``extra_latex`` are the question's own text/latex (already
    built by the caller); this function only adds the answer/distractor/
    verification fields, matching every other generator's dict-return shape.
    """
    check(gy != 0, "y must never be zero -- a real number's conjugate is degenerate")
    check(gx != 0, "x must never be zero -- keeps the three misconceptions pairwise distinct")

    ans_x, ans_y = gx, -gy
    # Independent re-derivation: negate y a second, symbolic route (never the
    # same Python `-gy` object reused) and confirm x is untouched.
    check_equivalent(ans_y, -1 * gy, "reflected y-coordinate does not equal -y")
    check(ans_x == gx, "x-coordinate must be unchanged by conjugation")

    answer_text = _point_text(ans_x, ans_y)
    answer_latex = _point_latex(ans_x, ans_y)

    # Misconceptions (each checked pairwise distinct from the answer and from
    # each other, every seed, every tier -- guaranteed by gx != 0, gy != 0):
    #  - negates BOTH coordinates: reflects through the origin, not the real axis.
    #  - negates NEITHER: states the given point unchanged.
    #  - negates x instead of y: reflects through the imaginary axis instead.
    negate_both = _point_text(-gx, -gy)
    negate_neither = _point_text(gx, gy)
    negate_x_only = _point_text(-gx, gy)
    check_pairwise_distinct(
        (ans_x, ans_y),
        (-gx, -gy),
        (gx, gy),
        (-gx, gy),
        msg="a misconception distractor collides with the answer or another distractor",
    )

    return dict(
        text=extra_text,
        latex=extra_latex,
        answer={"x": numeric_answer(ans_x), "y": numeric_answer(ans_y)},
        answer_text=answer_text,
        answer_latex=answer_latex,
        working=[
            f"The complex conjugate keeps the real part and negates the imaginary part: "
            f"({rational_text(gx)}, {rational_text(gy)}) -> ({rational_text(gx)}, {rational_text(-gy)}).",
        ],
        distractors=pick_distractors(answer_text, [negate_both, negate_neither, negate_x_only]),
    )


# --- easy / medium: point given directly --------------------------------

_DIRECT_TEMPLATES = [
    "The complex number z is plotted at the point {point} in the complex plane. "
    "State the coordinates of the point representing its complex conjugate.",
    "In the complex plane, z is represented by the point {point}. Find the "
    "coordinates of the point that represents the complex conjugate of z.",
    "A complex number z corresponds to the point {point} in the complex plane. "
    "Give the coordinates of the point representing the complex conjugate of z.",
]


def _direct(rng, *, x_sign: int | None, allow_fraction: bool) -> dict:
    x = _draw_component(rng, sign=x_sign, allow_fraction=allow_fraction)
    y = _draw_component(rng, sign=None, allow_fraction=allow_fraction)
    point = _point_text(x, y)
    text = phrase(rng, _DIRECT_TEMPLATES, point=point)
    return _verify_and_build(x, y, text)


# --- hard: point stated indirectly, as a worded description -------------

_WORDED_TEMPLATES = [
    "The complex number z is plotted {absx} units to the {xdir} of the origin and "
    "{absy} units {ydir} the real axis. State the coordinates of the point "
    "representing its complex conjugate.",
    "z is located {absx} units to the {xdir} of the imaginary axis and {absy} "
    "units {ydir} the real axis in the complex plane. Find the coordinates of "
    "the point representing the complex conjugate of z.",
    "In the complex plane, the point representing z sits {absx} units {xdir} of "
    "the origin and {absy} units {ydir} the real axis. Give the coordinates of "
    "the point representing the complex conjugate of z.",
]


def _worded(rng) -> dict:
    # 2..9, never 1 -- "1 units" is a grammar defect ("units" stays plural in
    # the shared template wording, so the magnitude must not be 1).
    absx = rng.randint(2, 9)
    absy = rng.randint(2, 9)
    x_right = rng.choice([True, False])
    y_above = rng.choice([True, False])
    x = Fraction(absx) if x_right else Fraction(-absx)
    y = Fraction(absy) if y_above else Fraction(-absy)
    xdir = "right" if x_right else "left"
    ydir = "above" if y_above else "below"
    text = phrase(rng, _WORDED_TEMPLATES, absx=absx, absy=absy, xdir=xdir, ydir=ydir)
    return _verify_and_build(x, y, text)


# --- exam: given the CONJUGATE's own plotted point, find the ORIGINAL ---

_BACKWARD_TEMPLATES = [
    "The complex number w is the conjugate of z, and w is plotted at the point "
    "{point} in the complex plane. State the coordinates of the point "
    "representing z.",
    "In the complex plane, w denotes the complex conjugate of z. The point "
    "representing w is {point}. Find the coordinates of the point representing z.",
    "z is a complex number whose conjugate, w, is plotted at {point} in the "
    "complex plane. Give the coordinates of the point representing z.",
]


def _backward(rng) -> dict:
    # (p, q) is w's plotted point -- the KNOWN quantity here. z (the answer) is
    # (p, -q): the same negate-the-second-component operation, run on the
    # point the question actually gives, which is w's, not z's.
    p = _draw_component(rng, sign=None, allow_fraction=True)
    q = _draw_component(rng, sign=None, allow_fraction=True)
    point = _point_text(p, q)
    text = phrase(rng, _BACKWARD_TEMPLATES, point=point)
    return _verify_and_build(p, q, text)


@generator(
    "p19g_complex_conjugate_coordinate",
    level=11,
    presets={
        "easy": dict(variant="direct"),
        "medium": dict(variant="mixed_sign_fraction"),
        "hard": dict(variant="worded"),
        "exam": dict(variant="backward"),
    },
)
def complex_conjugate_coordinate(rng, variant="direct"):
    if variant == "direct":
        return _direct(rng, x_sign=1, allow_fraction=False)
    if variant == "mixed_sign_fraction":
        return _direct(rng, x_sign=-1, allow_fraction=True)
    if variant == "worded":
        return _worded(rng)
    if variant == "backward":
        return _backward(rng)
    raise ValueError(f"unknown variant: {variant!r}")
