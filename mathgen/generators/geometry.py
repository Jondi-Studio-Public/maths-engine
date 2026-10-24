"""Levels 11-12: trigonometry (Pythagoras, angles, sine rule), vectors."""

import math
import re

from sympy import (
    Matrix,
    sqrt,
    Integer,
    atan2,
    deg,
    N,
    sin as ssin,
    cos as scos,
    asin as sasin,
    acos as sacos,
    rad,
    Rational,
)

from .. import figures
from ..core import (
    attempts,
    latex_prompt,
    generator,
    check,
    pick_distractors,
    GRADER_TOLERANCE,
    latex_answer,
    check_distinct,
    rounds_unambiguously,
    stated_precision_tolerance,
    phrase,
)
from ..figure_kinds import space
from . import _space_kit as kit

# Primitive Pythagorean triples (a, b, c) kept small for clean numbers. These
# exist so side_hyp/side_leg's answers come out as whole numbers -- they are
# correct FOR THOSE TWO KINDS specifically. "angle" does not need a triple at
# all (its answer is already rounded to 1 d.p. and a calculator is allowed),
# and scaling a triple by k never changes the angle atan/asin/acos derive
# from it -- see `_free_angle` below, which "angle" uses instead
# whenever a skill opts in via `angle_style`.
TRIPLES = [(3, 4, 5), (5, 12, 13), (8, 15, 17), (7, 24, 25), (20, 21, 29), (9, 40, 41)]

# Angle pool for sine-rule questions (degrees).
ANGLES = [30, 40, 45, 50, 60, 70, 75, 80]


@generator(
    "trigonometry",
    level=11,
    presets={
        "easy": dict(kinds=("side_hyp", "side_leg"), k_hi=1),
        "medium": dict(kinds=("side_hyp", "side_leg", "angle"), k_hi=3),
        # k_lo=4: when a skill pins `kinds` to one value, medium and hard would otherwise both draw
        # `k` from the identical randint(1, 3) pool -- 18 outcomes (6 TRIPLES x k) shared between
        # two independently seeded difficulty tiers -- and coincide at the ~1/18 birthday rate
        # (confirmed: 4/60 collisions per skill). k_lo=4/k_hi=6 makes hard's pool {4,5,6} disjoint
        # from medium's {1,2,3}, so a scaled TRIPLES triple can never repeat across tiers.
        "hard": dict(kinds=("side_hyp", "side_leg", "angle", "sine_rule"), k_lo=4, k_hi=6),
        # Additive only -- "easy"/"medium"/"hard" above are untouched. `word_problem=True` is what
        # pythagorean-hypotenuse/ pythagorean-leg's own `exam` SOURCE opts into (each pins its own
        # `kinds` to one value). k_hi kept small because the exam tier's own values are read inside
        # a real-world scenario (a mast, a kite string, a drone) -- k_hi=6/k_hi=9 on the largest
        # TRIPLES entry would print a 240-370 m guy-wire, which a subject teacher reads as nonsense
        # however correct the arithmetic is.
        "exam": dict(kinds=("side_hyp", "side_leg"), k_lo=1, k_hi=2, word_problem=True),
    },
)
def trigonometry(
    rng,
    kinds=("side_hyp", "side_leg", "angle", "sine_rule"),
    k_hi=3,
    k_lo=1,
    angle_style="triple",
    angle_lo=3,
    angle_hi=60,
    exact_triple: bool = True,
    word_problem: bool = False,
    figure: bool = False,
    units: bool = False,
):
    """Right-triangle and sine-rule questions.

    With ``figure=True`` the ``side_hyp`` and ``side_leg`` kinds draw the triangle instead of describing it in
    words: a right triangle in one of twelve orientations with two sides labelled and the third marked with a
    letter, "not drawn to scale" and with the drawn leg proportions unrelated to the answer (see
    :func:`_side_figure`). ``units=True`` adds a length unit (cm, m or mm) to every label and to the prompt. The
    other kinds, and the word-problem form, ignore both. The words-only form (``figure=False``) is unchanged.
    """
    if not figure:
        return _trigonometry(
            rng, kinds, k_hi, k_lo, angle_style, angle_lo, angle_hi, exact_triple, word_problem, False, False
        )
    for _ in attempts(60, "a triangle picture that does not show its own answer"):
        try:
            return _trigonometry(
                rng, kinds, k_hi, k_lo, angle_style, angle_lo, angle_hi, exact_triple, word_problem, True, units
            )
        except _ShowsAnswer:
            continue
    raise AssertionError("unreachable")


class _ShowsAnswer(Exception):
    """A drawn triangle whose labels would show the answer; the caller draws another."""


def _legs_to_draw(rng, value_a, value_b):
    """Drawn leg lengths (a, b) for a not-to-scale right triangle: a ratio of 1.15 to 1.8 either way, the longer
    labelled leg drawn longer when the two labels differ by a quarter or more (the picture must not contradict
    them), and otherwise unrelated to the numbers, so the drawing is no help in finding the answer."""
    r = round(rng.uniform(1.15, 1.8), 3)
    if value_a is not None and value_b is not None:
        if value_a >= value_b * 1.25:
            return r, 1
        if value_b >= value_a * 1.25:
            return 1, r
    return (r, 1) if rng.random() < 0.5 else (1, r)


_SIDE_UNKNOWNS = ("x", "y", "z", "d", "s", "t")
_UNIT_NAMES = {"cm": "centimetres", "m": "metres", "mm": "millimetres"}
_HYP_PHRASES = (
    "In the right-angled triangle shown, find the length of the hypotenuse, {u}.",
    "Find the length of the hypotenuse {u} in the right-angled triangle shown.",
    "The diagram shows a right-angled triangle. Work out the length of the hypotenuse, {u}.",
    "Use the right-angled triangle shown to calculate the length of the hypotenuse, {u}.",
)
_LEG_PHRASES = (
    "In the right-angled triangle shown, find the length of the leg marked {u}.",
    "Find the length of the missing leg {u} in the right-angled triangle shown.",
    "The diagram shows a right-angled triangle. Work out the length of the side marked {u}.",
    "Use the right-angled triangle shown to calculate the length of the leg {u}.",
)


def _side_figure(rng, kind, a, b, c, answer_text, places, units):
    """(figure, prompt) for ``side_hyp`` (legs a, b given, hypotenuse unknown) or ``side_leg`` (hypotenuse c and
    leg a given, the other leg unknown). ``a`` is the given leg for ``side_leg``; ``b`` is unused there. The
    unknown side carries a letter; with ``units`` every length carries cm, m or mm and the prompt says which."""
    letter = rng.choice(_SIDE_UNKNOWNS)
    unit = rng.choice(tuple(_UNIT_NAMES)) if units else None

    def show(v):
        return f"{v} {unit}" if unit else str(v)

    if kind == "side_hyp":
        labels = {"a": show(a), "b": show(b), "c": letter}
        legs = _legs_to_draw(rng, a, b)
        phrase_pool = _HYP_PHRASES
    else:
        labels = {"c": show(c)}
        known, unknown = ("a", "b") if rng.random() < 0.5 else ("b", "a")
        labels[known], labels[unknown] = show(a), letter
        legs = _legs_to_draw(rng, None, None)
        phrase_pool = _LEG_PHRASES
    fig = space.right_triangle(
        legs=legs,
        labels=labels,
        orientation=rng.randrange(12),
        alt="A right-angled triangle with the lengths of two sides marked and the third side marked with a letter.",
    )
    if kit.leaks(fig, answer_text):
        raise _ShowsAnswer
    text = rng.choice(phrase_pool).format(u=letter)
    if places:
        text = text.rstrip(".") + ", to 1 decimal place."
    if unit:
        text += f" All lengths are in {_UNIT_NAMES[unit]}."
    return fig, text


def _read_right_triangle(question) -> str:
    """Re-derive the unknown side from the picture alone: the sides meeting at the right-angle mark are the
    legs, the third is the hypotenuse, and the unknown is the side whose label is a bare letter."""
    fig = question.figure
    right = [a for a in fig["angles"] if a.get("right")]
    check(len(right) == 1, "expected one right angle")
    corner = right[0]["at"]
    legs, hyp = [], None
    for seg in fig["segments"]:
        if corner in (seg["from"], seg["to"]):
            legs.append(seg)
        else:
            hyp = seg
    check(len(legs) == 2 and hyp is not None, "expected two legs and a hypotenuse")

    def length(seg):
        return float(seg["label"].split()[0])

    def unknown(seg):
        return seg["label"].split()[0].isalpha()

    if unknown(hyp):
        total = sum(length(leg) ** 2 for leg in legs)
    else:
        missing = [leg for leg in legs if unknown(leg)]
        known = [leg for leg in legs if not unknown(leg)]
        check(len(missing) == 1 and len(known) == 1, "expected exactly one side marked with a letter")
        total = length(hyp) ** 2 - length(known[0]) ** 2
    value = math.sqrt(total)
    return str(round(value)) if abs(value - round(value)) < 1e-9 else f"{value:.1f}"


figures.register_reader("trigonometry", _read_right_triangle)


def _trigonometry(
    rng,
    kinds,
    k_hi,
    k_lo,
    angle_style,
    angle_lo,
    angle_hi,
    exact_triple,
    word_problem,
    figure,
    units,
):
    kind = rng.choice(list(kinds))
    if kind == "sine_rule":
        # b = a * sin(B) / sin(A), rounded to 1 d.p.
        A_deg, B_deg = rng.sample(ANGLES, 2)
        a_side = rng.randint(5, 20)
        exact = a_side * math.sin(math.radians(B_deg)) / math.sin(math.radians(A_deg))
        ans = round(exact, 1)
        sym = float(N(a_side * ssin(rad(B_deg)) / ssin(rad(A_deg))))
        check(abs(sym - ans) < 0.05)
        # Misconceptions: flipped the ratio; forgot to divide; premature rounding.
        flipped = round(a_side * math.sin(math.radians(A_deg)) / math.sin(math.radians(B_deg)), 1)
        no_div = round(a_side * math.sin(math.radians(B_deg)), 1)
        cands = [f"b = {flipped}", f"b = {no_div}", f"b = {float(round(exact)):.1f}"]
        return dict(
            text=(
                f"In triangle ABC, angle A = {A_deg}°, angle B = {B_deg}° and side a = {a_side}. "
                f"Use the sine rule to find side b (1 d.p.)."
            ),
            latex=rf"A = {A_deg}^\circ,\ B = {B_deg}^\circ,\ a = {a_side}.\ \text{{Find }} b\ \text{{(1 d.p.)}}",
            answer=ans,
            answer_text=f"b = {ans}",
            answer_latex=f"b = {ans}",
            working=[f"b = a·sin(B)/sin(A) = {a_side}·sin({B_deg}°)/sin({A_deg}°) = {ans}"],
            distractors=pick_distractors(f"b = {ans}", cands),
        )
    if kind == "cosine_rule":
        # c^2 = a^2 + b^2 - 2ab cos(C), rounded to 1 d.p.
        a_side = rng.randint(5, 20)
        b_side = rng.randint(5, 20)
        C_deg = rng.choice(ANGLES)
        exact = math.sqrt(a_side**2 + b_side**2 - 2 * a_side * b_side * math.cos(math.radians(C_deg)))
        ans = round(exact, 1)
        sym = float(N(sqrt(Integer(a_side) ** 2 + Integer(b_side) ** 2 - 2 * a_side * b_side * scos(rad(C_deg)))))
        # Compare against the unrounded value -- comparing to the rounded
        # `ans` instead risks float noise crossing the rounding boundary for
        # "nice" angles (e.g. 30 deg) where sympy's exact trig value and
        # math.sin's float approximation land on opposite sides of *.x5.
        check(abs(sym - exact) < 1e-6)
        # Misconceptions: forgot the -2ab·cos(C) term (plain Pythagoras); sign
        # error on that term (+2ab·cos(C)); used sin instead of cos.
        no_cos_term = round(math.sqrt(a_side**2 + b_side**2), 1)
        sign_sq = a_side**2 + b_side**2 + 2 * a_side * b_side * math.cos(math.radians(C_deg))
        sin_sq = a_side**2 + b_side**2 - 2 * a_side * b_side * math.sin(math.radians(C_deg))
        cands = [f"c = {no_cos_term}"]
        if sign_sq >= 0:
            cands.append(f"c = {round(math.sqrt(sign_sq), 1)}")
        if sin_sq >= 0:
            cands.append(f"c = {round(math.sqrt(sin_sq), 1)}")
        return dict(
            text=(
                f"In triangle ABC, side a = {a_side}, side b = {b_side}, and the included angle "
                f"C = {C_deg}°. Use the cosine rule to find side c (1 d.p.)."
            ),
            latex=rf"a = {a_side},\ b = {b_side},\ C = {C_deg}^\circ.\ \text{{Find }} c\ \text{{(1 d.p.)}}",
            answer=ans,
            answer_text=f"c = {ans}",
            answer_latex=f"c = {ans}",
            working=[
                f"c² = a² + b² - 2ab·cos(C) = {a_side}² + {b_side}² - 2×{a_side}×{b_side}×cos({C_deg}°)",
                f"c = {ans}",
            ],
            distractors=pick_distractors(f"c = {ans}", cands),
        )
    if kind == "triangle_area":
        # Area = (1/2) a b sin(C), rounded to 1 d.p.
        a_side = rng.randint(4, 15)
        b_side = rng.randint(4, 15)
        C_deg = rng.choice(ANGLES)
        exact = 0.5 * a_side * b_side * math.sin(math.radians(C_deg))
        ans = round(exact, 1)
        sym = float(N(Rational(1, 2) * a_side * b_side * ssin(rad(C_deg))))
        check(abs(sym - exact) < 1e-6)
        # Misconceptions: forgot the 1/2; used cos instead of sin; multiplied
        # the sides only.
        no_half = round(a_side * b_side * math.sin(math.radians(C_deg)), 1)
        used_cos = round(0.5 * a_side * b_side * math.cos(math.radians(C_deg)), 1)
        cands = [f"Area = {no_half}", f"Area = {used_cos}", f"Area = {float(a_side * b_side):.1f}"]
        return dict(
            text=(
                f"Triangle ABC has side a = {a_side}, side b = {b_side}, and included angle "
                f"C = {C_deg}°. Find its area (1 d.p.)."
            ),
            latex=rf"a = {a_side},\ b = {b_side},\ C = {C_deg}^\circ.\ \text{{Find area}}\ \text{{(1 d.p.)}}",
            answer=ans,
            answer_text=f"Area = {ans}",
            answer_latex=latex_answer(f"Area = {ans}", plain=True),
            working=[f"Area = ½ab·sin(C) = 0.5×{a_side}×{b_side}×sin({C_deg}°) = {ans}"],
            distractors=pick_distractors(f"Area = {ans}", cands),
        )
    if kind == "triangle_area_reverse":
        # REVERSE of "triangle_area" above: given the area, one side, and
        # the included angle, find the OTHER side -- genuinely different
        # unknown from "triangle_area", which always gives both sides and
        # asks for the area.
        for _ in range(50):
            a_side = rng.randint(4, 15)
            b_side = rng.randint(4, 15)
            C_deg = rng.choice(ANGLES)
            exact_area = 0.5 * a_side * b_side * math.sin(math.radians(C_deg))
            area = round(exact_area, 1)
            exact_b = 2 * area / (a_side * math.sin(math.radians(C_deg)))
            ans = round(exact_b, 1)
            if abs(ans - b_side) < 0.05:
                break
        sym = float(N(2 * Rational(int(round(area * 10)), 10) / (a_side * ssin(rad(C_deg)))))
        check(abs(sym - exact_b) < 0.05)
        # Misconceptions: forgot the factor of 2 (used Area = a·b·sin C
        # directly, so divided by 2ab instead of ab); used cos instead of
        # sin; divided the area by the side alone, ignoring sin(C) entirely.
        no_double = round(area / (a_side * math.sin(math.radians(C_deg))), 1)
        cos_val = math.cos(math.radians(C_deg))
        cands = [f"b = {no_double}"]
        if abs(cos_val) > 1e-6:
            used_cos = round(2 * area / (a_side * cos_val), 1)
            cands.append(f"b = {used_cos}")
        cands.append(f"b = {round(area / a_side, 1)}")
        return dict(
            text=(
                f"Triangle ABC has side a = {a_side}, included angle C = {C_deg}°, and area "
                f"{area}. Find side b (1 d.p.)."
            ),
            latex=rf"a = {a_side},\ C = {C_deg}^\circ,\ \text{{Area}} = {area}.\ \text{{Find }} b\ \text{{(1 d.p.)}}",
            answer=ans,
            answer_text=f"b = {ans}",
            answer_latex=latex_answer(f"b = {ans}", plain=True),
            working=[f"b = 2×Area / (a·sin C) = 2×{area} / ({a_side}×sin({C_deg}°)) = {ans}"],
            distractors=pick_distractors(f"b = {ans}", cands),
        )
    if kind == "angle" and angle_style != "triple":
        # Right-triangle-angle's answer is an angle to 1 d.p., not a whole number, so unlike
        # side_hyp/side_leg it does not need a Pythagorean triple at all -- atan is scale-invariant,
        # so scaling a triple by k (the mechanism side_hyp/side_leg legitimately need for
        # whole-number answers) never changes the angle, capping this kind's answer pool at 6 values
        # regardless of how many (triple, k) pairs are drawn. Arbitrary integer legs make the pool
        # effectively unbounded. This branch is opt-in (`angle_style` defaults to "triple", the
        # original behaviour below) so side_hyp/side_leg and any bare `make("trigonometry", seed)`
        # call are untouched.
        return _free_angle(rng, angle_style, angle_lo, angle_hi)
    if kind in ("side_hyp", "side_leg") and not exact_triple:
        return _side_inexact(rng, kind, figure, units)
    a, b, c = rng.choice(TRIPLES)
    k = rng.randint(k_lo, k_hi)
    a, b, c = a * k, b * k, c * k
    if kind in ("side_hyp", "side_leg") and word_problem:
        return _side_word_problem(rng, kind, a, b, c)
    if kind == "side_hyp":
        # Given both legs, find the hypotenuse.
        check(sqrt(Integer(a) ** 2 + Integer(b) ** 2) == c)
        d = dict(
            text=(f"A right-angled triangle has legs of length {a} and {b}. Find the length of the hypotenuse."),
            latex=rf"\text{{Legs }} {a} \text{{ and }} {b}\text{{: find the hypotenuse.}}",
            answer=c,
            answer_text=str(c),
            answer_latex=str(c),
            working=[f"h = √({a}² + {b}²) = √{a * a + b * b} = {c}"],
            # Misconceptions: added the sides; forgot the square root; subtracted.
            distractors=pick_distractors(str(c), [str(a + b), str(a * a + b * b), str(abs(b - a))]),
        )
        if figure:
            d["figure"], d["text"] = _side_figure(rng, kind, a, b, c, str(c), False, units)
            d["latex"] = latex_prompt(d["text"])
        return d
    if kind == "side_leg":
        # Given hypotenuse and one leg, find the other leg.
        check(sqrt(Integer(c) ** 2 - Integer(a) ** 2) == b)
        d = dict(
            text=(f"A right-angled triangle has hypotenuse {c} and one leg of length {a}. Find the other leg."),
            latex=rf"\text{{Hypotenuse }} {c}\text{{, leg }} {a}\text{{: find the other leg.}}",
            answer=b,
            answer_text=str(b),
            answer_latex=str(b),
            working=[f"l = √({c}² - {a}²) = √{c * c - a * a} = {b}"],
            # Misconceptions: forgot the square root; subtracted/added lengths not squares.
            distractors=pick_distractors(str(b), [str(c * c - a * a), str(c - a), str(c + a)]),
        )
        if figure:
            d["figure"], d["text"] = _side_figure(rng, kind, a, b, c, str(b), False, units)
            d["latex"] = latex_prompt(d["text"])
        return d
    # Angle opposite side a, to 1 d.p.
    ans = round(math.degrees(math.atan2(a, b)), 1)
    check(abs(float(N(deg(atan2(Integer(a), Integer(b))))) - ans) < 0.05)
    # Misconceptions: flipped the ratio (gives the co-angle); used sin or cos
    # with the same two sides (a < b always holds for our triples, so valid).
    cands = [
        f"θ = {round(90 - ans, 1)}°",
        f"θ = {round(math.degrees(math.asin(a / b)), 1)}°",
        f"θ = {round(math.degrees(math.acos(a / b)), 1)}°",
    ]
    return dict(
        text=(
            f"A right-angled triangle has opposite side {a} and adjacent side {b}. "
            f"Find the angle θ in degrees (1 d.p.)."
        ),
        latex=rf"\text{{opposite }} {a}\text{{, adjacent }} {b}\text{{: find }} \theta \text{{ (1 d.p.)}}",
        answer=ans,
        answer_text=f"θ = {ans}°",
        answer_latex=rf"\theta = {ans}^\circ",
        working=[f"θ = tan⁻¹({a}/{b}) = {ans}°"],
        distractors=pick_distractors(f"θ = {ans}°", cands),
    )


# An angle kept away from 0/90 degrees keeps the wrong-ratio distractors
# (which plug the same ratio into the wrong inverse function) resolved from
# the true answer at 1 d.p. -- near the extremes, sin/tan or cos/tan of a
# small ratio round to the same tenth of a degree and a distractor silently
# disappears. Re-drawing is cheap and keeps every draw usable.
_ANGLE_LO_DEG, _ANGLE_HI_DEG = 10.0, 80.0
_ANGLE_MAX_TRIES = 200


def _free_angle(rng, style, lo, hi):
    """Replacement for the triple-based "angle" kind: draw
    arbitrary integer side lengths (no Pythagorean triple, so atan/asin/acos
    are not scale-locked to six possible answers) and find the angle to
    1 d.p. with a calculator-level inverse trig ratio.

    `style` selects which ratio(s) are in play -- "legs" always uses tan
    (both legs given, easy tier); "leg_hyp" always uses sin or cos (a leg and
    the hypotenuse given, so the learner must choose the right ratio instead
    of defaulting to tan, medium tier); "mixed" draws from all three (hard
    tier, also given the widest `lo`/`hi` band of the three tiers).

    `right-triangle-angle.yaml`'s exam source uses `angle_style: elevation_eye`
    (the opposite side must first be obtained by subtracting the observer's
    eye height, which changes the numbers that go into the ratio)."""
    if style == "elevation_eye":
        return _angle_elevation_eye(rng)
    if style == "legs":
        ratio = "tan"
    elif style == "leg_hyp":
        ratio = rng.choice(("sin", "cos"))
    elif style == "mixed":
        ratio = rng.choice(("tan", "sin", "cos"))
    else:
        raise ValueError(f"unknown angle_style {style!r}")

    if ratio == "tan":
        for _ in range(_ANGLE_MAX_TRIES):
            opp = rng.randint(lo, hi)
            adj = rng.randint(lo, hi)
            if adj == opp:
                continue
            exact = math.degrees(math.atan2(opp, adj))
            if _ANGLE_LO_DEG <= exact <= _ANGLE_HI_DEG:
                break
        ans = round(exact, 1)
        check(abs(float(N(deg(atan2(Integer(opp), Integer(adj))))) - ans) < 0.05)
        lo_leg, hi_leg = (opp, adj) if opp < adj else (adj, opp)
        wrong_ratio = lo_leg / hi_leg
        # Misconceptions: complementary angle (flipped the ratio); applied
        # sin or cos to the same two legs as if one of them were the
        # hypotenuse.
        cands = [
            f"θ = {round(90 - ans, 1)}°",
            f"θ = {round(math.degrees(math.asin(wrong_ratio)), 1)}°",
            f"θ = {round(math.degrees(math.acos(wrong_ratio)), 1)}°",
        ]
        return dict(
            text=(
                f"A right-angled triangle has opposite side {opp} and adjacent side {adj}. "
                f"Find the angle θ in degrees (1 d.p.)."
            ),
            latex=(
                rf"\text{{opposite }} {opp}\text{{, adjacent }} {adj}"
                rf"\text{{: find }} \theta \text{{ (1 d.p.)}}"
            ),
            answer=ans,
            answer_text=f"θ = {ans}°",
            answer_latex=rf"\theta = {ans}^\circ",
            working=[f"θ = tan⁻¹({opp}/{adj}) = {ans}°"],
            distractors=pick_distractors(f"θ = {ans}°", cands, tolerance=GRADER_TOLERANCE),
        )

    # sin or cos: given one leg and the hypotenuse.
    for _ in range(_ANGLE_MAX_TRIES):
        leg = rng.randint(lo, hi)
        hyp = leg + rng.randint(1, hi)
        ratio_val = leg / hyp
        exact = math.degrees(math.asin(ratio_val)) if ratio == "sin" else math.degrees(math.acos(ratio_val))
        if _ANGLE_LO_DEG <= exact <= _ANGLE_HI_DEG:
            break
    ans = round(exact, 1)
    if ratio == "sin":
        sym = float(N(deg(sasin(Rational(leg, hyp)))))
        side_word, inv_sym = "opposite", "sin⁻¹"
    else:
        sym = float(N(deg(sacos(Rational(leg, hyp)))))
        side_word, inv_sym = "adjacent to", "cos⁻¹"
    check(abs(sym - exact) < 1e-6)
    # Misconceptions: complementary angle; misapplied tan to the leg and the
    # hypotenuse as if they were the triangle's two legs (both ways round).
    cands = [
        f"θ = {round(90 - ans, 1)}°",
        f"θ = {round(math.degrees(math.atan2(leg, hyp)), 1)}°",
        f"θ = {round(math.degrees(math.atan2(hyp, leg)), 1)}°",
    ]
    return dict(
        text=(
            f"A right-angled triangle has hypotenuse {hyp} and the side {side_word} "
            f"angle θ is {leg}. Find θ in degrees (1 d.p.)."
        ),
        latex=(
            rf"\text{{hypotenuse }} {hyp}\text{{, {side_word} }} {leg}"
            rf"\text{{: find }} \theta \text{{ (1 d.p.)}}"
        ),
        answer=ans,
        answer_text=f"θ = {ans}°",
        answer_latex=rf"\theta = {ans}^\circ",
        working=[f"θ = {inv_sym}({leg}/{hyp}) = {ans}°"],
        distractors=pick_distractors(f"θ = {ans}°", cands, tolerance=GRADER_TOLERANCE),
    )


# The hard-tier fix for side_hyp/side_leg. Drawn INDEPENDENTLY of TRIPLES/k -- a scaled triple
# always sums/differences to a perfect square by construction, so the only way to force a genuinely
# irrational third side is to stop drawing from that table. Kept modest so the rounded answer stays
# a comfortable two-digit number; `side_leg` draws the hypotenuse from double this band so `c -
# leg_lo > 0` has room and the resulting leg is rarely tiny.
_INEXACT_LEG_LO, _INEXACT_LEG_HI = 6, 28
_INEXACT_HYP_LO, _INEXACT_HYP_HI = 20, 55
_INEXACT_MAX_TRIES = 200


def _angle_elevation_eye(rng):
    """Exam tier: angle of elevation of a tower's top from a
    student's eyes. The opposite side is H - e (tower height less eye height),
    so the subtraction changes which numbers enter tan; it is not a detour.
    The yaml's angle_lo/angle_hi are ignored here: H, d and e have their own
    ranges. Lengths are held in tenths of a metre so H - e has no float noise."""
    for _ in range(_ANGLE_MAX_TRIES):
        tower = rng.randint(10, 40)
        dist = rng.randint(15, 60)
        eye = rng.choice((1.4, 1.5, 1.6, 1.7, 1.8))
        eye10 = round(eye * 10)
        opp10 = tower * 10 - eye10
        exact = math.degrees(math.atan2(opp10, dist * 10))
        if not _ANGLE_LO_DEG <= exact <= _ANGLE_HI_DEG:
            continue
        ans = round(exact, 1)
        forgot = round(math.degrees(math.atan2(tower * 10, dist * 10)), 1)
        added = round(math.degrees(math.atan2(tower * 10 + eye10, dist * 10)), 1)
        comp = round(90 - ans, 1)
        if len({ans, forgot, added, comp}) < 4:
            continue
        break
    else:
        check(False)
    check(abs(float(N(deg(atan2(Rational(opp10, 10), Integer(dist))))) - ans) < 0.05)
    opp_text = f"{opp10 // 10}.{opp10 % 10}"
    # Misconceptions: forgot the eye height; added it instead of subtracting;
    # complementary angle (flipped the ratio).
    cands = [f"θ = {forgot}°", f"θ = {added}°", f"θ = {comp}°"]
    return dict(
        text=(
            f"A student whose eyes are {eye} m above the ground stands {dist} m from the base of "
            f"a vertical tower that is {tower} m tall. Find the angle of elevation of the top of "
            f"the tower from the student's eyes (1 d.p.)."
        ),
        latex=(
            rf"\text{{eye height }} {eye}\text{{ m, distance }} {dist}\text{{ m, tower }} {tower}"
            rf"\text{{ m: find the angle of elevation (1 d.p.)}}"
        ),
        answer=ans,
        answer_text=f"θ = {ans}°",
        answer_latex=rf"\theta = {ans}^\circ",
        working=[
            f"opposite = {tower} - {eye} = {opp_text} m",
            f"θ = tan⁻¹({opp_text}/{dist}) = {ans}°",
        ],
        distractors=pick_distractors(f"θ = {ans}°", cands, tolerance=GRADER_TOLERANCE),
    )


def _side_inexact(rng, kind, figure=False, units=False):
    """Hard tier for side_hyp/side_leg: the two given lengths are
    drawn INDEPENDENTLY of TRIPLES, so the true third side is (almost
    always) irrational and must be rounded to 1 d.p. -- a genuinely
    different reasoning step from every lower tier's always-exact scaled
    triple, not just a bigger version of the same shape.

    Guards, in order, on a plain bounded resample (this draws two small
    integers, not a truss grid, so a `for`/`while` with a `check(tries <=
    N, ...)` fallback is cheap):
      - an accidental perfect square, which would silently make the draw an
        EXACT answer in disguise -- the one thing this tier must not be;
      - the ".x5" rounding boundary (`rounds_unambiguously`), same
        discipline as `p7a_trig_ratio_elevation_build.py`'s
        `_on_rounding_boundary`;
      - a rounded value that DISPLAYS as a whole number (`...0` at 1 d.p.),
        so the decimal-ness that makes this tier a new reasoning shape is
        always visible in `answer_text`, never accidentally hidden.
    """
    radicand = c = a = b = None
    exact = None
    for _ in range(_INEXACT_MAX_TRIES):
        if kind == "side_hyp":
            a = rng.randint(_INEXACT_LEG_LO, _INEXACT_LEG_HI)
            b = rng.randint(_INEXACT_LEG_LO, _INEXACT_LEG_HI)
            radicand = a * a + b * b
        else:  # side_leg
            c = rng.randint(_INEXACT_HYP_LO, _INEXACT_HYP_HI)
            a = rng.randint(_INEXACT_LEG_LO, c - 1)
            radicand = c * c - a * a
        root = math.isqrt(radicand)
        if root * root == radicand:
            continue  # accidental perfect square -- not the shape we want
        exact = math.sqrt(radicand)
        if not rounds_unambiguously(exact, 1):
            continue
        if f"{round(exact, 1):.1f}".endswith(".0"):
            continue
        break
    else:
        check(False, "trigonometry inexact side: no non-degenerate draw found")
    ans = round(exact, 1)
    # Independent route: sympy's own sqrt (exact, symbolic), not a second
    # call to Python's math.sqrt -- the same discipline cosine_rule/
    # triangle_area above already use for their own rounded answers.
    sym = float(N(sqrt(Integer(radicand))))
    check(abs(sym - exact) < 1e-9, "inexact side: sympy/math sqrt disagree")

    answer_text = f"{ans:.1f}"
    if kind == "side_hyp":
        text = (
            f"A right-angled triangle has legs of length {a} and {b}. "
            f"Find the length of the hypotenuse, to 1 decimal place."
        )
        latex = (
            rf"\text{{Legs }} {a} \text{{ and }} {b}"
            rf"\text{{: find the hypotenuse, to 1 decimal place.}}"
        )
        working = [f"h = √({a}² + {b}²) = √{radicand} = {ans}"]
        # Misconceptions: added the legs; forgot the square root (the
        # radicand, un-rooted); subtracted the legs -- the same three as the
        # exact tiers, reframed as decimals so every option is formatted
        # alike (no option is the only decimal on offer).
        cands = [f"{float(a + b):.1f}", f"{float(radicand):.1f}", f"{float(abs(b - a)):.1f}"]
    else:
        text = (
            f"A right-angled triangle has hypotenuse {c} and one leg of length {a}. "
            f"Find the other leg, to 1 decimal place."
        )
        latex = (
            rf"\text{{Hypotenuse }} {c}\text{{, leg }} {a}"
            rf"\text{{: find the other leg, to 1 decimal place.}}"
        )
        working = [f"l = √({c}² - {a}²) = √{radicand} = {ans}"]
        cands = [f"{float(radicand):.1f}", f"{float(c - a):.1f}", f"{float(c + a):.1f}"]
    check_distinct(ans, *cands)
    d = dict(
        text=text,
        latex=latex,
        answer=ans,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=pick_distractors(answer_text, cands, tolerance=stated_precision_tolerance(1)),
    )
    if figure:
        d["figure"], d["text"] = _side_figure(rng, kind, a, b, c, answer_text, True, units)
        d["latex"] = latex_prompt(d["text"])
    return d


# Several phrasings per skill so a `prompt_shape` sweep sees genuine sentence variety rather than
# only the drawn numbers -- `{a}`/`{b}`/`{c}` are filled by `phrase`.
_HYP_EXAM_CONTEXTS = (
    "A guy-wire supports a radio mast. It is anchored to the ground {a} m "
    "from the base of the mast and attached to the mast at a point {b} m "
    "above the ground. Find the length of the guy-wire.",
    "A zip-line tower is {b} m tall. A support cable runs from the top of "
    "the tower to an anchor point on the ground {a} m from the base of the "
    "tower. Find the length of the support cable.",
    "A drone is flying {b} m directly above a point on the ground that is "
    "{a} m from its pilot. Find the straight-line distance from the pilot "
    "to the drone.",
)
_LEG_EXAM_CONTEXTS = (
    "A kite string, {c} m long, is anchored to the ground. The kite is "
    "flying directly above a point on the ground that is {a} m from the "
    "anchor. Find how high the kite is flying.",
    "A zip-line cable, {c} m long, runs from a platform down to the ground. "
    "The point where the cable meets the ground is {a} m from the base of "
    "the platform. Find the height of the platform.",
    "A cable connects the top of a tower to an anchor point on the ground. "
    "The cable is {c} m long and the anchor is {a} m from the base of the "
    "tower. Find the height of the tower.",
)


def _side_word_problem(rng, kind, a, b, c):
    """Exam tier for side_hyp/side_leg. `a, b, c` are the SAME
    scaled-triple draw `trigonometry` already made (legs a, b; hypotenuse
    c) -- re-verified here independently of that draw's own check, since
    this branch returns before the original `side_hyp`/`side_leg` blocks'
    own `check(...)` calls run."""
    if kind == "side_hyp":
        check(sqrt(Integer(a) ** 2 + Integer(b) ** 2) == c)
        text = phrase(rng, _HYP_EXAM_CONTEXTS, a=a, b=b)
        answer, answer_text = c, str(c)
        working = [f"h = √({a}² + {b}²) = √{a * a + b * b} = {c}"]
        # Same three misconceptions as the exact tiers (added-legs,
        # forgot-sqrt, subtracted), reframed for the word problem's own
        # scenario rather than "legs"/"hypotenuse" vocabulary.
        cands = [str(a + b), str(a * a + b * b), str(abs(b - a))]
    else:
        check(sqrt(Integer(c) ** 2 - Integer(a) ** 2) == b)
        text = phrase(rng, _LEG_EXAM_CONTEXTS, a=a, c=c)
        answer, answer_text = b, str(b)
        working = [f"l = √({c}² - {a}²) = √{c * c - a * a} = {b}"]
        cands = [str(c * c - a * a), str(c - a), str(c + a)]
    check_distinct(answer, *cands)
    return dict(
        text=text,
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=pick_distractors(answer_text, cands),
    )


def _pn(v) -> str:
    """A component as a factor: a negative value is bracketed."""
    return f"({v})" if v < 0 else str(v)


def _cross_working(a, b, na, nb, result):
    """Working for the cross product a x b of two integer 3-vectors (named na, nb)."""
    return [
        f"Use a × b = (a₂b₃ - a₃b₂, a₃b₁ - a₁b₃, a₁b₂ - a₂b₁) with a = {na} and b = {nb}.",
        f"x: {_pn(a[1])} × {_pn(b[2])} - {_pn(a[2])} × {_pn(b[1])} = {result[0]}; "
        f"y: {_pn(a[2])} × {_pn(b[0])} - {_pn(a[0])} × {_pn(b[2])} = {result[1]}; "
        f"z: {_pn(a[0])} × {_pn(b[1])} - {_pn(a[1])} × {_pn(b[0])} = {result[2]}.",
        f"{na} × {nb} = ({result[0]}, {result[1]}, {result[2]}).",
    ]


def _simplified(m, ssq) -> str:
    """ " = 3√5" when √ssq simplifies, else nothing."""
    if str(m) == f"sqrt({ssq})":
        return ""
    return " = " + re.sub(r"sqrt\((\d+)\)", "\u221a\\1", str(m).replace("*", ""))


def _dot_working(a, b, result, label="u · v"):
    terms = " + ".join(f"{_pn(x)} × {_pn(y)}" for x, y in zip(a, b))
    return [f"{label} = {terms} = {result}."]


@generator(
    "vectors",
    level=12,
    presets={
        "easy": dict(hi=5, kinds=("dot", "magnitude")),
        "medium": dict(hi=9, kinds=("dot", "cross", "magnitude")),
        "hard": dict(hi=15, kinds=("dot", "cross", "magnitude")),
    },
)
def vectors(rng, hi=9, kinds=("dot", "cross", "magnitude")):
    u = [rng.randint(-hi, hi) for _ in range(3)]
    v = [rng.randint(-hi, hi) for _ in range(3)]
    kind = rng.choice(list(kinds))
    su, sv = f"({u[0]}, {u[1]}, {u[2]})", f"({v[0]}, {v[1]}, {v[2]})"
    if kind == "dot":
        ans = sum(ui * vi for ui, vi in zip(u, v))
        check(Matrix(u).dot(Matrix(v)) == ans)
        # Misconceptions: dropped negative signs; forgot the z term; added
        # components instead of multiplying.
        cands = [
            sum(abs(ui * vi) for ui, vi in zip(u, v)),
            u[0] * v[0] + u[1] * v[1],
            sum(ui + vi for ui, vi in zip(u, v)),
        ]
        return dict(
            text=f"u = {su}, v = {sv}. Find u · v",
            latex=rf"\mathbf{{u}} = {su},\ \mathbf{{v}} = {sv}.\ \text{{Find }} \mathbf{{u}} \cdot \mathbf{{v}}",
            answer=ans,
            answer_text=str(ans),
            answer_latex=str(ans),
            distractors=pick_distractors(str(ans), [str(c) for c in cands]),
            working=_dot_working(u, v, ans),
        )
    if kind == "cross":
        w = Matrix(u).cross(Matrix(v))
        ans = [int(w[0]), int(w[1]), int(w[2])]
        # Independent check: result is orthogonal to both inputs.
        check(w.dot(Matrix(u)) == 0 and w.dot(Matrix(v)) == 0)
        # Misconceptions: flipped the order (negated result); forgot the minus
        # on the middle cofactor; multiplied elementwise.
        cands = [
            f"({-ans[0]}, {-ans[1]}, {-ans[2]})",
            f"({ans[0]}, {-ans[1]}, {ans[2]})",
            f"({u[0] * v[0]}, {u[1] * v[1]}, {u[2] * v[2]})",
        ]
        return dict(
            text=f"u = {su}, v = {sv}. Find u × v",
            latex=rf"\mathbf{{u}} = {su},\ \mathbf{{v}} = {sv}.\ \text{{Find }} \mathbf{{u}} \times \mathbf{{v}}",
            answer=ans,
            answer_text=f"({ans[0]}, {ans[1]}, {ans[2]})",
            answer_latex=rf"({ans[0]},\ {ans[1]},\ {ans[2]})",
            distractors=pick_distractors(f"({ans[0]}, {ans[1]}, {ans[2]})", cands, unordered=True),
            working=_cross_working(u, v, "u", "v", ans),
        )
    if kind == "add":
        ans = [u[0] + v[0], u[1] + v[1], u[2] + v[2]]
        check(list(Matrix(u) + Matrix(v)) == ans)
        # Misconceptions: subtracted instead of added; added only the first
        # two components and dropped z; multiplied elementwise instead.
        sub = [u[0] - v[0], u[1] - v[1], u[2] - v[2]]
        drop_z = [u[0] + v[0], u[1] + v[1], 0]
        mul = [u[0] * v[0], u[1] * v[1], u[2] * v[2]]
        cands = [
            f"({sub[0]}, {sub[1]}, {sub[2]})",
            f"({drop_z[0]}, {drop_z[1]}, {drop_z[2]})",
            f"({mul[0]}, {mul[1]}, {mul[2]})",
        ]
        return dict(
            text=f"u = {su}, v = {sv}. Find u + v",
            latex=rf"\mathbf{{u}} = {su},\ \mathbf{{v}} = {sv}.\ \text{{Find }} \mathbf{{u}} + \mathbf{{v}}",
            answer=ans,
            answer_text=f"({ans[0]}, {ans[1]}, {ans[2]})",
            answer_latex=rf"({ans[0]},\ {ans[1]},\ {ans[2]})",
            # unordered=True: same hazard class as the cross branch -- the `sub`/`drop_z`/`mul`
            # candidates are permutations of `ans` whenever `ans` has a permutation-symmetric shape,
            # and the skill's real grader (GRADERS["list"]) is order-insensitive (matches by
            # multiset). Measured through vector-addition's own sources: 15/5000 false accepts on
            # the easy source, 2/5000 on medium (the hard source uses kind "subtract", already
            # unaffected at 0/5000); this fix brings all sources to 0/5000.
            distractors=pick_distractors(f"({ans[0]}, {ans[1]}, {ans[2]})", cands, unordered=True),
            working=[
                f"Add the matching components: ({u[0]} + {_pn(v[0])}, {u[1]} + {_pn(v[1])}, {u[2]} + {_pn(v[2])}) "
                f"= ({ans[0]}, {ans[1]}, {ans[2]})."
            ],
        )
    if kind == "subtract":
        ans = [u[0] - v[0], u[1] - v[1], u[2] - v[2]]
        check(list(Matrix(u) - Matrix(v)) == ans)
        # Misconceptions: added instead of subtracted (the sibling slip of
        # the "add" branch's own `sub` distractor, in the other direction);
        # subtracted in the wrong order (v - u, a sign flip on every
        # component); dropped the z-component the same way "add" does.
        added = [u[0] + v[0], u[1] + v[1], u[2] + v[2]]
        wrong_order = [v[0] - u[0], v[1] - u[1], v[2] - u[2]]
        drop_z = [u[0] - v[0], u[1] - v[1], 0]
        cands = [
            f"({added[0]}, {added[1]}, {added[2]})",
            f"({wrong_order[0]}, {wrong_order[1]}, {wrong_order[2]})",
            f"({drop_z[0]}, {drop_z[1]}, {drop_z[2]})",
        ]
        return dict(
            text=f"u = {su}, v = {sv}. Find u - v",
            latex=rf"\mathbf{{u}} = {su},\ \mathbf{{v}} = {sv}.\ \text{{Find }} \mathbf{{u}} - \mathbf{{v}}",
            answer=ans,
            answer_text=f"({ans[0]}, {ans[1]}, {ans[2]})",
            answer_latex=rf"({ans[0]},\ {ans[1]},\ {ans[2]})",
            # unordered=True: the skill's grader (GRADERS["list"]) accepts the
            # three components in ANY order, so a candidate that is merely a
            # PERMUTATION of the correct answer -- `wrong_order` is `-ans`
            # componentwise, which is a permutation of `ans` whenever `ans`
            # itself has a permutation-symmetric pattern such as (a, -a, 0) --
            # must be dropped the same way an exactly-equal candidate is, or
            # the grader marks it correct despite `pick_distractors`' ordered
            # comparison offering it as wrong.
            distractors=pick_distractors(f"({ans[0]}, {ans[1]}, {ans[2]})", cands, unordered=True),
            working=[
                f"Subtract the matching components: ({u[0]} - {_pn(v[0])}, {u[1]} - {_pn(v[1])}, {u[2]} - {_pn(v[2])}) "
                f"= ({ans[0]}, {ans[1]}, {ans[2]})."
            ],
        )
    if kind == "cross_sum":
        # p is the function's own `u` (reused, not a rename); q is a THIRD
        # vector drawn only on this branch, so no other kind's rng stream
        # moves. usum = p + q is computed independently of the display text.
        q = [rng.randint(-hi, hi) for _ in range(3)]
        sq = f"({q[0]}, {q[1]}, {q[2]})"
        usum = [u[0] + q[0], u[1] + q[1], u[2] + q[2]]
        w = Matrix(usum).cross(Matrix(v))
        ans = [int(w[0]), int(w[1]), int(w[2])]
        # Independent check, the same route the "cross" branch uses: the
        # result is orthogonal to both the sum and v.
        check(w.dot(Matrix(usum)) == 0 and w.dot(Matrix(v)) == 0)
        # Misconceptions: crossed p and v alone, forgetting to add q first;
        # added correctly but crossed in the wrong order (v x (p+q) = -w,
        # reusing the "wrong-order-sign" misconception's own idea); crossed
        # the wrong PAIR (p x q) and multiplied that elementwise with v.
        forgot_q = Matrix(u).cross(Matrix(v))
        wrong_order = [-ans[0], -ans[1], -ans[2]]
        wrong_pair = Matrix(u).cross(Matrix(q))
        wrong_pair_mixed = [int(wrong_pair[0]) * v[0], int(wrong_pair[1]) * v[1], int(wrong_pair[2]) * v[2]]
        cands = [
            f"({int(forgot_q[0])}, {int(forgot_q[1])}, {int(forgot_q[2])})",
            f"({wrong_order[0]}, {wrong_order[1]}, {wrong_order[2]})",
            f"({wrong_pair_mixed[0]}, {wrong_pair_mixed[1]}, {wrong_pair_mixed[2]})",
        ]
        return dict(
            text=f"p = {su}, q = {sq}, v = {sv}. Find (p + q) × v",
            latex=rf"\mathbf{{p}} = {su},\ \mathbf{{q}} = {sq},\ \mathbf{{v}} = {sv}.\ \text{{Find }} (\mathbf{{p}} + \mathbf{{q}}) \times \mathbf{{v}}",
            answer=ans,
            answer_text=f"({ans[0]}, {ans[1]}, {ans[2]})",
            answer_latex=rf"({ans[0]},\ {ans[1]},\ {ans[2]})",
            distractors=pick_distractors(f"({ans[0]}, {ans[1]}, {ans[2]})", cands, unordered=True),
            working=[
                f"p + q = ({usum[0]}, {usum[1]}, {usum[2]}).",
                *_cross_working(usum, v, "(p + q)", "v", ans),
            ],
        )
    if kind == "dot_sum":
        # p is the function's own `u` (reused, not a rename); q is a THIRD
        # vector drawn only on this branch, so no other kind's rng stream
        # moves. usum = p + q is computed independently of the display text.
        q = [rng.randint(-hi, hi) for _ in range(3)]
        sq = f"({q[0]}, {q[1]}, {q[2]})"
        usum = [u[0] + q[0], u[1] + q[1], u[2] + q[2]]
        ans = sum(ui * vi for ui, vi in zip(usum, v))
        # Independent check, the same route the "dot" branch uses.
        check(Matrix(usum).dot(Matrix(v)) == ans)
        # Misconceptions: computed p . v alone, forgetting to add q first;
        # computed p.v * q.v -- multiplied the two dot products together
        # instead of adding them (breaks the (p+q).v == p.v + q.v identity,
        # which is NOT a valid distractor since it equals the answer);
        # added the components elementwise instead of multiplying at the
        # dot-product step (the "added-not-multiplied" misconception's own
        # idea, applied to the summed vector).
        p_dot_v = sum(ui * vi for ui, vi in zip(u, v))
        q_dot_v = sum(qi * vi for qi, vi in zip(q, v))
        forgot_q = p_dot_v
        multiplied_dots = p_dot_v * q_dot_v
        added_not_multiplied = sum(ui + vi for ui, vi in zip(usum, v))
        cands = [forgot_q, multiplied_dots, added_not_multiplied]
        return dict(
            text=f"p = {su}, q = {sq}, v = {sv}. Find (p + q) · v",
            latex=rf"\mathbf{{p}} = {su},\ \mathbf{{q}} = {sq},\ \mathbf{{v}} = {sv}.\ \text{{Find }} (\mathbf{{p}} + \mathbf{{q}}) \cdot \mathbf{{v}}",
            answer=ans,
            answer_text=str(ans),
            answer_latex=str(ans),
            distractors=pick_distractors(str(ans), [str(c) for c in cands]),
            working=[
                f"p + q = ({usum[0]}, {usum[1]}, {usum[2]}).",
                *_dot_working(usum, v, ans, "(p + q) · v"),
            ],
        )
    if kind == "magnitude_sum":
        # Addition step before the magnitude computation: w = u + v, then
        # |u + v|, not the magnitude of a single given vector. u and v are
        # both already drawn unconditionally above (every kind does), so no
        # extra rng draw here -- same convention as "dot"/"cross"/"add"/
        # "subtract", unlike "cross_sum"/"dot_sum" which need a THIRD vector.
        w = [u[0] + v[0], u[1] + v[1], u[2] + v[2]]
        ssq = sum(wi * wi for wi in w)
        m = sqrt(ssq)
        check(m**2 == ssq)
        # Misconceptions: computed |u| + |v| instead of |u + v| -- the
        # triangle-inequality slip. These are NOT equal in general (equal
        # only when u and v are non-negative scalar multiples of each
        # other), so this candidate is only offered when it is genuinely
        # distinct from `m` for THIS seed -- checked below against
        # `answer_alt` (the same 2-d.p. value the real grader's own 0.005
        # tolerance shortcut compares against, `mathgen/grading/typed.py`'s
        # `grade_expression`), not merely an exact sympy `!=`: a candidate
        # can be symbolically distinct from `m` yet still land inside that
        # 0.005 window whenever u and v are close to parallel (measured
        # directly -- an exact-`!=`-only check let 4/5000 seeds through in
        # a verification sweep). 0.02 is a safety margin
        # comfortably clear of the worst case (0.005 rounding + 0.005
        # tolerance = 0.01), not a tuned threshold; added the summed
        # components without squaring first (the "no-square-before-root"
        # idea, applied to the summed vector); forgot the square root on the
        # summed components (the "forgot-sqrt-magnitude" idea, applied to
        # the summed vector).
        mag_u = sqrt(sum(ui * ui for ui in u))
        mag_v = sqrt(sum(vi * vi for vi in v))
        sum_of_mags = mag_u + mag_v
        answer_alt = round(float(m), 2)
        cands = []
        if abs(float(sum_of_mags) - answer_alt) > 0.02:
            cands.append(str(sum_of_mags))
        cands.append(str(sum(abs(wi) for wi in w)))
        cands.append(str(ssq))
        distractors = pick_distractors(str(m), cands)
        if not distractors:
            # Every slip above collapsed onto the answer (a zero or single-axis
            # sum): offer whole-unit slips, well clear of the grader's 0.005
            # window. Only when none survive, so a draw that keeps its two
            # misconception distractors is unchanged.
            distractors = pick_distractors(str(m), [str(m + 1), str(m + 2)])
        return dict(
            text=f"u = {su}, v = {sv}. Find |u + v| (exact)",
            latex=rf"\mathbf{{u}} = {su},\ \mathbf{{v}} = {sv}.\ \text{{Find }} |\mathbf{{u}} + \mathbf{{v}}|",
            answer=str(m),
            answer_text=str(m),
            answer_latex=r"|\mathbf{u} + \mathbf{v}| = " + str(m).replace("sqrt", r"\sqrt"),
            answer_alt=answer_alt,
            distractors=distractors,
            working=[
                f"u + v = ({w[0]}, {w[1]}, {w[2]}).",
                f"|u + v| = √({' + '.join(f'{_pn(x)}²' for x in w)}) = √{ssq}" + _simplified(m, ssq) + ".",
            ],
        )
    # Magnitude, exact (may be a surd).
    ssq = sum(ui * ui for ui in u)
    m = sqrt(ssq)
    check(m**2 == ssq)
    # Misconceptions: added |components| without squaring; forgot the square
    # root; square-rooted the unsquared sum.
    cands = [str(sum(abs(ui) for ui in u)), str(ssq), str(sqrt(sum(abs(ui) for ui in u)))]
    # Fallbacks, offered only when the misconceptions above collapse onto the
    # answer (a zero or single-axis vector): whole-unit slips, well clear of
    # the grader's 0.005 window.
    cands += [str(m + 1), str(m + 2)]
    return dict(
        text=f"u = {su}. Find |u| (exact)",
        latex=rf"\mathbf{{u}} = {su}.\ \text{{Find }} |\mathbf{{u}}|",
        answer=str(m),
        answer_text=str(m),
        answer_latex=r"|\mathbf{u}| = " + str(m).replace("sqrt", r"\sqrt"),
        answer_alt=round(float(m), 2),
        distractors=pick_distractors(str(m), cands),
        working=[f"|u| = √({' + '.join(f'{_pn(x)}²' for x in u)}) = √{ssq}" + _simplified(m, ssq) + "."],
    )
