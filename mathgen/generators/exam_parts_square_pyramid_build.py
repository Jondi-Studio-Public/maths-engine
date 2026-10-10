"""Multi-part exam tier for `square-pyramid-volume-and-surface-area`.

One scenario (a real object shaped like a square-based pyramid), four parts, 6 marks:

(a) "Show that" the perpendicular height is h (stated, correct to 1 decimal place); the learner's answer is the
    checkpoint, the exact value of h² = l² - (s/2)², 1 mark.
(b) Volume from the STATED, rounded h, correct to 1 decimal place, 2 marks. No follow-through: everyone is told
    to use the stated h, so everyone has the same target.
(c) Total surface area including the base, 2 marks, independent of (a) and (b).
(d) The object is scaled by k in every length (2, 3 or 1/2); the new volume from the learner's own (b),
    correct to 1 decimal place, 1 mark, `follow_through="<k**3>*part_b"`.

Everything is built from the integers `s` (even) and `l` (so `h²` is an integer and the exact checkpoint needs no
decimals; a decimal `h²` would meet `grading/typed.py:_is_rounded_as_asked`, which reads the "1 decimal place" in
(a)'s prompt as a rounding instruction for the reply). `h = √(l² - (s/2)²)` is always irrational (a perfect
square is redrawn). Canonical answers are computed from exact Fractions and rounded once, at display; a draw
whose exact value sits on a rounding tie (so a float and an exact calculator could round differently) is
redrawn.
"""

from __future__ import annotations

import math
from fractions import Fraction

import sympy

from ..core import (
    Part,
    check,
    check_equivalent,
    generator,
    latex_answer,
    numeric_answer,
    phrase,
    pick_distractors,
    round_half_up,
    stated_precision_string,
)
from ..grading import typed

# key: (stem opening, name used in prompts, unit, s values, l extra range above s/2)
_OBJECTS = (
    ("A solid glass paperweight is in the shape of a square-based pyramid.", "paperweight", "cm", range(6, 21, 2), 10),
    ("A candle is moulded in the shape of a square-based pyramid.", "candle", "cm", range(6, 19, 2), 9),
    ("A wooden desk ornament is made in the shape of a square-based pyramid.", "ornament", "cm", range(8, 25, 2), 10),
    (
        "A cardboard gift box, closed at the base, is shaped like a square-based pyramid.",
        "gift box",
        "cm",
        range(8, 23, 2),
        9,
    ),
    ("A garden tent, with a sewn-in floor, is shaped like a square-based pyramid.", "tent", "m", range(4, 11, 2), 3),
    ("A school display includes a model of a pyramid with a square base.", "model", "m", range(2, 9, 2), 3),
    (
        "A pyramid-shaped play shelter, with a canvas floor, is set up in a school yard.",
        "shelter",
        "m",
        range(4, 9, 2),
        3,
    ),
)

_SECOND_SENTENCE = (
    "Its square base has side length {s} {u}, and each triangular face has a slant height of {l} {u}, "
    "measured from the midpoint of a base edge to the apex.",
    "The base is square with side length {s} {u}. Each of the four triangular faces has a slant height of "
    "{l} {u} (the height of the face from the midpoint of a base edge to the apex).",
)

# (k, words, follow_through expression in part_b)
_SCALES = (
    (Fraction(2), "twice", "8*part_b"),
    (Fraction(3), "three times", "27*part_b"),
    (Fraction(1, 2), "half", "part_b/8"),
)

_A_PROMPT = (
    "Show that the perpendicular height h of the {name} is {h} {u}, correct to 1 decimal place. "
    "As your answer, write down the exact value of h² in {u}².",
    "Show that the {name} has a perpendicular height h of {h} {u}, correct to 1 decimal place. "
    "As your answer, give the exact value of h² in {u}².",
)
_B_PROMPT = (
    "Using the rounded value h = {h} {u} stated in part (a), find the volume of the {name}, "
    "correct to 1 decimal place.",
    "Use h = {h} {u}, the rounded value stated in part (a), to find the volume of the {name}, "
    "correct to 1 decimal place.",
)
_C_PROMPT = (
    "Find the total surface area of the {name}, including its base, in {u}².",
    "Calculate the total surface area of the {name}, including the base, in {u}².",
)
_D_PROMPT = (
    "A similar {name} is made in which every length is {words} the corresponding length of the original. "
    "Using your answer to (b), find the volume of the new {name}, correct to 1 decimal place.",
    "Every length of a second, similar {name} is {words} the corresponding length of the first. "
    "Using your answer to (b), find the volume of the second {name}, correct to 1 decimal place.",
)


def _is_tie(value: Fraction) -> bool:
    """Whether `value` sits exactly half-way between two 1-decimal-place values."""
    return (value * 10).denominator == 2


def _tenths_of_sqrt(h_squared: int) -> int:
    """round(10 * sqrt(h_squared)) by integer arithmetic (h_squared is not a perfect square, so no tie)."""
    return (math.isqrt(400 * h_squared) + 1) // 2


def _draw(rng):
    for _ in range(500):
        name_row = rng.choice(_OBJECTS)
        s = rng.choice(list(name_row[3]))
        half = s // 2
        l = rng.randint(half + 1, half + name_row[4])
        h_squared = l * l - half * half
        if math.isqrt(h_squared) ** 2 == h_squared:
            continue
        h_tenths = _tenths_of_sqrt(h_squared)
        h_stated = Fraction(h_tenths, 10)
        volume_exact = Fraction(s * s) * h_stated / 3
        if _is_tie(volume_exact):
            continue
        k, words, ft = rng.choice(_SCALES)
        volume_b = round_half_up(volume_exact, 1)
        new_exact = k**3 * volume_b
        if _is_tie(new_exact):
            continue
        return name_row, s, l, h_squared, h_stated, volume_exact, volume_b, k, words, ft, new_exact
    raise RuntimeError("no square pyramid draw found")


def _one_dp(value: Fraction) -> str:
    return stated_precision_string(round_half_up(value, 1), 1)


def _part(**kw) -> Part:
    return Part(label=kw.pop("label", "x"), answer_kind="number", **kw)


def _check_wrong(prompt: str, answer, answer_text: str, candidates: list[str]) -> None:
    """Every distractor must be graded wrong by the part's own grader (unconditionally)."""
    probe = _part(prompt=prompt, answer=answer, answer_text=answer_text)
    for c in candidates:
        check(not typed.grade("number", probe, c).correct, f"distractor {c!r} is accepted for {answer_text!r}")


@generator("exam_parts_square_pyramid", level=11, presets={"exam": {}})
def exam_parts_square_pyramid(rng):
    (opening, name, u, _s_values, _extra), s, l, h2, h_stated, volume_exact, volume_b, k, words, ft, new_exact = _draw(
        rng
    )
    half = s // 2
    h_text = stated_precision_string(h_stated, 1)

    stem = opening + " " + phrase(rng, _SECOND_SENTENCE, s=s, l=l, u=u)

    # ---- canonical answers, then a second route for each ----
    sa = s * s + 2 * s * l
    volume_text = stated_precision_string(volume_b, 1)
    new_volume = round_half_up(new_exact, 1)
    new_text = stated_precision_string(new_volume, 1)

    root = sympy.sqrt(sympy.Integer(l) ** 2 - sympy.Rational(s, 2) ** 2)
    check_equivalent(root**2, h2, "h² disagrees with the exact route")
    check(round_half_up(Fraction(str(sympy.N(root, 30))), 1) == h_stated, "stated h disagrees with the exact route")
    check(abs(h_stated - Fraction(str(sympy.N(root, 30)))) <= Fraction(1, 20), "stated h is not h to 1 decimal place")
    check(h_stated * h_stated != h2, "h must be irrational")
    check(
        round_half_up(Fraction(s * s, 3) * h_stated, 1) == volume_b,
        "volume disagrees with the second route",
    )
    check(sa == 4 * Fraction(s * l, 2) + s * s, "surface area disagrees with the faces-plus-base route")
    check(new_volume == round_half_up(volume_b * k**3, 1), "scaled volume disagrees with the second route")

    # ---- prompts ----
    pa = phrase(rng, _A_PROMPT, name=name, h=h_text, u=u)
    pb = phrase(rng, _B_PROMPT, name=name, h=h_text, u=u)
    pc = phrase(rng, _C_PROMPT, name=name, u=u)
    pd = phrase(rng, _D_PROMPT, name=name, words=words)

    # ---- (a) distractors ----
    a_cands = [
        str(l * l + half * half),  # adds the squares instead of subtracting
        str(l * l - s * s) if l * l - s * s > 0 else None,  # whole base side instead of half
        str(l * l),  # forgets to subtract (s/2)²
    ]
    a_dis = pick_distractors(str(h2), [c for c in a_cands if c])
    _check_wrong(pa, h2, str(h2), a_dis)
    a_notes = {
        str(l * l + half * half): "adds-instead-of-subtracts-in-pythagoras",
        str(l * l - s * s): "uses-whole-base-side-not-half",
        str(l * l): "forgets-to-subtract-the-half-base-squared",
    }

    # ---- (b) distractors ----
    b_cands = [
        (_one_dp(Fraction(s * s * l, 3)), "uses-slant-height-as-perpendicular-height"),
        (_one_dp(Fraction(s * s) * h_stated), "no-third-factor-on-volume"),
        (_one_dp(Fraction(s * s) * h_stated / 2), "uses-half-not-third-on-volume"),
    ]
    b_dis = pick_distractors(volume_text, [c for c, _ in b_cands], tolerance=Fraction(1, 10))
    _check_wrong(pb, numeric_answer(volume_b), volume_text, b_dis)

    # ---- (c) distractors ----
    c_cands = [
        (str(2 * s * l), "omits-base-square"),
        (_one_dp(s * s + 2 * s * h_stated), "uses-apex-height-as-slant-height"),
        (str(s * s + 4 * s * l), "no-half-factor-on-laterals"),
    ]
    c_dis = pick_distractors(str(sa), [c for c, _ in c_cands])
    _check_wrong(pc, sa, str(sa), c_dis)

    # ---- (d) distractors ----
    d_cands = [
        (_one_dp(volume_b * k), "scales-volume-by-k"),
        (_one_dp(volume_b * k * k), "scales-volume-by-k-squared"),
        (volume_text, "forgets-volume-changes-when-scaled"),
    ]
    d_dis = pick_distractors(new_text, [c for c, _ in d_cands], tolerance=Fraction(1, 10))
    _check_wrong(pd, numeric_answer(new_volume), new_text, d_dis)

    k_text = f"{k.numerator}/{k.denominator}" if k.denominator != 1 else str(k.numerator)
    k3 = k**3
    k3_text = f"{k3.numerator}/{k3.denominator}" if k3.denominator != 1 else str(k3.numerator)

    parts = [
        dict(
            prompt=pa,
            answer_kind="number",
            answer=h2,
            answer_text=str(h2),
            answer_latex=latex_answer(sympy.Integer(h2)),
            marks=1,
            working=[
                f"The perpendicular height, the half base side and the slant height form a right-angled triangle: "
                f"h² + (s/2)² = l².",
                f"h² = {l}² - {half}² = {l * l} - {half * half} = {h2}.",
                f"h = √{h2} = {h_text} {u}, correct to 1 decimal place, as stated.",
            ],
            distractors=a_dis,
            option_notes={d: a_notes[d] for d in a_dis},
        ),
        dict(
            prompt=pb,
            answer_kind="number",
            answer=numeric_answer(volume_b),
            answer_text=volume_text,
            answer_latex=latex_answer(volume_text),
            marks=2,
            working=[
                "Volume of a pyramid = (1/3) × base area × perpendicular height.",
                f"V = (1/3) × {s}² × {h_text} = (1/3) × {s * s} × {h_text} = {volume_text} {u}³, "
                f"correct to 1 decimal place.",
            ],
            distractors=b_dis,
            option_notes={c: n for c, n in b_cands if c in b_dis},
        ),
        dict(
            prompt=pc,
            answer_kind="number",
            answer=sa,
            answer_text=str(sa),
            answer_latex=latex_answer(sympy.Integer(sa)),
            marks=2,
            working=[
                "Total surface area = base + four triangular faces.",
                f"Base: {s}² = {s * s} {u}².",
                f"Each face: (1/2) × {s} × {l} = {half * l} {u}², "
                f"so the four faces are 2 × {s} × {l} = {2 * s * l} {u}².",
                f"Total = {s * s} + {2 * s * l} = {sa} {u}².",
            ],
            distractors=c_dis,
            option_notes={c: n for c, n in c_cands if c in c_dis},
        ),
        dict(
            prompt=pd,
            answer_kind="number",
            answer=numeric_answer(new_volume),
            answer_text=new_text,
            answer_latex=latex_answer(new_text),
            marks=1,
            follow_through=ft,
            working=[
                f"Volumes of similar solids scale with the cube of the length scale factor: {k_text}³ = {k3_text}.",
                f"New volume = {k3_text} × {volume_text} = {new_text} {u}³, correct to 1 decimal place.",
            ],
            distractors=d_dis,
            option_notes={c: n for c, n in d_cands if c in d_dis},
        ),
    ]
    return dict(text=stem, parts=parts)
