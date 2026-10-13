"""Exam source of `annulus-cross-section-area`: the metal in a hollow tube, as a chain.

The practice tiers ask for the area of one annulus from its radii or its outer diameter and wall thickness. An
exam question uses the annulus as the cross-section of a solid: (a) the area of metal in the cross-section (the
learner must first find the missing radius from a diameter and a wall that lies on both sides), (b) the volume of
metal in a length of tube, and (c) its mass from the density of the metal, converting grams to kilograms.

## Construction

Diameters and the wall are drawn first, so every canonical value comes from exact arithmetic with the stated value
of pi (3.14159). The answers are rounded as the parts ask: area to 1 decimal place, volume to the nearest 10 cm^3,
mass to the nearest kilogram. Part (b) is carried from the rounded (a) and (c) from the rounded (b), as the prompt
says, so a draw is accepted only when the exact route (no intermediate rounding) and the chained route display the
same three answers, each kept clear of a rounding boundary. A learner who keeps full precision and one who uses the
rounded figures then agree.
"""

from __future__ import annotations

import re
from fractions import Fraction

from mathgen.answers import Distractor
from mathgen.core import (
    attempts,
    check,
    exact_decimal_string,
    generator,
    latex_answer,
    numeric_answer,
    pick_distractors,
    round_half_up,
    stated_precision_string,
)

PI = Fraction(314159, 100000)  # the value of pi the prompt tells the learner to use
_MARGIN = Fraction(1, 200)

# metal, article-free noun phrase for the object, density in g/cm^3
_METALS = [
    ("steel", "A hollow steel pipe", Fraction(785, 100)),
    ("aluminium", "A hollow aluminium pole", Fraction(27, 10)),
    ("copper", "A copper water pipe", Fraction(896, 100)),
    ("brass", "A hollow brass tube", Fraction(85, 10)),
    ("steel", "A steel conduit", Fraction(785, 100)),
]


def _round_to(value: Fraction, step: int) -> Fraction:
    return round_half_up(value / step, 0) * step


def _clear(value: Fraction, step) -> bool:
    """True when `value` is at least `_MARGIN * step` away from a rounding boundary of the grid `step`."""
    offset = (value / step) % 1
    return abs(offset - Fraction(1, 2)) * step > max(
        _MARGIN, Fraction(step) * Fraction(1, 10) if step >= 10 else _MARGIN
    )


@generator("exd_x_m1_pipe_mass", level=11, presets={"exam": {"tier": "exam"}})
def exd_x_m1_pipe_mass(rng, tier: str = "exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(2000, "a pipe whose exact and chained answers agree and avoid every rounding boundary"):
        metal, noun, density = rng.choice(_METALS)
        outer_d = Fraction(rng.randint(8, 40), 2)  # 4 .. 20 cm in halves
        wall = Fraction(rng.randint(1, 6), 2)  # 0.5 .. 3 cm
        length = rng.choice(range(20, 65, 5))
        shape = rng.choice(["outer", "inner"])
        inner_d = outer_d - 2 * wall
        if inner_d < 2:
            continue
        big_r, small_r = outer_d / 2, inner_d / 2
        area = PI * (big_r**2 - small_r**2)
        a1 = round_half_up(area, 1)
        vol, vol_chain = area * length, a1 * length
        v10, v10_chain = _round_to(vol, 10), _round_to(vol_chain, 10)
        mass_exact = vol * density / 1000
        mass_chain = v10_chain * density / 1000
        k, k_chain = round_half_up(mass_exact, 0), round_half_up(mass_chain, 0)
        if k < 3 or (v10, k) != (v10_chain, k_chain):
            continue
        if not (_clear(area * 10, 1) and _clear(vol, 10) and _clear(mass_exact, 1) and _clear(mass_chain, 1)):
            continue
        if not (_clear(vol_chain, 10) and k == round_half_up(v10 * density / 1000, 0)):
            continue
        prompt_numbers = {f"{float(x):g}" for x in (outer_d, wall, inner_d, density)}
        prompt_numbers |= {str(length), "3.14159", "1", "10"}
        if {stated_precision_string(a1, 1), str(int(v10)), str(int(k))} & prompt_numbers:
            continue  # an answer must not appear as a number in the prompt
        break

    def num(x: Fraction) -> str:
        return exact_decimal_string(x, 2)

    metal_word = {"steel": "steel", "aluminium": "aluminium", "copper": "copper", "brass": "brass"}[metal]
    if shape == "outer":
        size = f"an outer diameter of {num(outer_d)} cm and a wall thickness of {num(wall)} cm"
    else:
        size = f"a hole with a diameter of {num(inner_d)} cm and a wall thickness of {num(wall)} cm"
    stem = f"{noun} has {size}. Use π ≈ 3.14159."
    dens_text = f"{float(density):g}"
    a_text = stated_precision_string(a1, 1)
    v_text, k_text = str(int(v10)), str(int(k))
    p_a = "Find the area of metal in the cross-section of the pipe, in square centimetres, correct to 1 decimal place."
    p_b = (
        f"A section of the pipe is {length} cm long. Using your answer to (a), find the volume of {metal_word} "
        "in the section, in cubic centimetres, correct to the nearest 10."
    )
    p_c = (
        f"The {metal_word} has a density of {dens_text} g/cm³. Using your answer to (b), find the mass of the section, "
        "in kilograms, correct to the nearest kilogram."
    )
    shown = set(re.findall(r"\d+(?:\.\d+)?", " ".join([stem, p_a, p_b, p_c])))
    check(not ({a_text, v_text, k_text} & shown), "an answer appears as a number in the prompt")

    def tag(value, ident, places=1):
        text = (
            stated_precision_string(round_half_up(value, places), places) if places else str(int(round_half_up(value)))
        )
        return Distractor(text, ident, Fraction(text)) if ident else text

    a_cands = [
        tag(PI * big_r**2, "ignored-the-hole"),
        tag(PI * (big_r - small_r) ** 2, "subtracted-radii-before-squaring"),
        tag(PI * (big_r**2 + small_r**2), "added-radii-squares"),
        tag(PI * (big_r**2 - ((outer_d - wall) / 2) ** 2), "diameter-minus-wall-once"),
        tag(PI * (outer_d**2 - inner_d**2), "diameter-for-radius"),
    ]

    def tag10(value, ident=None):
        text = str(int(_round_to(value, 10)))
        return Distractor(text, ident, int(text)) if ident else text

    b_cands = [
        tag10(PI * big_r**2 * length, "ignored-the-hole"),
        tag10(PI * small_r**2 * length),  # the volume of the hole, not of the metal
        tag10(a1 * length * wall),
    ]

    def tagk(value, ident=None):
        text = str(int(round_half_up(value)))
        return Distractor(text, ident, int(text)) if ident else text

    c_cands = [
        tagk(v10 * density, "unit-factor-wrong-power"),
        tagk(v10 * density / 100, "unit-factor-wrong-power"),
        tagk(v10 * density / 10, "unit-factor-wrong-power"),
    ]
    a_d = pick_distractors(a_text, a_cands, 3, tolerance=Fraction(51, 1000))
    b_d = pick_distractors(v_text, b_cands, 3)
    c_d = pick_distractors(k_text, c_cands, 3)
    check(len(a_d) >= 3 and len(b_d) >= 2 and len(c_d) >= 2, "too few distractors survive")
    check(all(abs(Fraction(str(x)) - a1) > Fraction(51, 1000) for x in a_d), "an (a) distractor is accepted")

    return dict(
        text=stem,
        parts=[
            dict(
                prompt=p_a,
                answer_kind="number",
                answer=numeric_answer(a1),
                answer_text=a_text,
                answer_latex=latex_answer(a_text),
                marks=2,
                sensible_range=(0, None),
                working=[
                    f"Outer radius R = {num(big_r)} cm; inner radius r = R - wall = {num(small_r)} cm.",
                    f"A = π(R² - r²) = π({num(big_r)}² - {num(small_r)}²) = {a_text} cm² (1 d.p.).",
                ],
                distractors=a_d,
            ),
            dict(
                prompt=p_b,
                answer_kind="number",
                answer=int(v10),
                answer_text=v_text,
                answer_latex=latex_answer(v_text),
                marks=1,
                sensible_range=(0, None),
                follow_through=f"10*floor(part_a*{length}/10+1/2)",
                working=[
                    f"V = area × length = {a_text} × {length} = {float(vol_chain):.1f} cm³, which is {v_text} to the nearest 10."
                ],
                distractors=b_d,
            ),
            dict(
                prompt=p_c,
                answer_kind="number",
                answer=int(k),
                answer_text=k_text,
                answer_latex=latex_answer(k_text),
                marks=2,
                sensible_range=(0, None),
                follow_through=f"part_b*{density.numerator}/{density.denominator}/1000",
                working=[
                    f"Mass = {v_text} × {dens_text} = {float(v10 * density):.1f} g.",
                    f"÷ 1000 gives {float(v10 * density / 1000):.2f} kg, which is {k_text} kg to the nearest kilogram.",
                ],
                distractors=c_d,
            ),
        ],
    )
