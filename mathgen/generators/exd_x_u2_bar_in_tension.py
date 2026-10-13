"""Exam tier for stress, strain and Young's modulus: one tension test followed through four parts.

A tension-test report gives a rectangular bar's cross-section, original length, load and measured extension.
Nothing is handed over as a stress or a strain: the learner finds the stress (the area comes from two
dimensions and the load is in kN), the strain (in microstrain), the modulus (MPa against GPa), and finally
uses that modulus to predict the extension of a second bar. Each later part carries `follow_through`, so a
slip early on costs only the part where it was made.

## Construction: draw the modulus and the strain first

The modulus E (GPa) and the strain (microstrain) are drawn, which fixes the stress E * strain exactly (a
multiple of 0.5 MPa). The cross-section is drawn from sizes whose area is a multiple of 100 mm^2, so the load
in kN terminates in two decimal places. The extension is strain * length, which terminates in three. Every
printed figure is therefore exact and the key for each part is exact arithmetic on what the stem prints.
The last part is the only rounded answer; its value is kept clear of a rounding boundary.
"""

from __future__ import annotations

import random
import re
from fractions import Fraction

from mathgen.answers import Distractor
from mathgen.core import (
    attempts,
    check,
    draw_until,
    exact_decimal_string,
    generator,
    latex_answer,
    numeric_answer,
    pick_distractors,
    round_half_up,
    stated_precision_string,
    stated_precision_tolerance,
)

_E_GPA = list(range(40, 221, 10))
_STRAIN_MICRO = list(range(200, 1501, 50))
_WIDTHS = (20, 25, 30, 40, 50, 60)
_THICKNESSES = (5, 8, 10, 12, 15, 20)
_LENGTHS = list(range(400, 2501, 100))
_LOADS_2 = list(range(10, 151, 5))
_MARGIN = Fraction(1, 50)  # in units of the last displayed place

_BARS = (
    "A tie bar in a roof truss",
    "A tension member in a warehouse frame",
    "A hanger rod for a shade sail",
    "A specimen in a university materials laboratory",
    "A flat bar in a bridge inspection sample",
)
_SECOND = (
    "A second bar of the same material and cross-section is {length} mm long and carries a tensile load of {load} kN.",
    "Another bar, made of the same material with the same cross-section, is {length} mm long and is loaded in "
    "tension with {load} kN.",
)


def _fmt(value: Fraction, dp: int) -> str:
    return stated_precision_string(round_half_up(value, dp), dp)


def _shown(value: Fraction, dp: int) -> Fraction | None:
    """The value as it displays at `dp` places, or None when it would display as zero."""
    shown = round_half_up(value, dp)
    return shown if shown != 0 else None


def _tagged(key_text: str, dp: int, candidates: list[tuple[str, Fraction]]) -> list:
    out = []
    for ident, value in candidates:
        shown = _shown(value, dp)
        if shown is None:
            continue
        out.append(Distractor(stated_precision_string(shown, dp), ident, shown))
    return pick_distractors(key_text, out, 3, tolerance=stated_precision_tolerance(dp))


def _draw(rng: random.Random):
    e_gpa = rng.choice(_E_GPA)
    strain = rng.choice(_STRAIN_MICRO)
    width, thick = draw_until(
        rng,
        lambda r: (r.choice(_WIDTHS), r.choice(_THICKNESSES)),
        lambda wt: wt[0] * wt[1] % 100 == 0 and wt[0] != wt[1],
        what="a cross-section with area a multiple of 100 mm^2",
    )
    length = rng.choice(_LENGTHS)

    def second(r):
        return r.choice(_LENGTHS), r.choice(_LOADS_2)

    def good(pair):
        length2, load2 = pair
        if length2 == length:
            return False
        ext2 = Fraction(load2 * length2, width * thick) / e_gpa
        if not Fraction(1, 5) <= ext2 <= 8:
            return False
        offset = (ext2 * 100) % 1
        return abs(offset - Fraction(1, 2)) > _MARGIN

    length2, load2 = draw_until(rng, second, good, what="a second bar with a clear 2 d.p. extension")
    return e_gpa, strain, width, thick, length, length2, load2


def _clashes(prompt: str, keys: list[Fraction]) -> bool:
    """True when a printed number in the prompt equals one of the answers."""
    printed = {Fraction(m) for m in re.findall(r"\d+(?:\.\d+)?", prompt)}
    return any(key in printed for key in keys)


def _build(rng: random.Random) -> dict | None:
    e_gpa, strain, width, thick, length, length2, load2 = _draw(rng)
    area = width * thick
    stress = Fraction(e_gpa * strain, 1000)  # MPa
    load_kn = stress * area / 1000
    ext = Fraction(strain * length, 10**6)  # mm
    ext2 = Fraction(load2 * length2, area) / e_gpa

    # Independent route: from the quantities the stem prints.
    check(Fraction(load_kn * 1000, area) == stress, "stress does not follow from the printed load")
    check(ext / length * 10**6 == strain, "strain does not follow from the printed extension")
    check(stress * 1000 / strain == e_gpa, "modulus does not follow from stress and strain")
    check(Fraction(load2 * 1000 * length2, area * e_gpa * 1000) == ext2, "second extension mismatch")

    load_text = exact_decimal_string(load_kn, 3)
    ext_text = exact_decimal_string(ext, 4)
    stem = (
        f"{rng.choice(_BARS)} has a rectangular cross-section {width} mm by {thick} mm and an original length of "
        f"{length} mm. Under a tensile load of {load_text} kN the bar extends by {ext_text} mm."
    )

    stress_text = _fmt(stress, 1)
    strain_text = str(strain)
    e_text = _fmt(Fraction(e_gpa), 1)
    ext2_text = _fmt(ext2, 2)

    stress_distractors = _tagged(
        stress_text,
        1,
        [
            ("unit-slip-kn-as-n", stress / 1000),
            ("added-dimensions-for-area", load_kn * 1000 / (width + thick)),
            ("used-one-side-for-area", load_kn * 1000 / (max(width, thick) ** 2)),
        ],
    )
    strain_distractors = pick_distractors(
        strain_text,
        [
            Distractor(
                str(round_half_up(Fraction(10**6, strain), 0).numerator),
                "inverted-ratio",
                int(round_half_up(Fraction(10**6, strain), 0)),
            ),
            Distractor(str(strain * 1000), "unit-slip-mm-as-m", strain * 1000),
            Distractor(str(strain * 10), "unit-slip-mm-as-cm", strain * 10),
        ],
        3,
        tolerance=Fraction(1, 2),
    )
    e_distractors = _tagged(
        e_text,
        1,
        [
            ("mpa-gpa-slip", Fraction(e_gpa * 1000)),
            ("inverted-ratio", Fraction(strain, 1) / stress),
            ("mpa-gpa-slip", Fraction(e_gpa, 1000)),
        ],
    )
    ext2_distractors = _tagged(
        ext2_text,
        2,
        [
            ("ignored-length-change", Fraction(load2 * length, area) / e_gpa),
            ("mpa-gpa-slip", ext2 * 1000),
            ("unit-slip-kn-as-n", ext2 / 1000),
            ("ignored-length-change", ext * Fraction(load2 * 1000, 1) / (load_kn * 1000)),
        ],
    )
    if min(map(len, (stress_distractors, strain_distractors, e_distractors, ext2_distractors))) < 2:
        return None

    second = rng.choice(_SECOND).format(length=length2, load=load2)
    ext2_prompt = (
        f"{second} Using your Young's modulus from (c), find the extension of this bar in mm, "
        "correct to 2 decimal places."
    )
    if _clashes(f"{stem} {ext2_prompt}", [stress, Fraction(strain), Fraction(e_gpa), round_half_up(ext2, 2)]):
        return None
    return dict(
        text=stem,
        parts=[
            dict(
                prompt="Find the tensile stress in the bar, in MPa, correct to 1 decimal place.",
                answer_kind="number",
                answer=numeric_answer(round_half_up(stress, 1)),
                answer_text=stress_text,
                answer_latex=latex_answer(stress_text),
                marks=2,
                sensible_range=(0, None),
                working=[
                    f"Area = {width} × {thick} = {area} mm².",
                    f"σ = F/A = {load_text} × 1000 N ÷ {area} mm² = {stress_text} MPa.",
                ],
                distractors=stress_distractors,
            ),
            dict(
                prompt="Find the strain in the bar, in microstrain (1 microstrain = 10⁻⁶), as a whole number.",
                answer_kind="number",
                answer=strain,
                answer_text=strain_text,
                answer_latex=latex_answer(strain_text),
                marks=1,
                sensible_range=(0, None),
                working=[f"ε = δL/L = {ext_text} ÷ {length} = {strain} × 10⁻⁶, which is {strain} microstrain."],
                distractors=strain_distractors,
            ),
            dict(
                prompt="Hence find Young's modulus of the material, in GPa, correct to 1 decimal place.",
                answer_kind="number",
                answer=numeric_answer(Fraction(e_gpa)),
                answer_text=e_text,
                answer_latex=latex_answer(e_text),
                marks=2,
                sensible_range=(0, None),
                follow_through="1000*part_a/part_b",
                working=[
                    f"E = σ/ε = {stress_text} MPa ÷ ({strain} × 10⁻⁶) = {e_gpa * 1000} MPa.",
                    f"Dividing by 1000 gives E = {e_text} GPa.",
                ],
                distractors=e_distractors,
            ),
            dict(
                prompt=ext2_prompt,
                answer_kind="number",
                answer=numeric_answer(round_half_up(ext2, 2)),
                answer_text=ext2_text,
                answer_latex=latex_answer(ext2_text),
                marks=2,
                sensible_range=(0, None),
                follow_through=f"{load2}*{length2}/({area}*part_c)",
                working=[
                    "δL = F·L / (A·E), with F in newtons and E in MPa (1 GPa = 1000 MPa).",
                    f"δL = {load2 * 1000} × {length2} ÷ ({area} × {e_gpa * 1000}) = {ext2_text} mm.",
                ],
                distractors=ext2_distractors,
            ),
        ],
    )


@generator("exd_x_u2_bar_in_tension", level=19, presets={"exam": dict(tier="exam")})
def exd_x_u2_bar_in_tension(rng: random.Random, tier: str = "exam") -> dict:
    """Stress, strain, modulus and a predicted extension for a rectangular bar. See the module docstring."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(200, "a bar whose answers are not printed in the prompt"):
        built = _build(rng)
        if built is not None:
            return built
