"""Exam-tier source for `wave-frequency-from-speed-wavelength`: a sound wave that crosses into a new medium.

The stem gives the speed of sound in air (m/s) and a measured wavelength in centimetres. Part (a) finds the
frequency, which needs a unit conversion before `v = f * wavelength` is rearranged. Part (b) sends the same wave
into a second medium where only the speed is given: the learner must know that the frequency, not the wavelength,
is what stays fixed across the boundary. Part (b) carries `follow_through`, so a wrong frequency in (a) still
earns (b) when it is used correctly.

Built backwards from a whole-number frequency, so (a) is exact and (b) is one rounding of an exact quotient.
"""

from __future__ import annotations

from fractions import Fraction

from mathgen.answers import Distractor
from mathgen.core import (
    GRADER_TOLERANCE,
    attempts,
    check,
    generator,
    latex_answer,
    numeric_answer,
    pick_distractors,
    round_half_up,
    stated_precision_string,
)

# (medium, lowest speed in m/s, highest speed in m/s), speeds drawn to one decimal place.
_SECOND_MEDIA = [
    ("fresh water", 1400, 1500),
    ("sea water", 1480, 1540),
    ("ice", 3100, 3300),
]
_SOURCES = [
    "A speaker at a school in Hobart plays a steady tone.",
    "A speaker at a music festival in Byron Bay plays a steady tone.",
    "A tuning fork is struck in a music room in Perth.",
    "A whistle blows on a sports oval in Wagga Wagga.",
]


def _dp2(value: Fraction) -> Fraction:
    return round_half_up(value, 2)


@generator("exd_x_u3_wave_boundary", level=11, presets={"exam": dict(tier="exam")})
def exd_x_u3_wave_boundary(rng, tier: str = "exam") -> dict:
    """Exam tier: find a frequency, then the wavelength of the same wave in another medium."""
    check(tier == "exam", f"unknown tier {tier!r}")
    medium, v2_lo, v2_hi = rng.choice(_SECOND_MEDIA)
    source = rng.choice(_SOURCES)
    for _ in attempts(1000, "a frequency and wavelength with a clear rounding"):
        f = 10 * rng.randint(10, 150)  # 100 .. 1500 Hz
        lo, hi = -(-330_000 // f), 350_000 // f
        if lo > hi:
            continue
        tenths_cm = rng.randint(lo, hi)  # wavelength in tenths of a centimetre
        v1 = Fraction(f * tenths_cm, 1000)  # exact speed in air, m/s, at most 2 decimal places
        v2 = Fraction(rng.randint(v2_lo * 10, v2_hi * 10), 10)
        lam2 = v2 / f
        frac = (lam2 * 100) % 1
        if abs(frac - Fraction(1, 2)) > Fraction(1, 20) and lam2 >= Fraction(1, 10):
            break
    lam1_cm = Fraction(tenths_cm, 10)
    lam1_m = lam1_cm / 100
    check(v1 / lam1_m == f, "frequency does not reproduce")
    check(lam2 * f == v2, "second wavelength does not reproduce")

    f_text = str(f)
    lam2_text = stated_precision_string(_dp2(lam2), 2)
    v1_text = stated_precision_string(v1, 2)
    lam1_text = stated_precision_string(lam1_cm, 1)
    v2_text = stated_precision_string(v2, 1)

    def whole(value: Fraction, ident: str):
        n = round_half_up(value, 0)
        return Distractor(str(int(n)), ident, int(n)) if n > 0 else None

    def two_dp(value: Fraction, ident: str):
        n = _dp2(value)
        return Distractor(stated_precision_string(n, 2), ident, n) if n > 0 else None

    a_cand = [
        whole(v1 / lam1_cm, "unit-not-converted"),  # centimetres used as metres
        whole(v1 * lam1_m, "multiplied-instead-of-divided"),
        whole(v1 / (lam1_cm / 1000), "unit-not-converted"),  # divided by 1000 instead of 100
    ]
    b_cand = [
        two_dp(lam1_m, "frequency-changes-at-boundary"),  # wavelength carried over unchanged
        two_dp(lam1_m * v1 / v2, "inverted-ratio"),
        two_dp(lam2 * 100, "unit-not-converted"),  # left in centimetres
        two_dp(v2 * lam1_m, "multiplied-instead-of-divided"),
    ]
    a_d = pick_distractors(f_text, [c for c in a_cand if c], 3, tolerance=GRADER_TOLERANCE)
    b_d = pick_distractors(lam2_text, [c for c in b_cand if c], 3, tolerance=GRADER_TOLERANCE)
    check(len(a_d) >= 2 and len(b_d) >= 2, "too few distractors")

    stem = (
        f"{source} In air, where sound travels at {v1_text} m/s, the wavelength is {lam1_text} cm. "
        f"The sound then passes into {medium}, where it travels at {v2_text} m/s."
    )
    return dict(
        text=stem,
        parts=[
            dict(
                prompt="Find the frequency of the sound, in hertz.",
                answer_kind="number",
                answer=f,
                answer_text=f_text,
                answer_latex=latex_answer(f_text),
                marks=2,
                sensible_range=(0, None),
                working=[
                    f"Wavelength = {lam1_text} cm = {stated_precision_string(lam1_m, 3)} m.",
                    f"f = v ÷ wavelength = {v1_text} ÷ {stated_precision_string(lam1_m, 3)} = {f_text} Hz.",
                ],
                distractors=a_d,
            ),
            dict(
                prompt=f"Using your frequency from (a), find the wavelength of the sound in {medium}, in metres, "
                "correct to 2 decimal places.",
                answer_kind="number",
                answer=numeric_answer(_dp2(lam2)),
                answer_text=lam2_text,
                answer_latex=latex_answer(lam2_text),
                marks=2,
                sensible_range=(0, None),
                follow_through=f"{v2_text}/part_a",
                working=[
                    "The frequency does not change when the wave crosses the boundary.",
                    f"wavelength = v ÷ f = {v2_text} ÷ {f_text} = {lam2_text} m (2 d.p.).",
                ],
                distractors=b_d,
            ),
        ],
    )
