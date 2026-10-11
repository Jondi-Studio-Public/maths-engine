"""Exam tier for angle of inclination and gradient: the angle between two lines.

Each line is described in a different way (slope-intercept, general form with surd coefficients, or two points),
so the learner has to choose to read each gradient, turn it into an angle of inclination, and then combine the
two angles into an acute or obtuse angle between the lines. Built forwards: the two inclinations are drawn
first, then each line is written to have exactly that inclination.
"""

from __future__ import annotations

import math
import random

from ..core import check, generator, pick_distractors

_ANGLES = (30, 45, 60, 120, 135, 150)

# General-form coefficient pairs (a, b) for  a x + b y = c  with gradient -a/b, as printed text.
_GENERAL = {
    30: [("x", "- sqrt(3)y"), ("-x", "+ sqrt(3)y")],  # m = 1/sqrt(3)
    60: [("sqrt(3)x", "- y"), ("-sqrt(3)x", "+ y")],  # m = sqrt(3)
    120: [("sqrt(3)x", "+ y")],  # m = -sqrt(3)
    150: [("x", "+ sqrt(3)y")],  # m = -1/sqrt(3)
    45: [("x", "- y")],
    135: [("x", "+ y")],
}
_SLOPE = {
    30: "(sqrt(3)/3)x",
    45: "x",
    60: "sqrt(3)x",
    120: "-sqrt(3)x",
    135: "-x",
    150: "-(sqrt(3)/3)x",
}


def _describe_line(rng: random.Random, theta: int) -> str:
    forms = ["slope"]
    if theta in _GENERAL:
        forms.append("general")
    if theta in (45, 135):
        forms.append("points")
    form = rng.choice(forms)
    if form == "slope":
        c = rng.randint(-6, 6)
        tail = "" if c == 0 else f" {'+' if c > 0 else '-'} {abs(c)}"
        return f"y = {_SLOPE[theta]}{tail}"
    if form == "general":
        lhs_x, lhs_y = rng.choice(_GENERAL[theta])
        c = rng.randint(1, 9)
        return f"{lhs_x} {lhs_y} = {c}"
    x1, y1 = rng.randint(-5, 5), rng.randint(-5, 5)
    run = rng.randint(1, 4)
    rise = run if theta == 45 else -run
    return f"the line through ({x1}, {y1}) and ({x1 + run}, {y1 + rise})"


@generator("exd_x_f5_incl", level=12, presets={"exam": dict(tier="exam")})
def exd_x_f5_incl(rng: random.Random, tier: str = "exam") -> dict:
    """Acute or obtuse angle between two lines, each given in a different form."""
    if tier != "exam":
        raise ValueError(f"unknown tier: {tier}")
    while True:
        t1, t2 = rng.sample(_ANGLES, 2)
        gap = abs(t1 - t2)
        if gap != 90:
            break
    acute = min(gap, 180 - gap)
    obtuse = 180 - acute
    want = rng.choice(["acute", "obtuse"])
    key = acute if want == "acute" else obtuse
    other = obtuse if want == "acute" else acute
    first, second = _describe_line(rng, t1), _describe_line(rng, t2)
    lines = []
    for i, d in enumerate((first, second), start=1):
        lines.append(f"Line {i} is {d}." if d.startswith("the line") else f"Line {i}: {d}.")
    text = " ".join(lines) + f" Find the {want} angle between the two lines. Give the number of degrees only."
    # Independent route: the angle between lines also satisfies tan(phi) = |(m1 - m2)/(1 + m1 m2)| for the acute angle.
    m1, m2 = math.tan(math.radians(t1)), math.tan(math.radians(t2))
    phi = math.degrees(math.atan(abs((m1 - m2) / (1 + m1 * m2))))
    check(abs(phi - acute) < 1e-6, "tan formula disagrees with the difference of inclinations")
    answer_text = str(key)
    candidates = [
        str(other),  # the other of the acute/obtuse pair
        str(abs(90 - key)),  # complement of the angle between the lines
        str(t1),  # an inclination reported as the answer
        str(t2),
        str((t1 + t2) % 180),  # sums the inclinations
    ]
    distractors = pick_distractors(answer_text, candidates, n=3)
    check(len(distractors) == 3, "fewer than 3 distractors")
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer_text,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"The inclinations are {t1} and {t2} degrees, a difference of {gap}.",
            f"The acute angle between the lines is {acute} degrees and the obtuse angle is {obtuse} degrees.",
        ],
        distractors=distractors,
    )
