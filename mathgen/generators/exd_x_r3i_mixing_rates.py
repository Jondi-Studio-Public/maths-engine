"""Exam-tier source for `senior-formulate-differential-equations-including-the-logistic-equation-that`: a
well-mixed tank, worked as the rate-of-change equation dQ/dt = (rate in) - (rate out).

The stem describes a tank of V litres holding Q0 grams of a dissolved substance. A solution of concentration c
grams per litre flows in at f litres per minute and the well-mixed contents leave at the same rate. The parts
formulate and use the model numerically:

(a) the rate at which the substance enters (g/min)         number, 1 mark    c * f
(b) the rate at which it leaves at the start (g/min)       number, 2 marks   f * Q0 / V
(c) dQ/dt at t = 0 (g/min)                                 number, 1 mark    part_a - part_b
(d) the amount the tank approaches in the long run (g)     number, 2 marks   part_a * V / f

## Construction

Every quantity is drawn so the answers are exact: (b) is drawn to be an exact number of hundredths (V divides
100 f Q0), so (c) is too, and the long-run amount c * V is a whole number. Nothing is rounded.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import attempts, check, generator, phrase
from ._exd_distinct_parts import distinct_part_answers
from .exd_x_r3i_support import number_part

#: (opening sentence, substance, concentration choices in g/L, tank volumes in L, flow rates in L/min)
_SETTINGS = (
    (
        "A tank holds {V} litres of water with {Q0} grams of salt dissolved in it. Brine containing {c} grams of salt "
        "per litre is pumped in at {f} litres per minute, and the well-mixed solution leaves the tank at the same "
        "rate.",
        "salt",
        (5, 10, 15, 20, 25, 30, 35, 40),
        (200, 250, 400, 500, 800, 1000, 1500, 2000),
        (4, 5, 6, 8, 10, 12, 15, 20, 25),
    ),
    (
        "A winery vat holds {V} litres of grape juice containing {Q0} grams of sugar. Juice with {c} grams of sugar "
        "per litre flows in at {f} litres per minute, and the stirred mixture drains out at the same rate.",
        "sugar",
        (80, 90, 100, 110, 120, 140, 160, 180, 200),
        (500, 800, 1000, 1500, 2000, 2500),
        (5, 8, 10, 12, 15, 20, 25),
    ),
    (
        "An irrigation tank holds {V} litres of water with {Q0} grams of fertiliser dissolved in it. A solution "
        "containing {c} grams of fertiliser per litre is added at {f} litres per minute, and the well-mixed tank "
        "overflows at the same rate.",
        "fertiliser",
        (2, 3, 4, 5, 6, 8, 10),
        (500, 800, 1000, 1500, 2000),
        (4, 5, 8, 10, 12, 15, 20),
    ),
)


@generator("exd_x_r3i_mixing_rates", level=15, presets={"exam": dict(tier="exam")})
@distinct_part_answers
def exd_x_r3i_mixing_rates(rng, tier: str = "exam") -> dict:
    """Inflow rate, outflow rate, dQ/dt and the long-run amount for a well-mixed tank."""
    check(tier == "exam", f"unknown tier {tier!r}")
    template, substance, c_choices, v_choices, f_choices = rng.choice(_SETTINGS)
    for _ in attempts(1000, "a tank with exact hundredths"):
        c = rng.choice(c_choices)
        V = rng.choice(v_choices)
        f = rng.choice(f_choices)
        m = rng.choice([2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15])  # start at m/10 of the inflow concentration
        Q0 = V * c * m // 10
        if (V * c * m) % 10 != 0:
            continue
        r_in = Fraction(c * f)
        r_out = Fraction(f * Q0, V)
        net = r_in - r_out
        if (100 * r_out).denominator == 1 and f < V // 10 and r_out != r_in:
            break
    long_run = Fraction(c * V)
    check(r_in * V / f == long_run, "long-run amount disagrees with c * V")
    check(net == r_in - r_out and net * f == f * c * f - f * r_out, "net rate disagrees")

    stem = template.format(V=V, Q0=Q0, c=c, f=f) + phrase(
        rng,
        (
            " Let Q be the number of grams of {s} in the tank t minutes later.",
            " After t minutes the tank contains Q grams of {s}.",
        ),
        s=substance,
    )
    part_a = number_part(
        f"Find the rate, in grams per minute, at which {substance} enters the tank.",
        r_in,
        0,
        1,
        [f"Rate in = {c} g/L × {f} L/min = {c * f} g/min."],
        [
            ("added-instead-of-multiplied", Fraction(c + f)),
            ("divided-instead-of-multiplied", Fraction(c, f)),
            ("added-instead-of-multiplied", Fraction(c - f)),
        ],
        sensible_range=(0, None),
    )
    part_b = number_part(
        f"Find the rate, in grams per minute, at which {substance} leaves the tank at the start, correct to 2 "
        "decimal places.",
        r_out,
        2,
        2,
        [f"Concentration at the start = {Q0}/{V} g/L, so rate out = {f} × {Q0}/{V} = {float(r_out):.2f} g/min."],
        [
            ("outflow-at-inflow-concentration", r_in),
            ("outflow-forgot-to-divide-by-volume", Fraction(f * Q0)),
            ("outflow-adds-flow-to-volume", Fraction(f * Q0, V + f)),
        ],
        sensible_range=(0, None),
    )
    part_c = number_part(
        "Find the value of dQ/dt when t = 0, in grams per minute, correct to 2 decimal places.",
        net,
        2,
        1,
        [f"dQ/dt = (rate in) - (rate out) = {c * f} - {float(r_out):.2f} = {float(net):.2f}."],
        [
            ("sign-of-result-reversed", -net),
            ("added-instead-of-subtracted", r_in + r_out),
            ("ignores-outflow", r_in),
            ("reports-outflow-as-net-rate", r_out),
        ],
        follow_through="part_a - part_b",
    )
    part_d = number_part(
        f"Find the amount of {substance}, in grams, that the tank approaches in the long run.",
        long_run,
        0,
        2,
        [f"dQ/dt = 0 when ({f}/{V})Q = {c * f}, so Q = {c * f} × {V}/{f} = {int(long_run)} g."],
        [
            ("equilibrium-forgot-flow-divisor", Fraction(c * f * V)),
            ("equilibrium-inverted-rate-constant", Fraction(c * f * f, V)),
            ("equilibrium-is-initial-amount", Fraction(Q0)),
        ],
        follow_through=f"part_a*{V}/{f}",
        sensible_range=(0, None),
    )
    return dict(text=stem, parts=[part_a, part_b, part_c, part_d])
