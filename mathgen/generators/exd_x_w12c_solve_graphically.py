"""Exam chain for solving a linear equation graphically: the crossing of two drawn lines, in a real situation.

Two lines show the total cost of two plans (a joining fee plus a charge for each week, hour or month) against time.
The lines are drawn through lattice points and cross at a whole-number time. Set the way a Year 8 exam sets it:

(a) the time at which the two plans cost the same, read from the graph      number, 1 mark
(b) the cost at that time, by substituting (a) into plan A's rule           number, 2 marks, follow-through (a)
(c) the greatest whole number of periods for which the cheaper-to-start
    plan is cheaper                                                          number, 1 mark,  follow-through (a)
(d) the saving from the better plan over a longer stay                      number, 2 marks, follow-through (a)

## The algebra (checked on every question)

The graph is read back from the figure's through points (the crossing is solved exactly with Fractions) and compared
with the constructed crossing. The rules A: C = fA + rA t and B: C = fB + rB t are built from the same lines, and
both must give the same cost at the crossing. The saving over a stay of N periods is |(rA - rB)(N - t*)|, which is
also found by evaluating the two rules at N.

## Distractors

* (a): the cost where the crossing is read, one gridline out, the point where a line meets the vertical axis.
* (b): the weekly charge times the time with the fee forgotten, the fee plus the time (charge not multiplied), the
  fee and charge swapped.
* (c): the crossing time itself (the plans tie there), one more than the crossing, the cost at the crossing.
* (d): the difference in charges times the whole stay, the cost of one plan, the difference in joining fees.
"""

from __future__ import annotations

from fractions import Fraction as F

from mathgen.core import attempts, check, generator, phrase
from mathgen.generators._exd_x_w12c_kit import redraw, int_part
from mathgen.generators.pv1a_line_figure_build import Window, line_figure, read_intersection, read_line

_ALT = "straight lines on a coordinate grid"

# One entry per situation: wording and the axis label.
_CONTEXTS = (
    dict(
        unit="week",
        plural="weeks",
        steps=(10,),
        axis="Time (weeks)",
        intro="Two gyms in {town} each charge a joining fee and then a fee for every week of membership. "
        "The graph shows the total cost of membership at each gym over time.",
        a_name="Gym A",
        b_name="Gym B",
        towns=("Newcastle", "Geelong", "Wollongong", "Townsville"),
        who=("Maya", "Liam", "Priya", "Jack"),
        stay="{who} wants to join for {n} weeks.",
    ),
    dict(
        unit="hour",
        plural="hours",
        steps=(10, 20),
        axis="Time (hours)",
        intro="Two shops in {town} hire out kayaks. Each charges a fixed hire fee plus an amount for every hour. "
        "The graph shows the total cost of hiring a kayak from each shop.",
        a_name="Shop A",
        b_name="Shop B",
        towns=("Noosa", "Port Macquarie", "Cairns", "Albany"),
        who=("Sam", "Chloe", "Ben", "Aisha"),
        stay="{who} wants to hire a kayak for {n} hours.",
    ),
    dict(
        unit="month",
        plural="months",
        steps=(10, 20),
        axis="Time (months)",
        intro="Two mobile phone plans from providers in {town} each charge a connection fee and then a charge every month. "
        "The graph shows the total cost of each plan over time.",
        a_name="Plan A",
        b_name="Plan B",
        towns=("Hobart", "Darwin", "Bendigo", "Mackay"),
        who=("Noah", "Zoe", "Ethan", "Mia"),
        stay="{who} wants a plan for {n} months.",
    ),
)


def _build(rng):
    """Read where two plans cost the same, check it with their rules, then use it to answer a question about a stay."""
    ctx = rng.choice(_CONTEXTS)
    ys = rng.choice(ctx["steps"])  # dollars per grid step on the cost axis
    for _ in attempts(3000, "two plans that cross at a whole-number time inside the grid"):
        m_steep, m_flat = sorted(rng.sample((1, 2, 3), 2), reverse=True)
        c_steep, c_flat = rng.randint(1, 4), rng.randint(2, 8)
        if c_flat <= c_steep or (c_flat - c_steep) % (m_steep - m_flat):
            continue
        xi = (c_flat - c_steep) // (m_steep - m_flat)
        yi = c_steep + m_steep * xi
        if 2 <= xi <= 6 and 3 <= yi <= 9 and c_flat * ys <= 100:
            break
    # The steeper plan starts cheaper (smaller fee); which plan is labelled A is drawn at random.
    steep_is_a = rng.random() < 0.5
    (mA, cA), (mB, cB) = (
        ((m_steep, c_steep), (m_flat, c_flat)) if steep_is_a else ((m_flat, c_flat), (m_steep, c_steep))
    )
    win = Window(0, 9, 0, 9, xs=1, ys=ys)
    fig = line_figure(
        win,
        [((0, cA), (xi, yi)), ((0, cB), (xi, yi))],
        _ALT,
        labels=["A", "B"],
        x_label=ctx["axis"],
        y_label="Total cost ($)",
    )
    # Second route: read the two lines back from the figure and solve them exactly.
    got_x, got_y = read_intersection(fig, 0, 1)
    check((got_x, got_y) == (F(xi), F(yi * ys)), "the lines do not cross at the constructed point")
    (rm_a, rc_a), (rm_b, rc_b) = read_line(fig, 0), read_line(fig, 1)
    fee_a, rate_a, fee_b, rate_b = int(rc_a), int(rm_a), int(rc_b), int(rm_b)
    check((fee_a, rate_a, fee_b, rate_b) == (cA * ys, mA * ys, cB * ys, mB * ys), "the rules are not the lines")
    cost = fee_a + rate_a * xi
    check(cost == fee_b + rate_b * xi == yi * ys, "the two rules disagree at the crossing")

    cheaper_first = "A" if steep_is_a else "B"  # the plan that is cheaper before the crossing
    other = "B" if steep_is_a else "A"
    n_stay = xi + rng.randint(2, 10)
    diff_rate = abs(rate_a - rate_b)
    saving = diff_rate * (n_stay - xi)
    both = (fee_a + rate_a * n_stay, fee_b + rate_b * n_stay)
    check(abs(both[0] - both[1]) == saving, "the saving is not the gap between the two costs")
    better = "A" if both[0] < both[1] else "B"
    plan = {"A": ctx["a_name"], "B": ctx["b_name"]}
    unit, plural = ctx["unit"], ctx["plural"]
    article = "an" if unit == "hour" else "a"
    who = rng.choice(ctx["who"])
    stem = phrase(rng, (ctx["intro"],), town=rng.choice(ctx["towns"])) + (
        f" Line A shows the cost of {plan['A']} and line B the cost of {plan['B']}."
    )

    part_a = int_part(
        f"Use the graph to find the number of {plural} for which the two total costs are equal.",
        xi,
        1,
        [f"The lines cross at ({xi}, {yi * ys}).", f"The costs are equal after {xi} {plural}."],
        [
            (yi * ys, "swapped-coordinates"),
            (xi + 1, "reads-one-gridline-out"),
            (xi - 1, "reads-one-gridline-out"),
            (c_flat * ys, "reads-where-a-line-meets-the-y-axis"),
        ],
        lo=0,
        hi=60,
    )

    part_b = int_part(
        f"The rule for {plan['A']} is C = {fee_a} + {rate_a}t and the rule for {plan['B']} is C = {fee_b} + {rate_b}t, "
        f"where C is the total cost in dollars and t is the time in {plural}. "
        f"Substitute your answer to part (a) into the rule for {plan['A']}. What is the total cost, in dollars, at that time?",
        cost,
        2,
        [
            f"{plan['A']}: C = {fee_a} + {rate_a} × {xi} = {cost}.",
            f"{plan['B']}: C = {fee_b} + {rate_b} × {xi} = {cost}, so the two costs agree.",
        ],
        [
            (rate_a * xi, "dropped-intercept-in-prediction"),
            (fee_a + xi, "substituted-without-multiplying"),
            (fee_a * xi + rate_a, "gradient-and-intercept-swapped"),
            (fee_a + rate_a + xi, "substituted-without-multiplying"),
        ],
        lo=0,
        hi=1000,
        follow_through=f"{fee_a} + {rate_a}*part_a",
    )
    part_c = int_part(
        f"{plan[cheaper_first]} costs less than {plan[other]} for a short time, then costs more. "
        f"What is the greatest whole number of {plural} for which {plan[cheaper_first]} is cheaper?",
        xi - 1,
        1,
        [
            f"At {xi} {plural} the two costs are equal, so {plan[cheaper_first]} is not cheaper then.",
            (
                f"It is cheaper for 1 {unit}."
                if xi == 2
                else f"It is cheaper for 1 and 2 {plural}."
                if xi == 3
                else f"It is cheaper for every whole number of {plural} from 1 to {xi - 1}."
            ),
        ],
        [
            (xi, "strict-bound-taken-as-value"),
            (xi + 1, "reverses-the-inequality-direction"),
            (yi * ys, "swapped-coordinates"),
        ],
        lo=0,
        hi=60,
        follow_through="part_a - 1",
    )
    part_d = int_part(
        f"{ctx['stay'].format(who=who, n=n_stay)} How many dollars does {who} save by choosing the cheaper plan instead of the dearer one?",
        saving,
        2,
        [
            f"Past {xi} {plural}, {plan[better]} is cheaper. The charges differ by ${diff_rate} {article} {unit}.",
            f"The saving is {diff_rate} × ({n_stay} - {xi}) = {saving} dollars.",
            f"Check: {plan['A']} costs {both[0]} and {plan['B']} costs {both[1]} for {n_stay} {plural}.",
        ],
        [
            (diff_rate * n_stay, "compares-gradients-only"),
            (min(both), "difference-not-taken"),
            (abs(fee_a - fee_b), "compares-starting-values-only"),
            (diff_rate * (n_stay + xi), "substitution-sign-error"),
        ],
        lo=0,
        hi=2000,
        follow_through=f"{diff_rate}*({n_stay} - part_a)",
    )
    return dict(text=stem, figure=fig, parts=[part_a, part_b, part_c, part_d])


@generator("exd_x_w12c_solve_graphically", level=12, presets={"exam": {"tier": "exam"}})
def exd_x_w12c_solve_graphically(rng, tier: str = "exam"):
    """Read where two plans cost the same, check it with their rules, then use it to answer a question about a stay."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return redraw(rng, _build)
