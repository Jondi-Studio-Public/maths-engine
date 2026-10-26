"""`pw1_piecewise_cost_graph`: read a drawn piece-wise LINEAR cost graph (ACMGM047, the linear half).

``senior-read-a-piece-wise-linear-cost-graph``. A cost-against-usage graph made of two or three straight sections, with a
fixed charge where it meets the cost axis and a rate that changes at each kink. Each context has its own realistic
ranges (a taxi flagfall of a few dollars and a rate of a few dollars a kilometre, a quarterly electricity supply charge
and a rate of tens of cents a kilowatt-hour, and so on). Money is whole cents, shown in dollars to two places. Every
vertex, and the point where two plans meet, lies on the gridlines (the usage step and the cost step are chosen per
draw), so each answer is read, not estimated. Nothing the keys need is printed: the figure carries axis labels and, on
the exam tier, the two plan names.

Tiers (the ``tier`` parameter; unknown tiers raise ``ValueError``):

* easy: the cost at a stated usage (in either section), or the fixed charge, the cost at no usage.
* medium: the rate (gradient) of one section between two stated usages, or the usage at which the rate first changes.
* hard: the usage that gives a stated cost in a later section, or the cost beyond the right end of the drawing when
  the last section continues at its rate.
* exam: two plans on one graph. Plan A is a straight line (fixed charge stated), Plan B has two sections. Parts: (a) the
  cost of B at a stated usage, (b) the rate of A, (c) the usage at which the plans cost the same, (d) the common
  cost from A's fixed charge, rate (b) and usage (c), with follow-through.

Keys are built from the chosen vertices (in cents) and re-derived by evaluating the piece-wise function at every whole
usage; the exam crossing is checked to be the only point where the two plans agree.
"""

from __future__ import annotations

import math
import random
import re
from fractions import Fraction

from .. import figures
from ..core import VerificationError, attempts, check, generator, phrase
from ._exd_x_w12a_kit import split_parts
from ._r3d_kit import number_fields as _number_fields
from ._r3d_kit import number_part as _number_part
from ._spread_kit import NEAR, mixed, uses_rng

TOPIC = "pw1_piecewise_cost_graph"
ALT = "a graph of cost against usage made of straight sections, with a different rate in each section"
ALT_TWO = "a graph of cost against usage for two plans, one a single straight line and one with two sections"

# Per context: usage gridline steps (gx), number of usage steps (n), fixed charge and rate ranges in cents, cost
# gridline steps in cents (gy).
CONTEXTS = [
    dict(
        intro="The graph shows the monthly cost of a mobile data plan.",
        two="The graph shows the monthly cost of two mobile data plans.",
        tag="Plan",
        things="plans",
        qty="data used",
        short="data",
        unit="gigabyte",
        units="gigabytes",
        cost="monthly cost",
        fixed="monthly fee",
        rstep=1,
        x_label="Data used (GB)",
        y_label="Monthly cost ($)",
        gx=tuple(range(1, 11)),
        max=100,
        n=(10, 32),
        fee=(1000, 6000),
        rate=(100, 1200),
        gy=(25, 50, 100, 200, 250, 500, 1000),
    ),
    dict(
        intro="The graph shows the fare for a taxi trip.",
        two="The graph shows the fare charged by two taxi companies.",
        tag="Company",
        things="companies",
        qty="distance travelled",
        short="travel",
        unit="kilometre",
        units="kilometres",
        cost="fare",
        fixed="flagfall",
        rstep=1,
        x_label="Distance (km)",
        y_label="Fare ($)",
        gx=tuple(range(1, 11)),
        max=60,
        n=(10, 32),
        fee=(300, 1200),
        rate=(100, 400),
        gy=(25, 50, 100, 200, 250, 500),
    ),
    dict(
        intro="The graph shows a household's electricity bill for one quarter.",
        two="The graph shows the quarterly bill from two electricity retailers.",
        tag="Retailer",
        things="retailers",
        qty="electricity used",
        short="electricity",
        unit="kilowatt-hour",
        units="kilowatt-hours",
        cost="bill",
        fixed="supply charge",
        rstep=1,
        x_label="Electricity used (kWh)",
        y_label="Bill ($)",
        gx=tuple(range(10, 201, 5)),
        max=5000,
        n=(10, 32),
        fee=(2000, 9000),
        rate=(15, 45),
        gy=(250, 500, 1000, 1250, 2000, 2500, 5000),
    ),
    dict(
        intro="The graph shows the cost of hiring a trailer.",
        two="The graph shows the cost of hiring a trailer from two companies.",
        tag="Company",
        things="companies",
        qty="hire time",
        short="hire",
        unit="day",
        units="days",
        cost="hire cost",
        fixed="booking fee",
        rstep=50,
        x_label="Hire time (days)",
        y_label="Cost ($)",
        gx=(1,),
        max=14,
        n=(10, 32),
        fee=(2000, 9000),
        rate=(1000, 6000),
        gy=(200, 250, 400, 500, 1000, 2000, 2500, 5000),
    ),
    dict(
        intro="The graph shows the charge for sending a parcel by courier.",
        two="The graph shows the charge made by two courier services for a parcel.",
        tag="Service",
        things="services",
        qty="parcel mass",
        short="parcel mass",
        unit="kilogram",
        units="kilograms",
        cost="charge",
        fixed="pick-up fee",
        rstep=1,
        x_label="Mass (kg)",
        y_label="Charge ($)",
        gx=tuple(range(1, 7)),
        max=50,
        n=(10, 32),
        fee=(500, 2000),
        rate=(200, 900),
        gy=(50, 100, 200, 250, 500),
    ),
    dict(
        intro="The graph shows a household's water bill for one quarter.",
        two="The graph shows the quarterly water bill from two water suppliers.",
        tag="Supplier",
        things="suppliers",
        qty="water used",
        short="water",
        unit="kilolitre",
        units="kilolitres",
        cost="bill",
        fixed="service charge",
        rstep=1,
        x_label="Water used (kL)",
        y_label="Bill ($)",
        gx=tuple(range(5, 61)),
        max=400,
        n=(10, 32),
        fee=(3000, 10000),
        rate=(200, 500),
        gy=(250, 500, 1000, 2000, 2500),
    ),
    dict(
        intro="The graph shows the cost of hiring an electric scooter.",
        two="The graph shows the cost of hiring an electric scooter from two companies.",
        tag="Company",
        things="companies",
        qty="ride time",
        short="riding",
        unit="minute",
        units="minutes",
        cost="hire cost",
        fixed="unlock fee",
        rstep=1,
        x_label="Ride time (minutes)",
        y_label="Hire cost ($)",
        gx=tuple(range(1, 16)),
        max=120,
        n=(10, 32),
        fee=(100, 500),
        rate=(20, 90),
        gy=(25, 50, 100, 200, 250, 500),
    ),
    dict(
        intro="The graph shows the charge for printing at a print shop.",
        two="The graph shows the charge made by two print shops.",
        tag="Shop",
        things="shops",
        qty="pages printed",
        short="printing",
        unit="page",
        units="pages",
        cost="charge",
        fixed="setup fee",
        rstep=1,
        x_label="Pages printed",
        y_label="Charge ($)",
        gx=tuple(range(5, 201, 5)),
        max=2000,
        n=(10, 32),
        fee=(200, 1000),
        rate=(5, 30),
        gy=(25, 50, 100, 200, 250, 500),
    ),
    dict(
        intro="The graph shows a household's gas bill for one quarter.",
        two="The graph shows the quarterly gas bill from two gas retailers.",
        tag="Retailer",
        things="retailers",
        qty="gas used",
        short="gas",
        unit="megajoule",
        units="megajoules",
        cost="bill",
        fixed="supply charge",
        rstep=1,
        x_label="Gas used (MJ)",
        y_label="Bill ($)",
        gx=tuple(range(50, 501, 25)),
        max=8000,
        n=(10, 32),
        fee=(3000, 9000),
        rate=(2, 8),
        gy=(500, 1000, 2000, 2500, 5000),
    ),
    dict(
        intro="The graph shows the cost of renting a self-storage unit.",
        two="The graph shows the cost of renting a self-storage unit from two storage companies.",
        tag="Company",
        things="companies",
        qty="rental time",
        short="rental",
        unit="week",
        units="weeks",
        cost="rental cost",
        fixed="admin fee",
        rstep=50,
        x_label="Rental time (weeks)",
        y_label="Rental cost ($)",
        gx=(1, 2, 3, 4, 5, 6),
        max=52,
        n=(10, 32),
        fee=(2000, 8000),
        rate=(1500, 4500),
        gy=(500, 1000, 2000, 2500, 5000),
    ),
    dict(
        intro="The graph shows the monthly cost of a mobile phone plan, by the call time used.",
        two="The graph shows the monthly cost of two mobile phone plans, by the call time used.",
        tag="Plan",
        things="plans",
        qty="call time",
        short="calls",
        unit="minute",
        units="minutes",
        cost="monthly cost",
        fixed="plan fee",
        rstep=1,
        x_label="Call time (minutes)",
        y_label="Monthly cost ($)",
        gx=tuple(range(5, 101)),
        max=2000,
        n=(10, 32),
        fee=(1000, 4000),
        rate=(8, 40),
        gy=(250, 500, 1000, 1250, 2000, 2500),
    ),
    dict(
        intro="The graph shows the charge for road freight of a load.",
        two="The graph shows the charge made by two road freight companies.",
        tag="Company",
        things="companies",
        qty="load mass",
        short="freight",
        unit="kilogram",
        units="kilograms",
        cost="freight charge",
        fixed="booking fee",
        rstep=5,
        x_label="Load mass (kg)",
        y_label="Freight charge ($)",
        gx=tuple(range(10, 201, 5)),
        max=3000,
        n=(10, 32),
        fee=(2000, 8000),
        rate=(50, 250),
        gy=(500, 1000, 2000, 2500, 5000),
    ),
    dict(
        intro="The graph shows the cost of a delivery of garden soil.",
        two="The graph shows the cost of a delivery of garden soil from two suppliers.",
        tag="Supplier",
        things="suppliers",
        qty="soil delivered",
        short="soil",
        unit="tonne",
        units="tonnes",
        cost="delivery cost",
        fixed="delivery fee",
        rstep=50,
        x_label="Soil delivered (tonnes)",
        y_label="Delivery cost ($)",
        gx=(1, 2, 3, 4, 5),
        max=30,
        n=(10, 32),
        fee=(3000, 9000),
        rate=(2000, 6000),
        gy=(500, 1000, 2000, 2500, 5000),
    ),
]

K_MIN, K_MAX = 8, 40  # the tallest point is this many gridlines up


def _pick_context(rng):
    """A context, the ones with a wide range of usages twice as likely (their crossings take more values)."""
    return rng.choices(CONTEXTS, weights=[2 if c["max"] >= 100 else 1 for c in CONTEXTS])[0]


# ---------------------------------------------------------------- money


def _d(cents):
    """Cents as a Fraction of dollars."""
    return Fraction(cents, 100)


def _m(cents):
    """Cents as dollars for a stem: $8 or $8.50."""
    cents = int(cents)
    return f"{cents // 100}" if cents % 100 == 0 else f"{cents // 100}.{cents % 100:02d}"


def _text(v, dp):
    return f"{float(v):.{dp}f}" if dp else str(int(v))


# ---------------------------------------------------------------- the shape


class _Retry(Exception):
    """This draw cannot be read cleanly off the grid: draw again."""


class Shape:
    """A piece-wise linear cost function in whole cents. Vertices sit on the gridlines (usage step gx, cost step gy)."""

    def __init__(self, xs, ys, gx, gy):
        self.xs = list(xs)
        self.ys = list(ys)
        self.gx, self.gy = gx, gy
        self.g = gy
        self.rates = [Fraction(ys[i + 1] - ys[i], xs[i + 1] - xs[i]) for i in range(len(xs) - 1)]
        self.fee = ys[0]

    @classmethod
    def from_rates(cls, xs, fee, rates, gx, gy):
        ys = [fee]
        for i, r in enumerate(rates):
            ys.append(ys[-1] + r * (xs[i + 1] - xs[i]))
        return cls(xs, ys, gx, gy)

    @property
    def right(self):
        return self.xs[-1]

    def cost(self, x):
        for i, r in enumerate(self.rates):
            if x <= self.xs[i + 1]:
                return self.ys[i] + r * (x - self.xs[i])
        return self.ys[-1] + self.rates[-1] * (x - self.xs[-1])

    def section(self, x):
        for i in range(len(self.rates)):
            if x <= self.xs[i + 1]:
                return i
        return len(self.rates) - 1


def _bends(a, b):
    """The rate changes by at least a fifth, so the kink can be seen."""
    return abs(a - b) * 5 >= max(a, b)


def _rate_choices(ctx, gy, dx):
    """Rates (cents per unit) in the context's range for which a section dx units wide rises by whole cost steps."""
    step = gy // math.gcd(gy, dx)
    lo, hi = ctx["rate"]
    return list(range(-(-lo // step) * step, hi + 1, step))


def _fee_choices(ctx, gy, least=1):
    lo, hi = ctx["fee"]
    return [k * gy for k in range(max(least, -(-lo // gy)), hi // gy + 1)]


def _draw_shape(rng, ctx, nseg):
    for _ in attempts(4000, "a cost graph that fits the grid"):
        gx = rng.choice(ctx["gx"])
        if ctx["max"] // gx < 8:
            continue
        n = rng.randint(8, min(14, ctx["max"] // gx))
        cuts = sorted(rng.sample(range(2, n - 1), nseg - 1))
        ms = [0, *cuts, n]
        if any(b - a < 2 for a, b in zip(ms, ms[1:])):
            continue
        gy = rng.choice(ctx["gy"])
        fees = _fee_choices(ctx, gy)
        if not fees:
            continue
        rates = []
        for a, b in zip(ms, ms[1:]):
            options = _rate_choices(ctx, gy, (b - a) * gx)
            if not options:
                break
            rates.append(rng.choice(options))
        if len(rates) < nseg or not all(_bends(a, b) for a, b in zip(rates, rates[1:])):
            continue
        sh = Shape.from_rates([m * gx for m in ms], rng.choice(fees), rates, gx, gy)
        if K_MIN <= max(sh.ys) // gy <= K_MAX:
            return sh
    raise AssertionError("unreachable")


def _check_shape(sh):
    """Every vertex on the gridlines, and the function re-derived from its vertices at every whole usage."""
    for x, y in zip(sh.xs, sh.ys):
        check(x % sh.gx == 0 and y % sh.gy == 0, "a vertex is off the gridlines")
    for x in range(sh.right + 1):
        sec = sh.section(x)
        check(
            sh.xs[sec] <= x <= sh.xs[sec + 1]
            and sh.cost(x)
            == sh.ys[sec] + (sh.ys[sec + 1] - sh.ys[sec]) * Fraction(x - sh.xs[sec], sh.xs[sec + 1] - sh.xs[sec]),
            "cost differs between the two routes",
        )


# ---------------------------------------------------------------- the figure


def _plane(gx, gy, top, right):
    return dict(
        x_range=(0, right + gx),
        y_range=(0, float(_d(top + 2 * gy))),
        grid=1,
        x_grid=gx,
        y_grid=float(_d(gy)),
        equal_scale=False,
    )


def _pt(x, cents):
    return [x, float(_d(cents))]


def _figure(ctx, sh):
    segments = [{"from": _pt(sh.xs[i], sh.ys[i]), "to": _pt(sh.xs[i + 1], sh.ys[i + 1])} for i in range(len(sh.rates))]
    points = [{"at": _pt(x, y)} for x, y in zip(sh.xs, sh.ys)]
    return figures.cartesian(
        alt=ALT,
        segments=segments,
        points=points,
        x_label=ctx["x_label"],
        y_label=ctx["y_label"],
        **_plane(sh.gx, sh.gy, max(sh.ys), sh.right),
    )


def _figure_two(ctx, a, b):
    tag = ctx["tag"]
    segments = [{"from": _pt(0, a.ys[0]), "to": _pt(a.right, a.ys[-1]), "label": f"{tag} A"}]
    for i in range(len(b.rates)):
        seg = {"from": _pt(b.xs[i], b.ys[i]), "to": _pt(b.xs[i + 1], b.ys[i + 1])}
        if i == 0:
            seg["label"] = f"{tag} B"
        segments.append(seg)
    points = [{"at": _pt(x, y)} for x, y in zip(b.xs, b.ys)]
    return figures.cartesian(
        alt=ALT_TWO,
        segments=segments,
        points=points,
        x_label=ctx["x_label"],
        y_label=ctx["y_label"],
        **_plane(a.gx, a.gy, max(*a.ys, *b.ys), a.right),
    )


# ---------------------------------------------------------------- the readers


def _lines(fig):
    """Segments as (x0, y0, x1, y1) in dollars."""
    return [tuple(Fraction(str(v)) for v in (*s["from"], *s["to"])) for s in fig["segments"]]


def _eval(segs, x):
    """Value of a chain of segments at x, the last one extended beyond its end."""
    x = Fraction(x)
    for x0, y0, x1, y1 in segs:
        if x <= x1:
            return y0 + (y1 - y0) / (x1 - x0) * (x - x0)
    x0, y0, x1, y1 = segs[-1]
    return y1 + (y1 - y0) / (x1 - x0) * (x - x1)


def _fmt(v, dp=2):
    v = Fraction(v)
    check((v * 10**dp).denominator == 1, "a read value has more places than the answer shows")
    return _text(v, dp)


def read_single(question) -> str:
    """Re-derive the answer to a single-answer question from the figure and the numbers the prompt states."""
    segs = _lines(question.figure)
    text = question.text
    nums = [int(n) for n in re.findall(r"\d+", text)]
    if "continues" in text:
        return _fmt(_eval(segs, nums[0]))
    if "$" in text:  # the cost is stated: find the usage that gives it
        target = Fraction(re.search(r"\$(\d+(?:\.\d+)?)", text).group(1))
        for x0, y0, x1, y1 in segs:
            if y0 <= target <= y1:
                return _fmt(x0 + (target - y0) / ((y1 - y0) / (x1 - x0)), 0)
        raise ValueError("cost is not on the graph")
    if "different rate" in text:  # the first kink
        return _fmt(segs[0][2], 0)
    if "each extra" in text:
        a, b = nums[0], nums[1]
        return _fmt((_eval(segs, b) - _eval(segs, a)) / (b - a))
    return _fmt(_eval(segs, nums[0]))


def read_exam(question) -> str:
    segs = _lines(question.figure)
    labels = [s.get("label") for s in question.figure["segments"]]
    a_seg = [segs[0]]
    b_segs = segs[1:]
    check(labels[0] is not None and labels[1] is not None, "the plans are not labelled")
    prompts = split_parts(question.text)
    u = int(re.search(r"\d+", prompts["a"]).group())
    a_rate = (a_seg[0][3] - a_seg[0][1]) / (a_seg[0][2] - a_seg[0][0])
    cross = [x for x in range(int(a_seg[0][2]) + 1) if _eval(a_seg, x) == _eval(b_segs, x)]
    check(len(cross) == 1, "the plans do not meet exactly once")
    return f"(a) {_fmt(_eval(b_segs, u))}; (b) {_fmt(a_rate)}; (c) {cross[0]}; (d) {_fmt(_eval(a_seg, cross[0]))}"


def _read(question) -> str:
    return read_exam(question) if question.parts else read_single(question)


figures.register_reader(TOPIC, _read)


# ---------------------------------------------------------------- wording

READ_ASK = (
    "{intro} Use the graph to find the {cost}, in dollars, when the {qty} is {u_text}.",
    "{intro} What is the {cost}, in dollars, for {u_text} of {qty_short}?",
    "{intro} According to the graph, how many dollars is the {cost} when the {qty} is {u_text}?",
    "Use the graph. {intro} Find the {cost} in dollars when the {qty} is {u_text}.",
)
FEE_ASK = (
    "{intro} The {fixed} is the {cost} when the {qty} is 0 {units}. What is the {fixed}, in dollars?",
    "{intro} Find the {fixed}, in dollars. It is the {cost} when the {qty} is 0 {units}.",
    "{intro} Use the graph to find the {fixed}, in dollars: the {cost} when the {qty} is 0 {units}.",
)
RATE_ASK = (
    "{intro} Between {a} and {b} {units}, by how many dollars does the {cost} rise for each extra {unit}?",
    "{intro} For {qty} from {a} to {b} {units}, how many dollars does each extra {unit} add to the {cost}?",
    "{intro} Find the rate, in dollars for each extra {unit}, when the {qty} is between {a} and {b} {units}.",
)
KINK_ASK = (
    "{intro} At what {qty}, in {units}, does the {cost} first start to rise at a different rate?",
    "{intro} How many {units} is the {qty} when the rate of increase of the {cost} first changes to a different rate?",
    "{intro} Find the {qty}, in {units}, at which the graph first changes to a different rate.",
)
USAGE_ASK = (
    "{intro} The {cost} was ${c}. What was the {qty}, in {units}?",
    "{intro} Find the {qty}, in {units}, when the {cost} is ${c}.",
    "{intro} For how many {units} of {qty_short} is the {cost} equal to ${c}?",
)
EXTEND_ASK = (
    "{intro} The last section continues at the same rate beyond the graph. What is the {cost}, in dollars, when the {qty} is {v} {units}?",
    "{intro} Suppose the last section continues at the same rate. Find the {cost}, in dollars, for {v} {units}.",
    "{intro} If the last section continues at the same rate, how many dollars is the {cost} when the {qty} is {v} {units}?",
)

EXAM_A = (
    "What is the {cost} of {tag} B, in dollars, when the {qty} is {u_text}?",
    "Use the graph to find what {tag} B charges, in dollars, when the {qty} is {u_text}.",
    "How many dollars does {tag} B cost when the {qty} is {u_text}?",
)
EXAM_B = (
    "{tag} A rises at the same rate throughout. Find that rate, in dollars for each extra {unit}.",
    "Find the rate, in dollars for each extra {unit}, at which the {cost} of {tag} A rises.",
    "By how many dollars does the {cost} of {tag} A rise for each extra {unit}?",
)
EXAM_C = (
    "For what {qty}, in {units}, do {tag} A and {tag} B charge the same?",
    "At what {qty}, in {units}, do the two {things} cost the same?",
    "Find the {qty}, in {units}, at which {tag} A and {tag} B give the same {cost}.",
)
EXAM_D = (
    "Use {tag} A's fixed charge of ${fa}, your rate from (b) and your {qty} from (c) to find the {cost}, in dollars, at which the two {things} charge the same.",
    "Hence find the {cost}, in dollars, that both {things} charge at the {qty} in (c), using {tag} A's fixed charge of ${fa} and your answer to (b).",
    "Starting from {tag} A's fixed charge of ${fa}, use your answers to (b) and (c) to find the common {cost}, in dollars.",
)


def _amount(ctx, n):
    return f"{n} {ctx['unit'] if n == 1 else ctx['units']}"


def _slots(ctx, **extra):
    return dict(
        intro=ctx["intro"],
        cost=ctx["cost"],
        qty=ctx["qty"],
        qty_short=ctx["short"],
        unit=ctx["unit"],
        units=ctx["units"],
        fixed=ctx["fixed"],
        tag=ctx["tag"],
        things=ctx["things"],
        **extra,
    )


# ---------------------------------------------------------------- single-answer tiers


def _interior(rng, sh, section=None):
    """A usage on a usage gridline strictly inside a section (not at a vertex) whose cost is on a cost gridline."""
    pool = [
        x
        for x in range(sh.gx, sh.right, sh.gx)
        if x not in sh.xs and (section is None or sh.section(x) == section) and sh.cost(x) % sh.gy == 0
    ]
    if not pool:
        raise _Retry
    return rng.choice(pool)


def _whole_or_none(v):
    v = Fraction(v)
    return int(v) if v.denominator == 1 and v > 0 else None


def _easy(rng):
    ctx = _pick_context(rng)
    sh = _draw_shape(rng, ctx, rng.choice((2, 2, 3)))
    _check_shape(sh)
    top, gx = max(sh.ys), sh.gx
    if rng.random() < 0.25:  # the fixed charge
        key = _d(sh.fee)
        wrong = [
            (_d(sh.rates[0]), "pwg-reads-the-rate-for-the-cost"),
            (_d(sh.ys[1]), "pwg-cost-at-the-breakpoint-for-the-start"),
            (_d(sh.cost(gx)), "pwg-reads-the-neighbouring-gridline"),
            (Fraction(sh.xs[1]), "pwg-breakpoint-for-the-cost"),
        ]
        text = phrase(rng, FEE_ASK, **_slots(ctx))
        working = [f"The graph meets the cost axis at ${_m(sh.fee)}, so the {ctx['fixed']} is {_m(sh.fee)}."]
        check(sh.cost(0) == sh.fee, "fixed charge differs between routes")
    else:
        section = rng.choice(range(len(sh.rates)))
        u = _interior(rng, sh, section)
        key = _d(sh.cost(u))
        sec = sh.section(u)
        wrong = [
            (key - _d(sh.fee), "pwg-leaves-out-the-fixed-charge"),
            (_d(sh.fee + sh.rates[0] * u), "pwg-first-rate-for-all-units"),
            (_d(sh.cost(u + gx)), "pwg-reads-the-neighbouring-gridline"),
            (_d(sh.cost(u - gx)), "pwg-reads-the-neighbouring-gridline"),
            (Fraction(u), "pwg-reads-the-usage-for-the-cost"),
            (_d(sh.rates[sec]), "pwg-reads-the-rate-for-the-cost"),
            (_d(sh.ys[sec]), "pwg-cost-at-the-breakpoint-for-the-cost"),
        ]
        text = phrase(rng, READ_ASK, u_text=_amount(ctx, u), **_slots(ctx))
        working = [f"Go up from {_amount(ctx, u)} to the graph and across to the cost axis: {_text(key, 2)}."]
    return dict(
        text=text, key=key, dp=2, wrong=wrong, working=working, fig=_figure(ctx, sh), hi=_d(top), step=_d(sh.gy)
    )


def _medium(rng):
    ctx = _pick_context(rng)
    sh = _draw_shape(rng, ctx, rng.choice((2, 3, 3)))
    _check_shape(sh)
    top = max(sh.ys)
    if rng.random() < 0.6:
        s = rng.randrange(len(sh.rates))
        a, b = sh.xs[s], sh.xs[s + 1]
        key = _d(sh.rates[s])
        check(_d(sh.cost(b) - sh.cost(a)) / (b - a) == key, "gradient differs between routes")
        others = [_d(r) for i, r in enumerate(sh.rates) if i != s]
        wrong = [
            (others[0], "pwg-other-section-rate"),
            (_d(sh.cost(b) - sh.cost(a)), "pwg-total-change-for-rate"),
            (_d(sh.cost(b)), "pwg-cost-at-the-end-for-rate"),
            (_d(sh.cost(b)) / b, "pwg-average-rate-for-gradient"),
            (Fraction(b - a), "pwg-run-for-rate"),
            (_d(sh.fee), "pwg-fixed-charge-for-rate"),
            (others[-1], "pwg-other-section-rate"),
        ]
        text = phrase(rng, RATE_ASK, a=a, b=b, **_slots(ctx))
        working = [
            f"From {a} to {b} {ctx['units']} the cost rises from {_m(sh.cost(a))} to {_m(sh.cost(b))}.",
            f"Rate = {_m(sh.cost(b) - sh.cost(a))} / {b - a} = {_text(key, 2)} for each extra {ctx['unit']}.",
        ]
        return dict(
            text=text,
            key=key,
            dp=2,
            wrong=wrong,
            working=working,
            fig=_figure(ctx, sh),
            hi=_d(top),
            step=Fraction(1, 20),
        )
    key = Fraction(sh.xs[1])
    check(sh.rates[0] != sh.rates[1], "no change at the first kink")
    wrong = [
        (_d(sh.ys[1]), "pwg-cost-at-the-change-for-usage"),
        (Fraction(sh.xs[2]), "pwg-other-breakpoint"),
        (Fraction(sh.right), "pwg-end-of-graph-for-breakpoint"),
        (_d(sh.rates[0]), "pwg-reads-the-rate-for-the-cost"),
        (_d(sh.fee), "pwg-fixed-charge-for-breakpoint"),
    ]
    text = phrase(rng, KINK_ASK, **_slots(ctx))
    working = [f"The graph bends first at {sh.xs[1]} {ctx['units']}, where the rate changes."]
    return dict(
        text=text, key=key, dp=0, wrong=wrong, working=working, fig=_figure(ctx, sh), hi=Fraction(sh.right), step=sh.gx
    )


def _hard(rng):
    ctx = _pick_context(rng)
    sh = _draw_shape(rng, ctx, rng.choice((2, 3, 3)))
    _check_shape(sh)
    top, gx = max(sh.ys), sh.gx
    if rng.random() < 0.6:
        section = rng.randrange(1, len(sh.rates))
        u = _interior(rng, sh, section)
        c = sh.cost(u)
        found = [x for x in range(sh.right + 1) if sh.cost(x) == c]
        check(found == [u], "usage is not unique")
        r = sh.rates[section]
        wrong = [
            (_d(c), "pwg-reads-the-cost-for-the-usage"),
            ((c - sh.fee) / sh.rates[0], "pwg-first-rate-for-all-units"),
            (Fraction(u - sh.xs[section]), "pwg-forgets-the-breakpoint-offset"),
            (Fraction(c) / r, "pwg-divides-cost-by-rate"),
            (Fraction(c - sh.fee) / r, "pwg-divides-cost-by-rate"),
            (Fraction(sh.xs[section]), "pwg-breakpoint-for-the-usage"),
            (Fraction(u + gx), "pwg-reads-the-neighbouring-gridline"),
        ]
        text = phrase(rng, USAGE_ASK, c=_m(c), **_slots(ctx))
        working = [f"Go across from ${_m(c)} to the graph, then down to the usage axis: {_amount(ctx, u)}."]
        return dict(
            text=text,
            key=Fraction(u),
            dp=0,
            wrong=wrong,
            working=working,
            fig=_figure(ctx, sh),
            hi=Fraction(sh.right),
            step=gx,
        )
    d = rng.randint(2, 6) * gx
    v = sh.right + d
    key = _d(sh.cost(v))
    r_last, r_first = sh.rates[-1], sh.rates[0]
    check(sh.ys[-1] + r_last * d == sh.cost(v), "extension differs between routes")
    wrong = [
        (_d(sh.ys[-1] + r_first * d), "pwg-extends-at-the-wrong-rate"),
        (_d(r_last * v), "pwg-extends-from-the-origin"),
        (_d(sh.fee + r_last * v), "pwg-extends-from-the-fixed-charge"),
        (_d(sh.ys[-1] + d), "pwg-extra-units-not-multiplied-by-rate"),
        (_d(sh.ys[-1] + r_last * (d + gx)), "pwg-extends-one-unit-too-far"),
        (_d(sh.ys[-1] + r_last * (d - gx)), "pwg-extends-one-unit-too-far"),
    ]
    text = phrase(rng, EXTEND_ASK, v=v, **_slots(ctx))
    working = [
        f"At {sh.right} {ctx['units']} the cost is {_m(sh.ys[-1])} and the last section rises {_m(r_last)} for each extra {ctx['unit']}.",
        f"Cost at {v} {ctx['units']} = {_m(sh.ys[-1])} + {d} x {_m(r_last)} = {_text(key, 2)}.",
    ]
    return dict(text=text, key=key, dp=2, wrong=wrong, working=working, fig=_figure(ctx, sh), hi=None, step=_d(sh.gy))


# ---------------------------------------------------------------- the options


def _options(key, wrong, dp, step, hi, unbounded=None):
    """The wrong options in a random rank order: named slips, with plain near misses one step apart as fillers.

    A whole-number answer is offered only whole-number options."""
    wrong = [(v, t) for v, t in wrong if v is not None and v > 0 and (dp > 0 or Fraction(v).denominator == 1)]
    near = [(key + d * step, NEAR) for d in (-3, -2, -1, 1, 2, 3) if key + d * step > 0]
    return mixed(key, wrong, fillers=near, lo=step if dp == 0 else Fraction(1, 100), hi=hi, unbounded=unbounded)


def number_fields(key, dp, wrong, step, hi):
    opts = _options(key, wrong, dp, step, hi, "open" if hi is None else None)
    return _number_fields(key, dp, opts, nonneg=True)


def part(prompt, key, dp, marks, working, wrong, step, hi, **kw):
    return _number_part(prompt, key, dp, marks, working, _options(key, wrong, dp, step, hi), nonneg=True, **kw)


# ---------------------------------------------------------------- the exam


def _draw_exam(rng, ctx):
    """Plan A a straight line, Plan B two sections, meeting once at a point on the gridlines (all in cents).

    The usage step, the number of steps and the crossing are drawn first, then the rates and cost step are searched
    for, so the crossing takes many different values."""
    for _ in attempts(400, "a crossing that two plans can be drawn to meet at"):
        gx = rng.choice(ctx["gx"])
        if ctx["max"] // gx < 10:
            continue
        n = rng.randint(max(ctx["n"][0], 10), min(ctx["n"][1], ctx["max"] // gx))
        m = rng.randint(3, n - 5)
        for _ in range(150):
            found = _try_exam(rng, ctx, gx, n, m)
            if found is not None:
                return found
    raise AssertionError("unreachable")


def _try_exam(rng, ctx, gx, n, m):
    lo_r, hi_r = ctx["rate"]
    right, xc = n * gx, m * gx
    step = ctx["rstep"]
    band = list(range(-(-lo_r // step) * step, hi_r + 1, step))
    r = rng.choice(band)  # Plan A's rate
    # a cost step that makes the graph 8 to 40 gridlines tall, and puts Plan A on a gridline within the graph
    # (so its rate can be read)
    gys = [
        g
        for g in ctx["gy"]
        if K_MIN + 2 <= (r * right + ctx["fee"][1] // 2) / g <= K_MAX - 1 and g // math.gcd(g, r * gx) <= n - 2
    ]
    if not gys:
        return None
    gy = rng.choice(gys)
    fees = _fee_choices(ctx, gy, least=2)
    if not fees:
        return None
    fa = rng.choice(fees)
    cstar = fa + r * xc
    a = Shape([0, right], [fa, fa + r * right], gx, gy)
    late = rng.random() < 0.5
    found = None
    if late:  # B starts below A, bends at x1 still below A, then climbs through the crossing to the right-hand edge
        x1 = rng.randint(2, m - 1) * gx
        lows = [v for v in band if v < r]
        rb1 = rng.choice(lows) if lows else None
        ups = []
        if rb1 is not None:
            ups = [
                v
                for v in band
                if v > r
                and _bends(rb1, v)
                and gy <= cstar - rb1 * x1 - v * (xc - x1) < fa
                and (cstar - rb1 * x1 - v * (xc - x1)) % gy == 0
            ]
        if ups:
            rb2 = rng.choice(ups)
            found = (cstar - rb1 * x1 - rb2 * (xc - x1), x1, rb1, rb2)
    else:  # B starts below A, climbs steeply through the crossing to x1, then carries on to the right-hand edge
        x1 = rng.randint(m + 1, n - 1) * gx
        steep = [v for v in band if v > r and gy <= cstar - v * xc < fa and (cstar - v * xc) % gy == 0]
        if steep:
            rb1 = rng.choice(steep)
            bent = [v for v in band if _bends(rb1, v)]
            found = (cstar - rb1 * xc, x1, rb1, rng.choice(bent)) if bent else None
    if found is None:
        return None
    fb, x1, rb1, rb2 = found
    b = Shape.from_rates([0, x1, right], fb, [rb1, rb2], gx, gy)
    d0, d1, d2 = b.cost(0) - a.cost(0), b.cost(x1) - a.cost(x1), b.cost(right) - a.cost(right)
    if not (d0 < 0 < d2 and ((late and d1 < 0) or (not late and d1 > 0))) or b.cost(xc) != cstar:
        return None
    if not K_MIN <= max(*a.ys, *b.ys) // gy <= K_MAX:
        return None
    us = [x for x in range(gx, right, gx) if x not in (x1, xc) and b.cost(x).denominator == 1 and b.cost(x) % gy == 0]
    if not us:
        return None
    check(a.fee % gy == 0 and b.fee % gy == 0, "a fixed charge is off the gridlines")
    return a, b, xc, rng.choice(us)


def _exam(rng):
    ctx = _pick_context(rng)
    for _ in attempts(60, "an exam draw with three usable wrong options per part"):
        a, b, xc, u = _draw_exam(rng, ctx)
        top = max(*a.ys, *b.ys)
        check(a.cost(xc) == b.cost(xc), "the plans do not agree at the crossing")
        try:
            parts = _exam_parts(rng, ctx, _slots(ctx), a, b, xc, u, top)
        except VerificationError as exc:
            if "usable distractors" not in str(exc):
                raise
            continue
        return dict(text=_exam_stem(rng, ctx, a), parts=list(parts), figure=_figure_two(ctx, a, b))
    raise AssertionError("unreachable")


def _exam_parts(rng, ctx, slots, a, b, xc, u, top):
    gx, gy = a.gx, a.gy
    ra, rb1, rb2 = a.rates[0], b.rates[0], b.rates[1]
    cstar = a.cost(xc)
    bu = b.cost(u)
    hi = _d(top)
    cents = _d(gy)
    pa = part(
        phrase(rng, EXAM_A, u_text=_amount(ctx, u), **slots),
        _d(bu),
        2,
        1,
        [f"Read {tag_b(ctx)} at {_amount(ctx, u)}: {_m(bu)}."],
        [
            (_d(a.cost(u)), "pwg-reads-the-other-plan"),
            (_d(b.fee + rb1 * u), "pwg-first-rate-for-all-units"),
            (_d(b.cost(u + gx)), "pwg-reads-the-neighbouring-gridline"),
            (_d(b.cost(u - gx)), "pwg-reads-the-neighbouring-gridline"),
            (_d(b.ys[1]), "pwg-cost-at-the-breakpoint-for-the-cost"),
            (_d(bu - b.fee), "pwg-leaves-out-the-fixed-charge"),
        ],
        cents,
        hi,
    )
    pb = part(
        phrase(rng, EXAM_B, **slots),
        _d(ra),
        2,
        2,
        [
            f"{tag_a(ctx)} rises from {_m(a.ys[0])} to {_m(cstar)} over {xc} {ctx['units']}.",
            f"Rate = {_text(_d(ra), 2)}.",
        ],
        [
            (_d(rb1), "pwg-reads-the-other-plan"),
            (_d(rb2), "pwg-reads-the-other-plan"),
            (_d(cstar), "pwg-cost-at-the-end-for-rate"),
            (_d(a.fee), "pwg-fixed-charge-for-rate"),
            (_d(cstar - a.fee), "pwg-total-change-for-rate"),
            (_d(cstar) / xc, "pwg-average-rate-for-gradient"),
            (Fraction(xc), "pwg-run-for-rate"),
        ],
        Fraction(1, 20),
        hi,
    )
    pc = part(
        phrase(rng, EXAM_C, **slots),
        Fraction(xc),
        0,
        2,
        [f"The lines cross at {xc} {ctx['units']}, where both cost {_m(cstar)}."],
        [
            (_d(cstar), "pwg-crossing-cost-for-usage"),
            (Fraction(b.xs[1]), "pwg-kink-for-crossing"),
            (Fraction(b.right), "pwg-end-of-graph-for-breakpoint"),
            ((a.fee - b.fee) / (rb1 - ra) if rb1 > ra else None, "pwg-first-rate-for-all-units"),
            (Fraction(xc + gx), "pwg-reads-the-neighbouring-gridline"),
            (Fraction(xc - gx), "pwg-reads-the-neighbouring-gridline"),
        ],
        gx,
        Fraction(b.right),
    )
    pd = part(
        phrase(rng, EXAM_D, fa=_m(a.fee), **slots),
        _d(cstar),
        2,
        2,
        [f"Cost = {_m(a.fee)} + {_text(_d(ra), 2)} x {xc} = {_m(cstar)}."],
        [
            (_d(ra * xc), "pwg-leaves-out-the-fixed-charge"),
            (Fraction(xc), "pwg-crossing-usage-for-cost"),
            (_d(a.fee) + xc, "pwg-rate-left-out"),
            (_d(a.fee + ra), "pwg-rate-for-one-unit-only"),
            (_d(b.fee + rb1 * xc), "pwg-first-rate-for-all-units"),
            (_d(a.fee + ra * (xc + gx)), "pwg-reads-the-neighbouring-gridline"),
        ],
        cents,
        hi,
        follow=f"{float(_d(a.fee))} + part_b*part_c",
        sensible=(0, None),
    )
    return pa, pb, pc, pd


def _exam_stem(rng, ctx, a):
    return phrase(
        rng,
        (
            "{two} {tag} A has a fixed charge of ${fa} and charges the same rate for every {unit}.",
            "{two} {tag} A has a fixed charge of ${fa}, and its rate for each {unit} never changes.",
            "{two} {tag} A charges ${fa} plus a constant rate for each {unit}.",
        ),
        two=ctx["two"],
        tag=ctx["tag"],
        fa=_m(a.fee),
        unit=ctx["unit"],
    )


def tag_a(ctx):
    return f"{ctx['tag']} A"


def tag_b(ctx):
    return f"{ctx['tag']} B"


# ---------------------------------------------------------------- the generator

TIERS = {"easy": _easy, "medium": _medium, "hard": _hard}


@generator(
    TOPIC,
    level=13,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
@uses_rng
def pw1_piecewise_cost_graph(rng: random.Random, tier: str = "easy") -> dict:
    if tier == "exam":
        return _exam(rng)
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(200, "a draw with enough distinct wrong options"):
        try:
            q = TIERS[tier](rng)
        except _Retry:
            continue
        try:
            fields = number_fields(q["key"], q["dp"], q["wrong"], q["step"], q["hi"])
        except VerificationError as exc:  # too few usable wrong options for this draw: draw again
            if "usable distractors" not in str(exc):
                raise
            continue
        return dict(text=q["text"], latex=q["text"], working=q["working"], figure=q["fig"], **fields)
    raise AssertionError("unreachable")
