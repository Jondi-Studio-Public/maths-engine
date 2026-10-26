"""`pw1_piecewise_cost_graph`: read a drawn piece-wise LINEAR cost graph (ACMGM047, the linear half).

``senior-read-a-piece-wise-linear-cost-graph``. A cost-against-usage graph made of two or three straight sections, with a
fixed charge where it meets the cost axis and a rate that changes at each kink. Every vertex and every whole-number
usage lies on a gridline (rates are whole multiples of the cost gridline step), so each answer is read, not estimated.
Nothing the keys need is printed: the figure carries axis labels and, on the exam tier, the two plan names.

Tiers (the ``tier`` parameter; unknown tiers raise ``ValueError``):

* easy: the cost at a stated usage (in either section), or the fixed charge, the cost at no usage.
* medium: the rate (gradient) of one section between two stated usages, or the usage at which the rate first changes.
* hard: the usage that gives a stated cost in a later section, or the cost beyond the right end of the drawing when
  the last section continues at its rate.
* exam: two plans on one graph. Plan A is a straight line (fixed charge stated), Plan B has two sections. Parts: (a) the
  cost of B at a stated usage, (b) the rate of A, (c) the usage at which the plans cost the same, (d) the common
  cost from A's fixed charge, rate (b) and usage (c), with follow-through.

The key is built from the chosen vertices and re-derived by evaluating the piece-wise function at every whole usage
(an independent route); the exam crossing is checked to be the only point where the two plans agree.
"""

from __future__ import annotations

import random
import re
from fractions import Fraction

from .. import figures
from ..core import VerificationError, attempts, check, generator, phrase
from ._exd_x_w12a_kit import split_parts
from ._r3d_kit import number_fields as _number_fields
from ._exd_x_pc_kit import count_part as _count_part
from ._spread_kit import NEAR, mixed, uses_rng

TOPIC = "pw1_piecewise_cost_graph"
ALT = "a graph of cost against usage made of straight sections, with a different rate in each section"
ALT_TWO = "a graph of cost against usage for two plans, one a single straight line and one with two sections"

CONTEXTS = [
    dict(
        intro="The graph shows the monthly cost of a mobile data plan.",
        two="The graph shows the monthly cost of two mobile data plans.",
        tag="Plan",
        things="plans",
        qty="data used",
        unit="gigabyte",
        units="gigabytes",
        cost="monthly cost",
        fixed="monthly fee",
        x_label="Data used (GB)",
        y_label="Monthly cost ($)",
    ),
    dict(
        intro="The graph shows the fare for a taxi trip.",
        two="The graph shows the fare charged by two taxi companies.",
        tag="Company",
        things="companies",
        qty="distance travelled",
        unit="kilometre",
        units="kilometres",
        cost="fare",
        fixed="flagfall",
        x_label="Distance (km)",
        y_label="Fare ($)",
    ),
    dict(
        intro="The graph shows a household's electricity bill for one quarter.",
        two="The graph shows the quarterly bill from two electricity retailers.",
        tag="Retailer",
        things="retailers",
        qty="electricity used",
        unit="kilowatt-hour",
        units="kilowatt-hours",
        cost="bill",
        fixed="supply charge",
        x_label="Electricity used (kWh)",
        y_label="Bill ($)",
    ),
    dict(
        intro="The graph shows the cost of hiring a trailer.",
        two="The graph shows the cost of hiring a trailer from two companies.",
        tag="Company",
        things="companies",
        qty="hire time",
        unit="day",
        units="days",
        cost="hire cost",
        fixed="booking fee",
        x_label="Hire time (days)",
        y_label="Cost ($)",
    ),
    dict(
        intro="The graph shows the charge for sending a parcel by courier.",
        two="The graph shows the charge made by two courier services for a parcel.",
        tag="Service",
        things="services",
        qty="parcel mass",
        unit="kilogram",
        units="kilograms",
        cost="charge",
        fixed="pick-up fee",
        x_label="Mass (kg)",
        y_label="Charge ($)",
    ),
]

K_MIN, K_MAX = 10, 26  # the tallest point is this many gridlines up


# ---------------------------------------------------------------- the shape


class Shape:
    """A piece-wise linear cost function with whole-number vertices. Costs and rates are in dollars."""

    def __init__(self, xs, fee, rates, g):
        self.xs = list(xs)  # vertex usages, xs[0] == 0
        self.rates = list(rates)  # dollars per unit in each section
        self.g = g
        ys = [fee]
        for i, r in enumerate(rates):
            ys.append(ys[-1] + r * (xs[i + 1] - xs[i]))
        self.ys = ys
        self.fee = fee

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


def _draw_shape(rng, nseg):
    for _ in attempts(2000, "a cost graph that fits the grid"):
        g = rng.choice((1, 2, 2, 5, 5))
        right = rng.randint(9, 12)
        cuts = sorted(rng.sample(range(2, right - 1), nseg - 1))
        xs = [0, *cuts, right]
        if any(b - a < 2 for a, b in zip(xs, xs[1:])):
            continue
        ks = [rng.randint(1, 4) for _ in range(nseg)]
        if any(a == b for a, b in zip(ks, ks[1:])):
            continue
        fee_k = rng.randint(1, 6)
        sh = Shape(xs, fee_k * g, [k * g for k in ks], g)
        top = max(sh.ys) // g
        if K_MIN <= top <= K_MAX:
            return sh
    raise AssertionError("unreachable")


def _check_shape(sh):
    """The function re-derived from its own vertices at every whole usage, and every one of those on a gridline."""
    for x in range(sh.right + 1):
        v = sh.cost(x)
        check(v % sh.g == 0, "a whole-usage point is off the cost gridlines")
        sec = sh.section(x)
        check(
            sh.ys[sec] + sh.rates[sec] * (x - sh.xs[sec]) == v and sh.xs[sec] <= x <= sh.xs[sec + 1],
            "cost differs between the two routes",
        )


# ---------------------------------------------------------------- the figure


def _plane(g, top, right):
    return dict(
        x_range=(0, right + 1),
        y_range=(0, top + 2 * g),
        grid=1,
        x_grid=1,
        y_grid=g,
        equal_scale=False,
    )


def _figure(ctx, sh):
    segments = [{"from": [sh.xs[i], sh.ys[i]], "to": [sh.xs[i + 1], sh.ys[i + 1]]} for i in range(len(sh.rates))]
    points = [{"at": [x, y]} for x, y in zip(sh.xs, sh.ys)]
    return figures.cartesian(
        alt=ALT,
        segments=segments,
        points=points,
        x_label=ctx["x_label"],
        y_label=ctx["y_label"],
        **_plane(sh.g, max(sh.ys), sh.right),
    )


def _figure_two(ctx, a, b):
    tag = ctx["tag"]
    segments = [{"from": [0, a.ys[0]], "to": [a.right, a.ys[-1]], "label": f"{tag} A"}]
    for i in range(len(b.rates)):
        seg = {"from": [b.xs[i], b.ys[i]], "to": [b.xs[i + 1], b.ys[i + 1]]}
        if i == 0:
            seg["label"] = f"{tag} B"
        segments.append(seg)
    points = [{"at": [x, y]} for x, y in zip(b.xs, b.ys)]
    return figures.cartesian(
        alt=ALT_TWO,
        segments=segments,
        points=points,
        x_label=ctx["x_label"],
        y_label=ctx["y_label"],
        **_plane(a.g, max(*a.ys, *b.ys), a.right),
    )


# ---------------------------------------------------------------- the readers


def _lines(fig):
    """Chains of (x0, y0, x1, y1) segments: the labelled first segment starts a plan, the next unlabelled ones follow."""
    return [
        (Fraction(s["from"][0]), Fraction(s["from"][1]), Fraction(s["to"][0]), Fraction(s["to"][1]))
        for s in fig["segments"]
    ]


def _eval(segs, x):
    """Value of a chain of segments at x, the last one extended beyond its end."""
    x = Fraction(x)
    for x0, y0, x1, y1 in segs:
        if x <= x1:
            return y0 + (y1 - y0) / (x1 - x0) * (x - x0)
    x0, y0, x1, y1 = segs[-1]
    return y1 + (y1 - y0) / (x1 - x0) * (x - x1)


def _fmt(v):
    v = Fraction(v)
    check(v.denominator == 1, "a read value is not whole")
    return str(int(v))


def read_single(question) -> str:
    """Re-derive the answer to a single-answer question from the figure and the numbers the prompt states."""
    segs = _lines(question.figure)
    text = question.text
    nums = [int(n) for n in re.findall(r"\d+", text)]
    if "continues" in text:
        return _fmt(_eval(segs, nums[0]))
    if "$" in text:  # the cost is stated: find the usage that gives it
        target = int(re.search(r"\$(\d+)", text).group(1))
        for x0, y0, x1, y1 in segs:
            if y0 <= target <= y1:
                return _fmt(x0 + (target - y0) / ((y1 - y0) / (x1 - x0)))
        raise ValueError("cost is not on the graph")
    if "different rate" in text:  # the first kink
        return _fmt(segs[0][2])
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

    qty_short = {
        "data used": "data",
        "distance travelled": "travel",
        "electricity used": "electricity",
        "hire time": "hire",
        "parcel mass": "parcel mass",
    }[ctx["qty"]]
    return dict(
        intro=ctx["intro"],
        cost=ctx["cost"],
        qty=ctx["qty"],
        qty_short=qty_short,
        unit=ctx["unit"],
        units=ctx["units"],
        fixed=ctx["fixed"],
        tag=ctx["tag"],
        things=ctx["things"],
        **extra,
    )


# ---------------------------------------------------------------- single-answer tiers


def _interior(rng, sh, section=None):
    """A whole usage strictly inside a section (not at a vertex), optionally in a given section."""
    pool = [x for x in range(1, sh.right) if x not in sh.xs and (section is None or sh.section(x) == section)]
    return rng.choice(pool)


def _int_or_none(v):
    v = Fraction(v)
    return int(v) if v.denominator == 1 and v > 0 else None


def _easy(rng):
    ctx = rng.choice(CONTEXTS)
    sh = _draw_shape(rng, rng.choice((2, 2, 3)))
    _check_shape(sh)
    top = max(sh.ys)
    if rng.random() < 0.25:  # the fixed charge
        key = sh.fee
        wrong = [
            (sh.rates[0], "pwg-reads-the-rate-for-the-cost"),
            (sh.ys[1], "pwg-cost-at-the-breakpoint-for-the-start"),
            (sh.cost(1), "pwg-reads-the-neighbouring-gridline"),
            (sh.xs[1], "pwg-breakpoint-for-the-cost"),
        ]
        text = phrase(rng, FEE_ASK, **_slots(ctx))
        working = [f"The graph meets the cost axis at ${key}, so the {ctx['fixed']} is {key}."]
        check(sh.cost(0) == key, "fixed charge differs between routes")
    else:
        section = rng.choice(range(len(sh.rates)))
        u = _interior(rng, sh, section)
        key = sh.cost(u)
        sec = sh.section(u)
        wrong = [
            (key - sh.fee, "pwg-leaves-out-the-fixed-charge"),
            (sh.fee + sh.rates[0] * u, "pwg-first-rate-for-all-units"),
            (sh.cost(u + 1), "pwg-reads-the-neighbouring-gridline"),
            (sh.cost(u - 1), "pwg-reads-the-neighbouring-gridline"),
            (u, "pwg-reads-the-usage-for-the-cost"),
            (sh.rates[sec], "pwg-reads-the-rate-for-the-cost"),
            (sh.ys[sec], "pwg-cost-at-the-breakpoint-for-the-cost"),
        ]
        text = phrase(rng, READ_ASK, u_text=_amount(ctx, u), **_slots(ctx))
        working = [f"Go up from {_amount(ctx, u)} to the graph and across to the cost axis: {key}."]
    fig = _figure(ctx, sh)
    return text, key, wrong, working, fig, top, sh.g


def _medium(rng):
    ctx = rng.choice(CONTEXTS)
    sh = _draw_shape(rng, rng.choice((2, 3, 3)))
    _check_shape(sh)
    top = max(sh.ys)
    if rng.random() < 0.5:
        s = rng.randrange(len(sh.rates))
        a, b = sh.xs[s], sh.xs[s + 1]
        key = sh.rates[s]
        check(Fraction(sh.cost(b) - sh.cost(a), b - a) == key, "gradient differs between routes")
        others = [r for i, r in enumerate(sh.rates) if i != s]
        wrong = [
            (others[0], "pwg-other-section-rate"),
            (sh.cost(b) - sh.cost(a), "pwg-total-change-for-rate"),
            (sh.cost(b), "pwg-cost-at-the-end-for-rate"),
            (_int_or_none(Fraction(sh.cost(b), b)), "pwg-average-rate-for-gradient"),
            (b - a, "pwg-run-for-rate"),
            (sh.fee, "pwg-fixed-charge-for-rate"),
            (others[-1], "pwg-other-section-rate"),
        ]
        text = phrase(rng, RATE_ASK, a=a, b=b, **_slots(ctx))
        working = [
            f"From {a} to {b} {ctx['units']} the cost rises from {sh.cost(a)} to {sh.cost(b)}.",
            f"Rate = {sh.cost(b) - sh.cost(a)} / {b - a} = {key} for each extra {ctx['unit']}.",
        ]
    else:
        key = sh.xs[1]
        check(sh.rates[0] != sh.rates[1], "no change at the first kink")
        wrong = [
            (sh.ys[1], "pwg-cost-at-the-change-for-usage"),
            (sh.xs[2], "pwg-other-breakpoint"),
            (sh.right, "pwg-end-of-graph-for-breakpoint"),
            (sh.rates[0], "pwg-reads-the-rate-for-the-cost"),
            (sh.fee, "pwg-fixed-charge-for-breakpoint"),
        ]
        text = phrase(rng, KINK_ASK, **_slots(ctx))
        working = [f"The graph bends first at {key} {ctx['units']}, where the rate changes."]
    fig = _figure(ctx, sh)
    return text, key, wrong, working, fig, top, sh.g


def _hard(rng):
    ctx = rng.choice(CONTEXTS)
    sh = _draw_shape(rng, rng.choice((2, 3, 3)))
    _check_shape(sh)
    top = max(sh.ys)
    if rng.random() < 0.6:
        section = rng.randrange(1, len(sh.rates))
        u = _interior(rng, sh, section)
        c = sh.cost(u)
        key = u
        found = [x for x in range(sh.right + 1) if sh.cost(x) == c]
        check(found == [u], "usage is not unique")
        r = sh.rates[section]
        wrong = [
            (c, "pwg-reads-the-cost-for-the-usage"),
            (_int_or_none(Fraction(c - sh.fee, sh.rates[0])), "pwg-first-rate-for-all-units"),
            (u - sh.xs[section], "pwg-forgets-the-breakpoint-offset"),
            (_int_or_none(Fraction(c, r)), "pwg-divides-cost-by-rate"),
            (_int_or_none(Fraction(c - sh.fee, r)), "pwg-divides-cost-by-rate"),
            (sh.xs[section], "pwg-breakpoint-for-the-usage"),
            (u + 1, "pwg-reads-the-neighbouring-gridline"),
        ]
        text = phrase(rng, USAGE_ASK, c=c, **_slots(ctx))
        working = [f"Go across from ${c} to the graph, then down to the usage axis: {_amount(ctx, u)}."]
        fig = _figure(ctx, sh)
        return text, key, wrong, working, fig, top, sh.g
    d = rng.randint(2, 6)
    v = sh.right + d
    key = sh.cost(v)
    check(sh.ys[-1] + sh.rates[-1] * d == key, "extension differs between routes")
    wrong = [
        (sh.ys[-1] + sh.rates[0] * d, "pwg-extends-at-the-wrong-rate"),
        (sh.rates[-1] * v, "pwg-extends-from-the-origin"),
        (sh.fee + sh.rates[-1] * v, "pwg-extends-from-the-fixed-charge"),
        (sh.ys[-1] + d, "pwg-extra-units-not-multiplied-by-rate"),
        (sh.ys[-1] + sh.rates[-1] * (d + 1), "pwg-extends-one-unit-too-far"),
        (sh.ys[-1] + sh.rates[-1] * (d - 1), "pwg-extends-one-unit-too-far"),
    ]
    text = phrase(rng, EXTEND_ASK, v=v, **_slots(ctx))
    working = [
        f"At {sh.right} {ctx['units']} the cost is {sh.ys[-1]} and the last section rises {sh.rates[-1]} for each extra {ctx['unit']}.",
        f"Cost at {v} {ctx['units']} = {sh.ys[-1]} + {d} x {sh.rates[-1]} = {key}.",
    ]
    fig = _figure(ctx, sh)
    return text, key, wrong, working, fig, None, sh.g


def _bounds(top, floor=1):
    return dict(lo=floor, hi=top)


# ---------------------------------------------------------------- the options


def _kind(v):
    """What a no-maths learner can see of a whole number: its length, and whether it is a multiple of 5 or of 10."""
    return (len(str(abs(int(v)))), v % 5 == 0, v % 10 == 0, v % 2)


def _options(key, wrong, g, lo, hi, unbounded=None):
    """The wrong options in a random rank order: named slips first, plain near misses one gridline step (g) apart.

    Named slips that look like the key (same length, same multiple of 5 and of 10) are kept when there are enough,
    so that no wrong option is the only one of its kind."""
    wrong = [(v, t) for v, t in wrong if v is not None]
    like = [(v, t) for v, t in wrong if _kind(v) == _kind(key)]
    if len(like) >= 2:
        wrong = like
    near = [
        (key + d * g, NEAR) for d in (-3, -2, -1, 1, 2, 3) if key + d * g >= lo and _kind(key + d * g) == _kind(key)
    ]
    near = near or [(key + d * g, NEAR) for d in (-3, -2, -1, 1, 2, 3) if key + d * g >= lo]
    return mixed(key, wrong, fillers=near, lo=lo, hi=hi, unbounded=unbounded)


def count_part(prompt, key, marks, working, wrong, **kw):
    wrong = [(v, t) for v, t in wrong if v is not None]
    like = [(v, t) for v, t in wrong if _kind(v) == _kind(key)]
    return _count_part(prompt, key, marks, working, like if len(like) >= 2 else wrong, **kw)


def number_fields(key, wrong, g, *, lo, hi, unbounded=None):
    return _number_fields(key, 0, _options(key, wrong, g, lo, hi, unbounded), nonneg=True)


# ---------------------------------------------------------------- the exam


def _draw_exam(rng):
    for _ in attempts(4000, "two plans that meet once on a graph that fits the grid"):
        g = rng.choice((1, 2, 2, 5, 5))
        right = rng.randint(9, 12)
        ka = rng.randint(1, 3)
        fa = rng.randint(3, 8)
        xs_cross = rng.randint(3, right - 3)
        late = rng.random() < 0.5
        if late:
            x1 = rng.randint(2, xs_cross - 1)
            kb1 = rng.randint(1, max(1, ka - 1)) if ka > 1 else None
            if kb1 is None:
                continue
            kb2 = rng.randint(ka + 1, ka + 3)
            fb = fa + ka * xs_cross - kb1 * x1 - kb2 * (xs_cross - x1)
        else:
            x1 = rng.randint(xs_cross + 2, right - 1)
            kb1 = rng.randint(ka + 1, ka + 3)
            kb2 = rng.randint(1, 5)
            fb = fa + (ka - kb1) * xs_cross
        if not (1 <= fb < fa) or kb1 == kb2:
            continue
        a = Shape([0, right], fa * g, [ka * g], g)
        b = Shape([0, x1, right], fb * g, [kb1 * g, kb2 * g], g)
        diffs = [b.cost(x) - a.cost(x) for x in range(right + 1)]
        zeros = [x for x, dv in zip(range(right + 1), diffs) if dv == 0]
        if zeros != [xs_cross]:
            continue
        if not all(dv < 0 for dv in diffs[:xs_cross]) or not all(dv > 0 for dv in diffs[xs_cross + 1 :]):
            continue
        top = max(*a.ys, *b.ys) // g
        if not K_MIN <= top <= K_MAX:
            continue
        _check_shape(a)
        _check_shape(b)
        return g, a, b, xs_cross
    raise AssertionError("unreachable")


def _exam(rng):
    ctx = rng.choice(CONTEXTS)
    for _ in attempts(60, "an exam draw with three usable wrong options per part"):
        g, a, b, xc = _draw_exam(rng)
        top = max(*a.ys, *b.ys)
        slots = _slots(ctx)
        u = rng.choice([x for x in range(1, b.right) if x not in (b.xs[1], xc)])
        cstar = a.cost(xc)
        check(cstar == b.cost(xc), "the plans do not agree at the crossing")
        try:
            pa, pb, pc, pd = _exam_parts(rng, ctx, slots, a, b, xc, u, top)
        except VerificationError as exc:
            if "usable distractors" not in str(exc):
                raise
            continue
        fig = _figure_two(ctx, a, b)
        return dict(text=_exam_stem(rng, ctx, a), parts=[pa, pb, pc, pd], figure=fig)
    raise AssertionError("unreachable")


def _exam_parts(rng, ctx, slots, a, b, xc, u, top):
    ra, rb1, rb2 = a.rates[0], b.rates[0], b.rates[1]
    cstar = a.cost(xc)
    pa = count_part(
        phrase(rng, EXAM_A, u_text=_amount(ctx, u), **slots),
        b.cost(u),
        1,
        [f"Read {tag_b(ctx)} at {_amount(ctx, u)}: {b.cost(u)}."],
        [
            (a.cost(u), "pwg-reads-the-other-plan"),
            (b.fee + rb1 * u, "pwg-first-rate-for-all-units"),
            (b.cost(u + 1), "pwg-reads-the-neighbouring-gridline"),
            (b.cost(u - 1), "pwg-reads-the-neighbouring-gridline"),
            (b.ys[1], "pwg-cost-at-the-breakpoint-for-the-cost"),
            (b.cost(u) - b.fee, "pwg-leaves-out-the-fixed-charge"),
        ],
        step=a.g,
        lo=1,
        hi=top,
    )
    pb = count_part(
        phrase(rng, EXAM_B, **slots),
        ra,
        2,
        [f"{tag_a(ctx)} rises from {a.ys[0]} to {a.ys[-1]} over {a.right} {ctx['units']}.", f"Rate = {ra}."],
        [
            (rb1, "pwg-reads-the-other-plan"),
            (rb2, "pwg-reads-the-other-plan"),
            (a.ys[-1], "pwg-cost-at-the-end-for-rate"),
            (a.fee, "pwg-fixed-charge-for-rate"),
            (a.ys[-1] - a.fee, "pwg-total-change-for-rate"),
            (_int_or_none(Fraction(a.ys[-1], a.right)), "pwg-average-rate-for-gradient"),
        ],
        step=a.g,
        lo=1,
        hi=top,
    )
    pc = count_part(
        phrase(rng, EXAM_C, **slots),
        xc,
        2,
        [f"The lines cross at {xc} {ctx['units']}, where both cost {cstar}."],
        [
            (cstar, "pwg-crossing-cost-for-usage"),
            (b.xs[1], "pwg-kink-for-crossing"),
            (b.right, "pwg-end-of-graph-for-breakpoint"),
            (_int_or_none(Fraction(a.fee - b.fee, rb1 - ra)) if rb1 > ra else None, "pwg-first-rate-for-all-units"),
            (xc + 1, "pwg-reads-the-neighbouring-gridline"),
            (xc - 1, "pwg-reads-the-neighbouring-gridline"),
        ],
        step=a.g,
        lo=1,
        hi=b.right,
    )
    pd = count_part(
        phrase(rng, EXAM_D, fa=a.fee, **slots),
        cstar,
        2,
        [f"Cost = {a.fee} + {ra} x {xc} = {cstar}."],
        [
            (ra * xc, "pwg-leaves-out-the-fixed-charge"),
            (xc, "pwg-crossing-usage-for-cost"),
            (a.fee + xc, "pwg-rate-left-out"),
            (a.fee + ra, "pwg-rate-for-one-unit-only"),
            (b.fee + rb1 * xc, "pwg-first-rate-for-all-units"),
            (a.fee + ra * (xc + 1), "pwg-reads-the-neighbouring-gridline"),
        ],
        step=a.g,
        lo=1,
        hi=top,
        follow=f"{a.fee} + part_b*part_c",
    )
    if None in (pa, pb, pc, pd):
        raise VerificationError("too few usable distractors")
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
        fa=a.fee,
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
    for _ in attempts(60, "a draw with enough distinct wrong options"):
        text, key, wrong, working, fig, top, g = TIERS[tier](rng)
        bounds = dict(lo=1, hi=top) if top is not None else dict(lo=1, hi=None, unbounded="open")
        try:
            fields = number_fields(key, wrong, g, **bounds)
        except VerificationError as exc:  # too few usable wrong options for this draw: draw again
            if "usable distractors" not in str(exc):
                raise
            continue
        return dict(text=text, latex=text, working=working, figure=fig, **fields)
    raise AssertionError("unreachable")
