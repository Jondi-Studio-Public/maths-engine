"""Generates `pv1b_step_graph` questions: read a step-function graph, deciding the step and the open or closed
endpoint at stated inputs.

One topic, ``pv1b_step_graph``, four tiers that each change the learner's
DECISION, not just the numbers:

- ``interior`` (easy): the stated input lies strictly inside a step; locate the
  step that covers it and read its height.
- ``boundary`` (medium): the input is exactly a step end; decide which dot
  (open or closed) applies. The closed side is drawn at random per seed, so
  "take the lower step" is not a learnable rule.
- ``two_reads`` (hard): two inputs, one usually on a boundary; read both and add.
- ``reverse`` (exam): given a charge, find the greatest input that still earns
  it, which is the closed right end of the step at that height (every step is
  ``(a, b]`` on this tier).

The figure is a ``figures.cartesian`` of horizontal ``segments`` with
``open_start``/``open_end`` set, one convention per figure. Nothing the key needs
is printed; the prompt states the boundary convention only generically ("use the
value shown by the filled circle"). The answer is an int, charges (or inputs on
the exam tier) are drawn so they spread widely.
"""

from __future__ import annotations

import re
from fractions import Fraction

from .. import figures
from ..core import check, check_pairwise_distinct, generator, phrase, rational_text

MAX_X = 20  # the widest window's right end (grid units)
ALT = "a step graph made of horizontal segments on a coordinate grid"

# name, intro, subject phrase, noun, quantity, units, verb, charge word, reverse
# wording, "greatest ..." question, axis labels.
CONTEXTS = [
    dict(
        intro="The graph shows the charge for sending a parcel by post, according to its mass.",
        subj="a parcel",
        noun="parcel",
        qty="mass",
        unit1="kg",
        unitn="kg",
        verb="weighs {amt}",
        cost="charge",
        paid="cost ${c} to send",
        ask="the greatest mass, in kilograms, the parcel could have had",
        x_label="Mass (kg)",
        y_label="Charge ($)",
    ),
    dict(
        intro="The graph shows the fee for parking in a car park, according to the time parked.",
        subj="a car",
        noun="car",
        qty="time parked",
        unit1="hour",
        unitn="hours",
        verb="is parked for {amt}",
        cost="fee",
        paid="was charged a fee of ${c}",
        ask="the greatest time, in hours, the car could have been parked for",
        x_label="Time parked (hours)",
        y_label="Fee ($)",
    ),
    dict(
        intro="The graph shows the fare for a bus trip, according to the distance travelled.",
        subj="a trip",
        noun="trip",
        qty="distance",
        unit1="km",
        unitn="km",
        verb="covers {amt}",
        cost="fare",
        paid="cost ${c}",
        ask="the greatest distance, in kilometres, the trip could have covered",
        x_label="Distance (km)",
        y_label="Fare ($)",
    ),
    dict(
        intro="The graph shows the freight charge for a crate, according to its mass.",
        subj="a crate",
        noun="crate",
        qty="mass",
        unit1="kg",
        unitn="kg",
        verb="weighs {amt}",
        cost="freight charge",
        paid="was charged ${c} to freight",
        ask="the greatest mass, in kilograms, the crate could have had",
        x_label="Mass (kg)",
        y_label="Freight charge ($)",
    ),
]


# ---------------------------------------------------------------- the figure


def _draw_widths(rng, k: int, wide: set[int]) -> list[int]:
    """k step widths (grid units), every index in ``wide`` at least 2, total <= MAX_X."""
    w = [rng.choice((1, 2, 2, 3, 3, 4)) for _ in range(k)]
    for i in wide:
        w[i] = max(w[i], 2)
    for _ in range(4 * MAX_X):
        if sum(w) <= MAX_X:
            break
        shrinkable = [i for i in range(k) if w[i] > (2 if i in wide else 1)]
        w[rng.choice(shrinkable)] -= 1
    return w


def _draw_heights(rng, k: int) -> list[int]:
    """k strictly increasing whole-dollar charges."""
    h = [rng.randint(1, 6)]
    for _ in range(k - 1):
        h.append(h[-1] + rng.randint(1, 4))
    return h


def _bounds(widths: list[int]) -> list[int]:
    b = [0]
    for w in widths:
        b.append(b[-1] + w)
    return b


def _figure(ctx: dict, bounds: list[int], heights: list[int], closed_right: bool) -> dict:
    """Steps (a, b] when ``closed_right`` (open left, closed right), else [a, b)."""
    segments = [
        {
            "from": [bounds[i], heights[i]],
            "to": [bounds[i + 1], heights[i]],
            "open_start": closed_right,
            "open_end": not closed_right,
        }
        for i in range(len(heights))
    ]
    return figures.cartesian(
        x_range=(0, bounds[-1] + 1),
        y_range=(0, heights[-1] + 1),
        alt=ALT,
        grid=1,
        equal_scale=False,
        segments=segments,
        x_label=ctx["x_label"],
        y_label=ctx["y_label"],
    )


# ---------------------------------------------------------------- the reader


def step_value(fig: dict, x: int) -> int:
    """The height the filled circle gives at ``x``: the segment that covers ``x``
    once each segment's open/closed ends are honoured."""
    hits = []
    for s in fig["segments"]:
        (a, h), (b, _) = s["from"], s["to"]
        lo_ok = x > a or (x == a and not s["open_start"])
        hi_ok = x < b or (x == b and not s["open_end"])
        if lo_ok and hi_ok:
            hits.append(h)
    check(len(hits) == 1, f"step graph gives {len(hits)} values at x = {x}")
    return hits[0]


def read_pv1b_step_graph(question) -> str:
    """Re-derive the answer from the figure plus the numbers the prompt states."""
    fig, text = question.figure, question.text
    charge = re.search(r"\$(\d+)", text)
    if charge:  # reverse read: the closed right end of the step at that height
        c = int(charge.group(1))
        ends = [s["to"][0] for s in fig["segments"] if s["from"][1] == c and not s["open_end"]]
        if len(ends) != 1:
            raise ValueError("no unique closed right end at that charge")
        return str(ends[0])
    xs = [int(n) for n in re.findall(r"\d+", text)]
    return str(sum(step_value(fig, x) for x in xs))


figures.register_reader("pv1b_step_graph", read_pv1b_step_graph)


# ---------------------------------------------------------------- wording


def _amt(ctx: dict, x: int) -> str:
    return f"{x} {ctx['unit1'] if x == 1 else ctx['unitn']}"


def _cap(s: str) -> str:
    return s[0].upper() + s[1:]


def _conv(ctx: dict) -> str:
    return f"Where the {ctx['qty']} is at the end of a step, use the value shown by the filled circle."


EASY_PATTERNS = (
    "{intro} {Subj} {verb}. {conv} What is the {cost}, in dollars?",
    "{intro} Use the graph to find the {cost}, in dollars, for {subj} that {verb}. {conv}",
    "{intro} According to the graph, how many dollars is the {cost} when the {qty} is {amt}? {conv}",
    "{conv} {intro} Find the {cost}, in dollars, when {subj} {verb}.",
)
MEDIUM_PATTERNS = (
    "{intro} {Subj} {verb_exact}. {conv} What is the {cost}, in dollars?",
    "{intro} Use the graph to find the {cost}, in dollars, for {subj} that {verb_exact}. {conv}",
    "{intro} According to the graph, how many dollars is the {cost} when the {qty} is exactly {amt}? {conv}",
    "{conv} {intro} Find the {cost}, in dollars, when {subj} {verb_exact}.",
)
HARD_PATTERNS = (
    "{intro} One {noun} {va} and another {noun} {vb}. {conv} What is the total {cost}, in dollars, for the two?",
    "{intro} Consider two {nouns}: the first {va} and the second {vb}. {conv} Find the combined {cost}, in dollars.",
    "{conv} {intro} Find the {cost} in dollars for {subj} that {va} together with {subj} that {vb}.",
    "{intro} {Subj} {va}, and a second {noun} {vb}. {conv} How many dollars is the {cost} for both?",
)
EXAM_PATTERNS = (
    "{intro} {Subj} {paid}. {conv} What is {ask}?",
    "{intro} {conv} Given that {subj} {paid}, find {ask}.",
    "{intro} Use the graph to work out {ask}, given that {subj} {paid}. {conv}",
    "{conv} {intro} {Subj} {paid}. Find {ask}.",
)


def _slots(ctx: dict) -> dict:
    return dict(
        intro=ctx["intro"],
        subj=ctx["subj"],
        Subj=_cap(ctx["subj"]),
        noun=ctx["noun"],
        nouns=ctx["noun"] + "s",
        qty=ctx["qty"],
        cost=ctx["cost"],
        conv=_conv(ctx),
        ask=ctx["ask"],
    )


def _verb(ctx: dict, x: int, exact: bool = False) -> str:
    v = ctx["verb"].format(amt=_amt(ctx, x))
    if exact:
        # put "exactly" before the amount: "weighs exactly 3 kg"
        v = v.replace(_amt(ctx, x), "exactly " + _amt(ctx, x))
    return v


# ---------------------------------------------------------------- distractors


def _finalize(key: int, candidates: list) -> list[str]:
    """Up to three distinct wrong values, in priority order, each different in VALUE
    from the key and from each other. ``None`` candidates are skipped."""
    chosen: list[Fraction] = []
    for c in candidates:
        if c is None:
            continue
        c = Fraction(c)
        if c == key or c in chosen or c < 0:
            continue
        chosen.append(c)
        if len(chosen) == 3:
            break
    check_pairwise_distinct(key, *chosen)
    return [rational_text(c) for c in chosen]


def _integer_or_none(v: Fraction):
    return int(v) if v.denominator == 1 else None


def _above(h: list[int], s: int):
    return h[s + 1] if s + 1 < len(h) else None


def _below(h: list[int], s: int):
    return h[s - 1] if s >= 1 else None


def _neighbour(h: list[int], s: int):
    """The step a learner slips to: the one above, else the one below."""
    return _above(h, s) if _above(h, s) is not None else _below(h, s)


# ---------------------------------------------------------------- the generator


@generator(
    "pv1b_step_graph",
    level=13,
    presets={
        "easy": {"variant": "interior"},
        "medium": {"variant": "boundary"},
        "hard": {"variant": "two_reads"},
        "exam": {"variant": "reverse"},
    },
)
def pv1b_step_graph(rng, variant: str = "interior") -> dict:
    ctx = rng.choice(CONTEXTS)
    k = rng.randint(4, 8)
    slots = _slots(ctx)
    closed_right = True if variant == "reverse" else rng.random() < 0.5

    if variant == "interior":
        j = rng.randrange(k)
        bounds = _bounds(_draw_widths(rng, k, {j}))
        heights = _draw_heights(rng, k)
        x = rng.randint(bounds[j] + 1, bounds[j + 1] - 1)
        key = heights[j]
        prev = heights[j - 1] if j >= 1 else 0
        interp = _integer_or_none(
            prev + Fraction(heights[j] - prev) * Fraction(x - bounds[j], bounds[j + 1] - bounds[j])
        )
        cands = [_above(heights, j), _below(heights, j), interp, x]
        fig = _figure(ctx, bounds, heights, closed_right)
        check(step_value(fig, x) == key, "interior read disagrees with the key")
        text = phrase(rng, EASY_PATTERNS, verb=_verb(ctx, x), amt=_amt(ctx, x), **slots)
        working = [f"The input {x} is strictly inside one step, so its height is the charge: {key}."]

    elif variant == "boundary":
        i = rng.randint(1, k - 1)  # boundary between step i-1 and step i
        bounds = _bounds(_draw_widths(rng, k, set()))
        heights = _draw_heights(rng, k)
        x = bounds[i]
        s = i - 1 if closed_right else i  # the step whose closed dot sits at x
        key = heights[s]
        mid = _integer_or_none(Fraction(heights[i - 1] + heights[i], 2))
        up, down = _above(heights, s), _below(heights, s)
        order = [up, down] if closed_right else [down, up]  # wrong side first
        cands = [order[0], mid, x, order[1]]
        fig = _figure(ctx, bounds, heights, closed_right)
        check(step_value(fig, x) == key, "boundary read disagrees with the key")
        text = phrase(rng, MEDIUM_PATTERNS, verb_exact=_verb(ctx, x, exact=True), amt=_amt(ctx, x), **slots)
        working = [f"{x} is the end of a step; the filled circle gives the charge: {key}."]

    elif variant == "two_reads":
        ja = rng.randrange(k)
        boundary_b = rng.random() < 0.7
        if boundary_b:
            ok = [i for i in range(1, k) if (i - 1 if closed_right else i) != ja]
            i = rng.choice(ok)
            sb = i - 1 if closed_right else i
            wide = {ja}
        else:
            jb = rng.choice([t for t in range(k) if t != ja])
            sb = jb
            wide = {ja, jb}
        bounds = _bounds(_draw_widths(rng, k, wide))
        heights = _draw_heights(rng, k)
        xa = rng.randint(bounds[ja] + 1, bounds[ja + 1] - 1)
        xb = bounds[i] if boundary_b else rng.randint(bounds[sb] + 1, bounds[sb + 1] - 1)
        fa, fb = heights[ja], heights[sb]
        key = fa + fb
        if boundary_b:  # the other dot at B
            other_b = heights[sb + 1] if closed_right else heights[sb - 1]
        else:
            other_b = _neighbour(heights, sb)
        na = _neighbour(heights, ja)
        fig = _figure(ctx, bounds, heights, closed_right)
        combined = None  # the learner adds the two inputs, then reads once
        if xa + xb < bounds[-1]:
            combined = step_value(fig, xa + xb)
        cands = [
            fa + other_b,
            None if na is None else na + fb,
            combined,
            xa + xb,
        ]
        check(step_value(fig, xa) + step_value(fig, xb) == key, "two-read total disagrees with the key")
        inputs = [xa, xb]
        if rng.random() < 0.5:
            inputs.reverse()
        text = phrase(rng, HARD_PATTERNS, va=_verb(ctx, inputs[0]), vb=_verb(ctx, inputs[1]), **slots)
        working = [f"The two charges are {fa} and {fb}, so the total is {key}."]

    elif variant == "reverse":
        i = rng.randrange(k)
        bounds = _bounds(_draw_widths(rng, k, set()))
        heights = _draw_heights(rng, k)
        c = heights[i]
        key = bounds[i + 1]
        mid = Fraction(bounds[i] + bounds[i + 1], 2)
        above_end = bounds[i + 2] if i + 1 < k else None
        cands = [bounds[i], above_end, c, _integer_or_none(mid)]
        fig = _figure(ctx, bounds, heights, True)
        ends = [s_["to"][0] for s_ in fig["segments"] if s_["from"][1] == c and not s_["open_end"]]
        check(ends == [key], "reverse read disagrees with the key")
        text = phrase(rng, EXAM_PATTERNS, paid=ctx["paid"].format(c=c), **slots)
        working = [f"The step at ${c} ends at {key}; its filled circle makes {key} the greatest."]
    else:
        raise ValueError(f"unknown variant {variant!r}")

    distractors = _finalize(key, cands)
    answer_text = rational_text(key)
    return {
        "text": text,
        "latex": text,
        "answer": key,
        "answer_text": answer_text,
        "answer_latex": answer_text,
        "distractors": distractors,
        "working": working,
        "figure": fig,
    }
