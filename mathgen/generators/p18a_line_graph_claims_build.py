"""Generates `f10-line-graph-change-over-time` questions (AC9M5ST02, Year 5): choose the correct claim about
change over time in a line graph, `answer: mcq`.

`f10-line-graph-change-over-time` (AC9M5ST02, Year 5). `answer: mcq`.

Scope: given a quantity's value at a labelled sequence of time points,
drawn as a line graph (a ``cartesian`` figure with the points joined by segments), and one stated claim
drawn from a fixed class (the value at a stated time point, the direction or
size of the change between two stated time points, or whether the value
rose, fell or stayed the same across every consecutive pair of time
points), classify the claim as supported or not supported by the data.

Every question draws the full sequence as a line graph on gridlines (every value sits on a gridline; the prompt
does not list the data), states one claim from
exactly one of the three fixed classes above, and asks whether the claim is
supported. The four options are always `Supported` plus three
`Not supported: <what the data actually show>` phrases in the same
grammatical shape as the claim itself, so no option is identifiable by its
own form, and exactly one is true by construction.

Construct-to-target: every draw first decides `want_supported`
(~30% True) and then builds data and claim to realise that outcome, so
`Supported` lands at 25-35% of draws per tier without post-hoc rejection.

Ladder (reasoning escalation, not size):
  easy:   reads a single named value directly off the sequence.
  medium: computes a directed difference between two named points -- the
          learner must subtract in the right order and get the sign right.
  hard:   checks a "rose/fell/stayed the same every <period>" claim against
          EVERY consecutive pair; when the claim is false, exactly one pair
          breaks the pattern and the "Not supported" options must name a
          specific pair, never a vague summary.
  exam:   two variants, both still one claim from the
          fixed class above, and both over a LONGER graph (seven or eight
          points): (a) the overall change from the first to the last
          named time point, over data that genuinely fluctuates in
          between, so the learner must read the two named points and
          ignore the interior wiggle; (b) the same every-consecutive-pair
          claim as hard, over the longer graph, where the one breaking
          pair is easy to pass over.
"""

from __future__ import annotations

import random
import re

from mathgen import figures
from mathgen.core import check, generator, latex_answer, phrase

TIERS = ("easy", "medium", "hard", "exam")

# ---------------------------------------------------------------------------
# Quantity / period banks -- variety for prompt-shape and rendered-text
# diversity. Ranges are chosen wide enough that the monotonic-series builder
# never needs to clamp (see `_monotonic_series`).
# ---------------------------------------------------------------------------

#   `periods` (optional): the recording intervals that fit the quantity; the default is every period kind.
#   `first`: the indefinite/generic form used once, to introduce the subject.
#   `later`: the definite form used in every claim and working line after
#     that first mention, so the noun the subject names (museum/company/
#     town/club/household) is never re-introduced with an indefinite
#     article. Quantities with no embedded common noun
#     ("the average daily temperature", "the amount of rainfall") use the
#     same phrase both times -- there is nothing to make definite.
_QUANTITIES = [
    dict(
        first="the number of visitors to a museum",
        later="the number of visitors to the museum",
        unit="people",
        lo=100,
        hi=1500,
        g=100,
        axis="Visitors",
    ),
    dict(
        first="a company's share price",
        later="the company's share price",
        unit="$",
        lo=5,
        hi=100,
        g=5,
        axis="Price ($)",
    ),
    dict(
        first="the average daily temperature",
        later="the average daily temperature",
        unit="°C",
        lo=16,
        hi=32,
        g=1,
        axis="Temperature (°C)",
        periods=("week",),
    ),
    dict(
        first="the population of a small town",
        later="the population of the town",
        unit="people",
        lo=1000,
        hi=10000,
        g=500,
        axis="Population",
        periods=("year",),
    ),
    dict(
        first="the amount of rainfall",
        later="the amount of rainfall",
        unit="mm",
        lo=10,
        hi=200,
        g=10,
        axis="Rainfall (mm)",
    ),
    dict(
        first="the number of active members of a club",
        later="the number of active members of the club",
        unit="members",
        lo=20,
        hi=200,
        g=10,
        axis="Members",
    ),
    dict(
        first="a household's weekly power bill",
        later="the household's weekly power bill",
        unit="$",
        lo=20,
        hi=105,
        g=5,
        axis="Bill ($)",
        periods=("week",),
    ),
]

_PERIOD_KINDS = ("year", "week")


def _labels(rng: random.Random, period_kind: str, n: int) -> tuple[list[str], str]:
    if period_kind == "year":
        start = rng.randint(2005, 2026 - n)
        return [str(start + i) for i in range(n)], "year"
    return [f"Week {i + 1}" for i in range(n)], "week"


def _cap(s: str) -> str:
    return s[0].upper() + s[1:]


def _fmt_value(v: int, unit: str) -> str:
    if unit == "$":
        return f"${v}"
    if unit == "°C":
        return f"{v}°C"
    return f"{v} {unit}"


def _fmt_magnitude(v: int, unit: str) -> str:
    """Like `_fmt_value` but for an unsigned magnitude (a delta)."""
    return _fmt_value(v, unit)


def _data_text(labels: list[str], values: list[int], unit: str) -> str:
    return ", ".join(f"({lbl}: {_fmt_value(v, unit)})" for lbl, v in zip(labels, values))


# The data are drawn as a line graph, never listed in the prompt.
INTRO_PATTERNS = [
    "{quantity_cap} was recorded for each {period}. The line graph shows the results.",
    "The line graph shows {quantity} for each {period}.",
    "{quantity_cap} was recorded for each {period} and drawn as a line graph.",
]


def _intro(rng: random.Random, q: dict, period: str) -> str:
    return phrase(rng, INTRO_PATTERNS, quantity=q["first"], quantity_cap=_cap(q["first"]), period=period)


def _draw(rng: random.Random, q: dict) -> int:
    """A value on a gridline: a whole number of gridline steps inside the quantity's range."""
    g = q["g"]
    return g * rng.randint(q["lo"] // g, q["hi"] // g)


def _xs(labels: list[str]) -> list[int]:
    return [int(label) if label.isdigit() else int(label.split()[-1]) for label in labels]


def _line_figure(q: dict, labels: list[str], values: list[int], period: str) -> dict:
    """The line graph: the points joined in time order, every value on a gridline."""
    xs = _xs(labels)
    g = q["g"]
    top = (max(values) // g + 1) * g
    points = [{"at": [x, v]} for x, v in zip(xs, values)]
    segments = [{"from": [xs[i], values[i]], "to": [xs[i + 1], values[i + 1]]} for i in range(len(xs) - 1)]
    return figures.cartesian(
        x_range=(xs[0] - 1, xs[-1] + 1),
        y_range=(0, top),
        grid=1,
        x_grid=1,
        y_grid=g,
        equal_scale=False,
        points=points,
        segments=segments,
        x_label=period.capitalize(),
        y_label=q["axis"],
        alt="a line graph",
    )


def _finish(text_body: str, claim: str, key: str, distractors: list[str], working: list[str], figure: dict) -> dict:
    # Plain-text mcq options (not symbolic expressions), so distinctness is
    # checked as a set of strings, not via `check_pairwise_distinct` (which
    # sympifies its arguments and would choke on "Not supported: ...").
    check(len(distractors) == 3, "line-graph-claims: exactly 3 distractors required")
    check(len({key, *distractors}) == 4, "line-graph-claims: options must be pairwise distinct")
    text = f'{text_body} A claim is made: "{claim}" Is this claim supported by the data?'
    return dict(
        text=text,
        answer=key,
        answer_text=key,
        answer_latex=latex_answer(key, plain=True),
        working=working,
        distractors=distractors,
        figure=figure,
    )


# ===========================================================================
# Claim class 1: the value at a stated time point.
# ===========================================================================


def _value_claim(rng: random.Random, q: dict, labels: list[str], period: str) -> dict:
    n = len(labels)
    g = q["g"]
    values = [_draw(rng, q) for _ in range(n)]
    idx = rng.randrange(n)
    actual = values[idx]
    want_supported = rng.random() < 0.30

    pool = sorted({v for v in values if v != actual})
    tries = 0
    while len(pool) < 3 and tries < 50:
        tries += 1
        span = max(3, (q["hi"] - q["lo"]) // (10 * g))
        cand = actual + rng.choice([-1, 1]) * g * rng.randint(1, span)
        if cand != actual and cand not in pool and q["lo"] - span * g <= cand <= q["hi"] + span * g:
            pool.append(cand)
    rng.shuffle(pool)

    if want_supported:
        claimed = actual
        key = "Supported"
        distractors = [f"Not supported: it was actually {_fmt_value(v, q['unit'])}" for v in pool[:3]]
    else:
        false_claim = pool[0]
        decoys = pool[1:3]
        tries = 0
        while len(decoys) < 2 and tries < 50:
            tries += 1
            cand = actual + rng.choice([-1, 1]) * g * rng.randint(1, 6)
            if cand not in decoys and cand != actual and cand != false_claim:
                decoys.append(cand)
        claimed = false_claim
        key = f"Not supported: it was actually {_fmt_value(actual, q['unit'])}"
        distractors = ["Supported"] + [f"Not supported: it was actually {_fmt_value(v, q['unit'])}" for v in decoys]

    # Independent check: recompute supported/not-supported from the raw
    # data and claim, freshly, and confirm it matches the assigned key.
    recomputed_supported = claimed == actual
    check(
        recomputed_supported == (key == "Supported"),
        "line-graph-claims value-claim: key does not match recomputed data",
    )

    claim = f"In {labels[idx]}, {q['later']} was {_fmt_value(claimed, q['unit'])}."
    text_body = _intro(rng, q, period)
    working = [
        f"Data: {_data_text(labels, values, q['unit'])}.",
        f"The actual value in {labels[idx]} is {_fmt_value(actual, q['unit'])}; "
        f"the claim states {_fmt_value(claimed, q['unit'])}.",
        f"So the claim is {'supported' if recomputed_supported else 'not supported'}.",
    ]
    return _finish(text_body, claim, key, distractors, working, _line_figure(q, labels, values, period))


# ===========================================================================
# Claim class 2: direction/size of change between two stated time points.
# ===========================================================================


def _delta_action(delta: int, unit: str) -> str:
    """The bare verb phrase for a change, with no subject pronoun -- used
    directly after a stated subject ("{quantity} {action}")."""
    if delta == 0:
        return "stayed the same"
    direction = "rose" if delta > 0 else "fell"
    return f"{direction} by {_fmt_magnitude(abs(delta), unit)}"


def _delta_fact(delta: int, unit: str) -> str:
    """The same change, with its own subject pronoun -- used standalone in
    an option/claim that names no other subject ("it rose by $5")."""
    return f"it {_delta_action(delta, unit)}"


def _decoy_deltas(rng: random.Random, actual_delta: int, count: int, g: int, bound: int | None = None) -> list[int]:
    candidates: set[int] = {0, -actual_delta}
    for d in (3, 5, 7, 10, 15, 20, -3, -5, -7, -10, -15, -20):
        candidates.add(actual_delta + d * g)
        candidates.add(actual_delta - d * g)
    candidates.discard(actual_delta)
    if bound is not None:
        # A change bigger than the quantity's whole range could never be drawn; keep to believable decoys when enough remain.
        near = {c for c in candidates if abs(c) <= bound}
        if len(near) >= count:
            candidates = near
    ordered = sorted(candidates)
    rng.shuffle(ordered)
    chosen = ordered[:count]
    tries = 0
    while len(chosen) < count and tries < 50:
        tries += 1
        cand = actual_delta + g * rng.randint(-40, 40)
        if cand != actual_delta and cand not in chosen:
            chosen.append(cand)
    return chosen


def _change_claim(
    rng: random.Random,
    q: dict,
    labels: list[str],
    period: str,
    i: int,
    j: int,
    overall: bool = False,
) -> dict:
    n = len(labels)
    values = [_draw(rng, q) for _ in range(n)]
    if overall:
        # exam variant A: the claim spans the whole sequence, so the data
        # must genuinely fluctuate in between -- otherwise the "every
        # consecutive pair" strategy from hard tier would also answer it,
        # and the tier would not be a new decision.
        tries = 0
        while tries < 60:
            tries += 1
            diffs = [values[k + 1] - values[k] for k in range(n - 1)]
            if any(d > 0 for d in diffs) and any(d < 0 for d in diffs):
                break
            values = [_draw(rng, q) for _ in range(n)]
        else:
            raise AssertionError("line-graph-claims: could not build a fluctuating exam series")

    actual_delta = values[j] - values[i]
    want_supported = rng.random() < 0.30
    true_fact = _delta_fact(actual_delta, q["unit"])

    if want_supported:
        claim_delta = actual_delta
        key = "Supported"
        decoy_deltas = _decoy_deltas(rng, actual_delta, 3, q["g"], q["hi"] - q["lo"])
        distractors = [f"Not supported: {_delta_fact(d, q['unit'])}" for d in decoy_deltas]
    else:
        decoy_deltas = _decoy_deltas(rng, actual_delta, 3, q["g"], q["hi"] - q["lo"])
        claim_delta = decoy_deltas[0]
        key = f"Not supported: {true_fact}"
        distractors = ["Supported"] + [f"Not supported: {_delta_fact(d, q['unit'])}" for d in decoy_deltas[1:3]]

    recomputed_supported = claim_delta == actual_delta
    check(
        recomputed_supported == (key == "Supported"),
        "line-graph-claims change-claim: key does not match recomputed delta",
    )

    claim_phrase = _delta_action(claim_delta, q["unit"])
    if overall:
        claim = (
            f"Overall, from {labels[i]} to {labels[j]}, {q['later']} {claim_phrase}, even "
            f"though it did not move the same way every {period} in between."
        )
    else:
        claim = f"Between {labels[i]} and {labels[j]}, {q['later']} {claim_phrase}."
    text_body = _intro(rng, q, period)
    working = [
        f"Data, in time order: {_data_text(labels, values, q['unit'])}.",
        f"Actual change from {labels[i]} to {labels[j]}: "
        f"{_fmt_value(values[j], q['unit'])} - {_fmt_value(values[i], q['unit'])} "
        f"-> {true_fact}.",
        f"The claim says {claim_phrase}, so the claim is {'supported' if recomputed_supported else 'not supported'}.",
    ]
    return _finish(text_body, claim, key, distractors, working, _line_figure(q, labels, values, period))


def _medium_change_claim(rng: random.Random, q: dict, labels: list[str], period: str) -> dict:
    n = len(labels)
    i, j = sorted(rng.sample(range(n), 2))
    return _change_claim(rng, q, labels, period, i, j, overall=False)


def _exam_overall_change_claim(rng: random.Random, q: dict, labels: list[str], period: str) -> dict:
    n = len(labels)
    return _change_claim(rng, q, labels, period, 0, n - 1, overall=True)


# ===========================================================================
# Claim class 3: rose/fell/stayed the same across every consecutive pair.
# ===========================================================================


def _diff_word(diff: int) -> str:
    if diff > 0:
        return "rose"
    if diff < 0:
        return "fell"
    return "same"


def _pair_phrase(word: str, a: str, b: str) -> str:
    if word == "same":
        return f"it stayed the same between {a} and {b}"
    return f"it {word} between {a} and {b}"


def _monotonic_series(rng: random.Random, q: dict, n: int, want_supported: bool):
    g = q["g"]
    lo, hi = q["lo"] // g, q["hi"] // g  # work in whole gridline steps, scale back at the end
    scale = max(1, (hi - lo) // (4 * n))
    margin = n * scale + scale
    if hi - lo <= 2 * margin:
        scale = 1
        margin = n + 1
        if hi - lo <= 2 * margin:
            margin = max(0, (hi - lo) // 2 - 1)
    start_lo = lo + margin
    start_hi = hi - margin
    if start_lo > start_hi:
        start_lo, start_hi = lo, hi
    direction = rng.choice(["rose", "fell", "same"])
    values = [rng.randint(start_lo, start_hi)]
    diffs: list[int] = []
    break_idx = None if want_supported else rng.randrange(n - 1)
    for i in range(n - 1):
        breaking = (not want_supported) and i == break_idx
        if breaking:
            if direction == "rose":
                d = -rng.randint(1, scale)
            elif direction == "fell":
                d = rng.randint(1, scale)
            else:
                d = rng.choice([-1, 1]) * rng.randint(1, scale)
        else:
            if direction == "rose":
                d = rng.randint(1, scale)
            elif direction == "fell":
                d = -rng.randint(1, scale)
            else:
                d = 0
        diffs.append(d)
        values.append(values[-1] + d)
    return [v * g for v in values], [d * g for d in diffs], direction, break_idx


def _false_option_for_pair(rng: random.Random, diffs: list[int], idx: int, labels: list[str]) -> str:
    actual_word = _diff_word(diffs[idx])
    false_word = rng.choice([w for w in ("rose", "fell", "same") if w != actual_word])
    return f"Not supported: {_pair_phrase(false_word, labels[idx], labels[idx + 1])}"


def _monotonic_claim(rng: random.Random, q: dict, labels: list[str], period: str) -> dict:
    n = len(labels)
    want_supported = rng.random() < 0.30
    values, diffs, direction, break_idx = _monotonic_series(rng, q, n, want_supported)

    if direction == "rose":
        claim = f"{_cap(q['later'])} rose every {period} from {labels[0]} to {labels[-1]}."
    elif direction == "fell":
        claim = f"{_cap(q['later'])} fell every {period} from {labels[0]} to {labels[-1]}."
    else:
        claim = f"{_cap(q['later'])} stayed the same throughout, from {labels[0]} to {labels[-1]}."

    if want_supported:
        check(
            all(_diff_word(d) == direction for d in diffs),
            "line-graph-claims monotonic-claim: supported draw is not actually uniform",
        )
        key = "Supported"
        idxs = rng.sample(range(n - 1), min(3, n - 1))
        distractors = [_false_option_for_pair(rng, diffs, idx, labels) for idx in idxs]
    else:
        mismatches = [i for i, d in enumerate(diffs) if _diff_word(d) != direction]
        check(len(mismatches) == 1, "line-graph-claims monotonic-claim: expected exactly one break")
        check(mismatches[0] == break_idx, "line-graph-claims monotonic-claim: break at wrong index")
        key = f"Not supported: {_pair_phrase(_diff_word(diffs[break_idx]), labels[break_idx], labels[break_idx + 1])}"
        other_idxs = [i for i in range(n - 1) if i != break_idx]
        chosen = rng.sample(other_idxs, min(2, len(other_idxs)))
        distractors = ["Supported"] + [_false_option_for_pair(rng, diffs, idx, labels) for idx in chosen]

    while len(distractors) < 3:
        # Only reachable for a pathologically short series; n >= 4 always
        # supplies enough pairs above, so this is a defensive fallback.
        distractors.append(f"Not supported: it stayed the same between {labels[0]} and {labels[-1]}")

    data_text = _data_text(labels, values, q["unit"])
    text_body = _intro(rng, q, period)
    working = [
        f"Data, in time order: {data_text}.",
        f"Consecutive changes: " + ", ".join(_diff_word(d) for d in diffs) + ".",
        f"Claimed pattern: {direction}. "
        + (
            f"Every pair matches, so the claim is supported."
            if want_supported
            else f"Pair {labels[break_idx]}-{labels[break_idx + 1]} breaks the pattern "
            f"({_diff_word(diffs[break_idx])}), so the claim is not supported."
        ),
    ]
    return _finish(text_body, claim, key, distractors, working, _line_figure(q, labels, values, period))


# ===========================================================================
# Top-level generator
# ===========================================================================


@generator(
    "p18a_line_graph_claims",
    level=9,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p18a_line_graph_claims(rng: random.Random, tier: str = "easy") -> dict:
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}")

    q = rng.choice(_QUANTITIES)
    period_kind = rng.choice(q.get("periods", _PERIOD_KINDS))

    if tier == "easy":
        n = rng.randint(4, 5)
        labels, period = _labels(rng, period_kind, n)
        return _value_claim(rng, q, labels, period)
    if tier == "medium":
        n = rng.randint(4, 5)
        labels, period = _labels(rng, period_kind, n)
        return _medium_change_claim(rng, q, labels, period)
    if tier == "hard":
        n = rng.randint(4, 5)
        labels, period = _labels(rng, period_kind, n)
        return _monotonic_claim(rng, q, labels, period)
    # exam: a longer graph
    n = rng.randint(7, 8)
    labels, period = _labels(rng, period_kind, n)
    if rng.random() < 0.5:
        return _exam_overall_change_claim(rng, q, labels, period)
    return _monotonic_claim(rng, q, labels, period)


# ===========================================================================
# The figure reader: re-derive the key from the drawn line graph and the stated claim alone
# ===========================================================================

_QUOTED_CLAIM = re.compile(r'A claim is made: "([^"]+)"')
_VALUE_CLAIM = re.compile(r"^In (?P<t>.+?), .+ was (?P<v>.+)\.$")


def _label_x(label: str) -> int:
    return int(label) if label.isdigit() else int(label.split()[-1])


def _read_line_graph(question) -> str:
    fig = question.figure
    quantity = next(qq for qq in _QUANTITIES if qq["axis"] == fig["y_label"])
    unit = quantity["unit"]
    week = fig["x_label"] == "Week"
    data = {int(p["at"][0]): p["at"][1] for p in fig["points"]}
    claim = _QUOTED_CLAIM.search(question.text).group(1)

    def name(x: int) -> str:
        return f"Week {x}" if week else str(x)

    m = _VALUE_CLAIM.match(claim)
    if m:
        actual = data[_label_x(m.group("t"))]
        claimed = int(re.search(r"-?\d+", m.group("v")).group())
        return "Supported" if claimed == actual else f"Not supported: it was actually {_fmt_value(actual, unit)}"
    m = re.match(r"^(?:Between|Overall, from) (.+?)(?: and | to )(.+?), ", claim)
    if m:
        x1, x2 = _label_x(m.group(1)), _label_x(m.group(2))
        actual = data[x2] - data[x1]
        if "stayed the same" in claim:
            claimed = 0
        else:
            mag = int(re.search(r"(?:rose|fell) by \$?(-?\d+)", claim).group(1))
            claimed = mag if "rose by" in claim else -mag
        return "Supported" if claimed == actual else f"Not supported: {_delta_fact(actual, unit)}"
    m = re.search(
        r"(rose|fell) every \S+ from (.+?) to (.+?)\.$|(stayed the same) throughout, from (.+?) to (.+?)\.$", claim
    )
    check(m is not None, "unreadable claim")
    direction = "same" if m.group(4) else m.group(1)
    xs = sorted(data)
    diffs = [data[b] - data[a] for a, b in zip(xs, xs[1:])]
    breaks = [i for i, d in enumerate(diffs) if _diff_word(d) != direction]
    if not breaks:
        return "Supported"
    check(len(breaks) == 1, "more than one breaking pair")
    i = breaks[0]
    return f"Not supported: {_pair_phrase(_diff_word(diffs[i]), name(xs[i]), name(xs[i + 1]))}"


figures.register_reader("p18a_line_graph_claims", _read_line_graph)
