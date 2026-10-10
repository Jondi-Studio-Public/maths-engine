"""Generates `f10-line-graph-change-over-time` questions (AC9M5ST02, Year 5): choose the correct claim about
change over time in a line graph, `answer: mcq`.

`f10-line-graph-change-over-time` (AC9M5ST02, Year 5). `answer: mcq`.

Scope: given a quantity's value at a labelled sequence of time points,
presented as an explicit list of time-value pairs, and one stated claim
drawn from a fixed class (the value at a stated time point, the direction or
size of the change between two stated time points, or whether the value
rose, fell or stayed the same across every consecutive pair of time
points), classify the claim as supported or not supported by the data.

Every question shows the full labelled sequence as an explicit list of
(time: value) pairs (never a rendered figure), states one claim from
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
          fixed class above, and both over a LONGER sequence displayed
          OUT OF TIME ORDER (the readings were filed in the order recorded,
          not sorted by time), so reading straight down the list no longer
          answers the claim: (a) the overall change from the first to the
          last named time point, over data that genuinely fluctuates in
          between, so the learner must locate the two named points in the
          jumbled list and ignore the interior wiggle; (b) the same
          every-consecutive-pair claim as hard, but the learner must first
          reconstruct the true time order from the labels themselves (a
          week number or a year is self-ordering) before consecutive pairs
          even exist to check -- a decision hard tier never asks for,
          because hard's list is already in time order.
"""

from __future__ import annotations

import random

from mathgen.core import check, generator, latex_answer, phrase

TIERS = ("easy", "medium", "hard", "exam")

# ---------------------------------------------------------------------------
# Quantity / period banks -- variety for prompt-shape and rendered-text
# diversity. Ranges are chosen wide enough that the monotonic-series builder
# never needs to clamp (see `_monotonic_series`).
# ---------------------------------------------------------------------------

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
        lo=50,
        hi=2000,
    ),
    dict(first="a company's share price", later="the company's share price", unit="$", lo=5, hi=500),
    dict(first="the average daily temperature", later="the average daily temperature", unit="°C", lo=-5, hi=40),
    dict(first="the population of a small town", later="the population of the town", unit="people", lo=500, hi=20000),
    dict(first="the amount of rainfall", later="the amount of rainfall", unit="mm", lo=0, hi=200),
    dict(
        first="the number of active members of a club",
        later="the number of active members of the club",
        unit="members",
        lo=10,
        hi=300,
    ),
    dict(first="a household's monthly power bill", later="the household's monthly power bill", unit="$", lo=50, hi=400),
]

_PERIOD_KINDS = ("year", "week")


def _labels(rng: random.Random, period_kind: str, n: int) -> tuple[list[str], str]:
    if period_kind == "year":
        start = rng.randint(2005, 2022)
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


# Never say "the table below" -- the data are always an
# inline list, never a rendered table or figure. Every pattern below
# describes the list itself ("recorded", "here is", "as follows").
INTRO_PATTERNS = [
    "{quantity_cap} was recorded for each {period}: {data}.",
    "Here is {quantity} recorded for each {period}: {data}.",
    "{quantity_cap} for each {period} was recorded as follows: {data}.",
]

# Exam tier only: the data are shown in the order the
# readings were filed, not in time order, so the claim -- a change between
# two named points, or an every-consecutive-pair check -- cannot be
# resolved by reading straight down the list; the learner must place the
# points in time order themselves first.
ORDER_NOTE_PATTERNS = [
    "{quantity_cap} was recorded on several occasions. The readings, in the order they were "
    "filed rather than in time order, are: {data}.",
    "Here is {quantity}, one reading per {period}, listed in the order the readings were "
    "filed rather than in time order: {data}.",
]


def _intro(rng: random.Random, q: dict, period: str, data_text: str) -> str:
    return phrase(
        rng, INTRO_PATTERNS, quantity=q["first"], quantity_cap=_cap(q["first"]), period=period, data=data_text
    )


def _intro_shuffled(rng: random.Random, q: dict, period: str, data_text: str) -> str:
    return phrase(
        rng, ORDER_NOTE_PATTERNS, quantity=q["first"], quantity_cap=_cap(q["first"]), period=period, data=data_text
    )


def _shuffled_pairs(rng: random.Random, labels: list[str], values: list[int]):
    """Returns `(display_labels, display_values)`: the same (label, value)
    pairs as `labels`/`values`, permuted so the display order is never the
    time order (`n >= 4` always, so a non-identity permutation exists)."""
    order = list(range(len(labels)))
    identity = order[:]
    while order == identity:
        rng.shuffle(order)
    return [labels[i] for i in order], [values[i] for i in order]


def _finish(text_body: str, claim: str, key: str, distractors: list[str], working: list[str]) -> dict:
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
    )


# ===========================================================================
# Claim class 1: the value at a stated time point.
# ===========================================================================


def _value_claim(rng: random.Random, q: dict, labels: list[str], period: str) -> dict:
    n = len(labels)
    values = [rng.randint(q["lo"], q["hi"]) for _ in range(n)]
    idx = rng.randrange(n)
    actual = values[idx]
    want_supported = rng.random() < 0.30

    pool = sorted({v for v in values if v != actual})
    tries = 0
    while len(pool) < 3 and tries < 50:
        tries += 1
        span = max(3, (q["hi"] - q["lo"]) // 10)
        cand = actual + rng.choice([-1, 1]) * rng.randint(1, span)
        if cand != actual and cand not in pool and q["lo"] - span <= cand <= q["hi"] + span:
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
            cand = actual + rng.choice([-1, 1]) * rng.randint(1, 6)
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
    text_body = _intro(rng, q, period, _data_text(labels, values, q["unit"]))
    working = [
        f"Data: {_data_text(labels, values, q['unit'])}.",
        f"The actual value in {labels[idx]} is {_fmt_value(actual, q['unit'])}; "
        f"the claim states {_fmt_value(claimed, q['unit'])}.",
        f"So the claim is {'supported' if recomputed_supported else 'not supported'}.",
    ]
    return _finish(text_body, claim, key, distractors, working)


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


def _decoy_deltas(rng: random.Random, actual_delta: int, count: int) -> list[int]:
    candidates: set[int] = {0, -actual_delta}
    for d in (3, 5, 7, 10, 15, 20, -3, -5, -7, -10, -15, -20):
        candidates.add(actual_delta + d)
        candidates.add(actual_delta - d)
    candidates.discard(actual_delta)
    ordered = sorted(candidates)
    rng.shuffle(ordered)
    chosen = ordered[:count]
    tries = 0
    while len(chosen) < count and tries < 50:
        tries += 1
        cand = actual_delta + rng.randint(-40, 40)
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
    shuffle_display: bool = False,
) -> dict:
    n = len(labels)
    values = [rng.randint(q["lo"], q["hi"]) for _ in range(n)]
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
            values = [rng.randint(q["lo"], q["hi"]) for _ in range(n)]
        else:
            raise AssertionError("line-graph-claims: could not build a fluctuating exam series")

    actual_delta = values[j] - values[i]
    want_supported = rng.random() < 0.30
    true_fact = _delta_fact(actual_delta, q["unit"])

    if want_supported:
        claim_delta = actual_delta
        key = "Supported"
        decoy_deltas = _decoy_deltas(rng, actual_delta, 3)
        distractors = [f"Not supported: {_delta_fact(d, q['unit'])}" for d in decoy_deltas]
    else:
        decoy_deltas = _decoy_deltas(rng, actual_delta, 3)
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
    if shuffle_display:
        disp_labels, disp_values = _shuffled_pairs(rng, labels, values)
        text_body = _intro_shuffled(rng, q, period, _data_text(disp_labels, disp_values, q["unit"]))
    else:
        text_body = _intro(rng, q, period, _data_text(labels, values, q["unit"]))
    working = [
        f"Data, in time order: {_data_text(labels, values, q['unit'])}.",
        f"Actual change from {labels[i]} to {labels[j]}: "
        f"{_fmt_value(values[j], q['unit'])} - {_fmt_value(values[i], q['unit'])} "
        f"-> {true_fact}.",
        f"The claim says {claim_phrase}, so the claim is {'supported' if recomputed_supported else 'not supported'}.",
    ]
    return _finish(text_body, claim, key, distractors, working)


def _medium_change_claim(rng: random.Random, q: dict, labels: list[str], period: str) -> dict:
    n = len(labels)
    i, j = sorted(rng.sample(range(n), 2))
    return _change_claim(rng, q, labels, period, i, j, overall=False)


def _exam_overall_change_claim(rng: random.Random, q: dict, labels: list[str], period: str) -> dict:
    n = len(labels)
    return _change_claim(rng, q, labels, period, 0, n - 1, overall=True, shuffle_display=True)


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
    lo, hi = q["lo"], q["hi"]
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
    return values, diffs, direction, break_idx


def _false_option_for_pair(rng: random.Random, diffs: list[int], idx: int, labels: list[str]) -> str:
    actual_word = _diff_word(diffs[idx])
    false_word = rng.choice([w for w in ("rose", "fell", "same") if w != actual_word])
    return f"Not supported: {_pair_phrase(false_word, labels[idx], labels[idx + 1])}"


def _monotonic_claim(
    rng: random.Random, q: dict, labels: list[str], period: str, shuffle_display: bool = False
) -> dict:
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
    if shuffle_display:
        disp_labels, disp_values = _shuffled_pairs(rng, labels, values)
        text_body = _intro_shuffled(rng, q, period, _data_text(disp_labels, disp_values, q["unit"]))
        sort_step = [f"Sorted into time order first: {data_text}."]
    else:
        text_body = _intro(rng, q, period, data_text)
        sort_step = []
    working = sort_step + [
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
    return _finish(text_body, claim, key, distractors, working)


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
    period_kind = rng.choice(_PERIOD_KINDS)

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
    # exam
    n = rng.randint(5, 6)
    labels, period = _labels(rng, period_kind, n)
    if rng.random() < 0.5:
        return _exam_overall_change_claim(rng, q, labels, period)
    return _monotonic_claim(rng, q, labels, period, shuffle_display=True)
