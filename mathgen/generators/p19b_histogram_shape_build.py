"""Generates `p19b_histogram_shape` questions: classify a histogram as symmetric, skewed left, skewed right or
bimodal.

Scope: given a histogram's
class-frequency data or a description of its shape, classify that shape as
symmetric, skewed left, skewed right, or bimodal. `answer: mcq`, four
plain-text options, ALL FOUR always offered (the answer-space
contract), so every draw has exactly 3 distractors by construction -- there
is no misconception-specific distractor selection to get wrong here, only
the key varies.

## The classifier (`_classify`), the one thing this generator and its oracle
each write independently

`_classify(freqs) -> label | None` returns a label only when exactly one of
the four shape rules fires; it is applied to every constructed table as an
in-generator check (construction draws the intended class, builds a
frequency list, and rejects/retries unless `_classify` confirms that exact
class -- construct-to-target). The independent oracle
 parses the rendered table back
out of `question.text` and applies its own, separately-written copy of the
same rule.

The four rules are disjoint by construction, not merely by assertion:

- `symmetric`: exactly one local peak, centred (`|2*peak_index -
  (n-1)| <= 1`), and every mirror pair `freqs[i]`/`freqs[n-1-i]` differs by
  at most 1. Construction builds the two "arms" as an increasing sequence of
  *actual* sampled integers and its near-mirror, with the two centre classes
  forced to differ by exactly 1-2 so an even-length table still has a single
  strict peak (a true tie would register as *zero* peaks under the strict
  `>` peak test below, not two).
- `skewed left` / `skewed right`: exactly one local peak, NOT centred (peak
  index <= 1, or >= n-2), with a strictly monotonic run from the peak to the
  far end (the "tail"). Right-skew has its peak near the START (index 0 or
  1) and a strictly *decreasing* tail toward the end (a long tail of
  ever-smaller frequencies stretching towards the larger class values);
  left-skew is built by taking a right-skew table and reversing it, which
  turns "peak near start, decreasing tail" into "peak near end, increasing
  head" -- a rule genuinely different in shape from symmetric (peak not
  centred) and mutually exclusive with the other skew direction by which end
  the peak sits at.
- `bimodal`: exactly TWO local peaks with exactly one valley class strictly
  between them, and that valley's frequency at most half the smaller peak's
  (a real, unambiguous dip, never a plateau). Because this branch requires
  `len(peaks) == 2`, it can never collide with the single-peak symmetric or
  skewed rules -- a table can only ever satisfy at most one branch.

A table with zero peaks (uniform/flat, explicitly rejected), more than two
peaks, two peaks with no valley cell between them, or a single off-centre
peak that fails the strict monotonic-tail test returns `None` and is
resampled, so no draw satisfies two rules and no draw is uniform.

## Convention pinned in every prompt

Every question -- at every tier -- states: "Skewed right (or positively
skewed) means the longer tail is on the right, towards larger values;
skewed left (or negatively skewed) means the longer tail is on the left,
towards smaller values." This removes the classic
skewed-right-vs-positively-skewed confusion by construction rather than by
hoping the learner already knows it.

## Tiers (each changes the DECISION, not the numbers)

The medium, hard and exam tiers draw a ``histogram`` figure (equal-width classes; every bar ends on a gridline
or the half line between two) and print no class-frequency table.

- **easy**: a word description of the shape (never using "symmetric",
  "skewed" or "bimodal" itself) -- e.g. "most values are small, with a few
  very large ones stretching out to the right". DECISION: match a verbal
  description to the vocabulary. (No figure: there is nothing to draw.)
- **medium**: one histogram of 7-9 classes. DECISION: read the peak and
  tail off the drawn bars.
- **hard**: the same kind of histogram, but every value is subtracted from a
  stated number (``M - x``) and the question asks for the shape of the NEW
  values. DECISION: classify the drawn shape, then reflect it: skewed left
  and skewed right swap, symmetric and bimodal do not -- a reasoning step
  neither easy nor medium requires. (This replaces the earlier "table rows
  printed out of class order" step, which a drawn histogram cannot have.)
- **exam**: two histograms drawn as two panels (two independently drawn
  classes, same variable and class width, different class counts so their
  ranges differ), each in class order; the prompt asks for the shape of
  whichever has the WIDER range (more classes) or the LARGER total count
  (construction guarantees the two differ on the stated criterion).
  DECISION: the learner must first decide which histogram the question is
  about, then classify it -- a comparison step no lower tier has, and still
  exactly one key.

## Not built

A hard-tier "bimodal decoy" (a skewed table whose tail has a small bump) is not built: a bump breaks
the strict monotonic-tail rule stated above for `skewed`, so such a table satisfies neither rule and
cannot be an unambiguous item. The out-of-order presentation is built instead.
"""

from __future__ import annotations

from .. import figures
from ..core import check, generator, phrase
from ..figure_kinds.data import histogram, histogram_panel

TOPIC = "p19b_histogram_shape"
_TIERS = ("easy", "medium", "hard", "exam")

_LABELS = ("symmetric", "skewed left", "skewed right", "bimodal")

_CONVENTION = (
    "Skewed right (or positively skewed) means the longer tail is on the right, towards "
    "larger values; skewed left (or negatively skewed) means the longer tail is on the "
    "left, towards smaller values."
)

_CONTEXTS = (
    dict(var="delivery times", unit="minutes", axis="Delivery time (minutes)", lo_lo=0, lo_hi=10, width=10),
    dict(var="plant heights", unit="cm", axis="Height (cm)", lo_lo=20, lo_hi=50, width=5),
    dict(var="test scores", unit="marks", axis="Test score (marks)", lo_lo=0, lo_hi=20, width=10),
    dict(var="waiting times at a clinic", unit="minutes", axis="Waiting time (minutes)", lo_lo=0, lo_hi=10, width=10),
    dict(var="lengths of bike rides", unit="km", axis="Length (km)", lo_lo=0, lo_hi=5, width=5),
    dict(var="daily screen time", unit="minutes", axis="Screen time (minutes)", lo_lo=0, lo_hi=40, width=15),
)


# ============================================================================
# The classifier -- construction's own copy; the oracle writes a second one.
# ============================================================================


def _local_peaks(freqs: list[int]) -> list[int]:
    n = len(freqs)
    peaks = []
    for i in range(n):
        left_ok = (i == 0) or (freqs[i] > freqs[i - 1])
        right_ok = (i == n - 1) or (freqs[i] > freqs[i + 1])
        if left_ok and right_ok:
            peaks.append(i)
    return peaks


def _classify(freqs: list[int]) -> str | None:
    n = len(freqs)
    if n < 5 or any(f <= 0 for f in freqs) or len(set(freqs)) == 1:
        return None
    peaks = _local_peaks(freqs)

    if len(peaks) == 1:
        p = peaks[0]
        mirror_ok = all(abs(freqs[i] - freqs[n - 1 - i]) <= 1 for i in range(n))
        centred = abs(2 * p - (n - 1)) <= 1
        if mirror_ok and centred:
            return "symmetric"
        if p <= 1 and n - 1 - p >= 3:
            tail = freqs[p:]
            if all(tail[i] > tail[i + 1] for i in range(len(tail) - 1)):
                if p == 0 or freqs[0] <= freqs[1]:
                    return "skewed right"
        if p >= n - 2 and p >= 3:
            head = freqs[: p + 1]
            if all(head[i] < head[i + 1] for i in range(len(head) - 1)):
                if p == n - 1 or freqs[n - 1] <= freqs[n - 2]:
                    return "skewed left"
        return None

    if len(peaks) == 2:
        p1, p2 = peaks
        if p2 - p1 < 2:
            return None
        valley = min(freqs[p1 + 1 : p2])
        smaller = min(freqs[p1], freqs[p2])
        if 2 * valley <= smaller:
            return "bimodal"
        return None

    return None


# ============================================================================
# Table construction (draw the CLASS first, then build data that realises it)
# ============================================================================


def _build_symmetric(rng, n: int) -> list[int]:
    half = n // 2
    # Gaps of >= 3 between consecutive "arm" values guarantee that shaving
    # 1-2 off the top value to break the centre tie (even n) still leaves a
    # strictly descending run into the rest of the mirrored arm -- a smaller
    # gap can let the shaved peak dip BELOW the second-highest arm value and
    # create a spurious second local peak there (caught empirically: gap=1
    # let [2,6,8,9] -> right=[7,8,6,2], an internal bump at the second slot).
    left = [rng.randint(2, 4)]
    for _ in range(half - 1):
        left.append(left[-1] + rng.randint(3, 5))
    if n % 2 == 1:
        centre = left[-1] + rng.randint(2, 5)
        freqs = left + [centre] + list(reversed(left))
    else:
        # d must be exactly 1: the mirror rule tolerates a difference of at
        # most 1 between paired classes, and this is the pair the "shave"
        # falls on (freqs[half-1] vs freqs[half]).
        right = [left[-1] - 1] + list(reversed(left[:-1]))
        freqs = left + right
    check(_classify(freqs) == "symmetric", "p19b_histogram_shape: symmetric construction misfired")
    return freqs


def _build_skewed(rng, n: int, direction: str) -> list[int]:
    peak_idx = rng.choice([0, 1])
    tail_len = n - 1 - peak_idx
    peak = tail_len + rng.randint(9, 16)
    tail = sorted(rng.sample(range(1, peak), tail_len), reverse=True)
    if peak_idx == 1:
        entry = rng.randint(2, peak - 2)
        freqs = [entry, peak] + tail
    else:
        freqs = [peak] + tail
    if direction == "left":
        freqs = list(reversed(freqs))
    check(_classify(freqs) == f"skewed {direction}", f"p19b_histogram_shape: skewed-{direction} construction misfired")
    return freqs


def _build_bimodal(rng, n: int) -> list[int]:
    left_len = rng.choice([2, 3])
    right_len = n - 1 - left_len
    peak1 = rng.randint(15, 22)
    hump1 = (sorted(rng.sample(range(3, peak1), left_len - 1)) + [peak1]) if left_len > 1 else [peak1]
    peak2 = rng.randint(15, 22)
    tail2 = sorted(rng.sample(range(3, peak2), right_len - 1), reverse=True) if right_len > 1 else []
    hump2 = [peak2] + tail2
    smaller = min(peak1, peak2)
    valley_val = rng.randint(1, max(1, smaller // 2 - 1))
    freqs = hump1 + [valley_val] + hump2
    check(_classify(freqs) == "bimodal", "p19b_histogram_shape: bimodal construction misfired")
    return freqs


_BUILDERS = {
    "symmetric": lambda rng, n: _build_symmetric(rng, n),
    "skewed left": lambda rng, n: _build_skewed(rng, n, "left"),
    "skewed right": lambda rng, n: _build_skewed(rng, n, "right"),
    "bimodal": lambda rng, n: _build_bimodal(rng, n),
}


def _draw_table(rng, label: str, n: int):
    ctx = rng.choice(_CONTEXTS)
    lo = rng.randint(ctx["lo_lo"], ctx["lo_hi"])
    width = ctx["width"]
    freqs = _BUILDERS[label](rng, n)
    bounds = [(lo + i * width, lo + (i + 1) * width - 1) for i in range(n)]
    return ctx, bounds, freqs


# ============================================================================
# Easy: verbal description (never the answer's own words)
# ============================================================================

_DESCRIPTIONS = {
    "symmetric": (
        "Most values cluster around the middle of the range, with roughly equal numbers tailing off on either side.",
        "The values pile up in the centre of the range and fall away by similar amounts on both sides.",
        "There is a single cluster of common values in the middle, with matching, similar "
        "tapers to the left and to the right.",
    ),
    "skewed right": (
        "Most values are small, with a few much larger ones stretching out to the right in a long tail.",
        "The bulk of the values sit near the low end of the range, tailing off gradually "
        "towards a handful of very large values.",
        "Values pile up near the smaller end of the range, with a thin, stretched-out tail "
        "reaching towards the larger values.",
    ),
    "skewed left": (
        "Most values are large, with a few much smaller ones stretching out to the left in a long tail.",
        "The bulk of the values sit near the high end of the range, tailing off gradually "
        "towards a handful of very small values.",
        "Values pile up near the larger end of the range, with a thin, stretched-out tail "
        "reaching towards the smaller values.",
    ),
    "bimodal": (
        "There are two separate clusters of common values, with a clear dip of much less common values between them.",
        "The values form two distinct peaks, separated by a stretch of far less common values in between.",
        "Two different groups of common values stand out, with an obvious gap of rarer values separating them.",
    ),
}

_EASY_TEMPLATES = (
    "{convention} A histogram of {var} ({unit}) has this shape: {description} What shape is this?",
    "{convention} The distribution of {var} ({unit}) is described as follows: {description} Classify this shape.",
    "{convention} A histogram shows {var} ({unit}) with this pattern: {description} What is "
    "the shape of this distribution?",
)


def _build_easy(rng):
    label = rng.choice(_LABELS)
    ctx = rng.choice(_CONTEXTS)
    description = rng.choice(_DESCRIPTIONS[label])
    text = phrase(
        rng, _EASY_TEMPLATES, convention=_CONVENTION, var=ctx["var"], unit=ctx["unit"], description=description
    )
    working = [f"The description matches the '{label}' pattern: no other class fits this wording."]
    distractors = [lab for lab in _LABELS if lab != label]
    return dict(
        text=text,
        latex=text,
        answer=label,
        answer_text=label,
        answer_latex=f"plain:{label}",
        distractors=distractors,
        working=working,
    )


# ============================================================================
# Figure helpers (medium, hard and exam draw the histogram)
# ============================================================================


def _panel(ctx, bounds, freqs, label=""):
    """A histogram panel: class ``k`` runs from its lower bound to the next class's lower bound."""
    width = ctx["width"]
    edges = [bounds[0][0] + i * width for i in range(len(freqs) + 1)]
    check(max(freqs) <= 30, "p19b_histogram_shape: frequency above the 30 the axis can show")
    axis_max = -(-max(freqs) // 2) * 2 + 2
    return histogram_panel(
        edges=edges, frequencies=freqs, freq_max=axis_max, freq_step=2, minor_lines=True, label=label
    )


def _figure(ctx, panels):
    return histogram(panels=panels, alt="a histogram" if len(panels) == 1 else "two histograms", x_label=ctx["axis"])


def _result(text, label, figure, working):
    distractors = [lab for lab in _LABELS if lab != label]
    return dict(
        text=text,
        latex=text,
        answer=label,
        answer_text=label,
        answer_latex=f"plain:{label}",
        distractors=distractors,
        working=working,
        figure=figure,
    )


# ============================================================================
# Medium: one histogram
# ============================================================================

_MEDIUM_TEMPLATES = (
    "{convention} A survey recorded {var} ({unit}), grouped into equal-width classes and drawn as the "
    "histogram. What is the shape of this distribution?",
    "{convention} The histogram shows {var} ({unit}) in equal-width classes. Classify the shape of this distribution.",
    "{convention} These are {var} ({unit}), grouped into equal-width classes and drawn from smallest to "
    "largest class. What shape does this histogram have?",
    "{convention} Look at the histogram of {var} ({unit}). Which shape does the distribution have?",
    "{convention} {var_cap} ({unit}) were collected and drawn in a histogram. Describe the shape of the "
    "distribution by choosing the best name for it.",
)


def _build_medium(rng):
    label = rng.choice(_LABELS)
    n = rng.choice([7, 8, 9])
    ctx, bounds, freqs = _draw_table(rng, label, n)
    text = phrase(
        rng,
        _MEDIUM_TEMPLATES,
        convention=_CONVENTION,
        var=ctx["var"],
        var_cap=ctx["var"].capitalize(),
        unit=ctx["unit"],
    )
    working = [
        f"Class frequencies, in order: {', '.join(str(f) for f in freqs)}.",
        f"This matches the '{label}' pattern.",
    ]
    return _result(text, label, _figure(ctx, [_panel(ctx, bounds, freqs)]), working)


# ============================================================================
# Hard: one histogram, then every value is subtracted from a number
# ============================================================================

_REFLECTED = {
    "symmetric": "symmetric",
    "bimodal": "bimodal",
    "skewed left": "skewed right",
    "skewed right": "skewed left",
}

_HARD_TEMPLATES = (
    "{convention} The histogram shows {var} ({unit}). Every value is subtracted from {top} to make a new data set "
    "(new value = {top} - old value). What is the shape of the histogram of the new values?",
    "{convention} A histogram of {var} ({unit}) is drawn. Each value x is replaced by {top} - x. What shape does the "
    "histogram of the new values have?",
    "{convention} The histogram shows {var} ({unit}). A new data set is made by taking each value away from {top}. "
    "Classify the shape of the histogram of the new data set.",
)


def _build_hard(rng):
    drawn = rng.choice(_LABELS)
    n = rng.choice([7, 8, 9])
    ctx, bounds, freqs = _draw_table(rng, drawn, n)
    label = _REFLECTED[drawn]
    top = bounds[0][0] + n * ctx["width"]
    text = phrase(rng, _HARD_TEMPLATES, convention=_CONVENTION, var=ctx["var"], unit=ctx["unit"], top=top)
    working = [
        f"The histogram's shape is '{drawn}'.",
        f"Subtracting each value from {top} reverses the order of the values, so the shape becomes '{label}'.",
    ]
    return _result(text, label, _figure(ctx, [_panel(ctx, bounds, freqs)]), working)


# ============================================================================
# Exam: two histograms (two panels), classify the one picked out by a stated comparison
# ============================================================================

_EXAM_TEMPLATES_RANGE = (
    "{convention} Two groups' {var} ({unit}) were recorded in equal-width classes of the same width and drawn as "
    "Group A and Group B. Classify the shape of whichever group's data has the WIDER range (the greater number "
    "of classes).",
    "{convention} {var_cap} ({unit}) were grouped into equal-width classes for two separate groups, Group A and "
    "Group B. State the shape of the group whose data spans the WIDER range (more classes).",
    "{convention} Group A and Group B each have a histogram of {var} ({unit}), with the same class width. Which "
    "shape does the group with the WIDER range (the greater number of classes) have?",
)
_EXAM_TEMPLATES_TOTAL = (
    "{convention} Two groups' {var} ({unit}) were recorded in equal-width classes of the same width and drawn as "
    "Group A and Group B. Classify the shape of whichever group has the LARGER total number of observations.",
    "{convention} {var_cap} ({unit}) were grouped into equal-width classes for two separate groups, Group A and "
    "Group B. State the shape of the group with the LARGER total number of observations.",
    "{convention} Group A and Group B each have a histogram of {var} ({unit}), with the same class width. Which "
    "shape does the group with the LARGER total number of observations have?",
)


def _build_exam(rng):
    """Draw two histograms with two INDEPENDENTLY drawn classes and, for the
    `range` criterion, two distinct class counts, then name whichever one the
    stated criterion (wider range, or larger total) actually decides as the
    winner -- the answer is always the WINNER's shape, never a fixed one of
    the two draws."""
    label_1 = rng.choice(_LABELS)
    label_2 = rng.choice(_LABELS)
    ctx = rng.choice(_CONTEXTS)
    lo = rng.randint(ctx["lo_lo"], ctx["lo_hi"])
    width = ctx["width"]
    criterion = rng.choice(("range", "total"))

    for _ in range(200):
        if criterion == "range":
            n_1, n_2 = rng.sample([7, 8, 9], 2)
        else:
            n_1 = rng.choice([7, 8, 9])
            n_2 = rng.choice([7, 8, 9])
        freqs_1 = _BUILDERS[label_1](rng, n_1)
        freqs_2 = _BUILDERS[label_2](rng, n_2)
        if criterion == "range" and n_1 != n_2:
            one_wins = n_1 > n_2
            break
        if criterion == "total" and sum(freqs_1) != sum(freqs_2):
            one_wins = sum(freqs_1) > sum(freqs_2)
            break
    else:
        check(False, "p19b_histogram_shape/exam: could not draw a decidable pair in 200 attempts")

    if one_wins:
        winner_label, winner_n, winner_freqs = label_1, n_1, freqs_1
        loser_label, loser_n, loser_freqs = label_2, n_2, freqs_2
    else:
        winner_label, winner_n, winner_freqs = label_2, n_2, freqs_2
        loser_label, loser_n, loser_freqs = label_1, n_1, freqs_1

    bounds_winner = [(lo + i * width, lo + (i + 1) * width - 1) for i in range(winner_n)]
    bounds_loser = [(lo + i * width, lo + (i + 1) * width - 1) for i in range(loser_n)]

    # Which letter (A/B) the winner prints as is itself randomised, so the
    # key is not predictable from "always Group A" or "always Group B".
    winner_is_a = rng.choice([True, False])
    panel_winner = _panel(ctx, bounds_winner, winner_freqs, "Group A" if winner_is_a else "Group B")
    panel_loser = _panel(ctx, bounds_loser, loser_freqs, "Group B" if winner_is_a else "Group A")
    panels = [panel_winner, panel_loser] if winner_is_a else [panel_loser, panel_winner]
    n_a = winner_n if winner_is_a else loser_n
    n_b = loser_n if winner_is_a else winner_n
    total_a = sum(winner_freqs if winner_is_a else loser_freqs)
    total_b = sum(loser_freqs if winner_is_a else winner_freqs)

    templates = _EXAM_TEMPLATES_RANGE if criterion == "range" else _EXAM_TEMPLATES_TOTAL
    text = phrase(
        rng,
        templates,
        convention=_CONVENTION,
        var=ctx["var"],
        var_cap=ctx["var"].capitalize(),
        unit=ctx["unit"],
    )

    if criterion == "range":
        working = [
            f"Group A spans {n_a} classes; Group B spans {n_b} classes.",
            f"The wider-range group's shape is '{winner_label}'.",
        ]
    else:
        working = [
            f"Group A total = {total_a}; Group B total = {total_b}.",
            f"The larger-total group's shape is '{winner_label}'.",
        ]
    return _result(text, winner_label, _figure(ctx, panels), working)


_TIER_BUILDERS = {
    "easy": _build_easy,
    "medium": _build_medium,
    "hard": _build_hard,
    "exam": _build_exam,
}


@generator(
    TOPIC,
    level=19,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p19b_histogram_shape(rng, tier: str = "easy"):
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    return _TIER_BUILDERS[tier](rng)


# ============================================================================
# The figure reader (medium, hard and exam draw a figure; easy does not)
# ============================================================================


def _read_shape(question) -> str:
    panels = question.figure["panels"]
    text = question.text
    if len(panels) == 2:
        by = {p["label"]: p["frequencies"] for p in panels}
        if "WIDER range" in text:
            chosen = max(by.values(), key=len)
        else:
            chosen = max(by.values(), key=sum)
        label = _classify(chosen)
    else:
        label = _classify(panels[0]["frequencies"])
        if (
            "subtracted" in text
            or "replaced by" in text
            or "take each value away" in text
            or "taking each value away" in text
        ):
            label = _REFLECTED[label]
    check(label is not None, "the drawn histogram has no unambiguous shape")
    return label


figures.register_reader(TOPIC, _read_shape)
