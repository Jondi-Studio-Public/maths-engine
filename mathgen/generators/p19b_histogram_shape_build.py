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

- **easy**: a word description of the shape (never using "symmetric",
  "skewed" or "bimodal" itself) -- e.g. "most values are small, with a few
  very large ones stretching out to the right". DECISION: match a verbal
  description to the vocabulary.
- **medium**: a class-frequency table, printed IN class order (7-9 classes,
  equal width, a stated variable and unit). DECISION: read the peak and
  tail directly off an already-ordered table.
- **hard**: the SAME kind of table, but the (interval, frequency) rows are
  printed in a shuffled, non-class order. DECISION: the learner must first
  reorder the rows by their class lower bound before the medium-tier
  judgement even becomes possible -- a decoding step neither easy nor
  medium requires.
- **exam**: two histograms (two independently drawn classes, same variable
  and class width, different class counts so their ranges differ), each
  printed in class order; the prompt asks for the shape of whichever
  histogram has the WIDER range (more classes) or the LARGER total count
  (construction guarantees the two histograms differ on the stated
  criterion). DECISION: the learner must first decide which histogram the
  question is about, then classify it -- a comparison step no lower tier
  has, and still exactly one key.

## Not built

A hard-tier "bimodal decoy" (a skewed table whose tail has a small bump) is not built: a bump breaks
the strict monotonic-tail rule stated above for `skewed`, so such a table satisfies neither rule and
cannot be an unambiguous item. The out-of-order presentation is built instead.
"""

from __future__ import annotations

from ..core import check, generator, phrase

TOPIC = "p19b_histogram_shape"
_TIERS = ("easy", "medium", "hard", "exam")

_LABELS = ("symmetric", "skewed left", "skewed right", "bimodal")

_CONVENTION = (
    "Skewed right (or positively skewed) means the longer tail is on the right, towards "
    "larger values; skewed left (or negatively skewed) means the longer tail is on the "
    "left, towards smaller values."
)

_CONTEXTS = (
    dict(var="delivery times", unit="minutes", lo_lo=0, lo_hi=10, width=10),
    dict(var="plant heights", unit="cm", lo_lo=20, lo_hi=50, width=5),
    dict(var="test scores", unit="marks", lo_lo=0, lo_hi=20, width=10),
    dict(var="daily screen time", unit="minutes", lo_lo=0, lo_hi=40, width=15),
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


def _interval_label(bound) -> str:
    return f"{bound[0]}-{bound[1]}"


def _table_text(ctx, bounds, freqs, *, order=None) -> str:
    idx = order if order is not None else list(range(len(freqs)))
    return ", ".join(f"{_interval_label(bounds[i])}: {freqs[i]}" for i in idx)


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
# Medium: class-frequency table, printed in class order
# ============================================================================

_MEDIUM_TEMPLATES = (
    "{convention} A survey recorded {var} ({unit}), grouped into equal-width classes: "
    "{table}. What is the shape of this distribution?",
    "{convention} The frequency table below groups {var} ({unit}) into equal-width classes, "
    "listed in class order: {table}. Classify the shape of this distribution.",
    "{convention} These are {var} ({unit}), grouped into equal-width classes and listed from "
    "smallest to largest class: {table}. What shape does this histogram have?",
)


def _build_medium(rng):
    label = rng.choice(_LABELS)
    n = rng.choice([7, 8, 9])
    ctx, bounds, freqs = _draw_table(rng, label, n)
    table = _table_text(ctx, bounds, freqs)
    text = phrase(rng, _MEDIUM_TEMPLATES, convention=_CONVENTION, var=ctx["var"], unit=ctx["unit"], table=table)
    working = [
        f"Class frequencies, in order: {', '.join(str(f) for f in freqs)}.",
        f"This matches the '{label}' pattern.",
    ]
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
# Hard: same table, rows printed OUT of class order
# ============================================================================

_HARD_TEMPLATES = (
    "{convention} A frequency table for {var} ({unit}) was typed up with its rows in the "
    "wrong order: {table}. Reorder the classes from smallest to largest, then classify the "
    "shape of this distribution.",
    "{convention} These class-frequency rows for {var} ({unit}) are listed in no particular "
    "order: {table}. Sort them by class first, then state the shape of this distribution.",
    "{convention} The rows of this {var} ({unit}) frequency table have been shuffled: "
    "{table}. Put the classes back in order, then classify the shape.",
)


def _build_hard(rng):
    label = rng.choice(_LABELS)
    n = rng.choice([7, 8, 9])
    ctx, bounds, freqs = _draw_table(rng, label, n)
    order = list(range(n))
    tries = 0
    while True:
        rng.shuffle(order)
        tries += 1
        if order != list(range(n)) or tries >= 10:
            break
    table = _table_text(ctx, bounds, freqs, order=order)
    text = phrase(rng, _HARD_TEMPLATES, convention=_CONVENTION, var=ctx["var"], unit=ctx["unit"], table=table)
    working = [
        f"Sorted by class: {', '.join(f'{_interval_label(bounds[i])}: {freqs[i]}' for i in range(n))}.",
        f"This matches the '{label}' pattern.",
    ]
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
# Exam: two histograms, classify the one picked out by a stated comparison
# ============================================================================

_EXAM_TEMPLATES_RANGE = (
    "{convention} Two groups' {var} ({unit}) were recorded, in equal-width classes of the "
    "same width. Group A: {table_a}. Group B: {table_b}. Classify the shape of whichever "
    "group's data has the WIDER range (the greater number of classes).",
    "{convention} {var_cap} ({unit}) were grouped into equal-width classes for two separate "
    "groups. Group A: {table_a}. Group B: {table_b}. State the shape of the group whose "
    "data spans the WIDER range (more classes).",
)
_EXAM_TEMPLATES_TOTAL = (
    "{convention} Two groups' {var} ({unit}) were recorded, in equal-width classes of the "
    "same width. Group A: {table_a}. Group B: {table_b}. Classify the shape of whichever "
    "group has the LARGER total number of observations.",
    "{convention} {var_cap} ({unit}) were grouped into equal-width classes for two separate "
    "groups. Group A: {table_a}. Group B: {table_b}. State the shape of the group with the "
    "LARGER total number of observations.",
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
    table_winner = _table_text(ctx, bounds_winner, winner_freqs)
    table_loser = _table_text(ctx, bounds_loser, loser_freqs)

    # Which letter (A/B) the winner prints as is itself randomised, so the
    # key is not predictable from "always Group A" or "always Group B".
    winner_is_a = rng.choice([True, False])
    table_a = table_winner if winner_is_a else table_loser
    table_b = table_loser if winner_is_a else table_winner
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
        table_a=table_a,
        table_b=table_b,
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
    distractors = [lab for lab in _LABELS if lab != winner_label]
    return dict(
        text=text,
        latex=text,
        answer=winner_label,
        answer_text=winner_label,
        answer_latex=f"plain:{winner_label}",
        distractors=distractors,
        working=working,
    )


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
