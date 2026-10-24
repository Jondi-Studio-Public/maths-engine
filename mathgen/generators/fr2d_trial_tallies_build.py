"""Generator for `f10-trial-tallies-compare-outcomes`: read tallies from repeated chance trials (Years 3 to 5).

Three tiers, all answered with a whole number and all without fractions:

* easy (Year 3): read one tally, compare two tallies, or find how many times the most common outcome came up.
* medium (Year 4): compare an outcome's tally with the count equal likelihood would give, from a spinner whose
  sections are all the same size.
* hard (Year 5): two tallies of one experiment (after the first trials, and after more), then the extra trials on
  their own, a gap between two outcomes in the extra trials, or the most common outcome against equal likelihood.

Tallies are written as four strokes with a diagonal line through them for each group of five and single strokes for the rest. Every key is
re-derived a second way (summing the tally text back, or counting from the table), and every option is a different
whole number from the key by construction.
"""

from __future__ import annotations

from mathgen.answers import Distractor
from mathgen.core import (
    GRADER_TOLERANCE,
    attempts,
    check,
    generator,
    latex_answer,
    phrase,
    pick_distractors,
)

from ._fr2d_kit import Short, int_d, need, redraw

_GATE = "||||/"  # four strokes with a diagonal through them: plain ASCII, so it needs no special font

# lead sentence, outcomes, the thing that is done once per trial, and a noun for a trial
_CONTEXTS = (
    dict(
        lead="Priya spun a spinner with {k} equal sections and recorded the colour it landed on each time.",
        cats=["red", "blue", "green", "yellow", "purple"],
        trial="spin",
    ),
    dict(
        lead="Tom pulled a counter from a bag, wrote down its colour and put it back, again and again.",
        cats=["red", "blue", "green", "white", "black"],
        trial="draw",
    ),
    dict(
        lead="A class dropped a drawing pin {n} times. It landed point up or point down, and the class tallied each.",
        cats=["point up", "point down"],
        trial="drop",
    ),
    dict(
        lead="Hana spun a spinner marked with shapes. She tallied the shape it stopped on each time.",
        cats=["circle", "square", "triangle", "star"],
        trial="spin",
    ),
)
_EQUAL_SPINNERS = (
    dict(
        lead="A spinner has {k} sections that are all the same size, each a different colour. "
        "Mei spun it {n} times and tallied the colour each time.",
        cats=["red", "blue", "green", "yellow", "purple"],
        trial="spin",
    ),
    dict(
        lead="A game board spinner has {k} equal sections, each a different colour. Jack spun it {n} times and "
        "tallied where it stopped.",
        cats=["orange", "pink", "green", "blue", "grey"],
        trial="spin",
    ),
)


def _tally(n: int) -> str:
    groups = [_GATE] * (n // 5) + (["|" * (n % 5)] if n % 5 else [])
    return " ".join(groups)


def _untally(text: str) -> int:
    """The count a tally string stands for (the second route to every key)."""
    return sum(5 if g == _GATE else len(g) for g in text.split())


def _table(cats, counts) -> str:
    return "; ".join(f"{c}: {_tally(n)}" for c, n in zip(cats, counts))


def _numbers(cats, counts) -> str:
    return ", ".join(f"{c} {n}" for c, n in zip(cats, counts))


def _finish(stem, key, cands, working):
    options = pick_distractors(
        str(key), [c for c in cands if c.value > 0 and c.value != key], 3, tolerance=GRADER_TOLERANCE
    )
    need(options)
    working = [line[:1].upper() + line[1:] for line in working]
    return dict(
        text=stem,
        answer_kind="number",
        answer=key,
        answer_text=str(key),
        answer_latex=latex_answer(str(key)),
        working=working,
        distractors=options,
    )


def _distinct_counts(rng, k, lo, hi, total_lo, total_hi):
    for _ in attempts(2000, "distinct tally counts"):
        counts = [rng.randint(lo, hi) for _ in range(k)]
        if len(set(counts)) == k and total_lo <= sum(counts) <= total_hi:
            return counts
    raise AssertionError  # pragma: no cover


_EASY_ASK = {
    "read": ("How many times did {x} come up?", "How many {x} results were there?"),
    "more": (
        "How many more times did {x} come up than {y}?",
        "{x} came up more often than {y}. How many more times?",
    ),
    "most": (
        "How many times did the outcome that came up most often come up?",
        "The most common result came up how many times?",
    ),
    "fewer": (
        "How many fewer times did the least common outcome come up than the most common outcome?",
        "What is the gap between the count for the most common outcome and the count for the least common?",
    ),
}


def _easy(rng):
    ctx = rng.choice(_CONTEXTS)
    pool = ctx["cats"]
    k = 2 if len(pool) == 2 else rng.randint(3, min(4, len(pool)))
    cats = [pool[i] for i in sorted(rng.sample(range(len(pool)), k))]
    counts = _distinct_counts(rng, k, 2, 14, 18, 44)
    total = sum(counts)
    lead = ctx["lead"].format(k=k, n=total)
    kind = rng.choice(["read", "more", "most", "fewer"] if k >= 3 else ["read", "more", "most"])
    order = sorted(range(k), key=lambda i: counts[i])
    lo, hi = order[0], order[-1]
    check(all(_untally(_tally(n)) == n for n in counts), "tally text does not stand for its count")

    def as_groups(n):  # a learner who counts each gate as one mark
        return n // 5 + n % 5

    if kind == "read":
        i = rng.randrange(k)
        x, key = cats[i], counts[i]
        cands = [
            int_d(rng.choice([c for j, c in enumerate(counts) if j != i]), "fr2d-reads-the-wrong-outcome"),
            int_d(as_groups(key), "fr2d-tally-group-as-one"),
            int_d(key + 1, "fr2d-tally-off-by-one"),
            int_d(total, "fr2d-reports-the-total"),
            int_d(key - 1, "fr2d-tally-off-by-one"),
        ]
        gate_word = "group" if key // 5 == 1 else "groups"
        working = [f"{x}: {_tally(key)} is {key // 5} {gate_word} of 5 and {key % 5} more, so {key}."]
        ask = phrase(rng, _EASY_ASK["read"], x=x)
    elif kind == "more":
        i, j = rng.sample(range(k), 2)
        if counts[i] < counts[j]:
            i, j = j, i
        key = counts[i] - counts[j]
        cands = [
            int_d(counts[i] + counts[j], "fr2d-adds-instead-of-compares"),
            int_d(counts[i], "fr2d-reads-the-wrong-outcome"),
            int_d(counts[j], "fr2d-reads-the-wrong-outcome"),
            int_d(abs(as_groups(counts[i]) - as_groups(counts[j])), "fr2d-tally-group-as-one"),
            int_d(key + 1, "fr2d-tally-off-by-one"),
        ]
        working = [f"{cats[i]}: {counts[i]}. {cats[j]}: {counts[j]}.", f"{counts[i]} - {counts[j]} = {key}."]
        ask = phrase(rng, _EASY_ASK["more"], x=cats[i], y=cats[j])
    elif kind == "most":
        key = counts[hi]
        cands = [
            int_d(counts[lo], "fr2d-reads-the-least-common"),
            int_d(total, "fr2d-reports-the-total"),
            int_d(counts[hi] - counts[lo], "fr2d-gap-for-count"),
            int_d(counts[order[-2]], "fr2d-reads-the-wrong-outcome"),
            int_d(key - 1, "fr2d-tally-off-by-one"),
        ]
        working = [f"The longest tally is {cats[hi]}: {_tally(key)}, which is {key}."]
        ask = phrase(rng, _EASY_ASK["most"])
    else:
        key = counts[hi] - counts[lo]
        cands = [
            int_d(counts[hi], "fr2d-gap-for-count"),
            int_d(counts[hi] + counts[lo], "fr2d-adds-instead-of-compares"),
            int_d(counts[lo], "fr2d-gap-for-count"),
            int_d(counts[hi] - counts[order[1]], "fr2d-reads-the-wrong-outcome"),
            int_d(key + 1, "fr2d-tally-off-by-one"),
        ]
        working = [f"Most: {counts[hi]}. Least: {counts[lo]}.", f"{counts[hi]} - {counts[lo]} = {key}."]
        ask = phrase(rng, _EASY_ASK["fewer"])
    stem = f"{lead} The tally of the results: {_table(cats, counts)}. {ask}"
    return _finish(stem, key, cands, working)


def _equal_counts(rng, k, m):
    """Tallies for k equally likely outcomes in k*m trials: none exactly m, at least one above and one below."""
    n = k * m
    for _ in attempts(3000, "tallies around an equal share"):
        cuts = sorted(rng.sample(range(1, n), k - 1))
        counts = [b - a for a, b in zip([0, *cuts], [*cuts, n])]
        if min(counts) < 2 or m in counts or max(counts) <= m or min(counts) >= m:
            continue
        if len(set(counts)) < k - 1:
            continue
        return counts
    raise AssertionError  # pragma: no cover


_MEDIUM_ASK = (
    "If every colour were equally likely, each would come up the same number of times. "
    "How many {word} times than that did {x} come up?",
    "Equal likelihood would give each colour the same count. How many {word} times than that count did {x} come up?",
)


def _medium(rng):
    ctx = rng.choice(_EQUAL_SPINNERS)
    k = rng.choice([3, 4, 5])
    m = rng.randint(5, 12)
    n = k * m
    cats = ctx["cats"][:k] if k < 5 else ctx["cats"]
    counts = _equal_counts(rng, k, m)
    check(sum(counts) == n, "the tallies do not add to the number of trials")
    more = rng.random() < 0.6
    pool = [i for i in range(k) if (counts[i] > m) == more]
    i = rng.choice(pool)
    key = abs(counts[i] - m)
    check(n // k == m and n % k == 0, "the equal share is not a whole number")
    other = rng.choice([j for j in range(k) if j != i])
    cands = [
        int_d(counts[i], "fr2d-count-for-difference-from-expected"),
        int_d(m, "fr2d-equal-share-for-difference"),
        int_d(counts[i] + m, "fr2d-adds-instead-of-compares"),
        int_d(abs(counts[other] - m), "fr2d-reads-the-wrong-outcome"),
        int_d(abs(counts[i] - counts[other]), "fr2d-compares-with-another-outcome"),
        int_d(key + 1, "fr2d-tally-off-by-one"),
    ]
    stem = (
        phrase(rng, [ctx["lead"]], k=k, n=n)
        + f" The tally of the results: {_table(cats, counts)}. "
        + phrase(rng, _MEDIUM_ASK, word="more" if more else "fewer", x=cats[i])
    )
    working = [
        f"{n} spins shared equally among {k} colours is {n} / {k} = {m} each.",
        f"{cats[i]} came up {counts[i]} times, so the difference is |{counts[i]} - {m}| = {key}.",
    ]
    return _finish(stem, key, cands, working)


_HARD_CONTEXTS = (
    dict(
        lead="Ruby spun a spinner with {k} equal sections. She tallied the results after her first {f} spins and "
        "again after {t} spins altogether.",
        cats=["red", "blue", "green", "yellow"],
        trial="spin",
    ),
    dict(
        lead="A class drew a counter from a bag and put it back each time. They tallied the colours after "
        "{f} draws and again after {t} draws altogether. The bag holds the same number of counters of each of "
        "its {k} colours.",
        cats=["red", "blue", "green", "white"],
        trial="draw",
    ),
)


def _hard(rng):
    ctx = rng.choice(_HARD_CONTEXTS)
    k = rng.choice([3, 4])
    cats = ctx["cats"][:k]
    f = rng.choice([20, 24, 30])
    extra = rng.choice([k * 10, k * 12, k * 15]) if k == 4 else rng.choice([30, 36, 45])
    t = f + extra
    for _ in attempts(3000, "tallies for a first set and the extra trials"):
        first = _split(rng, f, k)
        more = _split(rng, extra, k)
        if min(first) < 2 or min(more) < 3 or len(set(more)) < k:
            continue
        break
    total_counts = [a + b for a, b in zip(first, more)]
    check(sum(total_counts) == t and sum(more) == extra, "tallies do not add to their trials")
    kind = rng.choice(["extra", "gap", "share"])
    trial = ctx["trial"]
    lead = ctx["lead"].format(k=k, f=f, t=t)
    shown = (
        f"After {f} {trial}s: {_numbers(cats, first)}. After {t} {trial}s altogether: {_numbers(cats, total_counts)}."
    )
    if kind == "extra":
        i = rng.randrange(k)
        key = total_counts[i] - first[i]
        check(key == more[i], "the extra count differs from the difference of the two tallies")
        cands = [
            int_d(total_counts[i], "fr2d-reads-total-not-extra"),
            int_d(first[i], "fr2d-reads-first-tally-not-extra"),
            int_d(total_counts[i] + first[i], "fr2d-adds-instead-of-subtracts"),
            int_d(extra, "fr2d-reports-the-total"),
            int_d(abs(total_counts[i] - first[i] - 1), "fr2d-tally-off-by-one"),
        ]
        ask = f"How many of the extra {extra} {trial}s were {cats[i]}?"
        working = [
            f"{cats[i]} after {t}: {total_counts[i]}. After {f}: {first[i]}.",
            f"{total_counts[i]} - {first[i]} = {key}.",
        ]
    elif kind == "gap":
        i, j = sorted(rng.sample(range(k), 2), key=lambda a: -more[a])
        key = (total_counts[i] - first[i]) - (total_counts[j] - first[j])
        check(key == more[i] - more[j] > 0, "gap disagrees with the extra counts")
        cands = [
            int_d(total_counts[i] - total_counts[j], "fr2d-gap-over-all-trials"),
            int_d(first[i] - first[j], "fr2d-gap-over-first-trials"),
            int_d(more[i] + more[j], "fr2d-adds-instead-of-compares"),
            int_d(more[i], "fr2d-reads-the-wrong-outcome"),
            int_d(key + 1, "fr2d-tally-off-by-one"),
        ]
        ask = f"In the extra {extra} {trial}s only, how many more times did {cats[i]} come up than {cats[j]}?"
        working = [
            f"Extra {cats[i]}: {total_counts[i]} - {first[i]} = {more[i]}. Extra {cats[j]}: {total_counts[j]} - {first[j]} = {more[j]}.",
            f"{more[i]} - {more[j]} = {key}.",
        ]
    else:
        if t % k:
            raise Short
        share = t // k
        hi = max(range(k), key=lambda a: total_counts[a])
        key = total_counts[hi] - share
        if key <= 0:
            raise Short
        cands = [
            int_d(total_counts[hi], "fr2d-count-for-difference-from-expected"),
            int_d(share, "fr2d-equal-share-for-difference"),
            int_d(total_counts[hi] - first[hi], "fr2d-reads-total-not-extra"),
            int_d(total_counts[hi] - min(total_counts), "fr2d-compares-with-another-outcome"),
            int_d(key + 1, "fr2d-tally-off-by-one"),
        ]
        ask = (
            f"If every outcome were equally likely, each would come up the same number of times in {t} {trial}s. "
            f"How many more times than that did the most common outcome come up?"
        )
        working = [
            f"{t} / {k} = {share} each.",
            f"Most common: {cats[hi]} with {total_counts[hi]}.",
            f"{total_counts[hi]} - {share} = {key}.",
        ]
    return _finish(f"{lead} {shown} {ask}", key, cands, working)


def _split(rng, n, k):
    cuts = sorted(rng.sample(range(1, n), k - 1))
    return [b - a for a, b in zip([0, *cuts], [*cuts, n])]


@generator(
    "fr2d_trial_tallies",
    level=4,
    presets={"easy": {"tier": "easy"}, "medium": {"tier": "medium"}, "hard": {"tier": "hard"}},
)
@redraw
def fr2d_trial_tallies(rng, tier: str = "easy"):
    return {"easy": _easy, "medium": _medium, "hard": _hard}[tier](rng)
