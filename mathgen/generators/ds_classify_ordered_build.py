"""Classify several variables in order: `ds_classify_ordered`, appended to the `classify-data` skill.

The learner types one word per variable, in the order the variables are listed (``nominal, discrete,
continuous``). The variable pools and the reasons are imported from `r3j_classify_build`, not copied.

Classes are drawn independently with repeats allowed, so the reply space is 4**3 = 64 (medium) and 4**4 = 256
(hard) and no pattern, such as "all four classes once", can be guessed.

* medium: three plain variables (no wording trap), at least two different classes.
* hard: four variables, at least three different classes and at least two whose wording invites a mistake
  (numerals used as labels, ranks and ratings, a measurement recorded in whole units).
"""

from __future__ import annotations

from ..core import attempts, check, generator
from ._sr2a_kit import listed
from .r3j_classify_build import _WHY, _WRONG_TAG, CLASSES, POOL

TOPIC = "ds_classify_ordered"
_WORD = {c: c.split()[0].lower() for c in CLASSES}
_LABELS = "ABCD"

_LEADS = [
    "A survey records these variables: {items}. Classify each variable as nominal, ordinal, discrete or "
    "continuous. Give your answers in the order listed, separated by commas.",
    "Here are {n} variables from a data set: {items}. State the type of each (nominal, ordinal, discrete or "
    "continuous), in the order A to {last}.",
    "A study collects these variables: {items}. List the type of each one in order, choosing from nominal, "
    "ordinal, discrete and continuous.",
]


def _draw(rng, n, min_classes, twists):
    for _ in attempts(500, "variables of the wanted mix of classes"):
        classes = [rng.choice(CLASSES) for _ in range(n)]
        if len(set(classes)) < min_classes:
            continue
        flags = [rng.random() < 0.5 if twists else False for _ in range(n)]
        if twists and sum(flags) < twists:
            continue
        picked = []
        for c, f in zip(classes, flags):
            pool = [p for p in POOL if p[1] == c and p[2] == f and p not in picked]
            picked.append(rng.choice(pool))
        if len({p[0] for p in picked}) == n:
            return picked
    raise AssertionError("unreachable")


def _build(rng, n, min_classes, twists):
    picked = _draw(rng, n, min_classes, twists)
    items = [_WORD[p[1]] for p in picked]
    listing = "; ".join(f"({_LABELS[i]}) {p[0]}" for i, p in enumerate(picked))
    text = rng.choice(_LEADS).format(items=listing, n=n, last=_LABELS[n - 1])
    wrong = []
    for i, p in enumerate(picked):
        for c in CLASSES:
            if c != p[1]:
                wrong.append((items[:i] + [_WORD[c]] + items[i + 1 :], _WRONG_TAG[(p[1], c)]))
    rng.shuffle(wrong)
    working = [f"{_LABELS[i]}: {_WORD[p[1]]}, because {_WHY[p[1]]}." for i, p in enumerate(picked)]
    q = listed(text, items, working, wrong)
    check(q is not None, "too few distractors survived")
    # second route: every key word is one of the four classes and the pool tags agree with the words
    check(all(w in _WORD.values() for w in q["answer"]) and len(q["answer"]) == n, "key words")
    return q


@generator(
    TOPIC,
    level=8,
    presets={"medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def ds_classify_ordered(rng, tier="medium"):
    if tier == "medium":
        return _build(rng, 3, 2, 0)
    if tier != "hard":
        raise ValueError(f"unknown tier {tier!r}")
    return _build(rng, 4, 3, 2)
