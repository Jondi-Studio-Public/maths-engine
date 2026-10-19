"""Generates `fr2c_growing_patterns` and `fr2c_growing_patterns_exam` questions: growing matchstick figures.

One skill: ``f10-growing-visual-patterns`` (``answer: number``). Figures 1, 2 and 3 of a pattern made of matchsticks are
drawn with the existing ``cartesian`` figure kind (one segment for each match; the plane carries no scale). The patterns are a row
of squares (3n + 1 matches), a strip of triangles (2n + 1), a double row of squares (5n + 2) and a row of squares with a
diagonal match in each (4n + 1).

- easy: the number of matches in Figure 4, 5 or 6, a figure not far past the three drawn, which is found by seeing how many matches each figure adds.
- medium: the number of matches in a figure well beyond the drawn ones (Figure 8 to 20), which needs the rule.
- hard: the largest figure that can be built from a stated number of matches (some are left over), which needs the rule
  and a division that is not exact.

``fr2c_growing_patterns_exam`` is the chain: (a) matches in Figure 5, (b) a rule in n (an expression), (c) matches in a far
figure (follows through from (b)), (d) which figure uses exactly a stated number of matches (follows through from (b)).

The match counts are taken from the drawn segments (after removing shared matches) and checked against the rule.
"""

from __future__ import annotations

import re

import sympy
from sympy import Symbol

from mathgen import figures
from mathgen.core import attempts, check, generator, phrase

from . import _fr2c_kit as kit
from .exd_x_r2a_common import at, pretty

_N = Symbol("n")
_H = 0.866
_GAP = 1.4
_NAMES = ("Mia", "Tom", "Aiko", "Dev", "Leila", "Sam", "Jess", "Kofi", "Priya", "Marcus", "Zoe", "Ben")
_STICKS = ("match", "matchstick", "toothpick", "straw")

# pattern -> (matches added by each figure, matches in Figure 1)
_PATTERNS = {"squares": (3, 4), "triangles": (2, 3), "double": (5, 7), "braced": (4, 5)}


def _edges(pattern: str, n: int):
    """The distinct matches of Figure ``n`` as ((x1, y1), (x2, y2)) pairs in matchstick lengths, and its width."""
    edges: set = set()

    def add(a, b):
        edges.add((a, b) if a <= b else (b, a))

    if pattern in ("squares", "braced"):
        for i in range(n):
            add((i, 0), (i + 1, 0))
            add((i, 1), (i + 1, 1))
            add((i, 0), (i, 1))
            add((i + 1, 0), (i + 1, 1))
            if pattern == "braced":
                add((i, 0), (i + 1, 1))
        return edges, n, 1
    if pattern == "double":
        for i in range(n):
            for j in range(2):
                add((i, j), (i + 1, j))
                add((i, j + 1), (i + 1, j + 1))
                add((i, j), (i, j + 1))
                add((i + 1, j), (i + 1, j + 1))
        return edges, n, 2
    for t in range(n):  # triangles
        if t % 2 == 0:
            x = t / 2
            a, b, c = (x, 0), (x + 1, 0), (x + 0.5, _H)
        else:
            x = (t - 1) / 2
            a, b, c = (x + 0.5, _H), (x + 1.5, _H), (x + 1, 0)
        add(a, b)
        add(b, c)
        add(a, c)
    return edges, (n + 1) / 2, _H


def _figure(pattern: str):
    segs, labels = [], []
    ox = 1.2
    top = 0.0
    spans = []
    for n in (1, 2, 3):
        edges, width, height = _edges(pattern, n)
        for (x1, y1), (x2, y2) in sorted(edges):
            segs.append(
                {"from": [round(ox + x1, 4), round(1.8 + y1, 4)], "to": [round(ox + x2, 4), round(1.8 + y2, 4)]}
            )
        labels.append({"at": [round(ox + width / 2, 4), 0.9], "label": f"Figure {n}"})
        spans.append((ox, ox + width))
        top = max(top, height)
        ox += width + _GAP
    return figures.cartesian(
        x_range=(0.2, round(ox - _GAP + 1.0, 4)),
        y_range=(0.2, round(1.8 + top + 0.8, 4)),
        grid=1000,
        alt="Figures 1, 2 and 3 of a pattern made of matchsticks.",
        points=labels,
        segments=segs,
        x_label="",
        y_label="",
        tick_labels=False,
    )


def _drawn_counts(fig) -> list[int]:
    """Matches in each of the three drawn figures, from the segments alone (each is nearest to its label)."""
    centres = sorted(p["at"][0] for p in fig["points"])
    counts = [0, 0, 0]
    for s in fig["segments"]:
        mid = (s["from"][0] + s["to"][0]) / 2
        counts[min(range(3), key=lambda i: abs(centres[i] - mid))] += 1
    return counts


def _rule(pattern: str):
    d, c1 = _PATTERNS[pattern]
    return d, c1 - d  # matches = d * n + (c1 - d)


def _verified_figure(pattern: str):
    fig = _figure(pattern)
    d, k = _rule(pattern)
    counts = _drawn_counts(fig)
    check(counts == [d * n + k for n in (1, 2, 3)], f"the drawn {pattern} figures do not follow {d}n + {k}")
    return fig


def _stem(rng, pattern, stick):
    shape = {
        "squares": "squares in a row",
        "triangles": "triangles in a strip",
        "double": "squares in two rows",
        "braced": "squares in a row, each with a diagonal match across it",
    }[pattern]
    return phrase(
        rng,
        (
            "The pictures show Figures 1, 2 and 3 of a pattern. Each figure is made from {shape}, using {sticks}.",
            "A pattern of {shape} is built from {sticks}. Figures 1, 2 and 3 are drawn.",
            "{who} builds a pattern of {shape} out of {sticks}. The first three figures are shown.",
        ),
        shape=shape,
        sticks=stick + "es" if stick.endswith("h") else stick + "s",
        who=rng.choice(_NAMES),
    )


# ------------------------------------------------------------------------------------------ questions


def _easy(rng):
    pattern = rng.choice(list(_PATTERNS))
    stick = rng.choice(_STICKS)
    d, k = _rule(pattern)
    c1 = d + k
    n = rng.choice([4, 5, 6])
    value = d * n + k
    c3 = d * 3 + k
    wrong = [
        ("fr2c-added-one-each-time", c3 + (n - 3)),
        ("fr2c-multiplied-the-first-figure", n * c1),
        ("fr2c-rule-without-constant", n * d),
        ("fr2c-added-the-first-figure-again", c3 + (n - 3) * c1),
        ("fr2c-figure-number-off-by-one", d * (n + 1) + k),
        ("fr2c-figure-number-off-by-one", d * (n - 1) + k),
    ]
    sticks = stick + "es" if stick.endswith("h") else stick + "s"
    text = f"{_stem(rng, pattern, stick)} How many {sticks} are in Figure {n}?"
    steps = " + ".join([str(d)] * (n - 3))
    working = [
        f"Each figure uses {d} more than the one before: Figure 3 has {c3}, so Figure {n} has {c3} + {steps} = {value}."
    ]
    return pattern, text, value, wrong, working


def _medium(rng):
    pattern = rng.choice(list(_PATTERNS))
    stick = rng.choice(_STICKS)
    d, k = _rule(pattern)
    c1 = d + k
    n = rng.choice([8, 9, 10, 12, 15, 20])
    value = d * n + k
    wrong = [
        ("fr2c-multiplied-the-first-figure", n * c1),
        ("fr2c-rule-without-constant", n * d),
        ("fr2c-figure-number-off-by-one", d * (n + 1) + k),
        ("fr2c-figure-number-off-by-one", d * (n - 1) + k),
        ("fr2c-added-one-each-time", d * 3 + k + (n - 3)),
        ("fr2c-rule-constant-taken-as-first-figure", d * n + c1),
    ]
    sticks = stick + "es" if stick.endswith("h") else stick + "s"
    text = f"{_stem(rng, pattern, stick)} How many {sticks} are in Figure {n}?"
    working = [f"The rule is {d} x n + {k}.", f"For n = {n}: {d} x {n} + {k} = {value}."]
    return pattern, text, value, wrong, working


def _hard(rng):
    pattern = rng.choice(list(_PATTERNS))
    stick = rng.choice(_STICKS)
    d, k = _rule(pattern)
    c1 = d + k
    for _ in attempts(200, "a stock of sticks that leaves some over"):
        n = rng.randint(12, 40)
        left = rng.randint(1, d - 1)
        stock = d * n + k + left
        if stock % 10 == 0 or stock % 5 == 0:
            break
    sticks = stick + "es" if stick.endswith("h") else stick + "s"
    name = rng.choice(_NAMES)
    text = (
        f"{_stem(rng, pattern, stick)} {name} has {stock} {sticks}. What is the largest figure that {name} can build?"
    )
    value = n
    wrong = [
        ("fr2c-rule-without-constant", stock // d),
        ("fr2c-leftover-rounded-up", n + 1),
        ("fr2c-figure-number-off-by-one", n - 1),
        ("fr2c-multiplied-the-first-figure", stock // c1),
        ("fr2c-rule-constant-taken-as-first-figure", (stock - c1) // d),
    ]
    working = [
        f"The number of {sticks} in Figure n is {d} x n + {k}.",
        f"{stock} - {k} = {stock - k}, and {stock - k} / {d} = {n} with {left} left over.",
        f"So the largest figure is Figure {n}.",
    ]
    return pattern, text, value, wrong, working


_TIERS = {"easy": _easy, "medium": _medium, "hard": _hard}


@generator(
    "fr2c_growing_patterns",
    level=6,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def fr2c_growing_patterns(rng, tier="easy"):
    """A drawn growing matchstick pattern with a number answer. See the module docstring for the tiers."""
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(60, "a pattern question with three different wrong answers"):
        pattern, text, value, wrong, working = _TIERS[tier](rng)
        try:
            fields = kit.number_fields(value, [(m, w) for m, w in wrong if w > 0])
        except kit.Redraw:
            continue
        break
    return dict(text=text, working=working, figure=_verified_figure(pattern), **fields)


def _read_pattern(question) -> str:
    c1, c2, c3 = _drawn_counts(question.figure)
    d = c2 - c1
    check(c3 - c2 == d, "the drawn figures do not grow by a constant number")
    text = question.text
    m = re.search(r"in Figure (\d+)\?", text)
    if m:
        return str(c1 + (int(m.group(1)) - 1) * d)
    stock = int(re.search(r"has (\d+)", text).group(1))
    return str((stock - c1) // d + 1)


figures.register_reader("fr2c_growing_patterns", _read_pattern)


# ------------------------------------------------------------------------------------------ exam chain


@generator("fr2c_growing_patterns_exam", level=9, presets={"exam": {}})
def fr2c_growing_patterns_exam(rng):
    """Matches in Figure 5, a rule in n, a far figure, and the figure that uses an exact number of matches."""
    return kit.build_with_redraw(rng, _build_fr2c_growing_patterns_exam, what="fr2c_growing_patterns_exam")


def _build_fr2c_growing_patterns_exam(rng):
    pattern = rng.choice(list(_PATTERNS))
    stick = rng.choice(_STICKS)
    sticks = stick + "es" if stick.endswith("h") else stick + "s"
    d, k = _rule(pattern)
    c1 = d + k
    far = rng.choice([30, 40, 50, 60])
    target = rng.randint(12, 45)
    exact = d * target + k
    rule = d * _N + k
    check(sympy.simplify(rule.subs(_N, 1) - c1) == 0, "the rule disagrees with Figure 1")
    stem = f"{_stem(rng, pattern, stick)}"
    rule_text = pretty(rule)
    parts = [
        kit.num_part(
            f"How many {sticks} are in Figure 5?",
            d * 5 + k,
            1,
            [f"Each figure uses {d} more than the one before, so Figure 5 has {d * 3 + k} + {d} + {d} = {d * 5 + k}."],
            [
                ("fr2c-added-one-each-time", d * 3 + k + 2),
                ("fr2c-multiplied-the-first-figure", 5 * c1),
                ("fr2c-rule-without-constant", 5 * d),
                ("fr2c-figure-number-off-by-one", d * 4 + k),
            ],
            sensible_range=(0, None),
        ),
        kit.expr_part(
            f"Write a rule for the number of {sticks} in Figure n, in terms of n.",
            rule,
            2,
            [
                f"Each figure adds {d}, so the rule starts with {d}n. Figure 1 has {c1}, so add {k}.",
                f"The rule is {rule_text}.",
            ],
            [
                ("fr2c-rule-without-constant", d * _N),
                ("fr2c-rule-constant-taken-as-first-figure", d * _N + c1),
                ("fr2c-multiplied-the-first-figure", c1 * _N),
                ("fr2c-added-one-each-time", _N + c1),
            ],
            var=_N,
        ),
        kit.num_part(
            f"Use your rule to find the number of {sticks} in Figure {far}.",
            d * far + k,
            1,
            [f"{d} x {far} + {k} = {d * far + k}."],
            [
                ("fr2c-multiplied-the-first-figure", far * c1),
                ("fr2c-rule-without-constant", far * d),
                ("fr2c-figure-number-off-by-one", d * (far + 1) + k),
                ("fr2c-figure-number-off-by-one", d * (far - 1) + k),
            ],
            follow_through=at("b", far, "n"),
            sensible_range=(0, None),
        ),
        kit.num_part(
            f"A figure in the pattern uses exactly {exact} {sticks}. Using your rule, which figure is it?",
            target,
            2,
            [f"{d}n + {k} = {exact}, so {d}n = {exact - k} and n = {target}."],
            [
                ("fr2c-rule-without-constant", (exact // d)),
                ("fr2c-figure-number-off-by-one", target + 1),
                ("fr2c-figure-number-off-by-one", target - 1),
                ("fr2c-multiplied-the-first-figure", exact // c1),
                ("fr2c-rule-constant-taken-as-first-figure", (exact - c1) // d),
            ],
            follow_through=f"({exact} - {at('b', 0, 'n')})/({at('b', 1, 'n')} - {at('b', 0, 'n')})",
            sensible_range=(0, None),
        ),
    ]
    return dict(text=stem, figure=_verified_figure(pattern), parts=parts)
