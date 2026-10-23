"""Generates `fig_prob_tree` questions: read and complete a probability tree diagram (AC9M9P01).

One skill: ``f10-probability-tree-diagrams`` (``answer: fraction``). A bag holds counters of two colours and two
are drawn one after the other; the tree diagram is the stimulus, drawn with ``tree_diagram``. Three tiers, each a
genuine step up in what the learner has to do with the tree:

- easy (with replacement): every branch is labelled; one outcome at the end of the tree is marked ``?``. The
  answer is the product of the two branch probabilities along its path.
- medium (without replacement): every branch is labelled, and the second-stage probabilities differ from the
  first-stage ones. The question asks for "two different colours", which is two paths, so the answer is a sum of
  two products.
- hard (without replacement): two of the three sets of branches have one branch left blank (``?``). The learner
  completes them (each set of branches totals 1), then finds the probability that at least one counter is a
  stated colour, by adding three paths or by taking 1 minus the one path that has neither.

No probability the learner is asked for is printed on the tree: the marked outcome of the easy tier shows ``?``,
and the totals of the other tiers are never drawn. The prompt never states a colour count, so the tree is the only
source of the numbers. The figure reader re-derives each answer from the tree alone (plus the colour named in the
hard-tier sentence).
"""

from __future__ import annotations

import re
from fractions import Fraction

from .. import figures
from ..core import (
    attempts,
    check,
    check_pairwise_distinct,
    generator,
    latex_answer,
    phrase,
    pick_distractors,
    rational_text,
)
from ..figure_kinds import space
from . import _space_kit as kit

_COLOURS = (
    ("red", "blue"),
    ("green", "yellow"),
    ("black", "white"),
    ("orange", "purple"),
    ("pink", "grey"),
    ("brown", "green"),
)
_NOUNS = (("counter", "counters"), ("marble", "marbles"), ("ball", "balls"))

_ALT = "A two-stage probability tree for drawing two items one after the other."


def _lab(colour: str) -> str:
    return colour.capitalize()


def _frac(p: Fraction) -> str:
    check(0 < p < 1, "a branch probability must lie strictly between 0 and 1")
    return rational_text(p)


def _build_tree(colours, first, second, blank=frozenset(), asked=None):
    """The tree for two draws.

    ``first[i]`` is the probability of colour ``i`` on the first draw; ``second[i][j]`` that of colour ``j`` on the
    second draw after colour ``i`` on the first. ``blank`` holds paths drawn as ``?``: ``(i,)`` or ``(i, j)``.
    ``asked`` is the leaf ``(i, j)`` that shows ``?`` as its probability."""
    children = []
    for i, ci in enumerate(colours):
        kids = []
        for j, cj in enumerate(colours):
            leaf = {
                "branch": "?" if (i, j) in blank else _frac(second[i][j]),
                "label": _lab(cj),
                "outcome": f"{_lab(ci)}, {_lab(cj)}",
            }
            if asked == (i, j):
                leaf["outcome_p"] = "?"
            kids.append(leaf)
        children.append({"branch": "?" if (i,) in blank else _frac(first[i]), "label": _lab(ci), "children": kids})
    return space.tree_diagram(alt=_ALT, root={"label": "Start", "children": children})


def _paths(fig):
    """Every leaf of the tree as (outcome colours, [branch texts])."""
    out = []
    for first in fig["root"]["children"]:
        for second in first["children"]:
            out.append(((first["label"], second["label"]), [first["branch"], second["branch"]], second))
    return out


def _complete(fig):
    """The tree's branch probabilities as Fractions, each blank filled in as 1 minus its siblings."""
    root = fig["root"]["children"]
    first = [space.parse_probability(n["branch"]) for n in root]
    second = [[space.parse_probability(k["branch"]) for k in n["children"]] for n in root]

    def fill(group):
        blanks = [i for i, v in enumerate(group) if v is None]
        check(len(blanks) <= 1, "more than one blank among sibling branches")
        if blanks:
            group[blanks[0]] = 1 - sum(v for v in group if v is not None)
        return group

    first = fill(first)
    second = [fill(g) for g in second]
    return first, second


def _leaf_probability(first, second, i, j) -> Fraction:
    return first[i] * second[i][j]


# ------------------------------------------------------------------------------------------- tiers


def _draw_counts(rng, lo=5, hi=9):
    """(red, blue) counts with both at least 2 (so no without-replacement branch is impossible)."""
    for _ in attempts(100, "counts of two colours"):
        total = rng.randint(lo, hi)
        a = rng.randint(2, total - 2)
        if total - a >= 2:
            return a, total - a


def _easy(rng):
    cols = rng.choice(_COLOURS)
    cols = cols if rng.random() < 0.5 else (cols[1], cols[0])
    noun, plural = rng.choice(_NOUNS)
    for _ in attempts(100, "an easy tree"):
        total = rng.randint(5, 10)
        a = rng.randint(1, total - 1)
        if 2 * a == total:
            continue
        p = [Fraction(a, total), Fraction(total - a, total)]
        asked = (rng.randrange(2), rng.randrange(2))
        second = [p, p]
        fig = _build_tree(cols, p, second, asked=asked)
        answer = p[asked[0]] * p[asked[1]]
        if kit.leaks(fig, rational_text(answer)):
            continue
        break
    c1, c2 = cols[asked[0]], cols[asked[1]]
    text = phrase(
        rng,
        (
            "A {noun} is drawn at random from a bag, its colour is noted and it is put back. A second {noun} is "
            "then drawn. The tree diagram shows the probabilities. Find the probability that the first {noun} is "
            "{c1} and the second is {c2} (marked ? on the tree).",
            "A bag holds {plural} of two colours. One is drawn at random, its colour is noted and it is returned "
            "to the bag, then a second is drawn. The tree diagram shows the probabilities. Find the probability "
            "that the first is {c1} and the second is {c2} (marked ? on the tree).",
        ),
        noun=noun,
        plural=plural,
        c1=c1,
        c2=c2,
    )
    q1, q2 = p[asked[0]], p[asked[1]]
    other = [p[1 - asked[0]], p[1 - asked[1]]]
    cands = [
        q1 + q2,  # adds the branches instead of multiplying
        q1,  # stops after the first branch
        q2,
        other[0] * q2,  # takes the other colour on the first draw
        q1 * other[1],  # takes the other colour on the second draw
        q1 * q1,  # uses the first colour twice
        q2 * q2,
        1 - answer,  # the probability of anything else
        other[1] * q2,
    ]
    working = [
        f"Multiply along the branches: {rational_text(q1)} × {rational_text(q2)} = {rational_text(answer)}.",
    ]
    return fig, text, answer, cands, working


def _medium(rng):
    cols = rng.choice(_COLOURS)
    cols = cols if rng.random() < 0.5 else (cols[1], cols[0])
    noun, plural = rng.choice(_NOUNS)
    for _ in attempts(100, "a medium tree"):
        a, b = _draw_counts(rng)
        t = a + b
        first = [Fraction(a, t), Fraction(b, t)]
        second = [
            [Fraction(a - 1, t - 1), Fraction(b, t - 1)],  # after the first colour
            [Fraction(a, t - 1), Fraction(b - 1, t - 1)],  # after the second colour
        ]
        fig = _build_tree(cols, first, second)
        answer = first[0] * second[0][1] + first[1] * second[1][0]
        if not kit.leaks(fig, rational_text(answer)):
            break
    text = phrase(
        rng,
        (
            "Two {plural} are drawn at random from a bag of {c1} and {c2} {plural}, one after the other, and the "
            "first is not put back. The tree diagram shows the probabilities. Find the probability that the two "
            "{plural} are different colours.",
            "A {noun} is drawn at random from a bag holding {c1} and {c2} {plural} and kept out. A second {noun} is "
            "then drawn. The tree diagram shows the probabilities. Find the probability that the two {plural} are "
            "different colours.",
            "A bag has {c1} {plural} and {c2} {plural}. Two are taken out one after the other without being put "
            "back, and the tree diagram shows the probabilities. What is the probability that they are different "
            "colours?",
        ),
        noun=noun,
        plural=plural,
        c1=cols[0],
        c2=cols[1],
    )
    path1, path2 = first[0] * second[0][1], first[1] * second[1][0]
    pq = first[0] * first[1]
    cands = [
        path1,  # only one of the two orders
        path2,
        2 * pq,  # with-replacement probabilities for both orders
        1 - answer,  # the probability of two of the same colour
        first[0] + second[0][1],  # adds along one path
        path1 * path2,  # multiplies the two orders instead of adding
    ]
    working = [
        "Different colours means first one colour then the other, in either order.",
        f"{rational_text(path1)} + {rational_text(path2)} = {rational_text(answer)}.",
    ]
    return fig, text, answer, cands, working


def _hard(rng):
    cols = rng.choice(_COLOURS)
    cols = cols if rng.random() < 0.5 else (cols[1], cols[0])
    noun, plural = rng.choice(_NOUNS)
    for _ in attempts(200, "a hard tree"):
        a, b = _draw_counts(rng, 6, 10)
        t = a + b
        first = [Fraction(a, t), Fraction(b, t)]
        second = [
            [Fraction(a - 1, t - 1), Fraction(b, t - 1)],
            [Fraction(a, t - 1), Fraction(b - 1, t - 1)],
        ]
        groups = [(), (0,), (1,)]
        rng.shuffle(groups)
        blank = set()
        for group in groups[:2]:  # two of the three sets of branches have one blank
            blank.add(group + (rng.randrange(2),))
        fig = _build_tree(cols, first, second, blank=blank)
        target = rng.randrange(2)  # the colour the question asks "at least one" of
        other = 1 - target
        answer = 1 - first[other] * second[other][other]
        if not kit.leaks(fig, rational_text(answer)):
            break
    text = phrase(
        rng,
        (
            "Two {plural} are drawn at random from a bag, one after the other, and the first is not put back. "
            "The tree diagram is partly filled in. Complete it, then find the probability that at least one of "
            "the {plural} is {colour}.",
            "A {noun} is drawn at random from a bag and kept out, then a second {noun} is drawn. Some of the "
            "probabilities on the tree diagram are missing. Complete it, then find the probability that at least "
            "one {noun} is {colour}.",
        ),
        noun=noun,
        plural=plural,
        colour=cols[target],
    )
    neither = first[other] * second[other][other]
    t_first = [first[target] * second[target][j] for j in (0, 1)]
    cands = [
        neither,  # forgot to take it from 1
        first[target],  # only the first draw
        first[target] * second[target][target],  # both draws are the colour
        sum(t_first),  # leaves out the path that starts with the other colour
        1 - first[other] * first[other],  # with-replacement numbers for the complement
        first[target] * second[target][target] + first[other] * second[other][target],  # drops a path
    ]
    working = [
        "Each set of branches adds to 1, so a missing branch is 1 minus the others.",
        f"P(at least one) = 1 - P(neither) = 1 - {rational_text(neither)} = {rational_text(answer)}.",
    ]
    return fig, text, answer, cands, working


_TIERS = {"easy": _easy, "medium": _medium, "hard": _hard}


@generator(
    "fig_prob_tree",
    level=9,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def fig_prob_tree(rng, tier="easy"):
    """A two-stage probability tree with a fraction answer. See the module docstring for the three tiers."""
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(40, "a tree with three distinct wrong answers"):
        fig, text, answer, candidates, working = _TIERS[tier](rng)
        check(0 < answer < 1, "a tree probability must lie strictly between 0 and 1")
        answer_text = rational_text(answer)
        pool = [rational_text(c) for c in candidates if 0 < c < 1 and c != answer]
        distractors = pick_distractors(answer_text, pool)
        if len(distractors) == 3:
            break
    check_pairwise_distinct(answer, *[Fraction(d) for d in distractors])
    return dict(
        text=text,
        answer={"p": answer.numerator, "q": answer.denominator},
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        distractors=distractors,
        working=working,
        figure=fig,
    )


# ------------------------------------------------------------------------------------------- reader


def _read_tree(question) -> str:
    """Re-derive the answer from the tree alone: the marked outcome's path product (easy), the two
    different-colour paths added (medium), or at least one of the named colour once blanks are filled (hard)."""
    fig = question.figure
    first, second = _complete(fig)
    leaves = _paths(fig)
    marked = [
        (idx, leaf) for idx, (_cols, _branches, leaf) in enumerate(leaves) if leaf.get("outcome_p") == space.BLANK
    ]
    if marked:
        check(len(marked) == 1, "more than one marked outcome")
        i, j = divmod(marked[0][0], 2)
        value = _leaf_probability(first, second, i, j)
    else:
        text = question.text
        if "different colours" in text:
            value = _leaf_probability(first, second, 0, 1) + _leaf_probability(first, second, 1, 0)
        else:
            m = re.search(r" is (\w+)\.$", text)
            check(m is not None, "cannot find the colour in the question")
            colour = m.group(1).capitalize()
            names = [n["label"] for n in fig["root"]["children"]]
            check(colour in names, f"{colour} is not a colour on the tree")
            target = names.index(colour)
            other = 1 - target
            value = 1 - _leaf_probability(first, second, other, other)
    return rational_text(value)


figures.register_reader("fig_prob_tree", _read_tree)
