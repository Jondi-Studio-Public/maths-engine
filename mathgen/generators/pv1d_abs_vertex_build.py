"""Generates `pv1d_abs_vertex` questions: read the vertex of the graph of y = |x - h| + k from a figure.

One NEW senior figure skill.

- ``pv1d_abs_vertex`` (senior-use-and-apply-the-notation-left-x-right, ACMSM098)
  -- the learner is shown the graph of ``y = |x - h| + k`` on an integer grid and
  states its vertex ``(h, k)``. The key is a pair (``answer: pair``, the same shape
  and grader as ``quadratic-vertex``); the picture is a ``figures.cartesian`` figure
  built from a sampled V-shaped curve (the web component draws curves but not their
  labels, so two curves are told apart by a lettered point on each).

Tiers escalate by the DECISION the learner makes, never by number size:

- easy   -- the vertex is marked with the point P: read its coordinates.
- medium -- a single unmarked V: locate the corner where the two arms meet.
- hard   -- two V graphs and a lettered point on an ARM of each: first decide which
            graph the named point lies on, then locate THAT graph's corner (the other
            graph's corner is a distractor on every draw).
- exam   -- two V graphs and no lettered point: the graph is selected by a property of
            the picture, whether it meets the x-axis (k > 0 never meets it, k < 0 crosses
            it twice). The prompt asks for either the one that never meets the axis or
            the one that crosses it twice, 50/50, so the answer graph is not always the
            higher one.

Every value the key needs lies on the grid: h and k are integers in [-5, 5], the
slope of each arm is +-1, the corner is an exact sample of the curve, and a lettered
point sits on a grid intersection of an arm (never at a vertex). Nothing the key
needs is printed: the point labels are the single letters P and Q, and the alt text
names only the object.

Distractors, each a named misconception:

- ``other-graph-vertex``  the vertex of the OTHER graph (hard, exam: on every draw)
- ``swapped-coordinates`` (k, h)
- ``sign-of-h``           (-h, k), the ``|x + h|`` slip
- ``sign-of-k``           (h, -k)
- ``x-intercept``         (h + |k|, 0), only when the graph meets the x-axis (k < 0)
- ``y-intercept``         (0, |h| + k)

|h| != |k|, h != 0 and k != 0 on EVERY draw, so the first four distractors are
pairwise distinct from the key and from each other by construction.
"""

from __future__ import annotations

import re

from mathgen import figures
from mathgen.core import (
    GRADER_TOLERANCE,
    check,
    check_pairwise_distinct,
    generator,
    pick_distractors,
)

TOPIC = "pv1d_abs_vertex"

_LO, _HI = -10, 10  # the window, both axes
_N = 2 * (_HI - _LO) + 1  # step 0.5, so every integer x is a sample
_SAMPLE_Y = (_LO, _HI - 0.5)  # sample_curve lets y reach 10 exactly, not 10.5
_RANGE = tuple(v for v in range(-5, 6) if v != 0)
_CELLS = tuple((h, k) for h in _RANGE for k in _RANGE if abs(h) != abs(k))
_POS_CELLS = tuple(c for c in _CELLS if c[1] > 0)
_NEG_CELLS = tuple(c for c in _CELLS if c[1] < 0)

_ALT_ONE = "a V-shaped graph on a coordinate grid"
_ALT_TWO = "two V-shaped graphs on a coordinate grid"

_THROUGH_RE = re.compile(r"through the point marked ([A-Z])\b")
# Exam selectors: a graph that never meets the x-axis has k > 0; one that crosses it
# at two points has k < 0. Every positive template contains one of the first group,
# every negative template one of the second, and no template contains both.
_NEVER_WORDS = ("never meets", "stays above", "does not touch")
_TWICE_WORDS = ("twice", "two points", "two different points")


def _fmt(pair) -> str:
    return f"({pair[0]}, {pair[1]})"


# --------------------------------------------------------------- the picture


def _curve(h: int, k: int) -> dict:
    pieces = figures.sample_curve(lambda x: abs(x - h) + k, _LO, _HI, _SAMPLE_Y, n=_N)
    check(len(pieces) == 1, "a V graph must be one unbroken piece")
    return {"pieces": pieces}


def _samples(curve: dict) -> list:
    return [p for piece in curve["pieces"] for p in piece]


def _corner_of_curve(curve: dict) -> tuple[int, int]:
    """The corner of one drawn V: the lowest sample (the graph opens upward)."""
    pts = _samples(curve)
    lowest = min(range(len(pts)), key=lambda i: (pts[i][1], pts[i][0]))
    check(0 < lowest < len(pts) - 1, "the corner must lie inside the window, with both arms drawn")
    x, y = pts[lowest]
    check(x == int(x) and y == int(y), "the corner must be on a grid intersection")
    return int(x), int(y)


def _corner_from_arms(curve: dict) -> tuple[int, int]:
    """Independent route to the corner: extend the two end samples back along their
    arms (slope -1 on the left, +1 on the right) and intersect the lines."""
    pts = _samples(curve)
    (x1, y1), (x2, y2) = pts[0], pts[-1]
    c1, c2 = x1 + y1, y2 - x2  # left arm y = -x + c1, right arm y = x + c2
    check((c1 - c2) % 2 == 0, "the two arms do not meet on the grid")
    return int((c1 - c2) // 2), int((c1 + c2) // 2)


def _curve_with_point(spec: dict, letter: str) -> dict:
    marked = [p for p in spec["points"] if p.get("label") == letter]
    check(len(marked) == 1, f"the point marked {letter} must appear exactly once")
    at = tuple(marked[0]["at"])
    holders = [c for c in spec["curves"] if any(tuple(p) == at for p in _samples(c))]
    check(len(holders) == 1, f"the point marked {letter} must lie on exactly one graph")
    return holders[0]


def _read_spec(spec: dict, text: str) -> str:
    """Re-derive the key from the figure and the prompt's selector alone."""
    curves = spec["curves"]
    if len(curves) == 1:
        chosen = curves[0]
    else:
        letters = _THROUGH_RE.findall(text)
        if letters:
            check(len(letters) == 1, "the prompt must name one point")
            chosen = _curve_with_point(spec, letters[0])
        else:
            never = any(w in text for w in _NEVER_WORDS)
            twice = any(w in text for w in _TWICE_WORDS)
            check(never != twice, "the prompt must select the graph by exactly one axis property")
            low = {id(c): _corner_of_curve(c)[1] for c in curves}
            wanted = [c for c in curves if (low[id(c)] > 0) == never]
            check(len(wanted) == 1, "exactly one graph must have the stated axis property")
            chosen = wanted[0]
    return _fmt(_corner_of_curve(chosen))


def read_pv1d_abs_vertex(question) -> str:
    """The registered plausibility reader: figure + prompt only, never the key."""
    return _read_spec(question.figure, question.text)


figures.register_reader(TOPIC, read_pv1d_abs_vertex)


# ------------------------------------------------------------------ the prompts


def _pick(rng, pool):
    return pool[rng.randrange(len(pool))]


_EASY_INTRO = (
    "The graph of y = |x - h| + k is drawn on the grid.",
    "A V-shaped graph of the form y = |x - h| + k is shown.",
    "The grid shows the graph of y = |x - h| + k.",
    "Here is the graph of an absolute value function, y = |x - h| + k.",
)
_EASY_MARK = (
    "The point P marks its vertex.",
    "Its corner is marked with the point P.",
    "P is the point where its two arms meet.",
)
_EASY_ASK = (
    "Give the coordinates of the vertex as (h, k).",
    "State the vertex (h, k).",
    "Read the coordinates of P from the grid and give the vertex as (h, k).",
    "What is the vertex? Write it as (h, k).",
)

_MEDIUM_INTRO = _EASY_INTRO
_MEDIUM_SELECT = (
    "Its two straight arms meet at a sharp corner.",
    "Find the point where the two arms of the V meet.",
    "The V has one corner, where its two arms join.",
)
_MEDIUM_ASK = (
    "Give the coordinates of the vertex as (h, k).",
    "State the vertex (h, k).",
    "Which point is the vertex? Write it as (h, k).",
    "Read the vertex (h, k) from the grid.",
)

_HARD_INTRO = (
    "Two graphs of the form y = |x - h| + k are drawn on the same grid.",
    "The grid shows two V-shaped graphs, each of the form y = |x - h| + k.",
    "Two different graphs of y = |x - h| + k are shown, with one point marked on each.",
    "A pair of V-shaped graphs, both of the form y = |x - h| + k, is drawn.",
)
_HARD_SELECT = (
    "Consider the graph through the point marked {L}.",
    "Look at the graph through the point marked {L}, not the other one.",
    "Focus on the graph through the point marked {L}.",
)
_HARD_ASK = (
    "Give its vertex as (h, k).",
    "State the coordinates of its vertex, (h, k).",
    "Which point is its vertex? Write it as (h, k).",
    "Read the vertex (h, k) of that graph from the grid.",
)

_EXAM_INTRO = (
    "Two graphs of the form y = |x - h| + k are drawn on the same grid.",
    "The grid shows two V-shaped graphs, each of the form y = |x - h| + k.",
    "Two different graphs of y = |x - h| + k are shown.",
    "A pair of V-shaped graphs, both of the form y = |x - h| + k, is drawn.",
)
_EXAM_NEVER = (
    "Exactly one of them never meets the x-axis.",
    "One of them stays above the x-axis for every value of x.",
    "One of the two graphs does not touch or cross the x-axis.",
)
_EXAM_TWICE = (
    "Exactly one of them crosses the x-axis at two points.",
    "One of them cuts the x-axis twice.",
    "One of the two graphs meets the x-axis at two different points.",
)
_EXAM_ASK = (
    "Give the vertex (h, k) of that graph.",
    "State the coordinates of its vertex, (h, k).",
    "Which point is the vertex of that graph? Write it as (h, k).",
    "Read the vertex (h, k) of that graph from the grid.",
)


# --------------------------------------------------------------- answer + options


def _options(h: int, k: int, other=None):
    """(distractor texts, option_notes). Every candidate is a different pair from the
    key and from every other candidate, proved with check_pairwise_distinct."""
    named = []
    if other is not None:
        named.append((tuple(other), "other-graph-vertex"))
    named += [((k, h), "swapped-coordinates"), ((-h, k), "sign-of-h"), ((h, -k), "sign-of-k")]
    if k < 0:
        named.append(((h + abs(k), 0), "x-intercept"))
    named.append(((0, abs(h) + k), "y-intercept"))
    unique, seen = [], {(h, k)}
    for pair, note in named:  # the other graph's vertex may coincide with a misreading
        if pair not in seen:
            seen.add(pair)
            unique.append((pair, note))
    check_pairwise_distinct(
        (h, k), *[p for p, _ in unique], msg="a distractor collides with the key or another distractor"
    )
    answer_text = _fmt((h, k))
    texts = pick_distractors(answer_text, [_fmt(p) for p, _ in unique], n=6, tolerance=GRADER_TOLERANCE)
    notes = {_fmt(p): note for p, note in unique}
    return texts, {t: notes[t] for t in texts}


def _compatible(a, b) -> bool:
    """Two V graphs that read cleanly together: no shared arm line, neither corner on
    the other graph, and corners not on top of one another."""
    (h1, k1), (h2, k2) = a, b
    if max(abs(h1 - h2), abs(k1 - k2)) < 2:
        return False
    if k1 - h1 == k2 - h2 or k1 + h1 == k2 + h2:
        return False
    return k1 != abs(h1 - h2) + k2 and k2 != abs(h2 - h1) + k1


def _arm_point(rng, vertex):
    h, k = vertex
    d = rng.randint(1, 3)
    side = rng.choice((-1, 1))
    return (h + side * d, k + d)


def _on_graph(point, vertex) -> bool:
    return point[1] == abs(point[0] - vertex[0]) + vertex[1]


def _finish(rng, text, answer, other, spec_curves, points, alt):
    h, k = answer
    spec = figures.cartesian(
        x_range=(_LO, _HI),
        y_range=(_LO, _HI),
        grid=1,
        equal_scale=True,
        alt=alt,
        points=points,
        curves=spec_curves,
    )
    answer_text = _fmt(answer)
    # Re-derive the key from the drawn figure two ways; a corrupted reader fails the build.
    check(_read_spec(spec, text) == answer_text, "the figure does not read back to the key")
    for curve in spec["curves"]:
        check(_corner_from_arms(curve) == _corner_of_curve(curve), "arm route and corner route disagree")
    distractors, _notes = _options(
        h, k, other
    )  # the notes name each distractor's misconception (tests); Question.option_notes stays unset
    figures.assert_no_leak(spec, [answer_text, *distractors])
    return dict(
        text=text,
        answer={"x": h, "y": k},
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"The vertex is where the two arms meet: read across to x = {h} and up to y = {k}, "
            f"so the vertex is {answer_text}."
        ],
        distractors=distractors,
        figure=spec,
    )


# ------------------------------------------------------------------------ tiers


def _single(rng, marked: bool):
    h, k = _pick(rng, _CELLS)
    if marked:
        text = " ".join((_pick(rng, _EASY_INTRO), _pick(rng, _EASY_MARK), _pick(rng, _EASY_ASK)))
        points = [{"at": [h, k], "label": "P"}]
    else:
        text = " ".join((_pick(rng, _MEDIUM_INTRO), _pick(rng, _MEDIUM_SELECT), _pick(rng, _MEDIUM_ASK)))
        points = []
    return _finish(rng, text, (h, k), None, [_curve(h, k)], points, _ALT_ONE)


def _hard(rng):
    letter = rng.choice(("P", "Q"))
    other_letter = "Q" if letter == "P" else "P"
    for _ in range(400):
        a, b = _pick(rng, _CELLS), _pick(rng, _CELLS)
        if not _compatible(a, b):
            continue
        pa, pb = _arm_point(rng, a), _arm_point(rng, b)
        if _on_graph(pa, b) or _on_graph(pb, a):
            continue
        if max(abs(pa[0] - pb[0]), abs(pa[1] - pb[1])) < 2:
            continue
        break
    else:
        raise AssertionError("pv1d_abs_vertex hard: no clean pair of graphs in 400 draws")
    select = _pick(rng, _HARD_SELECT).format(L=letter)
    text = " ".join((_pick(rng, _HARD_INTRO), select, _pick(rng, _HARD_ASK)))
    curves = [_curve(*a), _curve(*b)]
    points = [{"at": list(pa), "label": letter}, {"at": list(pb), "label": other_letter}]
    if rng.random() < 0.5:
        curves.reverse()
    return _finish(rng, text, a, b, curves, points, _ALT_TWO)


def _exam(rng):
    never = rng.random() < 0.5
    for _ in range(400):
        a = _pick(rng, _POS_CELLS if never else _NEG_CELLS)
        b = _pick(rng, _NEG_CELLS if never else _POS_CELLS)
        if _compatible(a, b):
            break
    else:
        raise AssertionError("pv1d_abs_vertex exam: no clean pair of graphs in 400 draws")
    select = _pick(rng, _EXAM_NEVER if never else _EXAM_TWICE)
    text = " ".join((_pick(rng, _EXAM_INTRO), select, _pick(rng, _EXAM_ASK)))
    curves = [_curve(*a), _curve(*b)]
    if rng.random() < 0.5:
        curves.reverse()
    return _finish(rng, text, a, b, curves, [], _ALT_TWO)


_PRESETS = {
    "easy": {"variant": "marked"},
    "medium": {"variant": "unmarked"},
    "hard": {"variant": "two_graphs_lettered"},
    "exam": {"variant": "two_graphs_axis"},
}


@generator(TOPIC, level=12, presets=_PRESETS)
def pv1d_abs_vertex(rng, variant="marked"):
    if variant == "marked":
        return _single(rng, marked=True)
    if variant == "unmarked":
        return _single(rng, marked=False)
    if variant == "two_graphs_lettered":
        return _hard(rng)
    if variant == "two_graphs_axis":
        return _exam(rng)
    raise ValueError(f"unknown variant: {variant!r}")
