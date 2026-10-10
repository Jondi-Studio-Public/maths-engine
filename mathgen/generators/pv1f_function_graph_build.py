"""Generates `pv1f_graph_is_function` questions: decide from a drawn relation whether it is the graph of a
function.

``pv1f_graph_is_function`` (senior-understand-the-concept-of-the-graph-of-a, ACMMM024): a
relation is drawn on an integer grid and the learner picks the one correct statement. EVERY
option pairs a verdict with a witness line,

    "It is a function: the line x = a meets the graph once."
    "It is not a function: the line x = a meets the graph more than once."

so the key is never the only option of its shape: every question offers exactly two options
of each verdict. The answer class (function / not a function) is drawn FIRST and uniformly.
For a non-function the key's line meets the relation at two (or more) points and every
distractor's line meets it at most once; for a function the key's line meets it at one point,
one distractor claims the same of a line that meets nothing, and the other two claim a
failure at lines that meet it once. No line is ever named by two options, so a repeated line cannot point at the key. The lure
of the tier (the one-dot jump, the touching line, the dot's line) is offered as a false
"It is not a function" claim for a function and as a false "It is not a function" claim at the
continuous join / touching line for a non-function.
Tiers escalate by the decision made:

- easy:   one clean relation (segment or arc, circle or sideways parabola); the witness line
          is read off the picture;
- medium: a graph of line segments with open and closed end dots; at the join x = j the
          verdict turns on whether BOTH facing end dots are filled;
- hard:   curved relations whose lures are lines that only touch the graph once (the
          tangent to a circle or sideways parabola, the steep middle of a cubic);
- exam:   a curve plus one separate dot; the verdict turns on whether the dot shares an
          x-value with a different point of the graph.

Every curve is sampled with step 1/8; filled and open dots are the figure's own
``points`` and ``segments`` end flags. The topic registers a figure reader that parses each
option's line from the option text and counts, from ``question.figure`` alone, how many
points of the drawn graph lie on it.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction

from mathgen import figures
from mathgen.core import check, generator, mcq_options, pick_distractors

_LO, _HI = -10, 10
_TIERS = ("easy", "medium", "hard", "exam")
_PRESETS = {t: {"tier": t} for t in _TIERS}
_EPS = 1e-9


def _pick(rng, pool):
    return pool[rng.randrange(len(pool))]


def _fnum(x) -> float:
    return float(x)


# ----------------------------------------------------------------------- relations
# Every relation is a dict: curves (list of pieces), segments, points, meets(a) -> int,
# is_function, and optional named lure lines. Coordinates are integers wherever the
# learner must read them.


def _dot(x, y, open_=False) -> dict:
    d = {"at": [x, y]}
    if open_:
        d["open"] = True
    return d


def _sample_function(f, x1, x2) -> list:
    """The polyline of y = f(x) on [x1, x2] in steps of 1/8 (every integer is a sample)."""
    n = (x2 - x1) * 8 + 1
    return [[_clean(x1 + i / 8), _clean(f(Fraction(x1) + Fraction(i, 8)))] for i in range(n)]


def _clean(v):
    r = round(float(v), 6)
    return int(r) if r == int(r) else r


def _rel_segment(rng) -> dict:
    x1 = rng.randint(-8, 2)
    x2 = x1 + rng.randint(4, 10)
    x2 = min(x2, 8)
    for _ in range(100):
        m = _pick(rng, (-3, -2, -1, 1, 2, 3))
        y1 = rng.randint(-8, 8)
        y2 = y1 + m * (x2 - x1)
        if abs(y2) <= 8:
            break
    else:
        m, y1, y2 = 1, -4, -4 + (x2 - x1)
        check(abs(y2) <= 8)
    return dict(
        name="segment",
        is_function=True,
        curves=[],
        points=[],
        segments=[{"from": [x1, y1], "to": [x2, y2], "open_start": False, "open_end": False}],
        meets=lambda a: 1 if x1 <= a <= x2 else 0,
        domain=(x1, x2),
        lures=[],
        at=lambda x: Fraction(y1) + m * (x - x1),
    )


def _rel_arc(rng, *, hard: bool = False) -> dict:
    for _ in range(200):
        s = _pick(rng, (-1, 1))
        h = rng.randint(-5, 5)
        c = rng.randint(-6, 6)
        x1 = rng.randint(-8, 2)
        x2 = min(8, x1 + rng.randint(5, 11))
        ys = [s * (x - h) ** 2 + c for x in range(x1, x2 + 1)]
        if x1 < h < x2 and max(abs(y) for y in ys) <= 8 and max(ys) - min(ys) >= 3:
            break
    else:
        s, h, c, x1, x2 = 1, 0, -4, -3, 3
    f = lambda x: s * (x - h) ** 2 + c  # noqa: E731
    pts = [_dot(x1, f(x1)), _dot(x2, f(x2))]
    return dict(
        name="arc",
        is_function=True,
        curves=[{"pieces": [_sample_function(f, x1, x2)]}],
        segments=[],
        points=pts,
        meets=lambda a: 1 if x1 <= a <= x2 else 0,
        domain=(x1, x2),
        lures=[h],
        at=lambda x: Fraction(f(x)),
    )


def _rel_cubic(rng) -> dict:
    for _ in range(200):
        a = _pick(rng, (-1, 1))
        b = rng.randint(-5, 5)
        c = rng.randint(-4, 4)
        wl, wr = rng.randint(1, 2), rng.randint(1, 2)
        x1, x2 = b - wl, b + wr
        f = lambda x, a=a, b=b, c=c: a * (x - b) ** 3 + c  # noqa: E731
        if abs(f(x1)) <= 9 and abs(f(x2)) <= 9 and x2 - x1 >= 3 and -9 <= x1 and x2 <= 9:
            break
    else:
        a, b, c, x1, x2 = 1, 0, 0, -2, 2
        f = lambda x: x**3  # noqa: E731
    return dict(
        name="cubic",
        is_function=True,
        curves=[{"pieces": [_sample_function(f, x1, x2)]}],
        segments=[],
        points=[_dot(x1, f(x1)), _dot(x2, f(x2))],
        meets=lambda a_: 1 if x1 <= a_ <= x2 else 0,
        domain=(x1, x2),
        lures=[b],
        at=lambda x: Fraction(f(x)),
    )


def _circle_pieces(cx, cy, r) -> list:
    """One closed polyline: 160 angle samples plus every lattice point of the circle."""
    pts = {}
    for i in range(160):
        t = 2 * math.pi * i / 160
        pts[t] = (cx + r * math.cos(t), cy + r * math.sin(t))
    for dx in range(-r, r + 1):
        dy2 = r * r - dx * dx
        dy = math.isqrt(dy2)
        if dy * dy == dy2:
            for sy in (dy, -dy) if dy else (0,):
                t = math.atan2(sy, dx) % (2 * math.pi)
                pts[t] = (cx + dx, cy + sy)
    ordered = [list(map(_clean, pts[t])) for t in sorted(pts)]
    ordered.append(ordered[0])
    return [ordered]


def _rel_circle(rng) -> dict:
    r = _pick(rng, (2, 3, 4, 5, 5))
    cx = rng.randint(-9 + r, 9 - r)
    cy = rng.randint(-9 + r, 9 - r)

    def meets(a):
        d = abs(a - cx)
        return 2 if d < r else (1 if d == r else 0)

    return dict(
        name="circle",
        is_function=False,
        curves=[{"pieces": _circle_pieces(cx, cy, r)}],
        segments=[],
        points=[],
        meets=meets,
        domain=(cx - r, cx + r),
        lures=[cx - r, cx + r],
        lattice=[
            a for a in range(cx - r + 1, cx + r) if math.isqrt(r * r - (a - cx) ** 2) ** 2 == r * r - (a - cx) ** 2
        ],
    )


def _rel_sideways(rng) -> dict:
    t = _pick(rng, (2, 3, 4))
    k = _pick(rng, (-1, 1))
    lo = -9 + (t * t if k < 0 else 0)
    hi = 9 - (t * t if k > 0 else 0)
    c = rng.randint(lo, hi)
    pieces = [[_clean(c + k * (Fraction(i, 8)) ** 2), _clean(Fraction(i, 8))] for i in range(-8 * t, 8 * t + 1)]
    ends = [c + k * t * t, c + k * t * t]

    def meets(a):
        u = Fraction(a - c, k)
        if u < 0 or u > t * t:
            return 0
        return 1 if u == 0 else 2

    return dict(
        name="sideways",
        is_function=False,
        curves=[{"pieces": [pieces]}],
        segments=[],
        points=[_dot(ends[0], t), _dot(ends[1], -t)],
        meets=meets,
        domain=(min(c, c + k * t * t), max(c, c + k * t * t)),
        lures=[c],
        lattice=[c + k * s * s for s in range(1, t + 1)],
    )


def _rel_joins(rng, *, closed: bool) -> dict:
    """Three line segments. At x = j the graph jumps: with both facing end dots filled
    (``closed``) it is not a function, with exactly one filled it is. At x = j2 the segments
    meet in one point (no dots)."""
    for _ in range(300):
        xs = sorted(rng.sample(range(-8, 9), 4))
        if min(b - a for a, b in zip(xs, xs[1:])) < 2:
            continue
        jump_first = rng.randrange(2) == 0
        j, j2 = (xs[1], xs[2]) if jump_first else (xs[2], xs[1])
        ya, yb, ym = (rng.randint(-8, 8) for _ in range(3))
        yl, yr = rng.randint(-8, 8), rng.randint(-8, 8)
        if jump_first:  # S1 ends at the jump, S2 starts after it and runs to the smooth join
            ys = [(ya, yl), (yr, ym), (ym, yb)]
        else:  # S1 runs to the smooth join, S2 ends at the jump, S3 starts after it
            ys = [(ya, ym), (ym, yl), (yr, yb)]
        if abs(yl - yr) >= 2 and all(abs(p - q) >= 1 for p, q in ys):
            break
    else:
        raise AssertionError("no join layout found")
    cuts = list(zip(xs, xs[1:]))
    open_left = open_right = False
    if not closed:
        open_left, open_right = (True, False) if rng.randrange(2) else (False, True)
    jump_seg_end = 0 if jump_first else 1  # the segment whose END is at the jump
    segments = []
    for idx, ((xa, xb), (ya_, yb_)) in enumerate(zip(cuts, ys)):
        seg = {"from": [xa, ya_], "to": [xb, yb_]}
        if idx == 0:
            seg["open_start"] = False
        if idx == 2:
            seg["open_end"] = False
        if idx == jump_seg_end:
            seg["open_end"] = open_left
        if idx == jump_seg_end + 1:
            seg["open_start"] = open_right
        segments.append(seg)
    lo, hi = xs[0], xs[-1]

    def meets(a):
        if a == j:
            return 2 if closed else 1
        return 1 if lo <= a <= hi else 0

    return dict(
        name="joins",
        is_function=not closed,
        curves=[],
        points=[],
        segments=segments,
        meets=meets,
        domain=(lo, hi),
        lures=[j, j2],
        jump=j,
        smooth=j2,
    )


def _rel_stray_dot(rng, *, variant: str) -> dict:
    """A function graph plus one separate filled dot. ``variant``: 'above' (the dot sits over
    a different point of the graph: not a function), 'on' (on the graph itself) or 'outside'
    (at an x the graph does not reach): the last two leave a function."""
    base = _rel_segment(rng) if rng.randrange(2) else _rel_arc(rng)
    x1, x2 = base["domain"]
    if variant == "outside":
        j = _pick(rng, [x for x in range(-9, 10) if x < x1 or x > x2])
        yd = rng.randint(-8, 8)
    else:
        j = _pick(rng, list(range(x1 + 1, x2)))
        yc = int(base["at"](j))
        if variant == "on":
            yd = yc
        else:
            yd = _pick(rng, [yc + d for d in (-6, -5, -4, -3, -2, 2, 3, 4, 5, 6) if abs(yc + d) <= 9])
    meets_base = base["meets"]

    def meets(a):
        n = meets_base(a)
        return n if (a != j or variant == "on") else n + 1

    out = dict(base)
    out.update(
        name="stray_" + variant,
        is_function=variant != "above",
        points=list(base["points"]) + [_dot(j, yd)],
        meets=meets,
        jump=j,
        lures=[j],
        dot=(j, yd),
    )
    return out


# ----------------------------------------------------------------------- the reader


def _counts_at(fig: dict, a: float) -> int:
    """How many distinct points of the drawn graph lie on the line x = a."""
    ys = set()
    for curve in fig["curves"]:
        for piece in curve["pieces"]:
            for (x0, y0), (x1, y1) in zip(piece, piece[1:]):
                lo, hi = min(x0, x1), max(x0, x1)
                if lo - _EPS <= a <= hi + _EPS:
                    y = y0 if abs(x1 - x0) < _EPS else y0 + (y1 - y0) * (a - x0) / (x1 - x0)
                    ys.add(round(y, 5))
    excluded = set()
    for s in fig["segments"]:
        (x0, y0), (x1, y1) = s["from"], s["to"]
        lo, hi = min(x0, x1), max(x0, x1)
        if lo - _EPS <= a <= hi + _EPS:
            y = y0 if abs(x1 - x0) < _EPS else y0 + (y1 - y0) * (a - x0) / (x1 - x0)
            y = round(y, 5)
            if abs(a - x0) < _EPS and s.get("open_start"):
                excluded.add(y)
                continue
            if abs(a - x1) < _EPS and s.get("open_end"):
                excluded.add(y)
                continue
            ys.add(y)
    for p in fig["points"]:
        if abs(p["at"][0] - a) < _EPS:
            y = round(p["at"][1], 5)
            if p.get("open"):
                excluded.add(y)
            else:
                ys.add(y)
    # an open dot removes its own point from a segment or curve, unless a filled dot or a
    # filled end of another piece restores it
    closed_ends = set()
    for s in fig["segments"]:
        for end, flag in ((s["from"], s.get("open_start")), (s["to"], s.get("open_end"))):
            if abs(end[0] - a) < _EPS and flag is False:
                closed_ends.add(round(end[1], 5))
    for p in fig["points"]:
        if abs(p["at"][0] - a) < _EPS and not p.get("open"):
            closed_ends.add(round(p["at"][1], 5))
    return len(ys - (excluded - closed_ends))


def _graph_is_function(fig: dict) -> bool:
    return all(_counts_at(fig, x / 2) <= 1 for x in range(-2 * _HI, 2 * _HI + 1))


_OPT_RE = re.compile(r"^It is (not )?a function: the line x = (-?\d+) meets the graph (once|more than once)\.$")


def _statement_true(fig: dict, option: str, is_function: bool) -> bool:
    m = _OPT_RE.match(option)
    check(m is not None, f"unparseable option {option!r}")
    verdict_not, a, how = m.group(1), int(m.group(2)), m.group(3)
    n = _counts_at(fig, a)
    if verdict_not:
        return how == "more than once" and n >= 2
    return how == "once" and is_function and n == 1


def _read_from_figure(fig: dict, options: list) -> str:
    is_function = _graph_is_function(fig)
    true = [o for o in options if _statement_true(fig, o, is_function)]
    check(len(true) == 1, f"{len(true)} of the options are true for the figure")
    return true[0]


def _read_function(question) -> str:
    # Reader contract: return the option TEXT, not its index.
    return _read_from_figure(question.figure, mcq_options(question))


figures.register_reader("pv1f_graph_is_function", _read_function)

# ----------------------------------------------------------------------- options


def _text(verdict: str, a: int) -> str:
    if verdict == "F":
        return f"It is a function: the line x = {a} meets the graph once."
    return f"It is not a function: the line x = {a} meets the graph more than once."


def _lines(rel: dict, wanted) -> list:
    return [a for a in range(-9, 10) if wanted(rel["meets"](a))]


def _build_options(rng, rel: dict, cls: str, key_a: int, forced=()):
    """(key text, [3 distractor texts]) with exactly two options of each verdict.

    ``forced`` are (verdict, a) distractors the tier insists on; each is checked false.
    A statement is true when its verdict matches the relation and its witness line meets the
    graph as claimed."""
    meets = rel["meets"]

    def true(verdict, a):
        return meets(a) >= 2 if verdict == "N" else (rel["is_function"] and meets(a) == 1)

    key = (cls, key_a)
    check(true(*key), "the key is not true for the relation")
    chosen = [key]
    for opt in forced:
        check(not true(*opt), f"forced distractor {opt} is true")
        if opt not in chosen:
            chosen.append(opt)
    at_most_once = _lines(rel, lambda n: n <= 1)
    zero = _lines(rel, lambda n: n == 0)
    for _ in range(8):
        n_count = sum(1 for v, _a in chosen if v == "N")
        f_count = len(chosen) - n_count
        if n_count >= 2 and f_count >= 2:
            break
        fresh = {a for _v, a in chosen}  # a line already named by an option
        if n_count < 2:
            pool = [a for a in at_most_once if ("N", a) not in chosen]
            pool = [a for a in pool if a not in fresh] or pool
            check(bool(pool), "no line left for an 'is not a function' distractor")
            opt = ("N", _pick(rng, pool))
        else:
            # a function: the extra 'is a function' option claims the same of a line that
            # meets nothing; a non-function: any line meeting it at most once
            pool = [a for a in (zero if cls == "F" else at_most_once) if ("F", a) not in chosen]
            pool = [a for a in pool if a not in fresh] or pool
            check(bool(pool), "no line left for an 'is a function' distractor")
            opt = ("F", _pick(rng, pool))
        check(not true(*opt), "a distractor is true")
        chosen.append(opt)
    check(len(chosen) == 4 and len(set(chosen)) == 4, "need four distinct options")
    check(sum(1 for v, _a in chosen if v == "N") == 2, "two options of each verdict")
    return _text(*key), [_text(*o) for o in chosen[1:]]


# ----------------------------------------------------------------------- the prompts

_POOLS = {
    "easy": (
        (
            "A graph is drawn on the grid.",
            "The diagram shows a graph drawn on a grid.",
            "A relation is graphed below.",
            "Here is the graph of a relation.",
        ),
        (
            "Each grid square is 1 unit wide. Use vertical lines to decide whether it is the graph of a function.",
            "Think about vertical lines drawn through the graph.",
            "Test the graph with vertical lines x = a.",
        ),
        (
            "Which statement is correct?",
            "Choose the one correct statement.",
            "Pick the statement that is true for this graph.",
        ),
    ),
    "medium": (
        (
            "The graph is made of straight segments. A filled dot belongs to the graph and an open dot does not.",
            "Straight line segments make up this graph; a filled end dot is on the graph, an open end dot is not.",
            "This graph has several segments. Filled dots are points of the graph, open dots are not.",
            "The segments below form a graph in which a filled dot is included and an open dot is left out.",
        ),
        (
            "Think carefully about where two segments meet.",
            "Look closely at the place where the segments come together.",
            "Consider each place where one segment ends and another begins.",
        ),
        (
            "Which statement about the graph is correct?",
            "Select the one true statement.",
            "Which of these statements is true?",
        ),
    ),
    "hard": (
        (
            "A curved graph is shown on the grid.",
            "The grid shows a curve.",
            "Below is a curved graph drawn on a grid.",
            "A curve is drawn on the grid shown.",
        ),
        (
            "A vertical line that only touches the graph meets it at one point.",
            "Remember that a vertical line which just touches the curve meets it once, not twice.",
            "A vertical line that grazes the curve meets it at a single point.",
        ),
        (
            "Which statement is correct?",
            "Decide which one statement is true.",
            "Which of the statements below is true for this graph?",
        ),
    ),
    "exam": (
        (
            "The graph consists of a line or curve together with one separate filled dot.",
            "A line or curve and a single extra filled dot are drawn on the grid; every filled dot belongs to the graph.",
            "This graph is made of one line or curve and one more filled dot, and all of it counts as the graph.",
            "On the grid are a line or curve and one separate filled dot, all of which is part of the graph.",
        ),
        ("Include the dot when you decide.", "The dot is part of the graph.", "Count the dot as part of the graph."),
        ("Which statement is correct?", "Choose the one true statement.", "Which of these statements is true?"),
    ),
}

_WORKING = {
    "F": "No vertical line meets the graph more than once, so it is the graph of a function. Check the line x = {a}: it meets the graph once.",
    "N": "Check the line x = {a}: it meets the graph at more than one point, so this is not the graph of a function.",
}


# ----------------------------------------------------------------------- the draw


def _once_lines(rel: dict, avoid) -> list:
    """Lines meeting the graph exactly once that are not in ``avoid``."""
    return [a for a in range(-9, 10) if rel["meets"](a) == 1 and a not in avoid]


def _setup(rng, tier: str, cls: str):
    """(relation, key_a, forced distractors). No line is ever named by two options: a
    line named twice would tell the learner where the key is without reading the figure."""
    if tier == "easy":
        rel = (
            _rel_circle(rng)
            if cls == "N" and rng.randrange(2)
            else (_rel_sideways(rng) if cls == "N" else (_rel_segment(rng) if rng.randrange(2) else _rel_arc(rng)))
        )
        if cls == "N":
            key_a = _pick(rng, rel["lattice"])
        else:
            key_a = _pick(rng, _once_lines(rel, ()))
        return rel, key_a, ()
    if tier == "medium":
        rel = _rel_joins(rng, closed=cls == "N")
        j, j2 = rel["jump"], rel["smooth"]
        if cls == "N":
            return rel, j, (("N", j2),)  # the continuous join is the lure: it meets once
        # a function: the key is some other line; "not a function" at the one-dot jump and at
        # the continuous join are the lures
        return rel, _pick(rng, _once_lines(rel, (j, j2))), (("N", j), ("N", j2))
    if tier == "hard":
        if cls == "N":
            rel = _rel_circle(rng) if rng.randrange(2) else _rel_sideways(rng)
            key_a = _pick(rng, rel["lattice"])
            lure = _pick(rng, [a for a in rel["lures"] if rel["meets"](a) == 1])
            return rel, key_a, (("N", lure),)
        rel = _rel_cubic(rng) if rng.randrange(2) else _rel_arc(rng)
        lure = rel["lures"][0]
        return rel, _pick(rng, _once_lines(rel, (lure,))), (("N", lure),)
    variant = "above" if cls == "N" else _pick(rng, ("on", "outside"))
    rel = _rel_stray_dot(rng, variant=variant)
    j = rel["jump"]
    if cls == "N":
        return rel, j, ()
    # a function: the key is another line; "not a function" at the dot's line is the lure
    return rel, _pick(rng, _once_lines(rel, (j,))), (("N", j),)


def _build(rng, tier: str):
    cls = _pick(rng, ("F", "N"))  # the answer class FIRST, uniform
    for _ in range(40):
        try:
            rel, key_a, forced = _setup(rng, tier, cls)
            key, distractors = _build_options(rng, rel, cls, key_a, forced)
            break
        except AssertionError:
            continue
        except Exception as exc:  # noqa: BLE001 - a failed layout draw is retried, VerificationError included
            if type(exc).__name__ != "VerificationError":
                raise
            continue
    else:
        check(False, "no layout found")
    alt = "a graph on a coordinate grid"
    fig = figures.cartesian(
        x_range=(_LO, _HI),
        y_range=(_LO, _HI),
        alt=alt,
        grid=1,
        equal_scale=True,
        curves=rel["curves"],
        segments=rel["segments"],
        points=rel["points"],
    )
    # the construction's own counts must agree with what the picture shows at every option's line
    for opt in [key, *distractors]:
        a = int(_OPT_RE.match(opt).group(2))
        check(_counts_at(fig, a) == rel["meets"](a), f"figure and construction disagree at x = {a}")
    check(_graph_is_function(fig) == rel["is_function"], "figure and construction disagree on the verdict")
    open_, tip, ask = _POOLS[tier]
    text = " ".join((_pick(rng, open_), _pick(rng, tip), _pick(rng, ask)))
    return fig, text, key, distractors, cls, key_a


@generator("pv1f_graph_is_function", level=13, presets=_PRESETS)
def pv1f_graph_is_function(rng, tier="easy"):
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    fig, text, key, distractors, cls, key_a = _build(rng, tier)
    check(len({key, *distractors}) == 4, "four distinct options")
    picked = pick_distractors(key, distractors, n=3)
    check(len(picked) == 3, "the mcq must offer three distractors")
    figures.assert_no_leak(fig, [key, *picked])
    return dict(
        text=text,
        answer=key,
        answer_text=key,
        answer_latex=f"plain:{key}",
        distractors=picked,
        working=[_WORKING[cls].format(a=key_a)],
        figure=fig,
    )
