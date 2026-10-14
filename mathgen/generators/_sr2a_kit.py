"""Shared pieces for the ``sr2a_*`` generators (General Mathematics networks, bivariate data and distributions).

Not a generator module (the leading underscore keeps ``mathgen.generators`` from importing it as one).

* ``number_distractors`` and the ``*_part`` / ``single`` builders turn ``(value, misconception id)`` pairs into
  tagged ``Distractor``s that the real grader has been asked about and rejected, and into the dicts a generator
  returns for a single question or for one part of an exam chain;
* the small graph helpers (``edges_text``, ``random_connected``, ``components`` ...) serve the three network skills.
"""

from __future__ import annotations

from itertools import combinations
from statistics import correlation

from ..answers import Distractor
from ..core import (
    GRADER_TOLERANCE,
    Question,
    attempts,
    check,
    latex_answer,
    pick_distractors,
    stated_precision_tolerance,
)

LETTERS = "ABCDEFGHIJ"


# ---------------------------------------------------------------- numbers and graders


def num_text(value, dp: int = 0) -> str:
    """``value`` written to ``dp`` decimal places ("4", "2.5", "-0.30")."""
    if dp == 0:
        return str(int(round(value)))
    return f"{float(value):.{dp}f}"


def _stub(kind: str, answer, answer_text: str, text: str = "") -> Question:
    return Question(
        topic="sr2a-check",
        level=1,
        seed=0,
        text=text,
        latex=text,
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        answer_kind=kind,
    )


def rejected(kind: str, answer, answer_text: str, reply: str) -> bool:
    """True when the live grader of ``kind`` marks ``reply`` wrong for the keyed ``answer``."""
    from ..grading import grade

    return not grade(kind, _stub(kind, answer, answer_text), reply).correct


def number_distractors(key, wrong, *, dp: int = 0, lo=None, want: int = 3):
    """Up to ``want`` tagged number distractors for the numeric answer ``key`` shown to ``dp`` places.

    ``wrong`` lists ``(value, misconception id)`` in priority order. A value is dropped when it is below ``lo``,
    is the key itself, repeats an earlier option, or the real ``number`` grader would accept it. Returns the list
    (possibly shorter than ``want``; callers redraw when it is)."""
    key_text = num_text(key, dp)
    tolerance = stated_precision_tolerance(dp) if dp else 0
    cands = []
    for value, tag in wrong:
        if value is None or (lo is not None and value < lo):
            continue
        if dp == 0 and value != int(value):
            continue
        cands.append(Distractor(num_text(value, dp), tag, value if dp else int(value)))
    picked = pick_distractors(key_text, cands, want + 2, tolerance=tolerance)
    out = [d for d in picked if rejected("number", float(key) if dp else int(key), key_text, d.text)]
    return out[:want]


# ---------------------------------------------------------------- a single question


def single(
    text,
    key,
    working,
    wrong,
    *,
    dp: int = 0,
    lo=None,
    figure=None,
    want: int = 3,
):
    """The dict a generator returns for one ``number`` question, or None when too few distractors survive."""
    dis = number_distractors(key, wrong, dp=dp, lo=lo, want=want)
    if len(dis) < want:
        return None
    key_text = num_text(key, dp)
    out = dict(
        text=text,
        answer_kind="number",
        answer=float(key) if dp else int(key),
        answer_text=key_text,
        answer_latex=latex_answer(key_text),
        working=working,
        distractors=dis,
    )
    if figure is not None:
        out["figure"] = figure
    return out


def listed(
    text,
    items,
    working,
    wrong,
    *,
    kind: str = "ordered_list",
    figure=None,
    want: int = 3,
):
    """A question whose answer is a list of words (``ordered_list``): ``wrong`` is ``(list, misconception id)``."""
    answer_text = ", ".join(items)
    cands = []
    for value, tag in wrong:
        t = ", ".join(value)
        if t != answer_text and rejected(kind, list(items), answer_text, t):
            cands.append(Distractor(t, tag, list(value)))
    dis = pick_distractors(answer_text, cands, want)
    if len(dis) < want:
        return None
    out = dict(
        text=text,
        answer_kind=kind,
        answer=list(items),
        answer_text=answer_text,
        answer_latex="plain:" + answer_text,
        working=working,
        distractors=dis,
    )
    if figure is not None:
        out["figure"] = figure
    return out


# ---------------------------------------------------------------- parts of an exam chain


def number_part(prompt, key, marks, working, wrong, *, follow=None, dp: int = 0, lo=None, sensible=(-1, None)):
    """One ``number`` part. None when fewer than three distractors survive (the caller draws again)."""
    dis = number_distractors(key, wrong, dp=dp, lo=lo)
    if len(dis) < 3:
        return None
    key_text = num_text(key, dp)
    part = dict(
        prompt=prompt,
        answer_kind="number",
        answer=float(key) if dp else int(key),
        answer_text=key_text,
        answer_latex=latex_answer(key_text),
        marks=marks,
        sensible_range=sensible,
        working=working,
        distractors=dis,
    )
    if follow:
        part["follow_through"] = follow
    return part


def mcq_part(prompt, key, marks, working, wrong, *, want: int = 3):
    """One ``mcq`` part from ``(option text, misconception id)`` wrong options; None when too few survive."""
    cands = [Distractor(t, tag) for t, tag in wrong if t != key]
    dis = pick_distractors(key, cands, want)
    if len(dis) < want:
        return None
    return dict(
        prompt=prompt,
        answer_kind="mcq",
        answer=key,
        answer_text=key,
        answer_latex="plain:" + key,
        marks=marks,
        working=working,
        distractors=dis,
    )


def list_part(prompt, items, marks, working, wrong, *, want: int = 3):
    """One ``ordered_list`` part of words, from ``(list, misconception id)`` wrong answers."""
    q = listed(prompt, items, working, wrong, want=want)
    if q is None:
        return None
    return dict(
        prompt=prompt,
        answer_kind="ordered_list",
        answer=q["answer"],
        answer_text=q["answer_text"],
        answer_latex=q["answer_latex"],
        marks=marks,
        working=working,
        distractors=q["distractors"],
    )


def matrix_cells(rows) -> str:
    """The canonical spelling of a 0/1 matrix answer: ``[[1, 0], [0, 1]]``."""
    return "[" + ", ".join("[" + ", ".join(str(v) for v in row) + "]" for row in rows) + "]"


def matrix_distractors(rows, wrong, *, want: int = 3):
    """Tagged matrix distractors from ``(rows, misconception id)``; each is rejected by the matrix grader."""
    answer_text = matrix_cells(rows)
    out, seen = [], {answer_text}
    for value, tag in wrong:
        t = matrix_cells(value)
        if t in seen or not rejected("matrix", [list(r) for r in rows], answer_text, t):
            continue
        seen.add(t)
        out.append(Distractor(t, tag, [list(r) for r in value]))
        if len(out) == want:
            break
    return out


def matrix_single(text, rows, working, wrong, *, want: int = 3):
    """One ``matrix`` question, or None when fewer than ``want`` distractors survive."""
    dis = matrix_distractors(rows, wrong, want=want)
    if len(dis) < want:
        return None
    answer_text = matrix_cells(rows)
    return dict(
        text=text,
        answer_kind="matrix",
        answer=[list(r) for r in rows],
        answer_text=answer_text,
        answer_latex="plain:" + answer_text,
        working=working,
        distractors=dis,
    )


def matrix_part(prompt, rows, marks, working, wrong, *, want: int = 3):
    """One ``matrix`` part, or None when fewer than ``want`` distractors survive."""
    q = matrix_single(prompt, rows, working, wrong, want=want)
    if q is None:
        return None
    q.pop("text")
    return dict(prompt=prompt, marks=marks, **q)


# ---------------------------------------------------------------- bivariate data


def corr(xs, ys) -> float:
    """Pearson's r for two lists of numbers."""
    return correlation(list(map(float, xs)), list(map(float, ys)))


def curvature_gain(xs, ys) -> float:
    """How much a quadratic fits better than a straight line: R-squared(quadratic) - R-squared(line)."""
    import numpy as np

    x, y = np.array(xs, dtype=float), np.array(ys, dtype=float)
    total = float(((y - y.mean()) ** 2).sum())
    if total == 0:
        return 0.0

    def r2(deg):
        resid = y - np.polyval(np.polyfit(x, y, deg), x)
        return 1 - float((resid**2).sum()) / total

    return r2(2) - r2(1)


def strength_band(r: float) -> str:
    """strong (|r| >= 0.75), moderate (0.5 <= |r| < 0.75) or weak."""
    a = abs(r)
    return "strong" if a >= 0.75 else "moderate" if a >= 0.5 else "weak"


def clear_of_band_edges(r: float, margin: float = 0.05) -> bool:
    """True when |r| is not within ``margin`` of a strength boundary (0.5 or 0.75)."""
    return all(abs(abs(r) - edge) > margin for edge in (0.5, 0.75))


# ---------------------------------------------------------------- small simple graphs

Edge = tuple[int, int]


def norm(u: int, v: int) -> Edge:
    return (u, v) if u < v else (v, u)


def edge_name(edge: Edge) -> str:
    return LETTERS[edge[0]] + LETTERS[edge[1]]


def edges_text(edges) -> str:
    """``AB, AC, BD``: the edges in alphabetical order."""
    return ", ".join(edge_name(e) for e in sorted(edges))


def degrees(n: int, edges) -> list[int]:
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    return deg


def neighbours(n: int, edges) -> list[set[int]]:
    out: list[set[int]] = [set() for _ in range(n)]
    for u, v in edges:
        out[u].add(v)
        out[v].add(u)
    return out


def components(n: int, edges) -> list[set[int]]:
    """The connected components, each a set of vertices, smallest vertex first."""
    nbr = neighbours(n, edges)
    seen: set[int] = set()
    out = []
    for start in range(n):
        if start in seen:
            continue
        comp, stack = {start}, [start]
        while stack:
            for w in nbr[stack.pop()]:
                if w not in comp:
                    comp.add(w)
                    stack.append(w)
        seen |= comp
        out.append(comp)
    return out


def distance(n: int, edges, a: int, b: int) -> int | None:
    """The fewest edges on a walk from ``a`` to ``b`` (breadth-first search), or None if there is no walk."""
    nbr = neighbours(n, edges)
    frontier, seen, steps = {a}, {a}, 0
    while frontier:
        if b in frontier:
            return steps
        frontier = {w for v in frontier for w in nbr[v]} - seen
        seen |= frontier
        steps += 1
    return None


def random_connected(rng, n: int, m: int) -> list[Edge]:
    """A connected simple graph on ``n`` vertices with ``m`` edges: a random tree, then random extra edges."""
    check(n - 1 <= m <= n * (n - 1) // 2, "no connected simple graph with that many edges")
    order = list(range(n))
    rng.shuffle(order)
    edges = {norm(order[i], order[rng.randrange(i)]) for i in range(1, n)}
    rest = [p for p in combinations(range(n), 2) if p not in edges]
    rng.shuffle(rest)
    edges.update(rest[: m - len(edges)])
    return sorted(edges)


def counted(n: int, singular: str, plural: str | None = None) -> str:
    """``1 road``, ``2 roads``."""
    return f"{n} {singular if n == 1 else plural or singular + 's'}"


__all__ = [
    "clear_of_band_edges",
    "corr",
    "curvature_gain",
    "strength_band",
    "GRADER_TOLERANCE",
    "LETTERS",
    "attempts",
    "check",
    "components",
    "counted",
    "degrees",
    "distance",
    "edge_name",
    "edges_text",
    "list_part",
    "listed",
    "matrix_cells",
    "matrix_distractors",
    "matrix_part",
    "matrix_single",
    "mcq_part",
    "neighbours",
    "norm",
    "num_text",
    "number_distractors",
    "number_part",
    "random_connected",
    "rejected",
    "single",
]
