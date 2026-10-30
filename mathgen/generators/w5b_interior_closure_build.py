"""Generates `w5b_interior_closure` and `exd_x_w5b_interior_closure`: interior, closure, boundary and limit points.

One second-year skill, ``uc2-interior-closure-boundary`` (domain ``metric-spaces-and-topology``). Subsets of R are
finite unions of pieces, each one of

- ``("I", a, b, lc, rc)``: an interval from a to b, closed at the left end when ``lc`` and at the right when ``rc``;
- ``("P", p)``: the single point p;
- ``("Q", a, b)``: the rational numbers strictly between a and b;
- ``("S", c, sgn)``: the sequence ``{c + sgn/n : n = 1, 2, 3, ...}``, which has the single limit point c.

Everything is exact (`Fraction`). `analyse` merges the interval pieces into maximal components (so ``(0, 1] ∪ (1, 2)``
is ``(0, 2)`` while ``(0, 1) ∪ (1, 2)`` stays in two parts) and reads the interior, closure, boundary, isolated points,
limit points and ``closure minus S`` off the components. The planar sets are discs, a punctured disc, annuli,
half-planes and rectangles; points are classified by exact integer comparison.

Every single-part question has a set of numbers (or of lattice points) for its answer, because the engine grades a
skill's single-part questions with the one grader its `answer` kind names. The exam chain grades each part with its own
grader.

Tiers:

- ``easy``: one interval, an interval and a point, or two intervals: the boundary points, or the endpoints of the interior
  or of the closure.
- ``medium``: two to four pieces, with touching intervals and isolated points (an interval that meets a point it does not
  contain merges with it): boundary points, endpoints of the interior or closure, isolated points.
- ``hard``: sequences such as ``{1/n}`` and ``{2 - 1/n}`` (the set of limit points; the closure points missing from the
  set); sets built from ``ℚ ∩ (a, b)`` (endpoints of the interior, which ignores the rational pieces, and of the
  closure); and planar sets (which of six points lie in the interior, closure or boundary of a disc, punctured disc,
  annulus, half-plane or rectangle).
- ``exam`` (topic ``exd_x_w5b_interior_closure``): a five-part chain on one subset of R: the interior (an interval answer),
  the boundary points (or the points of the closure that are missing from the set) as a set, their number, their sum and
  their mean by follow-through.
"""

from __future__ import annotations

import itertools
import random
from fractions import Fraction as Fr
from types import SimpleNamespace

import sympy

from ..answers import Distractor
from ..core import attempts, check, generator
from ..grading import GRADERS
from ..grading.inequality import read_solution_set
from ..grading.inequality import _same_set as _same_real_set  # noqa: PLC2701
from ._r3j_kit import set_part, single_set
from ._spread_kit import uses_rng
from ._w5b_balance import balanced
from ._w5a_support import Link, X, ineq_part, join_signed, link_part, msay

TOPIC = "w5b_interior_closure"
EXAM_TOPIC = "exd_x_w5b_interior_closure"


# -- numbers and text ---------------------------------------------------------------------------------------------------------


def num(x) -> str:
    """An integer or a half as a learner reads it: 3, -4, 1.5, -2.5."""
    x = Fr(x)
    if x.denominator == 1:
        return str(x.numerator)
    check(x.denominator == 2, "only integers and halves are shown")
    return str(float(x))


def set_item(x) -> str:
    x = Fr(x)
    if x.denominator in (1, 2):
        return num(x)
    return f"{x.numerator}/{x.denominator}"


def _interval_text(a, b, lc, rc) -> str:
    return f"{'[' if lc else '('}{num(a)}, {num(b)}{']' if rc else ')'}"


def _seq_expr(c, sgn) -> str:
    one = "1/n" if sgn > 0 else "-1/n"
    if c == 0:
        return one
    return f"{num(c)} + 1/n" if sgn > 0 else f"{num(c)} - 1/n"


def piece_x(p) -> X:
    kind = p[0]
    if kind == "I":
        t = _interval_text(*p[1:])
        return X(t, t)
    if kind == "P":
        t = "{" + num(p[1]) + "}"
        return X(t, r"\{" + num(p[1]) + r"\}")
    if kind == "Q":
        t = _interval_text(p[1], p[2], p[3], p[4])
        return X("ℚ ∩ " + t, r"\mathbb{Q} \cap " + t)
    e = _seq_expr(p[1], p[2])
    return X("{" + e + " : n = 1, 2, 3, …}", r"\{" + e.replace("1/n", r"\frac{1}{n}") + r" : n = 1, 2, 3, \ldots\}")


def set_x(pieces) -> X:
    xs = [piece_x(p) for p in pieces]
    if len(xs) > 1:  # keep "ℚ ∩ (a, b)" visibly one piece of the union
        xs = [X("(" + x.p + ")", "(" + x.t + ")") if p[0] == "Q" else x for p, x in zip(pieces, xs)]
    return X(" ∪ ".join(x.p for x in xs), r" \cup ".join(x.t for x in xs))


def union_text(comps) -> str:
    """Closed or open intervals joined by union signs; the empty set is ∅."""
    if not comps:
        return "∅"
    return " ∪ ".join(_interval_text(*c) for c in comps)


# -- the exact structure of a subset of R -------------------------------------------------------------------------------------


def merge_reals(reals):
    """Merge intervals (a, b, lc, rc), points included as (p, p, True, True), into maximal components."""
    out: list[list] = []
    for a, b, lc, rc in sorted(reals, key=lambda t: (t[0], not t[2])):
        if out and (a < out[-1][1] or (a == out[-1][1] and (out[-1][3] or lc))):
            cur = out[-1]
            if b > cur[1]:
                cur[1], cur[3] = b, rc
            elif b == cur[1]:
                cur[3] = cur[3] or rc
        else:
            out.append([a, b, lc, rc])
    return [tuple(c) for c in out]


def analyse(pieces) -> SimpleNamespace:
    reals = [(p[1], p[2], p[3], p[4]) for p in pieces if p[0] == "I"] + [
        (p[1], p[1], True, True) for p in pieces if p[0] == "P"
    ]
    comps = merge_reals(reals)
    qs = [(p[1], p[2]) for p in pieces if p[0] == "Q"]
    seqs = [p[1] for p in pieces if p[0] == "S"]
    wide = [c for c in comps if c[0] < c[1]]
    iso = [c[0] for c in comps if c[0] == c[1]]
    finite = not qs and not seqs
    boundary = sorted({c[0] for c in comps} | {c[1] for c in comps}) if finite else None
    missing = None
    if not qs:
        missing = set()
        for a, b, lc, rc in wide:
            if not lc:
                missing.add(a)
            if not rc:
                missing.add(b)
        for c in seqs:
            if not any(x[0] <= c <= x[1] for x in comps):
                missing.add(c)
        missing = sorted(missing)
    closure = merge_reals([(a, b, True, True) for a, b, _l, _r in wide] + [(a, b, True, True) for a, b in qs])
    limits = None if wide or qs else sorted(set(seqs))
    return SimpleNamespace(
        comps=comps,
        interior=[(a, b, False, False) for a, b, _l, _r in wide],
        iso=iso,
        boundary=boundary,
        missing=missing,
        closure=closure,
        limits=limits,
        q_length=sum(b - a for a, b in qs),
        has_iso_or_seq=bool(iso or seqs),
        has_q=bool(qs),
        has_seq=bool(seqs),
    )


# -- random subsets -----------------------------------------------------------------------------------------------------------


def _end(rng) -> int:
    return rng.choice((0, 1))


def _extent(p):
    if p[0] == "P":
        return p[1], p[1]
    if p[0] == "S":
        return (p[1] - 1, p[1]) if p[2] < 0 else (p[1], p[1] + 1)
    return p[1], p[2]


def _moved(p, d):
    if p[0] == "P":
        return ("P", p[1] + d)
    if p[0] == "S":
        return ("S", p[1] + d, p[2])
    return (p[0], p[1] + d, p[2] + d) + tuple(p[3:])


def build_set(rng: random.Random, kinds, *, halves=False, touch=0.0, limit_point=0.0, reach=9, scale=1):
    """Pieces laid out left to right with gaps, then slid to a random place in [-reach, reach]; `kinds` lists the
    piece types in order ("I", "P", "Q", "S")."""
    for _ in attempts(200, "a set that fits in range"):
        pos = Fr(0)
        pieces = []
        prev = None
        for k in kinds:
            touching = prev in ("I", "P") and k in ("I", "P") and "I" in (prev, k) and rng.random() < touch
            gap = Fr(0) if touching else Fr(rng.randint(1, 3 * scale))
            if prev is None:
                gap = Fr(0)
            elif prev in ("Q", "S") or k in ("Q", "S"):
                gap = max(gap, Fr(1))
            pos += gap
            if touching and prev == "I":
                pieces[-1] = pieces[-1][:4] + (False,)  # the point at its right end is not already in it
            if k == "I":
                length = Fr(rng.randint(1, 4 * scale)) if not halves else Fr(rng.randint(2, 8), 2)
                lc = False if (touching and prev == "P") else bool(_end(rng))
                pieces.append(("I", pos, pos + length, lc, bool(_end(rng))))
                pos += length
            elif k == "P":
                pieces.append(("P", pos))
            elif k == "Q":
                length = Fr(rng.randint(2, 4))
                pieces.append(("Q", pos, pos + length, False, False))
                pos += length
            else:
                sgn = rng.choice((1, -1))
                c = pos if sgn > 0 else pos + 1
                pieces.append(("S", c, sgn))
                if rng.random() < limit_point:
                    pieces.append(("P", c))
                pos += 1
            prev = k
        lo = min(_extent(p)[0] for p in pieces)
        hi = max(_extent(p)[1] for p in pieces)
        if hi - lo <= 2 * reach:
            d = Fr(rng.randint(int(-reach - lo), int(reach - hi)))
            return [_moved(p, d) for p in pieces]
    raise AssertionError("unreachable")


def _shuffled(rng, pieces):
    pieces = list(pieces)
    if rng.random() < 0.5:
        rng.shuffle(pieces)
    return pieces


def _probe(kind: str, answer, text: str):
    return SimpleNamespace(answer=answer, answer_text=text, answer_kind=kind, text="S")


def _ineq_wrongs(key: str, cands):
    """Wrong interval answers, as (text, tag), dropped when the grader reads them as the key or cannot read them."""
    expected = read_solution_set(key)
    check(expected is not None, f"unreadable key {key!r}")
    out, seen = [], {key}
    for text, tag in cands:
        if text in seen:
            continue
        read = read_solution_set(text)
        if read is None or _same_real_set(expected[0], read[0]):
            continue
        seen.add(text)
        out.append((text, tag))
    return out


def _ineq_question(prompt, key, working, cands):
    wrongs = _ineq_wrongs(key, cands)
    part = ineq_part(prompt, key, 1, working, wrongs)
    return _as_question(part)


def _set_question(prompt, items, working, slips, *, single=False):
    """A question whose answer is a set of numbers; `prompt` is a (text, latex) pair."""
    if single:
        q = single_set(prompt[0], items, working, slips, keep=3)
        q["latex"] = prompt[1]
        return q
    part = set_part(prompt[0], items, 1, working, slips)
    part["prompt_latex"] = prompt[1]
    return _as_question(part)


def _as_question(part: dict) -> dict:
    q = dict(part)
    q["text"] = q.pop("prompt")
    if "prompt_latex" in q:
        q["latex"] = q.pop("prompt_latex")
    q.pop("marks", None)
    return q


# -- the wrong-answer builders ------------------------------------------------------------------------------------------------


def _closed(comps):
    return [(a, b, True, True) for a, b, _l, _r in comps]


def _interior_cands(an):
    comps = an.comps
    wide = [c for c in comps if c[0] < c[1]]
    cands = [
        (union_text([(a, b, lc or True, rc or True) for a, b, lc, rc in wide]), "ic-interior-keeps-endpoints"),
        (union_text([(a, b, lc, rc) for a, b, lc, rc in wide]), "ic-interior-keeps-endpoints"),
        (union_text(_closed(wide)), "ic-interior-keeps-endpoints"),
    ]
    cands.append((union_text([(a, b, True, False) for a, b, _l, _r in wide]), "ic-interior-keeps-endpoints"))
    cands.append((union_text([(a, b, False, True) for a, b, _l, _r in wide]), "ic-interior-keeps-endpoints"))
    if len(wide) >= 2:
        cands.append((union_text(an.interior[:-1]), "ic-interior-drops-a-piece"))
        cands.append(
            (union_text([(an.interior[0][0], an.interior[-1][1], False, False)]), "ic-touching-intervals-merged")
        )
    # a merge across a shared endpoint that is NOT in the set
    touching = [(x, y) for x, y in zip(wide, wide[1:]) if x[1] == y[0]]
    for x, y in touching:
        merged = [c for c in an.interior if c not in (x[:2] + (False, False), y[:2] + (False, False))]
        merged.append((x[0], y[1], False, False))
        cands.append((union_text(sorted(merged)), "ic-touching-intervals-merged"))
    return cands


def _closure_cands(an):
    cl = an.closure
    cands = [
        (union_text(an.interior), "ic-closure-drops-open-ends"),
        (union_text([(a, b, False, False) for a, b, _l, _r in cl]), "ic-closure-drops-open-ends"),
        (union_text([(a, b, True, False) for a, b, _l, _r in cl]), "ic-closure-keeps-open-end"),
        (union_text([(a, b, False, True) for a, b, _l, _r in cl]), "ic-closure-keeps-open-end"),
    ]
    comps = [c for c in an.comps if c[0] < c[1]]
    cands.append((union_text([(a, b, lc, rc) for a, b, lc, rc in comps]), "ic-closure-is-the-set"))
    if len(cl) >= 2:
        cands.append((union_text([(cl[0][0], cl[-1][1], True, True)]), "ic-closure-fills-the-gaps"))
        cands.append((union_text(cl[:-1]), "ic-closure-drops-a-piece"))
    return cands


def _int_part_key(value: int) -> str:
    return str(value)


# -- wrong-answer builders for the exam counts and sums ----------------------------------------------------------------------------


def _members(comps):
    """The boundary-candidate points that belong to the set itself: closed ends and isolated points."""
    out = set()
    for a, b, lc, rc in comps:
        if a == b:
            out.add(a)
            continue
        if lc:
            out.add(a)
        if rc:
            out.add(b)
    return sorted(out)


def _bd_wrongs(an):
    """Slips for 'how many boundary points' on a finite set."""
    comps, pts = an.comps, an.boundary
    n = len(pts)
    wide = [c for c in comps if c[0] < c[1]]
    wrongs = [
        (len(_members(comps)), "ic-boundary-endpoints-in-set-only"),
        (2 * len(wide), "ic-boundary-misses-isolated-point"),
        (len(comps), "ic-boundary-counts-pieces"),
        (n + 1, "dg-near-miss-value"),
        (n - 1, "dg-near-miss-value"),
        (n + 2, "dg-near-miss-value"),
    ]
    return [(int(v), t) for v, t in wrongs if v >= 0]


def _bd_sum_wrongs(an):
    pts = an.boundary
    s = sum(pts)
    wrongs = [
        (sum(_members(an.comps)), "ic-boundary-endpoints-in-set-only"),
        (sum(x for x in pts if x not in an.iso), "ic-boundary-misses-isolated-point"),
        (s + 1, "dg-near-miss-value"),
        (s - 1, "dg-near-miss-value"),
        (-s, "calc-sign-slip"),
    ]
    return [(int(v), t) for v, t in wrongs]


# -- set-of-numbers slips ---------------------------------------------------------------------------------------------------------


def _texts(points):
    return [set_item(x) for x in sorted(points)]


def _ends(ivs):
    return sorted({x for a, b, _l, _r in ivs for x in (a, b)})


def _slips(key_points, cands):
    """(items, tag) slips as text, dropping empty ones and any equal to the key; a near miss or two is always added."""
    key = sorted(key_points)
    out, seen = [], {tuple(key)}
    fallback = [
        ([x + 1 if i == 0 else x for i, x in enumerate(key)], "dg-near-miss-value"),
        ([x - 1 if i == len(key) - 1 else x for i, x in enumerate(key)], "dg-near-miss-value"),
        (key + [key[-1] + 1], "dg-near-miss-value"),
    ]
    for pts, tag in list(cands) + fallback:
        pts = sorted(set(pts))
        if not pts or tuple(pts) in seen:
            continue
        seen.add(tuple(pts))
        out.append((_texts(pts), tag))
    return out


def _bd_set_slips(an):
    pts, comps = an.boundary, an.comps
    cands = [
        (_members(comps), "ic-boundary-endpoints-in-set-only"),
        ([x for x in pts if x not in an.iso], "ic-boundary-misses-isolated-point"),
        ([x for x in pts if x != comps[0][0]], "ic-boundary-misses-open-left-end"),
        (
            set(pts) | {x for c in comps if c[0] < c[1] for x in (c[0] + 1,) if c[0] + 1 < c[1]},
            "ic-boundary-counts-interior-points",
        ),
        ([-x for x in pts], "calc-sign-slip"),
        (pts[:-1], "dg-near-miss-value"),
        (pts[1:], "dg-near-miss-value"),
    ]
    return _slips(pts, cands)


# -- single-part questions, all answered with a set of numbers ------------------------------------------------------------------


def _pick(rng, options):
    return options[rng.randrange(len(options))]()


def _form_bd_set(rng, pieces, an):
    s = set_x(_shuffled(rng, pieces))
    prompt = _pick(
        rng,
        [
            lambda: msay("Let S = ", s, " in R. Give the boundary of S as a set of numbers, in the form {a, b, c}."),
            lambda: msay("List the boundary points of S = ", s, " in R, in the form {a, b, c}."),
            lambda: msay("What is the boundary of S = ", s, " in R? Write it as a set of numbers {a, b, c}."),
        ],
    )
    items = _texts(an.boundary)
    working = [
        f"The components are {union_text(an.comps)}.",
        f"Their endpoints and isolated points are {{{', '.join(items)}}}.",
    ]
    return _set_question(prompt, items, working, _bd_set_slips(an), single=True)


def _form_interior_ends(rng, pieces, an):
    s = set_x(_shuffled(rng, pieces))
    key = _ends(an.interior)
    prompt = _pick(
        rng,
        [
            lambda: msay(
                "Let S = ",
                s,
                " as a subset of R. The interior of S is a union of open intervals. Give the set of all endpoints of those intervals, in the form {a, b, c}.",
            ),
            lambda: msay(
                "Find the interior of S = ",
                s,
                " in R. Write its interior as a union of the maximal open intervals contained in S, and list every endpoint of those intervals as a set {a, b, c}.",
            ),
            lambda: msay(
                "Consider S = ",
                s,
                " with the usual metric on R. Its interior is the union of the maximal open intervals contained in S; give the set {a, b, c} of all their endpoints.",
            ),
        ],
    )
    wide = [c for c in an.comps if c[0] < c[1]]
    listed = [p for p in pieces if p[0] == "I"]
    naive = _ends([(p[1], p[2], False, False) for p in listed])
    touching = [x for x, y in zip(wide, wide[1:]) if x[1] == y[0]]
    cands = [
        (naive, "ic-touching-intervals-not-merged"),
        (set(key) | set(an.iso), "ic-isolated-point-interior"),
        (set(key) | {p[i] for p in pieces if p[0] == "Q" for i in (1, 2)}, "ic-rational-interior"),
        ([x for x in key if x not in {t[1] for t in touching}], "ic-touching-intervals-merged"),
        (key[:-2], "ic-interior-drops-a-piece"),
        (key[2:], "ic-interior-drops-a-piece"),
        ([x + 1 if i == len(key) - 1 else x for i, x in enumerate(key)], "dg-near-miss-value"),
        ([-x for x in key], "calc-sign-slip"),
        ([key[0], key[-1]], "ic-closure-fills-the-gaps"),
    ]
    working = [
        f"The pieces merge into the components {union_text(an.comps)}.",
        f"Dropping the isolated points and the endpoints leaves {union_text(an.interior)}, whose endpoints are {{{', '.join(_texts(key))}}}.",
    ]
    return _set_question(prompt, _texts(key), working, _slips(key, cands), single=True)


def _form_closure_ends(rng, pieces, an):
    s = set_x(_shuffled(rng, pieces))
    cl = merge_reals(an.closure)
    key = _ends(cl)
    prompt = _pick(
        rng,
        [
            lambda: msay(
                "Let S = ",
                s,
                " as a subset of R. The closure of S is a union of closed intervals. Give the set of all endpoints of those intervals, in the form {a, b, c}.",
            ),
            lambda: msay(
                "Find the closure of S = ",
                s,
                " in R. Write its closure as a union of the maximal closed intervals it contains, and list every endpoint of those intervals as a set {a, b, c}.",
            ),
            lambda: msay(
                "Consider S = ",
                s,
                " with the usual metric on R. Its closure is the union of the maximal closed intervals it contains; give the set {a, b, c} of all their endpoints.",
            ),
        ],
    )
    wide = [c for c in an.comps if c[0] < c[1]]
    qs = [p for p in pieces if p[0] == "Q"]
    listed = [p for p in pieces if p[0] in ("I", "Q")]
    naive = _ends([(p[1], p[2], True, True) for p in listed])
    cands = [
        (naive, "ic-closure-not-merged"),
        (_members(wide), "ic-closure-drops-open-ends"),
        ([key[0], key[-1]], "ic-closure-fills-the-gaps"),
        (
            _ends([c for c in cl if not any(c[0] == p[1] and c[1] == p[2] for p in qs)]) if qs else key[:-2],
            "ic-rational-closure-itself" if qs else "ic-closure-drops-a-piece",
        ),
        (key[2:], "ic-closure-drops-a-piece"),
        ([x + 1 if i == 0 else x for i, x in enumerate(key)], "dg-near-miss-value"),
        ([-x for x in key], "calc-sign-slip"),
    ]
    working = [
        "Each interval component keeps all its points and gains its endpoints; rational intervals are dense in their closed interval; closed intervals that meet merge.",
        f"The closure is {union_text(cl)}, with endpoints {{{', '.join(_texts(key))}}}.",
    ]
    return _set_question(prompt, _texts(key), working, _slips(key, cands), single=True)


def _form_iso_set(rng, pieces, an):
    s = set_x(_shuffled(rng, pieces))
    key = list(an.iso)
    prompt = _pick(
        rng,
        [
            lambda: msay(
                "Let S = ",
                s,
                " in R. Give the set of isolated points of S, in the form {a, b}. (A point p of S is isolated if some open interval around p contains no other point of S.)",
            ),
            lambda: msay(
                "Which points of S = ",
                s,
                " are isolated points of S, that is points with a neighbourhood meeting S only at that point? Give them as a set {a, b}.",
            ),
            lambda: msay(
                "A point p of S = ",
                s,
                " is isolated when some interval (p - e, p + e) meets S only at p. List the isolated points of S as a set {a, b}.",
            ),
        ],
    )
    listed = [p[1] for p in pieces if p[0] == "P"]
    cands = [
        (listed, "ic-isolated-point-touching-interval"),
        ([c[0] for c in an.comps if c[0] < c[1] and c[2]] + key, "ic-isolated-point-limit-point"),
        (set(_ends([c for c in an.comps if c[0] < c[1]])) | set(key), "ic-boundary-counts-pieces"),
        (key[:-1], "dg-near-miss-value"),
        ([-x for x in key], "calc-sign-slip"),
        ([x + 1 for x in key], "dg-near-miss-value"),
    ]
    working = [
        f"The components are {union_text(an.comps)}; the degenerate ones are the isolated points {{{', '.join(_texts(key))}}}. A listed point that sits at an end of an interval is joined to it."
    ]
    return _set_question(prompt, _texts(key), working, _slips(key, cands), single=True)


def _easy_pieces(rng):
    shape = rng.choices(("one", "point", "two", "ipoint"), (4, 3, 2, 2))[0]
    if shape == "one":
        return build_set(rng, ["I"], halves=rng.random() < 0.5)
    if shape == "point":
        return build_set(rng, rng.choice((["I", "P"], ["P", "I"])))
    if shape == "two":
        return build_set(rng, ["I", "I"], halves=rng.random() < 0.3)
    return build_set(rng, ["P", "P"] if rng.random() < 0.3 else ["I", "P", "P"])


def _easy(rng: random.Random) -> dict:
    for _ in attempts(100, "an easy question"):
        pieces = _easy_pieces(rng)
        an = analyse(pieces)
        ask = rng.choices(("interior", "closure", "bd"), (3, 3, 4))[0]
        if ask == "interior" and not an.interior:
            continue
        if ask == "closure" and an.has_iso_or_seq:
            continue
        if ask == "bd" and len(an.boundary) < 2:
            continue
        break
    form = {"interior": _form_interior_ends, "closure": _form_closure_ends, "bd": _form_bd_set}[ask]
    return form(rng, pieces, an)


def _medium(rng: random.Random) -> dict:
    for _ in attempts(300, "a medium question"):
        kinds = rng.choice(
            (
                ["I", "I"],
                ["I", "I", "I"],
                ["I", "I", "P"],
                ["I", "P", "I"],
                ["I", "P", "P"],
                ["I", "I", "P", "P"],
                ["P", "I", "I"],
            )
        )
        pieces = build_set(rng, kinds, touch=0.5)
        an = analyse(pieces)
        ask = rng.choices(("interior", "closure", "bd", "iso"), (3, 2, 3, 2))[0]
        if ask == "interior" and not an.interior:
            continue
        if ask == "closure" and an.has_iso_or_seq:
            continue
        if ask == "iso" and not an.iso:
            continue
        if ask == "bd" and len(an.boundary) < 3:
            continue
        break
    form = {"interior": _form_interior_ends, "closure": _form_closure_ends, "bd": _form_bd_set, "iso": _form_iso_set}[
        ask
    ]
    return form(rng, pieces, an)


# -- hard: sequences, rationals, planar sets ------------------------------------------------------------------------------------


def _form_limits(rng, pieces, an):
    s = set_x(_shuffled(rng, pieces))
    key = list(an.limits)
    prompt = _pick(
        rng,
        [
            lambda: msay(
                "Let S = ",
                s,
                " in R. Give the set of limit points of S in the form {a, b}. (p is a limit point of S if every interval around p contains a point of S other than p.)",
            ),
            lambda: msay(
                "Find all the limit points of S = ",
                s,
                " in R, written as a set {a, b}. A limit point of S need not belong to S.",
            ),
            lambda: msay("Which real numbers are limit points of S = ", s, "? Give them as a set {a, b}."),
        ],
    )
    terms = [p[1] + Fr(p[2], n) for p in pieces if p[0] == "S" for n in (1, 2)]
    seq_limits = {p[1] for p in pieces if p[0] == "S"}
    cands = [
        (seq_limits | set(an.iso), "ic-isolated-point-limit-point"),
        (terms, "ic-sequence-terms-limit-points"),
        (an.iso, "ic-limit-point-must-belong"),
        ([-x for x in key], "calc-sign-slip"),
        (key[:-1], "ic-sequence-limit-missed"),
    ]
    working = [
        "Each sequence c ± 1/n has terms that get arbitrarily close to c without equalling it, so c is a limit point; the terms themselves and the single points are isolated.",
        f"The limit points are {{{', '.join(_texts(key))}}}.",
    ]
    return _set_question(prompt, _texts(key), working, _slips(key, cands), single=True)


def _form_missing_set(rng, pieces, an):
    s = set_x(_shuffled(rng, pieces))
    key = list(an.missing)
    prompt = _pick(
        rng,
        [
            lambda: msay(
                "Let S = ",
                s,
                " in R. Give the set of points of the closure of S that do not belong to S, in the form {a, b}.",
            ),
            lambda: msay(
                "S = ",
                s,
                " is a subset of R. Which points does its closure contain that S itself does not? Give them as a set {a, b}.",
            ),
            lambda: msay("Find the limit points of S = ", s, " that are not in S, as a set {a, b}."),
        ],
    )
    seq_limits = sorted({p[1] for p in pieces if p[0] == "S" and p[1] in key})
    open_ends = sorted(set(key) - set(seq_limits))
    cands = [
        (open_ends, "ic-sequence-limit-missed"),
        (seq_limits, "ic-open-end-missed"),
        (set(key) | set(an.iso), "ic-isolated-point-limit-point"),
        (key[:-1], "dg-near-miss-value"),
        ([-x for x in key], "calc-sign-slip"),
    ]
    working = [
        "An open end of an interval and the limit of a sequence are in the closure; they are missing from S when S does not contain them.",
        f"These points are {{{', '.join(_texts(key))}}}.",
    ]
    return _set_question(prompt, _texts(key), working, _slips(key, cands), single=True)


def _sequence_pieces(rng):
    kinds = rng.choice(
        (["S"], ["S", "P"], ["S", "S"], ["S", "P", "S"], ["P", "S", "P"], ["S", "S", "P"], ["S", "P", "P"])
    )
    return build_set(rng, kinds, limit_point=0.25)


def _mixed_seq_pieces(rng):
    for _ in attempts(50, "a set with a missing closure point"):
        kinds = rng.choice((["I", "S"], ["S", "I"], ["I", "S", "P"], ["I", "I", "S"], ["I", "S", "I"], ["S", "I", "P"]))
        pieces = build_set(rng, kinds, limit_point=0.2)
        if analyse(pieces).missing:
            return pieces
    raise AssertionError("unreachable")


def _q_set(rng):
    kinds = rng.choice(
        (
            ["Q"],
            ["Q", "I"],
            ["I", "Q"],
            ["Q", "Q"],
            ["Q", "I", "I"],
            ["I", "Q", "I"],
            ["Q", "P"],
            ["Q", "I", "P"],
            ["I", "Q", "P"],
        )
    )
    return build_set(rng, kinds)


def _hard_real(rng):
    sub = rng.choices(("limits", "missing", "q_interior", "q_closure"), (3, 3, 2, 2))[0]
    if sub == "limits":
        pieces = _sequence_pieces(rng)
        return _form_limits(rng, pieces, analyse(pieces))
    if sub == "missing":
        pieces = _mixed_seq_pieces(rng)
        return _form_missing_set(rng, pieces, analyse(pieces))
    for _ in attempts(100, "a rational set"):
        pieces = _q_set(rng)
        an = analyse(pieces)
        if sub == "q_interior" and an.interior:
            return _form_interior_ends(rng, pieces, an)
        if sub == "q_closure" and not an.iso:
            return _form_closure_ends(rng, pieces, an)
    raise AssertionError("unreachable")


# -- planar sets ---------------------------------------------------------------------------------------------------------------------


def _shift(var: str, v: int) -> str:
    return var if v == 0 else f"{var} - {v}" if v > 0 else f"{var} + {-v}"


def _sq(c):
    def one(var, v):
        return f"{var}²" if v == 0 else f"({_shift(var, v)})²"

    return f"{one('x', c[0])} + {one('y', c[1])}"


def _lin(a, b) -> str:
    def term(coef, v, first):
        if coef == 0:
            return ""
        mag = "" if abs(coef) == 1 else str(abs(coef))
        if first:
            return ("-" if coef < 0 else "") + mag + v
        return f" {'-' if coef < 0 else '+'} {mag}{v}"

    lead = term(a, "x", True)
    return lead + term(b, "y", not lead)


class Planar:
    """A planar set with exact classification of lattice points."""

    def __init__(self, kind, text, **kw):
        self.kind, self.text = kind, text
        self.__dict__.update(kw)

    def classify(self, p):
        """('int'|'bd'|'ext', member) for the lattice point p."""
        x, y = p
        if self.kind in ("disc", "punct", "annulus"):
            d2 = (x - self.c[0]) ** 2 + (y - self.c[1]) ** 2
            if self.kind == "disc":
                lo, hi = None, self.r**2
                member = d2 < hi or (self.closed and d2 == hi)
            elif self.kind == "punct":
                lo, hi = 0, self.r**2
                member = 0 < d2 < hi
            else:
                lo, hi = self.r1**2, self.r2**2
                member = (d2 > lo or (self.lo_closed and d2 == lo)) and (d2 < hi or (self.hi_closed and d2 == hi))
            inside = (lo is None or d2 > lo) and d2 < hi
            on = d2 == hi or (lo is not None and d2 == lo)
            return ("int" if inside else "bd" if on else "ext"), member
        if self.kind == "half":
            v = (self.a * x + self.b * y - self.k) * self.sense
            return ("int" if v < 0 else "bd" if v == 0 else "ext"), v < 0 or (self.closed and v == 0)
        a, b, c, d = self.box
        lx, rx, ly, ry = self.flags
        inside = a < x < b and c < y < d
        on_or_in = a <= x <= b and c <= y <= d
        okx = a < x < b or (x == a and lx) or (x == b and rx)
        oky = c < y < d or (y == c and ly) or (y == d and ry)
        return ("int" if inside else "bd" if on_or_in else "ext"), okx and oky


def _rand_planar(rng):
    k = rng.choice(["disc", "disc", "punct", "annulus", "rect", "sqminus", "half", "half"])
    c = (rng.randint(-3, 3), rng.randint(-3, 3))
    if k == "disc":
        r, closed = rng.randint(2, 6), rng.random() < 0.5
        return Planar("disc", f"{{(x, y) : {_sq(c)} {'≤' if closed else '<'} {r * r}}}", c=c, r=r, closed=closed)
    if k == "punct":
        r = rng.randint(2, 6)
        return Planar("punct", f"{{(x, y) : 0 < {_sq(c)} < {r * r}}}", c=c, r=r)
    if k == "annulus":
        r1 = rng.randint(1, 4)
        r2 = r1 + rng.randint(1, 4)
        lo_closed, hi_closed = rng.random() < 0.5, rng.random() < 0.5
        text = f"{{(x, y) : {r1 * r1} {'≤' if lo_closed else '<'} {_sq(c)} {'≤' if hi_closed else '<'} {r2 * r2}}}"
        return Planar("annulus", text, c=c, r1=r1, r2=r2, lo_closed=lo_closed, hi_closed=hi_closed)
    if k == "half":
        a, b = rng.choice([(1, 0), (0, 1), (1, 1), (1, -1), (2, 1), (1, 2), (2, -1), (3, 1), (1, -2), (3, -2)])
        kk, closed, sense = rng.randint(-4, 6), rng.random() < 0.5, rng.choice((1, -1))
        op = {(1, False): "<", (1, True): "≤", (-1, False): ">", (-1, True): "≥"}[(sense, closed)]
        return Planar(
            "half", f"{{(x, y) : {_lin(a, b)} {op} {kk}}}", a=a, b=b, k=kk, closed=closed, sense=sense, c=(0, 0)
        )
    a = rng.randint(-3, 1)
    b = a + rng.randint(2, 5)
    c0 = rng.randint(-3, 1)
    d0 = c0 + rng.randint(2, 5)
    if k == "sqminus":
        d0 = c0 + (b - a)
        return Planar(
            "rect",
            f"the square [{a}, {b}] × [{c0}, {d0}] with its boundary removed",
            box=(a, b, c0, d0),
            flags=(False,) * 4,
            c=((a + b) // 2, (c0 + d0) // 2),
        )
    fl = tuple(bool(rng.randint(0, 1)) for _ in range(4))
    text = f"{'[' if fl[0] else '('}{a}, {b}{']' if fl[1] else ')'} × {'[' if fl[2] else '('}{c0}, {d0}{']' if fl[3] else ')'}"
    return Planar("rect", text, box=(a, b, c0, d0), flags=fl, c=((a + b) // 2, (c0 + d0) // 2))


def _planar_points(rng, sh: Planar, n_in: int, n_bd: int):
    """Six lattice points with the wanted numbers of interior and boundary points."""
    ints, bds, exts = [], [], []
    cx, cy = sh.c
    for x in range(cx - 9, cx + 10):
        for y in range(cy - 9, cy + 10):
            where, _ = sh.classify((x, y))
            (ints if where == "int" else bds if where == "bd" else exts).append((x, y))
    n_bd, n_in = min(n_bd, len(bds)), min(n_in, len(ints))
    pts = rng.sample(ints, n_in) + rng.sample(bds, n_bd) + rng.sample(exts, 6 - n_in - n_bd)
    rng.shuffle(pts)
    return pts


def _form_planar_set(rng):
    sh = _rand_planar(rng)
    for _ in attempts(50, "a planar question with a non-trivial answer"):
        pts = _planar_points(rng, sh, rng.randint(1, 3), rng.choice((1, 2, 2, 3)))
        cls = [sh.classify(p) for p in pts]
        ask = rng.choice(("interior", "closure", "boundary"))
        want = {
            "interior": lambda w, m: w == "int",
            "closure": lambda w, m: w != "ext",
            "boundary": lambda w, m: w == "bd",
        }[ask]
        key = [p for p, (w, m) in zip(pts, cls) if want(w, m)]
        if 1 <= len(key) <= 4:
            break
    word = {"interior": "in the interior of", "closure": "in the closure of", "boundary": "on the boundary of"}[ask]
    plist = ", ".join(f"({x}, {y})" for x, y in pts)
    s = X(sh.text, sh.text)
    prompt = _pick(
        rng,
        [
            lambda: msay(
                "Let S = ",
                s,
                " as a subset of R². Which of the points ",
                X(plist, plist),
                f" lie {word} S? Give them as a set.",
            ),
            lambda: msay(
                "In R² with the usual metric let S = ",
                s,
                ". From the points ",
                X(plist, plist),
                f", pick those {word.replace('in the', 'in the').replace('on the', 'on the')} S, as a set.",
            ),
            lambda: msay(
                "S = ",
                s,
                " is a subset of the plane. List, as a set, the points among ",
                X(plist, plist),
                f" that lie {word} S.",
            ),
        ],
    )
    members = [p for p, (_w, m) in zip(pts, cls) if m]
    interior = [p for p, (w, _m) in zip(pts, cls) if w == "int"]
    closure = [p for p, (w, _m) in zip(pts, cls) if w != "ext"]
    boundary = [p for p, (w, _m) in zip(pts, cls) if w == "bd"]
    other = {"interior": closure, "closure": interior, "boundary": interior}[ask]
    cands = [
        (members, "ic-count-membership-not-region"),
        (other, "ic-closure-interior-confused"),
        (boundary if ask != "boundary" else members, "ic-closure-interior-confused"),
        (key[:-1], "dg-near-miss-value"),
        (key[1:], "dg-near-miss-value"),
        ([p for p in key if p != sh.c], "ic-punctured-centre-not-boundary")
        if sh.kind == "punct"
        else (set(key) | set(boundary), "ic-closure-interior-confused"),
        (set(key) | {p for p in pts if p not in key}, "mb-counted-outside"),
    ]
    items = [(x, y) for x, y in key]
    seen, slips = {tuple(sorted(items))}, []
    for pts_, tag in cands:
        t = tuple(sorted(set(pts_)))
        if t and t not in seen:
            seen.add(t)
            slips.append(([(x, y) for x, y in t], tag))
    working = [
        "Compare each point's squared distance from the centre, or its value in the inequality, with the bounds: strictly inside is interior; equal to a bound is boundary.",
        f"The points asked for are {', '.join(f'({x}, {y})' for x, y in key)}.",
    ]
    q = single_set(prompt[0], items, working, slips, keep=3)
    q["latex"] = prompt[1]
    return q


def _hard(rng: random.Random) -> dict:
    return _form_planar_set(rng) if rng.random() < 0.38 else _hard_real(rng)


# -- the generators -----------------------------------------------------------------------------------------------------------------


@generator(
    TOPIC,
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def w5b_interior_closure(rng: random.Random, tier: str = "easy") -> dict:
    check(tier in ("easy", "medium", "hard"), f"unknown tier {tier!r}")
    return {"easy": _easy, "medium": _medium, "hard": _hard}[tier](rng)


def _exam_pieces(rng):
    """A subset with intervals and points (flavour A: finite boundary) or with a sequence too (flavour B)."""
    if rng.random() < 0.5:
        kinds = rng.choice(
            (["I", "I", "P"], ["I", "P", "I"], ["I", "I", "I"], ["I", "I", "P", "P"], ["P", "I", "I"], ["I", "P", "P"])
        )
        flavour = "A"
        pieces = build_set(rng, kinds, touch=0.4, reach=50, scale=5)
    else:
        kinds = rng.choice((["I", "S"], ["S", "I"], ["I", "S", "P"], ["I", "I", "S"], ["I", "S", "I"], ["S", "I", "P"]))
        flavour = "B"
        pieces = build_set(rng, kinds, touch=0.0, limit_point=0.2, reach=50, scale=5)
    return flavour, pieces


@generator(EXAM_TOPIC, level=19, presets={"exam": dict(tier="exam")})
@uses_rng
def exd_x_w5b_interior_closure(rng: random.Random, tier: str = "exam") -> dict:
    """A five-part chain: interior, the boundary points (or closure points missing from S), their count, sum and mean."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(300, "an exam set"):
        flavour, pieces = _exam_pieces(rng)
        an = analyse(pieces)
        pts = an.boundary if flavour == "A" else an.missing
        if not an.interior or pts is None or len(pts) < 3 or sum(pts) == 0:
            continue
        if flavour == "A" and len(an.iso) == 0 and rng.random() < 0.5:
            continue
        break
    s = set_x(_shuffled(rng, pieces))
    items = [set_item(x) for x in pts]
    n, total = len(pts), int(sum(pts))
    mean = Fr(total, n)
    what = "boundary" if flavour == "A" else "closure"
    if flavour == "A":
        stem = [
            lambda: msay("Let S = ", s, " be a subset of R with the usual metric."),
            lambda: msay("Consider the subset S = ", s, " of the real line."),
            lambda: msay("The set S = ", s, " lies in R."),
        ][rng.randrange(3)]()
        set_prompt = "List the boundary of S as a set of numbers {a, b, c}."
        count_prompt = "How many points does the boundary of S contain?"
        sum_prompt = "Find the sum of the boundary points."
        ft_prompt = "Hence find the mean of the boundary points, as a fraction or whole number."
        set_working = [
            f"Endpoints of the components {union_text(an.comps)} and the isolated points give {{{', '.join(items)}}}."
        ]
        set_slips = _bd_set_slips(an)
        count_wrongs = _bd_wrongs(an)
        sum_wrongs = _bd_sum_wrongs(an)
    else:
        stem = [
            lambda: msay("Let S = ", s, " be a subset of R with the usual metric."),
            lambda: msay("Consider the subset S = ", s, " of the real line."),
            lambda: msay("The set S = ", s, " lies in R."),
        ][rng.randrange(3)]()
        set_prompt = "Give the set of points of the closure of S that do not belong to S, in the form {a, b, c}."
        count_prompt = "How many points of the closure of S are not in S?"
        sum_prompt = "Find the sum of the points of the closure of S that are not in S."
        ft_prompt = (
            "Hence find the mean of the points of the closure of S that are not in S, as a fraction or whole number."
        )
        set_working = [
            f"The open ends of the intervals and the limits of the sequences that S does not contain give {{{', '.join(items)}}}."
        ]
        seq_limits = sorted({p[1] for p in pieces if p[0] == "S"})
        open_ends = sorted(set(pts) - set(seq_limits))
        set_slips = [
            ([set_item(x) for x in open_ends], "ic-sequence-limit-missed"),
            ([set_item(x) for x in seq_limits], "ic-open-end-missed"),
            ([set_item(x) for x in sorted(set(pts) | set(an.iso))], "ic-isolated-point-limit-point"),
            ([set_item(x) for x in pts[:-1]], "dg-near-miss-value"),
            ([set_item(-x) for x in pts], "calc-sign-slip"),
        ]
        set_slips = [(a, t) for a, t in set_slips if a and sorted(a) != sorted(items)]
        count_wrongs = [
            (len(seq_limits), "ic-open-end-missed"),
            (len(open_ends), "ic-sequence-limit-missed"),
            (n + 1, "dg-near-miss-value"),
            (n - 1, "dg-near-miss-value"),
            (n + 2, "dg-near-miss-value"),
            (n + len(an.iso), "ic-isolated-point-limit-point"),
        ]
        sum_wrongs = [
            (sum(open_ends), "ic-sequence-limit-missed"),
            (sum(seq_limits), "ic-open-end-missed"),
            (total + 1, "dg-near-miss-value"),
            (total - 1, "dg-near-miss-value"),
            (-total, "calc-sign-slip"),
            (total + sum(an.iso), "ic-isolated-point-limit-point"),
        ]
    int_key = union_text(an.interior)
    interior_part = ineq_part(
        msay("Find the interior of S."),
        int_key,
        2,
        [
            f"The interval components are {union_text(an.comps)}; an interval's endpoints are not interior points and isolated points are not interior points.",
            f"The interior is {int_key}.",
        ],
        _ineq_wrongs(int_key, _interior_cands(an)),
    )
    set_p = set_part(msay(set_prompt)[0], items, 2, set_working, set_slips)
    set_p["prompt_latex"] = msay(set_prompt)[1]
    count_link = Link(
        msay(count_prompt),
        n,
        "number",
        1,
        [f"There are {n} such points."],
        balanced(n, [(int(v), t) for v, t in count_wrongs if v >= 0 and int(v) != n], whole=True),
    )
    sum_link = Link(
        msay(sum_prompt),
        total,
        "number",
        1,
        [f"{join_signed(items)} = {total}."],
        balanced(total, [(int(v), t) for v, t in sum_wrongs if int(v) != total], lo=None, whole=True),
    )
    mean_wrongs = [
        (sympy.Integer(total), "ic-mean-sum-not-divided"),
        (sympy.Rational(total, n + 1), "ic-mean-wrong-divisor"),
        (sympy.Rational(total, max(n - 1, 1)), "ic-mean-wrong-divisor"),
        (sympy.Rational(n, total), "ic-mean-inverted"),
        (sympy.Rational(pts[0] + pts[-1], 2), "ic-mean-midrange"),
    ]
    mean_wrongs = [
        (v, t) for v, t in mean_wrongs if sympy.Rational(v) != sympy.Rational(mean.numerator, mean.denominator)
    ]
    mean_link = Link(
        msay(ft_prompt),
        sympy.Rational(mean.numerator, mean.denominator),
        "fraction",
        2,
        [f"{total}/{n} = {sympy.Rational(mean.numerator, mean.denominator)}."],
        balanced(sympy.Rational(mean.numerator, mean.denominator), mean_wrongs, lo=None),
        ft="part_d/part_c",
    )
    _ = what
    parts = [interior_part, set_p, link_part(count_link), link_part(sum_link), link_part(mean_link)]
    stem_text, stem_tex = stem
    return dict(text=stem_text, latex=stem_tex, parts=parts)
