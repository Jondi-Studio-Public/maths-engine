"""Generates `w5b_metric_balls` and `exd_x_w5b_metric_balls`: distances and open balls in R^2 and R^3.

One second-year skill, ``uc2-metrics-and-balls-in-rn`` (domain ``metric-spaces-and-topology``). The metrics are the
taxicab metric ``d_1``, the Euclidean metric ``d_2``, the maximum metric ``d_inf`` and, as a minority, the weighted
taxicab metric ``d_w`` and the discrete metric ``d_0``. Everything is exact: lattice points, rational radii and SymPy
surds, so a distance such as ``sqrt(50)`` is stored as ``5*sqrt(2)``.

Tiers:

- ``easy``: the distance between two lattice points under a named metric.
- ``medium``: how many of six points lie in an open ball (points exactly on the sphere are a deliberate trap); and the
  norm-comparison questions: the largest ball of one metric inside a ball of another, the least ball of one metric
  containing a ball of another, the diameter of a ball measured in another metric.
- ``hard``: the least upper bound of the radii for which a ball fails to contain a finite set; the area or volume of a
  ball (named, or written as a set of inequalities), a weighted diamond, or a shell between two balls; the radius of the
  ball whose vertices are listed.
- ``exam`` (topic ``exd_x_w5b_metric_balls``): a five-part typed chain: two distances, a membership count, a containment
  radius, and the area or volume of that ball by follow-through.

The one fact behind the comparison questions: for norms ``a`` and ``b`` on R^n, ``B_a(c, s)`` lies inside ``B_b(c, r)``
exactly when ``s * C(b, a) <= r``, where ``C(b, a)`` is the largest value of ``||x||_b / ||x||_a``.
"""

from __future__ import annotations

import itertools
import random
from fractions import Fraction as Fr

import sympy

from ..answers import Distractor
from ..core import attempts, check, generator
from ._w5a_support import Link, X, astr, atext, dist_list, exam_dict, int_dist, msay, single

TOPIC = "w5b_metric_balls"
EXAM_TOPIC = "exd_x_w5b_metric_balls"

_SYM = {"1": "d_1", "2": "d_2", "inf": "d_∞", "disc": "d_0", "w": "d_w"}
_TEX = {"1": "d_1", "2": "d_2", "inf": r"d_\infty", "disc": "d_0", "w": "d_w"}
_WORD = {"1": "the taxicab metric", "2": "the Euclidean metric", "inf": "the maximum metric"}
_TEXB = {"1": "1", "2": "2", "inf": r"\infty"}
_BASIC = ("1", "2", "inf")


# -- the metrics: exact distances ---------------------------------------------------------------------------------------


def dist(m, p, q):
    """The exact distance between lattice points under metric m = (kind, weights)."""
    kind, w = m
    diffs = [abs(a - b) for a, b in zip(p, q)]
    if kind == "1":
        return sympy.Integer(sum(diffs))
    if kind == "inf":
        return sympy.Integer(max(diffs))
    if kind == "2":
        return sympy.sqrt(sympy.Integer(sum(d * d for d in diffs)))
    if kind == "w":
        return sympy.Integer(sum(wi * d for wi, d in zip(w, diffs)))
    return sympy.Integer(0 if tuple(p) == tuple(q) else 1)


def _side(m, p, c, r: Fr) -> int:
    """-1, 0 or 1 as the distance from c to p is less than, equal to or greater than the rational radius r."""
    kind, w = m
    diffs = [abs(a - b) for a, b in zip(p, c)]
    if kind == "2":
        v, r = sum(d * d for d in diffs), r * r
    elif kind == "1":
        v = sum(diffs)
    elif kind == "inf":
        v = max(diffs)
    elif kind == "w":
        v = sum(wi * d for wi, d in zip(w, diffs))
    else:
        v = 0 if tuple(p) == tuple(c) else 1
    return (v > r) - (v < r)


def _norm_ratio(b: str, a: str, n: int):
    """C(b, a): the largest value of ||x||_b / ||x||_a over R^n."""
    if a == b:
        return sympy.Integer(1)
    table = {
        ("1", "2"): sympy.sqrt(n),
        ("1", "inf"): sympy.Integer(n),
        ("2", "1"): sympy.Integer(1),
        ("2", "inf"): sympy.sqrt(n),
        ("inf", "1"): sympy.Integer(1),
        ("inf", "2"): sympy.Integer(1),
    }
    return table[(b, a)]


def _measure(kind: str, n: int, r, w=None):
    """The area (n = 2) or volume (n = 3) of the open ball of radius r."""
    r = sympy.sympify(r)
    if kind == "w":
        return 2 * r**2 / (w[0] * w[1])
    if kind == "1":
        return 2 * r**2 if n == 2 else sympy.Rational(4, 3) * r**3
    if kind == "inf":
        return (2 * r) ** n
    return sympy.pi * r**2 if n == 2 else sympy.Rational(4, 3) * sympy.pi * r**3


# -- text pieces ---------------------------------------------------------------------------------------------------------


def _pt(p) -> X:
    s = "(" + ", ".join(str(v) for v in p) + ")"
    return X(s, s)


def _pts(ps) -> X:
    s = ", ".join(_pt(p).p for p in ps)
    return X(s, s)


def _mname(m) -> X:
    return X(_SYM[m[0]], _TEX[m[0]])


def _rat(r) -> X:
    r = Fr(r)
    if r.denominator == 1:
        return X(str(r.numerator), str(r.numerator))
    return X(f"{r.numerator}/{r.denominator}", rf"\frac{{{r.numerator}}}{{{r.denominator}}}")


def _coord_names(n: int):
    return "x", "y", "z"


def _definition(m, n: int) -> X | None:
    """The formula for a metric that is not one of the three standard ones (they are named in the stem)."""
    kind, w = m
    if kind == "w":
        p = f"d_w(x, y) = {w[0]}|x_1 - y_1| + {w[1]}|x_2 - y_2|"
        return X(p, p)
    if kind == "disc":
        return X(
            "d_0(x, y) = 0 if x = y and d_0(x, y) = 1 otherwise",
            "d_0(x,y)=0\\text{ if }x=y,\\ d_0(x,y)=1\\text{ otherwise}",
        )
    return None


def _metric_phrase(m, n: int, rng: random.Random, formula: bool = False):
    """Pieces naming metric m: 'the taxicab metric d_1' (with its formula when asked), or the definition."""
    kind = m[0]
    if kind in _BASIC:
        pieces = [_WORD[kind] + " ", _mname(m)]
        if formula:
            terms = {
                "1": " + ".join(f"|x_{i + 1} - y_{i + 1}|" for i in range(n)),
                "inf": "max(" + ", ".join(f"|x_{i + 1} - y_{i + 1}|" for i in range(n)) + ")",
                "2": "√(" + " + ".join(f"(x_{i + 1} - y_{i + 1})²" for i in range(n)) + ")",
            }[kind]
            tex = {
                "1": " + ".join(f"|x_{i + 1} - y_{i + 1}|" for i in range(n)),
                "inf": r"\max(" + ", ".join(f"|x_{i + 1} - y_{i + 1}|" for i in range(n)) + ")",
                "2": r"\sqrt{" + " + ".join(f"(x_{i + 1} - y_{i + 1})^2" for i in range(n)) + "}",
            }[kind]
            pieces += [", where ", X(f"{_SYM[kind]}(x, y) = {terms}", f"{_TEX[kind]}(x,y)={tex}")]
        return pieces
    return ["the metric ", _definition(m, n)]


def _rand_pt(rng: random.Random, n: int, lo: int, hi: int):
    return tuple(rng.randint(lo, hi) for _ in range(n))


# -- exact point samplers --------------------------------------------------------------------------------------------------


def _sample_points(rng, m, n: int, r: Fr, n_in: int, n_on: int, total: int, centre):
    """`total` distinct lattice points around `centre`: n_in strictly inside the open ball, n_on exactly on the sphere
    (fewer if the sphere has too few lattice points) and the rest outside it."""
    reach = int(r) + 3
    ins, ons, outs = [], [], []
    zero = (0,) * n
    for off in itertools.product(range(-reach, reach + 1), repeat=n):
        s = _side(m, off, zero, r)
        if off != zero:  # the centre itself is never a sampled point
            (ins if s < 0 else ons if s == 0 else outs).append(off)
    n_on = min(n_on, len(ons))
    n_in = min(n_in, len(ins))
    picked = rng.sample(ins, n_in) + rng.sample(ons, n_on) + rng.sample(outs, total - n_in - n_on)
    pts = [tuple(c + o for c, o in zip(centre, off)) for off in picked]
    rng.shuffle(pts)
    return pts


def _count_in(m, pts, c, r: Fr) -> int:
    return sum(1 for p in pts if _side(m, p, c, r) < 0)


def _count_closed(m, pts, c, r: Fr) -> int:
    return sum(1 for p in pts if _side(m, p, c, r) <= 0)


# -- easy: a distance -----------------------------------------------------------------------------------------------------

_PYTH = ((3, 4), (4, 3), (6, 8), (8, 6), (5, 12), (12, 5), (0, 5), (5, 0), (2, 0), (0, 7))


def _easy(rng: random.Random) -> dict:
    kind = rng.choices(("1", "2", "inf", "w"), (3, 4, 3, 1))[0]
    n = 3 if (kind != "w" and rng.random() < 0.3) else 2
    m = (kind, tuple(rng.sample((2, 3, 4, 5), 2)) if kind == "w" else None)
    for _ in attempts(200, "two distinct points"):
        p = _rand_pt(rng, n, -6, 6)
        if kind == "2" and n == 2 and rng.random() < 0.35:
            dx, dy = rng.choice(_PYTH)
            q = (p[0] + rng.choice((-1, 1)) * dx, p[1] + rng.choice((-1, 1)) * dy)
        else:
            q = _rand_pt(rng, n, -6, 6)
        if p != q and len({abs(a - b) for a, b in zip(p, q)}) >= 2 and sum(abs(a - b) for a, b in zip(p, q)) >= 4:
            break
    value = dist(m, p, q)
    diffs = [abs(a - b) for a, b in zip(p, q)]
    signed = [a - b for a, b in zip(p, q)]
    sq = sympy.Integer(sum(d * d for d in diffs))
    wrongs: list = []
    if kind == "1":
        wrongs += [
            (dist(("2", None), p, q), "mb-wrong-norm-formula"),
            (dist(("inf", None), p, q), "mb-wrong-norm-formula"),
        ]
        wrongs += [(sympy.Integer(abs(sum(signed))), "mb-dropped-absolute-values")]
        wrongs += [(sympy.Integer(sum(diffs[:-1])), "mb-dropped-coordinate")] if n == 3 else []
    elif kind == "2":
        wrongs += [(sq, "forgot-square-root"), (dist(("1", None), p, q), "mb-wrong-norm-formula")]
        wrongs += [(dist(("inf", None), p, q), "mb-wrong-norm-formula")]
        wrongs += (
            [(sympy.sqrt(sympy.Integer(sum(d * d for d in diffs[:-1]))), "mb-dropped-coordinate")] if n == 3 else []
        )
        wrongs += [(sympy.sqrt(sympy.Integer(abs(sum(signed)) ** 2)), "mb-dropped-absolute-values")]
    elif kind == "inf":
        wrongs += [
            (dist(("1", None), p, q), "mb-wrong-norm-formula"),
            (dist(("2", None), p, q), "mb-wrong-norm-formula"),
        ]
        wrongs += [(sympy.Integer(min(diffs)), "mb-max-min-swap")]
    else:
        w = m[1]
        wrongs += [(dist(("1", None), p, q), "mb-wrong-norm-formula")]
        wrongs += [(sympy.Integer(w[1] * diffs[0] + w[0] * diffs[1]), "mb-weights-swapped")]
        wrongs += [(sympy.Integer(max(w[0] * diffs[0], w[1] * diffs[1])), "mb-max-min-swap")]
    if kind == "1":
        working = f"d_1 = {' + '.join(str(d) for d in diffs)} = {value}."
    elif kind == "inf":
        working = f"d_∞ = max({', '.join(str(d) for d in diffs)}) = {value}."
    elif kind == "2":
        working = f"d_2 = √({' + '.join(f'{d}²' for d in diffs)}) = √{sq} = {atext(value)}."
    else:
        working = f"d_w = {m[1][0]}·{diffs[0]} + {m[1][1]}·{diffs[1]} = {value}."
    lead = rng.randrange(3)
    name = _metric_phrase(m, n, rng, formula=rng.random() < 0.35)
    if lead == 0:
        prompt = msay(
            "Find the distance ",
            X(f"{_SYM[kind]}(x, y)", f"{_TEX[kind]}(x,y)"),
            " between x = ",
            _pt(p),
            " and y = ",
            _pt(q),
            " in R",
            _sup(n),
            " under ",
            *name,
            ".",
        )
    elif lead == 1:
        prompt = msay(
            "In R",
            _sup(n),
            " with ",
            *name,
            ", how far apart are the points ",
            _pt(p),
            " and ",
            _pt(q),
            "? Give the exact value.",
        )
    else:
        prompt = msay(
            "Two points of R",
            _sup(n),
            " are ",
            _pt(p),
            " and ",
            _pt(q),
            ". Using ",
            *name,
            ", determine the distance from the first to the second.",
        )
    wrongs += [(value + 1, "dg-near-miss-value"), (value + 2, "dg-near-miss-value")] + (
        [(value - 1, "dg-near-miss-value")] if value > 1 else []
    )
    return single(prompt, value, [working], wrongs)


def _sup(n: int) -> str:
    return "²" if n == 2 else "³"


# -- medium: counting points in a ball, and comparing balls of different metrics -------------------------------------------


def _medium_count(rng: random.Random) -> dict:
    kind = rng.choices(("1", "2", "inf", "w", "disc"), (3, 3, 3, 1, 1))[0]
    n = 3 if (kind in _BASIC and rng.random() < 0.25) else 2
    m = (kind, tuple(rng.sample((2, 3, 4), 2)) if kind == "w" else None)
    c = _rand_pt(rng, n, -4, 4)
    total = 6
    if kind == "disc":
        r = rng.choice((Fr(1, 2), Fr(1), Fr(3, 2), Fr(2)))
        extra = 1 if rng.random() < 0.6 else 0
        others = set()
        for _ in attempts(200, "distinct points"):
            others.add(_rand_pt(rng, n, -4, 4))
            others.discard(c)
            if len(others) >= total - extra:
                break
        pts = sorted(others)[: total - extra] + ([c] if extra else [])
        rng.shuffle(pts)
    else:
        r = Fr(rng.randint(2, 4 if n == 3 else 7))
        n_on = rng.choice((1, 1, 2, 2, 0))
        pts = _sample_points(rng, m, n, r, min(rng.choice((0, 1, 2, 2, 3, 4)), total - 1 - n_on), n_on, total, c)
    key = _count_in(m, pts, c, r)
    closed = _count_closed(m, pts, c, r)
    wrongs = [(closed, "mb-closed-ball-count")]
    for other in _BASIC:
        if other != kind:
            wrongs.append((_count_in((other, None), pts, c, r), "mb-wrong-norm-formula"))
    wrongs += [(total - key, "mb-counted-outside"), (total, "mb-counted-outside")]
    wrongs += [
        (key + 1, "dg-near-miss-value"),
        (key - 1, "dg-near-miss-value"),
        (key + 2, "dg-near-miss-value"),
        (key - 2, "dg-near-miss-value"),
    ]
    wrongs = [(v, t) for v, t in wrongs if v >= 0]
    dists = [str(dist(m, p, c)) if kind != "2" else atext(dist(m, p, c)) for p in pts]
    working = [
        "Distances from c: " + ", ".join(f"{_pt(p).p} is {d}" for p, d in zip(pts, dists)) + ".",
        f"A point is in the open ball only if its distance is strictly less than {_rat(r).p}, so {key} of them are.",
    ]
    name = _metric_phrase(m, n, rng)
    ball = X(f"{{x : {_SYM[kind]}(x, c) < r}}", f"\\{{x : {_TEX[kind]}(x,c) < r\\}}")
    lead = rng.randrange(3)
    if lead == 0:
        prompt = msay(
            "Work in R",
            _sup(n),
            " with ",
            *name,
            ". Let c = ",
            _pt(c),
            " and r = ",
            _rat(r),
            ", and let B be the open ball ",
            ball,
            ". How many of the points ",
            _pts(pts),
            " lie in B?",
        )
    elif lead == 1:
        prompt = msay(
            "Take ",
            *name,
            " on R",
            _sup(n),
            ", the centre c = ",
            _pt(c),
            " and the radius r = ",
            _rat(r),
            ". Of the points ",
            _pts(pts),
            ", how many belong to the open ball ",
            ball,
            "?",
        )
    else:
        prompt = msay(
            "The open ball of radius ",
            _rat(r),
            " about c = ",
            _pt(c),
            " in ",
            *name,
            " is the set ",
            ball,
            ". Count the points among ",
            _pts(pts),
            " that it contains.",
        )
    out = dict(
        text=prompt[0],
        latex=prompt[1],
        answer=key,
        answer_text=str(key),
        answer_latex=str(key),
        answer_kind="number",
        working=working,
        distractors=int_dist(key, wrongs),
    )
    check(len(out["distractors"]) >= 3, "too few distractors for a count")
    return out


def _medium_compare(rng: random.Random) -> dict:
    n = 3 if rng.random() < 0.4 else 2
    b, a = rng.choice((("1", "2"), ("1", "inf"), ("2", "inf")))  # pairs where the comparison constant exceeds 1
    r = rng.randint(2, 12)
    ratio = _norm_ratio(b, a, n)  # sup ||x||_b / ||x||_a
    form = rng.randrange(3)
    c = _rand_pt(rng, n, -5, 5)
    cx = _pt(c)
    if form == 0:
        value = sympy.Rational(r) / ratio
        prompt = _compare_prompt(rng, "largest", a, b, n, cx, r)
        wrongs = [
            (sympy.Rational(r) * ratio, "mb-inverted-comparison-constant"),
            (sympy.Integer(r), "mb-ball-radius-not-scaled"),
            (sympy.Rational(r) / _wrong_dim_ratio(b, a, n), "mb-dimension-constant-wrong"),
            (sympy.Rational(r) * _wrong_dim_ratio(b, a, n), "mb-dimension-constant-wrong"),
            (sympy.Rational(r, 2), "mb-diameter-radius"),
        ]
        working = [
            f"{_SYM[a]}-balls of radius s lie inside the {_SYM[b]}-ball of radius r exactly when s·C ≤ r, where C is the largest value of ||x||_{_pp(b)}/||x||_{_pp(a)}, here C = {atext(ratio)}.",
            f"So the largest s is {r}/{atext(ratio)} = {atext(value)}.",
        ]
    elif form == 1:
        value = sympy.Rational(r) * ratio
        prompt = _compare_prompt(rng, "least", a, b, n, cx, r)
        wrongs = [
            (sympy.Rational(r) / ratio, "mb-inverted-comparison-constant"),
            (sympy.Integer(r), "mb-ball-radius-not-scaled"),
            (sympy.Rational(r) * _wrong_dim_ratio(b, a, n), "mb-dimension-constant-wrong"),
            (sympy.Rational(r) / _wrong_dim_ratio(b, a, n), "mb-dimension-constant-wrong"),
            (sympy.Integer(2 * r), "mb-diameter-radius"),
        ]
        working = [
            f"B_a(c, r) lies inside B_b(c, s) exactly when s ≥ r·C, where C is the largest value of ||x||_{_pp(b)}/||x||_{_pp(a)}, here C = {atext(ratio)}.",
            f"So the least s is {r}·{atext(ratio)} = {atext(value)}.",
        ]
    else:
        value = 2 * sympy.Rational(r) * ratio
        prompt = _compare_prompt(rng, "diameter", a, b, n, cx, r)
        wrongs = [
            (sympy.Rational(r) * ratio, "mb-diameter-radius"),
            (2 * sympy.Rational(r) / ratio, "mb-inverted-comparison-constant"),
            (sympy.Integer(2 * r), "mb-ball-radius-not-scaled"),
            (2 * sympy.Rational(r) * _wrong_dim_ratio(b, a, n), "mb-dimension-constant-wrong"),
            (sympy.Integer(r), "mb-ball-radius-not-scaled"),
        ]
        working = [
            f"The {_SYM[b]}-diameter of B_a(c, r) is 2r times the largest value of ||x||_{_pp(b)}/||x||_{_pp(a)}, here {atext(ratio)}.",
            f"So the diameter is 2·{r}·{atext(ratio)} = {atext(value)}.",
        ]
    wrongs += [(2 * value, "mb-diameter-radius"), (value / 2, "mb-diameter-radius")]
    return single(prompt, value, working, wrongs)


def _pp(kind: str) -> str:
    return {"1": "1", "2": "2", "inf": "∞"}[kind]


def _wrong_dim_ratio(b: str, a: str, n: int):
    """The comparison constant for the wrong dimension (2 for 3, 3 for 2): the usual slip."""
    return _norm_ratio(b, a, 5 - n)


def _ball(kind: str, centre: str, radius: str) -> X:
    return X(f"B_{_pp(kind)}({centre}, {radius})", f"B_{{{_TEXB[kind]}}}({centre},{radius})")


def _compare_prompt(rng, form: str, a: str, b: str, n: int, cx: X, r: int):
    """The prompt for comparing the a-ball and the b-ball about the centre c. `form` is largest / least / diameter."""
    na, nb = _WORD[a] + " ", _WORD[b] + " "
    ma, mb = _mname((a, None)), _mname((b, None))
    head = ["Let c = ", cx, " in R", _sup(n), ". "]
    if form == "largest":
        return msay(
            *head,
            "Write ",
            _ball(a, "c", "s"),
            " for the open ball of radius s about c under ",
            na,
            ma,
            " and ",
            _ball(b, "c", "r"),
            " for the open ball of radius r about c under ",
            nb,
            mb,
            ", with r = ",
            str(r),
            ". Find the largest s for which ",
            _ball(a, "c", "s"),
            " is contained in ",
            _ball(b, "c", "r"),
            ".",
        )
    if form == "least":
        return msay(
            *head,
            "Write ",
            _ball(a, "c", "r"),
            " for the open ball of radius r = ",
            str(r),
            " about c under ",
            na,
            ma,
            " and ",
            _ball(b, "c", "s"),
            " for the open ball of radius s about c under ",
            nb,
            mb,
            ". Find the least s for which ",
            _ball(a, "c", "r"),
            " is contained in ",
            _ball(b, "c", "s"),
            ".",
        )
    return msay(
        *head,
        "Let ",
        _ball(a, "c", "r"),
        " be the open ball of radius r = ",
        str(r),
        " about c under ",
        na,
        ma,
        ". Measure distances in it with ",
        nb,
        mb,
        ": find the supremum of ",
        X(f"{_SYM[b]}(x, y)", f"{_TEX[b]}(x,y)"),
        " over all points x and y of the ball (its diameter).",
    )


# -- hard ---------------------------------------------------------------------------------------------------------------------


def _hard_contain(rng: random.Random) -> dict:
    kind = rng.choices(("1", "2", "inf", "w"), (3, 3, 3, 1))[0]
    n = 3 if (kind != "w" and rng.random() < 0.3) else 2
    m = (kind, tuple(rng.sample((2, 3, 4), 2)) if kind == "w" else None)
    c = _rand_pt(rng, n, -3, 3)
    k = rng.choice((4, 4, 5))
    for _ in attempts(300, "a set with a clear farthest point"):
        pts = [p for p in {_rand_pt(rng, n, -6, 6) for _ in range(k)} if p != c]
        if len(pts) < k:
            continue
        ds = [dist(m, p, c) for p in pts]
        top = max(ds)
        if ds.count(top) == 1 and len(set(ds)) >= k - 1:
            break
    pts.sort()
    ds = [dist(m, p, c) for p in pts]
    value = max(ds)
    wrongs = [(min(ds), "mb-containment-nearest-point")]
    wrongs += [(max(dist((o, None), p, c) for p in pts), "mb-wrong-norm-formula") for o in _BASIC if o != kind]
    wrongs += [
        (sum(ds), "mb-containment-sum"),
        (value + 1, "mb-containment-open-closed"),
        (value / 2, "mb-diameter-radius"),
    ]
    working = [
        "Distances from c: " + ", ".join(f"{_pt(p).p} is {atext(d)}" for p, d in zip(pts, ds)) + ".",
        f"B(c, r) contains a point exactly when r is larger than its distance, so it contains all of S exactly when r > {atext(value)}.",
    ]
    nm = _metric_phrase(m, n, rng)
    if rng.random() < 0.5:
        prompt = msay(
            "In R",
            _sup(n),
            " with ",
            *nm,
            ", let c = ",
            _pt(c),
            " and S = {",
            _pts(pts),
            "}. The open ball B(c, r) contains every point of S exactly when r > r0. Find r0.",
        )
    else:
        prompt = msay(
            "Let c = ",
            _pt(c),
            " and S = {",
            _pts(pts),
            "} in R",
            _sup(n),
            " under ",
            *nm,
            ". What is the infimum of the radii r for which the open ball B(c, r) contains all of S?",
        )
    return single(prompt, value, working, wrongs)


def _shift(var: str, v: int) -> str:
    return var if v == 0 else f"{var} - {v}" if v > 0 else f"{var} + {-v}"


def _inequality(kind: str, c, rr: Fr) -> X:
    xs, ys = _shift("x", c[0]), _shift("y", c[1])
    rs = _rat(rr)
    if kind == "1":
        return X(f"|{xs}| + |{ys}| < {rs.p}", f"|{xs}| + |{ys}| < {rs.t}")
    if kind == "inf":
        return X(f"max(|{xs}|, |{ys}|) < {rs.p}", f"\\max(|{xs}|, |{ys}|) < {rs.t}")
    sq = _rat(rr * rr)
    return X(f"({xs})² + ({ys})² < {sq.p}", f"({xs})^2 + ({ys})^2 < {sq.t}")


def _hard_shell(rng: random.Random) -> dict:
    kind = rng.choice(_BASIC)
    c = _rand_pt(rng, 2, -4, 4)
    inner = sympy.Integer(rng.randint(1, 4))
    outer = inner + rng.randint(1, 5)
    unit = _measure(kind, 2, 1)
    value = unit * (outer**2 - inner**2)
    wrongs = [(unit * (outer - inner) ** 2, "mb-shell-difference-squared"), (unit * outer**2, "mb-shell-outer-only")]
    wrongs += [(_measure(k, 2, 1) * (outer**2 - inner**2), "mb-area-wrong-shape") for k in _BASIC if k != kind]
    wrongs += [(2 * value, "mb-area-side-not-diameter")]
    m = (kind, None)
    prompt = msay(
        "In R² under ",
        *_metric_phrase(m, 2, rng),
        ", let A be the set of points x with ",
        str(inner),
        " ≤ ",
        X(f"{_SYM[kind]}(x, c)", f"{_TEX[kind]}(x,c)"),
        " < ",
        str(outer),
        ", where c = ",
        _pt(c),
        ". Find the Euclidean area of A. Give the exact value.",
    )
    working = [
        f"A ball of radius R has area {atext(unit)}R², so A has area {atext(unit)}({outer}² − {inner}²) = {atext(value)}."
    ]
    return single(prompt, value, working, wrongs)


def _hard_weighted(rng: random.Random) -> dict:
    w = tuple(rng.sample((2, 3, 4, 5), 2))
    c = _rand_pt(rng, 2, -4, 4)
    rr = Fr(rng.randint(1, 12), rng.choice((1, 2)))
    r = sympy.Rational(rr.numerator, rr.denominator)
    value = _measure("w", 2, r, w)
    wrongs = [
        (_measure("1", 2, r), "mb-area-wrong-shape"),
        (4 * r**2 / (w[0] * w[1]), "mb-area-side-not-diameter"),
        (r**2 / (w[0] * w[1]), "mb-area-diamond-no-half"),
        (2 * r**2 * w[0] * w[1], "mb-weights-multiplied"),
        (2 * r**2 / (w[0] + w[1]), "mb-weights-added"),
    ]
    d = f"d_w(x, y) = {w[0]}|x_1 - y_1| + {w[1]}|x_2 - y_2|"
    prompt = msay(
        "On R² define ",
        X(d, d),
        ". Find the area of the open ball ",
        X("{x : d_w(x, c) < r}", r"\{x : d_w(x,c) < r\}"),
        " about c = ",
        _pt(c),
        " when r = ",
        _rat(rr),
        ". Give the exact value.",
    )
    working = [
        f"The ball is a diamond reaching {atext(r / w[0])} from the centre along the first axis and {atext(r / w[1])} along the second, so its area is 2·({atext(r / w[0])})({atext(r / w[1])}) = {atext(value)}."
    ]
    return single(prompt, value, working, wrongs)


def _hard_ball_measure(rng: random.Random) -> dict:
    form = rng.choices(("named", "setbuilder", "volume"), (3, 3, 2))[0]
    n = 3 if form == "volume" else 2
    kind = rng.choice(_BASIC)
    c = _rand_pt(rng, n, -4, 4)
    rr = Fr(rng.randint(1, 12), rng.choice((1, 2, 2)))
    r = sympy.Rational(rr.numerator, rr.denominator)
    value = _measure(kind, n, r)
    wrongs = [(_measure(k, n, r), "mb-area-wrong-shape") for k in _BASIC if k != kind]
    wrongs += [(_measure(kind, n, 2 * r), "mb-area-side-not-diameter"), (value / 2, "mb-area-diamond-no-half")]
    if kind == "inf":
        wrongs += [(r**n, "mb-area-side-not-diameter")]
    if kind == "1" and n == 2:
        wrongs += [(r**2, "mb-area-diamond-no-half")]
    word = "area" if n == 2 else "volume"
    if form == "setbuilder":
        prompt = msay(
            "Find the area of the region of the plane given by ", _inequality(kind, c, rr), ". Give the exact value."
        )
    else:
        m = (kind, None)
        prompt = msay(
            "In R",
            _sup(n),
            " under ",
            *_metric_phrase(m, n, rng),
            ", find the ",
            word,
            " of the open ball ",
            X(f"{{x : {_SYM[kind]}(x, c) < r}}", f"\\{{x : {_TEX[kind]}(x,c) < r\\}}"),
            " about c = ",
            _pt(c),
            " when r = ",
            _rat(rr),
            ". Give the exact value.",
        )
    working = [
        f"The open ball of radius r has {word} {atext(_measure(kind, n, sympy.Symbol('r')))}; with r = {rr} this is {atext(value)}."
    ]
    return single(prompt, value, working, wrongs)


def _hard_vertices(rng: random.Random) -> dict:
    kind = rng.choice(("1", "inf"))
    cx, cy = _rand_pt(rng, 2, -5, 5)
    r = rng.randint(2, 14)
    if kind == "1":
        vs = [(cx + r, cy), (cx, cy + r), (cx - r, cy), (cx, cy - r)]
    else:
        vs = [(cx + r, cy + r), (cx - r, cy + r), (cx - r, cy - r), (cx + r, cy - r)]
    wrongs = [
        (sympy.Integer(2 * r), "mb-radius-from-side"),
        (sympy.sqrt(2) * r, "mb-radius-from-side"),
        (sympy.Rational(r, 2), "mb-radius-from-side"),
        (sympy.Integer(r + 1), "mb-ball-shape-mixup"),
        (sympy.sqrt(2) * r / 2, "mb-ball-shape-mixup"),
    ]
    if rng.random() < 0.5:
        prompt = msay(
            "The open square with vertices ",
            _pts(sorted(vs)),
            " is the open ball B(c, r) = {x : d(x, c) < r} about its centre c for exactly one of the metrics d_1, d_2, d_∞ on R². Find the radius r.",
        )
    else:
        prompt = msay(
            "Exactly one of d_1, d_2, d_∞ makes the open square with vertices ",
            _pts(sorted(vs)),
            " an open ball about its centre. Find that ball's radius.",
        )
    if kind == "1":
        how = "the vertices lie on the axes through the centre, at distance r from it: a d_1 ball is a diamond (a square turned through 45°)"
    else:
        how = "the sides are parallel to the axes and each is 2r long: a d_∞ ball is an axis-parallel square"
    working = [f"The centre is ({cx}, {cy}); {how}.", f"So r = {r}."]
    return single(prompt, sympy.Integer(r), working, wrongs)


def _hard(rng: random.Random) -> dict:
    fn = rng.choices((_hard_contain, _hard_ball_measure, _hard_shell, _hard_weighted, _hard_vertices), (4, 4, 2, 2, 3))[
        0
    ]
    return fn(rng)


def _medium(rng: random.Random) -> dict:
    return rng.choices((_medium_count, _medium_compare), (4, 5))[0](rng)


# -- the generators ------------------------------------------------------------------------------------------------------------


@generator(
    TOPIC,
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def w5b_metric_balls(rng: random.Random, tier: str = "easy") -> dict:
    check(tier in ("easy", "medium", "hard"), f"unknown tier {tier!r}")
    return {"easy": _easy, "medium": _medium, "hard": _hard}[tier](rng)


@generator(EXAM_TOPIC, level=19, presets={"exam": dict(tier="exam")})
def exd_x_w5b_metric_balls(rng: random.Random, tier: str = "exam") -> dict:
    """A five-part chain: two distances, a membership count, a containment radius and the measure of that ball."""
    check(tier == "exam", f"unknown tier {tier!r}")
    n = 3 if rng.random() < 0.35 else 2
    m1, m2, m3 = (rng.sample(_BASIC, 2) + [rng.choice(_BASIC)])[:3]
    m4 = rng.choice([k for k in _BASIC if k != m3] if rng.random() < 0.7 else list(_BASIC))
    P = _rand_pt(rng, n, -4, 4)
    for _ in attempts(100, "P distinct from Q"):
        Q = _rand_pt(rng, n, -5, 5)
        if Q != P and dist(("1", None), P, Q) != dist(("inf", None), P, Q):
            break
    r_c = Fr(rng.randint(2, 4 if n == 3 else 6))
    mm3 = (m3, None)
    n_on = rng.choice((1, 1, 2, 0))
    pts = _sample_points(rng, mm3, n, r_c, min(rng.choice((0, 1, 2, 2, 3)), 4 - n_on), n_on, 5, P)
    key_c = _count_in(mm3, pts, P, r_c)
    mm4 = (m4, None)
    ds4 = [dist(mm4, p, P) for p in pts]
    r0 = max(ds4)
    area = _measure(m4, n, r0)
    word = "area" if n == 2 else "volume"
    d1, d2 = dist((m1, None), P, Q), dist((m2, None), P, Q)
    t = _SYM
    metrics = [
        "the metrics ",
        X("d_1", "d_1"),
        " (taxicab), ",
        X("d_2", "d_2"),
        " (Euclidean) and ",
        X("d_∞", r"d_\infty"),
        " (maximum)",
    ]
    ball = X("B_d(P, r) = {x : d(x, P) < r}", r"B_d(P,r)=\{x : d(x,P) < r\}")
    shape = rng.randrange(3)
    if shape == 0:
        stem = msay(
            "In R",
            _sup(n),
            " let P = ",
            _pt(P),
            " and Q = ",
            _pt(Q),
            ", and let S = {",
            _pts(pts),
            "}. For a metric d write ",
            ball,
            " for the open ball about P. The metric d is one of ",
            *metrics,
            ".",
        )
    elif shape == 1:
        stem = msay(
            "Consider the points P = ",
            _pt(P),
            " and Q = ",
            _pt(Q),
            " of R",
            _sup(n),
            " and the set S = {",
            _pts(pts),
            "}. Three metrics are in use: ",
            *metrics,
            ". The open ball of radius r about P in a metric d is ",
            ball,
            ".",
        )
    else:
        stem = msay(
            "A set S = {",
            _pts(pts),
            "} and two points P = ",
            _pt(P),
            " and Q = ",
            _pt(Q),
            " lie in R",
            _sup(n),
            ". We use ",
            *metrics,
            ", and write ",
            ball,
            " for the open ball about P.",
        )
    diffs = [abs(a - b) for a, b in zip(P, Q)]
    a_part = Link(
        msay("Find ", X(f"{t[m1]}(P, Q)", f"{_TEX[m1]}(P,Q)"), "."),
        d1,
        "expression",
        1,
        [f"Coordinate differences are {', '.join(str(d) for d in diffs)}; {t[m1]}(P, Q) = {atext(d1)}."],
        _distance_wrongs(m1, P, Q),
        sensible=None,
    )
    b_part = Link(
        msay("Find ", X(f"{t[m2]}(P, Q)", f"{_TEX[m2]}(P,Q)"), "."),
        d2,
        "expression",
        1,
        [f"{t[m2]}(P, Q) = {atext(d2)}."],
        _distance_wrongs(m2, P, Q),
    )
    wrongs_c = [(_count_closed(mm3, pts, P, r_c), "mb-closed-ball-count")]
    wrongs_c += [(_count_in((o, None), pts, P, r_c), "mb-wrong-norm-formula") for o in _BASIC if o != m3]
    wrongs_c += [(5 - key_c, "mb-counted-outside"), (5, "mb-counted-outside")]
    wrongs_c += [
        (key_c + 1, "dg-near-miss-value"),
        (key_c - 1, "dg-near-miss-value"),
        (key_c + 2, "dg-near-miss-value"),
    ]
    wrongs_c = [(v, t) for v, t in wrongs_c if v >= 0]
    c_part = Link(
        msay("How many points of S lie in ", _ball(m3, "P", str(r_c)), "?"),
        sympy.Integer(key_c),
        "number",
        2,
        [
            f"Distances from P under {t[m3]} are {', '.join(atext(dist(mm3, p, P)) for p in pts)}; those strictly below {r_c} count."
        ],
        wrongs_c,
    )
    wrongs_d = [(min(ds4), "mb-containment-nearest-point")]
    wrongs_d += [(max(dist((o, None), p, P) for p in pts), "mb-wrong-norm-formula") for o in _BASIC if o != m4]
    wrongs_d += [(sum(ds4), "mb-containment-sum"), (r0 + 1, "mb-containment-open-closed")]
    d_part = Link(
        msay("The ball ", _ball(m4, "P", "r"), " contains every point of S exactly when r > r0. Find r0."),
        r0,
        "expression",
        2,
        [f"Distances from P under {t[m4]} are {', '.join(atext(d) for d in ds4)}; the largest is {atext(r0)}."],
        wrongs_d,
        sensible=None,
    )
    ft = {
        ("1", 2): "2*part_d**2",
        ("inf", 2): "4*part_d**2",
        ("2", 2): "pi*part_d**2",
        ("1", 3): "4/3*part_d**3",
        ("inf", 3): "8*part_d**3",
        ("2", 3): "4/3*pi*part_d**3",
    }[(m4, n)]
    wrongs_e = [(_measure(k, n, r0), "mb-area-wrong-shape") for k in _BASIC if k != m4]
    wrongs_e += [
        (_measure(m4, n, 2 * r0), "mb-area-side-not-diameter"),
        (_measure(m4, n, r0) / 2, "mb-area-diamond-no-half"),
    ]
    e_part = Link(
        msay("Hence find the ", word, " of the open ball ", _ball(m4, "P", "r0"), ". Give the exact value."),
        area,
        "expression",
        2,
        [
            f"The ball of radius r has {word} {atext(_measure(m4, n, sympy.Symbol('r')))}; at r = {atext(r0)} that is {atext(area)}."
        ],
        wrongs_e,
        ft=ft,
    )
    return exam_dict(stem, [a_part, b_part, c_part, d_part, e_part])


def _distance_wrongs(kind: str, P, Q):
    diffs = [abs(a - b) for a, b in zip(P, Q)]
    signed = [a - b for a, b in zip(P, Q)]
    out = []
    for o in _BASIC:
        if o != kind:
            out.append((dist((o, None), P, Q), "mb-wrong-norm-formula"))
    out.append(
        (sympy.Integer(sum(d * d for d in diffs)), "forgot-square-root")
        if kind == "2"
        else (sympy.Integer(min(diffs)), "mb-max-min-swap")
    )
    out.append((sympy.Integer(abs(sum(signed))), "mb-dropped-absolute-values"))
    return out
