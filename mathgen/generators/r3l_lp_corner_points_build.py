"""Generates `uc2-lp-formulation-and-corner-points` questions (easy, medium, hard) and the exam chain
`exd_x_r3l_lp_corner_points`.

``uc2-lp-formulation-and-corner-points`` (``optimisation``, answer ``number``): a two-variable linear programme told as
a story. The learner writes the constraints and the objective, finds the corner points of the feasible region and reads
the optimum off them. Every corner point has integer coordinates, so every key is an integer.

* easy: a maximisation with two resource limits. The greatest profit.
* medium: a minimisation (a feed or diet mix) with three minimum-requirement constraints, so the feasible region is
  unbounded and some of the intersections of constraint lines are not corner points. The least cost.
* hard: a maximisation with three resource limits and an unknown unit profit k on the second product. The value of k
  at which every plan along the edge of one named resource limit is equally good (the objective's gradient is
  parallel to that constraint's normal).
* exam (``exd_x_r3l_lp_corner_points``): a three-constraint maximisation. (a) the corner points (a ``points`` answer),
  (b) the optimal x, (c) the optimal y, (d) the greatest profit (follows through from (b) and (c)).

Corner points come from intersecting every pair of constraint lines and the axes and keeping the feasible ones
(`_r3l_kit.corner_points`). The optimum is re-derived by a brute-force scan of every integer point of a bounding box
(maximisation) or minimisation: for two variables the optimum of a linear objective over a polygon is at a corner,
and the scan confirms that no lattice point of a big box does better than the best corner.
"""

from __future__ import annotations

from fractions import Fraction
from math import gcd

from ..answers import Distractor
from ..core import attempts, check, generator, pick_distractors
from ..grading.typed import points_answer_text
from ._r3l_kit import (
    all_intersections,
    corner_points,
    feasible,
    linear_text,
    objective,
)

# (who, x name, y name, period, [(resource, unit singular, unit plural)])
_MAX_SCENARIOS = (
    (
        "A bakery in Fremantle",
        "tray of sourdough loaves",
        "tray of fruit buns",
        "day",
        [("Oven time", "hour", "hours"), ("Preparation time", "hour", "hours"), ("Flour", "kilogram", "kilograms")],
        ("trays of sourdough loaves", "trays of fruit buns"),
    ),
    (
        "A furniture workshop in Launceston",
        "table",
        "chair",
        "week",
        [
            ("Cutting time", "hour", "hours"),
            ("Sanding time", "hour", "hours"),
            ("Timber", "square metre", "square metres"),
        ],
        ("tables", "chairs"),
    ),
    (
        "A craft brewery in Brisbane",
        "keg of lager",
        "keg of pale ale",
        "week",
        [("Tank time", "hour", "hours"), ("Bottling time", "hour", "hours"), ("Malt", "kilogram", "kilograms")],
        ("kegs of lager", "kegs of pale ale"),
    ),
    (
        "A jeweller in Adelaide",
        "silver ring",
        "silver pendant",
        "month",
        [("Silver", "gram", "grams"), ("Polishing time", "hour", "hours"), ("Casting time", "hour", "hours")],
        ("silver rings", "silver pendants"),
    ),
)
# (who, x name, y name, [(requirement, unit singular, unit plural)], (x plural, y plural))
_MIN_SCENARIOS = (
    (
        "A dairy farmer near Colac mixes two feed supplements",
        "bag of supplement A",
        "bag of supplement B",
        [("Protein", "gram", "grams"), ("Calcium", "gram", "grams"), ("Fibre", "gram", "grams")],
        ("bags of supplement A", "bags of supplement B"),
    ),
    (
        "A market gardener in the Swan Valley blends two fertilisers",
        "bag of fertiliser A",
        "bag of fertiliser B",
        [
            ("Nitrogen", "kilogram", "kilograms"),
            ("Phosphorus", "kilogram", "kilograms"),
            ("Potassium", "kilogram", "kilograms"),
        ],
        ("bags of fertiliser A", "bags of fertiliser B"),
    ),
    (
        "A stockfeed supplier in Dubbo makes a mix from two grains",
        "tonne of grain A",
        "tonne of grain B",
        [("Energy", "unit", "units"), ("Protein", "unit", "units"), ("Minerals", "unit", "units")],
        ("tonnes of grain A", "tonnes of grain B"),
    ),
)


def _qty(n, unit) -> str:
    return f"{n} {unit[1] if n == 1 else unit[2]}"


def _scan_best(constraints, coefs, sense, box):
    """The best objective value over every lattice point of [0, box]^2 that is feasible (an independent check)."""
    best = None
    rows = [(int(a), int(b), int(r), s == "<=") for a, b, r, s in constraints]
    p, q = int(coefs[0]), int(coefs[1])
    for x in range(box + 1):
        for y in range(box + 1):
            if all((a * x + b * y <= r) if le else (a * x + b * y >= r) for a, b, r, le in rows):
                v = p * x + q * y
                if best is None or (v > best if sense == "max" else v < best):
                    best = v
    return best


def _draw_region(rng, m, mode):
    """(constraints, corner points) with every corner point integral, built edge by edge.

    A maximisation region has the frontier V0 = (0, Y0), V1, ..., Vm = (X, 0), each edge a positive multiple t of the
    direction (b, -a) of a constraint a x + b y <= r, the slopes steepening as x grows (a/b increasing). A
    minimisation region walks the same way with the slopes flattening (a/b decreasing) and ">=" constraints.
    Every vertex is an integer point, every constraint is a real edge, and the constraints come out in random
    order."""
    sense = "<=" if mode == "max" else ">="
    for _ in attempts(2000, "a lattice-cornered region"):
        dirs = sorted({(rng.randint(1, 6), rng.randint(1, 6)) for _ in range(m + 2)}, key=lambda ab: Fraction(*ab))
        dirs = [d for d in dirs if gcd(*d) == 1]
        if len(dirs) < m:
            continue
        chosen = sorted(rng.sample(dirs, m), key=lambda ab: Fraction(*ab), reverse=(mode == "min"))
        ts = [rng.randint(1, 4) for _ in range(m)]
        y0 = sum(t * a for t, (a, _b) in zip(ts, chosen))
        x_end = sum(t * b for t, (_a, b) in zip(ts, chosen))
        if y0 > 36 or x_end > 36:
            continue
        verts = [(0, y0)]
        for t, (a, b) in zip(ts, chosen):
            verts.append((verts[-1][0] + t * b, verts[-1][1] - t * a))
        cons = []
        for (a, b), v in zip(chosen, verts):
            cons.append((a, b, a * v[0] + b * v[1], sense))
        if max(r for _a, _b, r, _s in cons) > 90:
            continue
        pts = corner_points(cons)
        check(
            all(p[0].denominator == 1 and p[1].denominator == 1 for p in pts)
            and len(pts) == (m + 2 if mode == "max" else m + 1),
            "staircase region has the wrong corner points",
        )
        rng.shuffle(cons)
        return cons, pts
    raise AssertionError  # pragma: no cover


def _draw_objective(rng, pts, sense):
    """Positive integer unit values (p, q) with a unique best corner point."""
    for _ in attempts(500, "an objective with a unique optimum"):
        p, q = rng.randint(2, 9), rng.randint(2, 9)
        if p == q:
            continue
        vals = sorted(((p * x + q * y), (x, y)) for x, y in pts)
        best = vals[-1][0] if sense == "max" else vals[0][0]
        if sum(1 for v, _ in vals if v == best) != 1:
            continue
        runner = vals[-2][0] if sense == "max" else vals[1][0]
        if abs(best - runner) < 2:
            continue
        return p, q
    raise AssertionError  # pragma: no cover


def _best_point(pts, coefs, sense):
    pick = max if sense == "max" else min
    return pick(pts, key=lambda p: objective(coefs, p))


def _int(v) -> int:
    v = Fraction(v)
    check(v.denominator == 1, "non-integral value")
    return int(v)


def _constraint_text(cons, names=("x", "y")):
    return "; ".join(f"{linear_text([a, b], names)} {'≤' if s == '<=' else '≥'} {r}" for a, b, r, s in cons)


# -- stems --------------------------------------------------------------------------------------------------------------


def _max_story(rng, cons, scenario, p, q, ask_k=False):
    who, xn, yn, period, resources, _ = scenario
    lines = [f"{who} makes two products each {period}: a {xn} and a {yn}."]
    for (a, b, r, _), (res, u1, u2) in zip(cons, resources):
        unit = (None, u1, u2)
        lines.append(
            f"{res}: each {xn} takes {_qty(a, unit)} and each {yn} takes {_qty(b, unit)}, with {_qty(r, unit)} "
            f"available each {period}."
        )
    return lines


def _easy(rng):
    scenario = rng.choice(_MAX_SCENARIOS)
    cons, pts = _draw_region(rng, 2, "max")
    p, q = _draw_objective(rng, pts, "max")
    best = _best_point(pts, (p, q), "max")
    value = _int(objective((p, q), best))
    check(_scan_best(cons, (p, q), "max", 45) == value, "lattice scan disagrees with the corner-point optimum")
    who, xn, yn, period, _res, (xs, ys) = scenario
    lines = _max_story(rng, cons, scenario, p, q)
    lines.append(
        f"Profit is ${p} on each {xn} and ${q} on each {yn}. Find the greatest profit that can be made in a {period}."
    )
    wrong = []
    others = [pt for pt in pts if pt != best]
    for pt in sorted(others, key=lambda t: -objective((p, q), t)):
        wrong.append(
            Distractor(str(_int(objective((p, q), pt))), "r3l-lp-nonoptimal-corner", _int(objective((p, q), pt)))
        )
    inter = [
        pt for pt in all_intersections(cons) if pt not in pts and pt[0].denominator == 1 and pt[1].denominator == 1
    ]
    for pt in inter:
        v = _int(objective((p, q), pt))
        wrong.insert(1, Distractor(str(v), "r3l-lp-infeasible-vertex", v))
    swapped = _int(max(objective((q, p), pt) for pt in pts))
    wrong.insert(2, Distractor(str(swapped), "r3l-lp-profits-swapped", swapped))
    distractors = pick_distractors(str(value), wrong, 4)
    check(len(distractors) >= 3, "too few distractors")
    working = [
        f"Let x be the number of {xs} and y the number of {ys}. The constraints are {_constraint_text(cons)}, with x, y ≥ 0; maximise {linear_text([p, q], ('x', 'y'))}.",
        "Corner points: " + ", ".join(f"({_int(x)}, {_int(y)})" for x, y in pts) + ".",
        f"The objective is greatest at ({_int(best[0])}, {_int(best[1])}), where it equals {value}.",
    ]
    return dict(
        text=" ".join(lines),
        answer_kind="number",
        answer=value,
        answer_text=str(value),
        answer_latex=str(value),
        working=working,
        distractors=distractors,
    )


def _medium(rng):
    who, xn, yn, reqs, (xs, ys) = rng.choice(_MIN_SCENARIOS)
    cons, pts = _draw_region(rng, 3, "min")
    p, q = _draw_objective(rng, pts, "min")
    best = _best_point(pts, (p, q), "min")
    value = _int(objective((p, q), best))
    # lattice scan: the least cost over lattice points of a big box (corner values are integers)
    check(_scan_best(cons, (p, q), "min", 70) == value, "lattice scan disagrees with the corner-point optimum")
    lines = [f"{who} each day."]
    for (a, b, r, _), (res, u1, u2) in zip(cons, reqs):
        unit = (None, u1, u2)
        lines.append(
            f"{res}: each {xn} gives {_qty(a, unit)} and each {yn} gives {_qty(b, unit)}; at least {_qty(r, unit)} are needed."
        )
    lines.append(
        f"A {xn} costs ${p} and a {yn} costs ${q}. Find the least daily cost of a mix that meets every requirement."
    )
    wrong = []
    for pt in sorted((t for t in pts if t != best), key=lambda t: objective((p, q), t)):
        v = _int(objective((p, q), pt))
        wrong.append(Distractor(str(v), "r3l-lp-nonoptimal-corner", v))
    for pt in all_intersections(cons):
        if pt not in pts and pt[0].denominator == 1 and pt[1].denominator == 1:
            v = _int(objective((p, q), pt))
            if 0 < v < value:
                wrong.insert(0, Distractor(str(v), "r3l-lp-infeasible-vertex", v))
    top = _int(max(objective((p, q), t) for t in pts))
    wrong.append(Distractor(str(top), "r3l-lp-maximised-not-minimised", top))
    distractors = pick_distractors(str(value), wrong, 4)
    check(len(distractors) >= 2, "too few distractors")
    working = [
        f"Let x be the number of {xs} and y the number of {ys}. Minimise {linear_text([p, q], ('x', 'y'))} subject to {_constraint_text(cons)}, x, y ≥ 0.",
        "Corner points of the (unbounded) feasible region: "
        + ", ".join(f"({_int(x)}, {_int(y)})" for x, y in pts)
        + ".",
        f"The cost is least at ({_int(best[0])}, {_int(best[1])}): ${value}.",
    ]
    return dict(
        text=" ".join(lines),
        answer_kind="number",
        answer=value,
        answer_text=str(value),
        answer_latex=str(value),
        working=working,
        distractors=distractors,
    )


def _hard(rng):
    scenario = rng.choice(_MAX_SCENARIOS)
    who, xn, yn, period, resources, _ = scenario
    for _ in attempts(500, "a region with the named edge"):
        cons, pts = _draw_region(rng, 3, "max")
        i = rng.randrange(3)
        a, b, r, _s = cons[i]
        t = rng.randint(1, 3)
        p, k = a * t, b * t
        if a == b or b < 2 or p > 30 or k > 30:
            continue
        # the edge on constraint i is optimal exactly when (p, k) is parallel to (a, b)
        edge = [pt for pt in pts if a * pt[0] + b * pt[1] == r]
        if len(edge) < 2:
            continue
        # independent check: at k all edge points tie and beat every other corner point; off k the optimum is unique
        vals = {pt: objective((p, k), pt) for pt in pts}
        top = max(vals.values())
        if {pt for pt, v in vals.items() if v == top} != set(edge):
            continue
        unique_off = True
        for dk in (Fraction(-1, 2), Fraction(1, 2)):
            v2 = [objective((p, k + dk), pt) for pt in pts]
            unique_off = unique_off and v2.count(max(v2)) == 1
        if unique_off:
            break
    else:  # pragma: no cover
        raise AssertionError
    res, u1, u2 = resources[i]
    lines = _max_story(rng, cons, scenario, p, k)
    lines.append(
        f"Profit is ${p} on each {xn} and ${'k'} on each {yn}. For one particular value of k, every plan along the whole "
        f"edge of the feasible region on the {res.lower()} limit gives the same greatest profit. Find k."
    )
    wrong_vals = [
        (Fraction(p * a, b), "r3l-lp-isoprofit-normal-swapped"),
        (Fraction(r, b), "r3l-lp-intercept-for-profit"),
        (Fraction(p * b, a) + 1, None),
        (Fraction(p, 1), "r3l-lp-isoprofit-normal-swapped"),
        (Fraction(a * b), None),
    ]
    wrong = []
    for v, tag in wrong_vals:
        if v.denominator == 1 and v > 0:
            wrong.append(Distractor(str(int(v)), tag, int(v)))
    distractors = pick_distractors(str(k), wrong, 4)
    check(len(distractors) >= 2, "too few distractors")
    working = [
        f"The {res.lower()} limit is {linear_text([a, b], ('x', 'y'))} ≤ {r}. Profit {p}x + ky is constant along that edge when its coefficients ({p}, k) are parallel to ({a}, {b}), so {p}/{a} = k/{b}.",
        f"So k = {p}·{b}/{a} = {k}.",
    ]
    return dict(
        text=" ".join(lines),
        answer_kind="number",
        answer=k,
        answer_text=str(k),
        answer_latex=str(k),
        working=working,
        distractors=distractors,
    )


@generator(
    "r3l_lp_corner_points",
    level=20,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r3l_lp_corner_points(rng, tier="easy"):
    """A two-variable linear programme from a story; the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam ---------------------------------------------------------------------------------------------------------------


@generator("exd_x_r3l_lp_corner_points", level=20, presets={"exam": dict(tier="exam")})
def exd_x_r3l_lp_corner_points(rng, tier="exam"):
    """Exam chain: corner points, optimal x, optimal y, greatest profit (follows through from x and y)."""
    check(tier == "exam", f"unknown tier {tier!r}")
    scenario = rng.choice(_MAX_SCENARIOS)
    who, xn, yn, period, _res, (xs, ys) = scenario
    for _ in attempts(200, "an interior optimum"):
        cons, pts = _draw_region(rng, 3, "max")
        p, q = _draw_objective(rng, pts, "max")
        best = _best_point(pts, (p, q), "max")
        if best[0] > 0 and best[1] > 0:
            break
    bx, by = _int(best[0]), _int(best[1])
    value = _int(objective((p, q), best))
    check(_scan_best(cons, (p, q), "max", 45) == value, "lattice scan disagrees with the corner-point optimum")
    lines = _max_story(rng, cons, scenario, p, q)
    lines.append(
        f"Profit is ${p} on each {xn} and ${q} on each {yn}. Let x be the number of {xs} and y the number of {ys} made "
        f"each {period}."
    )
    pt_value = [[_int(x), _int(y)] for x, y in pts]
    a_text = points_answer_text(pt_value)
    infeasible = [
        t for t in all_intersections(cons) if t not in pts and t[0].denominator == 1 and t[1].denominator == 1
    ]
    cands = []
    if infeasible:
        extra = [[_int(x), _int(y)] for x, y in sorted(pts + [infeasible[0]])]
        cands.append(Distractor(points_answer_text(extra), "r3l-lp-infeasible-vertex", extra))
    cons_only = [[_int(x), _int(y)] for x, y in pts if x > 0 and y > 0]
    if cons_only and len(cons_only) < len(pt_value):
        cands.append(Distractor(points_answer_text(cons_only), "r3l-lp-axis-corners-forgotten", cons_only))
    origin_less = [v for v in pt_value if v != [0, 0]]
    cands.append(Distractor(points_answer_text(origin_less), "r3l-lp-corner-omitted", origin_less))
    other = [v for v in pt_value if v != [bx, by]]
    cands.append(Distractor(points_answer_text(other), "r3l-lp-corner-omitted", other))
    a_dist = pick_distractors(a_text, cands, 3)
    runner = sorted((t for t in pts if t != best), key=lambda t: -objective((p, q), t))
    b_dist = pick_distractors(
        str(bx),
        [Distractor(str(_int(t[0])), "r3l-lp-nonoptimal-corner", _int(t[0])) for t in runner]
        + [Distractor(str(by), "r3l-lp-coordinates-swapped", by)],
        3,
    )
    c_dist = pick_distractors(
        str(by),
        [Distractor(str(_int(t[1])), "r3l-lp-nonoptimal-corner", _int(t[1])) for t in runner]
        + [Distractor(str(bx), "r3l-lp-coordinates-swapped", bx)],
        3,
    )
    d_dist = pick_distractors(
        str(value),
        [
            Distractor(str(_int(objective((p, q), t))), "r3l-lp-nonoptimal-corner", _int(objective((p, q), t)))
            for t in runner
        ]
        + [Distractor(str(q * bx + p * by), "r3l-lp-profits-swapped", q * bx + p * by)],
        3,
    )
    check(all(len(x) >= 2 for x in (a_dist, b_dist, c_dist, d_dist)), "too few distractors")
    return dict(
        text=" ".join(lines),
        parts=[
            dict(
                prompt="Write the constraints as inequalities in x and y and list all the corner points of the feasible region.",
                answer_kind="points",
                answer=pt_value,
                answer_text=a_text,
                marks=3,
                working=[
                    f"Constraints: {_constraint_text(cons)}, x, y ≥ 0.",
                    "Intersect each pair of boundary lines and keep the points that satisfy every constraint.",
                ],
                distractors=a_dist,
            ),
            dict(
                prompt=f"Maximise the profit {linear_text([p, q], ('x', 'y'))} over the corner points. How many {xs} should be made?",
                answer_kind="number",
                answer=bx,
                answer_text=str(bx),
                marks=2,
                sensible_range=(0, None),
                working=[f"The profit is greatest at ({bx}, {by}), so x = {bx}."],
                distractors=b_dist,
            ),
            dict(
                prompt=f"How many {ys} should be made at the optimal plan?",
                answer_kind="number",
                answer=by,
                answer_text=str(by),
                marks=1,
                sensible_range=(0, None),
                working=[f"At the optimal corner ({bx}, {by}), y = {by}."],
                distractors=c_dist,
            ),
            dict(
                prompt=f"What is the greatest profit per {period}, in dollars?",
                answer_kind="number",
                answer=value,
                answer_text=str(value),
                marks=2,
                follow_through=f"{p}*part_b + {q}*part_c",
                sensible_range=(0, None),
                working=[f"{p}·{bx} + {q}·{by} = {value}."],
                distractors=d_dist,
            ),
        ],
    )
