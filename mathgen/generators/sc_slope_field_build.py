"""Generates `sc_slope_field` questions (easy, medium, hard) and the exam chain `sc_slope_field_chain`.

Skill `slope-field-readings` (answer `number`): read numbers off the slope field of a first-order equation
dy/dx = f(x, y), where f is a short polynomial in x and y. Every answer is typed; no part is a multiple choice.

* easy: the slope of the segment at one stated grid point, for f such as x + y, xy, x - y^2 or x^2 + 2y.
* medium, one of three:
    - a short table of slopes read at grid points of dy/dx = ax + by (+ c) fixes the unknown constants, and the
      slope at another stated point is asked for;
    - an unknown parameter k in the equation (kx + y, x^2 - ky, kxy ...) is found from one stated slope, a
      horizontal segment or a segment parallel to a stated line;
    - the y-coordinate at which the segment above a stated x is horizontal.
* hard, one of two:
    - a walk along the field: from a stated start, move across by a step of width h one or two times, each time
      along the segment at the point reached, and give the y-coordinate arrived at (Euler's method read from the
      field, in plain words);
    - how many of the integer grid points in a stated window have slope zero (or a stated slope).
* exam (`sc_slope_field_chain`): for a separable equation dy/dx = ky, axy, x/y, y/x or ax^2 through a stated point,
  the slope at the point, a two-step walk with the slope re-read at the new point, the exact value of the solution
  at the end of the walk, and the amount by which the walk misses it. Each part follows through from the last.

Every key is re-derived by a second route: the slope by SymPy substitution, the table constants by SymPy's linear
solver, the grid counts by SymPy evaluation of each point and the exact solution by checking, with SymPy, that it
satisfies the equation and the starting condition.
"""

from __future__ import annotations

import itertools
import random
from fractions import Fraction

import sympy

from ..core import attempts, check, generator, join_terms, paren_negative, phrase, round_half_up
from ._company_kit import keep_company, lone_sets, numeric_pool
from .exd_x_r2b_support import decimal_places
from .exd_x_r2b_support import exact_number_part as _exact_number_part
from .exd_x_r2b_support import number_part as _number_part
from .r4e_common import Redraw, clear_of_boundary, draw, real, single


#: the question being built draws from this, so the key's rank among the options follows the seed
_RNG: list = [None]


def _company(key, wrong, places, lo=None):
    """`wrong` reordered so the key is never the only option with a visible feature (`_company_kit`)."""
    key = Fraction(key)
    seen_values = {key}
    cands = []
    for v, t in wrong:
        if (
            v is not None
            and Fraction(v) not in seen_values
            and (lo is None or v >= lo)
            and (key.denominator != 1 or places or Fraction(v).denominator == 1)
        ):
            seen_values.add(Fraction(v))
            cands.append((Fraction(v), t))
    shown = max([places] + [decimal_places(v, 4) or 0 for v, _ in cands[:6]])
    places = min(shown, 4)
    step = Fraction(1, 10**places)
    pool = numeric_pool(key, step if places else 1)
    for off in (3, 5, 7, 10, 12, 15, 20, 25, 30, 40, 50):  # plain misses at mixed distances, near and far
        for sign in (-1, 1):
            v = key + sign * Fraction(off, 10) * (1 if places else 1)
            if places == 0:
                v = key + sign * (off // 5 + 1)
            pool.append((v, "near-miss-value"))
    pool = [
        (v, t)
        for v, t in pool
        if decimal_places(Fraction(v), places) is not None
        and abs(Fraction(v) - key) > max(Fraction(51, 1000), step)
        and (lo is None or Fraction(v) >= lo)
        and (key.denominator != 1 or places or Fraction(v).denominator == 1)
    ]
    random.Random(repr((str(key), places))).shuffle(pool)
    places_arg = places if places else None
    for item in pool:  # a part with fewer than three distinct slips is topped up from the plain near misses
        if len(cands) >= 3:
            break
        if Fraction(item[0]) not in seen_values:
            seen_values.add(Fraction(item[0]))
            cands.append(item)
    # a uniformly random rank for the key: how many of the three shown wrong options lie below it
    rng = _RNG[0] or random.Random(repr(("rank", str(key), [str(v) for v, _ in cands])))
    seen_c = {Fraction(v) for v, _ in cands}
    below = [c for c in cands if c[0] < key] + [c for c in pool if c[0] < key and Fraction(c[0]) not in seen_c]
    above = [c for c in cands if c[0] > key] + [c for c in pool if c[0] > key and Fraction(c[0]) not in seen_c]

    def has_company(t):  # not the only option on its side of zero
        return key == 0 or any((c[0] > 0) == (key > 0) and c[0] != 0 for c in below[:t] + above[: 3 - t])

    feasible = [t for t in range(4) if len(below) >= t and len(above) >= 3 - t and has_company(t)]
    target = rng.choice(feasible) if feasible else None
    if target is not None:
        pick = below[:target] + above[: 3 - target]
        # every question carries a plain near miss, whatever the rank, so the fillers do not mark one rank
        if not any(t == "near-miss-value" for _, t in pick):
            pool_below = [c for c in pool if c[0] < key and c not in pick]
            pool_above = [c for c in pool if c[0] > key and c not in pick]
            sides = [(i, f) for i, f in ((0, pool_below), (1, pool_above)) if f and (target if i == 0 else 3 - target)]
            if sides:
                side, fill = sides[rng.randrange(len(sides))]
                at = (target - 1) if side == 0 else (target + (3 - target) - 1)
                pick[at] = fill[0]
        cands = pick + [c for c in cands if c not in pick]
    ordered = keep_company(key, cands, pool, places=places_arg, max_pool=40, shift_cost=6.0)
    return _resample(key, ordered, cands, pool, places_arg, target)


def _resample(key, ordered, cands, pool, places, target=None):
    """If the key still has a feature to itself, try many triples of options (tagged slips preferred) and keep the best."""
    tagged = {Fraction(v) for v, _ in cands}
    shown = ordered[:3]
    if len(shown) < 3 or not lone_sets([key, *(v for v, _ in shown)])[0]:
        return ordered
    if target is None:
        target = sum(1 for v, _ in shown if v < key)
    from ._company_kit import _PLACES

    rng = random.Random(repr(("resample", str(key), places)))
    everything = list({Fraction(v): (v, t) for v, t in [*cands, *pool] if Fraction(v) != key}.values())

    def score(trio):
        _PLACES[0] = places
        try:
            sets = lone_sets([key, *(v for v, _ in trio)])
        finally:
            _PLACES[0] = None
        return (
            10 * len(sets[0])
            + 0.4 * sum(len(x) for x in sets[1:])
            + 0.3 * sum(Fraction(v) not in tagged for v, _ in trio)
            + 8 * abs(sum(1 for v, _ in trio if v < key) - target)
        )

    best, best_score = list(shown), score(shown)
    close = sorted((c for c in everything if abs(Fraction(c[0]) - key) <= 1), key=lambda c: abs(Fraction(c[0]) - key))[
        :12
    ]
    trials = [rng.sample(everything, 3) for _ in range(120)]
    trials += [list(t) for t in itertools.combinations(close, 3)]
    trials += [[a, b, c] for a, b in itertools.combinations(close, 2) for c in cands if c not in (a, b)]
    for trio in trials:
        if len({Fraction(v) for v, _ in trio}) < 3:
            continue
        sc = score(trio)
        if sc < best_score - 1e-9:
            best, best_score = trio, sc
    return best + [c for c in ordered if not any(c is x or c == x for x in best)]


def exact_number_part(prompt, value, marks, working, wrong, lo=None, **kw):
    places = decimal_places(Fraction(value), kw.get("max_places", 8)) or 0
    return _exact_number_part(prompt, value, marks, working, _company(value, wrong, places, lo), **kw)


def _fixed_part(prompt, value, marks, working, wrong, **kw):
    """A decimal answer stated to the places it needs (at least one), so every option is printed to the same places."""
    kw.pop("max_places", None)
    kw.pop("suffix", None)
    places = max(decimal_places(Fraction(value), 4) or 0, 1)
    return number_part(prompt, value, places, marks, working, wrong, **kw)


def number_part(prompt, value, places, marks, working, wrong, **kw):
    rounded = round_half_up(Fraction(value), places)
    wrong = [(round_half_up(Fraction(v), places), t) for v, t in wrong if v is not None]
    return _number_part(prompt, value, places, marks, working, _company(rounded, wrong, places), **kw)


x, y = sympy.symbols("x y")

# A right-hand side is a list of (coefficient, (power of x, power of y)).
_MONO = {(1, 0): "x", (0, 1): "y", (1, 1): "xy", (2, 0): "x^2", (0, 2): "y^2", (0, 0): ""}


def _text(terms) -> str:
    return join_terms([(c, _MONO[m]) for c, m in terms])


def _val(terms, px, py) -> Fraction:
    px, py = Fraction(px), Fraction(py)
    return sum((Fraction(c) * px ** m[0] * py ** m[1] for c, m in terms), Fraction(0))


def _slope(terms, px, py) -> Fraction:
    """The slope at (px, py), checked against SymPy's substitution into the same expression."""
    value = _val(terms, px, py)
    expr = sum((c * x ** m[0] * y ** m[1] for c, m in terms), sympy.Integer(0))
    got = expr.subs(
        {
            x: sympy.Rational(Fraction(px).numerator, Fraction(px).denominator),
            y: sympy.Rational(Fraction(py).numerator, Fraction(py).denominator),
        }
    )
    check(got == sympy.Rational(value.numerator, value.denominator), "slope disagrees with SymPy")
    return value


def _nonzero(rng, pool):
    return rng.choice([v for v in pool if v != 0])


def _rhs(rng, kinds) -> list:
    kind = rng.choice(kinds)
    a = rng.choice((-3, -2, -1, 1, 2, 3))
    b = rng.choice((-3, -2, -1, 1, 1, 2, 3))
    c = rng.choice((0, 0, -3, -2, -1, 1, 2, 3))
    if kind == "lin":
        terms = [(a, (1, 0)), (b, (0, 1)), (c, (0, 0))]
    elif kind == "prod":
        terms = [(rng.choice((-2, -1, 1, 1, 2)), (1, 1)), (c, (0, 0))]
    elif kind == "xy2":
        terms = [(a, (1, 0)), (rng.choice((-2, -1, -1, 1, 1, 2)), (0, 2)), (c, (0, 0))]
    elif kind == "x2y":
        terms = [(rng.choice((-2, -1, 1, 1, 2)), (2, 0)), (b, (0, 1)), (c, (0, 0))]
    elif kind == "x2y2":
        terms = [(1, (2, 0)), (rng.choice((-1, 1)), (0, 2)), (c, (0, 0))]
    else:
        raise ValueError(kind)
    return [(co, m) for co, m in terms if co != 0]


# -- easy ---------------------------------------------------------------------------------------------------------

_EASY_STEMS = (
    "Consider the slope field of dy/dx = {f}, drawn on a grid. What is the slope of the segment at the grid point ({px}, {py})?",
    "The point ({px}, {py}) lies on a solution curve of dy/dx = {f}. Find the gradient of the curve at that point.",
    "For the differential equation dy/dx = {f}, find the value of dy/dx where x = {px} and y = {py}.",
    "In the direction field of dy/dx = {f}, a short segment is drawn at ({px}, {py}). Find its slope.",
)


def _eval_variant(terms, px, py, *, sign_lost=False, prod_as_sum=False, drop=None):
    """The slope with one typical slip: a negative square kept negative, xy read as x + y, or a term dropped."""
    total = Fraction(0)
    for c, m in terms:
        if drop is not None and m == drop:
            continue
        if m == (1, 1) and prod_as_sum:
            total += c * (px + py)
            continue
        part = Fraction(c)
        for base, power in ((px, m[0]), (py, m[1])):
            term = Fraction(base) ** power
            if sign_lost and power == 2 and base < 0:
                term = -term
            part *= term
        total += part
    return total


def _easy(rng):
    terms = _rhs(rng, ("lin", "lin", "prod", "xy2", "x2y"))
    px, py = rng.randint(-4, 4), rng.randint(-4, 4)
    key = _slope(terms, px, py)
    if key == 0 or abs(key) > 30 or (px, py) == (0, 0):
        raise Redraw("a flat or oversized slope")
    stem = phrase(rng, _EASY_STEMS, f=_text(terms), px=px, py=py)
    wrong = [
        (_val(terms, py, px), "ode-coordinates-swapped"),
        (-key, "sign-of-result-reversed"),
        (_eval_variant(terms, px, py, sign_lost=True), "slope-field-square-sign-lost"),
        (_eval_variant(terms, px, py, prod_as_sum=True), "slope-field-product-as-sum"),
        (_eval_variant(terms, px, py, drop=(0, 0)), "dropped-constant-term"),
        (_val(terms, px, 0), "ode-slope-from-one-variable-only"),
        (_val(terms, 0, py), "ode-slope-from-one-variable-only"),
    ]
    working = [f"dy/dx = {_text(terms)} at ({px}, {py}): the slope is {int(key)}."]
    part = exact_number_part("", key, 1, working, wrong, suffix="Give a whole number.")
    part["prompt"] = "Give a whole number."
    return single(stem, part)


# -- medium -------------------------------------------------------------------------------------------------------


def _solve_constants(readings, with_c):
    """a, b (and c) from the readings (x, y, slope), by SymPy's linear solver."""
    a, b, c = sympy.symbols("a b c")
    eqs = [a * px + b * py + (c if with_c else 0) - s for px, py, s in readings]
    sol = sympy.solve(eqs, [a, b, c] if with_c else [a, b], dict=True)
    return sol[0] if sol else None


_TABLE_STEMS = (
    "Readings from a slope field show a slope of {readings}. The field comes from dy/dx = {rule}, where {consts} constants. Find the slope of the segment at ({qx}, {qy}).",
    "A slope field is drawn for dy/dx = {rule}, where {consts} constants. Its segments have a slope of {readings}. What is the slope of the segment at ({qx}, {qy})?",
)


def _list_text(items) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _medium_table(rng):
    with_c = rng.random() < 0.55
    a, b = _nonzero(rng, range(-4, 5)), _nonzero(rng, range(-4, 5))
    c = _nonzero(rng, range(-5, 6)) if with_c else 0
    truth = [(a, (1, 0)), (b, (0, 1)), (c, (0, 0))]
    truth = [(co, m) for co, m in truth if co != 0]
    need = 3 if with_c else 2
    pts = []
    for _ in attempts(200, "distinct grid points"):
        p = (rng.randint(-3, 3), rng.randint(-3, 3))
        if p not in pts:
            pts.append(p)
        if len(pts) == need + 1:
            break
    *given, query = pts
    readings = [(px, py, _val(truth, px, py)) for px, py in given]
    sol = _solve_constants([(px, py, s) for px, py, s in readings], with_c)
    if sol is None or len(sol) != (3 if with_c else 2):
        raise Redraw("the readings do not fix the constants")
    a_s, b_s = sympy.symbols("a b")
    check(sol[a_s] == a and sol[b_s] == b, "the table constants disagree")
    qx, qy = query
    key = _slope(truth, qx, qy)
    if key == 0 or abs(key) > 40:
        raise Redraw("a flat or oversized answer")
    if with_c:
        check(sol[sympy.Symbol("c")] == c, "the table constant c disagrees")
    nearest = min(readings, key=lambda r: (abs(r[0] - qx) + abs(r[1] - qy), readings.index(r)))[2]
    wrong = [
        (Fraction(a) * qx + Fraction(b) * qy, "dropped-constant-term") if with_c else None,
        (Fraction(a) * qy + Fraction(b) * qx + c, "ode-coordinates-swapped"),
        (nearest, "slope-field-table-nearest-reading"),
        (sum((r[2] for r in readings), Fraction(0)) / len(readings), "slope-field-table-averages-readings"),
        (-key, "sign-of-result-reversed"),
    ]
    wrong = [w for w in wrong if w is not None]
    reading_text = _list_text([f"{int(s)} at ({px}, {py})" for px, py, s in readings])
    consts = "a, b and c are" if with_c else "a and b are"
    rule = "ax + by + c" if with_c else "ax + by"
    shape = rng.randrange(2)
    stem = _TABLE_STEMS[shape].format(readings=reading_text, rule=rule, consts=consts, qx=qx, qy=qy)
    working = [
        f"The readings give a = {a}, b = {b}" + (f" and c = {c}." if with_c else "."),
        f"At ({qx}, {qy}): " + join_terms([(a, f"({qx})"), (b, f"({qy})"), (c, "")]) + f" = {int(key)}.",
    ]
    part = exact_number_part("", key, 2, working, wrong, suffix="Give a whole number.")
    part["prompt"] = "Give a whole number."
    return single(stem, part)


_MONO_POOL = ((1, 0), (0, 1), (1, 1), (2, 0), (0, 2))
_PARAM_STEMS = (
    "Consider the slope field of dy/dx = {rule}. The segment at the point ({px}, {py}) has slope {s}. Find the value of k.",
    "Consider the slope field of dy/dx = {rule}. The segment at the point ({px}, {py}) is {shape}. Find the value of k.",
    "Consider the slope field of dy/dx = {rule}. The segment drawn at ({px}, {py}) is parallel to the line y = {sx} + 4. What is k?",
)


def _param_rule(k_mono, other, o, c) -> str:
    kt = "k" + _MONO[k_mono]
    rest = _text([(o, other)] + ([(c, (0, 0))] if c else []))
    if rest.startswith("-"):
        return f"{kt} - {rest[1:]}"
    return f"{kt} + {rest}"


def _medium_param(rng):
    k_mono, other = rng.sample(_MONO_POOL, 2)
    o = rng.choice((-2, -1, 1, 1, 2, 3))
    c = rng.choice((0, 0, -3, -2, -1, 1, 2, 3))
    k = _nonzero(rng, range(-5, 6))
    px, py = rng.randint(-3, 3), rng.randint(-3, 3)
    fk = Fraction(px) ** k_mono[0] * Fraction(py) ** k_mono[1]
    fo = Fraction(px) ** other[0] * Fraction(py) ** other[1]
    if fk == 0:
        raise Redraw("the parameter drops out at this point")
    rest = o * fo + c
    s = k * fk + rest
    if abs(s) > 40 or fo == 0 and c == 0:
        raise Redraw("an oversized or trivial slope")
    # second route: solve the stated condition for k with SymPy
    ks = sympy.Symbol("k")
    expr = ks * x ** k_mono[0] * y ** k_mono[1] + o * x ** other[0] * y ** other[1] + c
    solved = sympy.solve(sympy.Eq(expr.subs({x: px, y: py}), sympy.Rational(s.numerator, s.denominator)), ks)
    check(solved == [k], "the parameter disagrees with SymPy")
    rule = _param_rule(k_mono, other, o, c)
    variant = rng.randrange(3)
    if variant == 1 and s != 0:
        variant = 0
    if variant == 2 and s == 0:
        variant = 1
    stem = _PARAM_STEMS[variant].format(
        rule=rule, s=int(s), sx={1: "x", -1: "-x"}.get(int(s), f"{int(s)}x"), px=px, py=py, shape="horizontal"
    )
    swapped_fk = Fraction(py) ** k_mono[0] * Fraction(px) ** k_mono[1]
    swapped_rest = o * Fraction(py) ** other[0] * Fraction(px) ** other[1] + c
    wrong = [
        (-k, "sign-of-result-reversed"),
        (s - rest, "slope-field-parameter-not-divided"),
        ((s - c) / fk, "slope-field-parameter-rest-dropped"),
        ((s + rest) / fk, "sign-error-moving-term"),
        ((s - swapped_rest) / swapped_fk if swapped_fk != 0 else None, "ode-coordinates-swapped"),
    ]
    wrong = [w for w in wrong if w[0] is not None]
    working = [
        f"At ({px}, {py}) the slope is " + join_terms([(1, f"k({int(fk)})"), (int(rest), "")]) + f" = {int(s)}.",
        f"So k = ({int(s)} - {paren_negative(int(rest))})/{paren_negative(int(fk))} = {k}.",
    ]
    part = exact_number_part("", k, 2, working, wrong, suffix="Give a whole number.")
    part["prompt"] = "Give a whole number."
    return single(stem, part)


_HORIZ_STEMS = (
    "Consider the slope field of dy/dx = {f}. Above x = {p}, the segment is horizontal at exactly one grid point. What is the y-coordinate of that point?",
    "A slope field is drawn for dy/dx = {f}. Along the line x = {p}, one segment is horizontal. Find the value of y there.",
)


def _medium_horizontal(rng):
    b = rng.choice((-3, -2, -1, 1, 2, 3))
    g = rng.choice(((1, 0), (2, 0)))
    a = rng.choice((-3, -2, -1, 1, 2, 3))
    c = rng.choice((0, -4, -3, -2, -1, 1, 2, 3, 4))
    p = rng.randint(-3, 3)
    terms = [(a, g), (b, (0, 1)), (c, (0, 0))]
    terms = [(co, m) for co, m in terms if co != 0]
    gp = Fraction(a) * Fraction(p) ** g[0] + c
    key = -gp / b
    if key.denominator != 1 or key == 0 or p == 0 or gp == 0:
        raise Redraw("no clean horizontal point")
    key = Fraction(key)
    check(_slope(terms, p, key) == 0, "the segment is not horizontal")
    yy = sympy.Symbol("yy")
    solved = sympy.solve(sum((co * sympy.Integer(p) ** m[0] * yy ** m[1] for co, m in terms), sympy.Integer(0)), yy)
    check(solved == [sympy.Rational(key.numerator, key.denominator)], "the horizontal point disagrees with SymPy")
    stem = phrase(rng, _HORIZ_STEMS, f=_text(terms), p=p)
    wrong = [
        (gp / b, "slope-field-zero-slope-sign-slip"),
        (-gp, "slope-field-zero-slope-coefficient-dropped"),
        (gp, "slope-field-zero-slope-coefficient-dropped"),
        (Fraction(p), "slope-field-reads-x-as-y"),
        (-key, "sign-of-result-reversed"),
    ]
    working = [f"Set {_text(terms)} = 0 with x = {p}: {int(b)}y = {int(-gp)}, so y = {int(key)}."]
    part = exact_number_part("", key, 2, working, wrong, suffix="Give a whole number.")
    part["prompt"] = "Give a whole number."
    return single(stem, part)


# -- hard ---------------------------------------------------------------------------------------------------------

_WALK_STEMS = (
    "Trace a path on the slope field of dy/dx = {f}. The path starts at the point ({x0}, {y0}). Each step moves {h} to the right along the segment at the point where the step starts. After {n}, what is the y-coordinate of the path?",
    "A path follows the slope field of dy/dx = {f}. Starting at ({x0}, {y0}), every step goes {h} across in the x direction, along the segment drawn where the step begins. Find the y-coordinate reached after {n}.",
)
_STEP_WORDS = {1: "one step", 2: "two steps"}


def _neg(v) -> str:
    return f"({_decimal(v)})" if v < 0 else _decimal(v)


def _decimal(v) -> str:
    return f"{float(v):g}"


def _hard_walk(rng):
    terms = _rhs(rng, ("lin", "prod", "xy2", "x2y", "x2y2"))
    x0, y0 = rng.randint(-3, 3), rng.randint(-3, 3)
    h = rng.choice((Fraction(1, 2), Fraction(1, 2), Fraction(1, 5), Fraction(1, 10)))
    n = rng.choice((1, 2, 2))
    xs, ys, ss = [Fraction(x0)], [Fraction(y0)], []
    for _ in range(n):
        s = _slope(terms, xs[-1], ys[-1])
        ss.append(s)
        xs.append(xs[-1] + h)
        ys.append(ys[-1] + h * s)
    if 0 in ss or abs(ss[0]) > 25 or any(abs(s) > 40 for s in ss) or abs(ys[-1]) > 60:
        raise Redraw("a flat or oversized slope")
    if n == 2 and (ss[0] == ss[1] or ys[2] == ys[1]):
        raise Redraw("the slope does not change")
    if decimal_places(ys[-1], 4) is None:
        raise Redraw("a long decimal")
    key = ys[-1]
    if n == 1:
        wrong = [
            (y0 + ss[0], "euler-step-size-not-applied"),
            (h * ss[0], "euler-increment-reported-as-estimate"),
            (y0 - h * ss[0], "subtracted-instead-of-added"),
            (y0 + h * _val(terms, xs[1], ys[0]), "slope-field-slope-read-at-step-end"),
        ]
    else:
        wrong = [
            (ys[1] + h * ss[0], "euler-gradient-uses-old-point"),
            (ys[1] + ss[1], "euler-step-size-not-applied"),
            (y0 + h * ss[1], "euler-restarts-from-initial-value"),
            (ys[1], "slope-field-stops-after-one-step"),
            (h * ss[1], "euler-increment-reported-as-estimate"),
        ]
    stem = phrase(rng, _WALK_STEMS, f=_text(terms), x0=x0, y0=y0, h=_decimal(h), n=_STEP_WORDS[n])
    working = [
        f"Step {i + 1}: from ({_decimal(xs[i])}, {_decimal(ys[i])}) the slope is {_decimal(ss[i])}, so y becomes {_decimal(ys[i + 1])}."
        for i in range(n)
    ]
    part = _fixed_part("", key, 3, working, wrong)
    part["prompt"] = part["prompt"].lstrip(", ").capitalize()
    return single(stem, part)


_COUNT_STEMS = (
    "In the slope field of dy/dx = {f}, a segment is drawn at every grid point (x, y) with whole-number coordinates, {xlo} ≤ x ≤ {xhi} and {ylo} ≤ y ≤ {yhi}. At how many of these grid points {cond}?",
    "Consider the slope field of dy/dx = {f}. Segments are drawn only at points whose coordinates are whole numbers in the window {xlo} ≤ x ≤ {xhi}, {ylo} ≤ y ≤ {yhi}. How many of the drawn segments {cond2}?",
)


def _hard_count(rng):
    terms = _rhs(rng, ("lin", "prod", "xy2", "x2y", "x2y2"))
    xlo, xhi = -rng.randint(2, 5), rng.randint(2, 5)
    ylo, yhi = -rng.randint(2, 5), rng.randint(2, 5)
    m = rng.choice((0, 0, 0, 1, -1, 2))
    window = [(px, py) for px in range(xlo, xhi + 1) for py in range(ylo, yhi + 1)]
    hits = [p for p in window if _val(terms, *p) == m]
    expr = sum((c * x ** mo[0] * y ** mo[1] for c, mo in terms), sympy.Integer(0))
    check(len(hits) == sum(1 for px, py in window if expr.subs({x: px, y: py}) == m), "grid count disagrees with SymPy")
    key = len(hits)
    if not 4 <= key <= 24:
        raise Redraw("too few or too many grid points")
    if key <= 6 and rng.random() < (0.93 if key == 4 else 0.6):
        raise Redraw("thin out the commonest small counts")
    inner = sum(1 for px, py in hits if xlo < px < xhi and ylo < py < yhi)
    columns = len({px for px, _ in hits})
    greater = sum(1 for p in window if _val(terms, *p) > m)
    differs = len(window) - key
    cond = "is the segment horizontal" if m == 0 else f"does the segment have slope {m}"
    cond2 = "are horizontal" if m == 0 else f"have slope {m}"
    stem = phrase(rng, _COUNT_STEMS, f=_text(terms), xlo=xlo, xhi=xhi, ylo=ylo, yhi=yhi, cond=cond, cond2=cond2)
    wrong = [
        (inner, "slope-field-window-boundary-dropped"),
        (columns, "slope-field-counts-columns-not-points"),
        (greater, "slope-field-wrong-condition-counted"),
        (key + 1, "slope-field-grid-points-miscounted"),
        (key - 1, "slope-field-grid-points-miscounted"),
        (differs, "slope-field-wrong-condition-counted"),
    ]
    working = [f"Solving {_text(terms)} = {m} over the window gives the {key} points {', '.join(map(str, hits))}."]
    part = exact_number_part("", key, 2, working, wrong, lo=0, suffix="Give a whole number.")
    part["prompt"] = "Give a whole number."
    return single(stem, part)


@generator(
    "sc_slope_field",
    level=17,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def sc_slope_field(rng, tier: str = "easy"):
    """Slope readings from dy/dx = f(x, y); see the module docstring for the ladder."""
    _RNG[0] = rng
    if tier == "easy":
        return draw(_easy, rng)
    if tier == "medium":
        pick = rng.random()
        return draw(_medium_table if pick < 0.4 else (_medium_param if pick < 0.8 else _medium_horizontal), rng)
    if tier == "hard":
        return draw(_hard_walk if rng.random() < 0.65 else _hard_count, rng)
    raise ValueError(f"unknown tier {tier!r}")


# -- exam chain ---------------------------------------------------------------------------------------------------

_X0, _Y0 = sympy.symbols("x0 y0")


def _families():
    """(name, text of f, f(x, y) as a function of Fractions, SymPy f, SymPy solution through (x0, y0))."""
    return {
        "ky": lambda k: (
            join_terms([(k, "y")]),
            lambda px, py: k * py,
            k * y,
            _Y0 * sympy.exp(k * (x - _X0)),
        ),
        "axy": lambda k: (
            join_terms([(k, "xy")]),
            lambda px, py: k * px * py,
            k * x * y,
            _Y0 * sympy.exp(k * (x**2 - _X0**2) / 2),
        ),
        "x/y": lambda k: (
            ("" if k == 1 else "-") + "x/y",
            lambda px, py: k * px / py,
            k * x / y,
            sympy.sqrt(_Y0**2 + k * (x**2 - _X0**2)),
        ),
        "y/x": lambda k: ("y/x", lambda px, py: py / px, y / x, _Y0 * x / _X0),
        "ax2": lambda k: (
            join_terms([(k, "x^2")]),
            lambda px, py: k * px**2,
            k * x**2,
            _Y0 + k * (x**3 - _X0**3) / 3,
        ),
    }


_CHAIN_STEMS = (
    "The point ({x0}, {y0}) lies on the solution curve of dy/dx = {f}. To trace the curve, a path is built from the slope field: starting at that point, each step goes {h} across along the segment where the step begins.",
    "Consider a curve through ({x0}, {y0}) that satisfies dy/dx = {f}. Its slope field is used to estimate the curve by a path of two steps, each {h} wide in the x direction and each taken along the segment at its starting point.",
)


def _chain(rng):
    name = rng.choice(("ky", "ky", "axy", "axy", "x/y", "y/x"))
    k = rng.choice((-2, -1, 1, 2, 3)) if name in ("ky", "axy", "ax2") else rng.choice((1, -1))
    text, f, f_sym, sol = _families()[name](k)
    if name == "y/x":
        x0, y0 = rng.choice((1, 2, 3)), rng.choice((1, 2, 3, 4))
    elif name == "x/y":
        x0, y0 = rng.randint(-2, 3), rng.choice((1, 2, 4, 5))
    else:
        x0, y0 = rng.randint(-2, 2), rng.choice((1, 2, 3))
    h = rng.choice((Fraction(1, 2), Fraction(1, 5), Fraction(1, 10), Fraction(1, 4)))
    x1, x2 = x0 + h, x0 + 2 * h
    s0 = Fraction(f(Fraction(x0), Fraction(y0)))
    y1 = y0 + h * s0
    if y1 == 0:
        raise Redraw("a zero denominator")
    s1 = Fraction(f(x1, y1))
    y2 = y1 + h * s1
    if name == "x/y" and (y2 <= 0 or y0**2 + k * (x2**2 - x0**2) <= 0):
        raise Redraw("the path leaves the solution's domain")
    if 0 in (s0, s1, y2) or s0 == s1 or abs(y2) > 40 or any(decimal_places(v, 4) is None for v in (s0, y1, s1, y2)):
        raise Redraw("a flat, oversized or long-decimal path")
    # second route: the stated solution satisfies the equation and the starting condition (SymPy)
    solution = sol.subs({_X0: x0, _Y0: y0})
    check(sympy.simplify(solution.subs(x, x0) - y0) == 0, "the solution misses the starting point")
    probe = sympy.Rational(x0) + sympy.Rational(1, 3)
    residual = sympy.diff(solution, x).subs(x, probe) - f_sym.subs({x: probe, y: solution.subs(x, probe)})
    check(abs(sympy.N(residual, 30)) < sympy.Float("1e-20"), "the solution does not satisfy the equation")
    x2_exact = sympy.Rational(x2.numerator, x2.denominator)
    exact = real(solution.subs(x, x2_exact))
    at_x1 = real(solution.subs(x, sympy.Rational(x1.numerator, x1.denominator)))
    miss = abs(exact - y2)
    if not (clear_of_boundary(exact, 3, Fraction(1, 10)) and clear_of_boundary(miss, 3, Fraction(1, 10))):
        raise Redraw("a key too close to a rounding tie")
    exact3 = round_half_up(exact, 3)
    miss3 = round_half_up(miss, 3)
    if miss3 < Fraction(1, 100) or miss3 > 5:
        raise Redraw("a miss too small or too large to compare")
    answers = [s0, y1, s1, y2, exact3, miss3]
    if len({str(float(v)) for v in answers}) < 6:
        raise Redraw("two parts share an answer")
    hs = _decimal(h)
    stem = phrase(rng, _CHAIN_STEMS, f=text, x0=x0, y0=y0, h=hs)
    slope_code = {
        "ky": f"({k})*part_b",
        "axy": f"({k})*({x1})*part_b",
        "x/y": f"({k})*({x1})/part_b",
        "y/x": f"part_b/({x1})",
        "ax2": f"({k})*({x1})**2 + 0*part_b",
    }[name]
    wrong_const = None
    if name in ("ky", "axy", "ax2"):
        wrong_const = real(sol.subs({_X0: 0, _Y0: y0}).subs(x, x2_exact))
    exact_wrong = [
        (y2, "slope-field-euler-reported-as-exact"),
        (at_x1, "slope-field-exact-at-wrong-x"),
        (wrong_const, "slope-field-constant-not-fixed"),
        (y1, "slope-field-euler-reported-as-exact"),
    ]
    exact_wrong = [w for w in exact_wrong if w[0] is not None]
    wrong_a = [
        (s1, "euler-gradient-uses-old-point"),
        (-s0, "sign-of-result-reversed"),
        (_val_swap(name, k, x0, y0), "ode-coordinates-swapped"),
    ]
    parts = [
        _fixed_part(
            f"Find the slope of the segment at ({x0}, {y0}).",
            s0,
            1,
            [f"At ({x0}, {y0}) the slope is {_decimal(s0)}."],
            wrong_a,
            max_places=4,
            suffix="Give the exact decimal value.",
        ),
        _fixed_part(
            f"Following that segment across by {hs}, find the y-coordinate of the path at x = {_decimal(x1)}.",
            y1,
            1,
            [f"y = {y0} + {hs} x {_neg(s0)} = {_decimal(y1)}."],
            [
                (y0 + s0, "euler-step-size-not-applied"),
                (h * s0, "euler-increment-reported-as-estimate"),
                (y0 - h * s0, "subtracted-instead-of-added"),
            ],
            follow_through=f"{y0} + ({h})*part_a",
            max_places=4,
            suffix="Give the exact decimal value.",
        ),
        _fixed_part(
            f"Find the slope of the segment at the point ({_decimal(x1)}, y) where y is your answer to the previous part.",
            s1,
            1,
            [f"The slope at ({_decimal(x1)}, {_decimal(y1)}) is {_decimal(s1)}."],
            [
                (s0, "euler-gradient-uses-old-point"),
                (Fraction(f(x1, Fraction(y0))), "euler-gradient-uses-old-point"),
                (Fraction(f(Fraction(x0), y1)), "euler-gradient-uses-old-point"),
            ],
            follow_through=slope_code,
            max_places=4,
            suffix="Give the exact decimal value.",
        ),
        _fixed_part(
            f"Take a second step of width {hs} along that segment. Find the y-coordinate of the path at x = {_decimal(x2)}.",
            y2,
            2,
            [f"y = {_decimal(y1)} + {hs} x {_neg(s1)} = {_decimal(y2)}."],
            [
                (y1 + h * s0, "euler-gradient-uses-old-point"),
                (y1 + s1, "euler-step-size-not-applied"),
                (y0 + h * s1, "euler-restarts-from-initial-value"),
                (h * s1, "euler-increment-reported-as-estimate"),
            ],
            follow_through=f"part_b + ({h})*part_c",
            max_places=4,
            suffix="Give the exact decimal value.",
        ),
        number_part(
            f"Solve the separable equation exactly and find y at x = {_decimal(x2)} on the solution curve through ({x0}, {y0}).",
            exact,
            3,
            3,
            [f"The exact solution gives y = {float(exact):.3f} at x = {_decimal(x2)}."],
            exact_wrong,
            sensible_range=(-100, 100),
        ),
        number_part(
            "By how much does the path's y-coordinate at this x-value differ from the exact value? Give the positive difference.",
            miss,
            3,
            2,
            [f"|{float(exact):.3f} - {_neg(y2)}| = {float(miss3):.3f}."],
            [
                (exact - y2 if exact < y2 else y2 - exact, "error-sign-not-absolute"),
                (y2, "reports-estimate-not-error"),
                (abs(exact - y1), "slope-field-exact-at-wrong-x"),
                (exact, "reports-estimate-not-error"),
            ],
            follow_through="abs(part_e - part_d)",
            sensible_range=(0, 100),
        ),
    ]
    return dict(text=stem, parts=parts)


def _val_swap(name, k, x0, y0) -> Fraction:
    """The slope at (x0, y0) with the two coordinates swapped; zero denominators fall back to the slope itself."""
    px, py = Fraction(y0), Fraction(x0)
    try:
        if name == "ky":
            return k * py
        if name == "axy":
            return k * px * py
        if name == "x/y":
            return k * px / py
        if name == "y/x":
            return py / px
        return k * px**2
    except ZeroDivisionError:
        return Fraction(0)


@generator("sc_slope_field_chain", level=17, presets={"exam": dict(tier="exam")})
def sc_slope_field_chain(rng, tier: str = "exam"):
    """Slope, two-step walk, exact separable solution and the miss, for a separable dy/dx through a point."""
    _RNG[0] = rng
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return draw(_chain, rng)
