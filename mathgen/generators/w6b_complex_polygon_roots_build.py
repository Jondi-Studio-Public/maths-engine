"""Generates `w6b_complex_polygon_roots` (easy, medium, hard) and the exam chain `exd_x_w6b_complex_polygon_roots`.

``complex-polygon-vertices-and-roots-of-unity`` (``senior-mathematics``, answer ``expression``): the vertices of a
regular polygon in the complex plane and the sums and products of the roots of unity.

* easy: the vertex k steps anticlockwise from a vertex on one of the axes, for a regular 3-, 4-, 6-, 8- or 12-gon centred
  at the origin (an exact complex number).
* medium: either the vertex k steps along a regular 3-, 4- or 6-gon whose centre is not the origin, or a sum of powers of
  a primitive n-th root of unity ω that collapses by 1 + ω + … + ω^(n-1) = 0 (a whole number).
* hard: facts that hold for every regular polygon inscribed in a circle of radius r: the product of the distances from
  one vertex to all the others, the sum of their squares, the sum of the other roots of zⁿ = w, and Σ ω^(mk) for a
  multiple m of n or not.
* exam (``exd_x_w6b_complex_polygon_roots``): the roots of zⁿ = rⁿ as the vertices of a regular polygon: the radius, the
  side (following through from the radius), the perimeter (following through from the side) and the root with
  argument 2π/n.

Every result is re-derived by a second route: the vertices are generated from the n-th roots of unity numerically and
compared, and sums and products are evaluated numerically over the actual roots.
"""

from __future__ import annotations

import cmath
import math

import sympy

from ..core import attempts, check, generator, ordinal, phrase, rounds_unambiguously
from ._w6b_kit import I, cdist, cis, cx_text, cx_value, dist, gaussian, need, retry

_NAMES = {
    3: "an equilateral triangle",
    4: "a square",
    6: "a regular hexagon",
    8: "a regular octagon",
    12: "a regular 12-gon",
}


def _roots_numeric(n, centre=0, start=1):
    """The n vertices of a regular n-gon, anticlockwise, from ``start`` about ``centre`` (floating point)."""
    return [complex(centre) + (complex(start) - complex(centre)) * cmath.exp(2j * math.pi * k / n) for k in range(n)]


def _cis_exact(n, k):
    """cos(2πk/n) + i sin(2πk/n) in exact form (n among 3, 4, 6, 8, 12)."""
    rad = sympy.Rational(2 * k, n) * sympy.pi
    return sympy.expand(sympy.cos(rad) + I * sympy.sin(rad))


def _num_dist(value, tag):
    return dist(str(value), tag, value)


# -- easy -----------------------------------------------------------------------------------------------------

_EASY_STEMS = (
    "{Shape} is drawn on an Argand diagram with its centre at the origin. One vertex is at {z0}. Going anticlockwise "
    "around the polygon, which complex number is the vertex {steps} from it?",
    "The vertices of {shape} lie on a circle centred at the origin, and one of them represents {z0}. Find the "
    "complex number for the vertex reached by moving {steps} anticlockwise from it.",
    "On the complex plane, {z0} is a vertex of {shape} whose centre is the origin. State the vertex that lies {steps} "
    "after it in the anticlockwise direction.",
)


def _steps_words(k):
    return "one vertex" if k == 1 else f"{k} vertices"


def _easy(rng):
    n = rng.choice([3, 4, 6, 6, 8, 12])
    k = rng.randint(1, n - 1)
    if n == 4 and k == 2:
        k = 1
    r = rng.choice([2, 4, 6, 8])
    z0 = r if rng.random() < 0.6 else r * I
    unit = _cis_exact(n, k)
    w = sympy.expand(z0 * unit)
    # second route: floating-point vertices
    verts = _roots_numeric(n, 0, complex(z0))
    check(abs(verts[k] - complex(w)) < 1e-9, "the vertex differs from the numeric polygon")
    check(abs(abs(complex(w)) - r) < 1e-9, "the vertex is not on the circle")
    interior = sympy.Rational(180 * (n - 2), n)
    nearby = k + 1 if k + 1 < n else k - 1
    candidates = [
        cdist(sympy.expand(z0 * _cis_exact(n, n - k)), "polygon-vertex-wrong-direction"),
        cdist(sympy.expand(z0 * cis(k * interior)), "polygon-vertex-uses-interior-angle"),
        cdist(sympy.expand(z0 * _cis_exact(n, nearby)), "polygon-vertex-off-by-one"),
        cdist(sympy.expand(z0 * sympy.expand(sympy.im(unit) + I * sympy.re(unit))), "rotation-sine-and-cosine-swapped"),
    ]
    answer = cx_text(w)
    answer_value = cx_value(w)
    working = [
        f"The vertices are spaced 360°/{n} = {360 // n}° apart, so {k} step{'s' if k != 1 else ''} turn{'s' if k == 1 else ''} the vertex through {360 * k // n}°.",
        f"Multiply {cx_text(z0)} by {cx_text(unit)} to get {answer}.",
    ]
    return dict(
        text=phrase(
            rng,
            _EASY_STEMS,
            shape=_NAMES[n],
            Shape=_NAMES[n][0].upper() + _NAMES[n][1:],
            z0=cx_text(z0),
            steps=_steps_words(k),
        ),
        answer_kind="expression",
        answer=answer_value,
        answer_text=answer,
        working=working,
        distractors=need(answer, candidates, 4),
    )


# -- medium ---------------------------------------------------------------------------------------------------

_CENTRE_STEMS = (
    "{Shape} on the complex plane has its centre at {c}, and one vertex at {z0}. Going anticlockwise, what complex "
    "number is the vertex {steps} after {z0}?",
    "The point {z0} is a vertex of {shape} whose centre is {c}. Find the vertex {steps} further round in the "
    "anticlockwise direction.",
    "{Shape} is centred at {c} on an Argand diagram and has a vertex at {z0}. Give the vertex reached by moving "
    "{steps} anticlockwise.",
)
_SUM_STEMS = (
    "Let ω = cos(2π/{n}) + i sin(2π/{n}). Find the value of {expr}.",
    "ω is the {n_ord} root of unity with the smallest positive argument. Evaluate {expr}.",
    "Given that ω = cos(2π/{n}) + i sin(2π/{n}), calculate {expr}.",
)


def _medium_centre(rng):
    n = rng.choice([3, 4, 6, 6])
    k = rng.randint(1, n - 1)
    for _ in attempts(500, "a vertex and a centre"):
        ca, cb = rng.randint(-5, 5), rng.randint(-5, 5)
        p, q = 2 * rng.randint(-3, 3), 2 * rng.randint(-3, 3)
        if p == 0 or q == 0 or abs(p) == abs(q):
            continue
        break
    c, z0 = gaussian(ca, cb), gaussian(ca + p, cb + q)
    unit = _cis_exact(n, k)
    w = sympy.expand(c + (z0 - c) * unit)
    verts = _roots_numeric(n, complex(c), complex(z0))
    check(abs(verts[k] - complex(w)) < 1e-9, "the vertex differs from the numeric polygon")
    candidates = [
        cdist(sympy.expand(c + (z0 - c) * _cis_exact(n, n - k)), "polygon-vertex-wrong-direction"),
        cdist(sympy.expand(z0 * unit), "rotation-centre-ignored"),
        cdist(sympy.expand((z0 - c) * unit), "rotation-centre-not-added-back"),
        cdist(sympy.expand(c + (z0 - c) * _cis_exact(n, k + 1 if k + 1 < n else k - 1)), "polygon-vertex-off-by-one"),
        cdist(
            sympy.expand(c + (z0 - c) * cis(k * sympy.Rational(180 * (n - 2), n))), "polygon-vertex-uses-interior-angle"
        ),
    ]
    answer = cx_text(w)
    answer_value = cx_value(w)
    working = [
        f"Vector from the centre to the vertex: {cx_text(z0 - c)}.",
        f"Turn it through {360 * k // n}° (multiply by {cx_text(unit)}), then add the centre: {answer}.",
    ]
    return dict(
        text=phrase(
            rng,
            _CENTRE_STEMS,
            shape=_NAMES[n],
            Shape=_NAMES[n][0].upper() + _NAMES[n][1:],
            c=cx_text(c),
            z0=cx_text(z0),
            steps=_steps_words(k),
        ),
        answer_kind="expression",
        answer=answer_value,
        answer_text=answer,
        working=working,
        distractors=need(answer, candidates, 4),
    )


def _power_sum(omega, lo, hi, step=1):
    return sum(omega**k for k in range(lo, hi + 1, step))


def _medium_sum(rng):
    n = rng.choice([3, 4, 5, 6, 7, 8])
    omega = cmath.exp(2j * math.pi / n)
    kind = rng.choice(["window", "tail", "window", "constant"])
    if kind == "window":
        lo = rng.randint(2, 9)
        hi = lo + n - 1
        terms = f"ω^{lo} + ω^{lo + 1} + … + ω^{hi}"
        value = 0
        working = [f"These are {n} consecutive powers of ω, and 1 + ω + … + ω^{n - 1} = 0 for any such run."]
        candidates = [
            _num_dist(1, "roots-of-unity-sum-is-one"),
            _num_dist(n, "roots-of-unity-sum-counts-terms"),
            _num_dist(-1, "roots-of-unity-sum-drops-the-one"),
        ]
    elif kind == "tail":
        terms = "ω + ω² + … + ω^" + str(n - 1)
        value = -1
        working = [f"1 + ω + … + ω^{n - 1} = 0, so leaving out the 1 gives -1."]
        candidates = [
            _num_dist(0, "roots-of-unity-sum-forgets-the-one"),
            _num_dist(1, "roots-of-unity-sum-sign"),
            _num_dist(n - 1, "roots-of-unity-sum-counts-terms"),
        ]
    else:
        c = rng.randint(2, 9)
        terms = f"{c} + {c}ω + {c}ω² + … + {c}ω^{n - 1}"
        value = 0
        working = [f"Factor out {c}: {c}(1 + ω + … + ω^{n - 1}) = {c} × 0 = 0."]
        candidates = [
            _num_dist(c, "roots-of-unity-sum-factor-kept"),
            _num_dist(c * n, "roots-of-unity-sum-counts-terms"),
            _num_dist(c * (n - 1), "roots-of-unity-sum-counts-terms"),
        ]
    # second route: sum the actual powers
    if kind == "window":
        total = _power_sum(omega, lo, hi)
    elif kind == "tail":
        total = _power_sum(omega, 1, n - 1)
    else:
        total = c * _power_sum(omega, 0, n - 1)
    check(abs(total - value) < 1e-9, "numeric sum differs")
    answer = str(value)
    return dict(
        text=phrase(rng, _SUM_STEMS, n=n, n_ord=ordinal(n), expr=terms),
        answer_kind="expression",
        answer=answer,
        answer_text=answer,
        working=working,
        distractors=need(answer, candidates, 3),
    )


def _medium(rng):
    return rng.choice([_medium_centre, _medium_sum])(rng)


# -- hard -----------------------------------------------------------------------------------------------------

_PRODUCT_STEMS = (
    "A regular {n}-gon is inscribed in the circle |z| = {r} on an Argand diagram. Find the product of the lengths of the "
    "{m} straight lines joining one vertex to each of the other vertices.",
    "The {n} roots of z^{n} = {w} are the vertices of a regular polygon. Let V be one of them. Find the product of the "
    "distances from V to the other {m} vertices.",
    "A vertex V of a regular {n}-gon with circumradius {r} is joined to each of the other {m} vertices. What is the product "
    "of the {m} lengths?",
)
_SQUARES_STEMS = (
    "A regular {n}-gon is inscribed in a circle of radius {r}, centred at the origin of an Argand diagram. Find the sum of "
    "the squares of the distances from one vertex to every vertex of the polygon (the vertex itself included).",
    "The {n} roots of z^{n} = {w} lie at the vertices of a regular polygon. For one root V, find the sum of |V - z|² "
    "over all {n} roots z.",
    "Let V be a vertex of a regular {n}-gon whose vertices are at distance {r} from the centre. Calculate the sum of the "
    "squares of the distances from V to all {n} vertices.",
)
_OTHERS_STEMS = (
    "The {n} roots of z^{n} = {w} sum to zero. One of them is {z0}. Find the sum of the other {m} roots.",
    "One root of z^{n} = {w} is z₀ = {z0}. Without finding the others, state the sum of the remaining {m} roots.",
    "z^{n} = {w} has {n} roots in all, and {z0} is one. What do the other {m} roots add up to?",
)
_POWER_STEMS = (
    "Let ω = cos(2π/{n}) + i sin(2π/{n}). Find the value of 1 + ω^{m} + ω^({m}·2) + … + ω^({m}·{last_k}).",
    "ω is a primitive {n_ord} root of unity. Evaluate Σ ω^({m}k) for k = 0, 1, …, {last_k}.",
    "With ω = cos(2π/{n}) + i sin(2π/{n}), calculate the sum of ω^({m}k) as k runs from 0 to {last_k}.",
)


def _hard_product(rng):
    n = rng.choice([3, 4, 5, 6, 8])
    r = rng.choice([2, 3, 4, 5])
    value = n * r ** (n - 1)
    verts = _roots_numeric(n, 0, r)
    prod = 1.0
    for v in verts[1:]:
        prod *= abs(v - verts[0])
    check(abs(prod - value) < 1e-6 * value, "numeric product differs")
    candidates = [
        _num_dist(r ** (n - 1), "polygon-product-forgets-n"),
        _num_dist(n * r, "polygon-product-adds-instead"),
        _num_dist(n**1 * r**n, "polygon-product-uses-n-factors"),
        _num_dist((n - 1) * r ** (n - 1), "polygon-product-counts-n-minus-one"),
    ]
    return dict(
        text=phrase(rng, _PRODUCT_STEMS, n=n, r=r, m=n - 1, w=r**n),
        answer_kind="expression",
        answer=str(value),
        answer_text=str(value),
        working=[
            f"Take the vertex at {r}. The distances to the others are |{r} - {r}ω^k| = {r}|1 - ω^k| for k = 1, …, {n - 1}.",
            f"Since (1 - ω)(1 - ω²)…(1 - ω^{n - 1}) = {n}, the product is {r}^{n - 1} × {n} = {value}.",
        ],
        distractors=need(str(value), candidates, 4),
    )


def _hard_squares(rng):
    n = rng.choice([3, 4, 5, 6, 8, 12])
    r = rng.choice([2, 3, 4, 5, 6])
    value = 2 * n * r * r
    verts = _roots_numeric(n, 0, r)
    total = sum(abs(v - verts[0]) ** 2 for v in verts)
    check(abs(total - value) < 1e-6 * value, "numeric sum of squares differs")
    candidates = [
        _num_dist(n * r * r, "polygon-squares-forgets-factor-two"),
        _num_dist(4 * r * r, "polygon-squares-uses-one-diameter"),
        _num_dist(2 * (n - 1) * r * r, "polygon-squares-counts-n-minus-one"),
        _num_dist(n * r, "polygon-squares-forgets-to-square"),
    ]
    return dict(
        text=phrase(rng, _SQUARES_STEMS, n=n, r=r, w=r**n),
        answer_kind="expression",
        answer=str(value),
        answer_text=str(value),
        working=[
            f"With V = {r}, |V - z|² = |V|² + |z|² - 2Re(V z̄) = {2 * r * r} - 2Re(V z̄).",
            f"The vertices sum to 0, so the sum over all {n} vertices is {n} × {2 * r * r} = {value}.",
        ],
        distractors=need(str(value), candidates, 4),
    )


def _hard_others(rng):
    n = rng.choice([3, 4, 6, 8])
    r = rng.choice([2, 3])
    k = rng.randint(0, n - 1)
    root = sympy.expand(_cis_exact(n, k) * r)
    w = r**n
    ans = sympy.expand(-root)
    candidates = [
        cdist(root, "roots-sum-others-gives-the-root"),
        cdist(sympy.conjugate(root), "roots-sum-others-gives-the-conjugate"),
        cdist(sympy.Integer(0), "roots-sum-forgets-to-remove-the-root"),
        cdist(sympy.expand(-sympy.conjugate(root)), "roots-sum-others-conjugates-and-negates"),
    ]
    z0 = cx_text(root)
    # second route: the roots numerically
    verts = _roots_numeric(n, 0, complex(root))
    check(abs(sum(verts[1:]) - complex(ans)) < 1e-9, "numeric sum of the others differs")
    check(all(abs(v**n - w) < 1e-6 * w for v in verts), "not roots of z^n = w")
    answer = cx_text(ans)
    answer_value = cx_value(ans)
    return dict(
        text=phrase(rng, _OTHERS_STEMS, n=n, w=w, z0=z0, m=n - 1),
        answer_kind="expression",
        answer=answer_value,
        answer_text=answer,
        working=[
            f"z^{n} - {w} has no z^{n - 1} term, so the {n} roots sum to 0.",
            f"The other roots sum to -({z0}) = {answer}.",
        ],
        distractors=need(answer, candidates, 4),
    )


def _hard_power(rng):
    n = rng.choice([4, 6, 8, 9, 10, 12])
    if rng.random() < 0.4:
        m = n * rng.choice([1, 2, 3])
    else:
        m = rng.choice([x for x in range(2, 2 * n + 1) if x % n])
    omega = cmath.exp(2j * math.pi / n)
    total = sum(omega ** (m * k) for k in range(n))
    value = n if m % n == 0 else 0
    check(abs(total - value) < 1e-9, "numeric power sum differs")
    candidates = [
        _num_dist(n if value == 0 else 0, "roots-of-unity-power-sum-swapped"),
        _num_dist(1, "roots-of-unity-sum-is-one"),
        _num_dist(n - 1, "roots-of-unity-power-sum-counts-terms-less-one"),
        _num_dist(-1, "roots-of-unity-sum-drops-the-one"),
    ]
    if value:
        working = [f"ω^{m} = 1 because {n} divides {m}, so every term is 1 and the sum is {n}."]
    else:
        working = [
            f"The terms form a geometric series with ratio ω^{m}, which is not 1, and (ω^{m})^{n} = 1.",
            f"The sum is (1 - (ω^{m})^{n})/(1 - ω^{m}) = 0.",
        ]
    return dict(
        text=phrase(rng, _POWER_STEMS, n=n, n_ord=ordinal(n), m=m, lastm=m * (n - 1), last_k=n - 1),
        answer_kind="expression",
        answer=str(value),
        answer_text=str(value),
        working=working,
        distractors=need(str(value), candidates, 4),
    )


def _hard(rng):
    return rng.choice([_hard_product, _hard_squares, _hard_others, _hard_power, _hard_product, _hard_squares])(rng)


_EXAM_STEMS = (
    "The {n} roots of z^{n} = {w} are the vertices of a regular polygon on an Argand diagram, centred at the origin.",
    "On an Argand diagram, the solutions of z^{n} = {w} form the vertices of a regular polygon with its centre at the origin.",
    "Let P be the polygon whose vertices are the {n} roots of the equation z^{n} = {w} on the complex plane.",
)


@generator(
    "w6b_complex_polygon_roots",
    level=18,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def w6b_complex_polygon_roots(rng, tier="easy"):
    """Regular polygons in the complex plane and roots-of-unity sums; see the module docstring."""
    if tier == "easy":
        return retry(rng, _easy)
    if tier == "medium":
        return retry(rng, _medium)
    if not (tier == "hard"):
        raise ValueError(f"unknown tier {tier!r}")
    return retry(rng, _hard)


# -- exam -----------------------------------------------------------------------------------------------------


@generator("exd_x_w6b_complex_polygon_roots", level=18, presets={"exam": dict(tier="exam")})
def exd_x_w6b_complex_polygon_roots(rng, tier="exam"):
    """Exam chain: the roots of z^n = r^n as a regular polygon: radius, side, perimeter and a root."""
    if not (tier == "exam"):
        raise ValueError(f"unknown tier {tier!r}")
    return retry(rng, _exam)


# (n, r) pairs, each with r^n at most 3 x 10^8 so the constant stays a number a learner can read.
_EXAM_POLYGONS = [(n, r) for n, top in ((3, 600), (4, 120), (6, 26)) for r in range(2, top + 1)]


def _exam(rng):
    for _ in attempts(500, "a polygon whose 2-decimal side and perimeter agree"):
        n, r = rng.choice(_EXAM_POLYGONS)
        side = 2 * r * math.sin(math.pi / n)
        # the perimeter must come out the same from the exact side as from the 2-decimal side the learner follows through
        if (
            rounds_unambiguously(side, 2)
            and rounds_unambiguously(n * side, 2)
            and (round(n * side + 1e-12, 2) == round(n * round(side + 1e-12, 2) + 1e-12, 2))
        ):
            break
    w = r**n
    side2 = round(side + 1e-12, 2)
    perimeter = n * side2
    side_text, perimeter_text = f"{side2:.2f}", f"{perimeter:.2f}"
    root_exact = sympy.expand(r * _cis_exact(n, 1))
    # second route: the roots of z^n = r^n numerically are the polygon's vertices
    verts = _roots_numeric(n, 0, r)
    check(all(abs(v**n - w) < 1e-6 * w for v in verts), "not roots of z^n = w")
    check(abs(abs(verts[1] - verts[0]) - side) < 1e-9, "numeric side differs")
    check(abs(verts[1] - complex(root_exact)) < 1e-9, "numeric root differs")
    chord_cands = [
        _two_dp(2 * r, "polygon-side-uses-diameter"),
        _two_dp(r, "polygon-side-uses-radius")
        if n != 6
        else _two_dp(2 * r * math.sin(math.pi / (2 * n)), "polygon-side-half-angle-missing"),
        _two_dp(2 * r * math.sin(2 * math.pi / n), "polygon-side-uses-full-angle"),
        _two_dp(2 * r * math.sin(math.pi / n) * 2, "polygon-side-doubled"),
    ]
    radius_cands = [
        _num_dist(w // n, "polygon-radius-divides-by-n"),
        _num_dist(w, "polygon-radius-is-the-constant"),
        _num_dist(n * r, "polygon-radius-multiplies-by-n"),
        _num_dist(r + 1, "polygon-radius-off-by-one"),
    ]
    perim_cands = [
        _two_dp(side2 * (n - 1), "polygon-perimeter-counts-n-minus-one"),
        _two_dp(side2 * n * 2, "polygon-perimeter-doubled"),
        _two_dp(side2, "polygon-perimeter-is-one-side"),
        _two_dp(side2 * n / 2, "polygon-perimeter-halved"),
    ]
    root_cands = [
        cdist(sympy.conjugate(root_exact), "polygon-vertex-wrong-direction"),
        cdist(sympy.expand(r * _cis_exact(n, 2)), "polygon-vertex-off-by-one"),
        cdist(sympy.expand(root_exact / r), "polygon-vertex-forgets-modulus"),
    ]
    first = need(str(r), radius_cands, 3)
    return dict(
        text=phrase(rng, _EXAM_STEMS, n=n, w=w),
        parts=[
            dict(
                prompt="Find the modulus of each root.",
                answer_kind="number",
                answer=r,
                answer_text=str(r),
                marks=1,
                sensible_range=(0, None),
                working=[f"|z|^{n} = {w}, so |z| = {r}."],
                distractors=first,
            ),
            dict(
                prompt="Find the length of each side of the polygon, correct to 2 decimal places.",
                answer_kind="number",
                answer=side2,
                answer_text=side_text,
                marks=2,
                follow_through=f"2*part_a*sin(pi/{n})",
                sensible_range=(0, None),
                working=[f"Side = 2r sin(π/{n}) = 2 × {r} × sin(π/{n}) = {side_text}."],
                distractors=need(side_text, chord_cands, 3),
            ),
            dict(
                prompt="Hence find the perimeter of the polygon, correct to 2 decimal places.",
                answer_kind="number",
                answer=round(perimeter + 1e-12, 2),
                answer_text=perimeter_text,
                marks=1,
                follow_through=f"{n}*part_b",
                sensible_range=(0, None),
                working=[f"Perimeter = {n} × {side_text} = {perimeter_text}."],
                distractors=need(perimeter_text, perim_cands, 3),
            ),
            dict(
                prompt="Find, in exact form, the root whose argument is 2π/" + str(n) + ".",
                answer_kind="expression",
                answer=cx_value(root_exact),
                answer_text=cx_text(root_exact),
                marks=2,
                working=[f"z = {r}(cos(2π/{n}) + i sin(2π/{n})) = {cx_text(root_exact)}."],
                distractors=need(cx_text(root_exact), root_cands, 3),
            ),
        ],
    )


def _two_dp(value, tag):
    return dist(f"{value:.2f}", tag, round(value + 1e-12, 2))
