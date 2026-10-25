"""Generates `nx_complex_arg_ratio` (easy, medium, hard) and the exam chain `exd_nx_complex_arg_ratio`.

Skill ``complex-arg-ratio-geometry`` (``senior-mathematics``, answer ``expression``). The existing
``complex-rotation-and-vertex-completion`` covers rotating a point and completing a square or an equilateral triangle;
this one covers the rest of the geometric toolkit: the ratio (z1 - z2)/(z3 - z2) as the test for collinear and
perpendicular points and as the angle at a vertex, the image of a point under w = az + b, midpoint and distance.

* easy: the image of a point under w = az + b; the midpoint of two points; the distance between two points.
* medium: the ratio (C - B)/(A - B) as a + bi; the multiplier a from the images of two points under w = az + b; the
  real k that puts C = k + ci on the line through A and B.
* hard: the angle ABC in degrees from arg((C - B)/(A - B)) (right, 45 and 135 degree angles, and 30, 60, 120 and 150
  degrees with surd coordinates); the real k that makes the angle at B a right angle.
* exam (``exd_nx_complex_arg_ratio``): from A, B, C, (a) the ratio (C - B)/(A - B), (b) the angle ABC following through
  from (a), (c) the midpoint M of AC, (d) the point D that makes M the midpoint of BD, following through from (c).

Every key is re-derived by a second route in floating point (the angle from ``cmath.phase``, the collinearity and
perpendicularity from the cross and dot products of the sides).
"""

from __future__ import annotations

import cmath
import math

import re

import sympy

from ..core import attempts, check, generator, paren_negative, phrase, pick_distractors
from ._w6b_kit import I, cdist, cx_text, cx_value, dist, gaussian, need, num_text, plus, retry, want

_MULTIPLIERS = (2, 3, -1, -2, I, -I, 1 + I, 1 - I, 2 * I, 2 + I, 1 + 2 * I)
_PYTHAGOREAN = ((3, 4), (4, 3), (5, 12), (12, 5), (6, 8), (8, 6), (8, 15), (15, 8), (9, 12), (12, 9))


def _g(rng, span, *, parts_nonzero=True):
    for _ in attempts(500, "a Gaussian integer"):
        a, b = rng.randint(-span, span), rng.randint(-span, span)
        if parts_nonzero and (a == 0 or b == 0):
            continue
        return gaussian(a, b)
    raise AssertionError  # pragma: no cover


def _cmp(z) -> complex:
    return complex(sympy.N(z))


def _signed_terms(mag) -> str:
    """A positive two-term sum with its positive term first: ``√3 - 1``, ``2 + √3``."""
    terms = sorted(sympy.Add.make_args(sympy.expand(mag)), key=lambda t: (not t.is_positive, str(t)))
    out = ""
    for t in terms:
        text = num_text(abs(t))
        out += text if not out else (" + " if t.is_positive else " - ") + text
    return ("-" if terms[0].is_negative else "") + out


def _cx(z) -> str:
    """`cx_text`, except that an imaginary part with two terms prints as a signed bracket: ``3 - (1 + 2√3)i`` and not
    ``3 + (-1 - 2√3)i``."""
    z = sympy.expand(z)
    im_ = sympy.expand(sympy.im(z))
    if not isinstance(im_, sympy.Add):
        return cx_text(z)
    negative = bool(im_.is_negative)
    mag = sympy.expand(-im_ if negative else im_)
    # the larger term first and a leading minus kept inside the bracket only when the terms differ in sign
    imag = f"({_signed_terms(mag)})i"
    re_ = sympy.re(z)
    if re_ == 0:
        return f"-{imag}" if negative else imag
    return f"{cx_text(re_)} {'-' if negative else '+'} {imag}"


def _num(value, tag):
    return dist(str(value), tag, value)


NEAR = "near-miss-value"


_OFFSETS = [complex(a, b) for a in (-3, -2, -1, 0, 1, 2, 3) for b in (-3, -2, -1, 0, 1, 2, 3) if (a, b) != (0, 0)]


def _kind(z) -> str:
    """``real``, ``imaginary`` (pure) or ``mixed``."""
    z = sympy.expand(z)
    if sympy.im(z) == 0:
        return "real"
    return "imaginary" if sympy.re(z) == 0 else "mixed"


def _gaussian(z) -> bool:
    """Whether both parts of z are whole numbers."""
    z = sympy.expand(z)
    return bool(sympy.re(z).is_integer and sympy.im(z).is_integer)


def _signature(text: str, z):
    """What a learner can read off an option without working: its kind and how many digits it prints."""
    return _kind(z), len(re.findall(r"\d", text))


_STANDARD_DEGREES = (30, 45, 60, 90, 120, 135, 150)


def pick(rng, answer_text, cands, n, *, key=None, keyz=None, lo=None, hi=None, step=(1, 2, 3)):
    """`n` wrong options with the key's place among them drawn at random, not left to the order of the slips.

    A complex option has no order, so those are taken in a random order. For a numeric key the number of options below
    it is drawn at random (as far as the slips and plain near misses allow); named slips come before plain fillers."""
    cands = [c for c in cands if c is not None]
    if key is None:
        named = need(answer_text, cands, len(cands))
        if keyz is not None:  # the options share the key's shape: Gaussian-integer keys get Gaussian-integer options
            named = [d for d in named if _gaussian(sympy.sympify(d.value)) == _gaussian(keyz)]
        rng.shuffle(named)
        if keyz is None:
            return named[:n]
        # a complex option has no order, but its real and imaginary parts each have a rank among the options: the named
        # slips sit at fixed places, so one or two of them are mixed with near misses at random offsets
        keep = named[: rng.randint(1, min(2, n - 1, len(named)))] if named else []
        used = {d.text for d in keep} | {answer_text}
        offsets = rng.sample(_OFFSETS, len(_OFFSETS))
        for off in offsets:
            if len(keep) == n:
                break
            z = keyz + gaussian(int(off.real), int(off.imag))
            f = cdist(z, NEAR)
            if f.text not in used:
                used.add(f.text)
                keep.append(f)
        # a twin: one wrong option of the key's own kind and digit count, so the key is never the lone real / imaginary /
        # mixed option nor the lone one-digit (or digit-free) option
        sig = _signature(answer_text, keyz)
        if not any(_signature(d.text, sympy.sympify(d.value)) == sig for d in keep):
            for off in rng.sample(_OFFSETS, len(_OFFSETS)):
                z = keyz + gaussian(int(off.real), int(off.imag))
                f = cdist(z, NEAR)
                if f.text not in used and _signature(f.text, z) == sig and _gaussian(z) == _gaussian(keyz):
                    drop = next((d for d in reversed(keep) if d.misconception == NEAR), keep[-1])
                    keep[keep.index(drop)] = f
                    break
            else:
                want(False, "no twin for the key")
        rng.shuffle(keep)
        want(len(keep) >= 2, "too few options")
        return keep
    fillers = [_num(key + d, NEAR) for k in step for d in (-k, k)]
    named = pick_distractors(str(answer_text), cands, max(len(cands), 1))
    taken = {float(d.value) for d in named}
    extra = [
        f
        for f in fillers
        if float(f.value) not in taken and (lo is None or f.value > lo) and (hi is None or f.value < hi)
    ]
    named = [d for d in named if (lo is None or d.value > lo) and (hi is None or d.value < hi)]

    def side(items, below):
        got = [d for d in items if (d.value < key) == below]
        rng.shuffle(got)
        return got

    nb, na = side(named, True), side(named, False)
    fb, fa = side(extra, True), side(extra, False)
    pool_b, pool_a = nb + fb, na + fa
    want(len(pool_b) + len(pool_a) >= n and len(pool_b) + len(pool_a) >= 2, "too few options")
    low = max(0, n - len(pool_a))
    high = min(n, len(pool_b))
    want(low <= high, "no split")
    below = rng.randint(low, high)
    out = pool_b[:below] + pool_a[: n - below]
    if (
        lo is not None
        and hi is not None
        and key in _STANDARD_DEGREES
        and not any(d.value in _STANDARD_DEGREES for d in out)
    ):
        # an angle key that is a standard angle keeps company: one wrong option is a standard angle too
        twins = [v for v in _STANDARD_DEGREES if v != key and lo < v < hi and v not in {d.value for d in out}]
        want(bool(twins), "no standard-angle twin")
        out[rng.randrange(len(out))] = _num(rng.choice(twins), NEAR)
    rng.shuffle(out)
    return out


def _expr_fields(z):
    return dict(answer_kind="expression", answer=cx_value(z), answer_text=_cx(z))


def _num_fields(v):
    return dict(answer_kind="number", answer=v, answer_text=str(v))


def _deg_fields(v):
    return dict(answer_kind="number", answer=v, answer_text=str(v))


def _kc(c: int) -> str:
    """The imaginary part of k + ci as it follows the k: ``+ 3i`` or ``- 7i``."""
    return f"{'+' if c > 0 else '-'} {abs(c) if abs(c) != 1 else ''}i"


def _kminus(v: int) -> str:
    """``k - 3`` or ``k + 3`` (``k`` for 0), never ``k - (-3)``."""
    return "k" if v == 0 else f"k - {v}" if v > 0 else f"k + {-v}"


def _br(z) -> str:
    """The multiplier as text: a whole number (any sign) or i as it stands, anything else bracketed."""
    t = _cx(z)
    return t if t == "i" or sympy.im(z) == 0 else f"({t})"


def _term(z) -> str:
    """The constant as a signed term after ``z``: ``+ 3``, ``- 3i``, ``+ (2 - i)``, ``- (1 + 3i)`` (never ``+ (-3)``)."""
    re_, im_ = sympy.re(z), sympy.im(z)
    if re_ <= 0 and im_ <= 0:
        return f"- {_br_abs(-z)}"
    return f"+ {_br_abs(z)}"


def _br_abs(z) -> str:
    t = _cx(z)
    return t if sympy.im(z) == 0 or t == "i" or (sympy.re(z) == 0) else f"({t})"


# -- easy -----------------------------------------------------------------------------------------------------

_MAP_STEMS = (
    "A transformation of the complex plane is w = {a}z {b}. Find the image of the point {z} under this transformation.",
    "The point P represents z = {z}. P is mapped to P' by w = {a}z {b}. Which complex number does P' represent?",
    "Under the map z -> {a}z {b} of the Argand diagram, where does the point {z} go? Give the complex number "
    "that its image represents.",
)
_MID_STEMS = (
    "On an Argand diagram, A represents {a} and B represents {b}. Find the complex number represented by the midpoint of AB.",
    "The points {a} and {b} are the ends of a line segment in the complex plane. What complex number marks the midpoint "
    "of the segment?",
    "M is the midpoint of the interval joining the points {a} and {b}. Write M as a complex number.",
)
_DIST_STEMS = (
    "Find the distance between the points {a} and {b} on the complex plane.",
    "A and B represent {a} and {b} on an Argand diagram. How far apart are A and B?",
    "Two points of the complex plane represent {a} and {b}. Find the length of the segment joining them.",
)


def _bracket_free(z) -> str:
    return _cx(z)


def _easy_map(rng):
    a = rng.choice(_MULTIPLIERS)
    b = _g(rng, 6, parts_nonzero=False)
    z = _g(rng, 6)
    want(sympy.expand(a * z) != z and b != 0)
    w = sympy.expand(a * z + b)
    check(abs(_cmp(w) - (_cmp(a) * _cmp(z) + _cmp(b))) < 1e-9, "image differs")
    cands = [
        cdist(a * z, "nx-map-constant-dropped"),
        cdist(z + b, "nx-map-multiplier-dropped"),
        cdist(a * (z + b), "nx-map-adds-before-multiplying"),
        cdist(a * z - b, "nx-map-constant-subtracted"),
        cdist(
            sympy.expand(sympy.re(a) * sympy.re(z) + sympy.re(b))
            + I * sympy.expand(sympy.im(a) * sympy.im(z) + sympy.im(b)),
            "nx-map-multiplies-parts-separately",
        ),
    ]
    working = [
        f"Multiply: ({_cx(a)})({_cx(z)}) = {_cx(a * z)}.",
        f"Add the constant: {_cx(a * z)} {plus(_cx(b))} = {_cx(w)}." if b != 0 else "No constant to add.",
    ]
    return dict(
        text=phrase(rng, _MAP_STEMS, a=_br(a), b=_term(b), z=_cx(z)),
        working=working,
        distractors=pick(rng, _cx(w), cands, 3, keyz=w),
        **_expr_fields(w),
    )


def _easy_mid(rng):
    for _ in attempts(500, "two points of one parity"):
        a, b = _g(rng, 8), _g(rng, 8)
        if sympy.re(a - b) % 2 == 0 and sympy.im(a - b) % 2 == 0 and a != b:
            break
    m = sympy.expand((a + b) / 2)
    cands = [
        cdist(a + b, "nx-midpoint-sum-not-halved"),
        cdist(sympy.expand((a - b) / 2), "nx-midpoint-half-the-difference"),
        cdist(sympy.expand(sympy.re(a + b) / 2 + I * sympy.im(a + b)), "nx-midpoint-halves-real-part-only"),
        cdist(sympy.expand(sympy.im(a + b) / 2 + I * sympy.re(a + b) / 2), "nx-midpoint-parts-swapped"),
    ]
    working = [
        f"The midpoint is (a + b)/2 = ({_cx(a)} {plus(_cx(b))})/2.",
        f"= ({_cx(a + b)})/2 = {_cx(m)}.",
    ]
    return dict(
        text=phrase(rng, _MID_STEMS, a=_cx(a), b=_cx(b)),
        working=working,
        distractors=pick(rng, _cx(m), cands, 3, keyz=m),
        **_expr_fields(m),
    )


def _easy_dist(rng):
    p, q = rng.choice(_PYTHAGOREAN)
    p, q = rng.choice([-1, 1]) * p, rng.choice([-1, 1]) * q
    a = _g(rng, 6)
    b = sympy.expand(a + gaussian(p, q))
    d = int(math.hypot(p, q))
    check(d * d == p * p + q * q and abs(abs(_cmp(a) - _cmp(b)) - d) < 1e-9, "distance differs")
    cands = [
        _num(abs(p) + abs(q), "nx-distance-adds-the-parts"),
        _num(p * p + q * q, "nx-distance-forgets-square-root"),
        _num(max(abs(p), abs(q)), "nx-distance-takes-the-larger-part"),
        _num(abs(p - q) if abs(p) != abs(q) else abs(p), "nx-distance-subtracts-the-parts"),
    ]
    working = [
        f"a - b = {_cx(a - b)}.",
        f"|a - b| = √({paren_negative(p)}² + {paren_negative(q)}²) = √{p * p + q * q} = {d}.",
    ]
    return dict(
        text=phrase(rng, _DIST_STEMS, a=_cx(a), b=_cx(b)),
        working=working,
        distractors=pick(rng, str(d), cands, 3, key=d, lo=0),
        **_num_fields(d),
    )


def _easy(rng):
    return retry(rng, rng.choice([_easy_map, _easy_map, _easy_mid, _easy_dist]))


# -- medium ---------------------------------------------------------------------------------------------------

_RATIO_STEMS = (
    "The points A, B and C on an Argand diagram represent {a}, {b} and {c}. Write (C - B)/(A - B) in the form x + yi.",
    "On the complex plane, A = {a}, B = {b} and C = {c}. Find the ratio (c - b)/(a - b) of the complex numbers "
    "represented, in the form x + yi.",
    "Let A, B and C represent {a}, {b} and {c}. Calculate the quotient of the vectors BC and BA, that is "
    "(c - b)/(a - b), giving x + yi.",
)
_MULT_STEMS = (
    "A transformation w = az + b, where a and b are complex constants, maps {z1} to {w1} and {z2} to {w2}. Find a.",
    "The map z -> az + b of the complex plane sends the point {z1} to {w1} and the point {z2} to {w2}. Find the "
    "complex number a.",
    "Suppose w = az + b sends {z1} to {w1} and {z2} to {w2}. Work out the constant a.",
)
_COLL_STEMS = (
    "The points A, B and C represent {a}, {b} and k {c}, where k is real. A, B and C lie on a straight line, so "
    "(C - A)/(B - A) is real. Find k.",
    "On an Argand diagram, A = {a}, B = {b} and C = k {c} for a real number k. The three points are collinear. Find k.",
    "Find the real number k so that the points {a}, {b} and k {c} are collinear on the complex plane.",
)


def _medium_ratio(rng):
    d = _g(rng, 4, parts_nonzero=False)
    want(d != 0)
    m = rng.choice([1 + I, -1 + I, 1 - I, -1 - I, 2 + I, 1 + 2 * I, 2 - I, -2 + I, 3 + I, 2 * I, -I, 1 - 2 * I, 3 - I])
    b = _g(rng, 5)
    a = sympy.expand(b + d)
    c = sympy.expand(b + d * m)
    q = sympy.expand(m)
    check(abs((_cmp(c) - _cmp(b)) / (_cmp(a) - _cmp(b)) - _cmp(q)) < 1e-9, "ratio differs")
    inv = sympy.expand(sympy.conjugate(m) / (sympy.re(m) ** 2 + sympy.im(m) ** 2))
    cands = [
        cdist(inv, "nx-ratio-inverted"),
        cdist(-q, "nx-ratio-difference-reversed"),
        cdist(sympy.expand((c - b) - (a - b)), "nx-ratio-subtracted-not-divided"),
        cdist(sympy.expand(sympy.conjugate(q)), "nx-ratio-conjugated"),
    ]
    den = sympy.expand(d * sympy.conjugate(d))
    working = [
        f"c - b = {_cx(c - b)} and a - b = {_cx(a - b)}.",
        f"({_cx(c - b)})/({_cx(a - b)}): multiply top and bottom by the conjugate {_cx(sympy.conjugate(d))}"
        f" so the denominator becomes {den}.",
        f"The ratio is {_cx(q)}.",
    ]
    return dict(
        text=phrase(rng, _RATIO_STEMS, a=_cx(a), b=_cx(b), c=_cx(c)),
        working=working,
        distractors=pick(rng, _cx(q), cands, 3, keyz=q),
        **_expr_fields(q),
    )


def _medium_multiplier(rng):
    a = rng.choice([2, 3, -1, -2, I, -I, 1 + I, 1 - I, 2 * I, 2 + I, 1 + 2 * I, 2 - I, -1 + I, 3 - I])
    b = _g(rng, 6, parts_nonzero=False)
    z1, z2 = _g(rng, 5), _g(rng, 5)
    want(z1 != z2)
    w1, w2 = sympy.expand(a * z1 + b), sympy.expand(a * z2 + b)
    check(abs((_cmp(w1) - _cmp(w2)) / (_cmp(z1) - _cmp(z2)) - _cmp(a)) < 1e-9, "multiplier differs")
    dz = z1 - z2
    dw = w1 - w2
    s = sympy.expand(z1 + z2)
    cands = [
        cdist(dw, "nx-multiplier-difference-not-divided"),
        cdist(sympy.expand(dz / dw) if dw != 0 else 0, "nx-multiplier-ratio-inverted"),
        cdist(sympy.expand(w1 / z1), "nx-multiplier-from-one-point-only"),
        cdist(sympy.expand((w1 + w2) / s) if s != 0 else 0, "nx-multiplier-sums-not-differences"),
        cdist(sympy.expand(w1 - w2 - dz), "nx-multiplier-difference-of-differences"),
    ]
    working = [
        f"Subtracting the two equations removes b: w1 - w2 = a(z1 - z2).",
        f"w1 - w2 = {_cx(dw)} and z1 - z2 = {_cx(dz)}.",
        f"a = ({_cx(dw)})/({_cx(dz)}) = {_cx(a)}.",
    ]
    return dict(
        text=phrase(rng, _MULT_STEMS, z1=_cx(z1), w1=_cx(w1), z2=_cx(z2), w2=_cx(w2)),
        working=working,
        distractors=pick(rng, _cx(a), cands, 3, keyz=a),
        **_expr_fields(a),
    )


def _medium_collinear(rng):
    for _ in attempts(500, "a collinear draw"):
        a = _g(rng, 6)
        u, v = rng.randint(-3, 3), rng.randint(-3, 3)
        if u == 0 or v == 0:
            continue
        t = rng.choice([2, 3, -1, -2, 4, -3])
        break
    d = gaussian(2 * u, 2 * v) if rng.random() < 0.3 else gaussian(u, v)
    dx, dy = int(sympy.re(d)), int(sympy.im(d))
    b = sympy.expand(a + d)
    ax, ay = int(sympy.re(a)), int(sympy.im(a))
    k = ax + t * dx
    c = ay + t * dy
    want(c != 0 and k != ax and k != int(sympy.re(b)))
    point = gaussian(k, c)
    # second route: the cross product of the sides vanishes
    cross = (_cmp(b) - _cmp(a)).real * (_cmp(point) - _cmp(a)).imag - (_cmp(b) - _cmp(a)).imag * (
        _cmp(point) - _cmp(a)
    ).real
    check(abs(cross) < 1e-9, "points are not collinear")
    cands = [
        _num(ax + t * dy, "nx-collinear-parts-swapped"),
        _num(int(sympy.re(b)) + t * dx, "nx-collinear-wrong-base-point"),
        _num(ax - t * dx, "nx-collinear-sign-slip"),
        _num(ax + (c * dx) // dy, "nx-collinear-imaginary-offset-ignored") if (c * dx) % dy == 0 else None,
    ]
    working = [
        f"(C - A)/(B - A) is real, where C - A = ({_kminus(ax)}) {_kc(c - ay)} and B - A = {_cx(d)}.",
        f"Use the cross product: ({dx})({c - ay}) = ({dy})({_kminus(ax)}), so {_kminus(ax)} = {t * dx}.",
        f"k = {k}.",
    ]
    return dict(
        text=phrase(rng, _COLL_STEMS, a=_cx(a), b=_cx(b), c=_kc(c)),
        working=working,
        distractors=pick(rng, str(k), cands, 3, key=k),
        answer_kind="number",
        answer=k,
        answer_text=str(k),
    )


def _medium(rng):
    return retry(
        rng, rng.choice([_medium_ratio, _medium_ratio, _medium_multiplier, _medium_multiplier, _medium_collinear])
    )


# -- hard -----------------------------------------------------------------------------------------------------

_ANGLE_STEMS = (
    "On an Argand diagram, A, B and C represent {a}, {b} and {c}. Use arg((c - b)/(a - b)) to find the size of angle "
    "ABC in degrees.",
    "The vertices of triangle ABC are A = {a}, B = {b} and C = {c} on the complex plane. By finding the argument of "
    "(c - b)/(a - b), state angle ABC in degrees.",
    "Triangle ABC has A({a}), B({b}) and C({c}) on an Argand diagram. Find the angle at B, in degrees, using the "
    "argument of the quotient of the vectors BC and BA.",
)
_PERP_STEMS = (
    "The points A, B and C represent {a}, {b} and k {c}, where k is real. Angle ABC is a right angle, so "
    "(A - B)/(C - B) is purely imaginary. Find k.",
    "On the complex plane, A = {a}, B = {b} and C = k {c}, with k real. BA is perpendicular to BC. Find k.",
    "Find the real number k for which the line from {b} to {a} is perpendicular to the line from {b} to k {c}.",
)

_GAUSS_ANGLES = ((1 + I, 45), (-1 + I, 135), (I, 90), (1 - I, 45), (-1 - I, 135), (-I, 90))
_SURD_ANGLES = (
    (1 + sympy.sqrt(3) * I, 60),
    (-1 + sympy.sqrt(3) * I, 120),
    (sympy.sqrt(3) + I, 30),
    (-sympy.sqrt(3) + I, 150),
    (1 - sympy.sqrt(3) * I, 60),
    (-1 - sympy.sqrt(3) * I, 120),
    (sympy.sqrt(3) - I, 30),
    (-sympy.sqrt(3) - I, 150),
)


def _hard_angle(rng):
    surd = rng.random() < 0.5
    if surd:
        m, theta = rng.choice(_SURD_ANGLES)
        scale = 1
    else:
        m, theta = rng.choice(_GAUSS_ANGLES)
        scale = rng.choice([1, 2, 3])
    if surd:
        d = rng.choice([1, 2, 3, -1, -2, -3, I, 2 * I, -I, -2 * I])
    else:
        d = _g(rng, 4, parts_nonzero=False)
        want(d != 0)
    b = _g(rng, 4)
    a = sympy.expand(b + d)
    c = sympy.expand(b + d * m * scale)
    q = sympy.expand(m * scale)
    # second route: the angle between the two sides from the dot product
    u, w = _cmp(a) - _cmp(b), _cmp(c) - _cmp(b)
    cos_angle = (u.real * w.real + u.imag * w.imag) / (abs(u) * abs(w))
    check(abs(math.degrees(math.acos(max(-1, min(1, cos_angle)))) - theta) < 1e-7, "angle differs")
    check(abs(abs(math.degrees(cmath.phase(_cmp(q)))) - theta) < 1e-7, "argument differs")
    cands = [
        _num(180 - theta, "nx-angle-supplement-taken"),
        _num(abs(90 - theta), "nx-angle-complement-taken") if theta != 90 else None,
        _num(theta // 2, "nx-angle-halved") if theta % 2 == 0 else None,
        _num(2 * theta, "nx-angle-doubled") if 2 * theta < 180 else None,
    ]
    working = [
        f"(c - b)/(a - b) = ({_cx(c - b)})/({_cx(a - b)}) = {_cx(q)}.",
        f"The angle at B is |arg of this ratio| = {theta}°.",
    ]
    return dict(
        text=phrase(rng, _ANGLE_STEMS, a=_cx(a), b=_cx(b), c=_cx(c)),
        working=working,
        distractors=pick(rng, str(theta), cands, 3, key=theta, lo=0, hi=180, step=(5, 10, 15, 30)),
        **_deg_fields(theta),
    )


def _hard_perp(rng):
    for _ in attempts(500, "a perpendicular draw"):
        p, q = rng.randint(-4, 4), rng.randint(-4, 4)
        s = rng.choice([1, 2, 3, -1, -2, -3])
        if p == 0 or q == 0 or abs(p) == abs(q):
            continue
        break
    b = _g(rng, 5)
    bx, by = int(sympy.re(b)), int(sympy.im(b))
    a = gaussian(bx + p, by + q)
    k, c = bx - s * q, by + s * p
    point = gaussian(k, c)
    # second route: the dot product of the sides vanishes
    u, w = _cmp(a) - _cmp(b), _cmp(point) - _cmp(b)
    check(abs(u.real * w.real + u.imag * w.imag) < 1e-9, "sides are not perpendicular")
    want(c != 0 and k != 0)
    cands = [
        _num(bx + s * p, "nx-perp-uses-parallel-condition"),
        _num(bx - s * p, "nx-perp-parts-swapped"),
        _num(bx + p - s * q, "nx-perp-measured-from-the-wrong-vertex"),
        _num(-s * q, "nx-perp-base-point-ignored"),
    ]
    working = [
        f"BA = {_cx(a - b)}. C - B = ({_kminus(bx)}) {_kc(c - by)}.",
        f"BA is perpendicular to BC, so the dot product is zero: ({p})({_kminus(bx)}) + ({q})({c - by}) = 0.",
        f"{_kminus(bx)} = {-s * q}, so k = {k}.",
    ]
    return dict(
        text=phrase(rng, _PERP_STEMS, a=_cx(a), b=_cx(b), c=_kc(c)),
        working=working,
        distractors=pick(rng, str(k), cands, 3, key=k),
        answer_kind="number",
        answer=k,
        answer_text=str(k),
    )


def _hard(rng):
    return retry(rng, rng.choice([_hard_angle, _hard_angle, _hard_perp]))


@generator(
    "nx_complex_arg_ratio",
    level=17,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def nx_complex_arg_ratio(rng, tier="easy"):
    """Geometry of the complex plane by the ratio of two vectors, w = az + b, midpoints and distances."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    if tier != "hard":
        raise ValueError(f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam -----------------------------------------------------------------------------------------------------

_EXAM_STEMS = (
    "The points A, B and C on an Argand diagram represent the complex numbers {a}, {b} and {c}.",
    "Triangle ABC is drawn on the complex plane with A at {a}, B at {b} and C at {c}.",
    "On an Argand diagram the vertices of triangle ABC are A = {a}, B = {b} and C = {c}.",
)


def _sym(z) -> str:
    return f"{int(sympy.re(z))} + ({int(sympy.im(z))})*i"


def _xfields(z, marks, **extra):
    return dict(answer_kind="expression", answer=_sym(z), answer_text=_cx(z), marks=marks, **extra)


@generator("exd_nx_complex_arg_ratio", level=17, presets={"exam": dict(tier="exam")})
def exd_nx_complex_arg_ratio(rng, tier="exam"):
    """Exam chain: the ratio (C - B)/(A - B), the angle ABC from it, the midpoint of AC, the point D opposite B."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return retry(rng, _exam)


def _exam(rng):
    m, theta = rng.choice(_GAUSS_ANGLES)
    scale = rng.choice([1, 2, 3])
    p, q = rng.randint(-3, 3), rng.randint(-3, 3)
    want((p, q) != (0, 0))
    d = gaussian(p, q)
    b = _g(rng, 4)
    q_ratio = sympy.expand(m * scale)
    a = sympy.expand(b + d)
    c = sympy.expand(b + d * q_ratio)
    want(sympy.re(a + c) % 2 == 0 and sympy.im(a + c) % 2 == 0)
    mid = sympy.expand((a + c) / 2)
    dd = sympy.expand(2 * mid - b)
    check(dd == sympy.expand(a + c - b), "opposite vertex differs")
    check(abs(abs(math.degrees(cmath.phase(_cmp(q_ratio)))) - theta) < 1e-7, "angle differs")
    bx, by = int(sympy.re(b)), int(sympy.im(b))
    inv = sympy.expand(sympy.conjugate(q_ratio) / (sympy.re(q_ratio) ** 2 + sympy.im(q_ratio) ** 2))
    ratio_cands = [
        cdist(inv, "nx-ratio-inverted"),
        cdist(-q_ratio, "nx-ratio-difference-reversed"),
        cdist(sympy.expand(c - a), "nx-ratio-subtracted-not-divided"),
        cdist(sympy.expand(sympy.conjugate(q_ratio)), "nx-ratio-conjugated"),
    ]
    angle_cands = [
        _num(180 - theta, "nx-angle-supplement-taken"),
        _num(abs(90 - theta), "nx-angle-complement-taken") if theta != 90 else None,
        _num(2 * theta, "nx-angle-doubled") if 2 * theta < 180 else None,
        _num(theta // 2, "nx-angle-halved") if theta % 2 == 0 else None,
    ]
    mid_cands = [
        cdist(a + c, "nx-midpoint-sum-not-halved"),
        cdist(
            sympy.expand((c - a) / 2) if (sympy.re(c - a) % 2 == 0 and sympy.im(c - a) % 2 == 0) else a + c - b,
            "nx-midpoint-half-the-difference",
        ),
        cdist(sympy.expand(sympy.re(a + c) / 2 + I * sympy.im(a + c)), "nx-midpoint-halves-real-part-only"),
    ]
    d_cands = [
        cdist(mid, "nx-opposite-vertex-midpoint-taken"),
        cdist(sympy.expand(mid - b), "nx-opposite-vertex-reflected-wrongly"),
        cdist(sympy.expand(2 * mid + b), "nx-opposite-vertex-adds-b"),
        cdist(sympy.expand(a + c), "nx-opposite-vertex-sum-of-a-and-c"),
    ]
    stem = phrase(rng, _EXAM_STEMS, a=_cx(a), b=_cx(b), c=_cx(c))
    return dict(
        text=stem,
        parts=[
            dict(
                prompt="Find (c - b)/(a - b) in the form x + yi.",
                working=[
                    f"c - b = {_cx(c - b)}, a - b = {_cx(a - b)}.",
                    f"The ratio is ({_cx(c - b)})/({_cx(a - b)}) = {_cx(q_ratio)}.",
                ],
                distractors=pick(rng, _cx(q_ratio), ratio_cands, 3, keyz=q_ratio),
                **_xfields(q_ratio, 2),
            ),
            dict(
                prompt="Hence find the size of angle ABC, in degrees.",
                answer_kind="number",
                answer=theta,
                answer_text=str(theta),
                marks=1,
                follow_through="abs(arg(Subs(part_a,(i,),(I,))))*180/pi",
                sensible_range=(0, 180),
                working=[f"Angle ABC is the size of arg((c - b)/(a - b)), that is {theta}°."],
                distractors=pick(rng, str(theta), angle_cands, 3, key=theta, lo=0, hi=180, step=(5, 10, 15, 30)),
            ),
            dict(
                prompt="Find the complex number that represents M, the midpoint of AC.",
                working=[f"M = (a + c)/2 = ({_cx(a + c)})/2 = {_cx(mid)}."],
                distractors=pick(rng, _cx(mid), mid_cands, 3, keyz=mid),
                **_xfields(mid, 1),
            ),
            dict(
                prompt="ABCD is a parallelogram, so M is also the midpoint of BD. Hence find the complex number that "
                "represents D.",
                follow_through=f"2*part_c - ({bx} + ({by})*i)",
                working=[
                    "M is the midpoint of BD, so d = 2m - b.",
                    f"d = 2({_cx(mid)}) - ({_cx(b)}) = {_cx(dd)}.",
                ],
                distractors=pick(rng, _cx(dd), d_cands, 3, keyz=dd),
                **_xfields(dd, 2),
            ),
        ],
    )
