"""Features of power-function graphs: `vm_power_graph_features` (easy, medium, hard) and the exam chain
`vm_power_graph_features_exam`.

Skill `power-function-graph-features` (VCE Mathematical Methods). The sibling skill `power-functions-rational-index`
evaluates x^(p/q), finds domain and range endpoints and solves equations; this one is about the shape of the graph.
Answers are an ordered list (a word, or numbers), so every tier shares one answer kind.

* easy: odd, even or neither for a x^(p/q) and a x^(-n) (an even root limits the domain, so such a function is neither);
  the y-value at -X from the y-value at X by symmetry; the asymptote of a x^(-n) translated;
* medium: the quadrants the graph lies in from the signs of a and the parity of p and q; both asymptotes of
  a(x - h)^(-n) + k; the quadrants of the image after a reflection in an axis or the origin;
* hard: every x-coordinate where a x^(p/q) meets y = x; the y-coordinate on an image after a dilation and a reflection;
  the x-intercepts of a(x - h)^(-n) + k;
* exam: y = a x^(-n) through a point: a, the value at -u by symmetry (following through from a), the quadrants, and the
  positive x-intercept after a vertical translation (following through from a).

For x^(p/q) in lowest terms the graph has a mirror image through the origin when q is odd: odd when p is odd, even when p
is even. When q is even, x < 0 is outside the domain and the function is neither.
"""

from __future__ import annotations

from fractions import Fraction

import sympy

from ..core import attempts, check, generator, phrase
from .exd_x_r2b_support import Redraw, draw, words_part
from ._exd_distinct_parts import distinct_part_answers
from ._spread_kit import _RNG, mixed, uses_rng
from .exd_x_r2b_support import fraction_part as _fraction_part
from .r4e_common import single
from .r4e_power_functions_build import index_text, real_power

_POS = [(1, 2), (1, 3), (2, 3), (3, 2), (4, 3), (5, 3), (3, 4), (2, 5), (3, 5), (5, 2)]
_NEG = [(-1, 1), (-2, 1), (-3, 1), (-4, 1), (-1, 2), (-1, 3), (-2, 3), (-3, 2), (-4, 3), (-3, 4)]
_INDICES = _POS + _NEG


def pw(base: str, p: int, q: int) -> str:
    """base^(p/q) as written on paper: a positive whole index needs no brackets."""
    return f"{base}^{p}" if q == 1 and p > 0 else f"{base}^({index_text(p, q)})"


def coef(a: int) -> str:
    return "" if a == 1 else "-" if a == -1 else str(a)


def shifted(h: int) -> str:
    return "x" if h == 0 else f"(x - {h})" if h > 0 else f"(x + {-h})"


def with_const(text: str, k: int) -> str:
    return text if k == 0 else f"{text} + {k}" if k > 0 else f"{text} - {-k}"


def fx_text(a: int, p: int, q: int, h: int = 0, k: int = 0) -> str:
    return with_const(coef(a) + pw(shifted(h), p, q), k)


def parity(p: int, q: int) -> str:
    """odd, even or neither for x^(p/q) in lowest terms."""
    if q % 2 == 0:
        return "neither"
    return "odd" if p % 2 else "even"


def quads(sign: int, p: int, q: int, *, ignore_root=False, swap=False) -> list[int]:
    """Quadrants a x^(p/q) occupies. `ignore_root` pretends q is odd; `swap` swaps odd and even."""
    right = 1 if sign > 0 else 4
    if q % 2 == 0 and not ignore_root:
        return [right]
    odd = bool(p % 2) != swap
    left = (2 if sign > 0 else 3) if not odd else (3 if sign > 0 else 2)
    return sorted([right, left])


_REFLECT = {
    "x-axis": {1: 4, 4: 1, 2: 3, 3: 2},
    "y-axis": {1: 2, 2: 1, 3: 4, 4: 3},
    "origin": {1: 3, 3: 1, 2: 4, 4: 2},
}


def reflect(qs, axis: str) -> list[int]:
    return sorted(_REFLECT[axis][v] for v in qs)


def _spread(value, wrong, lo=None, hi=None, zero_ok=False, exclude=()):
    """Wrong answers for a single number, reordered so that the first three hold a random number below the key and the
    rest above it (`_spread_kit.mixed`), with plain near misses to make up a side that is short. A value of a power
    function at a non-zero x is never 0, so 0 is left out of the options unless `zero_ok`."""
    near = [
        (value + d, "near-miss-value")
        for d in (-4, -3, -2, -1, 1, 2, 3, 4)
        if (lo is None or value + d >= lo)
        and (hi is None or value + d <= hi)
        and (zero_ok or value + d != 0)
        and value + d not in exclude
    ]
    wrong = [w for w in wrong if (zero_ok or w[0] != 0) and w[0] not in exclude]
    return mixed(
        value,
        list(wrong),
        fillers=near,
        lo=lo,
        hi=hi,
        unbounded="open" if lo is None and hi is None else None,
        slip_range=(None, None) if hi is None else None,  # a stated upper bound binds the named slips too
    )


def fraction_part(prompt, value, marks, working, wrong, lo=None, hi=None, zero_ok=False, **kw):
    return _fraction_part(prompt, value, marks, working, _spread(Fraction(value), wrong, lo, hi, zero_ok=zero_ok), **kw)


def one_word(prompt, word, marks, working, wrong):
    return words_part(prompt, [word], marks, working, [([w], t) for w, t in wrong])


def nums(prompt, values, marks, working, wrong, lo=None, hi=None, zero_ok=False, exclude=(), choice=False):
    """`choice` marks a set of labels (quadrant numbers), not a quantity: no plain near misses are invented for it."""
    values = [int(v) for v in values]
    wrong = [([int(x) for x in vs], t) for vs, t in wrong if vs is not None]
    if len(values) == 1 and not choice:
        pairs = _spread(values[0], [(vs[0], t) for vs, t in wrong if len(vs) == 1], lo, hi, zero_ok, exclude)
        wrong = [([int(v)], t) for v, t in pairs]
    elif _RNG.get() is not None:
        _RNG.get().shuffle(wrong)
    return words_part(prompt, values, marks, working, wrong)


def num_text(values) -> str:
    return ", ".join(str(v) for v in values)


# -- easy ----------------------------------------------------------------------------------------------------------


def _easy_symmetry(rng):
    key = rng.choice(("odd", "even", "neither"))  # drawn first so that no answer is the usual one
    p, q = rng.choice([i for i in _INDICES if parity(*i) == key])
    a = rng.choice((1, 1, 2, 3, -1, -2, 4))
    others = [w for w in ("odd", "even", "neither") if w != key]
    by_p = "even" if p % 2 == 0 else "odd"
    swapped = {"odd": "even", "even": "odd"}.get(key)
    wrong = [
        (by_p, "vm-pf-parity-from-numerator-only"),
        (swapped, "vm-pf-odd-even-swapped"),
        ("neither", "vm-pf-neither-for-defined-function"),
        (others[0], "vm-pf-parity-guess"),
        (others[1], "vm-pf-parity-guess"),
    ]
    wrong = [(w, t) for w, t in wrong if w]
    why = {
        "neither": "an even root means f(x) is not defined for x < 0",
        "odd": "the root is odd and the power is odd, so f(-x) = -f(x)",
        "even": "the root is odd and the power is even, so f(-x) = f(x)",
    }[key]
    return single(
        f"Let f(x) = {fx_text(a, p, q)}.",
        one_word(
            "Is f odd, even or neither? Answer with one of the words odd, even or neither.",
            key,
            2,
            [f"In lowest terms the index is {index_text(p, q)}: {why}."],
            wrong,
        ),
    )


def _easy_mirror_value(rng):
    q = rng.choice((1, 3, 5))
    p = rng.choice([v for v in (-4, -3, -2, -1, 1, 2, 3, 4, 5) if (v, q) in _INDICES or q == 1 or v in (1, 2, 4, 5)])
    if q == 3 and p % 3 == 0 or q == 5 and p % 5 == 0:
        raise Redraw("index not in lowest terms")
    u = rng.choice((1, 2)) if abs(p) > 2 else rng.choice((1, 2, 3))
    big = u**q
    y = rng.choice([v for v in range(2, 13)]) * (u ** max(0, -p))
    sign = 1 if p % 2 == 0 else -1
    key = sign * y
    wrong = [
        (-key, "vm-pf-symmetry-sign-wrong"),
        (key * 2, "vm-pf-symmetry-doubled"),
        (key - 1, "near-miss-value"),
        (key + 1, "near-miss-value"),
        (key + 2, "near-miss-value"),
        (-key + 1, "near-miss-value"),
    ]
    kind = "even" if sign > 0 else "odd"
    stem = (
        f"A function f(x) = a{pw('x', p, q)} has a graph through the point ({big}, {y}), where a is a constant."
        if big != 1
        else f"A function f(x) = a{pw('x', p, q)} has a graph through the point (1, {y}), where a is a constant."
    )
    return single(
        stem,
        nums(
            f"What is the y-coordinate of the point on the graph with x-coordinate -{big}?",
            [key],
            2,
            [f"f is {kind} (the root is odd and the power is {'even' if sign > 0 else 'odd'}), so f(-{big}) = {key}."],
            [([w], t) for w, t in wrong],
        ),
    )


def _easy_asymptote(rng):
    n = rng.choice((1, 2, 3, 4)) if rng.random() < 0.6 else None
    p, q = (-n, 1) if n else rng.choice([(-1, 2), (-1, 3), (-2, 3), (-3, 2)])
    a = rng.choice((1, 2, 3, -1, -2, 5))
    h = rng.choice((0, 0, 2, 3, -1, -2, 4))
    k = rng.choice((-3, -2, -1, 1, 2, 4, 5))
    horizontal = rng.random() < 0.5
    rule = fx_text(a, p, q, h, k)
    if horizontal:
        key, prompt = k, "The graph has a horizontal asymptote with equation y = c. Find c."
        wrong = [
            (0, "vm-pf-asymptote-reads-zero"),
            (-k, "vm-pf-asymptote-sign-of-shift"),
            (a + k, "vm-pf-asymptote-takes-dilation"),
            (h, "vm-pf-asymptote-axes-swapped"),
        ]
    else:
        if h == 0:
            h = rng.choice((2, 3, -2, -3))
            rule = fx_text(a, p, q, h, k)
        key, prompt = h, "The graph has a vertical asymptote with equation x = c. Find c."
        wrong = [
            (0, "vm-pf-asymptote-reads-zero"),
            (-h, "vm-pf-asymptote-sign-of-shift"),
            (k, "vm-pf-asymptote-axes-swapped"),
            (h + 1, "near-miss-value"),
        ]
    working = ["Translating by (h, k) moves the asymptotes from x = 0 and y = 0 to x = h and y = k."]
    return single(f"Let y = {rule}.", nums(prompt, [key], 2, working, [([w], t) for w, t in wrong], zero_ok=True))


# -- medium --------------------------------------------------------------------------------------------------------


def _quad_wrongs(sign, p, q):
    right = quads(sign, p, q)
    return [
        (quads(-sign, p, q), "vm-pf-quadrant-sign-of-a-reversed"),
        (quads(1, p, q), "vm-pf-quadrant-ignores-sign-of-a"),
        (quads(sign, p, q, swap=True), "vm-pf-quadrant-parity-swapped"),
        (quads(sign, p, q, ignore_root=True), "vm-pf-quadrant-ignores-even-root-domain"),
        (right[:1], "vm-pf-quadrant-right-branch-only"),
    ], right


_QUAD_ASK = "State the quadrants in which the graph lies, as quadrant numbers from smallest to largest."


def _medium_quadrants(rng):
    sign = rng.choice((1, -1))
    kind = rng.choice(("odd", "even", "neither"))  # drawn first, so every quadrant set comes up about equally often
    p, q = rng.choice([i for i in _INDICES if parity(*i) == kind])
    a = sign * rng.choice((1, 2, 3, 4, 5))
    wrong, key = _quad_wrongs(sign, p, q)
    why = (
        "the graph is only defined for x > 0"
        if q % 2 == 0
        else ("it has a mirror image in the origin (odd)" if p % 2 else "it is symmetric about the y-axis (even)")
    )
    return single(
        f"Let y = {fx_text(a, p, q)}.",
        nums(
            _QUAD_ASK,
            key,
            2,
            [f"For x > 0 the sign of y is the sign of {a}; {why}."],
            wrong,
            choice=True,
        ),
    )


def _medium_both_asymptotes(rng):
    n = rng.choice((1, 2, 3, 4)) if rng.random() < 0.7 else None
    p, q = (-n, 1) if n else rng.choice([(-1, 2), (-1, 3), (-2, 3)])
    a = rng.choice((1, 2, 3, -1, -2, 4, -3))
    h = rng.choice((-4, -3, -2, -1, 1, 2, 3, 4, 5))
    k = rng.choice((-4, -3, -2, -1, 1, 2, 3, 5))
    rule = fx_text(a, p, q, h, k)
    wrong = [
        ([-h, k], "vm-pf-asymptote-sign-of-shift"),
        ([h, -k], "vm-pf-asymptote-sign-of-shift"),
        ([-h, -k], "vm-pf-asymptote-sign-of-shift"),
        ([k, h], "vm-pf-asymptote-axes-swapped"),
        ([0, 0], "vm-pf-asymptote-reads-zero"),
        ([h, a + k], "vm-pf-asymptote-takes-dilation"),
    ]
    return single(
        f"Let y = {rule}.",
        nums(
            "Give the x-value of the vertical asymptote x = h, then the y-value of the horizontal asymptote y = k, as h, k.",
            [h, k],
            2,
            [f"The asymptotes of y = a x^(-n) are x = 0 and y = 0; the shifts move them to x = {h} and y = {k}."],
            wrong,
        ),
    )


def _medium_reflection(rng):
    sign = rng.choice((1, -1))
    kind = rng.choice(("odd", "even", "neither"))
    p, q = rng.choice([i for i in _INDICES if parity(*i) == kind])
    a = sign * rng.choice((1, 2, 3))
    axis = rng.choice(("x-axis", "y-axis", "origin"))
    base = quads(sign, p, q)
    key = reflect(base, axis)
    other = [m for m in _REFLECT if m != axis]
    wrong = [
        (reflect(base, other[0]), "vm-pf-reflection-wrong-axis"),
        (reflect(base, other[1]), "vm-pf-reflection-wrong-axis"),
        (base, "vm-pf-reflection-forgotten"),
        (reflect(quads(-sign, p, q), axis), "vm-pf-quadrant-sign-of-a-reversed"),
        (quads(sign, p, q, swap=True), "vm-pf-quadrant-parity-swapped"),
    ]
    target = {
        "x-axis": "in the x-axis",
        "y-axis": "in the y-axis",
        "origin": "in both axes (a rotation of 180 degrees about the origin)",
    }[axis]
    return single(
        f"The graph of y = {fx_text(a, p, q)} is reflected {target}.",
        nums(
            "State the quadrants in which the image lies, as quadrant numbers from smallest to largest.",
            key,
            3,
            [f"The original lies in quadrants {num_text(base)}; the reflection sends them to {num_text(key)}."],
            wrong,
            choice=True,
        ),
    )


# -- hard ----------------------------------------------------------------------------------------------------------


def _meets(a: int, p: int, q: int, x: int) -> bool:
    """Whether a x^(p/q) = x at the integer x (x < 0 needs an odd root)."""
    if x == 0:
        return p > 0
    if x < 0 and q % 2 == 0:
        return False
    return a * real_power(x, p, q) == x


def _hard_meets_line(rng):
    for _ in attempts(200, "an index below 1 in lowest terms"):
        q = rng.choice((2, 3, 3, 4, 5))
        p = rng.randint(1, q - 1)
        if sympy.gcd(p, q) == 1:
            break
    w = rng.choice((2, 3)) if q <= 3 else 2
    root = w**q
    a = w ** (q - p)
    check(abs(root) <= 400, "root too large")
    key_points = [x for x in sorted({0, root, -root}) if _meets(a, p, q, x)]
    key = key_points[-1] - key_points[0]
    mirror = -root in key_points
    wrong = [
        (root, "vm-pf-intersection-misses-negative-root"),
        (2 * root, "vm-pf-intersection-symmetry-assumed"),
        (w ** (q - p), "vm-pf-intersection-index-not-undone"),
        (root + 1, "near-miss-value"),
        (1, "vm-pf-intersection-unit-assumed"),
        (root // 2, "near-miss-value"),
    ]
    rule = f"{a}{pw('x', p, q)}"
    return single(
        f"Let f(x) = {rule}.",
        nums(
            "The graph of y = f(x) meets the line y = x at several points. "
            "Find the distance along the x-axis between the leftmost and the rightmost of these points.",
            [key],
            4,
            [
                f"Check x = 0 and x = {root}: {a} × {root}^({index_text(p, q)}) = {a * real_power(root, p, q)}. By the symmetry of f the point at x = -{root} "
                f"{'also lies' if mirror else 'does not lie'} on the line, so the points run from {key_points[0]} to {key_points[-1]}."
            ],
            [([v], t) for v, t in wrong],
            lo=1,
        ),
    )


def _hard_image_value(rng):
    p, q = rng.choice(_INDICES)
    u = rng.choice((1, 2, 3)) if abs(p) <= 2 else rng.choice((1, 2))
    mult = rng.choice((1, 2, 3, 4))
    a = mult * u ** max(0, -p)
    axis = rng.choice(("x-axis", "y-axis"))
    big = u**q
    if axis == "y-axis":
        X = -big if q % 2 == 0 else rng.choice((-big, big))
        src, wrong_src = -X, X
    else:
        X = big if q % 2 == 0 else rng.choice((-big, big))
        src, wrong_src = X, -X
    f = lambda c: real_power(c, p, q) if (c > 0 or q % 2 == 1) else None  # noqa: E731
    fv = f(src)
    check(fv is not None, "x outside the domain")
    key = a * fv * (-1 if axis == "x-axis" else 1)
    check(sympy.sympify(key).is_integer, "non-integer key")
    wv = f(wrong_src)
    wrong = [
        (
            a * wv * (-1 if axis == "x-axis" else 1) if wv is not None else None,
            "vm-pf-reflection-forgotten" if axis == "y-axis" else "vm-pf-reflection-wrong-axis",
        ),
        (-key, "vm-pf-reflection-sign-lost"),
        (fv, "vm-pf-dilation-forgotten"),
        (f(src * a) * 1 if f(src * a) is not None and a != 1 else None, "vm-pf-dilation-applied-to-x"),
    ]
    wrong = [([int(v)], t) for v, t in wrong if v is not None and sympy.sympify(v).is_integer]
    where = "y-axis" if axis == "y-axis" else "x-axis"
    stem = phrase(
        rng,
        (
            "The graph of y = {f} is dilated by a factor of {a} from the x-axis and then reflected in the {ax}.",
            "Starting from y = {f}, the graph is stretched by a factor of {a} away from the x-axis and then reflected in the {ax}.",
        ),
        f=pw("x", p, q),
        a=a,
        ax=where,
    )
    if a == 1:
        stem = f"The graph of y = {pw('x', p, q)} is reflected in the {where}."
    working = [
        f"The image has rule y = {coef(-a) if axis == 'x-axis' else coef(a)}"
        f"{pw('x', p, q) if axis == 'x-axis' else pw('(-x)', p, q)}, so at x = {X} it is {int(key)}."
    ]
    return single(
        stem,
        nums(f"Find the y-coordinate of the point on the image with x-coordinate {X}.", [key], 4, working, wrong),
    )


def _hard_intercepts(rng):
    n = rng.choice((1, 2, 3, 4))
    h = rng.choice((-3, -2, -1, 0, 1, 2, 3))
    t0 = rng.choice((1, 2)) if n <= 3 else 1
    s = rng.choice((1, -1)) if n % 2 == 1 else 1
    k = rng.choice((-2, -1, 1, 2))
    a = -k * s * t0**n
    check(a != 0 and abs(a) <= 40, "dilation out of range")
    t_values = [s * t0] if n % 2 == 1 else [-t0, t0]
    key = h + max(t_values)
    wrong = [
        # the shift applied the wrong way, which for an even n is the same value as the other (left-most) root
        (h - max(t_values), "vm-pf-intercept-sign-of-shift" if n % 2 else "vm-pf-intercept-other-root"),
        (max(t_values), "vm-pf-intercept-shift-forgotten"),
        (h + 2 * max(t_values), "near-miss-value"),
        (h - min(t_values), "vm-pf-intercept-sign-of-shift" if n % 2 else "vm-pf-intercept-other-root"),
        (h + max(t_values) + 1, "near-miss-value"),
    ]
    rule = fx_text(a, -n, 1, h, k)
    return single(
        f"Let y = {rule}.",
        nums(
            "Find the x-coordinate of the right-most x-intercept.",
            [key],
            4,
            [
                f"Set y = 0: {coef(a)}{pw(shifted(h), -n, 1)} = {-k}, so {shifted(h)}^{n} = {s * t0**n}, giving {shifted(h).strip('()')} = "
                + (f"{t_values[0]}" if len(t_values) == 1 else "±" + str(t0))
                + f". The right-most intercept is x = {key}."
            ],
            [([v], t) for v, t in wrong],
            exclude=(h,),  # the line x = h is an asymptote, so no point of the graph has this x-coordinate
        ),
    )


_EASY = (_easy_symmetry, _easy_symmetry, _easy_mirror_value, _easy_asymptote)
_MEDIUM = (_medium_quadrants, _medium_both_asymptotes, _medium_reflection)
_HARD = (_hard_meets_line, _hard_image_value, _hard_intercepts)

_TIERS = {
    "easy": lambda rng: draw(rng.choice(_EASY), rng, 300),
    "medium": lambda rng: draw(rng.choice(_MEDIUM), rng, 300),
    "hard": lambda rng: draw(rng.choice(_HARD), rng, 300),
}


@generator(
    "vm_power_graph_features",
    level=17,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@uses_rng
def vm_power_graph_features(rng, tier: str = "easy"):
    """Odd or even, asymptotes, quadrants, reflections and intersections of power-function graphs."""
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    return _TIERS[tier](rng)


# -- exam chain ----------------------------------------------------------------------------------------------------


@distinct_part_answers
def _chain(rng):
    n = rng.choice((1, 2, 3, 4))
    u = rng.choice((3, 4, 6, 8, 10) if n == 1 else (3, 4, 5, 6) if n == 2 else (2, 3, 4))
    v = rng.choice((-5, -4, -3, -2, -1, 1, 2, 3, 4, 5))
    m = rng.choice((2, 3)) if n <= 3 else 2
    grow = m**n <= 16 and rng.random() < 0.5
    if grow:  # the translation moves the intercept out to u × m instead of in to u ÷ m
        v = rng.choice((-2, -1, 1, 2)) * m**n
    a = v * u**n
    k = -(v // m**n) if grow else -v * m**n
    if abs(k) > 140 or abs(a) > 3000:
        raise Redraw("values out of range")
    sign = 1 if a > 0 else -1
    stem = phrase(
        rng,
        (
            "The graph of y = a{f}, where a is a constant, passes through the point ({u}, {v}).",
            "A function has the rule y = a{f}, where a is a constant. Its graph passes through ({u}, {v}).",
            "The curve y = a{f} goes through the point ({u}, {v}), where a is a constant.",
        ),
        f=pw("x", -n, 1),
        u=u,
        v=v,
    )
    at_neg = v * (-1) ** n
    key_c = quads(sign, -n, 1)
    part_a = fraction_part(
        "Find the value of a.",
        a,
        2,
        [f"{v} = a × {u}^(-{n}), so a = {v} × {u}^{n} = {a}."],
        [
            (v, "vm-pf-a-forgets-power"),
            (Fraction(v, u**n), "vm-pf-a-divides"),
            (v * u, "vm-pf-a-index-ignored"),
            (-a, "vm-pf-a-sign-lost"),
            (v * u ** (n + 1), "vm-pf-a-extra-power"),
        ],
    )
    part_b = fraction_part(
        f"Use your value of a to find the y-coordinate of the point on the graph with x-coordinate -{u}.",
        at_neg,
        2,
        [f"y = {a} × (-{u})^(-{n}) = {at_neg}; the function is {'odd' if n % 2 else 'even'}, so f(-{u}) = {at_neg}."],
        [
            (-at_neg, "vm-pf-symmetry-sign-wrong"),
            (a * -u, "vm-pf-a-index-ignored"),
            (at_neg + 1, "near-miss-value"),
            (at_neg - 1, "near-miss-value"),
        ],
        lo=-8 * abs(at_neg),  # the point at -u is a mirror image of the given point (u, v), so no option is far bigger
        hi=8 * abs(at_neg),
        follow_through=f"part_a*(-{u})**(-{n})",
    )
    part_c = nums(
        _QUAD_ASK,
        key_c,
        2,
        [
            f"a is {'positive' if sign > 0 else 'negative'} and n = {n} is {'odd' if n % 2 else 'even'}: quadrants {num_text(key_c)}."
        ],
        [
            (quads(-sign, -n, 1), "vm-pf-quadrant-sign-of-a-reversed"),
            (quads(sign, -n, 1, swap=True), "vm-pf-quadrant-parity-swapped"),
            (quads(1, -n, 1), "vm-pf-quadrant-ignores-sign-of-a"),
            (quads(sign, -n, 1)[:1], "vm-pf-quadrant-right-branch-only"),
        ],
        choice=True,
    )
    root = Fraction(u * m) if grow else Fraction(u, m)
    direction = "up" if k > 0 else "down"
    part_d = fraction_part(
        f"The graph is translated {abs(k)} unit{'s' if abs(k) != 1 else ''} {direction} to give y = a{pw('x', -n, 1)} {'+' if k > 0 else '-'} {abs(k)}. "
        "Find its positive x-intercept.",
        root,
        3,
        [f"{a}x^(-{n}) = {-k}, so x^{n} = {Fraction(a, -k)} and x = {root}."],
        [
            (Fraction(u), "vm-pf-intercept-reads-given-point"),
            (Fraction(u, m) if grow else Fraction(u * m), "vm-pf-intercept-scaled-wrong-way"),
            (Fraction(a, -k), "vm-pf-root-not-taken"),
            (Fraction(m, u), "vm-pf-index-inverted"),
            (Fraction(u, m + 1), "near-miss-value"),
            (root + 1, "near-miss-value"),
        ],
        lo=0,
        follow_through=f"(part_a/({-k}))**(1/{n})",
    )
    return dict(text=stem, parts=[part_a, part_b, part_c, part_d])


@generator("vm_power_graph_features_exam", level=17, presets={"exam": dict(tier="exam")})
@uses_rng
def vm_power_graph_features_exam(rng, tier: str = "exam"):
    """y = a x^(-n) through a point: a, the value at -u, the quadrants, and a translated x-intercept."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return draw(_chain, rng)
