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
from .r4e_common import fraction_part, single
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


def one_word(prompt, word, marks, working, wrong):
    return words_part(prompt, [word], marks, working, [([w], t) for w, t in wrong])


def nums(prompt, values, marks, working, wrong):
    return words_part(
        prompt,
        [int(v) for v in values],
        marks,
        working,
        [([int(x) for x in vs], t) for vs, t in wrong if vs is not None],
    )


def num_text(values) -> str:
    return ", ".join(str(v) for v in values)


# -- easy ----------------------------------------------------------------------------------------------------------


def _easy_symmetry(rng):
    p, q = rng.choice(_INDICES)
    a = rng.choice((1, 1, 2, 3, -1, -2, 4))
    key = parity(p, q)
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
    stem = f"Let f(x) = {fx_text(a, p, q)}."
    return single(
        stem,
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
        (0, "vm-pf-symmetry-zero"),
        (key * 2, "vm-pf-symmetry-doubled"),
        (key - 1, "near-miss-value"),
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
            (a, "vm-pf-asymptote-takes-dilation"),
        ]
    working = ["Translating by (h, k) moves the asymptotes from x = 0 and y = 0 to x = h and y = k."]
    return single(f"Let y = {rule}.", nums(prompt, [key], 2, working, [([w], t) for w, t in wrong]))


# -- medium --------------------------------------------------------------------------------------------------------


def _quad_wrongs(sign, p, q):
    right = quads(sign, p, q)
    return [
        (quads(-sign, p, q), "vm-pf-quadrant-sign-of-a-reversed"),
        (quads(1, p, q), "vm-pf-quadrant-ignores-sign-of-a"),
        (quads(sign, p, q, swap=True), "vm-pf-quadrant-parity-swapped"),
        (quads(sign, p, q, ignore_root=True), "vm-pf-quadrant-ignores-even-root-domain"),
        ([1, 2, 3, 4], "vm-pf-quadrant-all-four"),
    ], right


def _medium_quadrants(rng):
    p, q = rng.choice(_INDICES)
    a = rng.choice((1, 2, 3, 4, -1, -2, -3, 5))
    sign = 1 if a > 0 else -1
    wrong, key = _quad_wrongs(sign, p, q)
    why = (
        "the graph is only defined for x > 0"
        if q % 2 == 0
        else ("it has a mirror image in the origin (odd)" if p % 2 else "it is symmetric about the y-axis (even)")
    )
    return single(
        f"Let y = {fx_text(a, p, q)}.",
        nums(
            "In which quadrants does the graph lie? List the quadrant numbers from smallest to largest.",
            key,
            2,
            [f"For x > 0 the sign of y is the sign of {a}; {why}."],
            wrong,
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
        ([k, h], "vm-pf-asymptote-axes-swapped"),
        ([0, 0], "vm-pf-asymptote-reads-zero"),
        ([h, a + k], "vm-pf-asymptote-takes-dilation"),
    ]
    return single(
        f"Let y = {rule}.",
        nums(
            "Give the x-value of the vertical asymptote, then the y-value of the horizontal asymptote.",
            [h, k],
            2,
            [f"The asymptotes of y = a x^(-n) are x = 0 and y = 0; the shifts move them to x = {h} and y = {k}."],
            wrong,
        ),
    )


def _medium_reflection(rng):
    p, q = rng.choice(_INDICES)
    a = rng.choice((1, 2, 3, -1, -2))
    sign = 1 if a > 0 else -1
    axis = rng.choice(("x-axis", "y-axis", "origin"))
    base = quads(sign, p, q)
    key = reflect(base, axis)
    other = [m for m in _REFLECT if m != axis]
    wrong = [
        (reflect(base, other[0]), "vm-pf-reflection-wrong-axis"),
        (reflect(base, other[1]), "vm-pf-reflection-wrong-axis"),
        (base, "vm-pf-reflection-forgotten"),
        (reflect(quads(-sign, p, q), axis), "vm-pf-quadrant-sign-of-a-reversed"),
    ]
    target = {
        "x-axis": "in the x-axis",
        "y-axis": "in the y-axis",
        "origin": "in both axes (a rotation of 180 degrees about the origin)",
    }[axis]
    return single(
        f"The graph of y = {fx_text(a, p, q)} is reflected {target}.",
        nums(
            "In which quadrants does the image lie? List the quadrant numbers from smallest to largest.",
            key,
            3,
            [f"The original lies in quadrants {num_text(base)}; the reflection sends them to {num_text(key)}."],
            wrong,
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
    sign = rng.choice((1, 1, -1))
    a = sign * w ** (q - p)
    check(abs(root) <= 400, "root too large")
    candidates = sorted({0, root, -root})
    key = [x for x in candidates if _meets(a, p, q, x)]
    check(0 in key and len(key) >= 2 or len(key) >= 1, "no intersection")
    pos_only = [x for x in key if x > 0]
    wrong = [
        ([x for x in key if x != 0] or [root], "vm-pf-intersection-misses-origin"),
        (sorted({0, root}), "vm-pf-intersection-misses-negative-root"),
        (sorted({0, root, -root}), "vm-pf-intersection-symmetry-assumed"),
        ([0, w ** (q - p)], "vm-pf-intersection-index-not-undone"),
        ([0, 1], "vm-pf-intersection-unit-assumed"),
    ]
    if not pos_only and 0 in key and len(key) == 1:
        wrong.append(([0, root], "vm-pf-intersection-symmetry-assumed"))
    rule = f"{coef(a)}{pw('x', p, q)}"
    return single(
        f"Let f(x) = {rule}.",
        nums(
            "Find the x-coordinates of all the points where the graph of y = f(x) meets the line y = x, from smallest to largest.",
            key,
            4,
            [
                f"Check x = 0 and x = {root}: {a} × {root}^({index_text(p, q)}) = {a * real_power(root, p, q)}. By the symmetry of f the point at x = -{root} "
                f"{'also lies' if -root in key else 'does not lie'} on the line."
            ],
            wrong,
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
        f"The image has rule y = {coef(a) if a != 1 else ''}{'-' if axis == 'x-axis' else ''}"
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
    key = sorted(h + t for t in t_values)
    wrong = [
        ([h - t for t in t_values], "vm-pf-intercept-sign-of-shift"),
        ([t for t in t_values], "vm-pf-intercept-shift-forgotten"),
        (
            [h + t0] if len(key) == 2 else [h + t0, h - t0],
            "vm-pf-intercept-misses-negative-root" if len(key) == 2 else "vm-pf-intercept-spurious-root",
        ),
        ([h + s * t0 * 2], "near-miss-value"),
    ]
    wrong = [(sorted(v), t) for v, t in wrong]
    rule = fx_text(a, -n, 1, h, k)
    return single(
        f"Let y = {rule}.",
        nums(
            "Find the x-intercepts, from smallest to largest.",
            key,
            4,
            [
                f"Set y = 0: {coef(a)}{pw(shifted(h), -n, 1)} = {-k}, so {shifted(h)}^{n} = {s * t0**n}, giving {shifted(h).strip('()')} = "
                + (f"{t_values[0]}" if len(t_values) == 1 else "±" + str(t0))
                + "."
            ],
            wrong,
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
def vm_power_graph_features(rng, tier: str = "easy"):
    """Odd or even, asymptotes, quadrants, reflections and intersections of power-function graphs."""
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    return _TIERS[tier](rng)


# -- exam chain ----------------------------------------------------------------------------------------------------


def _chain(rng):
    n = rng.choice((1, 2, 3, 4))
    u = rng.choice((2, 3, 4))
    v = rng.choice((-5, -4, -3, -2, -1, 1, 2, 3, 4, 5))
    m = rng.choice((2, 3)) if n <= 3 else 2
    a = v * u**n
    k = -v * m**n
    check(abs(k) <= 140 and abs(a) <= 1500, "values out of range")
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
    part_a = fraction_part(
        "Find the value of a.",
        a,
        2,
        [f"{v} = a × {u}^(-{n}), so a = {v} × {u}^{n} = {a}."],
        [
            (v, "vm-pf-a-forgets-power"),
            (Fraction(v, u**n), "vm-pf-a-divides"),
            (v * u, "vm-pf-a-index-ignored"),
            (-a, "near-miss-value"),
        ],
    )
    part_b = fraction_part(
        f"Use your value of a to find the y-coordinate of the point on the graph with x-coordinate -{u}.",
        at_neg,
        2,
        [
            f"y = {a} × (-{u})^(-{n}) = {at_neg}; the function is {'odd' if n % 2 else 'even'}, so this is {'-' if n % 2 else ''}{v}."
        ],
        [
            (-at_neg, "vm-pf-symmetry-sign-wrong"),
            (a * -u, "vm-pf-a-index-ignored"),
            (a * (-u) ** n, "vm-pf-a-index-ignored"),
            (0, "vm-pf-symmetry-zero"),
        ],
        follow_through=f"part_a*(-{u})**(-{n})",
    )
    key_c = quads(sign, -n, 1)
    part_c = nums(
        "In which quadrants does the graph lie? List the quadrant numbers from smallest to largest.",
        key_c,
        2,
        [f"a is {'positive' if sign > 0 else 'negative'} and n = {n} is {'odd' if n % 2 else 'even'}."],
        [
            (quads(-sign, -n, 1), "vm-pf-quadrant-sign-of-a-reversed"),
            (quads(sign, -n, 1, swap=True), "vm-pf-quadrant-parity-swapped"),
            (quads(1, -n, 1), "vm-pf-quadrant-ignores-sign-of-a"),
        ],
    )
    root = Fraction(u, m)
    direction = "up" if k > 0 else "down"
    part_d = fraction_part(
        f"The graph is translated {abs(k)} units {direction} to give y = a{pw('x', -n, 1)} {'+' if k > 0 else '-'} {abs(k)}. "
        "Find its positive x-intercept.",
        root,
        3,
        [f"{a}x^(-{n}) = {-k}, so x^{n} = {Fraction(a, -k)} and x = {root}."],
        [
            (Fraction(u), "vm-pf-intercept-shift-forgotten"),
            (Fraction(u * m), "vm-pf-intercept-sign-of-shift"),
            (Fraction(-a, k) if n == 1 else Fraction(a, -k), "vm-pf-root-not-taken"),
            (Fraction(m, u), "vm-pf-index-inverted"),
        ],
        follow_through=f"(part_a/({-k}))**(1/{n})",
    )
    return dict(text=stem, parts=[part_a, part_b, part_c, part_d])


@generator("vm_power_graph_features_exam", level=17, presets={"exam": dict(tier="exam")})
def vm_power_graph_features_exam(rng, tier: str = "exam"):
    """y = a x^(-n) through a point: a, the value at -u, the quadrants, and a translated x-intercept."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return draw(_chain, rng)
