"""Generates sphere questions (ACMSM103): find the centre and radius from the equation of a sphere in 3-D
Cartesian coordinates.

``senior-introduce-cartesian-coordinates-for-three-dimensional-space-including`` (ACMSM103), scope:
given the equation of a sphere in centre-radius form, or in expanded form that needs completing the square,
state its centre coordinates and radius as the ordered 4-tuple ``[a, b, c, r]``.

Single topic, ``pv0c_sphere_centre_radius``, ``answer: ordered_list``. Every
value in the graded answer is a plain Python ``int`` or ``fractions.Fraction``
(never a sympy type, never a float) so ``mathgen.grading.grade_ordered_list``'s
exact positional ``Fraction`` comparison is what actually decides the grade
-- matching the ``ucv1-gram-schmidt-2d`` precedent
(``mathgen/generators/p12f_geometry_calculus_build.py``).

Tiers (construct-to-target: draw ``(a, b, c, r)`` first, uniformly, THEN
build the printed equation from it -- never the reverse):

- ``easy`` -- centre-radius form printed directly, mixed signs, a zero
  coordinate rendered as a bare ``x^2``/``y^2``/``z^2`` on some draws.
  Decision: flip signs, take the square root of the printed constant.
- ``medium`` -- centre-radius form that needs normalising first: a common
  factor on every square, the constant moved to the left, or one square
  moved to the other side. Decision: rewrite to standard form before
  reading off the centre and radius.
- ``hard`` -- expanded form ``x^2 + y^2 + z^2 + Dx + Ey + Fz + G = 0``,
  ``a, b, c, r`` integers (so ``D, E, F`` are automatically even). Decision:
  complete the square in each variable.
- ``exam`` -- expanded form multiplied through by a leading coefficient
  ``k`` (``k in {2, 3}`` with an all-integer centre/radius, or ``k = 4``
  paired with a half-integer centre/radius so every displayed coefficient
  stays an integer). Decision: divide through by ``k`` first, then complete
  the square (sometimes with fractions).

Plausibility: ``|a|, |b|,
|c| <= 9``, ``1 <= r <= 12``, ``r^2 > 0`` always (r is drawn >= 1, so this is
automatic), and on hard/exam ``a^2 + b^2 + c^2 != r^2`` (``G != 0``) so the
constant term is never a trivially-recognisable ``-r^2``.

Independent check: for every draw, ``check_equivalent`` compares the
constructed centre-radius expression (built purely from ``a, b, c, r``)
against the ``D, E, F, G``-based expanded expression this module derives
FROM ``a, b, c, r`` by the standard completing-the-square formula
(``D = -2a``, etc) -- catching an arithmetic slip in either route.
"""

from __future__ import annotations

from fractions import Fraction
from math import isqrt

from ..core import (
    check,
    check_distinct,
    check_pairwise_distinct,
    generator,
    join_terms,
    pick_distractors,
    rational_text,
)

_MAX_ATTEMPTS = 500
_ANSWER_FORMAT = (
    "Give your answer as four numbers a, b, c, r separated by commas, "
    "where (a, b, c) is the centre and r is the radius."
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _sq_term(var: str, coord: Fraction, *, bare_zero: bool = False) -> str:
    """``(x - a)^2``-shaped term for one coordinate, e.g. ``_sq_term("x", 2)``
    -> ``"(x - 2)"`` (the caller appends ``^2``), ``_sq_term("x", -3)`` ->
    ``"(x + 3)"``. ``coord == 0`` renders as a bare ``x`` (so the caller's
    ``^2`` suffix gives the bare-square form the easy tier sometimes uses)
    unless ``bare_zero`` is False, in which case it renders ``(x - 0)``."""
    if coord == 0:
        return var if bare_zero else f"({var} - 0)"
    if coord > 0:
        return f"({var} - {rational_text(coord)})"
    return f"({var} + {rational_text(-coord)})"


def _defg(a: Fraction, b: Fraction, c: Fraction, r: Fraction):
    """The standard completing-the-square coefficients for
    ``x^2+y^2+z^2+Dx+Ey+Fz+G=0`` derived from ``(a, b, c, r)``:
    ``D=-2a, E=-2b, F=-2c, G=a^2+b^2+c^2-r^2``."""
    D, E, F = -2 * a, -2 * b, -2 * c
    G = a * a + b * b + c * c - r * r
    return D, E, F, G


def _to_int(value: Fraction, msg: str) -> int:
    check(value.denominator == 1, msg)
    return int(value)


#: Sample points used by `_verify_expansion` -- five points is enough to pin
#: down a degree-2 polynomial identity in x, y, z with a comfortable margin
#: over the 1 (constant) + 3 (linear) + 3 (pure-square) = 7 coefficients
#: actually varying here (the cross terms are always absent on both sides,
#: by construction, so 5 points independent enough to vary x, y, z and their
#: squares is sufficient to catch a sign/formula slip in D, E, F or G).
_SAMPLE_POINTS = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1))


def _verify_expansion(a, b, c, r, D, E, F, G) -> None:
    """Independent route, deliberately NOT a symbolic re-expansion (that
    would cost ~8 ms/call via SymPy and blow the generator's 15 ms budget --
    "slow cross-checks belong in tests, not in make()"): evaluate both
    ``(x-a)^2+(y-b)^2+(z-c)^2-r^2`` and ``x^2+y^2+z^2+Dx+Ey+Fz+G`` at several
    concrete integer points using plain ``Fraction`` arithmetic and require
    them to agree everywhere. `D, E, F, G` are computed by the `_defg`
    formula from `a, b, c, r`; this check evaluates the ORIGINAL
    centre-radius expression at each point independently of that formula, so
    a sign slip in `_defg` (e.g. `D = -a` instead of `-2a`) shows up as a
    mismatch at a point where it matters, exactly like a symbolic expansion
    check would, at a fraction of the cost."""
    for x, y, z in _SAMPLE_POINTS:
        lhs = (x - a) ** 2 + (y - b) ** 2 + (z - c) ** 2 - r**2
        rhs = x**2 + y**2 + z**2 + D * x + E * y + F * z + G
        check(lhs == rhs, f"sphere_centre_radius: expansion mismatch at ({x}, {y}, {z})")


def _exact_sqrt(value: Fraction) -> Fraction | None:
    """Exact rational square root of a positive ``Fraction``, or ``None`` if
    it is not one -- both numerator and denominator must be perfect
    squares."""
    if value <= 0:
        return None
    num, den = value.numerator, value.denominator
    rn, rd = isqrt(num), isqrt(den)
    if rn * rn == num and rd * rd == den:
        return Fraction(rn, rd)
    return None


def _fmt_answer(values) -> str:
    return ", ".join(rational_text(v) for v in values)


def _latex_num(v: Fraction) -> str:
    if v.denominator == 1:
        return str(v.numerator)
    sign = "-" if v < 0 else ""
    return rf"{sign}\frac{{{abs(v.numerator)}}}{{{v.denominator}}}"


def _latex_answer(values) -> str:
    return ", ".join(_latex_num(v) for v in values)


def _draw_plain(rng, *, allow_zero_coord: bool = True):
    """(a, b, c, r), all integers, |a|,|b|,|c| <= 9, 1 <= r <= 12. At most
    one of a, b, c is zero (keeps the printed equation from carrying more
    than one bare-square term at once); not all three share the same sign
    when at least two are nonzero (the "mixed signs" flavour the easy tier
    names)."""
    for _attempt in range(_MAX_ATTEMPTS):
        coords = [rng.randint(-9, 9) for _ in range(3)]
        zero_count = sum(1 for v in coords if v == 0)
        if zero_count > 1 or (zero_count > 0 and not allow_zero_coord):
            continue
        nonzero = [v for v in coords if v != 0]
        if len(nonzero) >= 2 and len({1 if v > 0 else -1 for v in nonzero}) == 1:
            continue  # all-same-sign nonzero coordinates -- redraw for mixed signs
        r = rng.randint(1, 12)
        a, b, c = (Fraction(v) for v in coords)
        return a, b, c, Fraction(r)
    raise RuntimeError("sphere_centre_radius: could not draw a plain (a, b, c, r)")


def _draw_half(rng):
    """(a, b, c, r) each independently an integer or a true half-integer
    (denominator 2), |a|,|b|,|c| <= 9, 1 <= r <= 12, with at least one of
    the four genuinely carrying denominator 2 (else this is just the plain
    hard-tier case in disguise)."""
    for _attempt in range(_MAX_ATTEMPTS):
        na, nb, nc = (rng.randint(-18, 18) for _ in range(3))
        nr = rng.randint(2, 24)
        a, b, c, r = (Fraction(na, 2), Fraction(nb, 2), Fraction(nc, 2), Fraction(nr, 2))
        if not any(v.denominator == 2 for v in (a, b, c, r)):
            continue
        return a, b, c, r
    raise RuntimeError("sphere_centre_radius: could not draw a half-integer (a, b, c, r)")


def _guard_hard(a, b, c, r):
    _D, _E, _F, G = _defg(a, b, c, r)
    return G != 0


# ---------------------------------------------------------------------------
# Distractors
# ---------------------------------------------------------------------------


def _base_distractors(a, b, c, r):
    """centre-sign-flipped and r-squared-for-r -- available on every tier."""
    candidates = []
    flipped = (-a, -b, -c, r)
    candidates.append(("centre-sign-flipped", flipped))
    r_sq = r * r
    if r_sq != r:
        candidates.append(("r-squared-for-r", (a, b, c, r_sq)))
    return candidates


def _expanded_distractors(a, b, c, r, D, E, F, G, k: Fraction | None = None):
    """coefficient-not-halved and constant-sign-error (hard/exam), plus
    not-divided-by-k when a leading coefficient k != 1 was used (exam)."""
    candidates = []
    not_halved = (-D, -E, -F, r)  # = (2a, 2b, 2c, r): read D, E, F straight
    # as the centre without ever halving them.
    candidates.append(("coefficient-not-halved", not_halved))

    wrong_sq = a * a + b * b + c * c + G  # sign error: +G instead of -G
    r_wrong = _exact_sqrt(wrong_sq)
    if r_wrong is not None:
        candidates.append(("constant-sign-error", (a, b, c, r_wrong)))

    if k is not None and k != 1:
        candidates.append(("not-divided-by-k", (a * k, b * k, c * k, r * k)))
    return candidates


def _pick(answer, labelled_candidates):
    """Filter labelled (name, tuple) candidates to genuinely distinct
    rational 4-tuples (unconditionally checked, ``check_distinct``/
    ``check_pairwise_distinct``), then format and hand to
    ``pick_distractors``."""
    offered = [answer]
    kept = []
    for _label, cand in labelled_candidates:
        if cand == answer or cand in offered:
            continue
        check_distinct(answer, cand, msg="a sphere distractor collided with the answer")
        check_pairwise_distinct(*offered, cand, msg="a sphere distractor collided with another option")
        offered.append(cand)
        kept.append(cand)
    answer_text = _fmt_answer(answer)
    return pick_distractors(answer_text, [_fmt_answer(cand) for cand in kept])


# ---------------------------------------------------------------------------
# easy
# ---------------------------------------------------------------------------

_EASY_PATTERNS = [
    "A sphere has equation {eq}. {fmt}",
    # Deliberately "is that of", never "describes a sphere" here: the
    # equation's right side can be a bare 1 (r=1), and mathgen.text_quality's
    # singular_count_plural check flags "1 describes" as a count/plural
    # mismatch ("describes" ends in "s" and is not in its verb allowlist).
    # "is" IS in that allowlist, so this wording is safe at every r.
    "The equation {eq} is that of a sphere. {fmt}",
    "A sphere is given by the equation {eq}. State its centre and radius. {fmt}",
]


def _pv0c_sphere_easy(rng):
    for _attempt in range(_MAX_ATTEMPTS):
        a, b, c, r = _draw_plain(rng)
        bare = {v: rng.random() < 0.6 for v in (a, b, c) if v == 0}
        tx = _sq_term("x", a, bare_zero=bare.get(a, False))
        ty = _sq_term("y", b, bare_zero=bare.get(b, False))
        tz = _sq_term("z", c, bare_zero=bare.get(c, False))
        r2 = r * r
        D, E, F, G = _defg(a, b, c, r)
        _verify_expansion(a, b, c, r, D, E, F, G)

        eq = f"{tx}^2 + {ty}^2 + {tz}^2 = {rational_text(r2)}"
        pattern = rng.choice(_EASY_PATTERNS)
        text = pattern.format(eq=eq, fmt=_ANSWER_FORMAT)

        answer = (a, b, c, r)
        answer_text = _fmt_answer(answer)
        distractors = _pick(answer, _base_distractors(a, b, c, r))
        if not distractors:
            continue
        working = [
            f"Flip the signs shown to read the centre: ({rational_text(a)}, {rational_text(b)}, {rational_text(c)}).",
            f"The right side is r^2 = {rational_text(r2)}, so r = {rational_text(r)}.",
        ]
        return dict(
            text=text,
            answer=list(answer),
            answer_text=answer_text,
            answer_latex=_latex_answer(answer),
            working=working,
            distractors=distractors,
        )
    raise RuntimeError("sphere_centre_radius easy: could not draw a usable question")


# ---------------------------------------------------------------------------
# medium
# ---------------------------------------------------------------------------

_MEDIUM_VARIANT_TEMPLATES = {
    "scaled": "The equation {eq} describes a sphere. {fmt}",
    "moved": "The equation {eq} describes a sphere. {fmt}",
    "split": "The equation {eq} describes a sphere. {fmt}",
}


def _pv0c_sphere_medium(rng):
    for _attempt in range(_MAX_ATTEMPTS):
        a, b, c, r = _draw_plain(rng, allow_zero_coord=False)
        tx = _sq_term("x", a)
        ty = _sq_term("y", b)
        tz = _sq_term("z", c)
        r2 = r * r
        D, E, F, G = _defg(a, b, c, r)
        _verify_expansion(a, b, c, r, D, E, F, G)

        variant = rng.choice(("scaled", "moved", "split"))
        if variant == "scaled":
            k = rng.choice((2, 3))
            eq = f"{k}{tx}^2 + {k}{ty}^2 + {k}{tz}^2 = {rational_text(k * r2)}"
            hint = f"Every square carries a common factor of {k} -- divide both sides by {k} first."
        elif variant == "moved":
            eq = f"{tx}^2 + {ty}^2 + {tz}^2 - {rational_text(r2)} = 0"
            hint = f"Move the constant back to the right: {tx}^2 + {ty}^2 + {tz}^2 = {rational_text(r2)}."
        else:
            eq = f"{tx}^2 + {ty}^2 = {rational_text(r2)} - {tz}^2"
            hint = f"Move every square to one side: {tx}^2 + {ty}^2 + {tz}^2 = {rational_text(r2)}."

        pattern = _MEDIUM_VARIANT_TEMPLATES[variant]
        text = pattern.format(eq=eq, fmt=_ANSWER_FORMAT)

        answer = (a, b, c, r)
        answer_text = _fmt_answer(answer)
        distractors = _pick(answer, _base_distractors(a, b, c, r))
        if not distractors:
            continue
        working = [
            hint,
            f"Centre ({rational_text(a)}, {rational_text(b)}, {rational_text(c)}), radius {rational_text(r)}.",
        ]
        return dict(
            text=text,
            answer=list(answer),
            answer_text=answer_text,
            answer_latex=_latex_answer(answer),
            working=working,
            distractors=distractors,
        )
    raise RuntimeError("sphere_centre_radius medium: could not draw a usable question")


# ---------------------------------------------------------------------------
# hard
# ---------------------------------------------------------------------------

_HARD_PATTERNS = [
    "A sphere has equation {eq}. {fmt}",
    "The equation {eq} describes a sphere. Complete the square to find its centre and radius. {fmt}",
    "Find the centre and radius of the sphere {eq}. {fmt}",
]


def _expanded_text(D, E, F, G, *, k: Fraction = Fraction(1)) -> str:
    kD = _to_int(k * D, "sphere_centre_radius: kD is not an integer")
    kE = _to_int(k * E, "sphere_centre_radius: kE is not an integer")
    kF = _to_int(k * F, "sphere_centre_radius: kF is not an integer")
    kG = _to_int(k * G, "sphere_centre_radius: kG is not an integer")
    kk = _to_int(k, "sphere_centre_radius: k is not an integer")
    return (
        join_terms(
            [
                (kk, "x^2"),
                (kk, "y^2"),
                (kk, "z^2"),
                (kD, "x"),
                (kE, "y"),
                (kF, "z"),
                (kG, ""),
            ]
        )
        + " = 0"
    )


def _pv0c_sphere_hard(rng):
    for _attempt in range(_MAX_ATTEMPTS):
        a, b, c, r = _draw_plain(rng)
        if not _guard_hard(a, b, c, r):
            continue
        D, E, F, G = _defg(a, b, c, r)
        _verify_expansion(a, b, c, r, D, E, F, G)

        eq = _expanded_text(D, E, F, G)
        pattern = rng.choice(_HARD_PATTERNS)
        text = pattern.format(eq=eq, fmt=_ANSWER_FORMAT)

        answer = (a, b, c, r)
        answer_text = _fmt_answer(answer)
        candidates = _base_distractors(a, b, c, r) + _expanded_distractors(a, b, c, r, D, E, F, G)
        distractors = _pick(answer, candidates)
        if not distractors:
            continue
        working = [
            f"Group and complete the square in each variable: D={rational_text(D)}, "
            f"E={rational_text(E)}, F={rational_text(F)}, G={rational_text(G)}.",
            f"Centre (-D/2, -E/2, -F/2) = ({rational_text(a)}, {rational_text(b)}, {rational_text(c)}); "
            f"r^2 = a^2+b^2+c^2-G = {rational_text(r * r)}, so r = {rational_text(r)}.",
        ]
        return dict(
            text=text,
            answer=list(answer),
            answer_text=answer_text,
            answer_latex=_latex_answer(answer),
            working=working,
            distractors=distractors,
        )
    raise RuntimeError("sphere_centre_radius hard: could not draw a usable question")


# ---------------------------------------------------------------------------
# exam
# ---------------------------------------------------------------------------

_EXAM_PATTERNS = [
    "A sphere has equation {eq}. {fmt}",
    "The equation {eq} describes a sphere. Divide through first, then complete the square. {fmt}",
    "Find the centre and radius of the sphere {eq}. {fmt}",
]


def _pv0c_sphere_exam_scaled(rng):
    a, b, c, r = _draw_plain(rng)
    if not _guard_hard(a, b, c, r):
        return None
    D, E, F, G = _defg(a, b, c, r)
    _verify_expansion(a, b, c, r, D, E, F, G)
    k = rng.choice((2, 3))
    eq = _expanded_text(D, E, F, G, k=Fraction(k))
    answer = (a, b, c, r)
    candidates = _base_distractors(a, b, c, r) + _expanded_distractors(a, b, c, r, D, E, F, G, k=Fraction(k))
    hint = f"Divide every term by {k} first, then complete the square as usual."
    return eq, answer, candidates, hint


def _pv0c_sphere_exam_half(rng):
    a, b, c, r = _draw_half(rng)
    if not _guard_hard(a, b, c, r):
        return None
    D, E, F, G = _defg(a, b, c, r)
    _verify_expansion(a, b, c, r, D, E, F, G)
    k = Fraction(4)
    eq = _expanded_text(D, E, F, G, k=k)
    answer = (a, b, c, r)
    candidates = _base_distractors(a, b, c, r) + _expanded_distractors(a, b, c, r, D, E, F, G, k=k)
    hint = "Divide every term by 4 first, then complete the square -- the centre may land on a half-integer."
    return eq, answer, candidates, hint


def _pv0c_sphere_exam(rng):
    for _attempt in range(_MAX_ATTEMPTS):
        variant = rng.choice(("scaled", "half"))
        built = _pv0c_sphere_exam_scaled(rng) if variant == "scaled" else _pv0c_sphere_exam_half(rng)
        if built is None:
            continue
        eq, answer, candidates, hint = built
        pattern = rng.choice(_EXAM_PATTERNS)
        text = pattern.format(eq=eq, fmt=_ANSWER_FORMAT)
        answer_text = _fmt_answer(answer)
        distractors = _pick(answer, candidates)
        if not distractors:
            continue
        a, b, c, r = answer
        working = [
            hint,
            f"Centre ({rational_text(a)}, {rational_text(b)}, {rational_text(c)}), radius {rational_text(r)}.",
        ]
        return dict(
            text=text,
            answer=list(answer),
            answer_text=answer_text,
            answer_latex=_latex_answer(answer),
            working=working,
            distractors=distractors,
        )
    raise RuntimeError("sphere_centre_radius exam: could not draw a usable question")


@generator(
    "pv0c_sphere_centre_radius",
    level=12,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def pv0c_sphere_centre_radius(rng, tier="easy"):
    if tier == "easy":
        return _pv0c_sphere_easy(rng)
    if tier == "medium":
        return _pv0c_sphere_medium(rng)
    if tier == "hard":
        return _pv0c_sphere_hard(rng)
    if tier == "exam":
        return _pv0c_sphere_exam(rng)
    raise ValueError(f"unknown tier {tier!r}")
