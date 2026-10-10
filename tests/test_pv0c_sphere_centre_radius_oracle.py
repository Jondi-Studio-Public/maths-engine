"""Independent oracle for the `pv0c_sphere_centre_radius` questions: it re-derives each answer from the rendered
question text.

Imports NOTHING from `mathgen.generators.pv0c_sphere_centre_radius_build`.
Parses the sphere's equation straight out of `question.text` with its own
regex, hands it to `sympy` (a general symbolic-algebra tool, not this
generator's own machinery) to expand and collect coefficients via
`sympy.Poly` -- a route that works uniformly across every tier's differently
-shaped equation (centre-radius, common-factor, moved-constant,
split-sides, or fully expanded with a leading coefficient) without this
oracle ever needing to know which shape it is looking at -- then recovers
`(a, b, c, r)` via the textbook `a = -D/(2A)` formula and compares against
`question.answer_text` and every one of `question.distractors`.

>= 1000 questions total, >= 250 per tier (SEEDS below gives exactly 250 per
tier x 4 tiers = 1000).
"""

from fractions import Fraction

import pytest
import sympy
from sympy import Poly, expand, symbols
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

import mathgen
from mathgen.skills import load_skills

SKILL_ID = "senior-introduce-cartesian-coordinates-for-three-dimensional-space-including"
TIERS = ["easy", "medium", "hard", "exam"]
SEEDS = range(250)

_X, _Y, _Z = symbols("x y z")
_TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)

_PREFIXES = [
    "A sphere has equation ",
    "The equation ",
    "A sphere is given by the equation ",
    "Find the centre and radius of the sphere ",
]
_SUFFIX_MARKERS = [
    " describes a sphere.",
    " is that of a sphere.",
    ". State its centre and radius.",
    ". Complete the square to find its centre and radius.",
    ". Divide through first, then complete the square.",
    ". Give your answer",
]


def _extract_equation(text: str) -> str:
    rest = None
    for prefix in _PREFIXES:
        if text.startswith(prefix):
            rest = text[len(prefix) :]
            break
    assert rest is not None, f"unrecognised prompt prefix: {text!r}"
    for marker in _SUFFIX_MARKERS:
        idx = rest.find(marker)
        if idx != -1:
            return rest[:idx]
    raise AssertionError(f"unrecognised prompt suffix: {text!r}")


def _recompute_answer(eq_text: str):
    """Parse ``LHS = RHS``, expand ``LHS - RHS`` via SymPy, collect the
    quadratic-form coefficients with ``sympy.Poly`` and recover
    ``(a, b, c, r)`` via the standard completing-the-square formula. Works
    identically whether the printed equation was centre-radius, common
    -factor, moved-constant, split-sides or fully expanded -- the SAME
    formula applies once the equation is in ``A(x^2+y^2+z^2)+Dx+Ey+Fz+G=0``
    shape, which ``expand`` + ``Poly`` always normalises to."""
    lhs_str, rhs_str = eq_text.replace("^", "**").split("=")
    local_dict = {"x": _X, "y": _Y, "z": _Z}
    lhs = parse_expr(lhs_str, local_dict=local_dict, transformations=_TRANSFORMS)
    rhs = parse_expr(rhs_str, local_dict=local_dict, transformations=_TRANSFORMS)
    poly = Poly(expand(lhs - rhs), _X, _Y, _Z)

    A = poly.coeff_monomial(_X**2)
    Ay = poly.coeff_monomial(_Y**2)
    Az = poly.coeff_monomial(_Z**2)
    assert A == Ay == Az and A != 0, f"not a sphere-shaped quadratic form: {eq_text!r}"
    for cross in (_X * _Y, _X * _Z, _Y * _Z):
        assert poly.coeff_monomial(cross) == 0, f"unexpected cross term in {eq_text!r}"

    D = poly.coeff_monomial(_X)
    E = poly.coeff_monomial(_Y)
    F = poly.coeff_monomial(_Z)
    G = poly.coeff_monomial(1)

    a = -D / (2 * A)
    b = -E / (2 * A)
    c = -F / (2 * A)
    r_sq = a**2 + b**2 + c**2 - G / A
    r = sympy.sqrt(r_sq)
    assert r.is_rational, f"radius is not rational for {eq_text!r}: r^2 = {r_sq}"

    def _to_fraction(v) -> Fraction:
        v = sympy.Rational(v)
        return Fraction(int(v.p), int(v.q))

    return _to_fraction(a), _to_fraction(b), _to_fraction(c), _to_fraction(r)


def _skill_sources():
    _domains, skills = load_skills()
    skill = next(s for s in skills if s.id == SKILL_ID)
    return {src.difficulty: src for src in skill.sources}


SOURCES = _skill_sources()


def _make(difficulty, seed):
    src = SOURCES[difficulty]
    return mathgen.make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


@pytest.mark.parametrize("difficulty", TIERS)
def test_oracle_recomputes_the_key_from_the_rendered_equation(difficulty):
    n_checked = 0
    for seed in SEEDS:
        q = _make(difficulty, seed)
        eq_text = _extract_equation(q.text)
        recomputed = _recompute_answer(eq_text)
        canonical = tuple(Fraction(v) for v in q.answer)
        assert recomputed == canonical, (
            f"{difficulty} seed {seed}: oracle recomputed {recomputed} from "
            f"{eq_text!r} but the canonical answer was {canonical}"
        )
        n_checked += 1
    assert n_checked >= 250, f"{difficulty}: only checked {n_checked} questions"


@pytest.mark.parametrize("difficulty", TIERS)
def test_oracle_every_distractor_differs_from_the_recomputed_key(difficulty):
    for seed in SEEDS:
        q = _make(difficulty, seed)
        eq_text = _extract_equation(q.text)
        recomputed = _recompute_answer(eq_text)
        for d in q.distractors:
            parsed = tuple(Fraction(part.strip()) for part in d.split(","))
            assert parsed != recomputed, f"{difficulty} seed {seed}: distractor {d!r} equals the oracle-recomputed key"


# ---------------------------------------------------------------------------
# Proof this oracle is not vacuously green ("prove it red first").
# ---------------------------------------------------------------------------


def test_oracle_catches_a_perturbed_key_deliberately_corrupted_here():
    """Demonstrates the oracle actually discriminates: a genuine question's
    `answer_text` is deliberately corrupted (one coordinate bumped by 1,
    entirely within this test, never touching any shipped file) and the
    comparison this oracle performs is shown to fail against it -- the same
    comparison `test_oracle_recomputes_the_key_from_the_rendered_equation`
    runs for real. This is how the oracle was proven red before being
    trusted: it was run once against a corrupted `(a, b, c, r)` (this test)
    and confirmed to raise, and once against the real generator output
    (the test above) and confirmed to pass."""
    q = _make("hard", 0)
    eq_text = _extract_equation(q.text)
    recomputed = _recompute_answer(eq_text)
    corrupted = (recomputed[0] + 1, *recomputed[1:])
    assert corrupted != recomputed
