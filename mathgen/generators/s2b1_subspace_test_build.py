"""Generates `s2b1_subspace_test` questions (easy, medium, hard) and the exam chain `exd_x_s2b1_subspace`.

``uc2-subspace-test`` (``linear-algebra``, answer ``number``): decide which sets are subspaces of R^3, of the quadratic
polynomials P_2 or of the 2x2 matrices M_2, and find the parameter that turns an affine set into a subspace. Counting
the subspaces in a list (or finding a parameter) gives a typed answer that cannot be guessed from the wording, and the
exam chain pairs a four-way classification (which of the three subspace tests fails first) with typed parts.

* easy: how many of three sets in R^3 are subspaces (a plane through the origin, a plane not through it, a cone, a
  half-space, a union of two planes, vectors with integer entries, ...).
* medium: how many of four sets are subspaces, in P_2 or M_2(R) or R^3, with definitions that hide the linear structure
  (``p(1) p(2) = 0``, ``det A = 0``, ``AB = BA``, ``p divisible by (x - a)^2``, ``A^2 = A``).
* hard: a parameter. The larger k for which {p : p(a) = c(k)} (or the affine plane x + 2y - z = c(k), or the trace
  condition) is a subspace, where c is quadratic in k; the k for which {A : A(1, 1)^T = (k - r, k^2 - r^2)^T} is a
  subspace; or how many of four sets built from planes and lines (unions, intersections, sums, complements) are subspaces.
* exam (``exd_x_s2b1_subspace``): S_k = {x in R^3 : u . x = c(k), v . x = 0} (one equation for some questions). (a) which
  of the three subspace tests (zero vector, addition, scalar multiplication) fails first at a stated k that is not a
  root (a four-way multiple choice), (b) the larger k for which S_k is a subspace, (c) its dimension at that k,
  (d) the smaller k (follows through from (b): the sum of the roots is known).

Every set is a membership predicate on exact coordinates. Whether it is a subspace is decided twice: by the rule that
defines it (a homogeneous linear condition, a span, an intersection of subspaces) and by testing the zero vector and
closure under addition and scalar multiples on a grid of small vectors, which must find a counterexample exactly when
the set is not a subspace.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from typing import Callable

import sympy

from ..answers import Distractor
from ..core import attempts, check, draw_until, generator, join_terms, phrase, pick_distractors
from ._s2b1_linalg import plain_stem, int_distractors

LEVEL = 19

Vec = tuple


@dataclass(frozen=True)
class SetSpec:
    """A set of vectors: its wording, membership test and whether it is a subspace by the rule that defines it."""

    text: str
    member: Callable[[Vec], bool]
    subspace: bool


# -- classification by testing ---------------------------------------------------------------------------------------------

_SCALARS = (Fraction(-1), Fraction(2), Fraction(1, 2))


def _grid(dim: int):
    """Small integer vectors, nearest the origin first."""
    vectors = [tuple(Fraction(v) for v in vec) for vec in product(range(-2, 3), repeat=dim)]
    vectors.sort(key=lambda v: (sum(abs(c) for c in v), v))
    return vectors


def _coef(c):
    """Coefficient text for a term written as ``x = c y``: 1 and -1 drop the digit."""
    return {1: "", -1: "-"}.get(c, str(c))


_GRIDS = {3: _grid(3), 4: _grid(4)}


def classify(member, dim: int) -> tuple[bool, bool, bool]:
    """(contains zero, closed under addition, closed under scalar multiples) tested on the small-vector grid."""
    zero = tuple(Fraction(0) for _ in range(dim))
    members = [v for v in _GRIDS[dim] if member(v)][:45]
    has_zero = member(zero)
    add = all(member(tuple(a + b for a, b in zip(u, v))) for u in members for v in members)
    scalar = all(member(tuple(s * c for c in v)) for v in members for s in _SCALARS)
    return has_zero, add, scalar


def is_subspace(member, dim: int) -> bool:
    return all(classify(member, dim))


def _first_failure(member, dim: int) -> int:
    """0 if the zero vector is missing, 1 if addition fails first, 2 if scalar multiplication fails first, 3 if none."""
    has_zero, add, scalar = classify(member, dim)
    return 0 if not has_zero else 1 if not add else 2 if not scalar else 3


# -- the sets ----------------------------------------------------------------------------------------------------------------


def _lin(coeffs, names):
    return join_terms([(int(c), n) for c, n in zip(coeffs, names)])


def _r3_sets(rng):
    """Named builders of sets in R^3: each returns a `SetSpec`."""
    a, b, c = (rng.choice([-3, -2, -1, 1, 2, 3]) for _ in range(3))
    d = rng.choice([-4, -2, 2, 3, 5])
    s, t = rng.choice([-2, -1, 1, 2]), rng.choice([-3, -2, 2, 3])
    names = ("x", "y", "z")
    dot = lambda v: a * v[0] + b * v[1] + c * v[2]  # noqa: E731
    plane = f"{_lin((a, b, c), names)} = 0"
    return {
        "plane": SetSpec(f"{{(x, y, z) : {plane}}}", lambda v: dot(v) == 0, True),
        "line": SetSpec(
            f"{{(x, y, z) : x = {_coef(s)}y and z = {_coef(t)}x}}",
            lambda v: v[0] == s * v[1] and v[2] == t * v[0],
            True,
        ),
        "plane2": SetSpec(
            f"{{(x, y, z) : x + y = 0 and {_lin((a, 0, c), names)} = 0}}",
            lambda v: v[0] + v[1] == 0 and a * v[0] + c * v[2] == 0,
            True,
        ),
        "affine": SetSpec(f"{{(x, y, z) : {_lin((a, b, c), names)} = {d}}}", lambda v: dot(v) == d, False),
        "cone": SetSpec("{(x, y, z) : xz = y²}", lambda v: v[0] * v[2] == v[1] ** 2, False),
        "product": SetSpec("{(x, y, z) : xy = 0}", lambda v: v[0] * v[1] == 0, False),
        "union": SetSpec(f"{{(x, y, z) : x = {_coef(s)}y or z = 0}}", lambda v: v[0] == s * v[1] or v[2] == 0, False),
        "halfspace": SetSpec(f"{{(x, y, z) : {_lin((a, b, c), names)} ≥ 0}}", lambda v: dot(v) >= 0, False),
        "octant": SetSpec("{(x, y, z) : x ≥ 0 and y ≥ 0}", lambda v: v[0] >= 0 and v[1] >= 0, False),
        "integers": SetSpec(
            "{(x, y, z) : x, y and z are all integers}", lambda v: all(c.denominator == 1 for c in v), False
        ),
        "abs": SetSpec("{(x, y, z) : |x| = |z|}", lambda v: abs(v[0]) == abs(v[2]), False),
        "sphere": SetSpec("{(x, y, z) : x² + y² + z² = 1}", lambda v: sum(c * c for c in v) == 1, False),
    }


_R3_YES = ("plane", "line", "plane2")
_R3_NO_EASY = ("affine", "cone", "halfspace", "octant", "integers", "union", "sphere")
_R3_NO_HARD = ("cone", "product", "union", "abs", "octant", "integers", "halfspace")


def _p2_sets(rng):
    a, b = rng.sample([-2, -1, 0, 1, 2], 2)
    c = rng.choice([-3, -2, 2, 3])
    v = lambda p, t: p[0] + p[1] * t + p[2] * t * t  # noqa: E731
    dv = lambda p, t: p[1] + 2 * p[2] * t  # noqa: E731
    return {
        "root": SetSpec(f"{{p : p({a}) = 0}}", lambda p: v(p, a) == 0, True),
        "equal": SetSpec(f"{{p : p({a}) = p({b})}}", lambda p: v(p, a) == v(p, b), True),
        "even": SetSpec("{p : p(-x) = p(x) for all x}", lambda p: p[1] == 0, True),
        "double": SetSpec(
            f"{{p : p is divisible by (x {'-' if a > 0 else '+'} {abs(a)})²}}"
            if a != 0
            else "{p : p is divisible by x²}",
            lambda p: v(p, a) == 0 and dv(p, a) == 0,
            True,
        ),
        "mixed": SetSpec(f"{{p : p'({a}) + p({b}) = 0}}", lambda p: dv(p, a) + v(p, b) == 0, True),
        "value": SetSpec(f"{{p : p({a}) = {c}}}", lambda p: v(p, a) == c, False),
        "degree": SetSpec("{p : p has degree exactly 2}", lambda p: p[2] != 0, False),
        "product": SetSpec("{p : p(1) p(2) = 0}", lambda p: v(p, 1) * v(p, 2) == 0, False),
        "nonneg": SetSpec("{p : p(x) ≥ 0 for every real x}", lambda p: _nonnegative(p), False),
        "coeffs": SetSpec(
            "{p : every coefficient of p is a non-negative integer}",
            lambda p: all(x >= 0 and x.denominator == 1 for x in p),
            False,
        ),
        "sum": SetSpec("{p : p(1) = 1}", lambda p: v(p, 1) == 1, False),
        "int": SetSpec(
            "{p : every coefficient of p is an integer}", lambda p: all(x.denominator == 1 for x in p), False
        ),
    }


def _nonnegative(p) -> bool:
    c, b, a = p
    if a > 0:
        return b * b - 4 * a * c <= 0
    if a == 0:
        return b == 0 and c >= 0
    return False


_P2_YES = ("root", "equal", "even", "double", "mixed")
_P2_NO = ("value", "degree", "product", "nonneg", "coeffs", "sum", "int")


def _m2_sets(rng):
    bm = draw_until(
        rng,
        lambda r: [r.randint(-2, 2) for _ in range(4)],
        lambda m: not (m[1] == 0 and m[2] == 0 and m[0] == m[3]),  # a scalar matrix commutes with everything
        what="a matrix that is not a multiple of the identity",
    )
    w = rng.choice([(1, 1), (1, -1), (2, 1), (1, 2)])
    det = lambda m: m[0] * m[3] - m[1] * m[2]  # noqa: E731

    def commutes(m):
        a, b, c, d = m
        e, f, g, h = bm
        return (a * e + b * g, a * f + b * h, c * e + d * g, c * f + d * h) == (
            e * a + f * c,
            e * b + f * d,
            g * a + h * c,
            g * b + h * d,
        )

    def square(m):
        a, b, c, d = m
        return (a * a + b * c, a * b + b * d, c * a + d * c, c * b + d * d)

    return {
        "trace0": SetSpec("{A : tr(A) = 0}", lambda m: m[0] + m[3] == 0, True),
        "symmetric": SetSpec("{A : Aᵀ = A}", lambda m: m[1] == m[2], True),
        "upper": SetSpec("{A : A is upper triangular}", lambda m: m[2] == 0, True),
        "commute": SetSpec(f"{{A : AB = BA}}, where B = [[{bm[0]}, {bm[1]}], [{bm[2]}, {bm[3]}]]", commutes, True),
        "kills": SetSpec(
            f"{{A : A({w[0]}, {w[1]})ᵀ = (0, 0)ᵀ}}",
            lambda m: m[0] * w[0] + m[1] * w[1] == 0 and m[2] * w[0] + m[3] * w[1] == 0,
            True,
        ),
        "det0": SetSpec("{A : det A = 0}", lambda m: det(m) == 0, False),
        "invertible": SetSpec("{A : A is invertible}", lambda m: det(m) != 0, False),
        "nonneg": SetSpec("{A : every entry of A is at least 0}", lambda m: all(x >= 0 for x in m), False),
        "idempotent": SetSpec("{A : A² = A}", lambda m: square(m) == tuple(m), False),
        "trace1": SetSpec("{A : tr(A) = 1}", lambda m: m[0] + m[3] == 1, False),
        "fixed": SetSpec(
            f"{{A : A({w[0]}, {w[1]})ᵀ = ({w[0]}, {w[1]})ᵀ}}",
            lambda m: m[0] * w[0] + m[1] * w[1] == w[0] and m[2] * w[0] + m[3] * w[1] == w[1],
            False,
        ),
        "sym_or_skew": SetSpec(
            "{A : Aᵀ = A or Aᵀ = -A}", lambda m: m[1] == m[2] or (m[0] == 0 and m[3] == 0 and m[1] == -m[2]), False
        ),
    }


_M2_YES = ("trace0", "symmetric", "upper", "commute", "kills")
_M2_NO = ("det0", "invertible", "nonneg", "idempotent", "trace1", "fixed", "sym_or_skew")

_SPACES = {
    "R3": ("ℝ³", 3, _r3_sets, _R3_YES, _R3_NO_HARD),
    "P2": ("P₂, the space of polynomials of degree at most 2", 3, _p2_sets, _P2_YES, _P2_NO),
    "M2": ("M₂(ℝ), the space of 2×2 real matrices", 4, _m2_sets, _M2_YES, _M2_NO),
}


def _checked(spec: SetSpec, dim: int) -> SetSpec:
    """The rule-based verdict must agree with the tested one."""
    check(is_subspace(spec.member, dim) == spec.subspace, f"{spec.text}: the tests disagree with the defining rule")
    return spec


# -- count questions -------------------------------------------------------------------------------------------------------


def _count_distractors(specs, dim, count):
    flags = [classify(s.member, dim) for s in specs]
    total = len(specs)
    zero = sum(1 for f in flags if f[0])
    add = sum(1 for f in flags if f[1])
    scalar = sum(1 for f in flags if f[2])
    cands = [
        (zero, "subspace-zero-vector-only"),
        (add, "subspace-addition-only"),
        (scalar, "subspace-scalar-only"),
        (total - count, "subspace-count-complement"),
        (count + 1, "subspace-miscounted"),
        (count - 1, "subspace-miscounted"),
    ]
    cands = [(v, t) for v, t in cands if 0 <= v <= total]
    # Any other count in range is a plausible miscount, so every key keeps enough options.
    cands += [(v, "subspace-miscounted") for v in range(total + 1)]
    return pick_distractors(str(count), int_distractors(cands, count), 4)


_ROMAN = ("(i)", "(ii)", "(iii)", "(iv)", "(v)")


def _count_question(rng, space_key, n_sets, yes_pool, no_pool, tier_note):
    label, dim, builder, _yes, _no = _SPACES[space_key]
    for _ in attempts(100, "a list of sets with enough distractors"):
        library = builder(rng)
        n_yes = rng.randint(0, n_sets)
        chosen_yes = rng.sample(yes_pool, n_yes) if n_yes <= len(yes_pool) else None
        if chosen_yes is None:
            continue
        chosen_no = rng.sample(no_pool, n_sets - n_yes)
        keys = [*chosen_yes, *chosen_no]
        rng.shuffle(keys)
        specs = [_checked(library[k], dim) for k in keys]
        count = sum(1 for s in specs if s.subspace)
        distractors = _count_distractors(specs, dim, count)
        if len(distractors) >= 3:
            break
    listing = "; ".join(f"{_ROMAN[i]} {s.text}" for i, s in enumerate(specs))
    failures = {
        0: "does not contain the zero vector",
        1: "is not closed under addition",
        2: "is not closed under scalar multiplication",
    }
    working = [
        "Test each set for the zero vector, closure under addition and closure under scalar multiplication; a set is a subspace only if all three hold.",
        *[
            f"{_ROMAN[i]} {'is a subspace' if s.subspace else failures[_first_failure(s.member, dim)]}."
            for i, s in enumerate(specs)
        ],
        f"So {count} of the {n_sets} sets are subspaces.",
    ]
    stem = phrase(
        rng,
        (
            "Consider the following subsets of {label}: {listing}. How many of them are subspaces?",
            "Each of the sets {listing} is a subset of {label}. How many of the {n} sets are subspaces of {short}?",
        ),
        label=label,
        listing=listing,
        n=n_sets,
        short=label.split(",")[0],
    )
    return dict(
        text=stem,
        answer_kind="number",
        answer=count,
        answer_text=str(count),
        answer_latex=str(count),
        working=working,
        distractors=distractors,
    )


def _retry(rng, build):
    for _ in attempts(100, "a question with enough distractors"):
        result = build(rng)
        if result is not None:
            return result
    raise AssertionError  # pragma: no cover


def _easy(rng):
    return _count_question(rng, "R3", 3, _R3_YES, _R3_NO_EASY, "")


def _medium(rng):
    space = rng.choice(["P2", "M2", "R3"])
    _label, _dim, _builder, yes, no = _SPACES[space]
    return _count_question(rng, space, 4, yes, no, "")


# -- hard ------------------------------------------------------------------------------------------------------------------


def _quadratic_in_k(rng):
    """(r1, r2, text, value): c(k) = (k - r1)(k - r2) with distinct non-zero integer roots, written as a polynomial in k."""
    for _ in attempts(200, "two distinct non-zero integer roots"):
        r1, r2 = rng.sample([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5], 2)
        if r1 + r2 != 0:
            break
    text = join_terms([(1, "k²"), (-(r1 + r2), "k"), (r1 * r2, "")])
    return min(r1, r2), max(r1, r2), text, (lambda k: (k - r1) * (k - r2))


def _hard_quadratic(rng):
    space = rng.choice(["P2", "R3", "M2"])
    r1, r2, ctext, cfun = _quadratic_in_k(rng)
    a = rng.choice([-2, -1, 1, 2])
    if space == "P2":
        label = "the space P₂ of polynomials of degree at most 2"
        set_text = f"S_k = {{p : p({a}) = {ctext}}}"
        member = lambda k: lambda p: p[0] + p[1] * a + p[2] * a * a == cfun(k)  # noqa: E731
        dim = 3
    elif space == "R3":
        label = "ℝ³"
        coeffs = (rng.choice([1, 2, 3]), rng.choice([-2, -1, 1, 2]), rng.choice([-3, -1, 1, 2]))
        set_text = f"S_k = {{(x, y, z) : {_lin(coeffs, ('x', 'y', 'z'))} = {ctext}}}"
        member = lambda k: lambda v: sum(c * x for c, x in zip(coeffs, v)) == cfun(k)  # noqa: E731
        dim = 3
    else:
        label = "the space M₂(ℝ) of 2×2 real matrices"
        set_text = f"S_k = {{A : tr(A) = {ctext}}}"
        member = lambda k: lambda m: m[0] + m[3] == cfun(k)  # noqa: E731
        dim = 4
    # Both roots give subspaces, and a non-root does not (decided by testing, independent of the algebra above).
    check(is_subspace(member(r1), dim) and is_subspace(member(r2), dim), "a root of c(k) does not give a subspace")
    check(not is_subspace(member(r2 + 1), dim) and not is_subspace(member(r1 - 1), dim), "a non-root gives a subspace")
    check(
        sympy.solve(sympy.Symbol("k") ** 2 - (r1 + r2) * sympy.Symbol("k") + r1 * r2, sympy.Symbol("k"))
        in ([r1, r2], [r2, r1]),
        "SymPy's roots differ",
    )
    cands = [
        (r1, "subspace-other-root"),
        (-r2, "sign-error-moving-term"),
        (r1 + r2, "subspace-parameter-sum-for-root"),
        (r1 * r2, "subspace-parameter-product-for-root"),
        (-r1, "sign-error-moving-term"),
    ]
    distractors = pick_distractors(str(r2), int_distractors(cands, r2), 4)
    if len(distractors) < 3:
        return None
    stem = phrase(
        rng,
        (
            "For a real number k let {s} be a subset of {label}. S_k is a subspace for exactly two values of k. Find the larger one.",
            "In {label}, for each real k define {s}. Two values of k make S_k a subspace. What is the larger of them?",
        ),
        s=set_text,
        label=label,
    )
    working = [
        "S_k is the solution set of a linear equation with right-hand side c(k); it contains the zero vector, and is a subspace, exactly when c(k) = 0.",
        f"c(k) = {ctext} = (k - ({r1}))(k - ({r2})), so the values are {r1} and {r2} and the larger is {r2}.",
    ]
    return dict(
        text=stem,
        answer_kind="number",
        answer=r2,
        answer_text=str(r2),
        answer_latex=str(r2),
        working=working,
        distractors=distractors,
    )


def _hard_matrix_root(rng):
    r = rng.choice([-4, -3, -2, 2, 3, 4])
    w = rng.choice([(1, 1), (1, 2), (2, 1)])
    # {A : A w = (k - r, k² - r²)}: a subspace exactly when both components vanish, which happens only at k = r
    # (k² - r² = (k - r)(k + r), and k + r is 0 only at -r, where k - r = -2r != 0).
    member = lambda k: (  # noqa: E731
        lambda m: m[0] * w[0] + m[1] * w[1] == k - r and m[2] * w[0] + m[3] * w[1] == k * k - r * r
    )
    check(is_subspace(member(r), 4), "k = r does not give a subspace")
    check(not is_subspace(member(-r), 4) and not is_subspace(member(r + 1), 4), "another k gives a subspace")
    cands = [
        (-r, "subspace-other-root"),
        (r * r, "subspace-parameter-product-for-root"),
        (r + 1, "subspace-miscounted"),
        (-r * r, "sign-error-moving-term"),
        (0, "subspace-parameter-zero"),
    ]
    distractors = pick_distractors(str(r), int_distractors(cands, r), 4)
    if len(distractors) < 3:
        return None
    wt = f"({w[0]}, {w[1]})ᵀ"
    stem = phrase(
        rng,
        (
            "For a real number k let S_k = {{A : A{w} = (k {rsign}, k² - {r2})ᵀ}} be a subset of the space M₂(ℝ) of 2×2 real matrices. "
            "For which value of k is S_k a subspace?",
            "In the space M₂(ℝ) of 2×2 real matrices define S_k = {{A : A{w} = (k {rsign}, k² - {r2})ᵀ}} for each real k. Exactly one "
            "value of k makes S_k a subspace. Find it.",
        ),
        w=wt,
        rsign=f"- {r}" if r > 0 else f"+ {-r}",
        r2=str(r * r),
    )
    working = [
        "S_k is a subspace exactly when the right-hand side is the zero vector, so both k - r = 0 and k² - r² = 0 must hold.",
        f"The first gives k = {r}, and then k² - {r * r} = 0 as well, so the only value is k = {r}.",
    ]
    return dict(
        text=stem,
        answer_kind="number",
        answer=r,
        answer_text=str(r),
        answer_latex=str(r),
        working=working,
        distractors=distractors,
    )


def _plane_pair(rng):
    """Two distinct planes U, W through the origin in R^3 (normals n1, n2) and a line L = span(v) inside U."""
    for _ in attempts(500, "two planes and a line inside the first"):
        n1 = [rng.choice([-2, -1, 0, 1, 2]) for _ in range(3)]
        n2 = [rng.choice([-2, -1, 0, 1, 2]) for _ in range(3)]
        v = [rng.choice([-2, -1, 0, 1, 2]) for _ in range(3)]
        if (
            any(n1)
            and any(n2)
            and any(v)
            and sympy.Matrix([n1, n2]).rank() == 2
            and sum(a * b for a, b in zip(n1, v)) == 0
        ):
            if sum(a * b for a, b in zip(n2, v)) != 0:
                return n1, n2, v
    raise AssertionError  # pragma: no cover


def _hard_conceptual(rng):
    n1, n2, v = _plane_pair(rng)
    names = ("x", "y", "z")
    in_u = lambda p: sum(a * b for a, b in zip(n1, p)) == 0  # noqa: E731
    in_w = lambda p: sum(a * b for a, b in zip(n2, p)) == 0  # noqa: E731
    on_line = lambda p: all(v[i] * p[j] == v[j] * p[i] for i in range(3) for j in range(i + 1, 3))  # noqa: E731
    u_text = f"U = {{(x, y, z) : {_lin(n1, names)} = 0}}"
    w_text = f"W = {{(x, y, z) : {_lin(n2, names)} = 0}}"
    l_text = f"L = span{{({', '.join(str(c) for c in v)})}}"
    pool = [
        (f"U ∪ W", lambda p: in_u(p) or in_w(p), False),
        (f"U ∩ W", lambda p: in_u(p) and in_w(p), True),
        ("U ∪ L", lambda p: in_u(p) or on_line(p), True),
        ("W ∪ L", lambda p: in_w(p) or on_line(p), False),
        ("the vectors of ℝ³ not in U", lambda p: not in_u(p), False),
        ("the vectors of ℝ³ not in U, together with 0", lambda p: not in_u(p) or all(c == 0 for c in p), False),
        ("W ∩ L", lambda p: in_w(p) and on_line(p), True),
        ("U ∩ (W ∪ L)", lambda p: in_u(p) and (in_w(p) or on_line(p)), False),
    ]
    for _ in attempts(100, "four conceptual sets with enough distractors"):
        chosen = rng.sample(pool, 4)
        specs = [SetSpec(t, m, ok) for t, m, ok in chosen]
        count = sum(1 for s in specs if s.subspace)
        if count in (0, 4) and rng.random() < 0.7:
            continue
        # The grid may be too coarse to expose a failure for an unlucky direction; never emit such a set.
        if any(is_subspace(s.member, 3) != s.subspace for s in specs):
            continue
        distractors = _count_distractors(specs, 3, count)
        if len(distractors) >= 3:
            break
    listing = "; ".join(f"{_ROMAN[i]} {s.text}" for i, s in enumerate(specs))
    stem = phrase(
        rng,
        (
            "In ℝ³ let {u}, {w} and {l}. Which of the following sets are subspaces of ℝ³? {listing}. Give the number of them that are.",
            "Let {u}, {w} and {l} be subsets of ℝ³. How many of the following sets are subspaces: {listing}?",
        ),
        u=u_text,
        w=w_text,
        l=l_text,
        listing=listing,
    )
    working = [
        "L lies inside U (the normal of U is perpendicular to the generator of L) but not inside W.",
        "An intersection of subspaces is a subspace, and a union of two subspaces is a subspace only when one contains the other.",
        f"So {count} of the four sets are subspaces.",
    ]
    return dict(
        text=stem,
        answer_kind="number",
        answer=count,
        answer_text=str(count),
        answer_latex=str(count),
        working=working,
        distractors=distractors,
    )


@generator(
    "s2b1_subspace_test",
    level=LEVEL,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@plain_stem
def s2b1_subspace_test(rng, tier="easy"):
    """Which sets are subspaces, and for which parameter; the tiers are described in the module docstring."""
    if tier == "easy":
        return _retry(rng, _easy)
    if tier == "medium":
        return _retry(rng, _medium)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _retry(rng, rng.choice([_hard_quadratic, _hard_quadratic, _hard_matrix_root, _hard_conceptual]))


# -- exam ----------------------------------------------------------------------------------------------------------------------

_TESTS = (
    "The zero vector is not in S_k",
    "S_k contains the zero vector but is not closed under addition",
    "S_k contains the zero vector and is closed under addition, but is not closed under scalar multiplication",
    "None of the tests fails: S_k is a subspace",
)


def _exam_try(rng):
    r1, r2, ctext, cfun = _quadratic_in_k(rng)
    n1 = [rng.choice([1, 2, 3]), rng.choice([-2, -1, 1, 2]), rng.choice([-3, -1, 1, 2])]
    two = rng.random() < 0.7
    n2 = [rng.choice([-2, -1, 1, 2]), rng.choice([-2, -1, 1, 2]), rng.choice([-2, -1, 1, 2])]
    if two and sympy.Matrix([n1, n2]).rank() != 2:
        return None
    names = ("x", "y", "z")

    def member(k):
        if two:
            return lambda v: sum(a * b for a, b in zip(n1, v)) == cfun(k) and sum(a * b for a, b in zip(n2, v)) == 0
        return lambda v: sum(a * b for a, b in zip(n1, v)) == cfun(k)

    check(is_subspace(member(r1), 3) and is_subspace(member(r2), 3), "a root of c(k) does not give a subspace")
    m = rng.choice([v for v in range(-3, 5) if cfun(v) != 0])
    check(_first_failure(member(m), 3) == 0, "a non-root should fail the zero-vector test")
    dim = 1 if two else 2
    # The dimension at a root, by the null space of the homogeneous system.
    rows = [n1, n2] if two else [n1]
    check(3 - sympy.Matrix(rows).rank() == dim, "the dimension at the root differs")

    options = list(_TESTS)
    answer = options[0]
    a_dist = [
        Distractor(options[1], "subspace-zero-vector-overlooked"),
        Distractor(options[2], "subspace-zero-vector-overlooked"),
        Distractor(options[3], "subspace-equation-means-subspace"),
    ]
    sum_roots = r1 + r2
    b_dist = pick_distractors(
        str(r2),
        int_distractors(
            [
                (r1, "subspace-other-root"),
                (-r2, "sign-error-moving-term"),
                (sum_roots, "subspace-parameter-sum-for-root"),
                (r1 * r2, "subspace-parameter-product-for-root"),
            ],
            r2,
        ),
        3,
    )
    c_dist = pick_distractors(
        str(dim),
        int_distractors(
            [
                (3, "subspace-dimension-ambient"),
                (dim + 1, "subspace-dimension-affine-for-linear"),
                (dim - 1, "dimension-counts-equations"),
                (0, "kernel-assumed-trivial"),
            ],
            dim,
        ),
        3,
    )
    d_dist = pick_distractors(
        str(r1),
        int_distractors(
            [
                (r2, "subspace-other-root"),
                (-r1, "sign-error-moving-term"),
                (sum_roots, "subspace-parameter-sum-for-root"),
                (r1 * r2, "subspace-parameter-product-for-root"),
            ],
            r1,
        ),
        3,
    )
    if not all(len(x) >= 2 for x in (b_dist, c_dist, d_dist)):
        return None
    eqs = f"{_lin(n1, names)} = {ctext}" + (f" and {_lin(n2, names)} = 0" if two else "")
    stem = f"For a real number k let S_k be the set of vectors (x, y, z) in ℝ³ satisfying {eqs}."
    return dict(
        text=stem,
        parts=[
            dict(
                prompt=(
                    f"Take k = {m}. Test S_k in the order: zero vector, closure under addition, closure under scalar multiplication. "
                    "Which statement is correct?"
                ),
                answer_kind="mcq",
                answer=answer,
                answer_text=answer,
                marks=2,
                working=[
                    f"At k = {m} the right-hand side {ctext} is {cfun(m)}, not 0, so (0, 0, 0) does not satisfy the first equation."
                ],
                distractors=a_dist,
            ),
            dict(
                prompt="Find the larger value of k for which S_k is a subspace of ℝ³.",
                answer_kind="number",
                answer=r2,
                answer_text=str(r2),
                marks=3,
                working=[
                    f"S_k contains 0 exactly when {ctext} = 0, that is (k - ({r1}))(k - ({r2})) = 0; the larger root is {r2}."
                ],
                distractors=b_dist,
            ),
            dict(
                prompt="For that value of k, find the dimension of S_k.",
                answer_kind="number",
                answer=dim,
                answer_text=str(dim),
                marks=2,
                working=[
                    f"S_k is the solution set of {len(rows)} independent homogeneous equation(s) in 3 unknowns, so its dimension is 3 - {len(rows)} = {dim}."
                ],
                distractors=c_dist,
            ),
            dict(
                prompt="Use your answer to (b), and the fact that the two values of k add up to "
                + str(sum_roots)
                + ", to find the smaller value of k for which S_k is a subspace.",
                answer_kind="number",
                answer=r1,
                answer_text=str(r1),
                marks=2,
                follow_through=f"{sum_roots} - part_b",
                working=[f"The roots add up to {sum_roots}, so the other root is {sum_roots} - ({r2}) = {r1}."],
                distractors=d_dist,
            ),
        ],
    )


@generator("exd_x_s2b1_subspace", level=LEVEL, presets={"exam": dict(tier="exam")})
@plain_stem
def exd_x_s2b1_subspace(rng, tier="exam"):
    """Exam chain: which subspace test fails, the parameters that make an affine set a subspace, and its dimension."""
    check(tier == "exam", f"unknown tier {tier!r}")
    return _retry(rng, _exam_try)
