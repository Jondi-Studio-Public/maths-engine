"""Generates `fy1la_span` questions (easy, medium, hard) and the exam chain `exd_x_fy1la_span`.

``ucv1-span-membership`` (``linear-algebra``, answer ``matrix``): a vector v is in span{u, w} exactly when
v = a u + b w has a solution; the answer is the pair of coefficients (a, b), which is unique because u and w are
independent.

* easy: u and w in ℝ³ with a 1 and a 0 in two shared coordinates, so a and b can be read from those coordinates and
  checked against the third.
* medium: u and w in ℝ³ with no such shortcut: a and b come from solving two equations by elimination and checking
  the third.
* hard: u and w in ℝ⁴, so there are four equations in two unknowns, and the learner has to choose two equations to
  solve and use the others only as a check.
* exam (``exd_x_fy1la_span``): u and w in ℝ³ and v with an unknown k as one coordinate. (a) det[u w v] as an
  expression in k, (b) the k for which v is in the span, (c) the coefficients (a, b) at that k, (d) how many pairs
  (a, b) work at a different k.

Each v is built as a u + b w (k replaces one coordinate in the exam), so the answer is known; it is re-derived by
Gauss-Jordan elimination in exact fractions and by SymPy's `linsolve`, and the other values of k are checked to leave v
outside the span.
"""

from __future__ import annotations

import sympy

from ..answers import Distractor
from ..core import VerificationError, attempts, check, generator, join_terms, phrase
from ._fy1la_kit import (
    column_value,
    consistent,
    det,
    expression_distractors,
    expression_part,
    fm,
    matrix_part,
    mcq_part,
    num_distractors,
    number_part,
    rref,
    vec_str,
    vector_distractors,
    with_columns,
)

_NOTE = "Give your answer as (a, b)."


def _combination(u, w, a, b):
    return [a * x + b * y for x, y in zip(u, w)]


def _solve(u, w, v):
    """(a, b) by elimination, or None when v is outside the span (or u, w are dependent)."""
    aug = with_columns([u, w, v])
    reduced, pivots = rref(aug)
    if pivots != [0, 1]:
        return None
    return reduced[0][2], reduced[1][2]


def _verify(u, w, v, a, b):
    got = _solve(u, w, v)
    check(got == (a, b), "elimination disagrees with the coefficients")
    x1, x2 = sympy.symbols("a b")
    solved = sympy.linsolve(
        [x1 * u[i] + x2 * w[i] - v[i] for i in range(len(v))],
        [x1, x2],
    )
    check(solved == sympy.FiniteSet((a, b)), "SymPy's coefficients differ")


def _wrong_candidates(v, a, b):
    cands = [
        ([b, a], "coefficients-order-swapped"),
        ([-a, -b], "span-coefficient-sign-slip"),
        ([a, -b], "span-coefficient-sign-slip"),
        ([-a, b], "span-coefficient-sign-slip"),
        ([v[0], v[1]], "vector-entries-read-as-coefficients"),
    ]
    return cands


def _two_good_pairs(u, w) -> bool:
    """At least two coordinate pairs whose 2×2 determinant is not zero, so no choice of two equations is a trap."""
    n = len(u)
    good = sum(1 for i in range(n) for j in range(i + 1, n) if u[i] * w[j] - u[j] * w[i] != 0)
    return good >= 2


def _draw(rng, tier):
    nz = lambda lo, hi: rng.choice([x for x in range(lo, hi + 1) if x != 0])  # noqa: E731
    for _ in attempts(2000, "a span question"):
        if tier == "easy":
            i, j = rng.sample(range(3), 2)
            i, j = min(i, j), max(i, j)
            k = 3 - i - j
            u, w = [0, 0, 0], [0, 0, 0]
            u[i], w[j] = 1, 1
            u[k], w[k] = nz(-3, 3), nz(-3, 3)
            a, b = nz(-4, 4), nz(-4, 4)
        elif tier == "medium":
            u = [rng.randint(-3, 4) for _ in range(3)]
            w = [rng.randint(-3, 4) for _ in range(3)]
            if sum(1 for x in u + w if x == 0) > 1:
                continue
            a, b = nz(-4, 4), nz(-4, 4)
        else:
            u = [rng.randint(-3, 4) for _ in range(4)]
            w = [rng.randint(-3, 4) for _ in range(4)]
            if sum(1 for x in u + w if x == 0) > 2:
                continue
            a, b = nz(-3, 4), nz(-3, 4)
        v = _combination(u, w, a, b)
        if tier != "easy" and not _two_good_pairs(u, w):
            continue
        if max(abs(x) for x in v) > 14 or all(x == 0 for x in v) or _solve(u, w, v) is None:
            continue
        return u, w, v, a, b
    raise VerificationError("unreachable")  # pragma: no cover - attempts() raises first


def _single(rng, tier):
    for _ in attempts(300, "a span question with distractors"):
        u, w, v, a, b = _draw(rng, tier)
        _verify(u, w, v, a, b)
        dist = vector_distractors(_wrong_candidates(v, a, b), [a, b])
        if dist is None:
            continue
        text = vec_str([a, b])
        n = len(u)
        stems = (
            "Let u = {u}, w = {w} and v = {v} in ℝ{sup}. The vector v is a linear combination of u and w. Find the scalars a and b "
            "such that v = a·u + b·w. " + _NOTE,
            "Write v = {v} as a linear combination a·u + b·w of u = {u} and w = {w}. " + _NOTE,
        )
        return dict(
            text=phrase(rng, stems, u=vec_str(u), w=vec_str(w), v=vec_str(v), sup="³" if n == 3 else "⁴"),
            answer_kind="matrix",
            answer=column_value([a, b]),
            answer_text=text,
            answer_latex="plain:" + text,
            working=[
                "The equations a·u + b·w = v, one per coordinate, are solved by elimination on two of them.",
                f"They give a = {a} and b = {b}; the remaining coordinate{'s' if n == 4 else ''} agree"
                f"{'' if n == 4 else 's'}, so v is in the span.",
            ],
            distractors=dist,
        )
    raise VerificationError("unreachable")  # pragma: no cover


@generator(
    "fy1la_span",
    level=16,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def fy1la_span(rng, tier="easy"):
    """Coefficients of a vector in the span of two others; the tiers are described in the module docstring."""
    check(tier in ("easy", "medium", "hard"), f"unknown tier {tier!r}")
    return _single(rng, tier)


# -- exam -------------------------------------------------------------------------------------------------------


@generator("exd_x_fy1la_span", level=16, presets={"exam": dict(tier="exam")})
def exd_x_fy1la_span(rng, tier="exam"):
    """Exam chain: det[u w v] in k, the k that puts v in the span, the coefficients, and a k that does not."""
    check(tier == "exam", f"unknown tier {tier!r}")
    k = sympy.Symbol("k")
    for _ in attempts(1000, "an exam span chain"):
        u = [rng.randint(-3, 4) for _ in range(3)]
        w = [rng.randint(-3, 4) for _ in range(3)]
        if sum(1 for x in u + w if x == 0) > 1:
            continue
        a, b = rng.choice([x for x in range(-3, 4) if x]), rng.choice([x for x in range(-3, 4) if x])
        v0 = _combination(u, w, a, b)
        t = rng.randrange(3)
        k0 = v0[t]
        if k0 == 0 or abs(k0) > 12 or max(abs(x) for x in v0) > 14 or _solve(u, w, v0) is None:
            continue
        shape = [("k" if i == t else v0[i]) for i in range(3)]

        def build(kv, shape=shape):
            return [kv if x == "k" else x for x in shape]

        d0, d1 = det(fm(with_columns([u, w, build(0)]))), det(fm(with_columns([u, w, build(1)])))
        C, D = int(d1 - d0), int(d0)
        if C == 0 or D != -C * k0:
            continue
        mat = sympy.Matrix(3, 3, lambda r, c: [u, w, [k if x == "k" else x for x in shape]][c][r])
        check(sympy.expand(mat.det() - (C * k + D)) == 0, "SymPy's determinant differs")
        check(sympy.solve(mat.det(), k) == [k0], "SymPy's root differs")
        _verify(u, w, v0, a, b)
        k1 = k0 + rng.choice([-3, -2, -1, 1, 2, 3])
        check(
            _solve(u, w, build(k1)) is None and not consistent(with_columns([u, w]), build(k1)),
            "v is in the span at k1",
        )
        det_text = join_terms([(C, "k"), (D, "")])
        dist_a = expression_distractors(
            str(C * k + D),
            [
                (join_terms([(C, "k"), (-D, "")]), "det-expansion-sign-slip"),
                (join_terms([(-C, "k"), (D, "")]), "det-expansion-sign-slip"),
                (join_terms([(C, "k")]), "dropped-constant-term-det"),
                (join_terms([(-C, "k"), (-D, "")]), "det-expansion-sign-slip"),
            ],
            "Find det[u w v] in terms of k.",
        )
        dist_b = num_distractors(
            k0,
            [
                (-k0, "sign-error-moving-term"),
                (-D, "forgot-to-divide-by-coefficient"),
                (D, "forgot-to-divide-by-coefficient"),
            ]
            + ([(C // D, "linear-solution-inverted")] if D and C % D == 0 else []),
        )
        wrong_a_b = [
            ([b, a], "coefficients-order-swapped"),
            ([-a, -b], "span-coefficient-sign-slip"),
            ([a, -b], "span-coefficient-sign-slip"),
            ([-a, b], "span-coefficient-sign-slip"),
        ]
        dist_c = vector_distractors(wrong_a_b, [a, b])
        if None in (dist_a, dist_b, dist_c):
            continue
        v_text = "(" + ", ".join("k" if x == "k" else str(x) for x in shape) + ")"
        options = ("No pair (a, b)", "Exactly one pair (a, b)", "Infinitely many pairs (a, b)")
        return dict(
            text=f"Let u = {vec_str(u)}, w = {vec_str(w)} and v = {v_text} in ℝ³, where k is a constant.",
            parts=[
                expression_part(
                    "Find det[u w v] in terms of k, where u, w and v are the columns of a 3×3 matrix.",
                    C * k + D,
                    det_text,
                    dist_a,
                    marks=2,
                    working=["Expand the determinant along the column containing k."],
                ),
                number_part(
                    "Find the value of k for which v lies in the span of u and w.",
                    k0,
                    dist_b,
                    marks=1,
                    working=[
                        f"v is in the span exactly when the three vectors are dependent: {det_text} = 0, so k = {k0}."
                    ],
                ),
                matrix_part(
                    f"For this value of k, find the scalars a and b with v = a·u + b·w. {_NOTE}",
                    [a, b],
                    dist_c,
                    marks=3,
                    vector=True,
                    working=[f"With k = {k0}, v = {vec_str(v0)}; elimination gives a = {a} and b = {b}."],
                ),
                mcq_part(
                    f"How many pairs (a, b) satisfy v = a·u + b·w when k = {k1}?",
                    options[0],
                    [
                        Distractor(options[1], "span-membership-assumed"),
                        Distractor(options[2], "span-dependence-means-infinite"),
                    ],
                    marks=2,
                    working=[f"{k1} is not {k0}, so det[u w v] is not zero and v is outside the span: no pair works."],
                ),
            ],
        )
    raise VerificationError("unreachable")  # pragma: no cover
