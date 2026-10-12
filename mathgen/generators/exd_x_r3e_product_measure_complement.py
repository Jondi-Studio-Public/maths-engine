"""Exam-tier source for `ucv1-product-measure-rectangle`: rectangles and their complement in a product of two finite
measure spaces.

X = {x1, x2, x3} and Y = {y1, y2, y3} carry point masses (positive rationals, so neither space has measure 1).
A is a non-empty proper subset of X and B of Y. The chain:

(a) mu(A)                                          fraction, 1 mark
(b) nu(B)                                          fraction, 1 mark
(c) (mu x nu)(A x B)                               fraction, 2 marks   part_a * part_b
(d) the measure of the points of X x Y outside A x B   fraction, 2 marks   mu(X) nu(Y) - part_c

(d) is the step that separates the rectangle from its complement: the complement is not the rectangle of the
complements. Every answer is an exact fraction; the key is re-derived from the point masses by summing over the
points of the product, not by the product rule it is testing.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import product

from ..answers import Distractor
from ..core import check, generator, pick_distractors, rational_text
from ._exd_r3_distinct import distinct_part_answers

_MASSES = sorted({Fraction(n, d) for d in (2, 3, 4, 6) for n in range(1, d + 2)})


def _fraction_part(prompt, value: Fraction, marks, working, distractors, follow_through=None):
    part = dict(
        prompt=prompt,
        answer_kind="fraction",
        answer={"p": value.numerator, "q": value.denominator},
        answer_text=rational_text(value),
        marks=marks,
        working=working,
        distractors=distractors,
    )
    if follow_through:
        part["follow_through"] = follow_through
    return part


def _options(key: Fraction, candidates, n: int = 3):
    out, seen = [], {key}
    for tag, value in candidates:
        value = Fraction(value)
        if value in seen or value <= 0:
            continue
        seen.add(value)
        out.append(Distractor(rational_text(value), tag, value=value))
    return pick_distractors(rational_text(key), out, n)


@generator("exd_x_r3e_product_measure_complement", level=19, presets={"exam": dict(tier="exam")})
@distinct_part_answers
def exd_x_r3e_product_measure_complement(rng, tier: str = "exam") -> dict:
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in range(2000):
        mu = [rng.choice(_MASSES) for _ in range(3)]
        nu = [rng.choice(_MASSES) for _ in range(3)]
        size_a, size_b = rng.choice([1, 2]), rng.choice([1, 2])
        ia = sorted(rng.sample(range(3), size_a))
        ib = sorted(rng.sample(range(3), size_b))
        ja = [i for i in range(3) if i not in ia]
        jb = [j for j in range(3) if j not in ib]
        mu_a, nu_b = sum(mu[i] for i in ia), sum(nu[j] for j in ib)
        mu_x, nu_y = sum(mu), sum(nu)
        rect = mu_a * nu_b
        outside = mu_x * nu_y - rect
        other_mu = mu[ja[0]]
        other_nu = nu[jb[0]]
        a_opts = _options(
            mu_a,
            [
                ("measure-of-complement-instead-of-set", mu_x - mu_a),
                ("used-total-space-measure", mu_x),
                ("read-the-wrong-point-mass", other_mu),
                ("counted-too-few-atoms", mu_a - mu[ia[-1]] if size_a == 2 else 0),
            ],
        )
        b_opts = _options(
            nu_b,
            [
                ("measure-of-complement-instead-of-set", nu_y - nu_b),
                ("used-total-space-measure", nu_y),
                ("read-the-wrong-point-mass", other_nu),
                ("counted-too-few-atoms", nu_b - nu[ib[-1]] if size_b == 2 else 0),
            ],
        )
        c_opts = _options(
            rect,
            [
                ("added-measures-instead-of-multiplying", mu_a + nu_b),
                ("used-total-space-measure", mu_x * nu_b),
                ("used-total-space-measure", mu_a * nu_y),
                ("measure-of-complement-instead-of-set", (mu_x - mu_a) * (nu_y - nu_b)),
            ],
        )
        d_opts = _options(
            outside,
            [
                ("complement-of-rectangle-as-rectangle-of-complements", (mu_x - mu_a) * (nu_y - nu_b)),
                ("rectangle-measure-for-complement", rect),
                ("used-total-space-measure", mu_x * nu_y),
                ("assumed-total-measure-is-one", 1 - rect),
            ],
        )
        if min(len(a_opts), len(b_opts), len(c_opts), len(d_opts)) >= 3 and rect != mu_x * nu_y:
            break
    else:
        check(False, "no measure spaces with enough distractors")

    # Independent route: sum the point masses of the product space directly.
    total = sum(mu[i] * nu[j] for i, j in product(range(3), range(3)))
    rect_direct = sum(mu[i] * nu[j] for i in ia for j in ib)
    outside_direct = sum(mu[i] * nu[j] for i, j in product(range(3), range(3)) if not (i in ia and j in ib))
    check(total == mu_x * nu_y and rect_direct == rect and outside_direct == outside, "product measure disagrees")

    def atoms(name, masses):
        return ", ".join(
            f"{name}(x{i + 1}) = {rational_text(m)}" if name == "mu" else f"{name}(y{i + 1}) = {rational_text(m)}"
            for i, m in enumerate(masses)
        )

    stem = (
        f"X = {{x1, x2, x3}} is a finite measure space with {atoms('mu', mu)}. "
        f"Y = {{y1, y2, y3}} is a finite measure space with {atoms('nu', nu)}. "
        f"A = {{{', '.join(f'x{i + 1}' for i in ia)}}} and B = {{{', '.join(f'y{j + 1}' for j in ib)}}}."
    )
    mu_x_text, nu_y_text = rational_text(mu_x), rational_text(nu_y)

    def tex(m):
        return str(m.numerator) if m.denominator == 1 else rf"\frac{{{m.numerator}}}{{{m.denominator}}}"

    def atoms_tex(sym, var, masses):
        return ",\\; ".join(rf"{sym}({var}_{i + 1}) = {tex(m)}" for i, m in enumerate(masses))

    stem_latex = (
        rf"X = \left\{{x_1, x_2, x_3\right\}}\text{{ is a finite measure space with }}{atoms_tex(chr(92) + 'mu', 'x', mu)}\text{{. }}"
        rf"Y = \left\{{y_1, y_2, y_3\right\}}\text{{ is a finite measure space with }}{atoms_tex(chr(92) + 'nu', 'y', nu)}\text{{. }}"
        rf"A = \left\{{{', '.join(f'x_{i + 1}' for i in ia)}\right\}}\text{{ and }}B = \left\{{{', '.join(f'y_{j + 1}' for j in ib)}\right\}}."
    )
    return dict(
        text=stem,
        latex=stem_latex,
        parts=[
            _fraction_part(
                "Find mu(A).",
                mu_a,
                1,
                [f"mu(A) = {' + '.join(rational_text(mu[i]) for i in ia)} = {rational_text(mu_a)}."],
                a_opts,
            ),
            _fraction_part(
                "Find nu(B).",
                nu_b,
                1,
                [f"nu(B) = {' + '.join(rational_text(nu[j]) for j in ib)} = {rational_text(nu_b)}."],
                b_opts,
            ),
            _fraction_part(
                "Hence find the product measure (mu x nu)(A x B).",
                rect,
                2,
                [
                    f"(mu x nu)(A x B) = mu(A) x nu(B) = {rational_text(mu_a)} x {rational_text(nu_b)} = {rational_text(rect)}."
                ],
                c_opts,
                follow_through="part_a*part_b",
            ),
            _fraction_part(
                "Find the product measure of the set of points in X x Y that are not in A x B.",
                outside,
                2,
                [
                    f"mu(X) = {mu_x_text} and nu(Y) = {nu_y_text}, so (mu x nu)(X x Y) = {rational_text(mu_x * nu_y)}.",
                    f"The measure outside A x B is {rational_text(mu_x * nu_y)} - {rational_text(rect)} = {rational_text(outside)}.",
                ],
                d_opts,
                follow_through=f"({rational_text(mu_x * nu_y)}) - part_c",
            ),
        ],
    )
