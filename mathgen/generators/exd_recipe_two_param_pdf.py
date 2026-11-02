"""Recipe subject `two_param_pdf`: a density with two unknown constants, exact answers only (VCE Exam 1 style).

The unknowns are called k (the scale) and w (the width of the support), both positive. Two families on [0, w]:

* power  -- f(x) = k x^n, n in 1..3. Area 1 gives k = (n + 1) / w^(n + 1); E(X) = (n + 1) w / (n + 2).
* linear -- f(x) = k x + b with the number b stated. Area 1 gives k = 2 (1 - b w) / w^2; E(X) = 2w/3 - b w^2/6, which
  makes E(X) a quadratic in w. Its second root always makes k negative, so exactly one root is admissible.

w (and k) are drawn first, so every stated and asked value is rational. The asks are generic over the properties
`PDF_DENSITY` and `UNKNOWN_PARAMETERS` (`_recipe_kit`): they solve area = 1, E(X) = given and P(X > m) from the
subject's density with SymPy and keep only admissible roots. The subject supplies the wrong answers its working
invites (`relation_slips`, `parameter_slips`, `tail_slips`). The ask names keep the brief's `find_d_from_mean` and
`find_c`; "d" and "c" are the first and second unknown, whatever the subject calls them.

`find_d_from_mean` carries no follow-through. A learner's wrong relation would have to be solved for w against E(X),
and the follow-through language can substitute into an earlier reply but cannot solve an equation. `find_c`
(substitute w into the relation) and `probability_above` (use k) do follow through.
"""

from __future__ import annotations

import functools
from fractions import Fraction

import sympy

from ..core import attempts, check, rational_text
from ..recipes import Subject, ask, subject
from ._r3f_kit import fraction_part
from ._recipe_kit import PDF_DENSITY, UNKNOWN_PARAMETERS, X, expression_part, formula

# The unknowns are k and w, not c and d: a labelled reply "(a) ... (b) ..." is split at label-like markers, and a
# "d)" or "c)" inside an answer such as "(4 - d)/(2d^2)" would read as the start of part (d) or (c).
C = sympy.Symbol("k")
D = sympy.Symbol("w")

_POWER_D = {
    1: (2, 3, 4, 5, 6, Fraction(3, 2), Fraction(5, 2)),
    2: (2, 3, 4, 5, Fraction(3, 2), Fraction(5, 2)),
    3: (2, 3, 4, Fraction(5, 2)),
}
_LINEAR_D = (1, 2, 3, 4, 5, 6)
_LINEAR_K = tuple(
    Fraction(a, b)
    for a, b in ((1, 12), (1, 10), (1, 8), (1, 6), (1, 5), (1, 4), (1, 3), (2, 5), (1, 2), (3, 5), (3, 4))
)


def _rat(value) -> sympy.Rational:
    value = Fraction(value)
    return sympy.Rational(value.numerator, value.denominator)


def _frac(value) -> Fraction:
    value = sympy.Rational(value)
    return Fraction(int(value.p), int(value.q))


@functools.lru_cache(maxsize=1024)
def _analyse(density, given_mean: Fraction) -> dict:
    """Solve the subject's density for its constants, by SymPy from the density alone: the relation area = 1 gives c in
    terms of d; E(X) = given then reduces to a polynomial in d, whose positive real roots are sorted into the
    admissible one and the rest. Cached: a few hundred (density, mean) pairs recur, and SymPy costs tens of ms each."""
    anti = sympy.Poly(density, X).integrate().as_expr()
    area = sympy.expand(anti.subs(X, D) - anti.subs(X, 0))
    (relation,) = sympy.solve(area - 1, C)
    first = sympy.Poly(sympy.expand(X * density), X).integrate().as_expr()
    mean = sympy.together((first.subs(X, D) - first.subs(X, 0)).subs(C, relation))
    numer, _ = sympy.fraction(sympy.together(mean - _rat(given_mean)))
    roots = [r for r in sympy.Poly(sympy.expand(numer), D).real_roots() if r > 0]
    return dict(relation=sympy.simplify(relation), anti=anti, roots=roots)


@subject("two_param_pdf")
class TwoParamPdf(Subject):
    families = frozenset({"pdf"})
    properties = frozenset({PDF_DENSITY, UNKNOWN_PARAMETERS})
    rv = "X"
    unknowns = (C, D)
    param_names = ("k", "w")

    def __init__(self, family, n, k, d, given_mean, threshold):
        self.family, self.n, self.k, self.d_value = family, n, k, d
        self.given_mean, self.threshold = given_mean, threshold
        self.density = C * X**n if family == "power" else C * X + _rat(k)
        self.lo, self.hi = sympy.Integer(0), D
        self.c_value = Fraction(n + 1) / d ** (n + 1) if family == "power" else 2 * (1 - k * d) / d**2
        self.truth = {C: self.c_value, D: self.d_value}
        self.analysis = _analyse(self.density, given_mean)
        self.valid_roots = [
            r for r in self.analysis["roots"] if self.valid({C: self.analysis["relation"].subs(D, r), D: r})
        ]
        self.other_roots = [r for r in self.analysis["roots"] if r not in self.valid_roots]

    # -- the subject interface -------------------------------------------------------------------------------

    @classmethod
    def draw(cls, rng, technology_active):
        if technology_active:
            raise ValueError("two_param_pdf asks exact answers only; it has no technology (decimal) form")
        for _ in attempts(500, "a two-parameter density with one admissible root and clean prompts"):
            if rng.random() < 0.5:
                n, k = rng.choice((1, 2, 3)), Fraction(0)
                d = Fraction(rng.choice(_POWER_D[n]))
                mean = Fraction(n + 1) * d / (n + 2)
                family = "power"
            else:
                n, k, d = 1, rng.choice(_LINEAR_K), Fraction(rng.choice(_LINEAR_D))
                if not Fraction(1, 5) <= k * d <= Fraction(4, 5):
                    continue
                mean = 2 * d / 3 - k * d * d / 6
                family = "linear"
            steps = [Fraction(j, 4) for j in range(1, 25) if Fraction(j, 4) < d]
            m = rng.choice(steps)
            s = cls(family, n, k, d, mean, m)
            check(len(s.valid_roots) == 1 and _frac(s.valid_roots[0]) == d, "the admissible root is not d")
            shown = {rational_text(mean), rational_text(m)} | ({rational_text(k)} if k else set())
            answers = {rational_text(s.c_value), rational_text(d), rational_text(s.tail(m, s.c_value))}
            if answers & shown or not 0 < s.tail(m, s.c_value) < 1:
                continue
            return s
        raise AssertionError("unreachable")

    def valid(self, values) -> bool:
        """Admissible when c and d are positive and the density is non-negative across [0, d]."""
        c, d = values[C], values[D]
        if not (c > 0 and d > 0):
            return False
        f = self.density.subs(C, c)
        return all(f.subs(X, d * sympy.Rational(i, 8)) >= 0 for i in range(9))

    def stem(self) -> str:
        if self.family == "power":
            body = "kx" + (f"^{self.n}" if self.n > 1 else "")
        else:
            body = f"kx + {rational_text(self.k)}"
        return (
            f"A continuous random variable X has probability density function f(x) = {body} for 0 ≤ x ≤ w, "
            "and f(x) = 0 otherwise, where k and w are positive constants."
        )

    def tail(self, m, c) -> Fraction:
        """P(X > m) for the true density (a Fraction), by exact integration."""
        anti = sympy.Poly(self.density.subs(C, _rat(c)), X).integrate().as_expr()
        return 1 - _frac(anti.subs(X, _rat(m)) - anti.subs(X, 0))

    # -- the wrong answers this subject's working invites ------------------------------------------------------

    def relation_slips(self):
        n, kk = self.n, _rat(self.k)
        if self.family == "power":
            return [
                ((n + 1) / D**n, "r3f-k-power-not-raised"),
                (D ** (n + 1) / (n + 1), "r3f-k-inverted"),
                (1 / D ** (n + 1), "tpp-normalisation-skips-the-divisor"),
                (1 / D, "r3f-cdf-assumes-uniform"),
            ]
        return [
            (2 / D**2, "tpp-normalisation-drops-the-k-term"),
            ((1 - kk * D) / D**2, "tpp-integral-of-cx-forgets-the-half"),
            (2 * (1 + kk * D) / D**2, "sign-error-moving-term"),
            (2 * (1 - kk) / D**2, "tpp-constant-integrates-to-itself"),
        ]

    def parameter_slips(self):
        e, n = self.given_mean, self.n
        slips = [(e, "tpp-mean-taken-as-d"), (2 * e, "r3f-mean-taken-as-midpoint")]
        if self.family == "power":
            slips.append((Fraction(n + 1) * e / (n + 2), "tpp-mean-ratio-inverted"))
        else:
            slips = [(_frac(r), "tpp-extraneous-root-kept") for r in self.other_roots if r.is_rational] + slips
            slips.append((3 * e / 2, "tpp-mean-ignores-the-k-term"))
        return slips

    def tail_slips(self):
        m, d, c, k = self.threshold, self.d_value, self.c_value, self.k
        key = self.tail(m, c)
        slips = [(1 - key, "gave-complement-probability"), (1 - m / d, "r3f-cdf-assumes-uniform")]
        if self.family == "power":
            slips.append((1 - (m / d) ** self.n, "tpp-cdf-power-not-raised"))
        else:
            slips.append((1 - c * m * m, "tpp-integral-of-cx-forgets-the-half"))
            slips.append((1 - c * m * m / 2, "tpp-cdf-drops-the-k-term"))
        return slips


# -- generic asks ---------------------------------------------------------------------------------------------


@ask(None, "normalisation_relation", needs=(PDF_DENSITY, UNKNOWN_PARAMETERS), default_marks=2)
def normalisation_relation(s, rng, ctx):
    first, second = s.param_names
    rel = s.analysis["relation"]
    prompt = f"Find {first} in terms of {second}."
    slips = [(e, tag, formula(e)) for e, tag in s.relation_slips()]
    return expression_part(
        prompt,
        rel,
        formula(rel),
        2,
        [
            f"The total area under the density is 1, so the integral of f from {s.lo} to {s.hi} equals 1.",
            f"{first} = {formula(rel)}.",
        ],
        slips,
    )


@ask(
    None,
    "find_d_from_mean",
    needs=(PDF_DENSITY, UNKNOWN_PARAMETERS),
    requires=("normalisation_relation",),
    default_marks=3,
)
def find_d_from_mean(s, rng, ctx):
    first, second = s.param_names
    value = _frac(s.valid_roots[0])
    working = [
        f"Substitute {first} = {formula(s.analysis['relation'])} into E({s.rv}) = {rational_text(s.given_mean)}.",
        f"This gives {second} = {rational_text(value)}"
        + (" (the other root makes the constant negative, so it is rejected)." if s.other_roots else "."),
    ]
    return fraction_part(
        f"Given that E({s.rv}) = {rational_text(s.given_mean)}, find the value of {second}.",
        value,
        3,
        working,
        s.parameter_slips(),
        sensible_range=(0, None),
    )


@ask(None, "find_c", needs=(PDF_DENSITY, UNKNOWN_PARAMETERS), requires=("normalisation_relation", "find_d_from_mean"))
def find_c(s, rng, ctx):
    first, second = s.param_names
    rel_label, d_label = ctx.label_of("normalisation_relation"), ctx.label_of("find_d_from_mean")
    value = s.c_value
    at_d = {sympy.Symbol(second): _rat(s.d_value)}
    slips = [(_frac(e.subs(at_d)), tag) for e, tag in s.relation_slips()]
    return fraction_part(
        f"Hence find the value of {first}.",
        value,
        1,
        [
            f"{first} = {formula(s.analysis['relation'])} with {second} = {rational_text(s.d_value)}, so {first} = {rational_text(value)}."
        ],
        slips,
        follow_through=f"Subs(part_{rel_label}, ({second},), (part_{d_label},))",
        sensible_range=(0, None),
    )


@ask(None, "probability_above", needs=(PDF_DENSITY, UNKNOWN_PARAMETERS), requires=("find_c",), default_marks=2)
def probability_above(s, rng, ctx):
    first, _ = s.param_names
    c_label = ctx.label_of("find_c")
    m = s.threshold
    anti = s.analysis["anti"]
    carried = 1 - (anti.subs(X, _rat(m)) - anti.subs(X, 0)).subs(D, _rat(s.d_value)).subs(
        C, sympy.Symbol(f"part_{c_label}")
    )
    value = s.tail(m, s.c_value)
    return fraction_part(
        f"Hence find P({s.rv} > {rational_text(m)}).",
        value,
        2,
        [
            f"P({s.rv} > {rational_text(m)}) = 1 - (the integral of f from 0 to {rational_text(m)}) = {rational_text(value)}."
        ],
        s.tail_slips(),
        follow_through=str(sympy.expand(carried)),
        sensible_range=(0, 1),
    )
