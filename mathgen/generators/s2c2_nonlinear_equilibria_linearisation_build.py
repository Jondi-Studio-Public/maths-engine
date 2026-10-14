"""Generates `s2c2_nonlinear_equilibria_linearisation` questions (easy, medium, hard) and the exam chain
`exd_x_s2c2_nonlinear_equilibria_linearisation`.

``uc2-nonlinear-equilibria-linearisation`` (``ordinary-differential-equations``, answer ``matrix``): linearising a
nonlinear two-dimensional system at an equilibrium. The models are competing species, predator and prey with
logistic growth in the prey, and the damped pendulum.

* easy: the Jacobian matrix at a stated equilibrium on the x-axis (or at (n pi, 0) for the pendulum), where the
  product rule decides the second row.
* medium: the Jacobian matrix at the coexistence equilibrium (or the pendulum equilibrium in 0 < x < 2 pi), which
  the learner must first find by solving the nullcline equations.
* hard: the two eigenvalues of the linearisation at an equilibrium (an ordered list), the parameters chosen so they
  are rational.
* exam (``exd_x_s2c2_nonlinear_equilibria_linearisation``): a competing-species or predator-prey system in four
  parts: (a) x*, (b) y* of the coexistence equilibrium, (c) the determinant of the Jacobian there (follows through
  from (a) and (b)), (d) the classification of the equilibrium (multiple choice).

Equilibria are found by Cramer's rule (competing) or from the two nullclines (predator-prey), and the Jacobian by
hand. SymPy then differentiates the vector field itself, substitutes the equilibrium, checks the field vanishes
there, and finds the eigenvalues, and so the classification, again.
"""

from __future__ import annotations

import math
from fractions import Fraction

import sympy

from ..answers import Distractor, exact_value
from ..core import attempts, check, generator, join_terms, pick_distractors, rational_text
from ._ucv1_linalg import column_value, fm, mat_str, matrix_distractors, rows_value, transpose, vec_str
from ._ucv1_texsay import say_marked

X, Y = sympy.symbols("x y")
FORM = "scalar_multiple"


def _sym(v):
    v = Fraction(v)
    return sympy.Rational(v.numerator, v.denominator)


def _point_text(model, x, y) -> str:
    if model.kind == "pendulum":
        n = int(x)
        return "(0, 0)" if n == 0 else ("(π, 0)" if n == 1 else f"({n}π, 0)")
    return f"({rational_text(x)}, {rational_text(y)})"


class _Model:
    """A system x' = f(x, y), y' = g(x, y) with its wording and its hand-built equilibria and Jacobian."""

    def __init__(self, kind, **p):
        self.kind, self.p = kind, p
        if kind == "compete":
            a, b, c, d, e, f = (p[k] for k in "abcdef")
            self.f = X * (a - b * X - c * Y)
            self.g = Y * (d - e * X - f * Y)
            self.text = (
                f"x' = x({join_terms([(a, ''), (-b, 'x'), (-c, 'y')])}), "
                f"y' = y({join_terms([(d, ''), (-e, 'x'), (-f, 'y')])})"
            )
        elif kind == "prey":
            a, b, c, d, e = (p[k] for k in "abcde")
            self.f = X * (a - b * X - c * Y)
            self.g = Y * (-d + e * X)
            self.text = (
                f"x' = x({join_terms([(a, ''), (-b, 'x'), (-c, 'y')])}), y' = y({join_terms([(-d, ''), (e, 'x')])})"
            )
        else:
            w2, c = p["w2"], p["c"]
            self.f = Y
            self.g = -w2 * sympy.sin(X) - c * Y
            swing = "-sin x" if w2 == 1 else f"-{w2} sin x"
            self.text = f"x' = y, y' = {swing} - {'' if c == 1 else c}y"

    # -- equilibria and Jacobians, by hand --------------------------------------------------------------------

    def interior(self):
        p = self.p
        if self.kind == "compete":
            a, b, c, d, e, f = (p[k] for k in "abcdef")
            det = b * f - c * e
            check(det != 0, "parallel nullclines")
            return Fraction(a * f - c * d, det), Fraction(b * d - a * e, det)
        a, b, c, d, e = (p[k] for k in "abcde")
        x = Fraction(d, e)
        return x, (a - b * x) / c

    def jacobian_at(self, x, y):
        p = self.p
        if self.kind == "compete":
            a, b, c, d, e, f = (p[k] for k in "abcdef")
            return fm([[a - 2 * b * x - c * y, -c * x], [-e * y, d - e * x - 2 * f * y]])
        if self.kind == "prey":
            a, b, c, d, e = (p[k] for k in "abcde")
            return fm([[a - 2 * b * x - c * y, -c * x], [e * y, -d + e * x]])
        cos_x = 1 if int(x) % 2 == 0 else -1
        return fm([[0, 1], [-p["w2"] * cos_x, -p["c"]]])

    def jacobian_without_product_rule(self, x, y):
        """d(u g)/dv taken as u dg/dv: the bracket's own derivative times the factor outside it."""
        p = self.p
        if self.kind == "compete":
            return fm([[-p["b"] * x, -p["c"] * x], [-p["e"] * y, -p["f"] * y]])
        return fm([[-p["b"] * x, -p["c"] * x], [p["e"] * y, 0]])

    def check_point(self, x, y, jac):
        """SymPy differentiates the field itself, evaluates at (x, y), and the entries must match `jac`."""
        pt = {X: sympy.pi * int(x) if self.kind == "pendulum" else _sym(x), Y: _sym(y)}
        field = sympy.Matrix([self.f, self.g])
        check(field.subs(pt) == sympy.zeros(2, 1), "not an equilibrium")
        theirs = field.jacobian([X, Y]).subs(pt)
        for i in range(2):
            for j in range(2):
                check(sympy.simplify(theirs[i, j] - _sym(jac[i][j])) == 0, "Jacobian disagrees with SymPy")


def _eigenvalues(jac):
    return list(sympy.Matrix([[_sym(v) for v in row] for row in jac]).eigenvals())


def classify(trace, det):
    """The type of an equilibrium from the trace and determinant of its Jacobian (no zero or repeated roots)."""
    trace, det = Fraction(trace), Fraction(det)
    check(det != 0 and trace != 0, "borderline linearisation")
    if det < 0:
        return "saddle"
    disc = trace * trace - 4 * det
    check(disc != 0, "repeated eigenvalue")
    if disc > 0:
        return "stable node" if trace < 0 else "unstable node"
    return "stable spiral" if trace < 0 else "unstable spiral"


def _check_classification(jac, label):
    roots = _eigenvalues(jac)
    if any(sympy.im(r) != 0 for r in roots):
        got = "stable spiral" if sympy.re(roots[0]) < 0 else "unstable spiral"
    elif roots[0] * roots[-1] < 0:
        got = "saddle"
    else:
        got = "stable node" if roots[0] < 0 else "unstable node"
    check(got == label, "SymPy classifies the equilibrium differently")


def _is_square(value) -> Fraction | None:
    """The exact square root of a non-negative Fraction, or None."""
    value = Fraction(value)
    if value < 0:
        return None
    num, den = math.isqrt(value.numerator), math.isqrt(value.denominator)
    return Fraction(num, den) if num * num == value.numerator and den * den == value.denominator else None


# ---------------------------------------------------------------------------------------------------------------
# parameter draws


def _draw_network(rng, kind, *, integer_equilibrium=False):
    """A competing-species or predator-prey model and its coexistence equilibrium."""
    for _ in attempts(4000, "a system with a coexistence equilibrium"):
        if kind == "compete":
            b, c, e, f = (rng.randint(1, 4) for _ in range(4))
            if b * f == c * e:
                continue
            if integer_equilibrium:
                x, y = rng.randint(1, 4), rng.randint(1, 4)
                a, d = b * x + c * y, e * x + f * y
            else:
                a, d = rng.randint(2, 18), rng.randint(2, 18)
            model = _Model("compete", a=a, b=b, c=c, d=d, e=e, f=f)
        else:
            b, c, e = (rng.randint(1, 4) for _ in range(3))
            if integer_equilibrium:
                x, y = rng.randint(1, 4), rng.randint(1, 4)
                a, d = b * x + c * y, e * x
            else:
                a, d = rng.randint(3, 20), rng.randint(1, 9)
            model = _Model("prey", a=a, b=b, c=c, d=d, e=e)
        x, y = model.interior()
        if x > 0 and y > 0 and x.denominator <= 6 and y.denominator <= 6:
            return model, (x, y)
    raise AssertionError("unreachable")


def _draw_pendulum(rng, *, n=None, rational_roots=False):
    """A damped pendulum y' = -w2 sin x - c y. With rational_roots the linearisation at (n pi, 0) has rational
    eigenvalues: roots k and -(k + c) at an odd n, roots -k1 and -k2 at an even n."""
    n = rng.choice([0, 1, 2, 3]) if n is None else n
    if not rational_roots:
        return _Model("pendulum", w2=rng.randint(1, 12), c=rng.randint(1, 7)), n
    if n % 2:
        k, c = rng.randint(1, 3), rng.randint(1, 5)
        return _Model("pendulum", w2=k * (k + c), c=c), n
    k1, k2 = rng.sample(range(1, 6), 2)
    return _Model("pendulum", w2=k1 * k2, c=k1 + k2), n


_CONTEXT = {
    "compete": "Two competing species of native fish have populations x and y, in hundreds, that satisfy ",
    "prey": "The populations of rabbits (x) and foxes (y) on an island, in hundreds, satisfy ",
    "pendulum": "A damped pendulum has angle x and angular velocity y, and satisfies ",
}


def _stem(model, ask):
    return say_marked(f"{_CONTEXT[model.kind]}«{model.text}». {ask}")


# ---------------------------------------------------------------------------------------------------------------
# single-answer tiers


def _jacobian_candidates(model, point, jac):
    x, y = point
    out = [(transpose(jac), "ode2-jacobian-transposed"), ([jac[1], jac[0]], "ode2-jacobian-rows-swapped")]
    if model.kind == "pendulum":
        out.append((model.jacobian_at(x + 1, y), "ode2-jacobian-at-wrong-equilibrium"))
        out.append((model.jacobian_at(0, 0), "ode2-jacobian-at-origin"))
        sign_flipped = model.jacobian_at(x, y)
        sign_flipped[1][1] = -sign_flipped[1][1]
        out.append((sign_flipped, "ode2-damping-term-sign"))
        return out
    out.append((model.jacobian_without_product_rule(x, y), "ode2-product-rule-missed"))
    out.append((model.jacobian_at(0, 0), "ode2-jacobian-at-origin"))
    ix, iy = model.interior()
    if (x, y) != (ix, iy):
        out.append((model.jacobian_at(ix, iy), "ode2-jacobian-at-wrong-equilibrium"))
    else:
        out.append(
            (model.jacobian_at(Fraction(model.p["a"], model.p["b"]), Fraction(0)), "ode2-jacobian-at-wrong-equilibrium")
        )
    return out


def _finish_matrix(text, latex, jac, candidates, working):
    answer = rows_value(jac)
    answer_text = mat_str(jac)
    distractors = pick_distractors(answer_text, matrix_distractors(candidates, answer), 4)
    check(len(distractors) >= 3, "too few Jacobian distractors")
    return dict(
        text=text,
        latex=latex,
        answer_kind="matrix",
        answer=answer,
        answer_text=answer_text,
        answer_latex="plain:" + answer_text,
        working=working,
        distractors=distractors,
    )


@generator(
    "s2c2_nonlinear_equilibria_linearisation",
    level=22,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def s2c2_nonlinear_equilibria_linearisation(rng, tier="easy"):
    kind = rng.choice(["compete", "prey", "pendulum"])
    return {"easy": _easy, "medium": _medium, "hard": _hard}[tier](rng, kind)


def _easy(rng, kind):
    if kind == "pendulum":
        model, n = _draw_pendulum(rng, n=rng.choice([1, 2, 3]))
        point = (n, 0)
    else:
        model, _ = _draw_network(rng, kind, integer_equilibrium=True)
        point = (Fraction(model.p["a"], model.p["b"]), Fraction(0))  # the equilibrium on the x-axis
    jac = model.jacobian_at(*point)
    model.check_point(*point, jac)
    shown = _point_text(model, *point)
    text, latex = _stem(model, f"Find the Jacobian matrix of the system at the equilibrium «{shown}».")
    working = [f"Differentiating each field and evaluating at {shown} gives {mat_str(jac)}."]
    return _finish_matrix(text, latex, jac, _jacobian_candidates(model, point, jac), working)


def _medium(rng, kind):
    if kind == "pendulum":
        model, _ = _draw_pendulum(rng, n=1)
        point = (1, 0)
        ask = "Find the Jacobian matrix of the system at its equilibrium with «0 < x < 2π»."
    else:
        model, point = _draw_network(rng, kind)
        ask = "Find the Jacobian matrix of the system at the equilibrium with «x > 0» and «y > 0»."
    jac = model.jacobian_at(*point)
    model.check_point(*point, jac)
    text, latex = _stem(model, ask)
    cands = _jacobian_candidates(model, point, jac)
    if kind != "pendulum":
        p = model.p
        # each bracket solved for its own axis, instead of the two brackets together
        wrong = (
            (Fraction(p["a"], p["b"]), Fraction(p["d"], p["f"]))
            if kind == "compete"
            else (Fraction(p["d"], p["e"]), Fraction(p["a"], p["c"]))
        )
        cands.insert(0, (model.jacobian_at(*wrong), "ode2-equilibrium-from-single-equations"))
    working = [f"The equilibrium is {_point_text(model, *point)}; the Jacobian there is {mat_str(jac)}."]
    return _finish_matrix(text, latex, jac, cands, working)


def _unit_last(vec):
    """`vec` scaled so its last entry is 1, or None when that entry is 0."""
    last = Fraction(vec[-1])
    return None if last == 0 else [Fraction(v) / last for v in vec]


def _eigenvector(jac, lam):
    """An eigenvector of the 2x2 `jac` for `lam`, from the second row: (lam - s, r)."""
    return [lam - jac[1][1], jac[1][0]]


def _hard(rng, kind):
    kind = "compete" if kind == "prey" else kind  # the coexistence point of a predator-prey model is never a saddle
    for _ in attempts(6000, "a saddle with rational eigenvalues and enough distractors"):
        if kind == "pendulum":
            model, n = _draw_pendulum(rng, n=rng.choice([1, 3]), rational_roots=True)
            point = (n, 0)
            ask_point = f"the equilibrium «{_point_text(model, n, 0)}»"
        else:
            model, point = _draw_network(rng, "compete", integer_equilibrium=True)
            ask_point = "the equilibrium with «x > 0» and «y > 0»"
        jac = model.jacobian_at(*point)
        trace, det = jac[0][0] + jac[1][1], jac[0][0] * jac[1][1] - jac[0][1] * jac[1][0]
        root = _is_square(trace * trace - 4 * det)
        if det >= 0 or root is None or jac[1][0] == 0:
            continue
        lam_pos, lam_neg = (trace + root) / 2, (trace - root) / 2
        if lam_pos <= 0 or lam_neg >= 0:
            continue
        vec = _unit_last(_eigenvector(jac, lam_pos))
        wrong = [
            (_unit_last(_eigenvector(jac, lam_neg)), "eigvec-other-eigenvalue"),
            ([vec[1], vec[0]], "eigvec-components-swapped"),
            ([-vec[0], vec[1]], "eigvec-component-sign-slip"),
            (_unit_last(_eigenvector(transpose(jac), lam_pos)), "ode2-jacobian-transposed"),
            (_unit_last([jac[0][0] - lam_pos, jac[0][1]]), "eigvec-row-as-vector"),
            (_unit_last([jac[1][0], jac[1][1] - lam_pos]), "eigvec-row-as-vector"),
        ]
        scored = [
            Distractor(vec_str(cand), tag, column_value(cand))
            for cand, tag in wrong
            if cand is not None and cand[0] * vec[1] - cand[1] * vec[0] != 0
        ]
        answer_text = vec_str(vec)
        distractors = pick_distractors(answer_text, scored, 4)
        if len(distractors) >= 3:
            break
    model.check_point(*point, jac)
    # SymPy's null space of J - lambda I is a second route to the same direction.
    ns = (sympy.Matrix([[_sym(v) for v in row] for row in jac]) - _sym(lam_pos) * sympy.eye(2)).nullspace()
    check(len(ns) == 1, "the eigenspace is not a line")
    check(_sym(vec[0]) * ns[0][1] - _sym(vec[1]) * ns[0][0] == 0, "SymPy finds a different eigenvector")
    text, latex = _stem(
        model,
        f"The linearisation at {ask_point} has one positive eigenvalue. Find an eigenvector for it, "
        "scaled so that its last entry is 1.",
    )
    return dict(
        text=text,
        latex=latex,
        answer_kind="matrix",
        answer=column_value(vec),
        answer_text=answer_text,
        answer_latex="plain:" + answer_text,
        answer_form=FORM,
        distractors=distractors,
        working=[
            f"The Jacobian is {mat_str(jac)}, with eigenvalues {rational_text(lam_neg)} and {rational_text(lam_pos)}.",
            f"(J - {rational_text(lam_pos)}I)v = 0 gives the direction {answer_text}.",
        ],
    )


# ---------------------------------------------------------------------------------------------------------------
# exam chain


class _Redraw(Exception):
    pass


@generator("exd_x_s2c2_nonlinear_equilibria_linearisation", level=22, presets={"exam": dict(tier="exam")})
def exd_x_s2c2_nonlinear_equilibria_linearisation(rng, tier="exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(60, "an exam question with enough distractors"):
        try:
            return _exam_once(rng)
        except _Redraw:
            continue
    raise AssertionError("unreachable")


def _classification_distractors(label):
    flip = {
        "stable node": "unstable node",
        "unstable node": "stable node",
        "stable spiral": "unstable spiral",
        "unstable spiral": "stable spiral",
        "saddle": "stable node",
    }
    swap = {
        "stable node": "stable spiral",
        "unstable node": "unstable spiral",
        "stable spiral": "stable node",
        "unstable spiral": "unstable node",
        "saddle": "unstable node",
    }
    ordered = [
        ("ode2-classification-stability-flip", flip[label]),
        ("ode2-classification-node-spiral-swapped", swap[label]),
        ("ode2-classification-saddle-misjudged", "saddle" if label != "saddle" else "stable spiral"),
        ("ode2-classification-stability-flip", "unstable spiral" if "spiral" not in label else "stable node"),
    ]
    seen, out = {label}, []
    for tag, text in ordered:
        if text not in seen:
            seen.add(text)
            out.append(Distractor(text, tag, text))
    return out


def _exam_once(rng):
    kind = rng.choice(["compete", "prey"])
    model, (xs, ys) = _draw_network(rng, kind)
    p = model.p
    jac = model.jacobian_at(xs, ys)
    model.check_point(xs, ys, jac)
    trace, det = jac[0][0] + jac[1][1], jac[0][0] * jac[1][1] - jac[0][1] * jac[1][0]
    k = p["b"] * p["f"] - p["c"] * p["e"] if kind == "compete" else p["c"] * p["e"]
    check(det == xs * ys * k, "the determinant is not x* y* K")
    if det == 0 or trace == 0 or trace * trace == 4 * det:
        raise _Redraw
    label = classify(trace, det)
    _check_classification(jac, label)

    if kind == "compete":
        a, b, c, d, e, f = (p[q] for q in "abcdef")
        d0 = b * f - c * e
        wrong_x = [
            ("ode2-equilibrium-from-single-equations", Fraction(a, b)),
            ("ode2-cramer-sign-slip", Fraction(a * f + c * d, d0)),
            ("ode2-coordinates-swapped", ys),
            ("ode2-solution-not-divided-by-determinant", Fraction(a * f - c * d)),
        ]
        wrong_y = [
            ("ode2-equilibrium-from-single-equations", Fraction(d, f)),
            ("ode2-cramer-sign-slip", Fraction(b * d + a * e, d0)),
            ("ode2-coordinates-swapped", xs),
            ("ode2-solution-not-divided-by-determinant", Fraction(b * d - a * e)),
        ]
        origin_det = Fraction(a * d)
    else:
        a, b, c, d, e = (p[q] for q in "abcde")
        wrong_x = [
            ("ode2-equilibrium-from-single-equations", Fraction(a, b)),
            ("ode2-coordinates-swapped", ys),
            ("ode2-solution-not-divided-by-determinant", Fraction(d)),
            ("ode2-ratio-inverted", Fraction(e, d)),
        ]
        wrong_y = [
            ("ode2-equilibrium-from-single-equations", Fraction(a, c)),
            ("ode2-cramer-sign-slip", (a + b * xs) / c),
            ("ode2-coordinates-swapped", xs),
            ("ode2-solution-not-divided-by-determinant", a - b * xs),
        ]
        origin_det = Fraction(-a * d)
    wrong_det = [
        ("ode2-trace-for-determinant", trace),
        ("ode2-determinant-sign-slip", -det),
        ("ode2-jacobian-at-wrong-equilibrium", origin_det),
        ("ode2-determinant-factor-left-out", xs * ys),
    ]

    def fraction_dist(correct, wrong):
        return pick_distractors(
            rational_text(correct),
            [Distractor(rational_text(v), tag, exact_value(v)) for tag, v in wrong if v != correct],
            4,
        )

    x_dist, y_dist, det_dist = fraction_dist(xs, wrong_x), fraction_dist(ys, wrong_y), fraction_dist(det, wrong_det)
    class_dist = _classification_distractors(label)
    if min(len(x_dist), len(y_dist), len(det_dist)) < 2 or len(class_dist) < 3:
        raise _Redraw

    stem, stem_latex = _stem(
        model,
        "The system has exactly one equilibrium «(x*, y*)» with «x* > 0» and «y* > 0», where both species survive.",
    )

    def prompt(template):
        t, tex = say_marked(template)
        return dict(prompt=t, prompt_latex=tex)

    return dict(
        text=stem,
        latex=stem_latex,
        parts=[
            dict(
                **prompt("Find «x*»."),
                answer_kind="fraction",
                answer=exact_value(xs),
                answer_text=rational_text(xs),
                marks=2,
                working=[
                    "Setting both brackets to zero and solving the two linear equations gives x* = "
                    + rational_text(xs)
                    + "."
                ],
                distractors=x_dist,
            ),
            dict(
                **prompt("Find «y*»."),
                answer_kind="fraction",
                answer=exact_value(ys),
                answer_text=rational_text(ys),
                marks=2,
                working=["The same two equations give y* = " + rational_text(ys) + "."],
                distractors=y_dist,
            ),
            dict(
                **prompt("Find the determinant of the Jacobian matrix of the system at «(x*, y*)»."),
                answer_kind="fraction",
                answer=exact_value(det),
                answer_text=rational_text(det),
                marks=2,
                follow_through=f"part_a*part_b*({k})",
                working=[f"The Jacobian is {mat_str(jac)}, with determinant {rational_text(det)}."],
                distractors=det_dist,
            ),
            dict(
                **prompt("Classify the equilibrium «(x*, y*)» using the linearisation."),
                answer_kind="mcq",
                answer=label,
                answer_text=label,
                marks=1,
                working=[
                    f"The trace is {rational_text(trace)} and the determinant is {rational_text(det)}; "
                    f"trace² - 4 × determinant = {rational_text(trace * trace - 4 * det)}, so it is a {label}."
                ],
                distractors=class_dist,
            ),
        ],
    )
