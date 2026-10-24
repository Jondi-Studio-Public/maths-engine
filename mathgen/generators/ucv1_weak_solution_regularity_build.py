"""Generates `ucv1-weak-solution-regularity` questions. `answer: mcq`.

Scope: a one-dimensional boundary value problem on a bounded interval (a, b) is fixed, its weak
solution u is defined by the weak form, and the Sobolev regularity of the data f is supplied, either as a label
("f is in H^k but not in H^(k+1)") or as an explicit function whose label follows from a stated fact. The learner
selects the statement that gives the exact regularity of u: "u is in H^m but not in H^(m+1)". General existence
proofs, nonlinear regularity theory, higher dimensions, corners and theorems the question does not supply are out of
scope.

Why exactly one option is true, for every question this generator can write (the argument is elementary because the
domain is an interval, so nothing is retrieved from a theorem that the question does not state):

  * The weak form, tested against smooth compactly supported v, says the differential equation holds in the sense of
    distributions on (a, b).
  * On a bounded interval, g is in H^m exactly when g and its m-th derivative are in L^2.
  * Second-order equations. -u'' = f gives u'' = -f. For -u'' + c u = f with c > 0, u'' = c u - f, and bootstrapping
    from u in H^1 raises the regularity of u to H^(k+2) when f is in H^k (c u is no less regular than u, hence more
    regular than f); u is not in H^(k+3) because then c u would be in H^(k+1) and f = c u - u'' in H^(k+1), which is
    false. For -(p u')' = f with p smooth and positive, u'' = -(f + p' u') / p, and the same bootstrap and the same
    argument give u in H^(k+2) but not in H^(k+3).
  * The fourth-order equation u'''' = f (a clamped beam, u in H^2_0) gives u'''' = f, so u is in H^(k+4) and not in
    H^(k+5).
  * Hence, whenever f is in H^k but not in H^(k+1), u is in H^m but not in H^(m+1) with m = k + 2 (second order) or
    m = k + 4 (fourth order). Every distractor names a different m, so it is false.
  * Explicit data: for alpha that is not an even integer, |x - x0|^alpha is in H^k(a, b) exactly when
    alpha > k - 1/2 (the derivative of order k behaves like |x - x0|^(alpha - k), which is square integrable exactly
    when 2 (alpha - k) > -1; the function is smooth elsewhere). The question states this as a fact. The generator
    recomputes k by comparing exact fractions, never floats.

Ladder (reasoning escalation, not size):
  easy:   -u'' = f with f's label supplied; find the label of u.
  medium: a lower-order term (-u'' + c u = f) or a variable coefficient (-(p(x) u')' = f); the label of f supplied,
          so the learner must see that the extra term does not change the regularity gain.
  hard:   f is an explicit power |x - x0|^alpha, so the label of f has to be found first from the stated fact, and
          then carried through one of the three second-order operators.
  exam:   the fourth-order clamped beam u'''' = f, where the gain is four derivatives, with f given by label or as
          an explicit power; two-derivative reasoning gives a wrong answer.
"""

from __future__ import annotations

from fractions import Fraction

from mathgen.core import check, generator

_TIERS = ("easy", "medium", "hard", "exam")

_SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")

_INTERVALS = [(0, 1), (0, 2), (0, 3), (-1, 1), (-1, 2), (1, 3), (-2, 2), (0, 4), (2, 4), (1, 4), (-3, 1), (0, 6)]
_DATA_NAMES = ("f", "g", "h")
_REACTION = (2, 3, 4, 9)
_COEFFICIENTS = [("1 + x²", "1 + x^{2}"), ("2 + sin x", r"2 + \sin x"), ("eˣ", "e^{x}")]
_ALPHAS = [Fraction(1, 2), Fraction(1), Fraction(3, 2), Fraction(5, 2), Fraction(3), Fraction(7, 2)]


def _num(value) -> str:
    """An interval endpoint or midpoint as plain text: 1, -1, 0.5."""
    value = Fraction(value)
    if value.denominator == 1:
        return str(value.numerator)
    return str(float(value))


def _h(order: int) -> str:
    return "L²" if order == 0 else "H" + str(order).translate(_SUP)


def _h_latex(order: int) -> str:
    return r"L^{2}" if order == 0 else f"H^{{{order}}}"


def _alpha_text(alpha: Fraction) -> str:
    return str(alpha.numerator) if alpha.denominator == 1 else f"{alpha.numerator}/{alpha.denominator}"


def _alpha_latex(alpha: Fraction) -> str:
    return str(alpha.numerator) if alpha.denominator == 1 else rf"\frac{{{alpha.numerator}}}{{{alpha.denominator}}}"


def _label_of_power(alpha: Fraction) -> int:
    """The largest k with alpha > k - 1/2, by exact fractions."""
    return max(k for k in range(20) if alpha > k - Fraction(1, 2))


class _Doc:
    """A question stem built as parallel plain text and LaTeX. Prose goes into \\text{}; math is given both ways."""

    def __init__(self):
        self.text: list[str] = []
        self.latex: list[str] = []

    def prose(self, s: str):
        self.text.append(s)
        self.latex.append(r"\text{" + s.replace("%", r"\%") + "}")

    def math(self, plain: str, latex: str):
        self.text.append(plain)
        self.latex.append(r"\," + latex + r"\,")

    def result(self) -> tuple[str, str]:
        return "".join(self.text), "".join(self.latex)


def _power_text(x0: Fraction, alpha: Fraction) -> tuple[str, str]:
    """|x - x0|^alpha as (plain text, LaTeX); x0 = 0 reads |x|^alpha."""
    if x0 == 0:
        base = "|x|"
    elif x0 > 0:
        base = f"|x - {_num(x0)}|"
    else:
        base = f"|x + {_num(-x0)}|"
    if alpha == 1:
        return base, base
    exponent = _alpha_text(alpha)
    plain = f"{base}^{exponent}" if alpha.denominator == 1 else f"{base}^({exponent})"
    return plain, f"{base}^{{{_alpha_latex(alpha)}}}"


def _stem(
    kind: str,
    a: int,
    b: int,
    fname: str,
    *,
    k: int | None,
    alpha: Fraction | None,
    x0: Fraction | None,
    c: int | None,
    coef: tuple[str, str] | None,
):
    """(text, latex) for the question; exactly one of `k` (label given) and `alpha` (explicit power) is set."""
    d = _Doc()
    dom = f"({a}, {b})"
    if kind == "beam":
        space, space_l = "H²₀", r"H^{2}_{0}"
    else:
        space, space_l = "H¹₀", r"H^{1}_{0}"
    d.prose("Let ")
    d.math("u", "u")
    d.prose(" be the weak solution in ")
    d.math(f"{space}{dom}", f"{space_l}{dom}")
    d.prose(" of ")
    if kind == "laplace":
        d.math(f"-u'' = {fname}", f"-u'' = {fname}")
        weak = (f"∫ u' v' dx = ∫ {fname} v dx", rf"\int u'\,v'\,dx = \int {fname}\,v\,dx")
    elif kind == "reaction":
        d.math(f"-u'' + {c}u = {fname}", f"-u'' + {c}u = {fname}")
        weak = (
            f"∫ (u' v' + {c}u v) dx = ∫ {fname} v dx",
            rf"\int (u'\,v' + {c}\,u\,v)\,dx = \int {fname}\,v\,dx",
        )
    elif kind == "variable":
        d.math(f"-(p(x) u')' = {fname}", rf"-(p(x)\,u')' = {fname}")
        weak = (f"∫ p u' v' dx = ∫ {fname} v dx", rf"\int p\,u'\,v'\,dx = \int {fname}\,v\,dx")
    else:
        d.math(f"u'''' = {fname}", f"u'''' = {fname}")
        weak = (f"∫ u'' v'' dx = ∫ {fname} v dx", rf"\int u''\,v''\,dx = \int {fname}\,v\,dx")
    d.prose(" on ")
    d.math(dom, dom)
    if kind == "beam":
        d.prose(" (a clamped beam)")
    d.prose(", meaning ")
    d.math(*weak)
    d.prose(" for every ")
    d.math(f"v ∈ {space}{dom}", rf"v \in {space_l}{dom}")
    d.prose(", where each integral is over ")
    d.math(dom, dom)
    d.prose(". ")
    if kind == "variable":
        d.prose("Here ")
        d.math(f"p(x) = {coef[0]}", f"p(x) = {coef[1]}")
        d.prose(" is smooth and positive. ")
    d.prose("Facts you may use: the weak form gives the differential equation in the sense of distributions on ")
    d.math(dom, dom)
    d.prose("; and on a bounded interval a function lies in ")
    d.math("Hᵐ", "H^{m}")
    d.prose(" exactly when it and its m-th derivative lie in ")
    d.math("L²", "L^{2}")
    d.prose(". ")
    if alpha is not None:
        d.prose("For ")
        d.math("α", r"\alpha")
        d.prose(" that is not an even integer, ")
        d.math("|x - x₀|^α", r"|x - x_{0}|^{\alpha}")
        d.prose(" lies in ")
        d.math("Hᵏ", "H^{k}")
        d.prose(" on the interval, for ")
        d.math("x₀", "x_{0}")
        d.prose(" inside it, exactly when ")
        d.math("α > k - 1/2", r"\alpha > k - \tfrac{1}{2}")
        d.prose(". The data are ")
        power, power_l = _power_text(x0, alpha)
        d.math(f"{fname}(x) = {power}", f"{fname}(x) = {power_l}")
        d.prose(". ")
    else:
        d.prose("The data satisfy ")
        d.math(f"{fname} ∈ {_h(k)}{dom}", rf"{fname} \in {_h_latex(k)}{dom}")
        d.prose(" but ")
        d.math(f"{fname} ∉ {_h(k + 1)}{dom}", rf"{fname} \notin {_h_latex(k + 1)}{dom}")
        d.prose(". ")
    d.prose("Which statement gives the exact regularity of u?")
    return d.result()


def _statement(m: int, a: int, b: int) -> str:
    return f"u ∈ {_h(m)}({a}, {b}), but u ∉ {_h(m + 1)}({a}, {b})"


_NOTE_UNDER = "under-counts-the-regularity-gain"
_NOTE_NO_GAIN = "assumes-no-regularity-gain"
_NOTE_OVER = "over-counts-the-regularity-gain"
_NOTE_LOSS = "loses-regularity-when-solving"
_NOTE_SECOND_ORDER = "uses-the-second-order-gain-for-the-fourth-order-equation"


def _note_for(offset: int, gain: int) -> str:
    """The error that gives an order `offset` away from the right one, when the equation gains `gain` derivatives."""
    if offset > 0:
        return _NOTE_OVER
    if offset == -gain:
        return _NOTE_NO_GAIN
    if offset < -gain:
        return _NOTE_LOSS
    if gain == 4 and offset == -2:
        return _NOTE_SECOND_ORDER
    return _NOTE_UNDER


def _options(rng, m: int, a: int, b: int, *, gain: int):
    """(key, distractors, option_notes). Each distractor names a different m, so each is false. The wrong orders are
    drawn from three below and three above the right one, so the key can sit anywhere in the order."""
    offsets = [o for o in (-3, -2, -1, 1, 2, 3) if m + o >= 1]
    if gain == 4:
        offsets = [o for o in (-4, -3, -2, -1, 1, 2, 3) if m + o >= 1]
    rng.shuffle(offsets)
    chosen = sorted(offsets[:3])
    check(len(chosen) == 3 and 0 not in chosen, "three distinct wrong orders are needed")
    notes = {}
    distractors = []
    for offset in chosen:
        text = _statement(m + offset, a, b)
        distractors.append(text)
        notes[text] = _note_for(offset, gain)
    return _statement(m, a, b), distractors, notes


def _draw_common(rng):
    a, b = rng.choice(_INTERVALS)
    fname = rng.choice(_DATA_NAMES)
    return a, b, fname


def _build(rng, kind: str, *, explicit: bool):
    a, b, fname = _draw_common(rng)
    gain = 4 if kind == "beam" else 2
    c = rng.choice(_REACTION) if kind == "reaction" else None
    coef = rng.choice(_COEFFICIENTS) if kind == "variable" else None
    if explicit:
        alpha = rng.choice(_ALPHAS)
        x0 = Fraction(a + b, 2)
        k = _label_of_power(alpha)
        # The label found a second way: the largest k whose k-th derivative ~ |t|^(alpha - k) is square integrable.
        check(
            all((alpha > j - Fraction(1, 2)) == (j <= k) for j in range(0, 12)),
            "label of the power disagrees with the integrability test",
        )
        text, latex = _stem(kind, a, b, fname, k=None, alpha=alpha, x0=x0, c=c, coef=coef)
    else:
        k = rng.randint(0, 3)
        text, latex = _stem(kind, a, b, fname, k=k, alpha=None, x0=None, c=c, coef=coef)
    m = k + gain
    key, distractors, notes = _options(rng, m, a, b, gain=gain)
    if explicit:
        working = [
            f"The function {fname}: |x - x₀|^α is in Hᵏ exactly when α > k - 1/2, and α = {_alpha_text(alpha)}, "
            f"so {fname} is in {_h(k)} but not in {_h(k + 1)}.",
        ]
    else:
        working = [f"{fname} is in {_h(k)} but not in {_h(k + 1)}."]
    if kind == "beam":
        working.append(
            f"The equation holds in the sense of distributions, so the fourth derivative of u equals {fname}, and u is "
            f"in {_h(m)} but not in {_h(m + 1)} (four derivatives gained)."
        )
    elif kind == "laplace":
        working.append(
            f"The equation gives u'' = -{fname}, so u'' has the regularity of {fname}: u is in {_h(m)} but not in "
            f"{_h(m + 1)} (two derivatives gained)."
        )
    else:
        working.append(
            f"The equation gives u'' as {fname} plus terms that are more regular than {fname}, so u'' has the "
            f"regularity of {fname}: u is in {_h(m)} but not in {_h(m + 1)} (two derivatives gained)."
        )
    return dict(
        text=text,
        latex=latex,
        answer=key,
        answer_text=key,
        working=working,
        distractors=distractors,
        option_notes=notes,
        answer_kind="mcq",
    )


def _digits(n: int) -> str:
    return str(n).translate(_SUP)


def _easy(rng):
    return _build(rng, "laplace", explicit=False)


def _medium(rng):
    return _build(rng, rng.choice(("reaction", "variable")), explicit=False)


def _hard(rng):
    return _build(rng, rng.choice(("laplace", "reaction", "variable")), explicit=True)


def _exam(rng):
    return _build(rng, "beam", explicit=rng.random() < 0.5)


_BUILDERS = {"easy": _easy, "medium": _medium, "hard": _hard, "exam": _exam}


@generator("ucv1_weak_solution_regularity", level=21, presets={tier: dict(tier=tier) for tier in _TIERS})
def ucv1_weak_solution_regularity(rng, tier="easy"):
    check(tier in _BUILDERS, f"unknown tier {tier!r}")
    return _BUILDERS[tier](rng)
