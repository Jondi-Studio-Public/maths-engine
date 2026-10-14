"""Shared helpers for the second-year multivariable integration builds (cylindrical, spherical, coordinates, double
integrals over general regions, reversed order, Jacobians).

* `etext` / `latex_of` spell an exact SymPy value (``8*pi/3``) the way a learner types it.
* `wrong_answers` turns tagged wrong values into `Distractor`s, drops any the real grader would accept, and checks
  that none repeats the answer or another option.
* `gauss_nested` is an independent numerical integral: Gauss-Legendre in each variable, with limits that may depend
  on the variables outside them. Generators use it as the second route to every key.
* `say_marked` is `_ucv1_texsay.say_marked` with the symbols these topics need (θ, π, √(...), differentials).
* `exam_part` builds one part of an exam chain.
"""

from __future__ import annotations

import re
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import sympy

from ..answers import Distractor, exact_value
from ..core import check, latex_answer, rational_text
from ..grading import grade
from . import _ucv1_texsay as _texsay
from ._ucv1_calc_markup import tex_value, value_string, value_text

PI = sympy.pi

# ---------------------------------------------------------------------------------------------------------------
# spelling


def display(value) -> str:
    """The way a learner reads an exact value: 3π/2, 2√3, -64√3·π/3 + 128π/3. The expression grader reads it back."""
    value = sympy.sympify(value)
    if value.free_symbols:
        return etext(value)  # a function of phi, theta or z keeps its typed spelling
    text = value_text(value)
    return re.sub(r"(√\d+)(?=π|[a-z(])", r"\1·", text)


def etext(value) -> str:
    """The typed spelling of an exact value: ``8*pi/3``, ``5/3``, ``2*sqrt(3)``."""
    value = sympy.nsimplify(value, rational=False) if isinstance(value, float) else sympy.sympify(value)
    return str(sympy.expand(value)).replace("**", "^")


def is_rational(value) -> bool:
    return bool(sympy.sympify(value).is_Rational)


def to_fraction(value) -> Fraction:
    value = sympy.sympify(value)
    return Fraction(int(value.p), int(value.q))


def latex_of(value) -> str:
    return latex_answer(sympy.sympify(value))


# ---------------------------------------------------------------------------------------------------------------
# question text with LaTeX

_DIFFERENTIAL = re.compile(r"(?<![A-Za-z\\])d(θ|φ|ρ|[A-Za-z])(?![A-Za-z])")


def mtex(plain: str) -> str:
    s = _texsay.tex(plain)
    s = re.sub(r"√\(([^()]*)\)", r"\\sqrt{\1}", s)
    s = _DIFFERENTIAL.sub(lambda m: r"\,\mathrm{d}" + m.group(1), s)
    for symbol, command in (
        ("θ", r"\theta "),
        ("π", r"\pi "),
        ("φ", r"\varphi "),
        ("ρ", r"\rho "),
        ("δ", r"\delta "),
        ("∬", r"\iint "),
        ("∂", r"\partial "),
        ("→", r"\to "),
        ("×", r"\times "),
        ("°", r"^{\circ}"),
    ):
        s = s.replace(symbol, command)
    return s


_MARKED = re.compile(r"«(.*?)»", re.S)


def say_marked(template: str) -> tuple[str, str]:
    """``(text, latex)`` for a sentence whose maths sits between « and »."""
    text: list[str] = []
    latex: list[str] = []
    from ..core import _escape_prose

    for i, part in enumerate(_MARKED.split(template)):
        if not part:
            continue
        if i % 2:
            text.append(part)
            latex.append(mtex(part) + "{}")
        else:
            text.append(part)
            latex.append(r"\text{" + _escape_prose(part) + "}")
    return "".join(text), "".join(latex)


def stem(template: str) -> dict:
    text, latex = say_marked(template)
    return dict(text=text, latex=latex)


def prompt(template: str) -> dict:
    text, latex = say_marked(template)
    return dict(prompt=text, prompt_latex=latex)


# ---------------------------------------------------------------------------------------------------------------
# distractors


def _grades(kind: str, answer, answer_text: str, option: str) -> bool:
    target = SimpleNamespace(answer=answer, answer_text=answer_text, answer_alt=None, answer_form=None)
    return grade(kind, target, option).correct


def expression_answer(value):
    """(kind, answer, answer text) for a single question graded as an expression, rational or not."""
    value = sympy.sympify(value)
    return "expression", etext(value), display(value)


def machine_answer(value):
    """(kind, answer value, answer text) for an exact value: a fraction for a rational, else an expression."""
    value = sympy.sympify(value)
    if value.is_Rational:
        f = to_fraction(value)
        return "fraction", exact_value(f), rational_text(f)
    return "expression", str(value), display(value)


def has_surd(value) -> bool:
    return any(a.exp.is_Rational and a.exp.q == 2 and a.base.is_number for a in sympy.sympify(value).atoms(sympy.Pow))


_SAMPLES = (0.37, 0.91, 1.13, 0.58)


def _same_function(a, b) -> bool:
    """Whether two expressions with free symbols agree at several sample points."""
    symbols = sorted((a.free_symbols | b.free_symbols), key=str)
    for k in range(3):
        point = {s: _SAMPLES[(i + k) % len(_SAMPLES)] for i, s in enumerate(symbols)}
        try:
            if abs(complex(sympy.N((a - b).subs(point)))) > 1e-9:
                return False
        except TypeError:
            return False
    return True


def _pi_degree(value) -> int:
    """Highest power of pi in the terms of `value`."""
    value = sympy.expand(sympy.sympify(value))
    return max((int(term.as_powers_dict().get(sympy.pi, 0)) for term in sympy.Add.make_args(value)), default=0)


def wrong_answers(
    value, candidates, limit: int = 4, kind: str | None = None, max_ratio: float = 25, allow_negative: bool = False
) -> list[Distractor]:
    """Distinct wrong options from ``[(misconception id, value), ...]``, in order, at most `limit`.

    An option is dropped when it equals the answer (to 1e-9), repeats another option's spelling, or is accepted by
    the real grader for this answer. For a rational answer every option is spelled as a fraction or integer; for
    an answer with pi, as an expression."""
    value = sympy.sympify(value)
    answer_kind, machine, answer_text = machine_answer(value)
    kind = kind or answer_kind
    if kind == "expression":
        machine = etext(value)
    symbolic = bool(value.free_symbols)
    target = None if symbolic else float(value)
    seen = {answer_text}
    out: list[Distractor] = []
    off_form: list[Distractor] = []
    for tag, wrong in candidates:
        if wrong is None:
            continue
        wrong = sympy.sympify(wrong)
        if wrong.has(sympy.zoo, sympy.nan, sympy.oo, sympy.asin, sympy.acos, sympy.atan, sympy.log) and not symbolic:
            continue
        if symbolic:
            if wrong == 0 or _same_function(wrong, value) or wrong.has(sympy.nan, sympy.zoo):
                continue
            same_form = True
        else:
            if not wrong.is_real:
                continue
            if abs(float(wrong) - target) < 1e-9:
                continue
            if wrong == 0 or wrong.as_numer_denom()[1] > 1000:
                continue
            same_form = _pi_degree(wrong) == _pi_degree(value) and (has_surd(value) or not has_surd(wrong))
            if abs(float(wrong)) > max_ratio * abs(target) or (
                float(wrong) < 0 < target and not tag.endswith("swapped") and not allow_negative
            ):
                continue
        if kind == "fraction":
            if not wrong.is_Rational:
                continue
            f = to_fraction(wrong)
            if f.denominator > 5000:
                continue
            text, val = rational_text(f), exact_value(f)
        else:
            text, val = display(wrong), etext(wrong)
        if text in seen:
            continue
        # a value this close could be read as the answer; the tests grade every option of every seed regardless
        if not symbolic and abs(float(wrong) - target) <= 1e-4 * max(1.0, abs(target)):
            continue
        seen.add(text)
        (out if same_form else off_form).append(Distractor(text, tag, val))
    out = out + off_form[:1]  # at most one option of a different form (pi squared, no pi): that alone is no giveaway
    if has_surd(value):
        # a root in the answer alone would give it away: offer an option that has one first
        out.sort(key=lambda d: not has_surd(sympy.sympify(str(d.value).replace("^", "**"))))
        if sum(has_surd(sympy.sympify(str(d.value).replace("^", "**"))) for d in out[:limit]) < 1:
            return []
    return out[:limit]


# ---------------------------------------------------------------------------------------------------------------
# an independent numerical integral

_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(24)


def gauss_nested(f, *limits) -> float:
    """Integral of ``f(v1, v2, ...)`` where limit k is ``(lo, hi)`` for variable k, each a number or a function of
    the variables before it (given as broadcastable arrays). Twenty-four-point Gauss-Legendre in each variable."""
    n, count = len(limits), len(_NODES)
    values: list[np.ndarray] = []
    weight = 1.0
    for k, (lo, hi) in enumerate(limits):
        shape = [1] * n
        shape[k] = count
        lo_v = np.asarray(lo(*values) if callable(lo) else lo, dtype=float)
        hi_v = np.asarray(hi(*values) if callable(hi) else hi, dtype=float)
        mid, half = (lo_v + hi_v) / 2, (hi_v - lo_v) / 2
        values.append(mid + half * _NODES.reshape(shape))
        weight = weight * _WEIGHTS.reshape(shape) * half
    return float(np.sum(f(*values) * weight))


def close(exact, numeric: float, what: str, rel: float = 1e-8) -> None:
    exact = float(sympy.sympify(exact))
    check(abs(numeric - exact) <= rel * max(1.0, abs(exact)), f"{what}: numerical {numeric!r} vs exact {exact!r}")


# ---------------------------------------------------------------------------------------------------------------
# exam parts


def exam_part(
    prompt_template: str,
    value,
    marks: int,
    working,
    candidates,
    *,
    follow_through: str | None = None,
    sensible_range=None,
    limit: int = 4,
    allow_negative: bool = False,
) -> dict | None:
    """One part of an exam chain, or None when it cannot be given two wrong options."""
    value = sympy.sympify(value)
    kind, machine, answer_text = machine_answer(value)
    wrong = wrong_answers(value, candidates, limit, allow_negative=allow_negative)
    if len(wrong) < 2:
        return None
    part = dict(
        **prompt(prompt_template),
        answer_kind=kind,
        answer=machine,
        answer_text=answer_text,
        answer_latex=latex_of(value),
        marks=marks,
        working=list(working),
        distractors=wrong,
    )
    if follow_through:
        part["follow_through"] = follow_through
    if sensible_range and kind == "fraction":
        part["sensible_range"] = sensible_range
    return part


# ---------------------------------------------------------------------------------------------------------------
# exact polynomial integration


def integrate_poly(expr, var, lo, hi):
    """Exact integral of the polynomial `expr` in `var` between `lo` and `hi` (which may involve other symbols),
    term by term."""
    expr = sympy.expand(expr)
    if not expr.has(var):
        return sympy.expand(expr * (hi - lo))
    poly = sympy.Poly(expr, var)
    return sympy.expand(sum(c * (hi ** (n + 1) - lo ** (n + 1)) / (n + 1) for (n,), c in poly.terms()))


def grid_integral(f, inside, x_range, y_range, n: int = 1400) -> float:
    """Midpoint-rule integral of `f(x, y)` over the points where `inside(x, y)` holds (NumPy arrays)."""
    (x0, x1), (y0, y1) = x_range, y_range
    hx, hy = (x1 - x0) / n, (y1 - y0) / n
    xs = x0 + (np.arange(n) + 0.5) * hx
    ys = y0 + (np.arange(n) + 0.5) * hy
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    mask = inside(gx, gy)
    return float(np.sum(np.where(mask, f(gx, gy) + 0 * gx, 0.0)) * hx * hy)


# ---------------------------------------------------------------------------------------------------------------
# exact values with e, ln and pi, spelled the way `_ucv1_calc_markup` does (what the expression grader reads)


def say_value(value) -> str:
    """An exact value for a worked line: 8π/3, 2√3, (e - 1)/2 (no `*` operators)."""
    return value_text(tidy(value))


def tidy(value):
    """The value with a common factor pulled out, so (e - 1)/2 rather than -1/2 + e/2."""
    value = sympy.sympify(value)
    return value if value.is_Rational else sympy.factor_terms(value)


def exact_answer(value):
    value = tidy(value)
    return "expression", value_string(value), value_text(value)


def exact_wrong(value, cands, limit=4, allow_negative=False, max_ratio=40.0, kind="expression"):
    """Distinct wrong options (never accepted by the grader) from [(tag, value), ...]."""
    value = sympy.sympify(value)
    ans_kind, machine, answer_text = exact_answer(value)
    target = float(value)
    seen = {answer_text}
    out = []
    off_form = []
    negatives = 0
    for tag, wrong in cands:
        wrong = sympy.sympify(wrong)
        if not wrong.is_real or wrong.has(sympy.zoo, sympy.nan, sympy.oo):
            continue
        w = float(wrong)
        if abs(w - target) < 1e-9 or abs(w) < 1e-12 or abs(w) > max_ratio * abs(target):
            continue
        if w < 0 < target and (not allow_negative or negatives >= 1):
            continue
        if kind == "fraction":
            if not wrong.is_Rational:
                continue
            f = to_fraction(wrong)
            if f.denominator > 5000:
                continue
            text, val = rational_text(f), exact_value(f)
        else:
            text = val = value_text(tidy(wrong))
            if len(text) > 40 or "/(" in text:
                continue
            if wrong.is_Rational and wrong.q > 1000:
                continue
        if text in seen:
            continue
        if abs(w - target) <= 1e-4 * max(1.0, abs(target)):
            continue
        negatives += w < 0
        seen.add(text)
        same_form = all(wrong.has(f) == value.has(f) for f in (sympy.log, sympy.pi, sympy.exp))
        (out if same_form else off_form).append(Distractor(text, tag, val))
    out = out + off_form[:1]  # at most one option of a different form: that alone is no giveaway
    return out[:limit]


def exact_part(template, value, marks, working, cands, follow_through=None, sensible_range=None, allow_negative=False):
    value = sympy.sympify(value)
    kind = "fraction" if value.is_Rational else "expression"
    wrong = exact_wrong(value, cands, 4, allow_negative=allow_negative, kind=kind)
    if len(wrong) < 2:
        return None
    if kind == "fraction":
        f = to_fraction(value)
        machine, answer_text = exact_value(f), rational_text(f)
    else:
        _, machine, answer_text = exact_answer(value)
    part = dict(
        **prompt(template),
        answer_kind=kind,
        answer=machine,
        answer_text=answer_text,
        answer_latex=tex_value(tidy(value)),
        marks=marks,
        working=list(working),
        distractors=wrong,
    )
    if follow_through:
        part["follow_through"] = follow_through
    if sensible_range and kind == "fraction":
        part["sensible_range"] = sensible_range
    return part
