"""Helpers shared by the `r3b_*` calculus builds (exponential and reciprocal integrals, exact definite integrals,
f'/f integrals, derivatives of ln, definite-integral properties, repeated and irreducible partial fractions and
trigonometric powers).

Not a generator module (the leading underscore keeps the package loader from importing it). It holds:

* three spellings of a SymPy expression: `canon` (what the grader reads, so `exp(2*x)` and `ln|x|`), `shown` (what a
  stem prints, so `e^(2x)`) and `tex` (LaTeX), and `say` to fill one wording pattern in both text and LaTeX;
* a numeric comparison of two expressions, used to drop a wrong option that is secretly the key, and numeric checks
  that an antiderivative differentiates back and that an exact integral matches quadrature;
* tagged distractors and part builders for the answer kinds these builds use (`expression`, `fraction`, `number`).

The grader reads an indefinite integral against `answer_text`, which carries the "+ C", so every antiderivative key
here is `answer = F` and `answer_text = "F + C"`; a wrong option that omits the constant is offered with
`const=False`.
"""

from __future__ import annotations

import functools
import math
import random
import re
from fractions import Fraction

import sympy
from sympy import Abs, Symbol, exp, log
from sympy.printing.str import StrPrinter

from ..answers import Distractor, exact_value
from ..core import (
    VerificationError,
    check,
    latex_answer,
    latex_from_inline,
    numeric_answer,
    rational_text,
    stated_precision_string,
)

x = Symbol("x")
C = Symbol("C")


class Redraw(VerificationError):
    """A draw that cannot give a sound question (too few distinct wrong answers, a degenerate value): draw again."""


def retry(build, tries: int = 60):
    """Wrap a builder so a draw that raises `Redraw` is redrawn, at most `tries` times and deterministically."""

    @functools.wraps(build)
    def run(rng, *args, **kwargs):
        for _ in range(tries):
            try:
                return build(rng, *args, **kwargs)
            except Redraw:
                continue
        raise Redraw(f"{build.__name__}: no acceptable draw in {tries} tries")

    return run


# -- three spellings of an expression -----------------------------------------------------------------------------


class _Printer(StrPrinter):
    """SymPy's string printer with `ln|u|` for a logarithm of an absolute value and either `exp(u)` or `e^(u)`."""

    def __init__(self, shown: bool, order=None):
        super().__init__({"order": order})
        self._shown = shown

    def _bars(self, inner):
        """`ln|u|`; with brackets, `ln(|u|)`, when u is not linear (the grader will not read `ln|x^2 + 1|`)."""
        linear = inner.is_polynomial(x) and sympy.Poly(inner, x).degree() <= 1
        text = self._print(inner)
        return f"ln|{text}|" if linear else f"ln(|{text}|)"

    def _print_log(self, expr):
        arg = expr.args[0]
        if isinstance(arg, Abs):
            return self._bars(arg.args[0])
        if isinstance(arg, sympy.Mul):
            bars = [a for a in arg.args if isinstance(a, Abs)]
            if len(bars) == 1:  # SymPy writes |2x| as 2|x|; print it as the learner would: ln|2x|
                return self._bars(sympy.Mul(*[a for a in arg.args if a is not bars[0]], bars[0].args[0]))
        return f"ln({self._print(arg)})"

    def _print_Abs(self, expr):
        return f"|{self._print(expr.args[0])}|"

    def _print_exp(self, expr):
        arg = expr.args[0]
        inner = self._print(arg)
        if not self._shown:
            return f"exp({inner})"
        return f"e^{inner}" if arg.is_Symbol or (arg.is_Integer and arg >= 0) else f"e^({inner})"

    def _print_Exp1(self, expr):
        return "e" if self._shown else "exp(1)"

    def _print_atan(self, expr):
        return f"arctan({self._print(expr.args[0])})"


def _clean(text: str, shown: bool) -> str:
    text = text.replace("**", "^")
    if shown:
        text = _TIMES.sub("", text)
        text = _DIGIT_TIMES.sub(" × ", text)
    return text


_TIMES = re.compile(r"(?<=[\dx)|])\*(?=[A-Za-z(])")
_DIGIT_TIMES = re.compile(r"(?<=[\d)])\*(?=\d)")


def _order(expr):
    """A number-only expression prints its terms with the constant last (`7e^2/2 - 7/2`); others keep SymPy's order."""
    return "rev-lex" if not expr.free_symbols else None


def canon(expr) -> str:
    """The text the grader reads: `3*exp(2*x)/2`, `ln|x + 1|`, `2^x/ln(2)`."""
    expr = sympy.sympify(expr)
    return _clean(_Printer(False, _order(expr)).doprint(expr), False)


def shown(expr) -> str:
    """The text a stem prints: `3e^(2x)/2`, `ln|x + 1|`."""
    expr = sympy.sympify(expr)
    return _clean(_Printer(True, _order(expr)).doprint(expr), True)


def tex(expr) -> str:
    """LaTeX of an expression, with \\ln and \\arctan."""
    return sympy.latex(sympy.sympify(expr), ln_notation=True, inv_trig_style="full")


def plus(n) -> str:
    """A signed term for a stem: 3 -> "+ 3", -3 -> "- 3", Rational(-1, 2) -> "- 1/2"."""
    n = sympy.nsimplify(n)
    return f"- {shown(-n)}" if n < 0 else f"+ {shown(n)}"


def math_(expr) -> str:
    """An expression as inline maths in a LaTeX pattern: \\(e^{2 x}\\)."""
    return rf"\({tex(expr)}\)"


def say(
    rng: random.Random, patterns, text_slots: dict, math_slots: dict, plain_slots: dict | None = None, suffix: str = ""
):
    """Choose one wording and fill it twice: `(text, latex)`.

    `suffix` is plain prose added to both. `text_slots` holds the strings for the plain stem; `math_slots` the same keys as LaTeX (already wrapped in
    `\\(...\\)` by `math_`). Plain-only slots (a word, a number that needs no maths) go in `plain_slots` and fill both."""
    pattern = patterns[rng.randrange(len(patterns))]
    plain = plain_slots or {}
    return pattern.format(**text_slots, **plain) + suffix, latex_from_inline(
        pattern.format(**math_slots, **plain) + suffix
    )


def _itg(f):
    return f if isinstance(f, Itg) else Itg(f)


def integral_text(f, lo=None, hi=None) -> str:
    """`∫ e^(2x) dx` or `∫ from 0 to 1 of e^(2x) dx` as plain text; `f` is an expression or an `Itg`."""
    f = _itg(f)
    body = f"({f.text})" if _needs_brackets(f.value) else f.text
    if lo is None:
        return f"∫ {body} dx"
    return f"∫ from {shown(sympy.sympify(lo))} to {shown(sympy.sympify(hi))} of {body} dx"


def integral_math(f, lo=None, hi=None) -> str:
    f = _itg(f)
    body = rf"\left({f.latex}\right)" if _needs_brackets(f.value) else f.latex
    if lo is None:
        return rf"\(\int {body}\,\mathrm{{d}}x\)"
    return rf"\(\int_{{{tex(sympy.sympify(lo))}}}^{{{tex(sympy.sympify(hi))}}} {body}\,\mathrm{{d}}x\)"


def _needs_brackets(expr) -> bool:
    return isinstance(sympy.sympify(expr), sympy.Add)


# -- numeric second routes ----------------------------------------------------------------------------------------

_POINTS = (0.37, 1.41, 2.63, 0.91, 3.17)


def _fn(expr):
    return sympy.lambdify(x, expr, modules=["math"])


@functools.lru_cache(maxsize=8192)
def _values(expr, points=_POINTS):
    """The expression at each point (None where undefined), kept: `lambdify` compiles code and costs ~2 ms, and the same
    candidate expressions come round for every question."""
    f = _fn(expr)
    out = []
    for p in points:
        try:
            value = complex(f(p))
        except (ValueError, ZeroDivisionError, OverflowError, TypeError):
            out.append(None)
            continue
        out.append(value.real if abs(value.imag) < 1e-12 else None)
    return tuple(out)


def same_function(a, b) -> bool:
    """Whether two expressions agree at five sample points. A point where exactly one is undefined is a difference."""
    va, vb = _values(sympy.sympify(a)), _values(sympy.sympify(b))
    for p, q in zip(va, vb):
        if (p is None) != (q is None):
            return False
        if p is not None and not math.isclose(p, q, rel_tol=1e-9, abs_tol=1e-9):
            return False
    return True


def check_antiderivative(F, f) -> None:
    """The second route to an antiderivative: F' must equal the integrand f at the sample points."""
    plain = sympy.sympify(F).replace(
        lambda e: isinstance(e, sympy.log) and isinstance(e.args[0], Abs), lambda e: log(e.args[0].args[0])
    )
    dF = sympy.diff(plain, x)
    check(same_function(dF, f), f"the antiderivative {F} does not differentiate to {f}")


def check_definite(value, f, lo, hi) -> None:
    """The second route to an exact definite integral: Simpson's rule on 2000 strips agrees with `value`."""
    g = _fn(sympy.sympify(f))
    a, b = float(sympy.N(sympy.sympify(lo), 30)), float(sympy.N(sympy.sympify(hi), 30))
    n = 2000
    h = (b - a) / n
    total = g(a) + g(b)
    for i in range(1, n):
        total += (4 if i % 2 else 2) * g(a + i * h)
    approx = total * h / 3
    exact = float(sympy.N(sympy.sympify(value), 30))
    check(math.isclose(approx, exact, rel_tol=1e-7, abs_tol=1e-9), f"quadrature {approx} disagrees with {exact}")


# -- distractors --------------------------------------------------------------------------------------------------


class Cand:
    """A wrong expression `expr` for the error `tag`; `const` says whether it carries a "+ C"."""

    __slots__ = ("expr", "tag", "const")

    def __init__(self, expr, tag: str, const: bool = True):
        self.expr, self.tag, self.const = expr, tag, const


def expr_text(expr, const: bool) -> str:
    expr = sympy.sympify(expr)
    if not const and not expr.free_symbols:
        expr = sympy.factor_terms(expr)  # 7(e^2 - 1)/2 reads better than -7/2 + 7e^2/2
    return canon(expr) + (" + C" if const else "")


def expr_dist(expr, tag: str, const: bool = True) -> Distractor:
    text = expr_text(expr, const)
    return Distractor(text, tag, text)


def pick_expr(key, key_const: bool, cands, count: int = 3, minimum: int = 2, rng=None) -> list[Distractor]:
    """The first `count` candidates that are a different answer from the key and from each other, at least `minimum`.

    With `rng` the candidates are taken in a seeded random order, so the options vary from question to question."""
    key = sympy.sympify(key)
    kept: list[Cand] = []
    cands = list(cands) if rng is None else rng.sample(list(cands), len(list(cands)))
    for cand in cands:
        if cand is None or cand.expr is None:
            continue
        expr = sympy.sympify(cand.expr)
        if expr.has(sympy.I, sympy.zoo, sympy.nan, sympy.oo):
            continue
        if cand.const == key_const and same_function(expr, key):
            continue
        if any(k.const == cand.const and same_function(k.expr, expr) for k in kept):
            continue
        kept.append(Cand(expr, cand.tag, cand.const))
        if len(kept) == count:
            break
    if len(kept) < minimum:
        raise Redraw("fewer than two distinct wrong expressions")
    return [expr_dist(k.expr, k.tag, k.const) for k in kept]


def key_fields(expr, const: bool = True) -> dict:
    """The answer fields of an expression key; the indefinite integral's key carries the "+ C" in `answer_text`."""
    expr = sympy.sympify(expr)
    shown_expr = sympy.factor_terms(expr) if not const and not expr.free_symbols else expr
    return dict(
        answer_kind="expression",
        answer=str(expr),
        answer_text=expr_text(expr, const),
        answer_latex=tex(shown_expr) + (" + C" if const else ""),
    )


def exact_fields(expr) -> dict:
    """The answer fields of an exact value such as `exp(2) - 1` or `5*ln(3)`."""
    return key_fields(expr, const=False)


# -- exact rationals ----------------------------------------------------------------------------------------------


def as_fraction(value) -> Fraction:
    value = sympy.Rational(value)
    return Fraction(int(value.p), int(value.q))


def frac_fields(value) -> dict:
    value = as_fraction(value)
    text = rational_text(value)
    return dict(answer_kind="fraction", answer=exact_value(value), answer_text=text, answer_latex=latex_answer(text))


def frac_dist(value, tag: str) -> Distractor:
    value = as_fraction(value)
    return Distractor(rational_text(value), tag, exact_value(value))


def int_fields(value) -> dict:
    """A whole-number `number` answer, graded to zero decimal places."""
    value = int(value)
    text = str(value)
    return dict(answer_kind="number", answer=numeric_answer(Fraction(value)), answer_text=text, answer_latex=text)


def int_dist(value, tag: str) -> Distractor:
    value = int(value)
    return Distractor(str(value), tag, numeric_answer(Fraction(value)))


def pick_values(key, cands, make_dist, count: int = 3, minimum: int = 2, rng=None) -> list[Distractor]:
    """The first `count` `(value, tag)` candidates that differ from `key` and from each other, as distractors."""
    seen = {sympy.nsimplify(key)}
    out = []
    cands = list(cands) if rng is None else rng.sample(list(cands), len(list(cands)))
    for value, tag in cands:
        if value is None:
            continue
        v = sympy.nsimplify(value)
        if v in seen or v.is_infinite:
            continue
        seen.add(v)
        out.append(make_dist(v, tag))
        if len(out) == count:
            break
    if len(out) < minimum:
        raise Redraw("fewer than two distinct wrong values")
    return out


def _is_rational(value) -> bool:
    return value is None or bool(sympy.nsimplify(value).is_Rational)


def pick_fractions(key, cands, count: int = 3, rng=None) -> list[Distractor]:
    """Wrong fractions; a candidate that is not a rational number (a power or a logarithm) is skipped."""
    return pick_values(key, [(v, t) for v, t in cands if _is_rational(v)], frac_dist, count, rng=rng)


def pick_ints(key, cands, count: int = 3, rng=None) -> list[Distractor]:
    return pick_values(
        key, [(v, t) for v, t in cands if v is None or sympy.Rational(v).q == 1], int_dist, count, rng=rng
    )


class Itg:
    """An integrand with its value and both spellings, for a stem that prints a form SymPy would simplify."""

    def __init__(self, value, text: str | None = None, latex: str | None = None):
        self.value = sympy.sympify(value)
        self.text = shown(self.value) if text is None else text
        self.latex = tex(self.value) if latex is None else latex


# -- parts of an exam chain ---------------------------------------------------------------------------------------


def part(prompt: str, prompt_latex: str, marks: int, working: list[str], fields: dict, distractors, **extra) -> dict:
    """One part of a multi-part question. `extra` carries `follow_through` and `sensible_range`."""
    out = dict(prompt=prompt, marks=marks, working=working, distractors=distractors, **fields)
    out["prompt_latex"] = prompt_latex if "\\(" not in prompt_latex else latex_from_inline(prompt_latex)
    out.update({k: v for k, v in extra.items() if v is not None})
    return out


def plain_prompt(text: str) -> tuple[str, str]:
    """A prompt with no maths in it: the same text as its own LaTeX source."""
    return text, text


def decimal_fields(value: Fraction, places: int) -> dict:
    text = stated_precision_string(value, places)
    return dict(answer_kind="number", answer=numeric_answer(value), answer_text=text, answer_latex=text)
