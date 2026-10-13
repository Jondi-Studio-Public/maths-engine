"""Shared maths markup for the first-year calculus builds (`ucv1_lhopital_build`, `ucv1_improper_integral_build`,
`ucv1_series_convergence_build`).

A formula is written once, as TeX, and rendered twice: `plain()` turns it into the learner-facing text (Unicode,
no TeX, no Python notation) and the TeX itself goes into the question's `latex` field. `say()` builds the pair for a
whole sentence from prose and `M(...)` formulas, so the two never drift apart. `value_text()` writes an exact
value (a rational, a multiple of pi, a logarithm, a power of e, a surd) the way a learner types it.

Only the TeX this module's own generators use is understood; anything else raises, so a typo fails loudly.
"""

from __future__ import annotations

import re
from fractions import Fraction

import sympy

from ..core import _escape_prose


class M(str):
    """A TeX fragment inside a `say()` sentence."""


_SIMPLE = {
    r"\infty": "∞",
    r"\pi": "π",
    r"\to": "→",
    r"\cdot": "·",
    r"\ln": "ln",
    r"\sin": "sin",
    r"\cos": "cos",
    r"\tan": "tan",
    r"\le": "≤",
    r"\ge": "≥",
    r"\,": " ",
    r"\;": " ",
    r"\left": "",
    r"\right": "",
    r"\!": "",
}
_FUNCTIONS = (r"\ln", r"\sin", r"\cos", r"\tan")
_WORD = re.compile(r"[\w.π∞²³]+")
_ATOM = re.compile(
    r"(?:\d+(?:\.\d+)?|[A-Za-zπ∞])(?:[²³]|\^(?:\d+|[A-Za-z]|\([^()]*\)))?"
    r"|(?:ln|sin|cos|tan)\([^()]*\)|√(?:\d+|[A-Za-z]|\([^()]*\))"
    r"|[A-Za-z]_(?:[A-Za-z0-9]|\([^()]*\))"
)
_PLAIN_EXPONENT = re.compile(r"\d+|[A-Za-z]")
_COMMAND = re.compile(r"\\([A-Za-z]+|.)", re.S)


def _group(s: str, i: int) -> tuple[str, int]:
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1 : j], j + 1
    raise ValueError(f"unbalanced braces in {s!r}")


def _arg(s: str, i: int) -> tuple[str, int]:
    while i < len(s) and s[i] == " ":
        i += 1
    if i >= len(s):
        raise ValueError(f"missing argument in {s!r}")
    if s[i] == "{":
        return _group(s, i)
    if s[i] == "\\":
        match = _COMMAND.match(s, i)
        return match.group(0), match.end()
    return s[i], i + 1


def _enclosed(e: str) -> bool:
    if not (e.startswith("(") and e.endswith(")")):
        return False
    depth = 0
    for k, ch in enumerate(e):
        depth += ch == "("
        depth -= ch == ")"
        if depth == 0 and k < len(e) - 1:
            return False
    return True


def _wrap(e: str) -> str:
    return e if _ATOM.fullmatch(e) or _enclosed(e) else f"({e})"


def _bound(e: str) -> str:
    return e if _WORD.fullmatch(e) else f"({e})"


def plain(tex: str) -> str:
    """The learner-facing text for a TeX fragment."""
    out: list[str] = []
    i = 0
    s = tex
    while i < len(s):
        c = s[i]
        if c == "\\":
            match = _COMMAND.match(s, i)
            name = "\\" + match.group(1)
            i = match.end()
            if name == r"\frac":
                a, i = _arg(s, i)
                b, i = _arg(s, i)
                out.append(f"{_wrap(plain(a))}/{_wrap(plain(b))}")
            elif name == r"\text":
                a, i = _arg(s, i)
                out.append(a)
            elif name == r"\sqrt":
                a, i = _arg(s, i)
                inner = plain(a)
                out.append("√" + (inner if _WORD.fullmatch(inner) else f"({inner})"))
            elif name in (r"\int", r"\sum", r"\lim"):
                sub = sup = None
                for _ in range(2):
                    j = i
                    while j < len(s) and s[j] == " ":
                        j += 1
                    if s[j : j + 1] == "_":
                        arg, i = _arg(s, j + 1)
                        sub = plain(arg)
                    elif s[j : j + 1] == "^":
                        arg, i = _arg(s, j + 1)
                        sup = plain(arg)
                if name == r"\lim":
                    out.append(f"lim({sub}) ")
                else:
                    sym = "∫" if name == r"\int" else "Σ"
                    out.append(sym + (f"_{_bound(sub)}" if sub else "") + (f"^{_bound(sup)}" if sup else "") + " ")
            elif name in _SIMPLE:
                word = _SIMPLE[name]
                if name in _FUNCTIONS:
                    # "\\ln x" keeps its space; "\\ln(x)" and "\\ln\\left(x\\right)" do not need one.
                    word += "" if s[i:].lstrip(" ").startswith(("(", "\\left")) else " "
                out.append(word)
                if name[1:].isalpha() and name not in (r"\le", r"\ge"):  # TeX ignores the spaces after a command word
                    i += len(s[i:]) - len(s[i:].lstrip(" "))
            else:
                raise ValueError(f"unsupported TeX command {name!r} in {tex!r}")
        elif c in "^_":
            arg, i = _arg(s, i + 1)
            e = plain(arg)
            if c == "^" and e in ("2", "3"):
                out.append("²" if e == "2" else "³")
            elif c == "^" and e in ("+", "-"):
                out.append("⁺" if e == "+" else "⁻")
            else:
                out.append(c + (e if _PLAIN_EXPONENT.fullmatch(e) else f"({e})"))
        elif c in "{}":
            i += 1
        else:
            out.append(c)
            i += 1
    return re.sub(r" {2,}", " ", "".join(out)).strip()


def say(*parts) -> tuple[str, str]:
    """``(text, latex)`` for a sentence made of prose strings and `M` formulas."""
    text: list[str] = []
    tex: list[str] = []
    for part in parts:
        if isinstance(part, M):
            text.append(plain(str(part)))
            tex.append(str(part))
        elif part:
            text.append(part)
            tex.append(r"\text{" + _escape_prose(part) + "}")
    joined = "".join(text)
    return re.sub(r" {2,}", " ", joined).strip(), "".join(tex)


# -- TeX snippets ----------------------------------------------------------------------------------------------


def coef(c: int) -> str:
    """The coefficient before a variable: "" for 1, "-" for -1."""
    return "" if c == 1 else "-" if c == -1 else str(c)


def signed(c: int, body: str) -> str:
    """`+ body` or `- body` for a term with integer coefficient c (body is what follows the coefficient)."""
    mag = abs(c)
    return f"{'+' if c > 0 else '-'} {'' if mag == 1 and body else mag}{body}"


def tex_rational(value) -> str:
    """A rational as TeX: 3, -3, \\frac{3}{2}, -\\frac{3}{2}."""
    value = Fraction(value)
    if value.denominator == 1:
        return str(value.numerator)
    sign = "-" if value < 0 else ""
    return f"{sign}\\frac{{{abs(value.numerator)}}}{{{value.denominator}}}"


def tex_power(p) -> str:
    """The exponent of x as TeX braces content: 3, 3/2, -2."""
    p = Fraction(p)
    return str(p.numerator) if p.denominator == 1 else f"{p.numerator}/{p.denominator}"


# -- exact values ----------------------------------------------------------------------------------------------


def value_text(expr) -> str:
    """An exact value the way a learner types it: ``3/2``, ``π/4``, ``ln(3)/2``, ``e^(3/2)``, ``2√3``.

    The text is also what the expression grader reads back, so every form here is one it parses."""
    expr = sympy.sympify(expr)
    s = sympy.sstr(expr).replace("**", "^")
    s = re.sub(r"exp\(([^()]*)\)", lambda m: _exp_text(m.group(1)), s)
    s = re.sub(r"\bE\b", "e", s)
    s = s.replace("log(", "ln(").replace("pi", "π")
    s = re.sub(r"sqrt\((\d+)\)", r"√\1", s)
    s = re.sub(r"(?<=\d)\*(?=[a-zπ√(])", "", s)
    s = s.replace("*", "·")
    return s


def _exp_text(arg: str) -> str:
    if re.fullmatch(r"\d+", arg):
        return "e" if arg == "1" else f"e^{arg}"
    return f"e^({arg})"


def value_string(expr) -> str:
    """The SymPy string stored as an expression answer (`Question.answer`)."""
    return sympy.sstr(sympy.sympify(expr))


def tex_value(expr) -> str:
    """TeX for an exact value, for `answer_latex`."""
    return sympy.latex(sympy.sympify(expr), ln_notation=True)


def poly_tex(coeffs, var: str = "x") -> str:
    """TeX for an integer polynomial given by its coefficients from the highest power down: ``x^{3} - 2x + 5``."""
    degree = len(coeffs) - 1
    pieces: list[str] = []
    for k, c in enumerate(coeffs):
        power = degree - k
        if c == 0:
            continue
        mag = abs(c)
        if power == 0:
            body = str(mag)
        else:
            body = ("" if mag == 1 else str(mag)) + (var if power == 1 else f"{var}^{{{power}}}")
        if not pieces:
            pieces.append(("-" if c < 0 else "") + body)
        else:
            pieces.append(f"{'-' if c < 0 else '+'} {body}")
    return " ".join(pieces) if pieces else "0"
