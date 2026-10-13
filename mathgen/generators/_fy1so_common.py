"""Shared helpers for the first-year series and ODE builds (`fy1so_*_build.py`).

Three things live here:

* **Markup.** A formula is written once, as TeX, and rendered twice: `plain()` turns it into the learner-facing text
  (Unicode, no TeX, no Python notation) and the TeX itself goes into the question's `latex` field. `say()` builds the
  ``(text, latex)`` pair for a whole sentence from prose and `M(...)` formulas, so the two never drift apart.
* **Exact values as typed text.** `expr_text()` writes a SymPy expression the way a learner types it (``2e^(3x) - 1``,
  ``√(x² + 4)``), and every form it writes is one the expression grader reads back.
* **Answer parts.** `Link` describes one answer (a fraction, a rounded decimal, an expression or a multiple-choice
  statement) with its tagged wrong answers, and `build_part()` / `build_single()` turn it into the dict a generator
  returns, with every wrong answer checked to be a different answer from the right one *as the grader sees it*.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from fractions import Fraction

import sympy

from ..answers import Distractor, exact_value
from ..core import (
    _escape_prose,
    check,
    latex_answer,
    numeric_answer,
    rational_text,
    round_half_up,
    stated_precision_string,
)

# -- markup ----------------------------------------------------------------------------------------------------


class M(str):
    """A TeX fragment inside a `say()` sentence."""


_SIMPLE = {
    r"\infty": "∞",
    r"\pi": "π",
    r"\to": "→",
    r"\cdot": "·",
    r"\ldots": "…",
    r"\dots": "…",
    r"\approx": "≈",
    r"\pm": "±",
    r"\ne": "≠",
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
_WORD = re.compile(r"[\w.π∞⁰¹²³⁴⁵⁶⁷⁸⁹]+")
_ATOM = re.compile(
    r"(?:\d+(?:\.\d+)?|[A-Za-zπ∞])(?:[⁰¹²³⁴⁵⁶⁷⁸⁹]|\^(?:\d+|[A-Za-z]|\([^()]*\)))?"
    r"|(?:ln|sin|cos|tan)\([^()]*\)|√(?:\d+|[A-Za-z]|\([^()]*\))"
    r"|[A-Za-z]_(?:[A-Za-z0-9]|\([^()]*\))"
    r"|\([^()]*\)(?:[⁰¹²³⁴⁵⁶⁷⁸⁹]|\^(?:\d+|[A-Za-z]))"
)
_PLAIN_EXPONENT = re.compile(r"\d+|[A-Za-z]")
_COMMAND = re.compile(r"\\([A-Za-z]+|.)", re.S)
_SUPERSCRIPT_DIGITS = dict(zip("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))


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
    """The learner-facing text for a TeX fragment. Only the TeX these builds use is understood; anything else
    raises, so a typo fails loudly."""
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
                sign = "√"
                if s[i : i + 1] == "[":  # an nth root: \sqrt[3]{n}
                    close = s.index("]", i)
                    sign = {"3": "∛", "4": "∜"}.get(s[i + 1 : close], f"{s[i + 1 : close]}√")
                    i = close + 1
                a, i = _arg(s, i)
                inner = plain(a)
                out.append(sign + (inner if _WORD.fullmatch(inner) else f"({inner})"))
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
                    word += "" if s[i:].lstrip(" ").startswith(("(", "\\left")) else " "
                out.append(word)
                if name[1:].isalpha() and name not in (r"\le", r"\ge"):  # TeX ignores the spaces after a command word
                    i += len(s[i:]) - len(s[i:].lstrip(" "))
            else:
                raise ValueError(f"unsupported TeX command {name!r} in {tex!r}")
        elif c in "^_":
            arg, i = _arg(s, i + 1)
            e = plain(arg)
            if c == "^" and re.fullmatch(r"\(\d\)", e):  # a derivative order: f^{(4)} -> f⁽⁴⁾
                out.append("⁽" + _SUPERSCRIPT_DIGITS[e[1]] + "⁾")
            elif c == "^" and e in _SUPERSCRIPT_DIGITS:
                out.append(_SUPERSCRIPT_DIGITS[e])
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


def tex_rational(value) -> str:
    """A rational as TeX: 3, -3, \\frac{3}{2}, -\\frac{3}{2}."""
    value = Fraction(value)
    if value.denominator == 1:
        return str(value.numerator)
    sign = "-" if value < 0 else ""
    return f"{sign}\\frac{{{abs(value.numerator)}}}{{{value.denominator}}}"


def tex_decimal(value, places: int) -> str:
    """A decimal at a fixed number of places as TeX."""
    return stated_precision_string(round_half_up(Fraction(value), places), places)


def coef_tex(c: int) -> str:
    """The coefficient before a variable: "" for 1, "-" for -1."""
    return "" if c == 1 else "-" if c == -1 else str(c)


def signed_tex(c, body: str) -> str:
    """`+ body` or `- body` for a term with coefficient c (c a Fraction or int; body what follows the coefficient)."""
    c = Fraction(c)
    mag = abs(c)
    lead = "" if mag == 1 and body else tex_rational(mag)
    return f"{'+' if c > 0 else '-'} {lead}{body}"


# -- exact values as typed text ----------------------------------------------------------------------------------


def _balanced(s: str, open_at: int) -> int:
    """Index of the bracket closing the one at `open_at`."""
    depth = 0
    for j in range(open_at, len(s)):
        depth += s[j] == "("
        depth -= s[j] == ")"
        if depth == 0:
            return j
    raise ValueError(f"unbalanced brackets in {s!r}")


def _rewrite_calls(s: str) -> str:
    """exp(a) -> e^(a), log(a) -> ln(a), sqrt(a) -> √(a), innermost first."""
    pattern = re.compile(r"(?<![A-Za-z])(exp|log|sqrt)\(")
    while True:
        calls = list(pattern.finditer(s))
        if not calls:
            return s
        m = calls[-1]
        start = m.end() - 1
        end = _balanced(s, start)
        arg = s[start + 1 : end]
        simple = re.fullmatch(r"-?[A-Za-z0-9]+", arg) is not None
        if m.group(1) == "exp":
            new = f"e^{arg}" if simple else f"e^({arg})"
        elif m.group(1) == "log":
            new = f"ln({arg})"
        else:
            new = f"√{arg}" if re.fullmatch(r"[A-Za-z0-9]+", arg) else f"√({arg})"
        s = s[: m.start()] + new + s[end + 1 :]


def expr_text(expr) -> str:
    """A SymPy expression the way a learner types it: ``2e^(3x) - 1``, ``x² e^(-x)``, ``√(x² + 4)``, ``1/(1 - x²)``.

    The text is also what the expression grader reads back, so every form written here is one it parses."""
    s = sympy.sstr(sympy.sympify(expr)).replace("**", "^")
    s = _rewrite_calls(s)
    s = re.sub(r"(?<![\w.])(\w)\^2(?![\d(])", r"\1²", s)
    s = re.sub(r"(?<![\w.])(\w)\^3(?![\d(])", r"\1³", s)
    s = re.sub(r"\)\^2(?![\d(])", ")²", s)
    s = re.sub(r"\)\^3(?![\d(])", ")³", s)
    # products: a number before a letter or bracket is a coefficient; otherwise a space reads as "times"
    s = re.sub(r"(?<=\d)\*(?=[A-Za-z(√])", "", s)
    s = re.sub(r"\*", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def expr_string(expr) -> str:
    """The SymPy string stored as an expression answer (`Question.answer`)."""
    return sympy.sstr(sympy.sympify(expr))


def expr_tex(expr) -> str:
    """TeX for an exact expression."""
    return sympy.latex(sympy.sympify(expr), ln_notation=True)


_PROBES = (Fraction(37, 100), Fraction(113, 100), Fraction(221, 100), Fraction(53, 10))


def same_function(a, b, symbols) -> bool:
    """Whether two expressions agree (to 1e-9) at several generic points: cheap where `simplify` is not. A point where
    exactly one is undefined or not real counts as a difference."""
    a, b = sympy.sympify(a), sympy.sympify(b)
    for p in _PROBES:
        sub = {}
        for j, s in enumerate(symbols):
            q = p + Fraction(j, 13)
            sub[s] = sympy.Rational(q.numerator, q.denominator)
        try:
            va, vb = complex(sympy.N(a.subs(sub), 20)), complex(sympy.N(b.subs(sub), 20))
        except (TypeError, ValueError, ZeroDivisionError):
            return False
        if abs(va - vb) > 1e-9 * max(1.0, abs(va), abs(vb)):
            return False
    return True


# -- answers -----------------------------------------------------------------------------------------------------


@dataclass
class Link:
    """One answer: its kind, its value, its prompt and its tagged wrong answers.

    ``kind`` is ``fraction`` (``value`` an int or Fraction), ``number`` (``value`` a Fraction or SymPy number, stated to
    ``places`` decimal places), ``expression`` (``value`` a SymPy expression in ``symbols``) or ``mcq`` (``value`` the
    right option's text, ``wrongs`` pairs of option text and misconception id).
    """

    prompt: tuple  # (text, latex)
    kind: str
    value: object
    marks: int = 1
    working: list = field(default_factory=list)
    wrongs: list = field(default_factory=list)  # (value or option text, misconception id), most telling first
    ft: str | None = None
    sensible: tuple | None = None
    places: int = 0
    symbols: tuple = ()
    minimum: int = 2  # fewest wrong answers this part must keep


def _frac(value) -> Fraction:
    value = sympy.nsimplify(value) if isinstance(value, sympy.Basic) and not value.is_Rational else value
    if isinstance(value, sympy.Basic):
        return Fraction(int(value.p), int(value.q))
    return Fraction(value)


def _shape(f: Fraction) -> str:
    return "integer" if f.denominator == 1 else "fraction"


def fraction_options(answer, wrongs, n: int = 4):
    """``(answer value, answer text, answer latex, [Distractor])`` for an exact rational answer.

    Wrong answers equal to the answer or to an earlier option are dropped. When the answer is the only option of its
    shape (the only whole number, or the only fraction) a same-shape wrong answer is brought forward so it cannot be
    picked without doing the working."""
    answer = _frac(answer)
    seen = [answer]
    pool: list[tuple[Fraction, str]] = []
    for value, misconception in wrongs:
        if value is None:
            continue
        try:
            v = _frac(value)
        except (TypeError, ValueError, ZeroDivisionError):
            continue
        if v in seen:
            continue
        seen.append(v)
        pool.append((v, misconception))
    chosen = pool[:n]
    if chosen and all(_shape(v) != _shape(answer) for v, _ in chosen):
        same = next((c for c in pool[n:] if _shape(c[0]) == _shape(answer)), None)
        if same is not None:
            chosen = chosen[:-1] + [same]
    options = [Distractor(rational_text(v), m, exact_value(v)) for v, m in chosen]
    latex = latex_answer(sympy.Rational(answer.numerator, answer.denominator))
    return exact_value(answer), rational_text(answer), latex, options


def number_options(answer, wrongs, places: int, n: int = 4):
    """``(answer float, answer text, [Distractor])`` for a decimal answer stated to `places` places. A wrong answer that
    rounds to the answer, or to an earlier option, is dropped; the rest differ from it by at least one unit in the last
    place, so the grader (which allows half a unit) refuses every one."""

    def rounded(v):
        exact = Fraction(str(sympy.N(sympy.sympify(v), 40))) if not isinstance(v, Fraction) else v
        return round_half_up(exact, places)

    ans = rounded(answer)
    seen = [ans]
    options: list[Distractor] = []
    for value, misconception in wrongs:
        if value is None:
            continue
        try:
            r = rounded(value)
        except (TypeError, ValueError, ZeroDivisionError):
            continue
        if r in seen:
            continue
        seen.append(r)
        options.append(Distractor(stated_precision_string(r, places), misconception, numeric_answer(r)))
        if len(options) == n:
            break
    return numeric_answer(ans), stated_precision_string(ans, places), options


def expression_options(answer, wrongs, symbols, n: int = 4):
    """``(answer string, answer text, answer latex, [Distractor])`` for an expression answer. A wrong answer equal as a
    function to the answer or to an earlier option is dropped."""
    answer = sympy.sympify(answer)
    seen = [answer]
    options: list[Distractor] = []
    for value, misconception in wrongs:
        if value is None:
            continue
        try:
            v = sympy.sympify(value)
        except (sympy.SympifyError, TypeError):
            continue
        if any(same_function(v, s, symbols) for s in seen):
            continue
        seen.append(v)
        options.append(Distractor(expr_text(v), misconception, expr_string(v)))
        if len(options) == n:
            break
    return expr_string(answer), expr_text(answer), expr_tex(answer), options


def build_answer(link: Link) -> dict:
    """The answer fields for one `Link`: answer, answer_text, answer_latex, answer_kind and distractors."""
    if link.kind == "fraction":
        answer, text, latex, options = fraction_options(link.value, link.wrongs)
    elif link.kind == "number":
        answer, text, options = number_options(link.value, link.wrongs, link.places)
        latex = text
    elif link.kind == "expression":
        answer, text, latex, options = expression_options(link.value, link.wrongs, link.symbols)
    elif link.kind == "mcq":
        text = link.value
        answer, latex = text, text
        seen = {text}
        options = []
        for option, misconception in link.wrongs:
            if option in seen:
                continue
            seen.add(option)
            options.append(Distractor(option, misconception))
    else:
        raise ValueError(f"unknown answer kind {link.kind!r}")
    check(len(options) >= link.minimum, f"fewer than {link.minimum} wrong answers for {link.prompt[0]!r}")
    return dict(answer=answer, answer_text=text, answer_latex=latex, answer_kind=link.kind, distractors=options)


def build_part(link: Link) -> dict:
    """One part of a multi-part question."""
    built = build_answer(link)
    part = dict(
        prompt=link.prompt[0],
        prompt_latex=link.prompt[1],
        marks=link.marks,
        working=list(link.working),
        **built,
    )
    if link.ft:
        part["follow_through"] = link.ft
    if link.sensible is not None:
        part["sensible_range"] = link.sensible
    return part


def build_single(prompt: tuple, link: Link) -> dict:
    """A single-answer question from a prompt ``(text, latex)`` and its `Link`."""
    built = build_answer(link)
    text, latex = prompt
    return dict(text=text, latex=latex, working=list(link.working), **built)


def n_to_places(value, places: int) -> Fraction:
    """`value` (a Fraction or SymPy number) rounded half up to `places` decimal places."""
    exact = value if isinstance(value, Fraction) else Fraction(str(sympy.N(sympy.sympify(value), 40)))
    return round_half_up(exact, places)


def rounds_clear(value, places: int, margin: float = 0.2) -> bool:
    """Whether `value` sits clear of a rounding boundary at `places` decimal places: at least `margin` of a unit in the
    last place from the halfway point, so working to a few more figures reaches the same rounded answer."""
    scaled = float(sympy.N(sympy.sympify(value), 30)) * 10**places
    return abs((scaled % 1) - 0.5) >= margin
