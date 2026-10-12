"""Shared pieces for the free-form proof generators (`proof_*_build.py`).

A free-form proof question is marked against a rubric, not against one expected answer, so a generator
returns a prompt plus a per-item marking dict (`Question.answer`):

    {"start": None, "target": None, "reference_lines": [...], "key_steps": [...],
     "traps": [...], "checks": []}

`reference_lines` is one worked proof in plain sentences, `key_steps` says what earns each rubric criterion
(`correctness`, `logical_completeness`, `definitions_assumptions`, `communication`) and `traps` names the
usual wrong turns, each with the question a marker should ask about it. `checks` is empty: nothing here can
be decided by a computer algebra system alone, so a person or a model marks the written argument.

Underscore-prefixed, so the generator registry does not import it as a generator module.
"""

from __future__ import annotations

import re

from ..core import VerificationError, check

try:  # a bounded stand-in for `while True:`; older cores do not carry it
    from ..core import attempts
except ImportError:  # pragma: no cover - exercised only by a core without `attempts`

    def attempts(limit: int = 1000, what: str = "an acceptable draw"):
        """Yield 0 .. limit - 1, then raise `VerificationError`."""
        yield from range(limit)
        raise VerificationError(f"no acceptable draw for {what} in {limit} tries")


CRITERIA = ("correctness", "logical_completeness", "definitions_assumptions", "communication")

_TEXT_ESCAPES = {
    "\\": r"\textbackslash{}",
    "{": r"\{",
    "}": r"\}",
    "%": r"\%",
    "&": r"\&",
    "#": r"\#",
    "_": r"\_",
    "$": r"\$",
    "^": r"\textasciicircum{}",
    "~": r"\textasciitilde{}",
}


def tex_text(s: str) -> str:
    """`s` as upright text inside a LaTeX formula (`\\text{...}`), every special character escaped."""
    return r"\text{" + "".join(_TEXT_ESCAPES.get(ch, ch) for ch in s) + "}"


_FRAC_RE = re.compile(r"\\frac\{([^{}]*)\}\{([^{}]*)\}")
_SQRT_RE = re.compile(r"\\sqrt\{([^{}]*)\}")
_POW_RE = re.compile(r"\^\{([^{}]*)\}")
_SUB_RE = re.compile(r"_\{([^{}]*)\}")
_SYMBOLS = (
    (r"\left", ""),
    (r"\right", ""),
    (r"\angle", "∠"),
    (r"\circ", "°"),
    (r"\ln", "ln"),
    (r"\log", "log"),
    (r"\sin", "sin"),
    (r"\cos", "cos"),
    (r"\int", "∫"),
    (r"\pi", "π"),
    (r"\cdot", "·"),
    (r"\times", "×"),
    (r"\leq", "≤"),
    (r"\geq", "≥"),
    (r"\le", "≤"),
    (r"\ge", "≥"),
    (r"\neq", "≠"),
    (r"\ne", "≠"),
    (r"\parallel", "∥"),
    (r"\perp", "⊥"),
    (r"\cap", "∩"),
    (r"\cup", "∪"),
    (r"\mid", "|"),
    (r"\cong", "≅"),
    (r"\sim", "~"),
    (r"\equiv", "≡"),
    (r"\Rightarrow", "⇒"),
    (r"\,", " "),
    (r"\;", " "),
    (r"\!", ""),
    (r"\{", "{"),
    (r"\}", "}"),
)


_TIDY = (
    (re.compile(r"\(1/-(\d+)\)"), r"(-1/\1)"),
    (re.compile(r"(?<![\w.])-1(?=[A-Za-z])"), "-"),
    (re.compile(r"\+ -(?=[\d(A-Za-z])"), "- "),
    (re.compile(r"(?<=[\w)]) - -(?=[\d(A-Za-z])"), " + "),
    (re.compile(r"(?<![\w.^/(-])1(?=[xy]\b)"), ""),
    (re.compile(r"\^\((-?)1x\)"), r"^(\1x)"),
    (re.compile(r"\^\((\w)\)"), r"^\1"),
    (re.compile(r"\(1\)(?=[a-z])"), ""),
    (re.compile(r"(?<![\w.^(/-])1(?=e\^)"), ""),
)


def tidy(s: str) -> str:
    """Plain text with the usual machine-spelling slips removed: `(1/-2)` as `(-1/2)`, `-1y` as `-y`, `1x` as `x`."""
    for pat, rep in _TIDY:
        s = pat.sub(rep, s)
    return s


def _group(part: str) -> str:
    """`part` bracketed unless it is a single token (a number or a name), so 1/x reads as 1/x, not (1)/(x)."""
    return part if re.fullmatch(r"-?[A-Za-z0-9.]+", part) else f"({part})"


def plain_from_tex(tex: str) -> str:
    """A readable plain-text spelling of a simple LaTeX formula (fractions in brackets, `^` for powers)."""
    s = tex
    for _ in range(4):
        s = _FRAC_RE.sub(lambda m: f"{_group(m.group(1))}/{_group(m.group(2))}", s)
        s = _SQRT_RE.sub(lambda m: f"√({m.group(1)})", s)
    s = _POW_RE.sub(lambda m: "^" + (m.group(1) if len(m.group(1)) == 1 else f"({m.group(1)})"), s)
    s = _SUB_RE.sub(lambda m: "_" + (m.group(1) if len(m.group(1)) == 1 else f"({m.group(1)})"), s)
    for src, dst in _SYMBOLS:
        s = s.replace(src, dst)
    s = s.replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", s).strip()


#: Symbols that a plain-text prompt spells with one Unicode character but a formula renderer needs as maths:
#: `∠ABC`, `35°` and the relation signs. `Prompt.t` pulls them out of prose and sends them to the LaTeX side as
#: real maths, with angle letters spaced so that no run of letters reads as a word.
_MATH_TOKEN_RE = re.compile(r"∠[A-Z]{3}|\d+(?:\.\d+)?°|[∥⊥×÷≠≤≥≅∩∪π]")
_MATH_TOKEN_TEX = {
    "∥": r"\parallel ",
    "⊥": r"\perp ",
    "×": r"\times ",
    "÷": r"\div ",
    "≠": r"\neq ",
    "≤": r"\leq ",
    "≥": r"\geq ",
    "≅": r"\cong ",
    "∩": r"\cap ",
    "∪": r"\cup ",
    "π": r"\pi ",
}


def _token_tex(token: str) -> str:
    if token.startswith("∠"):
        return r"\angle " + " ".join(token[1:])
    if token.endswith("°"):
        return token[:-1] + r"^{\circ}"
    return _MATH_TOKEN_TEX[token]


def prose_latex(prose: str) -> str:
    """LaTeX for a run of prose: upright text, with angles, degrees and relation signs as real maths."""
    out: list[str] = []
    pos = 0
    for match in _MATH_TOKEN_RE.finditer(prose):
        if match.start() > pos:
            out.append(tex_text(prose[pos : match.start()]))
        out.append(_token_tex(match.group(0)))
        pos = match.end()
    if pos < len(prose):
        out.append(tex_text(prose[pos:]))
    return "".join(out)


class Fm:
    """A formula held in both spellings: LaTeX for the renderer and plain text for the prompt's `text`.

    ``Fm("x^{2}") + " + 1"`` joins formulas (a bare string is used as is in both spellings).
    """

    def __init__(self, tex: str, plain: str | None = None) -> None:
        self.tex = tex
        self.plain = plain if plain is not None else plain_from_tex(tex)

    @staticmethod
    def of(x: Fm | str) -> Fm:
        return x if isinstance(x, Fm) else Fm(x, x)

    def __add__(self, other: Fm | str) -> Fm:
        o = Fm.of(other)
        return Fm(self.tex + o.tex, self.plain + o.plain)

    def __radd__(self, other: str) -> Fm:
        o = Fm.of(other)
        return Fm(o.tex + self.tex, o.plain + self.plain)

    def __str__(self) -> str:
        return self.plain


def frac_tex(num: int, den: int) -> str:
    """A reduced fraction as LaTeX (`\\frac{a}{b}`, a whole number when it divides)."""
    from fractions import Fraction

    f = Fraction(num, den)
    if f.denominator == 1:
        return str(f.numerator)
    sign = "-" if f < 0 else ""
    return f"{sign}\\frac{{{abs(f.numerator)}}}{{{f.denominator}}}"


def frac_plain(num: int, den: int) -> str:
    from fractions import Fraction

    f = Fraction(num, den)
    return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"


_AFTER_LETTER_TEXT = re.compile(r"([A-Za-z])(\\text\{)")


class Prompt:
    """A question built from prose and formulas, rendered twice: plain `text` and real-LaTeX `latex`."""

    def __init__(self) -> None:
        self._text: list[str] = []
        self._latex: list[str] = []

    def t(self, prose: str) -> Prompt:
        self._text.append(prose)
        self._latex.append(prose_latex(prose))
        return self

    def m(self, tex: Fm | str, plain: str | None = None) -> Prompt:
        if isinstance(tex, Fm):
            self._text.append(tex.plain)
            self._latex.append(tex.tex)
        else:
            self._text.append(plain if plain is not None else plain_from_tex(tex))
            self._latex.append(tex)
        return self

    @property
    def text(self) -> str:
        return tidy("".join(self._text))

    @property
    def latex(self) -> str:
        # A thin space between a formula that ends in a letter and the text that follows keeps the letter from
        # running into the next formula once the text is taken away (`dx` then `u` must not read as `dxu`).
        return _AFTER_LETTER_TEXT.sub(r"\1\\,\2", "".join(self._latex))


def marking(*, reference: list[str], steps: list[tuple[str, str]], traps: list[tuple[str, str]]) -> dict:
    """The per-item marking dict. `steps` are (criterion id, what earns it); `traps` are (name, question)."""
    check(len(reference) >= 2, "a reference proof needs at least two lines")
    ids = {cid for cid, _ in steps}
    check(ids <= set(CRITERIA), f"unknown criterion id in {ids - set(CRITERIA)}")
    check({"correctness", "logical_completeness"} <= ids, "key steps must cover correctness and logical_completeness")
    check(len(traps) >= 2, "an item needs at least two named traps")
    check(len({name for name, _ in traps}) == len(traps), "trap names must be distinct")
    return {
        "start": None,
        "target": None,
        "reference_lines": [tidy(line) for line in reference],
        "key_steps": [{"criterion_id": cid, "description": tidy(desc)} for cid, desc in steps],
        "traps": [{"name": name, "question": tidy(question)} for name, question in traps],
        "checks": [],
    }


def finish(prompt: Prompt, claim: str, mark: dict) -> dict:
    """The dict a free-form proof generator returns. `claim` is the statement the learner proves."""
    mark = dict(mark, reference_lines=[tidy(line) for line in mark["reference_lines"]])
    return dict(text=prompt.text, latex=prompt.latex, answer=mark, answer_text=tidy(claim))


#: Closing sentence variants. Every proof prompt ends with one of these, so the learner always knows the
#: argument is marked for its reasons and not only for its last line.
TAILS = (
    "Write your proof as a numbered list of steps and give the reason for each step.",
    "Set out your proof in short steps, saying why each step follows.",
    "Give a complete argument, one step per line, with the reason for each step beside it.",
    "Show every step of your reasoning, and give a reason for each one.",
    "Explain each line of your argument, so that someone else could check it.",
)
