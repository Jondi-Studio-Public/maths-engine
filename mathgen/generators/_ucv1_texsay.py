"""Question text and LaTeX written together for the Laplace, line integral and triple integral builds.

`mathgen.core.latex_prompt` guesses at the TeX of a plain sentence, and for exponentials (``e^(-4t)``), limits and
the triple integral sign it guesses badly. A builder here says its sentence once, as prose strings and `Math(...)`
fragments written in the same plain Unicode the learner reads, and gets back the plain text (unchanged) and a
LaTeX version with the maths set as maths: ``say("Find ", Math("F(s) = 3e^(-4t)"), ".")``.

`tex()` understands only what these builders write: ``^(...)`` and ``^n`` powers, ``²`` and ``³``, ``√n``,
``≤ ≥ ∫ ∭ ρ φ ∇ ·``, the function names ``sin cos``, literal braces, and a lone ``i`` or ``j`` as a unit vector.
Anything else passes through as it is.
"""

from __future__ import annotations

import re

from ..core import _escape_prose


class Math(str):
    """A fragment of plain Unicode maths inside a sentence."""


_SUPERSCRIPT = {"²": "2", "³": "3"}


def tex(plain: str) -> str:
    s = plain.replace("{", r"\{").replace("}", r"\}")
    s = re.sub(r"∫C\b", r"∫_C", s)
    s = re.sub(r"(?<![A-Za-z\\])([ij])(?![A-Za-z(])", r"\\mathbf{\1}", s)
    s = re.sub(r"\^\(([^()]*)\)", lambda m: "^{" + m.group(1) + "}", s)
    s = re.sub(r"\^(-?\d+)", r"^{\1}", s)
    s = re.sub(r"\^([A-Za-z])", r"^{\1}", s)
    s = re.sub("([²³])", lambda m: "^{" + _SUPERSCRIPT[m.group(1)] + "}", s)
    s = re.sub(r"√(\d+)", r"\\sqrt{\1}", s)
    s = re.sub(r"(?<![A-Za-z\\])(sin|cos)(?![A-Za-z])", r"\\\1", s)
    for symbol, command in (
        ("≤", r"\le "),
        ("≥", r"\ge "),
        ("∭", r"\iiint "),
        ("∫", r"\int "),
        ("∇", r"\nabla "),
        ("·", r"\cdot "),
        ("ρ", r"\rho "),
        ("φ", r"\varphi "),
        ("⁻¹", "^{-1}"),
    ):
        s = s.replace(symbol, command)
    return s


def say(*parts) -> tuple[str, str]:
    """``(text, latex)`` for prose strings and `Math` fragments."""
    text: list[str] = []
    latex: list[str] = []
    for part in parts:
        if isinstance(part, Math):
            text.append(str(part))
            latex.append(tex(str(part)) + "{}")  # {} keeps neighbouring words from running into one letter run
        elif part:
            text.append(part)
            latex.append(r"\text{" + _escape_prose(part) + "}")
    return "".join(text), "".join(latex)


_MARKED = re.compile(r"«(.*?)»", re.S)


def say_marked(template: str) -> tuple[str, str]:
    """`say()` for a sentence in which each maths fragment sits between « and »."""
    parts = _MARKED.split(template)
    return say(*(Math(p) if i % 2 else p for i, p in enumerate(parts)))
