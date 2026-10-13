"""Everything a page shows for one question, in one call.

    from mathgen import engine

    q = engine.make_for_skill(skill, seed=7)
    page = engine.present(skill, q)

``page`` is a JSON-able dict. Every string a learner reads comes as a pair:

- ``text``: plain Unicode, written as on paper (``x^2``, ``2x``, ``√21``; never Python's ``x**2``). Always set.
  Show it as is when the page renders no maths, and use it for speech, search and plain-text exports.
- ``tex``: the same content with each piece of mathematics wrapped in ``\\(...\\)`` and the prose left as
  ordinary text, or ``None`` when there is no mathematics to typeset. This is the input KaTeX's auto-render
  extension and MathJax read by default. Configure them with the ``\\(...\\)`` delimiter only, never ``$``:
  prose can contain a dollar sign ("$4.50"). Because prose stays outside the maths, a long prompt wraps on a
  phone screen the way ordinary text does.

Keys:

- ``prompt``: the question.
- ``figure``: ``None``, or ``{"svg", "alt", "spec"}``: the drawing as standalone SVG (`mathgen.render`), its
  accessible description, and the spec it was drawn from, for a platform that draws figures itself.
- ``options``: for a multiple-choice question, the options in the order `mathgen.core.mcq_options` gives, so
  grading the chosen index with `engine.grade` agrees with what was shown. Absent otherwise.
- ``answer``: the expected answer.
- ``working``: the worked solution, one pair per step (often ``[]``: not every generator writes one).
- ``parts``: a multi-part question's parts, each ``{"label", "prompt", "answer", "options"?, "working"}``,
  with the shared information in ``prompt`` (the stem). Absent for a single-answer question.

The question itself is not changed.
"""

from __future__ import annotations

import re
from typing import Any

from mathgen.core import latex_answer, mcq_options, paper_notation

# Undo `core.latex_prompt`'s escaping of prose inside \text{...}. Longest escapes first.
_UNESCAPE = (
    (re.compile(r"\\textbackslash\{\}"), lambda m: "\\"),
    (re.compile(r"\\textasciitilde\{\}"), lambda m: "~"),
    (re.compile(r"\\textasciicircum\{\}"), lambda m: "^"),
    (re.compile(r"\\([$%&#_{}])"), lambda m: m.group(1)),
)
_WORD = re.compile(r"[A-Za-z]{3,}")
_SPACING = r"(?:\s|\\[ ,;:!>]|\\q?quad\b)"
_EDGE_SPACE_LEAD = re.compile(rf"^{_SPACING}*")
_EDGE_SPACE_TRAIL = re.compile(rf"{_SPACING}+$")


def _segments(latex: str) -> list[tuple[str, str]] | None:
    r"""Split a `latex_prompt`-shaped string into ("text", prose) and ("math", latex) runs, or None when it is
    not that shape (an unbalanced brace, a nested group inside a prose \text{...}). Only a \text{...} at the top
    level is prose; one inside a maths group (``\frac{2.5\,\text{kN}}{A}``) stays part of the maths."""
    out: list[tuple[str, str]] = []
    math_start, i, depth = 0, 0, 0
    while i < len(latex):
        if depth == 0 and latex.startswith("\\text{", i):
            if i > math_start:
                out.append(("math", latex[math_start:i]))
            j, inner = i + 6, 1
            while j < len(latex) and inner > 0:
                c = latex[j]
                if c == "\\":
                    j += 2
                    continue
                inner += {"{": 1, "}": -1}.get(c, 0)
                j += 1
            if inner != 0:
                return None
            body = latex[i + 6 : j - 1]
            if re.search(r"(?<!\\)[{}]", body):
                return None
            for pattern, repl in _UNESCAPE:
                body = pattern.sub(repl, body)
            out.append(("text", body))
            i = math_start = j
            continue
        c = latex[i]
        if c == "\\":
            i += 2
            continue
        depth += {"{": 1, "}": -1}.get(c, 0)
        if depth < 0:
            return None
        i += 1
    if depth != 0:
        return None
    if math_start < len(latex):
        out.append(("math", latex[math_start:]))
    return out


def _math(latex: str) -> str:
    latex = latex.strip()
    return f"\\({latex}\\)" if latex else ""


def prompt_tex(text: str, latex: str) -> str | None:
    r"""The ``tex`` form of a prompt from its stored ``text`` and ``latex``: prose as text, maths in \(...\)."""
    if not latex or latex.startswith("plain:"):
        return None
    if latex == text:
        # `latex_prompt` returns a string unchanged when it is all prose it could not lift maths from, and when it
        # is pure maths with no word in it ("56 + 71").
        return None if _WORD.search(text) else _math(latex)
    segs = _segments(latex)
    if segs is None:
        return None
    if not any(kind == "text" for kind, _ in segs):
        return _math(latex)
    out = []
    for kind, s in segs:
        if kind == "text":
            out.append(s)
            continue
        # Spacing at the edge of a maths run (``\ ``, ``\,``, ``\quad``) becomes an ordinary space outside it,
        # so the prose reads naturally and no delimiter swallows half a command.
        lead, body, trail = _EDGE_SPACE_LEAD.match(s).group(), s, ""
        body = body[len(lead) :]
        m = _EDGE_SPACE_TRAIL.search(body)
        if m:
            trail, body = m.group(), body[: m.start()]
        out.append((" " if lead else "") + _math(body) + (" " if trail else ""))
    tex = re.sub(r" {2,}", " ", "".join(out)).strip()
    return tex if "\\(" in tex else None


def value_tex(text: str, latex: str | None = None) -> str | None:
    r"""The ``tex`` form of an answer or an option: ``\(...\)`` around its LaTeX, or None when it is not maths.

    ``latex`` is the generator's own LaTeX when it wrote one (``answer_latex``); otherwise it is derived from
    ``text`` with `mathgen.core.latex_answer`."""
    if not latex:
        try:
            latex = latex_answer(text)
        except Exception:  # an answer the converter cannot read is shown as text
            return None
    if not latex or latex.startswith("plain:"):
        return None
    return _math(latex)


def _pair(text: str, tex: str | None) -> dict:
    return {"text": paper_notation(text), "tex": tex}


def _answer_side(item, *, mcq: bool) -> dict:
    out: dict[str, Any] = {
        "answer": _pair(item.answer_text, value_tex(item.answer_text, getattr(item, "answer_latex", ""))),
        "working": [_pair(step, None) for step in (getattr(item, "working", None) or [])],
    }
    if mcq:
        out["options"] = [_pair(o, value_tex(o)) for o in mcq_options(item)]
    return out


def present(question, *, mcq: bool = False, theme: str = "light", draw_figure: bool = True) -> dict:
    """The presentation dict for ``question`` (see the module docstring).

    ``mcq``: include ``options`` (`engine.present` sets it from the skill's answer kind).
    ``theme``: passed to the SVG renderer. ``draw_figure=False`` leaves ``figure.svg`` as ``None`` for a
    platform that draws from the spec itself.
    """
    page: dict[str, Any] = {
        "prompt": _pair(question.text, prompt_tex(question.text, question.latex)),
        "figure": None,
    }
    if question.figure:
        svg = None
        if draw_figure:
            from mathgen.render import figure_svg

            svg = figure_svg(question.figure, theme=theme)
        page["figure"] = {"svg": svg, "alt": question.figure["alt"], "spec": question.figure}
    parts = getattr(question, "parts", None)
    if parts:
        stem = question.stem if question.stem is not None else question.text
        page["prompt"] = _pair(stem, prompt_tex(stem, _stem_latex(stem)))
        page["parts"] = []
        for p in parts:
            part_mcq = mcq or getattr(p, "answer_kind", None) == "mcq"
            page["parts"].append(
                {"label": p.label, "prompt": _pair(p.text, prompt_tex(p.text, p.prompt_latex or _stem_latex(p.text)))}
                | _answer_side(p, mcq=part_mcq)
            )
        page |= _answer_side(question, mcq=False)
        return page
    page |= _answer_side(question, mcq=mcq)
    return page


def _stem_latex(text: str) -> str:
    from mathgen.core import latex_from_inline, latex_prompt

    return latex_prompt(latex_from_inline(text))
