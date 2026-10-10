"""Modelled on `tests/test_p9r_answer_latex_mangled.py` -- same detector shape,
same sweep shape -- but reading `Question.latex` instead of `answer_latex`,
and with one addition: `\\begin{...}`/`\\end{...}` are stripped BEFORE the
letter-run check. `cases` and `matrix` (as in `\\begin{cases}`,
`\\begin{matrix}`) are LaTeX ENVIRONMENT names, not italic English words --
a generator that authors its own `\\begin{cases}...\\end{cases}` prompt
latex is real, hand-written LaTeX, and without stripping those environment
names first this detector would false-positive on it. Do NOT edit
`tests/test_p9r_answer_latex_mangled.py` to add the same stripping: its own
`answer_latex` sweep is green today with a latent, undisturbed false
positive on environment names it happens never to hit, and this file
does not touch it.

`mathgen/core.py`'s `latex_prompt` plus one hook in `make()` route every
generator's plain-text `text` through it whenever a generator didn't author
its own distinct `latex`, so no skill ships `latex == text` verbatim. This
sweep expects zero such skills.
"""

import re

import pytest

from mathgen.skills import load_skills
from mathgen import make
from scripts.regrade_sweep import _difficulties_for, _tuplify

#: `\text{...}`/`\mathrm{...}`/`\mathbf{...}`/`\operatorname{...}` render
#: their argument upright and legible -- a real word inside one of these is
#: NOT the italic-letters defect, so its contents are dropped before the
#: letter-run check runs at all (one level of `{}` nesting only; none of
#: this corpus nests deeper).
#: `latex_prompt`'s own prose-escaping macros for `^ ~ \` take an EMPTY
#: `{}` argument (`\textasciicircum{}`, not `\textasciicircum{x}`) -- a
#: second, inner level of braces that `_TEXT_COMMAND_GROUP_RE`'s one-level
#: `[^{}]*` cannot cross. Left unstripped, `\text{Solve 2\textasciicircum{}x}`
#: would NOT be recognised as one text group (the nested `{}` breaks the
#: match), and the surviving "Solve" would false-positive as mangled. Strip
#: these zero-content macros first so the text-group regex sees a clean
#: single level of nesting.
_ESCAPE_MACRO_RE = re.compile(r"\\text(?:asciicircum|asciitilde|backslash)\{\}")
_TEXT_COMMAND_GROUP_RE = re.compile(r"\\(?:text|mathrm|mathbf|operatorname)\{[^{}]*\}")
#: A subscript/superscript group's contents render as a short label
#: ("M_{max}", "V_{max}") -- KaTeX italicises it the same as any other
#: unescaped letters, but that is a separate, pre-existing typographic
#: convention this detector does not touch.
_SUBSCRIPT_GROUP_RE = re.compile(r"[_^]\{[^{}]*\}")
#: LaTeX environment names ("cases", "matrix", ...) are not italic English
#: words -- strip the whole `\begin{...}`/`\end{...}` token (name included)
#: before the letter-run check, or a genuine `\begin{cases}` false-positives.
_ENVIRONMENT_RE = re.compile(r"\\(?:begin|end)\{[^{}]*\}")
# The upper bound of 4 keeps a genuinely bare English word ("TOTAL", "CHANGE", "ALWAYS", "FINAL",
# "EXTERNAL", "FORTNIGHTLY" -- all 5+ letters, all measured in the same sweep) flaggable. Residual,
# accepted gap: a hand-rolled 3-4-letter all-caps English word ("RATE", "TRUE") shipped bare, with
# no `\text{}` and no `plain:` prefix, would slip through undetected -- not observed anywhere in the
# corpus (English prose here is never written in solid capitals), and not tightenable by
# adjacency/punctuation since a bare word is followed by ordinary punctuation exactly like a label
# is.
_CAPS_LABEL_RE = re.compile(r"(?<![A-Za-z])[A-Z]{3,4}(?![A-Za-z])")
_COMMAND_TOKEN_RE = re.compile(r"\\[A-Za-z]+")
_LETTER_RUN_RE = re.compile(r"[A-Za-z]{3,}")
SEEDS = 2


def _active_skills():
    _, skills = load_skills()
    return [s for s in skills if s.status == "active"]


def is_mangled(latex: str) -> bool:
    """Whether `latex` is hand-rolled plain text rather than real LaTeX or a
    `plain:`-prefixed fallback string."""
    if not latex or latex.startswith("plain:"):
        return False
    stripped = _ENVIRONMENT_RE.sub("", latex)
    stripped = _ESCAPE_MACRO_RE.sub("", stripped)
    stripped = _TEXT_COMMAND_GROUP_RE.sub("", stripped)
    stripped = _SUBSCRIPT_GROUP_RE.sub("", stripped)
    stripped = _CAPS_LABEL_RE.sub("", stripped)
    stripped = _COMMAND_TOKEN_RE.sub("", stripped)
    return bool(_LETTER_RUN_RE.search(stripped)) or "sqrt" in stripped.lower()


def _iter_prompt_latex():
    """Yield (skill_id, topic, difficulty, params, seed, latex) for every
    source/tier/seed combination of every active skill."""
    for skill in _active_skills():
        for src in skill.sources:
            params = _tuplify(src.params)
            for difficulty in _difficulties_for(src.topic, src.difficulty):
                for seed in range(SEEDS):
                    q = make(src.topic, seed=seed, difficulty=difficulty, **params)
                    yield skill.id, src.topic, difficulty, dict(src.params or {}), seed, q.latex


def test_no_active_skill_ships_mangled_prompt_latex():
    mangled = {}
    for skill_id, topic, difficulty, params, seed, latex in _iter_prompt_latex():
        if is_mangled(latex):
            mangled.setdefault(skill_id, (topic, latex))
    assert not mangled, (
        f"{len(mangled)} active skills ship a hand-rolled, non-LaTeX prompt "
        f"latex that KaTeX will render as italic maths letters: "
        f"{sorted(mangled.items())}"
    )


@pytest.mark.parametrize(
    "latex,expected",
    [
        ("8\\sqrt{7}", False),  # real LaTeX, backslash command
        ("\\frac{3}{4}", False),  # real LaTeX
        ("plain:Obtuse", False),  # already the fallback shape
        ("Obtuse", True),  # hand-rolled plain word
        ("8*sqrt(7)", True),  # hand-rolled surd text
        ("x = 3", False),  # single-letter relation, fine
        ("", False),  # empty falls back to plain text in the front-end math renderer
        ("E = 100\\,\\text{GPa}", False),  # unit word correctly wrapped in \text{}
        ("\\mathrm{False}", False),  # boolean wrapped in \mathrm{}
        ("\\sqrt{9} = sqrt(9)", True),  # real LaTeX prefix, plain-text tail
        ("M_{max} = 20\\,\\text{kN}\\cdot\\text{m}", False),  # subscript label, out of scope
        ("\\begin{cases} x = 1 \\\\ y = 2 \\end{cases}", False),  # environment name, not a word
        ("\\begin{matrix} 1 & 0 \\\\ 0 & 1 \\end{matrix}", False),
        ("\\text{Write }3.59 \\times 10^{8}\\text{ in engineering notation}", False),
        ("\\text{Solve 2\\textasciicircum{}x = 16}", False),  # latex_prompt's own
        # escape for an unresolved caret -- the empty {} argument must not
        # break the surrounding \text{} group match (see _ESCAPE_MACRO_RE).
        ("I = PRT/100", False),  # implied multiplication of single-letter
        # formula variables (Principal, Rate, Time) -- solid ALL-CAPS run,
        # see _CAPS_LABEL_RE.
        ("\\text{Midpoints of quadrilateral }ABCD: \\text{ find }S", False),
        # concatenated polygon vertex-label run, see _CAPS_LABEL_RE.
        ("TOTAL = 5", True),  # 5-letter all-caps word is above _CAPS_LABEL_RE's
        # length-4 bound and must still flag.
    ],
)
def test_is_mangled_detector_examples(latex, expected):
    assert is_mangled(latex) is expected
