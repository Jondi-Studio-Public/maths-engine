"""Detector: ``answer_latex`` carries a 3+-letter run or "sqrt" that is NOT part of a backslash-prefixed LaTeX
command token, and the whole string is not ``plain:``-prefixed. The naive whole-string "contains no backslash"
check is not enough by itself: `latex_answer`'s relation branch can join two plain-word
clauses with a ``,\\;`` separator ("left = 1,\\; right = 2") -- a genuine backslash appears in the string
while "left"/"right" still render as italic letter products. Stripping backslash-command tokens first and
checking what is left closes that hole.

Swept over every active skill's declared sources, at 2 seeds x every
difficulty tier the topic declares -- the same shape as `scripts/render_skill_questions.py`'s per-
source, per-tier loop, but reading `answer_latex` instead of the prompt.

Every generator's `answer_latex` is routed through `mathgen.core.latex_answer`, and this sweep expects zero hits.
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
#: this corpus nests deeper). `\mathcal{}`/`\mathtt{}` are deliberately NOT
#: in this list: both are genuine sympy-collision artefacts in this corpus
#: (`\mathcal{Categori}`, `\mathtt{\text{<class '...'>}}`) and neither
#: renders as a legible word either way.
_TEXT_COMMAND_GROUP_RE = re.compile(r"\\(?:text|mathrm|mathbf|operatorname)\{[^{}]*\}")
# Stripped before the letter-run check so it is not conflated with that defect.
_SUBSCRIPT_GROUP_RE = re.compile(r"[_^]\{[^{}]*\}")
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
    stripped = _TEXT_COMMAND_GROUP_RE.sub("", latex)
    stripped = _SUBSCRIPT_GROUP_RE.sub("", stripped)
    stripped = _COMMAND_TOKEN_RE.sub("", stripped)
    return bool(_LETTER_RUN_RE.search(stripped)) or "sqrt" in stripped.lower()


def _iter_answer_latex():
    """Yield (skill_id, topic, difficulty, params, seed, answer_latex) for
    every source/tier/seed combination of every active skill."""
    for skill in _active_skills():
        for src in skill.sources:
            params = _tuplify(src.params)
            for difficulty in _difficulties_for(src.topic, src.difficulty):
                for seed in range(SEEDS):
                    q = make(src.topic, seed=seed, difficulty=difficulty, **params)
                    yield skill.id, src.topic, difficulty, dict(src.params or {}), seed, q.answer_latex


def test_no_active_skill_ships_mangled_answer_latex():
    mangled = {}
    for skill_id, topic, difficulty, params, seed, latex in _iter_answer_latex():
        if is_mangled(latex):
            mangled.setdefault(skill_id, (topic, latex))
    assert not mangled, (
        f"{len(mangled)} active skills ship a hand-rolled, non-LaTeX "
        f"answer_latex that KaTeX will render as italic maths letters: "
        f"{sorted(mangled.items())}"
    )


@pytest.mark.parametrize(
    "latex,expected",
    [
        ("8\\sqrt{7}", False),  # real LaTeX, backslash command
        ("\\frac{3}{4}", False),  # real LaTeX
        ("plain:Obtuse", False),  # already the fallback shape
        ("Obtuse", True),  # hand-rolled plain word
        ("8*sqrt(7)", True),  # hand-rolled surd text (the original defect)
        ("left = 1,\\; right = 2", True),  # backslash present, but from the
        # joiner only -- "left"/"right" are
        # still unescaped letter runs
        ("R = 73,\\; \\alpha = 41.1", False),  # genuine math: R is one letter,
        # alpha is a real backslash command
        ("x = 3", False),  # single-letter relation, fine
        ("", False),  # empty falls back to plain text in the front-end math renderer
        ("E = 100\\,\\text{GPa}", False),  # unit word correctly wrapped in \text{}
        ("\\mathrm{False}", False),  # boolean wrapped in \mathrm{}
        ("\\sqrt{9} = sqrt(9)", True),  # real LaTeX prefix, plain-text tail
        ("M_{max} = 20\\,\\text{kN}\\cdot\\text{m}", False),  # subscript label, out of scope
    ],
)
def test_is_mangled_detector_examples(latex, expected):
    assert is_mangled(latex) is expected
