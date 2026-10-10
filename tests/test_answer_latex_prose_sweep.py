"""Sweeps every active skill's sources and fails any question whose `answer_latex` is plain prose, a bare
connective or otherwise not real LaTeX.

Walks every active skill's own declared `sources` at a handful of seeds each, and fails any question whose
`answer_latex` trips one of three rules:

  - **prose**: contains no backslash (not real LaTeX, and not
    `latex_answer`'s `plain:`-prefixed deliberate fallback), AND contains a
    run of 2+ ASCII letters KaTeX would italicise, or one of
    `core._BARE_KEYWORD_RE`'s bare English connectives ("or", "and", ...) --
    a plain-English `answer_latex` ("x = -1.69 or x = 1.19") renders as
    italicised, spaceless nonsense.

  - **dollar**: a non-`plain:` `answer_latex` containing an
    unescaped `$` anywhere. `$` opens/closes math mode in TeX, so a second,
    unpaired `$` is a parse error -- measured directly against this repo's
    own KaTeX: ``katex.renderToString("$542.00", {throwOnError: false})``
    returns ``<span class="katex-error" ... style="color:#cc0000">``, i.e.
    the learner's own answer rendered in red error styling on the feedback
    card.

  - **percent**: a non-`plain:` `answer_latex` containing an
    unescaped `%` anywhere. `%` starts a LaTeX comment, silently truncating
    everything after it -- no error, no red, just gone: measured directly,
    ``katex.renderToString("80%", {throwOnError: false})`` renders as `80`,
    and ``"1/2, 0.5, 50%"`` renders as `1/2, 0.5, 50` -- the percent sign,
    the entire point of a fraction/decimal/percent equivalence question,
    vanishes with no indication anything went wrong.

  "Unescaped" means the character is not immediately preceded by a
  backslash (`\\$`, `\\%` are the real LaTeX escapes for a literal dollar
  sign or percent sign and render correctly). A `plain:`-prefixed
  `answer_latex` is exempt from the dollar/percent rules because
  the front-end renderer's `plain:` branch renders the string as ordinary text, never
  through KaTeX at all, so a `$` or `%` inside it is inert.

Parametrized per skill id so xdist (`-n 8`) distributes the ~471 active skills across workers.
"""

from __future__ import annotations

import re

import pytest

from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import PRESETS, _BARE_KEYWORD_RE

SEEDS = 8
_DIFFICULTY_TIERS = ("easy", "medium", "hard")
#: A run of 2+ ASCII letters -- shorter than `core._LETTER_RUN_RE`'s 3+
#: (that one is tuned for `latex_prompt`'s prose-lifting, which must leave a
#: bare 1-2 letter variable alone; this one exists purely to catch KaTeX
#: italicising a whole English word or connective, so "or"/"dx" both count).
_LETTER_RUN_RE = re.compile(r"[A-Za-z]{2,}")

#: A `$` or `%` not immediately preceded by a backslash -- the real LaTeX
#: escape (`\$`, `\%`) renders a literal character and is not the defect;
#: an un-escaped one is a KaTeX parse error (`$`) or a silent truncation
#: (`%`). See the module docstring for the measured KaTeX evidence.
_UNESCAPED_DOLLAR_RE = re.compile(r"(?<!\\)\$")
_UNESCAPED_PERCENT_RE = re.compile(r"(?<!\\)%")


def _tuplify(params: dict) -> dict:
    """Mirrors `mathgen.skills._tuplify` -- copied, not imported, matching
    `scripts/regrade_sweep.py`'s own stated convention for this exact
    helper."""
    return {k: (tuple(v) if isinstance(v, list) else v) for k, v in (params or {}).items()}


def _difficulties_for(topic: str, explicit: str | None) -> list:
    """Mirrors `scripts/regrade_sweep.py`'s `_difficulties_for`."""
    if explicit:
        return [explicit]
    supported = [d for d in _DIFFICULTY_TIERS if d in PRESETS.get(topic, {})]
    return supported or [None]


def _is_prose_leak(answer_latex) -> bool:
    """True when `answer_latex` is the exact defect this file guards
    against: no backslash, no `plain:` fallback, and a letter run or bare
    keyword KaTeX would render as broken italic prose."""
    if answer_latex is None:
        return True
    text = str(answer_latex)
    if "\\" in text or text.startswith("plain:"):
        return False
    return bool(_LETTER_RUN_RE.search(text) or _BARE_KEYWORD_RE.search(text))


def _is_dollar_leak(answer_latex) -> bool:
    """True when a non-`plain:` `answer_latex` contains an unescaped `$`
    (KaTeX math-mode delimiter -> parse error, red `katex-error` styling)."""
    if answer_latex is None:
        return False
    text = str(answer_latex)
    if text.startswith("plain:"):
        return False
    return bool(_UNESCAPED_DOLLAR_RE.search(text))


def _is_percent_leak(answer_latex) -> bool:
    """True when a non-`plain:` `answer_latex` contains an unescaped `%`
    (LaTeX comment character -> silently truncates the rest of the answer,
    with no error and no visual indication)."""
    if answer_latex is None:
        return False
    text = str(answer_latex)
    if text.startswith("plain:"):
        return False
    return bool(_UNESCAPED_PERCENT_RE.search(text))


def _active_skills():
    _, skills = load_skills()
    return sorted(
        (s for s in skills if s.status == "active" and s.sources),
        key=lambda s: s.id,
    )


_ACTIVE_SKILLS = _active_skills()


@pytest.mark.parametrize("skill", _ACTIVE_SKILLS, ids=[s.id for s in _ACTIVE_SKILLS])
def test_answer_latex_has_no_italicised_prose(skill):
    """One walk over every (source, difficulty, seed) combination, checking
    all three `answer_latex`/KaTeX defect modes described in the module
    docstring: italicised prose, an unescaped `$`, and
    an unescaped `%`."""
    failures = []
    for src in skill.sources:
        params = _tuplify(src.params)
        for difficulty in _difficulties_for(src.topic, src.difficulty):
            for seed in range(SEEDS):
                question = make(src.topic, seed=seed, difficulty=difficulty, **params)
                answer_latex = question.answer_latex
                if _is_prose_leak(answer_latex):
                    failures.append((src.topic, difficulty, seed, "prose", answer_latex))
                if _is_dollar_leak(answer_latex):
                    failures.append((src.topic, difficulty, seed, "dollar", answer_latex))
                if _is_percent_leak(answer_latex):
                    failures.append((src.topic, difficulty, seed, "percent", answer_latex))
    assert not failures, (
        f"{skill.id}: answer_latex is unsafe for KaTeX -- 'prose' means "
        f"italicised English prose with no backslash and no plain: prefix, "
        f"'dollar' means an unescaped $ (KaTeX parse error), 'percent' "
        f"means an unescaped % (silently truncates the rest of the "
        f"answer): {failures[:5]}"
    )
