"""A learner-facing instruction that lives only in `text` never reaches the
screen: the front-end math renderer uses `latex` whenever it is present,
so a phrase like "(2 d.p.)" that appears only in `text` is invisible to the
learner while the grader still enforces it.

Parametrized over **skills**, not `TOPICS` -- a topic-parametrized sweep only
ever calls the bare `make(topic, seed)` default variant, which is exactly the
shape that carries no instruction to drop in the first place. The defect
lives in the params-selected variants (`kind`/`kinds`) that only a skill's own
`sources` reach.
"""

import re

import pytest

from mathgen import make
from mathgen.skills import load_skills

# Phrases that, if present in the plain `text`, are an instruction the learner must see to answer
# correctly (a rounding/format directive).
#
# Widened. The first draft of this pattern matched `decimal places` only in the plural and `exact
# value` only in that exact wording. Both gaps hid a real instance of the very defect this test
# exists to catch: `cartesian_polar_convert`'s `reverse` variant says "rounded to 1 decimal place"
# (singular) and its `axis` variant says "the exact distance from the origin". Neither was flagged;
# both were found by hand. A detector with a hole in it reads exactly like a clean result, so the
# singular/plural forms and the looser `exact <noun>` are spelled out here deliberately.
KEY = re.compile(
    r"(\d+\s*d\.?p\.?"
    r"|decimal place(?:s)?"
    r"|significant figure(?:s)?|\d+\s*s\.?f\.?"
    r"|to the nearest|nearest"
    r"|correct to|round(?:ed)? to"
    r"|in the form|in terms of"
    r"|give your answer|state your answer|leave your answer"
    r"|exact \w+|simplest form|lowest terms)",
    re.I,
)

SEEDS = range(1, 21)
# skills but a 20-seed sweep is the one that counts.


def _normalise(s: str) -> str:
    """lowercase, strip `\\text{`, `}` and backslashes, collapse whitespace --
    so an instruction wrapped as `\\text{ d.p.}` in `latex` still matches the
    same phrase found in `text`."""
    s = s.lower()
    s = s.replace(r"\text{", "")
    s = s.replace("{", "").replace("}", "")
    s = s.replace("\\", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


_domains, _skills = load_skills()
ACTIVE_SKILLS = [s for s in _skills if getattr(s, "status", "active") == "active" and getattr(s, "sources", None)]


@pytest.mark.parametrize("skill", ACTIVE_SKILLS, ids=lambda s: s.id)
def test_latex_carries_every_text_instruction(skill):
    """Every rounding/format instruction present in a question's `text` must
    also be present (once normalised) in its on-screen `latex`."""
    missing: dict[str, set[str]] = {}
    for src in skill.sources:
        for seed in SEEDS:
            q = make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))
            lx = q.latex or ""
            if not lx or lx.startswith("plain:"):
                continue
            norm_lx = _normalise(lx)
            phrases = set(p.lower() for p in KEY.findall(q.text or ""))
            for phrase in phrases:
                if _normalise(phrase) not in norm_lx:
                    missing.setdefault(f"{src.topic}:{src.difficulty}:{src.params}", set()).add(phrase)
    assert not missing, f"{skill.id}: on-screen latex drops instruction(s) present in text: {missing}"
