"""Stem wording lint: no boilerplate, no wrong article, no "1 <plural noun>".

Swept over every active skill's declared sources at 2 seeds x every tier the topic declares (the shape of
`test_p9r_answer_latex_mangled.py`). The prompt, the multiple-choice options and any structured stem are read.
There is no allowlist: a hit means the generator that produced the text is wrong.
"""

import re

import pytest

from mathgen import make
from mathgen.skills import load_skills
from scripts.regrade_sweep import _difficulties_for, _tuplify

SEEDS = 2

BANNED_PHRASES = ("Exam-style question", "Show working")

# "a"/"A" then a word that starts with a vowel letter. Words that start with a vowel letter but a consonant sound
# ("one", "uni...", "u..." as in unit/use/user, "euro") are fine. Function words after a bare variable "a"
# ("a and b", "a is positive") are not articles.
_ARTICLE_RE = re.compile(r"(?<![\w.'\-\\^_{(])([aA]) ([A-Za-z][\w\-]*)")
_CONSONANT_SOUND = re.compile(r"^(one|once|uni|u|eu|ubiq|use|usu|uti)", re.I)
_NOT_AN_ARTICLE = {"and", "or", "is", "are", "in", "if", "of", "on", "as", "at", "into", "equals", "i", "e", "x"}
# Capital "A" is a label ("Machine A is ...") unless it opens a sentence.
_SENTENCE_START = re.compile(r"(?:^|[.?!:]\s+)$")

# A whole-token 1 followed by a plural noun: "1 years", "1 successes". Not "21 years", "1.5 years", "1 m/s".
_ONE_PLURAL_RE = re.compile(r"(?<![\w.,/\-])(?:(\b[A-Za-z]+) )?1 ([A-Za-z]{3,})(?![\w.,/\-]*\))")
# "Machine 1 produces", "Year 1 sales": the 1 is a label, not a quantity.
_LABELS = {
    "machine",
    "item",
    "event",
    "year",
    "quarter",
    "bus",
    "train",
    "boat",
    "ferry",
    "group",
    "option",
    "case",
    "type",
    "stage",
    "step",
    "part",
    "test",
    "sample",
    "class",
    "box",
    "bag",
    "urn",
    "player",
    "worker",
    "batch",
    "line",
    "route",
    "service",
    "person",
    "week",
    "day",
    "month",
    "period",
    "trial",
    "question",
    "set",
    "list",
    "row",
    "column",
    "table",
    "figure",
    "plan",
    "path",
    "tank",
    "shape",
    "object",
    "container",
    "formula",
    "method",
}
_SINGULAR_S = {"is", "was", "has", "this", "plus", "minus", "its", "less", "thus", "does", "pass", "times", "as", "bus"}


def _active_skills():
    _, skills = load_skills()
    return [s for s in skills if s.status == "active"]


def _texts(q):
    out = [q.text]
    out.extend(str(d) for d in (q.distractors or []))
    if isinstance(q.stem, str):
        out.append(q.stem)
    return out


def wording_defects(text: str) -> list[str]:
    found = [p for p in BANNED_PHRASES if p.lower() in text.lower()]
    for m in _ARTICLE_RE.finditer(text):
        art, word = m.group(1), m.group(2)
        if word[0].lower() not in "aeiou" or len(word) < 2:
            continue
        if word.lower() in _NOT_AN_ARTICLE or _CONSONANT_SOUND.match(word):
            continue
        if art == "A" and not _SENTENCE_START.search(text[: m.start(1)]):
            continue
        found.append(f"'{art} {word}'")
    for m in _ONE_PLURAL_RE.finditer(text):
        if (m.group(1) or "").lower() in _LABELS:
            continue
        before = text[: m.start(2) - 2].rstrip()
        if before.endswith(("+", "-", "=", "*", "/", "^", "(", "\u2212", "|")):
            continue  # arithmetic, or a stem-and-leaf key ("5 | 1 means 51"), not a count
        w = m.group(2).lower()
        if w.endswith("s") and not w.endswith("ss") and w not in _SINGULAR_S:
            found.append(f"'1 {m.group(2)}'")
    return found


def test_stem_wording_has_no_defects():
    bad = {}
    for skill in _active_skills():
        for src in skill.sources:
            params = _tuplify(src.params)
            for difficulty in _difficulties_for(src.topic, src.difficulty):
                for seed in range(SEEDS):
                    q = make(src.topic, seed=seed, difficulty=difficulty, **params)
                    for t in _texts(q):
                        for d in wording_defects(t):
                            bad.setdefault((src.topic, difficulty, d), (skill.id, seed, t[:90]))
    assert not bad, f"{len(bad)} wording defects: " + "\n".join(f"{k}: {v}" for k, v in sorted(bad.items()))


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Exam-style question: find x.", True),
        ("Show working for this.", True),
        ("A investment is opened.", True),
        ("Invest in a investment fund.", True),
        ("Find a and b.", False),
        ("Machine A is faster than B.", False),
        ("A one-off fee and a unit rate and a euro.", False),
        ("a apple", True),
        ("a hat and a hexagon", False),
        ("Year 1 sales and Machine 1 makes", False),
        ("Write 1 hundredths as a decimal", True),
        ("for 1 years", True),
        ("there are 1 successes", True),
        ("21 years, 1.5 years, 1 year", False),
        ("for 1 steps", True),
        ("(y + 2)^2 + 1 gives a parabola", False),
        ("1 m/s", False),
    ],
)
def test_detector_examples(text, expected):
    assert bool(wording_defects(text)) is expected
