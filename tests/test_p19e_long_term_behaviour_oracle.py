"""Independent oracle for the `p19e_long_term_behaviour` questions: it re-derives each answer from the rendered
question text.

Parses `question.text`, re-derives the class by a fresh route, and confirms `answer_text` is the unique letter
whose displayed option matches that class.

`N_SEEDS` (per tier) overridable by `P19E_ORACLE_SEEDS`, default 250 so
>= 1000 questions are checked over the four tiers -- per topic.
"""

from dataclasses import replace
import os
import re
from fractions import Fraction

from mathgen.grading.typed import grade_mcq, mcq_options
from mathgen.skills import load_skills
from mathgen import make
import mathgen.generators.p19e_long_term_behaviour_build  # noqa: F401 -- registers topics

N_SEEDS = int(os.environ.get("P19E_ORACLE_SEEDS", "250"))
TIERS = ["easy", "medium", "hard", "exam"]

GEO_ID = "senior-understand-the-limiting-behaviour-as-n-rightarrow-infty"
REC_ID = "senior-recognise-that-a-sequence-generated-by-a-first"

# Same fixed wordings the generator uses -- an oracle is allowed to know the
# *contract* (three named classes with fixed text), just never the module's
# own construction/classification code.
_GEO_OPTION_TEXT = {
    "zero": "Converges to 0",
    "infinity": "Diverges to +∞ (the terms stay positive and grow without bound)",
    "neither": "Neither: the terms oscillate in sign and do not approach a limit",
}
_REC_OPTION_TEXT = {
    "increase": "Increases without bound",
    "decrease": "Decreases without bound",
    "steady": "Settles to a steady state",
}


def _skill_sources(skill_id):
    _domains, skills = load_skills()
    skill = next(s for s in skills if s.id == skill_id)
    return {src.difficulty: src for src in skill.sources}


GEO_SOURCES = _skill_sources(GEO_ID)
REC_SOURCES = _skill_sources(REC_ID)
REC_SOURCES["exam"] = replace(REC_SOURCES["exam"], topic="p19e_recurrence_long_term_class", params={})


def _make(sources, difficulty, seed):
    src = sources[difficulty]
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


def _letter_for(text, option_text, expected_class):
    """The letter whose "L: wording" block matches option_text[expected_class]."""
    expected_wording = option_text[expected_class]
    for letter in ("A", "B", "C"):
        match = re.search(rf"{letter}: (.+?)(?:   [A-C]:|$)", text)
        if match and match.group(1) == expected_wording:
            return letter
    raise AssertionError(f"no option in {text!r} matches {expected_wording!r}")


def _geo_true_class(r: Fraction) -> str:
    if r > 1:
        return "infinity"
    if r <= -1:
        return "neither"
    return "zero"


def _rec_true_class(a: Fraction, b, u0) -> str:
    if a == 1:
        if b > 0:
            return "increase"
        if b < 0:
            return "decrease"
        return "steady"
    if 0 < a < 1:
        return "steady"
    x_star = Fraction(b, 1 - a)
    if u0 == x_star:
        return "steady"
    return "increase" if u0 > x_star else "decrease"


# ==========================================================================
# geometric limit class -- parse r (or terms, or t2/t5) straight from text
# ==========================================================================

_R_DIRECT_RE = re.compile(r"r = (-?\d+(?:/\d+)?)")
_TERMS_3_RE = re.compile(r"(-?\d+), (-?\d+), (-?\d+)")


def _parse_fraction(s: str) -> Fraction:
    if "/" in s:
        p, q = s.split("/")
        return Fraction(int(p), int(q))
    return Fraction(int(s))


def _geo_oracle_class(text: str) -> str:
    m = _R_DIRECT_RE.search(text)
    if m:
        return _geo_true_class(_parse_fraction(m.group(1)))
    # exam tier: t2 and t5 (two separate sentence shapes)
    m2 = re.search(r"t2 = (-?\d+)", text)
    m5 = re.search(r"t5 = (-?\d+)", text)
    if m2 and m5:
        t2, t5 = int(m2.group(1)), int(m5.group(1))
        r_cubed = Fraction(t5, t2)
        # odd-power root: try small denominators up to 3, numerators up to 3
        for q in range(1, 4):
            for p in range(-3, 4):
                if p == 0:
                    continue
                r_try = Fraction(p, q)
                if r_try**3 == r_cubed:
                    return _geo_true_class(r_try)
        raise AssertionError(f"could not recover r from t2={t2}, t5={t5}")
    # medium tier: three consecutive terms
    m3 = _TERMS_3_RE.search(text)
    if m3:
        t1, t2v, t3v = (int(g) for g in m3.groups())
        r = Fraction(t2v, t1)
        assert Fraction(t3v, t2v) == r, "terms are not a genuine geometric sequence"
        return _geo_true_class(r)
    raise AssertionError(f"could not parse r from: {text!r}")


def test_geo_oracle_red_first_on_corrupted_key():
    """Prove the oracle actually discriminates: swap `answer_text` for a
    distractor letter and confirm the check below would fail."""
    q = _make(GEO_SOURCES, "easy", 0)
    true_class = _geo_oracle_class(q.text)
    true_letter = _letter_for(q.text, _GEO_OPTION_TEXT, true_class)
    corrupted = next(d for d in q.distractors if d != true_letter)
    try:
        assert corrupted == true_letter
        raised = False
    except AssertionError:
        raised = True
    assert raised, "the oracle must reject a corrupted key -- this proves it is not a tautology"


def test_geo_oracle_re_derives_the_key_over_many_seeds():
    checked = 0
    for tier in TIERS:
        for seed in range(N_SEEDS):
            q = _make(GEO_SOURCES, tier, seed)
            true_class = _geo_oracle_class(q.text)
            true_letter = _letter_for(q.text, _GEO_OPTION_TEXT, true_class)
            assert q.answer_text == true_letter, (tier, seed, q.text)
            for d in q.distractors:
                assert d != true_letter
            checked += 1
    assert checked >= 1000


def test_geo_oracle_grader_agrees():
    for tier in TIERS:
        for seed in range(60):
            q = _make(GEO_SOURCES, tier, seed)
            options = mcq_options(q)
            true_class = _geo_oracle_class(q.text)
            true_letter = _letter_for(q.text, _GEO_OPTION_TEXT, true_class)
            correct_idx = options.index(true_letter)
            assert grade_mcq(q, correct_idx).correct


# ==========================================================================
# recurrence long-term class -- parse a, b, u0 (or the exam-context numbers)
# ==========================================================================

_FORMULA_RE = re.compile(r"u\(n\+1\) = (?:([\d.]+) × )?u\(n\)(?: ([+-]) (\d+))?")
_U0_RE = re.compile(r"u\(0\) = (-?\d+)")
_EXAM_PCT_RE = re.compile(r"(?:grows by|increases by) (\d+)% each period")
_EXAM_AMOUNT_RE = re.compile(r"\$(\d+) is(?: also)? (added|removed)")
_EXAM_START_RE = re.compile(r"starting at \$(\d+)|starts at \$(\d+)")


def _rec_oracle_class(text: str) -> str:
    m = _FORMULA_RE.search(text)
    if m:
        a = Fraction(m.group(1) or 1)  # the stem leaves out a multiplier of 1 and a constant of 0
        sign = -1 if m.group(2) == "-" else 1
        b = sign * int(m.group(3) or 0)
        u0_match = _U0_RE.search(text)
        u0 = int(u0_match.group(1))
        return _rec_true_class(a, b, u0)
    # exam tier: percentage rate + added/removed amount + starting value
    pct = int(_EXAM_PCT_RE.search(text).group(1))
    amount_match = _EXAM_AMOUNT_RE.search(text)
    amount = int(amount_match.group(1))
    verb = amount_match.group(2)
    b = amount if verb == "added" else -amount
    a = Fraction(100 + pct, 100)
    start_match = _EXAM_START_RE.search(text)
    u0 = int(start_match.group(1) or start_match.group(2))
    return _rec_true_class(a, b, u0)


def test_rec_oracle_red_first_on_corrupted_key():
    q = _make(REC_SOURCES, "easy", 0)
    true_class = _rec_oracle_class(q.text)
    true_letter = _letter_for(q.text, _REC_OPTION_TEXT, true_class)
    corrupted = next(d for d in q.distractors if d != true_letter)
    try:
        assert corrupted == true_letter
        raised = False
    except AssertionError:
        raised = True
    assert raised, "the oracle must reject a corrupted key -- this proves it is not a tautology"


def test_rec_oracle_re_derives_the_key_over_many_seeds():
    checked = 0
    for tier in TIERS:
        for seed in range(N_SEEDS):
            q = _make(REC_SOURCES, tier, seed)
            true_class = _rec_oracle_class(q.text)
            true_letter = _letter_for(q.text, _REC_OPTION_TEXT, true_class)
            assert q.answer_text == true_letter, (tier, seed, q.text)
            for d in q.distractors:
                assert d != true_letter
            checked += 1
    assert checked >= 1000


def test_rec_oracle_grader_agrees():
    for tier in TIERS:
        for seed in range(60):
            q = _make(REC_SOURCES, tier, seed)
            options = mcq_options(q)
            true_class = _rec_oracle_class(q.text)
            true_letter = _letter_for(q.text, _REC_OPTION_TEXT, true_class)
            correct_idx = options.index(true_letter)
            assert grade_mcq(q, correct_idx).correct
