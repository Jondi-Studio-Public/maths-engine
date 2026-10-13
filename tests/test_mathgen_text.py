"""Learner-facing text: sign formatting, notation, prompt and working-text defects.

Part of the generator sweep, split across `tests/test_mathgen_*.py` so the CI
shards can balance it (`scripts/ci_shard.py` balances files by size). Verification
is built into the generators themselves (each independently re-checks its
answer with SymPy and raises VerificationError on mismatch), so constructing
thousands of questions IS the correctness test; the assertions here cover the
output contract on top of that. Set `MATHS_DEEP_SEEDS` for a deeper run of the
text checks.
"""

import re

import pytest

from mathgen import make, topics
from mathgen.core import PRESETS, REGISTRY
from tests.mathgen_sweep_support import NEW_KIND_VARIANTS, make_kind, TEXT_SEEDS, WORKING_SEEDS
from mathgen.text_quality import find_text_defects

TOPICS = [t for _lvl, t in topics()]
# Topics registered by a build module (`mathgen/generators/*_build.py`): these
# must meet the learner-facing text bar. Legacy topics keep their existing
# notation (their `latex` field is what the UI renders) and are out of scope
# until someone owns them.
BUILD_TOPICS = sorted(t for t, (_lvl, fn) in REGISTRY.items() if fn.__module__.rsplit(".", 1)[-1].endswith("_build"))

# ---------------------------------------------------------------------------
# Sign-formatting invariant: no sampled
# question should ever print a double sign ("+ -", "- -") or a coefficient of
# 1/-1 spelled out as "1x"/"-1x". Checked across every registered topic (200
# seeds, default params) and every explicit kind/param variant already
# enumerated in NEW_KIND_VARIANTS (50 seeds each, matching that test's own
# sample size to keep the suite fast).
_BAD_SIGN_PATTERNS = [
    re.compile(r"\+ -"),
    re.compile(r"- -"),
    re.compile(r"(?<![\d.])1x\b"),
    re.compile(r"(?<![\d.])-1x\b"),
]


def _assert_no_bad_signs(q, where):
    fields = [q.text, q.latex, q.answer_text, q.answer_latex, *q.distractors, *q.working]
    for s in fields:
        if not isinstance(s, str):
            continue
        for pat in _BAD_SIGN_PATTERNS:
            assert not pat.search(s), f"{where}: bad sign pattern {pat.pattern!r} in {s!r}"


@pytest.mark.parametrize("topic", TOPICS)
def test_no_bad_sign_patterns(topic):
    n = 60 if topic == "differential_equation" else 200
    for seed in range(n):
        q = make(topic, seed=seed)
        _assert_no_bad_signs(q, f"{topic} seed={seed}")


@pytest.mark.parametrize("topic,param,value", NEW_KIND_VARIANTS)
def test_no_bad_sign_patterns_new_kind_variants(topic, param, value):
    for seed in range(50):
        q = make_kind(topic, param, value, seed)
        _assert_no_bad_signs(q, f"{topic} {param}={value} seed={seed}")


# ---------------------------------------------------------------------------
# `scientific_notation`'s question used to render the "ordinary"
# number with Python's `:g`, which itself switches to exponent notation once
# the value's own exponent goes below -4 or reaches 6 -- so "Write 3.1e-05 in
# scientific notation" was asking the learner to transcribe an answer already
# shown to them, exactly at the large/small end of the range the skill exists
# to test. 27 of 120 draws (seeds 0-119, default params, this topic has no
# presets) showed the defect before the fix in mathgen/generators/
# curriculum_expansion.py; 0 of the same 120 after. Seeds 0-39 alone already
# reach both breaking exponents (-5 and 6, see the specific-value test below),
# so 40 seeds is enough to catch a regression.
_SCI_NOTATION_EXPONENT_RE = re.compile(r"\d[eE][+-]?\d")


def test_scientific_notation_never_shows_python_exponent_notation():
    for seed in range(40):
        q = make("scientific_notation", seed=seed)
        fields = [q.text, q.answer_text, *q.distractors]
        for s in fields:
            assert not _SCI_NOTATION_EXPONENT_RE.search(s), (
                f"scientific_notation seed={seed}: {s!r} already shows exponent "
                "notation, so there is nothing left to convert"
            )


def test_scientific_notation_extremes_render_positionally():
    """The specific values that broke today: an exponent below -4 (seed 27,
    1.2e-5) and an exponent at/above 6 (seed 4, 1.7e6) -- the ones `:g`
    reformats, not a mid-range value that never exercised it."""
    small = make("scientific_notation", seed=27)
    assert small.answer_text == "1.2 × 10^-5"
    assert "0.000012" in small.text
    assert not _SCI_NOTATION_EXPONENT_RE.search(small.text)

    large = make("scientific_notation", seed=4)
    assert large.answer_text == "1.7 × 10^6"
    assert "1700000" in large.text
    assert not _SCI_NOTATION_EXPONENT_RE.search(large.text)


# ---------------------------------------------------------------------------
# Learner-facing text invariant for build topics: no Python exponent notation
# (`5**3`), no asterisk as the multiplication sign in a prompt, no "1 parts".
# Patterns live in mathgen/text_quality.py, shared with
# scripts/render_skill_questions.py so the review artefact and this test agree.
# Checked over every difficulty preset the topic declares (wording differs per
# tier) plus the default, 60 seeds each.
#
# Topics below shipped before this test existed and still exhibit the defect
# named. Each is a known wording defect awaiting a fix; the tripwire test after this one fails as soon as a
# topic is fixed, so the entry must be removed in the same change. Never add
# a new build topic here: fix its wording instead.
TEXT_DEFECTS_GRANDFATHERED = {
    "exponent_laws_positive_zero": "double_star",
}


def _text_defects_over_tiers(topic):
    """`[(where, defect), ...]` across every preset and the default."""
    found = []
    for difficulty in [*sorted(PRESETS.get(topic, {})), None]:
        for seed in range(TEXT_SEEDS):
            q = make(topic, seed=seed, difficulty=difficulty)
            for defect in find_text_defects(q):
                found.append((f"{topic} difficulty={difficulty} seed={seed}", defect))
    return found


@pytest.mark.parametrize("topic", BUILD_TOPICS)
def test_no_bad_text_patterns(topic):
    if topic in TEXT_DEFECTS_GRANDFATHERED:
        pytest.skip(f"grandfathered {TEXT_DEFECTS_GRANDFATHERED[topic]}; see the tripwire test")
    found = _text_defects_over_tiers(topic)
    assert not found, "\n".join(f"{where}: {defect}" for where, defect in found[:10])


@pytest.mark.parametrize("topic,pattern", sorted(TEXT_DEFECTS_GRANDFATHERED.items()))
def test_grandfathered_text_defect_still_present(topic, pattern):
    assert topic in BUILD_TOPICS, f"{topic} is not a build topic; drop it from TEXT_DEFECTS_GRANDFATHERED"
    found = _text_defects_over_tiers(topic)
    names = {defect.split(" in ", 1)[0] for _where, defect in found}
    assert pattern in names, f"{topic} no longer shows {pattern}: remove it from TEXT_DEFECTS_GRANDFATHERED"
    assert names == {pattern}, f"{topic} shows other defects too: {sorted(names - {pattern})}"


# ---------------------------------------------------------------------------
# Three detectors for defect classes that shipped and that nothing could see.
# All three are HARD -- no backlog, no ratchet, because measuring each one across
# all 388 skills drove the real count to zero rather than pinning it:
#
#   signed_after_operator  "17 + -14"   7 topics, every one FIXED. mathgen.core
#                          gained `paren_negative` for the case where the
#                          operator is the thing being taught, beside the older
#                          `join_terms` for the case where it should simplify.
#   double_punctuation     "cost $9.."  1 skill, and it turned out to be a false
#                          positive on "p.a.," -- see below.
#   the lone-label check   in test_mathgen_options.py, `test_answer_is_not_the_only_labelled_option`.
#
# Two candidate patterns were MEASURED AND REJECTED, which is the part worth
# remembering. `double_space` flagged 19 skills and nearly every one was
# deliberate visual layout -- counting objects as "|#|  |#|  |#|", comparing two
# drawn lines, arrows, table columns, indented working. In a primary counting
# question the spacing IS the content. And the first `double_punctuation` regex
# flagged 11 skills, of which every one was an ellipsis ("4, 9, 14, 19, ...",
# "{1, ..., 10}", "0.666...") or an abbreviation ("6% p.a., compounded"). A rule
# that fires on correct notation trains people to ignore it. Measure the blast
# radius before adding a pattern, and when the measurement says the rule is
# wrong, fix the rule rather than pinning the skills it misjudges.


# ---------------------------------------------------------------------------
# Worked-solution text. `Question.working` is shown to the learner after they
# answer, and it used to be unchecked. It arrived as a
# ratchet -- a `WORKING_TEXT_BACKLOG` of the six then-dirty topics, skipped
# here and forced out one at a time by a tripwire test -- because failing all
# six at once would have turned the gate red. All six were fixed, the backlog reached empty, and the ratchet and
# its tripwire are now gone: the check below is unconditional over every build
# topic. A defect here means the generator is wrong and the generator is what
# gets fixed, so do not reintroduce a skip list.
# These defects can be rare, not systematic: they depend on a specific
# generated value landing on one (a duration of exactly one minute, an answer
# that happens to be a perfect square under a square root, ...), so a shallow
# seed count can miss one entirely -- timetable_itinerary_duration's "1
# minutes" needed seed 126 at the medium tier to show up at all, 1 in 800
# across seeds 0-199 x every tier. 60 matches TEXT_SEEDS so this check is at
# least as deep as the prompt check, but even 60 is not a guarantee: a
# 200-seed sweep across every build topic is what caught that one, and this
# suite runs on every pull request so the count here stays bounded rather than
# matching that sweep's depth.


def _working_defects_over_tiers(topic):
    """`[(where, defect), ...]` found in the worked-solution lines only."""
    found = []
    for difficulty in [*sorted(PRESETS.get(topic, {})), None]:
        for seed in range(WORKING_SEEDS):
            q = make(topic, seed=seed, difficulty=difficulty)
            for defect in find_text_defects(q, include_working=True):
                if " in working[" in defect:
                    found.append((f"{topic} difficulty={difficulty} seed={seed}", defect))
    return found


@pytest.mark.parametrize("topic", BUILD_TOPICS)
def test_no_bad_working_text(topic):
    found = _working_defects_over_tiers(topic)
    assert not found, "\n".join(f"{where}: {defect}" for where, defect in found[:10])


# ---------------------------------------------------------------------------
# Quotient rule: numerator and
# denominator are re-drawn whenever they are scalar multiples of each other,
# so the derivative must never be identically zero.
def test_quotient_rule_derivative_never_zero():
    for seed in range(500):
        q = make("differentiation", seed=seed, kinds=("quotient_rule",))
        assert q.answer not in ("0", 0), f"seed={seed}: degenerate zero derivative {q.answer!r}"
