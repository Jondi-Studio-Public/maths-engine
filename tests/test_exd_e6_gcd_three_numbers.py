"""A NEW `count` parameter on the `euclidean_algorithm_gcd` topic
(`mathgen/generators/university_foundations.py`).

`count=3` draws a THIRD positive integer and computes `gcd(gcd(first, second), third)` -- a genuine second
application of the Euclidean algorithm, verified independently against BOTH `sympy.gcd(sympy.gcd(first,
second), third)` and a three-way brute-force enumeration.
"""

from __future__ import annotations

import math
import re

import pytest

from mathgen import make
from mathgen.core import VerificationError
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "euclidean_algorithm_gcd"

# Matches the skill's hard source exactly: `{count: 3}` (number_hi=500 comes
# from the "hard" named preset via the merge, unchanged).
_HARD_PARAMS = dict(count=3)


def _make_hard(seed, **overrides):
    params = {**_HARD_PARAMS, **overrides}
    return make(TOPIC, seed=seed, difficulty="hard", **params)


# ---------------------------------------------------------------------------
# Text quality, prompt variety, determinism
# ---------------------------------------------------------------------------


def test_count3_no_bad_text_patterns_over_60_seeds():
    for seed in range(60):
        q = _make_hard(seed)
        defects = find_text_defects(q)
        assert defects == [], f"seed={seed}: {defects} ({q.text!r})"


def test_count3_distinct_raw_prompts_over_60_seeds():
    """Matches the library's own convention
    (at least `_MIN_DISTINCT = 10` distinct raw prompts over 60 seeds). `prompt_shape()` masks every digit run, and this
    generator's whole prompt is a single fixed sentence template around three
    numbers, so its OWN prompt_shape is a single constant string by design --
    the escalation is that the hard tier's shape (three numbers) differs from
    the easy/medium tiers' shape (two numbers), proved below, not that the
    hard tier is internally varied."""
    texts = {_make_hard(seed).text for seed in range(60)}
    assert len(texts) >= 10, texts


def test_count3_determinism_same_seed_is_byte_identical():
    for seed in (0, 1, 2, 3, 100, 12345):
        first = _make_hard(seed).to_json()
        second = _make_hard(seed).to_json()
        assert first == second


def test_count3_hard_adds_shape_over_easy_and_medium():
    """The new `count=3` hard tier's shape is not in the union of the
    UNCHANGED easy/medium tiers' shapes (both mirror the skill YAML's own
    `count`-default sources, unaffected by this change): a genuine escalation
    in the NUMBER OF QUANTITIES the prompt names (three, not two), not
    merely a bigger `number_hi`."""
    lower_shapes = {prompt_shape(make(TOPIC, seed=s, difficulty="easy")) for s in range(60)}
    lower_shapes |= {prompt_shape(make(TOPIC, seed=s, difficulty="medium")) for s in range(60)}
    hard_shapes = {prompt_shape(_make_hard(s)) for s in range(60)}
    new_shapes = hard_shapes - lower_shapes
    assert len(new_shapes) >= 1, (hard_shapes, lower_shapes)


# ---------------------------------------------------------------------------
# Independent oracle: re-derive gcd(a, b, c) from the rendered prompt text
# ALONE, using only Python's stdlib `math.gcd` -- never importing
# `mathgen.generators.university_foundations`, and never using `sympy`
# either, so this genuinely does not share a code path with the generator's
# own `check()` (which itself already cross-checks sympy AND enumeration).
# ---------------------------------------------------------------------------

_GCD3_RE = re.compile(r"^Use the Euclidean algorithm to find gcd\((\d+), (\d+), (\d+)\)\.$")


def _independent_gcd3(question_text: str) -> int:
    m = _GCD3_RE.match(question_text)
    assert m is not None, f"oracle: could not parse {question_text!r}"
    a, b, c = (int(g) for g in m.groups())
    return math.gcd(a, b, c)


def test_count3_answer_matches_independent_oracle_over_600_seeds():
    mismatches = []
    for seed in range(600):
        q = _make_hard(seed)
        oracle_answer = _independent_gcd3(q.text)
        if oracle_answer != q.answer:
            mismatches.append((seed, q.text, oracle_answer, q.answer))
    assert mismatches == [], mismatches


import mathgen.generators.university_foundations as _uf  # noqa: E402


def test_count3_corrupted_sympy_check_raises_verification_error(monkeypatch):
    original_gcd = _uf.sympy.gcd

    def _wrong_gcd(*args, **kwargs):
        result = original_gcd(*args, **kwargs)
        return result + 1

    monkeypatch.setattr(_uf.sympy, "gcd", _wrong_gcd)
    with pytest.raises(VerificationError):
        _make_hard(0)


def test_count3_corrupted_check_is_not_a_false_positive_when_uncorrupted():
    _make_hard(0)  # no exception


# ---------------------------------------------------------------------------
# Distractors: distinct, non-empty, never equal to the answer.
# ---------------------------------------------------------------------------


def test_count3_distractors_non_empty_and_never_equal_answer_over_300_seeds():
    for seed in range(300):
        q = _make_hard(seed)
        assert len(q.distractors) >= 1, (seed, q.text, q.distractors)
        for d in q.distractors:
            assert d != q.answer_text, (seed, q.text, d)


def test_count3_distractors_distinct_within_each_question_over_300_seeds():
    for seed in range(300):
        q = _make_hard(seed)
        assert len(set(q.distractors)) == len(q.distractors), (seed, q.text, q.distractors)


# ---------------------------------------------------------------------------
# Degeneracy fix: the hard tier's `answer == 1` rate and the "both misconceptions
# collapse into the true answer" rate must stay materially below the original
# measured degeneracy (82.2% / 51.8% over 600 seeds, `first`/`second`
# drawn independently and uniformly with no planted factor). This locks in
# the improvement from planting a shared factor `g` into `first`/`second`
# so their pair gcd is never forced to 1.
# ---------------------------------------------------------------------------


def test_count3_answer_is_1_rate_and_misconception_collapse_rate_over_600_seeds():
    n = 600
    answer_is_1 = 0
    both_collapse = 0
    for seed in range(n):
        q = _make_hard(seed)
        if q.answer == 1:
            answer_is_1 += 1
        m = _GCD3_RE.match(q.text)
        assert m is not None, q.text
        a, b, c = (int(x) for x in m.groups())
        first_two_only = math.gcd(a, b)
        last_two = math.gcd(b, c)
        distractor_set = set(q.distractors)
        if str(first_two_only) not in distractor_set and str(last_two) not in distractor_set:
            both_collapse += 1
    # The original measurement: answer==1 in 493/600 (82.2%), both-collapse
    # in 311/600 (51.8%). The two-number tier's own answer==1 baseline rate
    # is ~61% (P(gcd=1)=1/zeta(2) heuristic); a reasonable bar for the fix is
    # to land measurably below that, and to keep the collapse rate well
    # under half.
    assert answer_is_1 / n < 0.61, (answer_is_1, n)
    assert both_collapse / n < 0.30, (both_collapse, n)


# ---------------------------------------------------------------------------
# The real grader (mathgen.grading.GRADERS["number"]) -- skill's `answer: number`
# ---------------------------------------------------------------------------

from mathgen.grading.typed import GRADERS  # noqa: E402

_NUMBER_GRADER = GRADERS["number"]


def test_count3_grader_accepts_canonical_and_rejects_every_distractor_over_200_seeds():
    bad = []
    for seed in range(200):
        q = _make_hard(seed)
        canonical_result = _NUMBER_GRADER(q, q.answer_text)
        if not canonical_result.correct:
            bad.append(("canonical-rejected", seed, q.answer_text))
        for d in q.distractors:
            distractor_result = _NUMBER_GRADER(q, d)
            if distractor_result.correct:
                bad.append(("distractor-accepted", seed, d, q.answer_text))
    assert bad == [], bad


def test_count3_grader_rejects_deliberately_wrong_misconception_values():
    """The three misconception values, computed by hand against a fixed seed
    and checked they are actually wrong before asserting the grader rejects
    them (never a vacuous check).

    Fix: `first`/`second` are now redrawn inside the `count == 3` branch
    with a planted shared factor, so seed=16's numbers changed from the
    original (66, 480, 416) to (234, 420, 452) -- re-derived directly from
    `q.text` here (not hardcoded blind) so this test cannot silently drift
    from the generator again. gcd(234, 420, 452) = 2. first_two_only =
    gcd(234, 420) = 6 (stopped one Euclidean-algorithm pass short);
    last_two_only = gcd(420, 452) = 4 (ignored the first number); sum =
    234+420+452 = 1106 (reused-input-style candidate).
    """
    q = _make_hard(16)
    m = _GCD3_RE.match(q.text)
    assert m is not None, q.text
    a, b, c = (int(x) for x in m.groups())
    assert (a, b, c) == (234, 420, 452)
    answer = q.answer
    assert answer == 2
    first_two_only = math.gcd(a, b)
    last_two_only = math.gcd(b, c)
    total = a + b + c
    wrong_candidates = [first_two_only, last_two_only, total]
    assert first_two_only == 6
    assert last_two_only == 4
    assert total == 1106
    for wrong in wrong_candidates:
        assert wrong != answer, (wrong, answer)
        result = _NUMBER_GRADER(q, str(wrong))
        assert not result.correct, (wrong, q.answer_text)


# The render baseline replay is the authoritative proof (2 stored rows, both count-default calls);
# this is a lightweight structural regression check in addition.

_GCD2_RE = re.compile(r"^Use the Euclidean algorithm to find gcd\((\d+), (\d+)\)\.$")


def test_original_count2_default_shape_unaffected_over_300_seeds():
    for seed in range(300):
        for difficulty in ("easy", "medium", "hard"):
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            assert _GCD2_RE.match(q.text), (seed, difficulty, q.text)


def test_original_count2_explicit_matches_default_over_60_seeds():
    """`count=2` passed explicitly must be byte-identical to omitting it
    (the parameter's default), proving the new knob truly defaults to the
    old behaviour rather than merely happening to look the same."""
    for seed in range(60):
        implicit = make(TOPIC, seed=seed, difficulty="hard").to_json()
        explicit = make(TOPIC, seed=seed, difficulty="hard", count=2).to_json()
        assert implicit == explicit


# ---------------------------------------------------------------------------
# Skill loads via mathgen.skills.load_skills
# ---------------------------------------------------------------------------


def test_skill_loads_via_api_skills():
    from mathgen import skills as sm

    _domains, skill_list = sm.load_skills()
    ids = {s.id: s for s in skill_list}
    assert "euclidean-algorithm-gcd" in ids
    skill = ids["euclidean-algorithm-gcd"]
    hard_sources = [s for s in skill.sources if s.difficulty == "hard"]
    assert len(hard_sources) == 1
    assert hard_sources[0].params.get("count") == 3
