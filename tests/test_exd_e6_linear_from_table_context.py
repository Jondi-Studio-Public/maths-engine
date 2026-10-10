"""A NEW `kind="equation_context"` on the `linear_from_table` topic
(`mathgen/generators/foundations_algebra.py`).

The existing `kind="equation"` shape (a table "x: ... y: ...") reads m/b
straight off a table -- the HARD source was every skill source's only shape
at every difficulty for THIS skill's own m/b construction, differing only by
`hi` (9/9/15), i.e. range-only. The fix does not touch that shape's body at
all; it adds a new branch, `equation_context`, that replaces the table with a
short real-world (hire-company) scenario built from the SAME m/b
construction and answer shape, requiring the learner to translate prose into
`y = mx + b` before answering rather than read `m`/`b` off a table. The HARD
source is repointed to the new kind (`hi: 15` passed explicitly); easy/medium
keep `kind: equation` untouched.

**Note**: `kind="intercept_context"` shares `_linear_from_table_scenario` with this kind
and ADDS to this same test file rather than creating a second one, since both
skills share the `linear_from_table` generator function -- the intercept-context test
classes/functions are prefixed `test_intercept_context_*` so they do not
collide with this file's `test_equation_context_*` prefix.
"""

from __future__ import annotations

import re

import pytest

from mathgen import make
from mathgen.core import VerificationError
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "linear_from_table"


def _make_context(seed, difficulty="hard"):
    return make(TOPIC, seed=seed, difficulty=difficulty, kind="equation_context", hi=15)


def _make_equation(seed, difficulty="hard", hi=15):
    return make(TOPIC, seed=seed, difficulty=difficulty, kind="equation", hi=hi)


# Text quality, prompt variety, determinism
# ---------------------------------------------------------------------------


def test_equation_context_no_bad_text_patterns_over_60_seeds():
    for seed in range(60):
        q = _make_context(seed)
        defects = find_text_defects(q)
        assert defects == [], f"seed={seed}: {defects} ({q.text!r})"


def test_equation_context_prompt_variety():
    """>= 3 distinct prompt_shape outputs over 60 seeds for the new hard kind alone -- one per phrase-bank
    template."""
    shapes = {prompt_shape(_make_context(s)) for s in range(60)}
    assert len(shapes) >= 3, shapes


def test_equation_context_hard_adds_shapes_over_easy_and_medium():
    """The new hard kind's shapes are not a subset of the union of the
    UNCHANGED easy/medium (`kind: equation`) tiers' shapes -- a genuine
    escalation in kind (prose scenario vs table), not merely a bigger `hi`."""
    lower_shapes = {prompt_shape(_make_equation(s, "easy", hi=9)) for s in range(60)}
    lower_shapes |= {prompt_shape(_make_equation(s, "medium", hi=9)) for s in range(60)}
    hard_shapes = {prompt_shape(_make_context(s)) for s in range(60)}
    new_shapes = hard_shapes - lower_shapes
    assert len(new_shapes) >= 1, (hard_shapes, lower_shapes)


def test_equation_context_determinism_same_seed_is_byte_identical():
    for seed in (0, 1, 2, 3, 100, 12345):
        first = _make_context(seed).to_json()
        second = _make_context(seed).to_json()
        assert first == second


# ---------------------------------------------------------------------------
# Independent oracle: re-derive m/b from the rendered prompt text ALONE, with
# plain regex + int(), never importing mathgen.generators.foundations_algebra.
# ---------------------------------------------------------------------------

_QUESTION_TAIL = r" Write the equation for the total cost y after x days \(y = mx \+ b\)\.$"
_CONTEXT_RES = [
    re.compile(r"^A hire company charges a flat fee of \$(\d+) plus \$(\d+) per day\." + _QUESTION_TAIL),
    re.compile(
        r"^A car hire company charges a flat fee of \$(\d+), then \$(\d+) for each "
        r"day of hire\." + _QUESTION_TAIL
    ),
    re.compile(
        r"^A tool hire company charges \$(\d+) up front and \$(\d+) for every extra "
        r"day the tool is kept\." + _QUESTION_TAIL
    ),
]


def _match_context(question_text: str):
    for pattern in _CONTEXT_RES:
        m = pattern.match(question_text)
        if m is not None:
            return m
    return None


def _independent_m_b(question_text: str) -> tuple[int, int]:
    """Every template states the flat fee (b) before the per-day rate (m),
    in that order -- see `_LINEAR_TABLE_CONTEXT_TEMPLATES` (not imported
    here; verified independently by reading the templates once and encoding
    the same order by hand, the oracle discipline)."""
    m = _match_context(question_text)
    assert m is not None, f"oracle: could not parse {question_text!r}"
    b, rate = (int(v) for v in m.groups())
    return rate, b  # (m, b)


def test_equation_context_answer_matches_independent_parse_over_600_seeds():
    mismatches = []
    for seed in range(600):
        q = _make_context(seed)
        oracle_m, oracle_b = _independent_m_b(q.text)
        if (oracle_m, oracle_b) != (q.answer["m"], q.answer["b"]):
            mismatches.append((seed, q.text, (oracle_m, oracle_b), q.answer))
    assert mismatches == [], mismatches


def test_equation_context_answer_text_matches_independent_format_over_300_seeds():
    """`answer_text` ("y = {m}x + {b}") re-derived from the oracle's own
    (m, b), independent of the generator's own `m_str`/f-string logic."""
    for seed in range(300):
        q = _make_context(seed)
        oracle_m, oracle_b = _independent_m_b(q.text)
        m_str = "" if oracle_m == 1 else str(oracle_m)
        expected = f"y = {m_str}x + {oracle_b}"
        assert q.answer_text == expected, (seed, q.text, q.answer_text, expected)


def test_equation_context_m_and_b_always_strictly_positive_over_600_seeds():
    """Documents a design choice: m and b are drawn
    strictly positive (1..hi), not merely non-negative -- a hire-company flat
    fee/day-rate of exactly $0 reads oddly and this also sidesteps a
    zero-coefficient edge case in the misconception distractors."""
    for seed in range(600):
        q = _make_context(seed)
        assert q.answer["m"] >= 1, (seed, q.answer)
        assert q.answer["b"] >= 1, (seed, q.answer)


# ---------------------------------------------------------------------------
# The generator's own in-body check() actually fires on a corrupted scenario.
# ---------------------------------------------------------------------------

import mathgen.generators.foundations_algebra as _fa  # noqa: E402


def test_equation_context_corrupted_scenario_raises_verification_error(monkeypatch):
    def _scenario_missing_values(rng, patterns, **slots):
        return "A company charges an undisclosed amount."

    monkeypatch.setattr(_fa, "phrase", _scenario_missing_values)
    with pytest.raises(VerificationError):
        make(TOPIC, seed=0, difficulty="hard", kind="equation_context", hi=15)


def test_equation_context_uncorrupted_scenario_is_not_a_false_positive():
    make(TOPIC, seed=0, difficulty="hard", kind="equation_context", hi=15)  # no exception


# ---------------------------------------------------------------------------
# Distractors: distinct, non-empty, never equal to the answer.
# ---------------------------------------------------------------------------


def test_equation_context_distractors_non_empty_and_never_equal_answer_over_300_seeds():
    for seed in range(300):
        q = _make_context(seed)
        assert len(q.distractors) >= 1, (seed, q.text, q.distractors)
        for d in q.distractors:
            assert d != q.answer_text, (seed, q.text, d)


def test_equation_context_distractors_distinct_within_each_question_over_300_seeds():
    for seed in range(300):
        q = _make_context(seed)
        assert len(set(q.distractors)) == len(q.distractors), (seed, q.text, q.distractors)


# ---------------------------------------------------------------------------
# The real grader (mathgen.grading.GRADERS["pair"])
# ---------------------------------------------------------------------------

from mathgen.grading.typed import GRADERS  # noqa: E402

_PAIR_GRADER = GRADERS["pair"]


def test_equation_context_grader_accepts_canonical_and_rejects_every_distractor_over_200_seeds():
    bad = []
    for seed in range(200):
        q = _make_context(seed)
        canonical_result = _PAIR_GRADER(q, q.answer_text)
        if not canonical_result.correct:
            bad.append(("canonical-rejected", seed, q.answer_text))
        for d in q.distractors:
            distractor_result = _PAIR_GRADER(q, d)
            if distractor_result.correct:
                bad.append(("distractor-accepted", seed, d, q.answer_text))
    assert bad == [], bad


def test_equation_context_grader_rejects_deliberately_wrong_values():
    """The three misconception values, computed by hand against a fixed seed
    and checked they are actually wrong before asserting the grader rejects
    them (never a vacuous check).

    seed=0, difficulty=hard, hi=15: m=4, b=2 (see the render output quoted
    dropped_rate = "y = 2".
    """
    q = _make_context(0)
    assert q.answer == {"m": 4, "b": 2}
    assert q.answer_text == "y = 4x + 2"
    wrong_candidates = ["y = 2x + 4", "y = 4x", "y = 2", "y = 5x + 2", "y = 4x + 3"]
    for wrong in wrong_candidates:
        assert wrong != q.answer_text, (wrong, q.answer_text)
        result = _PAIR_GRADER(q, wrong)
        assert not result.correct, (wrong, q.answer_text)


# Original kind="equation" behaviour untouched
# ---------------------------------------------------------------------------


def test_original_equation_kind_unaffected_over_300_seeds():
    """Confirms the ORIGINAL `kind == "equation"` branch (easy/medium
    sources, and `kind="equation"` explicitly at any difficulty) still
    matches its own long-standing table shape and never mentions the new
    hire-company scenario wording."""
    pattern = re.compile(r"^x: .*   y: .*\. Find the equation of the line \(y = mx \+ b\)\.$")
    for seed in range(300):
        for difficulty, hi in (("easy", 9), ("medium", 9), ("hard", 15)):
            q = _make_equation(seed, difficulty, hi=hi)
            assert pattern.match(q.text), (seed, difficulty, q.text)
            assert "hire company" not in q.text


# ---------------------------------------------------------------------------
# Skill loads via mathgen.skills.load_skills
# ---------------------------------------------------------------------------


def test_skill_loads_via_api_skills():
    from mathgen import skills as sm

    _domains, skill_list = sm.load_skills()
    ids = {s.id: s for s in skill_list}
    assert "equation-from-table" in ids
    skill = ids["equation-from-table"]
    hard_sources = [s for s in skill.sources if s.difficulty == "hard"]
    assert len(hard_sources) == 1
    assert hard_sources[0].params.get("kind") == "equation_context"
    assert hard_sources[0].params.get("hi") == 15
    easy_medium = [s for s in skill.sources if s.difficulty in ("easy", "medium")]
    assert all(s.params.get("kind") == "equation" for s in easy_medium)


# Reuses `_linear_from_table_scenario` (the equation-context helper) rather than duplicating it. All test names
# below are prefixed `test_intercept_context_*` to avoid colliding with the
# `test_equation_context_*` names.
#
# The existing `kind="intercept"` shape (a table "x: ... y: ...", asking for b alone) was every
# source's only shape at every difficulty for this skill's own m/b construction, differing only by
# `hi` (9/9/15), i.e. range-only. This fix does not touch that shape's body at all; it adds a new
# branch, `intercept_context`, that replaces the table with the SAME hire-company scenario the equation-context kind uses,
# asking only for the y-intercept (the flat fee) rather than reading it off a table. The HARD source
# is repointed to the new kind (`hi: 15` passed explicitly); easy/medium keep `kind: intercept`
# untouched. ===========================================================================


def _make_intercept_context(seed, difficulty="hard"):
    return make(TOPIC, seed=seed, difficulty=difficulty, kind="intercept_context", hi=15)


def _make_intercept(seed, difficulty="hard", hi=15):
    return make(TOPIC, seed=seed, difficulty=difficulty, kind="intercept", hi=hi)


# Text quality, prompt variety, determinism
# ---------------------------------------------------------------------------


def test_intercept_context_no_bad_text_patterns_over_60_seeds():
    for seed in range(60):
        q = _make_intercept_context(seed)
        defects = find_text_defects(q)
        assert defects == [], f"seed={seed}: {defects} ({q.text!r})"


def test_intercept_context_prompt_variety():
    """>= 3 distinct prompt_shape outputs over 60 seeds for the new hard kind alone -- one per phrase-bank
    template."""
    shapes = {prompt_shape(_make_intercept_context(s)) for s in range(60)}
    assert len(shapes) >= 3, shapes


def test_intercept_context_hard_adds_shapes_over_easy_and_medium():
    """The new hard kind's shapes are not a subset of the union of the
    UNCHANGED easy/medium (`kind: intercept`) tiers' shapes -- a genuine
    escalation in kind (prose scenario vs table), not merely a bigger `hi`."""
    lower_shapes = {prompt_shape(_make_intercept(s, "easy", hi=9)) for s in range(60)}
    lower_shapes |= {prompt_shape(_make_intercept(s, "medium", hi=9)) for s in range(60)}
    hard_shapes = {prompt_shape(_make_intercept_context(s)) for s in range(60)}
    new_shapes = hard_shapes - lower_shapes
    assert len(new_shapes) >= 1, (hard_shapes, lower_shapes)


def test_intercept_context_determinism_same_seed_is_byte_identical():
    for seed in (0, 1, 2, 3, 100, 12345):
        first = _make_intercept_context(seed).to_json()
        second = _make_intercept_context(seed).to_json()
        assert first == second


# ---------------------------------------------------------------------------
# Independent oracle: re-derive b from the rendered prompt text ALONE, with
# plain regex + int(), never importing mathgen.generators.foundations_algebra.
# ---------------------------------------------------------------------------

_INTERCEPT_QUESTION_TAIL = r" What is the flat fee, in dollars \(the y-intercept\)\?$"
_INTERCEPT_CONTEXT_RES = [
    re.compile(r"^A hire company charges a flat fee of \$(\d+) plus \$(\d+) per day\." + _INTERCEPT_QUESTION_TAIL),
    re.compile(
        r"^A car hire company charges a flat fee of \$(\d+), then \$(\d+) for each "
        r"day of hire\." + _INTERCEPT_QUESTION_TAIL
    ),
    re.compile(
        r"^A tool hire company charges \$(\d+) up front and \$(\d+) for every extra "
        r"day the tool is kept\." + _INTERCEPT_QUESTION_TAIL
    ),
]


def _match_intercept_context(question_text: str):
    for pattern in _INTERCEPT_CONTEXT_RES:
        m = pattern.match(question_text)
        if m is not None:
            return m
    return None


def _independent_b(question_text: str) -> int:
    """Every template states the flat fee (b) before the per-day rate (m),
    in that order -- see `_LINEAR_TABLE_CONTEXT_TEMPLATES` (not imported
    here; verified independently by reading the templates once and encoding
    the same order by hand, the oracle discipline, same as above)."""
    m = _match_intercept_context(question_text)
    assert m is not None, f"oracle: could not parse {question_text!r}"
    b, _rate = (int(v) for v in m.groups())
    return b


def test_intercept_context_answer_matches_independent_parse_over_600_seeds():
    mismatches = []
    for seed in range(600):
        q = _make_intercept_context(seed)
        oracle_b = _independent_b(q.text)
        if oracle_b != q.answer:
            mismatches.append((seed, q.text, oracle_b, q.answer))
    assert mismatches == [], mismatches


def test_intercept_context_answer_text_matches_str_of_answer_over_300_seeds():
    for seed in range(300):
        q = _make_intercept_context(seed)
        oracle_b = _independent_b(q.text)
        assert q.answer_text == str(oracle_b), (seed, q.text, q.answer_text, oracle_b)


def test_intercept_context_b_always_strictly_positive_over_600_seeds():
    """`_linear_from_table_scenario` draws m and b strictly positive (1..hi,
    see the module comment) -- confirmed here too since this kind reuses the
    same helper unmodified."""
    for seed in range(600):
        q = _make_intercept_context(seed)
        assert q.answer >= 1, (seed, q.answer)


def test_intercept_context_rendered_scenario_contains_the_stated_b_value_over_300_seeds():
    """Independent check: the rendered
    scenario text actually contains the answer's `$b` value, so a future
    edit to `_linear_from_table_scenario`'s phrasing cannot silently desync
    the text from the answer. `_linear_from_table_scenario` already asserts
    this internally (via its own `check()`) for both m and b; this test
    re-derives it independently from the rendered text/answer alone rather
    than trusting that internal check."""
    for seed in range(300):
        q = _make_intercept_context(seed)
        assert f"${q.answer}" in q.text, (seed, q.text, q.answer)


# ---------------------------------------------------------------------------
# The generator's own in-body check() (inside the shared scenario helper)
# actually fires on a corrupted scenario.
# ---------------------------------------------------------------------------


def test_intercept_context_corrupted_scenario_raises_verification_error(monkeypatch):
    def _scenario_missing_values(rng, patterns, **slots):
        return "A company charges an undisclosed amount."

    monkeypatch.setattr(_fa, "phrase", _scenario_missing_values)
    with pytest.raises(VerificationError):
        make(TOPIC, seed=0, difficulty="hard", kind="intercept_context", hi=15)


def test_intercept_context_uncorrupted_scenario_is_not_a_false_positive():
    make(TOPIC, seed=0, difficulty="hard", kind="intercept_context", hi=15)  # no exception


# ---------------------------------------------------------------------------
# Distractors: distinct, non-empty, never equal to the answer.
# ---------------------------------------------------------------------------


def test_intercept_context_distractors_non_empty_and_never_equal_answer_over_300_seeds():
    for seed in range(300):
        q = _make_intercept_context(seed)
        assert len(q.distractors) >= 1, (seed, q.text, q.distractors)
        for d in q.distractors:
            assert d != q.answer_text, (seed, q.text, d)


def test_intercept_context_distractors_distinct_within_each_question_over_300_seeds():
    for seed in range(300):
        q = _make_intercept_context(seed)
        assert len(set(q.distractors)) == len(q.distractors), (seed, q.text, q.distractors)


# ---------------------------------------------------------------------------
# The real grader (mathgen.grading.GRADERS["number"])
# ---------------------------------------------------------------------------

from mathgen.grading.typed import GRADERS as _GRADERS  # noqa: E402

_NUMBER_GRADER = _GRADERS["number"]


def test_intercept_context_grader_accepts_canonical_and_rejects_every_distractor_over_200_seeds():
    bad = []
    for seed in range(200):
        q = _make_intercept_context(seed)
        canonical_result = _NUMBER_GRADER(q, q.answer_text)
        if not canonical_result.correct:
            bad.append(("canonical-rejected", seed, q.answer_text))
        for d in q.distractors:
            distractor_result = _NUMBER_GRADER(q, d)
            if distractor_result.correct:
                bad.append(("distractor-accepted", seed, d, q.answer_text))
    assert bad == [], bad


def test_intercept_context_grader_rejects_deliberately_wrong_values():
    """The two misconception values, computed by hand against a fixed seed
    and checked they are actually wrong before asserting the grader rejects
    them (never a vacuous check).

    seed=0, difficulty=hard, hi=15: same scenario draw as
    `test_equation_context_grader_rejects_deliberately_wrong_values` (m=4, b=2, since
    `_linear_from_table_scenario` is called with the same seed/rng state at the start of both
    `equation_context` and `intercept_context` -- both kinds call the helper as their very first random draw).
    days is drawn next from `rng.randint(2, 6)`; for seed=0 that draws days=4, so total_after_days = 4*4 + 2 =
    18. reported-rate = "4" (m instead of b); reported-total = "18" (total cost after 4 days instead of the
    flat fee).
    """
    q = _make_intercept_context(0)
    assert q.answer == 2
    assert q.answer_text == "2"
    wrong_candidates = ["4", "18", "6", "1"]
    for wrong in wrong_candidates:
        assert wrong != q.answer_text, (wrong, q.answer_text)
        result = _NUMBER_GRADER(q, wrong)
        assert not result.correct, (wrong, q.answer_text)


# Original kind="intercept" behaviour untouched
# ---------------------------------------------------------------------------


def test_original_intercept_kind_unaffected_over_300_seeds():
    """Confirms the ORIGINAL `kind == "intercept"` branch (easy/medium
    sources, and `kind="intercept"` explicitly at any difficulty) still
    matches its own long-standing table shape and never mentions the new
    hire-company scenario wording."""
    pattern = re.compile(r"^x: .*   y: .*\. Find the y-intercept\.$")
    for seed in range(300):
        for difficulty, hi in (("easy", 9), ("medium", 9), ("hard", 15)):
            q = _make_intercept(seed, difficulty, hi=hi)
            assert pattern.match(q.text), (seed, difficulty, q.text)
            assert "hire company" not in q.text


# ---------------------------------------------------------------------------
# Skill loads via mathgen.skills.load_skills
# ---------------------------------------------------------------------------


def test_intercept_skill_loads_via_api_skills():
    from mathgen import skills as sm

    _domains, skill_list = sm.load_skills()
    ids = {s.id: s for s in skill_list}
    assert "intercept" in ids
    skill = ids["intercept"]
    hard_sources = [s for s in skill.sources if s.difficulty == "hard"]
    assert len(hard_sources) == 1
    assert hard_sources[0].params.get("kind") == "intercept_context"
    assert hard_sources[0].params.get("hi") == 15
    easy_medium = [s for s in skill.sources if s.difficulty in ("easy", "medium")]
    assert all(s.params.get("kind") == "intercept" for s in easy_medium)
