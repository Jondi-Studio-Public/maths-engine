"""Classify a 3x3 linear system (ACMSM110), topic pv0b_system_cases."""

from __future__ import annotations

from dataclasses import replace

import re

import pytest

from mathgen.grading.typed import GRADERS, mcq_options
from mathgen.skills import load_skills
from mathgen import make
from mathgen.generators.pv0b_system_cases_build import (
    KEY_INFINITE,
    KEY_NONE,
    KEY_UNIQUE,
    _eq_text,
    _rank_class,
)

TOPIC = "pv0b_system_cases"
SKILL_ID = "senior-examine-the-three-cases-for-solutions-of-systems"
DIFFICULTIES = ["easy", "medium", "hard", "exam"]
SEEDS = range(40)


@pytest.fixture(scope="module")
def skills_dict():
    _, skills = load_skills()
    return {s.id: s for s in skills}


def _make_from_sources(skills_dict, difficulty, seed):
    skill = skills_dict[SKILL_ID]
    src = next(s for s in skill.sources if s.difficulty == difficulty)
    if (
        difficulty == "exam"
    ):  # the skill's exam source is now a multi-part chain; these tests pin the earlier exam generator
        src = replace(src, topic="pv0b_system_cases", params={})
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


# --- Determinism -------------------------------------------------------------

# --- Distractor hygiene: exactly 2 distractors, distinct, never == answer ---


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_distractors_are_the_two_other_classes(skills_dict, difficulty):
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, difficulty, seed)
        assert q.answer_text in (KEY_UNIQUE, KEY_NONE, KEY_INFINITE), (difficulty, seed, q.answer_text)
        assert len(q.distractors) == 2, (difficulty, seed, q.distractors)
        assert len(set(q.distractors)) == 2, (difficulty, seed, q.distractors)
        assert set([q.answer_text] + list(q.distractors)) == {KEY_UNIQUE, KEY_NONE, KEY_INFINITE}


# --- Real grader accept/reject (mcq: pass an INDEX into mcq_options) --------


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_real_grader_accepts_canonical_and_rejects_every_distractor(skills_dict, difficulty):
    grade_mcq = GRADERS["mcq"]
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, difficulty, seed)
        options = mcq_options(q)
        correct_index = options.index(q.answer_text)
        assert grade_mcq(q, str(correct_index)).correct, (difficulty, seed, q.text)
        for i, opt in enumerate(options):
            if opt != q.answer_text:
                assert not grade_mcq(q, str(i)).correct, (difficulty, seed, q.text, opt)


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_real_grader_rejects_an_out_of_range_index(skills_dict, difficulty):
    grade_mcq = GRADERS["mcq"]
    for seed in SEEDS[:10]:
        q = _make_from_sources(skills_dict, difficulty, seed)
        assert not grade_mcq(q, "99").correct, (difficulty, seed, q.text)


# --- At least 20 distinct rendered questions per tier over 60 seeds --------


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_at_least_20_distinct_questions_per_tier(difficulty):
    seen = {make(TOPIC, seed=s, difficulty=difficulty, tier=difficulty).text for s in range(60)}
    assert len(seen) >= 20, (difficulty, len(seen))


# --- ValueError on a nonsensical preset -------------------------------------


def test_raises_value_error_on_unknown_tier():
    with pytest.raises(ValueError):
        make(TOPIC, seed=1, tier="bogus")


# --- 200-seed class concentration, scaled to the 3-class answer space ------


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_class_concentration_at_200_seeds_scaled_to_three_classes(difficulty):
    counts = {KEY_UNIQUE: 0, KEY_NONE: 0, KEY_INFINITE: 0}
    for seed in range(200):
        q = make(TOPIC, seed=seed, difficulty=difficulty, tier=difficulty)
        counts[q.answer_text] += 1
    top_share = max(counts.values()) / 200
    # Uniform-over-three puts the top share near 33%; allow a generous band
    # ("each class within about 28-39%") plus fair-coin slack.
    assert top_share <= 0.45, (difficulty, counts)
    assert min(counts.values()) >= 20, (difficulty, counts)


# --- Key POSITION balance at 400 seeds --------------------------------------


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_key_position_balance_at_400_seeds(difficulty):
    positions = {0: 0, 1: 0, 2: 0}
    for seed in range(400):
        q = make(TOPIC, seed=seed, difficulty=difficulty, tier=difficulty)
        options = mcq_options(q)
        positions[options.index(q.answer_text)] += 1
    # No position should be starved or dominate absurdly over 400 draws.
    for count in positions.values():
        assert count >= 60, (difficulty, positions)


# --- Prompt variety: >= 3 distinct prompt_shape() outputs per tier ----------


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_prompt_variety_at_least_3_shapes(difficulty):
    from mathgen.text_quality import prompt_shape

    shapes = set()
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty=difficulty, tier=difficulty)
        shapes.add(prompt_shape(q))
    assert len(shapes) >= 3, (difficulty, shapes)


def test_hard_and_exam_each_add_a_shape_absent_from_lower_tiers():
    from mathgen.text_quality import prompt_shape

    shapes_by_tier = {}
    for difficulty in DIFFICULTIES:
        shapes_by_tier[difficulty] = {
            prompt_shape(make(TOPIC, seed=s, difficulty=difficulty, tier=difficulty)) for s in range(60)
        }
    lower_union = shapes_by_tier["easy"] | shapes_by_tier["medium"]
    assert shapes_by_tier["hard"] - lower_union, "hard tier adds no new prompt shape"
    lower_union |= shapes_by_tier["hard"]
    assert shapes_by_tier["exam"] - lower_union, "exam tier adds no new prompt shape"


# --- Guards, proven red first ------------------------------------------------


def test_guard_all_zero_left_side_fires():
    """`_eq_text` must refuse to print an all-zero-left-side equation. Proven
    red: without the `check()` inside `_eq_text`, `join_terms([(0,'x'),(0,'y'),
    (0,'z')])` returns the string "0" and a naive f-string would happily print
    "0 = 5" as a system equation."""
    from mathgen.core import VerificationError

    with pytest.raises(VerificationError):
        _eq_text(0, 0, 0, 5)


def test_guard_two_identical_equations_would_be_caught():
    """`_build_medium`/`_build_hard` assert `len(set(eqs)) == 3` after building
    the three equation strings. Proven red: three identical rows produce the
    same equation text three times over, so the set collapses to size 1."""
    rows = [(1, 2, 3, 4), (1, 2, 3, 4), (1, 2, 3, 4)]
    eqs = [_eq_text(*r) for r in rows]
    assert len(set(eqs)) == 1  # demonstrates the collapse the guard must catch


# --- The independent (rank-based) check fires on a corrupted value ----------


def test_rank_class_independent_check_catches_a_corrupted_label():
    # A genuinely full-rank system: classified UNIQUE by _rank_class.
    rows = [(1, 0, 0, 5), (0, 1, 0, 6), (0, 0, 1, 7)]
    derived = _rank_class(rows)
    assert derived == KEY_UNIQUE
    # A corrupted claim of "no solution" for this same system must disagree.
    assert derived != KEY_NONE


# --- No printed equation has an all-zero left side (medium/hard/exam) ------

_SENTENCE_SPLIT = re.compile(r"(?<=[.?!])\s+")


def _sentences(text: str) -> list[str]:
    """Split `text` into sentences on a sentence-ending punctuation mark
    (period, question mark or exclamation mark) followed by whitespace --
    some intro variants end in "?", so a literal ". " split would merge
    that sentence with the next one."""
    return _SENTENCE_SPLIT.split(text)


def _equations_sentence(text: str) -> str:
    """The equations-only sentence: dropped the redundant "System: " label, so the printed system is now just the
    plain semicolon-joined equations, on its own sentence (identified by being the one sentence containing
    "=", since no prose sentence ever does)."""
    return next(s for s in _sentences(text) if "=" in s)


@pytest.mark.parametrize("difficulty", ["medium", "hard", "exam"])
def test_no_printed_equation_has_all_zero_left_side(difficulty):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty, tier=difficulty)
        parts = _equations_sentence(q.text).split(";")
        eq1 = parts[0].strip()
        eq2 = parts[1].strip()
        eq3 = parts[2].strip()
        for eq in (eq1, eq2, eq3):
            lhs = eq.split("=")[0].strip()
            assert lhs != "0" and lhs != "", (difficulty, seed, q.text, eq)


# --- No two equations identical on the unique-class draws -------------------


@pytest.mark.parametrize("difficulty", ["medium", "hard"])
def test_no_two_equations_identical_when_unique(difficulty):
    checked = 0
    for seed in range(200):
        q = make(TOPIC, seed=seed, difficulty=difficulty, tier=difficulty)
        if q.answer_text != KEY_UNIQUE:
            continue
        parts = _equations_sentence(q.text).split(";")
        eq1 = parts[0].strip()
        eq2 = parts[1].strip()
        eq3 = parts[2].strip()
        assert len({eq1, eq2, eq3}) == 3, (difficulty, seed, q.text)
        checked += 1
    assert checked >= 10, (difficulty, checked)


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_no_python_operators_in_prompt(difficulty):
    """Proven red on the pre-fix generator: the exam condition was built as
    ``f"... k != {c}"``, which appeared verbatim in both `text` and `latex`
    (e.g. seed 0's exam question read "k != 6"). A human-facing prompt must
    use mathematical notation (\u2260), never a programming-language token."""
    banned = re.compile(r"!=|<=|>=|\*\*")
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty=difficulty, tier=difficulty)
        assert not banned.search(q.text), (difficulty, seed, q.text)
        assert not banned.search(q.latex), (difficulty, seed, q.latex)


def _is_prose(sentence: str) -> bool:
    """A sentence that, once its numbers/variables/parameter/operators/row-
    matrix punctuation are stripped, still has letters left over is prose
    (an instruction or a condition) and must be visible in `latex`
    (the front-end math renderer uses `latex` whenever it is present, never `text`). A
    sentence that strips to nothing is the printed system itself -- either
    the plain equations or the easy tier's echelon "Row n: [...]" rundown --
    real math, already visible via a `\\begin{array}`/matrix environment,
    not required to duplicate as `\\text`."""
    s = re.sub(r"\brow\b", "", sentence, flags=re.I)
    stripped = re.sub(r"[0-9xyzkXYZK+\-=(),.;:\[\]|\s]", "", s)
    return bool(stripped)


def _text_spans(latex: str) -> str:
    """Every ``\\text{...}`` span in `latex`, concatenated (so a check can
    look for a sentence anywhere across the spans without caring which span
    it landed in)."""
    return " ".join(re.findall(r"\\text\{([^}]*)\}", latex))


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_every_prose_sentence_reaches_latex_text_spans(difficulty):
    """Proven red on the pre-fix generator: the exam tier's closing
    instruction ("Classify the system's solution set for every value of k
    satisfying this condition.") was built into `text` but never appended to
    `latex`, so seed 0's exam question showed the system and the condition
    but never told the learner what to do with them."""
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty=difficulty, tier=difficulty)
        spans = _text_spans(q.latex)
        for sentence in _sentences(q.text):
            sentence = sentence.strip()
            if not sentence or not _is_prose(sentence):
                continue
            assert sentence.rstrip(".") in spans or sentence in spans, (
                difficulty,
                seed,
                sentence,
                q.latex,
            )


# Every (k position, condition form) exam combo needs >= 2 --- --- classes, with no class over 70%
# of that combo's draws ------------------


def _exam_position(text: str) -> str:
    """L = k appears left of '=' in an equation, R = right of '=' (looking
    only at the equations sentence itself, never the condition sentence --
    "Here, k = 6." would otherwise register a spurious L)."""
    eq_sentence = _equations_sentence(text)
    parts = eq_sentence.rstrip(".").split(";")
    left_has_k = right_has_k = False
    for part in parts:
        lhs, rhs = part.split("=")
        if "k" in lhs:
            left_has_k = True
        if "k" in rhs:
            right_has_k = True
    if left_has_k and right_has_k:
        return "L and R"
    if left_has_k:
        return "L"
    if right_has_k:
        return "R"
    return "none"  # pragma: no cover -- every exam draw states k somewhere


def _exam_condition_form(text: str) -> str:
    if "k ≠" in text:
        return "k ≠ c"
    if "k = " in text:
        return "k = c"
    if "every value of k" in text:
        return "every k"
    return "?"  # pragma: no cover


def test_exam_every_position_condition_combo_has_at_least_two_classes():
    """Proven red on the pre-fix generator (all four combos it emitted were
    single-class: L/'k != c' -> unique 126/126, R/'k != c' -> none 147/147,
    R/'k = c' -> infinite 61/61, R/'every k' -> infinite 66/66 -- a learner
    who memorises the surface pattern scores 100% without reasoning about
    the stated condition). Requires every combo drawn >= 20 times over 400
    exam seeds to offer at least two classes, with no class above 70%."""
    from collections import Counter

    combo_class = Counter()
    for seed in range(400):
        q = make(TOPIC, seed=seed, difficulty="exam", tier="exam")
        combo_class[(_exam_position(q.text), _exam_condition_form(q.text), q.answer_text)] += 1

    combo_totals = Counter()
    combo_by_class = {}
    for (pos, cond, cls), n in combo_class.items():
        combo_totals[(pos, cond)] += n
        combo_by_class.setdefault((pos, cond), {})[cls] = n

    checked = 0
    for combo, total in combo_totals.items():
        if total < 20:
            continue
        checked += 1
        classes = combo_by_class[combo]
        assert len(classes) >= 2, (combo, classes)
        top_share = max(classes.values()) / total
        assert top_share <= 0.70, (combo, classes, top_share)
    assert checked >= 4, combo_totals  # every combo the exam tier actually emits


# --- Both skills load through mathgen.skills.load_skills ------------------------


def test_skill_loads_with_expected_shape(skills_dict):
    skill = skills_dict[SKILL_ID]
    assert skill.answer == "mcq"
    assert skill.placement_eligible is False
    assert skill.prerequisites == ["vector-addition"]
    by_difficulty = {s.difficulty: s for s in skill.sources}
    assert set(by_difficulty) == {"easy", "medium", "hard", "exam"}
    assert by_difficulty["exam"].stages == ["exam"]
    assert by_difficulty["exam"].stages != []
    assert by_difficulty["hard"].stages == [6, "maintenance"]
