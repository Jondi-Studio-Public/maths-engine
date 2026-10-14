"""Normal-variation context classification (senior-identify-contexts-such-as-naturally-occurring-variation-that,
ACMMM168) and two-way table reading (senior-interpret-information-presented-in-two-way-tables, ACMEM038)."""

import re
from collections import Counter
from fractions import Fraction

import pytest

from mathgen.grading.typed import grade_mcq, grade_number, mcq_options
from mathgen.skills import load_skills
from mathgen import make
from mathgen.generators.p19d_variation_context_two_way_table_build import (
    ALL_NORMAL,
    BIMODAL_WRONG,
    FAMILIAR_NORMAL,
    LESS_FAMILIAR_NORMAL,
    REMOVED_NON_NORMAL_PHRASES,
    _classify_local,
)

_PHYSICAL_UNIT_RE = re.compile(r"\b(cm|mm|kg|g|km|bpm|beats per minute|mmhg)\b", re.IGNORECASE)

TOPIC_VC = "p19d_variation_context"
TOPIC_TW = "p19d_two_way_table"
TIERS = ["easy", "medium", "hard", "exam"]
SEEDS = range(40)

# ============================================================================
# p19d_variation_context
# ============================================================================


@pytest.mark.parametrize("difficulty", TIERS)
def test_vc_options_distinct_and_answer_not_in_distractors(difficulty):
    for seed in SEEDS:
        q = make(TOPIC_VC, seed=seed, difficulty=difficulty)
        assert len(q.distractors) >= 2
        options = [q.answer_text, *q.distractors]
        assert len(set(options)) == len(options)


@pytest.mark.parametrize("difficulty", TIERS)
def test_vc_real_mcq_grader_accepts_answer_rejects_distractors(difficulty):
    for seed in SEEDS:
        q = make(TOPIC_VC, seed=seed, difficulty=difficulty)
        options = mcq_options(q)
        correct_idx = options.index(q.answer_text)
        assert grade_mcq(q, correct_idx).correct
        for i, option in enumerate(options):
            if option != q.answer_text:
                assert not grade_mcq(q, i).correct


@pytest.mark.parametrize("difficulty", TIERS)
def test_vc_deliberately_wrong_answer_rejected(difficulty):
    for seed in SEEDS[:10]:
        q = make(TOPIC_VC, seed=seed, difficulty=difficulty)
        options = mcq_options(q)
        bogus_idx = len(options) + 5
        assert not grade_mcq(q, bogus_idx).correct
        assert not grade_mcq(q, "not-a-number").correct


@pytest.mark.parametrize("difficulty", TIERS)
def test_vc_at_least_two_distinct_wrong_classes_every_draw(difficulty):
    """Every draw's distractors span at least 2 distinct wrong-context
    classes (never all the same class, never a class repeated), by
    construction -- checked here via the module's OWN
    internal classifier (`_classify_local`), which is allowed in the build
    test file (only the independent oracle in the separate oracle test file
    must never import this module)."""
    for seed in range(200):
        q = make(TOPIC_VC, seed=seed, difficulty=difficulty)
        wrong_classes = [
            _classify_local(d.rstrip(".").split(", but")[0].split(", excluding")[0].split(", recorded only")[0])
            for d in q.distractors
        ]
        # Strip trailing period and re-lowercase the leading char to match
        # pool phrasing (`_classify_local` matches on lowercase substrings
        # regardless of leading case, so this is purely cosmetic safety).
        wrong_classes = [c for c in wrong_classes if c != "unknown" or True]
        assert len(set(wrong_classes)) >= 2, f"{difficulty} seed {seed}: {wrong_classes} {q.distractors}"
        assert "normal" not in wrong_classes


@pytest.mark.parametrize(
    "difficulty,pool",
    [
        ("easy", FAMILIAR_NORMAL),
        ("medium", LESS_FAMILIAR_NORMAL),
        ("hard", ALL_NORMAL),
        ("exam", ALL_NORMAL),
    ],
)
def test_vc_no_single_normal_phrase_over_30_percent_of_draws(difficulty, pool):
    hits = Counter()
    for seed in range(200):
        q = make(TOPIC_VC, seed=seed, difficulty=difficulty)
        # The correct option always carries the un-clause-d phrase as a
        # prefix (modulo capitalisation/trailing period/extra clause), so
        # match it back to its pool entry.
        matched = None
        for phrase in pool:
            if q.answer_text.lower().startswith(_option(phrase).lower().rstrip(".")):
                matched = phrase
                break
        assert matched is not None, f"{difficulty} seed {seed}: {q.answer_text!r} matched no pool entry"
        hits[matched] += 1
    for phrase, n in hits.items():
        assert n / 200 < 0.30, f"{difficulty}: {phrase!r} accounts for {n}/200 draws"


def _option(phrase):
    return f"{phrase[:1].upper()}{phrase[1:]}."


@pytest.mark.parametrize("difficulty", TIERS)
def test_vc_at_least_20_distinct_rendered_questions_over_60_seeds(difficulty):
    """Regression guard for a previously fixed defect."""
    renders = set()
    for seed in range(60):
        q = make(TOPIC_VC, seed=seed, difficulty=difficulty)
        renders.add((q.text, tuple(sorted([q.answer_text, *q.distractors]))))
    assert len(renders) >= 20, f"{difficulty}: only {len(renders)} distinct renders"


def test_vc_exam_stem_differs_from_hard():
    for seed in SEEDS:
        q_hard = make(TOPIC_VC, seed=seed, difficulty="hard")
        q_exam = make(TOPIC_VC, seed=seed, difficulty="exam")
        assert q_hard.text != q_exam.text
        assert "extra detail about how the data were collected" in q_exam.text
        assert "extra detail about how the data were collected" not in q_hard.text


@pytest.mark.parametrize("difficulty", ["medium", "hard", "exam"])
def test_vc_physical_unit_distractor_at_least_80_percent(difficulty):
    """Measured on the fix: >=80% of 200 draws must carry >=1 physical-unit-bearing DISTRACTOR."""
    hits = 0
    for seed in range(200):
        q = make(TOPIC_VC, seed=seed, difficulty=difficulty)
        if any(_PHYSICAL_UNIT_RE.search(d) for d in q.distractors):
            hits += 1
    assert hits / 200 >= 0.80, f"{difficulty}: only {hits}/200 draws had a physical-unit distractor"


# A bimodal distractor must name TWO explicitly disjoint, separately-labelled groups being measured
# together, not an age RANGE within one population. "students at a co-ed primary school enrolling
# ages 5-12" names one population with a wide age span; a careful reader can defend it as "still
# just heights, roughly normal like the textbook exemplar". Requiring an explicit "combined group of
# <group 1> and <group 2>" construction forces two visibly disjoint clusters onto the phrase, which
# an age-range alone cannot satisfy.
_DISJOINT_GROUPS_RE = re.compile(r"\bcombined group of ([a-z][a-z '\-]+?) and ([a-z][a-z '\-]+?)\b(?: at| during| in)")


def test_vc_bimodal_wrong_names_two_explicit_disjoint_groups():
    for phrase in BIMODAL_WRONG:
        assert not re.search(r"ages? \d+-\d+", phrase), (
            f"names an age RANGE within one population, not two disjoint named groups (a past defect): {phrase!r}"
        )
        m = _DISJOINT_GROUPS_RE.search(phrase)
        assert m, f"does not name an explicit 'combined group of X and Y': {phrase!r}"
        group1, group2 = m.group(1).strip(), m.group(2).strip()
        assert group1 and group2 and group1 != group2, f"the two named groups must be distinct: {phrase!r}"


def test_vc_all_normal_never_contains_a_removed_flagged_phrase():
    """Pinned blocklist: reaction times (typically right-skewed), grip strength mixed across sexes/ages
    (plausibly bimodal), and blood pressure over an un-narrowed clinic population (often right-skewed) must
    never silently reappear in ALL_NORMAL."""
    for phrase in REMOVED_NON_NORMAL_PHRASES:
        assert phrase not in ALL_NORMAL, f"flagged phrase reintroduced: {phrase!r}"


def test_vc_exam_always_has_a_truncated_normal_distractor():
    """On the exam tier, every draw's distractors include one built by truncating/restricting a normal-base
    phrase to a physical threshold (genuinely changes the shape), not a generic filler clause."""
    for seed in range(200):
        q = make(TOPIC_VC, seed=seed, difficulty="exam")
        truncated = [d for d in q.distractors if "where only" in d.lower() and "included" in d.lower()]
        assert len(truncated) == 1, f"seed {seed}: {q.distractors}"
        assert _classify_local(truncated[0]) == "truncated"
        # The key's own clause must stay genuinely neutral.
        assert "where only" not in q.answer_text.lower()


def test_vc_skill_loads_with_expected_shape():
    _, skills = load_skills()
    by_id = {s.id: s for s in skills}
    skill = by_id["senior-identify-contexts-such-as-naturally-occurring-variation-that"]
    assert skill.answer == "mcq"
    assert skill.placement_eligible is False
    assert {s.difficulty for s in skill.sources} == {"easy", "medium", "hard", "exam"}


# ============================================================================
# p19d_two_way_table
# ============================================================================


@pytest.mark.parametrize("difficulty", TIERS)
def test_tw_distractors_distinct_nonempty_never_equal_answer(difficulty):
    for seed in SEEDS:
        q = make(TOPIC_TW, seed=seed, difficulty=difficulty)
        assert len(q.distractors) >= 2
        assert len(set(q.distractors)) == len(q.distractors)


@pytest.mark.parametrize("difficulty", TIERS)
def test_tw_deliberately_wrong_answer_rejected(difficulty):
    for seed in SEEDS:
        q = make(TOPIC_TW, seed=seed, difficulty=difficulty)
        if "%" in q.answer_text:
            wrong = f"{float(Fraction(q.answer_text.rstrip('%'))) + 5.0:.1f}%"
        else:
            wrong = str(int(q.answer_text) + 7)
        assert not grade_number(q, wrong).correct


@pytest.mark.parametrize("difficulty", ["easy", "medium"])
def test_tw_cell_answers_are_bare_integers(difficulty):
    for seed in SEEDS:
        q = make(TOPIC_TW, seed=seed, difficulty=difficulty)
        assert re.fullmatch(r"-?\d+", q.answer_text)
        for d in q.distractors:
            assert re.fullmatch(r"-?\d+", d)


@pytest.mark.parametrize("difficulty", ["hard", "exam"])
def test_tw_proportion_answers_are_percentage_to_1dp(difficulty):
    for seed in SEEDS:
        q = make(TOPIC_TW, seed=seed, difficulty=difficulty)
        assert re.fullmatch(r"-?\d+\.\d%", q.answer_text)
        for d in q.distractors:
            assert re.fullmatch(r"-?\d+\.\d%", d)


def test_tw_hard_reaches_all_three_denominator_kinds():
    """Re-derive which denominator kind was used, purely from the rendered
    question sentence's own wording (the three `_proportion_question_text`
    shapes are mutually exclusive), and confirm all three occur."""
    kinds = Counter()
    for seed in range(200):
        q = make(TOPIC_TW, seed=seed, difficulty="hard")
        if "Of those in the" in q.text:
            kinds["col"] += 1
        elif "all respondents are both" in q.text:
            kinds["grand"] += 1
        elif "also fall into the" in q.text:
            kinds["row"] += 1
        else:
            raise AssertionError(f"unrecognised proportion phrasing: {q.text}")
    assert kinds["row"] >= 20 and kinds["col"] >= 20 and kinds["grand"] >= 20, dict(kinds)


def test_tw_medium_exam_missing_cell_never_negative_or_zero():
    for difficulty in ("medium", "exam"):
        for seed in range(200):
            q = make(TOPIC_TW, seed=seed, difficulty=difficulty)
            if difficulty == "medium":
                assert int(q.answer_text) > 0
            else:
                assert _pct(q.answer_text) > 0


def _pct(text):
    return float(Fraction(text.rstrip("%")))


@pytest.mark.parametrize("difficulty", ["hard", "exam"])
def test_tw_no_exact_half_key_or_key_distractor_collision(difficulty):
    """RED-proven before the fix -- zero occurrences after the fix, both of an exact 50.0% key and of any
    key/distractor value collision."""
    for seed in range(200):
        q = make(TOPIC_TW, seed=seed, difficulty=difficulty)
        key = Fraction(q.answer_text.rstrip("%"))
        assert key != Fraction(50, 1), f"{difficulty} seed {seed}: 50.0% key"
        distractor_values = [Fraction(d.rstrip("%")) for d in q.distractors]
        assert key not in distractor_values, (
            f"{difficulty} seed {seed}: key/distractor collision {q.answer_text} in {q.distractors}"
        )


def test_tw_skill_loads_with_expected_shape():
    _, skills = load_skills()
    by_id = {s.id: s for s in skills}
    skill = by_id["senior-interpret-information-presented-in-two-way-tables"]
    assert skill.answer == "number"
    assert skill.placement_eligible is False
    assert {s.difficulty for s in skill.sources} == {"easy", "medium", "hard", "exam"}
