"""Tests for `mathgen/generators/add_sub_number_sentence_build.py`: `add_sub_number_sentence` (skill
`f10-add-subtract-number-sentences`).

Generator-specific proofs: option shape (3 to 5 well-formed sentences, one blank each), the key's blank position
and operation varying, no distractor that solves to the answer, the mcq grader battery, the tier progression, the
in-generator solver firing on a corrupted sentence, and skill load. The independent oracle (which parses
`question.text` and the option strings and never imports the generator) is
`tests/test_add_sub_number_sentence_oracle.py`.
"""

from __future__ import annotations

import re
from collections import Counter

import pytest

import mathgen.generators.add_sub_number_sentence_build as gen
from mathgen import make
from mathgen.core import VerificationError
from mathgen.grading.typed import grade_mcq, mcq_options
from mathgen.skills import load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "add_sub_number_sentence"
TIERS = ("easy", "medium", "hard")


def _all(tier, seeds=range(60)):
    return [make(TOPIC, seed=s, difficulty=tier) for s in seeds]


@pytest.mark.parametrize("tier", TIERS)
def test_no_bad_text_patterns(tier):
    for q in _all(tier, range(40)):
        assert find_text_defects(q) == [], f"{tier}: {find_text_defects(q)} for {q.text!r}"


@pytest.mark.parametrize("tier", TIERS)
def test_prompt_variety_at_least_3_shapes_and_20_distinct_questions(tier):
    qs = _all(tier)
    assert len({prompt_shape(q) for q in qs}) >= 3
    assert len({q.text for q in qs}) >= 20


@pytest.mark.parametrize("tier", TIERS)
def test_determinism(tier):
    for seed in (0, 1, 7, 59):
        assert make(TOPIC, seed=seed, difficulty=tier).to_json() == make(TOPIC, seed=seed, difficulty=tier).to_json()


@pytest.mark.parametrize("tier", TIERS)
def test_options_are_three_to_five_well_formed_distinct_sentences(tier):
    sizes = Counter()
    for q in _all(tier, range(200)):
        options = [q.answer_text, *q.distractors]
        sizes[len(options)] += 1
        assert 3 <= len(options) <= 5
        assert len(set(options)) == len(options)
        for option in options:
            assert re.fullmatch(r"(\d+|□) [+-] (\d+|□) = (\d+|□)", option), option
            assert option.count("□") == 1
    assert set(sizes) == {3, 4, 5}, f"{tier}: option counts seen {dict(sizes)}"


@pytest.mark.parametrize("tier", TIERS)
def test_no_distractor_solves_to_the_answer_and_the_key_does(tier):
    for q in _all(tier, range(200)):
        key_value = gen._solve_text(q.answer_text)
        assert key_value is not None
        for d in q.distractors:
            assert gen._solve_text(d) != key_value, f"{tier}: {d!r} also solves to {key_value} ({q.text!r})"


@pytest.mark.parametrize("tier", TIERS)
def test_distractors_include_both_wrong_value_and_impossible_sentences(tier):
    wrong_value = impossible = 0
    for q in _all(tier, range(100)):
        for d in q.distractors:
            if gen._solve_text(d) is None:
                impossible += 1
            else:
                wrong_value += 1
    assert wrong_value > 40 and impossible > 40


@pytest.mark.parametrize("tier", TIERS)
def test_key_is_not_given_away_by_its_form(tier):
    """Over 300 draws the key's blank sits in every position, uses both operations and the key is not the only option
    of its kind."""
    positions, ops = Counter(), Counter()
    for q in _all(tier, range(300)):
        m = re.fullmatch(r"(\S+) ([+-]) (\S+) = (\S+)", q.answer_text)
        positions[[m[1], m[3], m[4]].index("□")] += 1
        ops[m[2]] += 1
    assert set(positions) == {0, 1, 2} and min(positions.values()) >= 30, positions
    assert set(ops) == {"+", "-"} and min(ops.values()) >= 60, ops


@pytest.mark.parametrize("tier", TIERS)
def test_key_letter_position_is_balanced_for_each_option_count(tier):
    by_count: dict[int, Counter] = {}
    for s in range(600):
        q = make(TOPIC, seed=s, difficulty=tier)
        options = mcq_options(q)
        by_count.setdefault(len(options), Counter())[options.index(q.answer_text)] += 1
    for n_options, counts in by_count.items():
        total = sum(counts.values())
        assert set(counts) == set(range(n_options)), f"{tier}: {n_options} options, key positions {dict(counts)}"
        for idx, n in counts.items():
            assert 0.6 / n_options <= n / total <= 1.6 / n_options, (
                f"{tier}: {n_options} options, index {idx}: {n}/{total}"
            )


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_accepts_the_key_and_rejects_every_other_option(tier):
    for q in _all(tier, range(60)):
        options = mcq_options(q)
        for i, option in enumerate(options):
            assert grade_mcq(q, str(i)).correct == (option == q.answer_text)
        assert not grade_mcq(q, "").correct
        assert not grade_mcq(q, str(len(options))).correct


def test_answer_latex_is_plain_text():
    for tier in TIERS:
        for q in _all(tier, range(20)):
            assert q.answer_latex == f"plain:{q.answer_text}"


# -- tier progression ------------------------------------------------------------------------------------------


def test_easy_unknown_is_the_end_result():
    """easy stories name every number they give and never hide the start or the change."""
    for q in _all("easy"):
        story = re.split(r" (?:Which|Choose)", q.text)[0]
        assert not re.search(r"\b[Ss]ome (?:more|away|of them|\w+s were)|start|at first", story), story


def test_medium_unknown_is_the_change():
    for q in _all("medium"):
        assert re.search(r"\b[Ss]ome\b", q.text), q.text
        assert not re.search(r"start with|at first", q.text), q.text


def test_hard_unknown_is_the_start_or_a_comparison():
    for q in _all("hard"):
        assert re.search(r"start with|at first|more .* than|fewer", q.text) or "who has" in q.text, q.text


def test_numbers_grow_with_the_tier():
    tops = {}
    for tier in TIERS:
        biggest = 0
        for q in _all(tier, range(100)):
            story = q.text.split("Which")[0].split("Choose")[0]
            biggest = max(biggest, *(int(n) for n in re.findall(r"\d+", story)))
        tops[tier] = biggest
    assert tops["easy"] <= 20 < tops["medium"] <= 50 < tops["hard"] <= 99


# -- the in-generator solver -----------------------------------------------------------------------------------


def test_solver_reads_every_blank_position():
    assert gen._solve_text("14 + □ = 31") == 17
    assert gen._solve_text("□ + 14 = 31") == 17
    assert gen._solve_text("14 + 17 = □") == 31
    assert gen._solve_text("31 - 14 = □") == 17
    assert gen._solve_text("31 - □ = 14") == 17
    assert gen._solve_text("□ - 14 = 31") == 45


def test_solver_returns_none_when_no_whole_number_fits():
    assert gen._solve_text("14 - 31 = □") is None
    assert gen._solve_text("31 + □ = 14") is None


def test_solver_fires_on_a_malformed_sentence():
    with pytest.raises(VerificationError):
        gen._solve_text("14 + 17 = 31")
    with pytest.raises(VerificationError):
        gen._solve_text("14 x □ = 31")


def test_equal_sentences_share_a_canonical_form():
    assert gen._canonical(("+", 14, None, 31)) == gen._canonical(("+", None, 14, 31))
    assert gen._canonical(("-", 31, 14, None)) == gen._canonical(("-", 31, None, 14))
    assert gen._canonical(("+", 14, None, 31)) != gen._canonical(("-", 31, 14, None))


def test_class_pools_never_overlap_and_the_matching_class_is_correct():
    for ka, kb, answer in ((14, 31, 17), (14, 31, 45), (6, 9, 15), (40, 7, 33)):
        matching, wrong, impossible = gen._classes(ka, kb, answer)
        for members in matching:
            assert all(gen._solve_cells(*s) == answer for s in members)
        for members in wrong:
            assert all(gen._solve_cells(*s) not in (None, answer) for s in members)
        for members in impossible:
            assert all(gen._solve_cells(*s) is None for s in members)


# -- skill load ------------------------------------------------------------------------------------------------


class TestSkillLoad:
    def test_skill_loads_with_expected_shape(self):
        _, skills = load_skills()
        skill = next(s for s in skills if s.id == "f10-add-subtract-number-sentences")
        assert skill.domain == "equations"
        assert skill.answer == "mcq"
        assert skill.placement_eligible is False
        assert skill.prerequisites == [
            "add-within-100",
            "f10-addition-facts-related-subtraction",
            "subtract-within-100",
        ]
        assert [s.difficulty for s in skill.sources] == ["easy", "medium", "hard"]
        stages = set()
        for source in skill.sources:
            assert source.topic == TOPIC
            stages.update(source.stages)
        assert stages == {0, 1, 2, 3, 4, 5, 6, "maintenance"}
