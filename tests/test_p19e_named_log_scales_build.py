"""Tests `p19e_named_log_scale_value`: values on named logarithmic scales, `answer: number`.

`p19e_named_log_scale_value` (`answer: number`)."""

import re
from collections import Counter

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make
import mathgen.generators.p19e_named_log_scales_build as m  # noqa: F401 -- registers the topic

SKILL_ID = "senior-interpret-and-use-logarithmic-scales-such-as-decibels"
TOPIC = "p19e_named_log_scale_value"
TIERS = ["easy", "medium", "hard", "exam"]
SEEDS = range(60)

_domains, _skills = load_skills()
SKILL = next(s for s in _skills if s.id == SKILL_ID)
SOURCES = {src.difficulty: src for src in SKILL.sources}

_BAD_PHRASE_RE = re.compile(r"find (?:the )?\[H\+\]\b", re.I)


def _make(difficulty, seed):
    src = SOURCES[difficulty]
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


@pytest.mark.parametrize("difficulty", TIERS)
def test_real_grader_accepts_canonical_and_rejects_every_distractor(difficulty):
    for seed in SEEDS:
        q = _make(difficulty, seed)
        assert grade_number(q, q.answer_text).correct is True, (difficulty, seed, q.text)
        for d in q.distractors:
            assert grade_number(q, d).correct is False, (difficulty, seed, q.text, d)


@pytest.mark.parametrize("difficulty", TIERS)
def test_real_grader_rejects_a_deliberately_wrong_answer(difficulty):
    for seed in SEEDS:
        q = _make(difficulty, seed)
        target = float(q.answer_text)
        wrong = target + 7.0
        assert grade_number(q, f"{wrong:.2f}").correct is False


def test_skill_loads_with_expected_shape():
    assert SKILL.answer == "number"
    assert SKILL.placement_eligible is False
    assert SKILL.domain == "functions-sequences"
    assert SKILL.prerequisites == ["function-notation"]
    assert {src.difficulty for src in SKILL.sources} == {"easy", "medium", "hard", "exam"}
    exam_src = next(s for s in SKILL.sources if s.difficulty == "exam")
    assert exam_src.stages == ["exam"]


@pytest.mark.parametrize("difficulty", TIERS)
def test_two_or_more_distractors_at_200_seeds_default_and_tier(difficulty):
    def _share(diff):
        n_ok = 0
        for seed in range(200):
            q = make(TOPIC, seed=seed, difficulty=diff)
            if len(q.distractors) >= 2:
                n_ok += 1
        return n_ok / 200

    assert _share(difficulty) >= 0.85, f"{difficulty}: <2 distractors too often"
    if difficulty == "easy":
        assert _share(None) >= 0.85, "default: <2 distractors too often"


@pytest.mark.parametrize("difficulty", TIERS)
def test_answer_concentration_at_200_seeds(difficulty):
    counts = Counter()
    for seed in range(200):
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        counts[q.answer_text] += 1
    total = sum(counts.values())
    top_share = counts.most_common(1)[0][1] / total
    assert top_share <= 0.15, f"{difficulty}: top answer share {top_share:.3f} exceeds 0.15"


@pytest.mark.parametrize("difficulty", TIERS)
def test_all_four_scales_appear(difficulty):
    """Every tier's 200-seed sweep touches all four named scales -- checked
    by a scale-specific fingerprint in the rendered text."""
    seen = set()
    for seed in range(200):
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        if "decibel" in q.text or "L = 10 log10" in q.text:
            seen.add("db")
        if "Richter" in q.text or "M = log10" in q.text:
            seen.add("richter")
        if "octave" in q.text or "log2" in q.text:
            seen.add("octave")
        if "pH" in q.text:
            seen.add("ph")
    assert seen == {"db", "richter", "octave", "ph"}, (difficulty, seen)


@pytest.mark.parametrize("difficulty", TIERS)
def test_forward_computation_only_never_solves_for_input(difficulty):
    for seed in SEEDS:
        q = _make(difficulty, seed)
        assert not _BAD_PHRASE_RE.search(q.text), q.text


@pytest.mark.parametrize("difficulty", ["medium", "hard", "exam"])
def test_instruction_phrase_in_text_and_latex(difficulty):
    for seed in range(20):
        q = _make(difficulty, seed)
        assert "decimal place" in q.text
        assert "decimal place" in q.latex


@pytest.mark.parametrize("difficulty", TIERS)
def test_answers_and_distractors_formatted_alike(difficulty):
    for seed in range(30):
        q = _make(difficulty, seed)
        answer_dp = len(q.answer_text.split(".", 1)[1]) if "." in q.answer_text else 0
        for d in q.distractors:
            d_dp = len(d.split(".", 1)[1]) if "." in d else 0
            assert d_dp == answer_dp, (difficulty, seed, q.answer_text, d)


def test_easy_answers_are_exact_integers_graded_exactly():
    for seed in range(60):
        q = _make("easy", seed)
        assert "." not in q.answer_text, q.answer_text
        for d in q.distractors:
            assert "." not in d, d


def test_display_helper_directly():
    from fractions import Fraction
    from mathgen.core import round_half_up
    from mathgen.generators.p19e_named_log_scales_build import _display

    assert _display(23.0, 0) == "23"
    assert _display(round_half_up(Fraction("21.83"), 1), 1) == "21.8"


def test_check_fires_on_a_corrupted_recompute():
    from mathgen.core import VerificationError, check

    with pytest.raises(VerificationError):
        check(abs(10 * 3 - 40) < 1e-9, "easy(db): recompute disagrees")


def test_easy_and_hard_distinct_prompt_counts_clear_a_higher_floor_at_200_seeds():
    """Easy was 96/200 and hard was 78/200 distinct prompts (too many repeats from a narrow value pool) --
    widened pools must clear well above those floors."""
    for difficulty, floor in (("easy", 100), ("hard", 105)):
        texts = {make(TOPIC, seed=seed, difficulty=difficulty).text for seed in range(200)}
        assert len(texts) >= floor, (difficulty, len(texts))


def test_medium_octave_frequencies_are_not_always_the_same_pair():
    """The medium-tier octave draw used to repeat
    f = 990 Hz / 1100 Hz almost exclusively. Over 200 seeds, no single
    octave frequency should dominate the octave-scale draws."""
    freqs = Counter()
    for seed in range(200):
        q = make(TOPIC, seed=seed, difficulty="medium")
        m = re.search(r"f = ([\d.]+) Hz", q.text)
        if m and ("octave" in q.text.lower() or "log2" in q.text):
            freqs[m.group(1)] += 1
    total = sum(freqs.values())
    assert total > 0
    assert len(freqs) >= 6, freqs
    top_share = freqs.most_common(1)[0][1] / total
    assert top_share <= 0.35, freqs


def test_exam_octave_frequencies_are_whole_hertz():
    """Exam-tier octave frequencies (f1, f2) must be plain whole hertz -- never a visibly back-constructed
    decimal like f2 = 6453.33 Hz reverse-engineered from a target 'nice' ratio."""
    n_octave = 0
    for seed in range(400):
        q = make(TOPIC, seed=seed, difficulty="exam")
        if "octave" not in q.text.lower():
            continue
        n_octave += 1
        for m in re.finditer(r"f[12] = ([\d.]+) Hz", q.text):
            assert "." not in m.group(1), (seed, q.text)
    assert n_octave > 0
