"""Exam source for `engineering-notation` (`topic=engineering_notation`,
`mathgen/generators/p8a_notation_build.py`), `variant="sum_difference"`.

The independent oracle below (`_oracle_exam`) parses `question.text` alone and re-derives the
engineering-notation answer from scratch with plain `decimal.Decimal` and a hand-copied SI-prefix table -- it
imports nothing from `mathgen.generators.p8a_notation_build`, so a slip in the generator's own
`_verify_engineering_pair`/`_verify_sig_fig_rounding` calls cannot also be baked into the check that is
supposed to catch it. `test_oracle_catches_a_corrupted_answer` proves the oracle is not vacuously true.
"""

from __future__ import annotations

import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from mathgen.grading.typed import grade_mcq, mcq_options
from mathgen.skills import load_skills
from mathgen import make
from mathgen.text_quality import prompt_shape

ROOT = Path(__file__).resolve().parents[1]
TOPIC = "engineering_notation"
SEEDS = range(40)
SHAPE_SEEDS = range(60)

# --- independent oracle ------------------------------------------------------

_TEXT_RE = re.compile(r": (\d\.\d{3}) × 10\^(-?\d+) ([+-]) (\d\.\d{3}) × 10\^(-?\d+)\.")
_OPTION_RE = re.compile(r"^(\d+(?:\.\d+)?) × 10\^(-?\d+) \(\1 (\w+)\)$")
# Hand-copied from the SI system's own definition, not imported from the
# generator -- the same discipline the pythagorean oracle used for the
# Pythagorean theorem itself.
_SI_INDEPENDENT = {
    -12: "pico",
    -9: "nano",
    -6: "micro",
    -3: "milli",
    3: "kilo",
    6: "mega",
    9: "giga",
    12: "tera",
}


def _oracle_exam(text: str) -> str:
    """Re-derive the engineering-notation answer from the rendered prompt
    alone: parse the two scientific-notation terms and the operator, align
    and combine them exactly in `Decimal`, regroup to the nearest lower
    multiple-of-3 exponent, and round the mantissa to 4 significant figures
    with `ROUND_HALF_UP` -- the taught convention, reimplemented here rather
    than called from the generator."""
    match = _TEXT_RE.search(text)
    assert match, f"could not parse exam-tier prompt: {text!r}"
    a_str, exp_a_str, op, b_str, exp_b_str = match.groups()
    a = Decimal(a_str).scaleb(int(exp_a_str))
    b = Decimal(b_str).scaleb(int(exp_b_str))
    value = a + b if op == "+" else a - b

    sci_exp = value.adjusted()
    eng_exp = 3 * (sci_exp // 3)
    mantissa = value.scaleb(-eng_exp)

    q_exp = mantissa.adjusted() - 4 + 1
    quantum = Decimal(1).scaleb(q_exp)
    rounded = mantissa.quantize(quantum, rounding=ROUND_HALF_UP)
    if rounded >= 1000:
        eng_exp += 3
        rounded = rounded.scaleb(-3)

    mantissa_str = format(rounded, "f")
    prefix = _SI_INDEPENDENT[eng_exp]
    return f"{mantissa_str} × 10^{eng_exp} ({mantissa_str} {prefix})"


def test_exam_tier_oracle_agrees():
    checked = 0
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="exam", variant="sum_difference")
        expected = _oracle_exam(q.text)
        assert expected == q.answer_text, (
            f"seed={seed}: oracle recomputed {expected!r} from the prompt, "
            f"generator answer_text is {q.answer_text!r}\n{q.text}"
        )
        checked += 1
    assert checked == len(SEEDS)


def test_oracle_catches_a_corrupted_answer():
    """Proves the oracle is not vacuously true: feed it a real question's
    prompt paired with a deliberately wrong `answer_text` and confirm it
    disagrees."""
    q = make(TOPIC, seed=0, difficulty="exam", variant="sum_difference")
    corrupted = "9.999 × 10^99 (9.999 unobtainium)"
    assert _oracle_exam(q.text) != corrupted
    # A genuinely plausible corruption too: take only the dominant term's
    # own regrouping (the "ignores-smaller-term" misconception)
    # and confirm the oracle still disagrees with it.
    for d in q.distractors:
        assert _oracle_exam(q.text) != d


# --- determinism -------------------------------------------------------------


def test_exam_tier_deterministic_across_seeds():
    for seed in SEEDS:
        q1 = make(TOPIC, seed=seed, difficulty="exam", variant="sum_difference")
        q2 = make(TOPIC, seed=seed, difficulty="exam", variant="sum_difference")
        assert q1.to_json() == q2.to_json()


# --- distractor hygiene --------------------------------------------------


def test_exam_tier_distractors_distinct_and_never_equal_answer():
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="exam", variant="sum_difference")
        assert q.answer_text not in q.distractors
        assert len(q.distractors) == len(set(q.distractors))
        assert len(q.distractors) >= 1


# --- formatting: every option states its own value to 4 significant figures


def test_exam_tier_options_state_a_precise_mantissa():
    """Every option (answer and every distractor) either has the full
    "mantissa × 10^exp (mantissa prefix)" shape -- matching `_OPTION_RE`,
    the SAME shape `_option_text`/`_option_text_sf` both already use for
    every other tier -- or is the bare scientific-notation form the
    "scientific-notation-exponent-grouping" misconception always was for
    this skill's other three tiers (no parenthetical prefix). Either way the
    mantissa itself must show at least 4 significant figures' worth of
    digits (never collapsed to a bare integer, which would silently
    understate the stated precision)."""
    digit_re = re.compile(r"^\d(\.\d+)?$")
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="exam", variant="sum_difference")
        for text in [q.answer_text, *q.distractors]:
            full = _OPTION_RE.match(text)
            if full:
                mantissa_str = full.group(1)
            else:
                # bare scientific-notation form: "m × 10^exp"
                bare = re.match(r"^(\d(?:\.\d+)?) × 10\^-?\d+$", text)
                assert bare, f"seed={seed}: option {text!r} matches neither known option shape"
                mantissa_str = bare.group(1)
            digits = mantissa_str.replace(".", "").lstrip("0") or "0"
            assert len(digits) >= 4, f"seed={seed}: option {text!r} does not state 4 significant figures"
        # The canonical answer itself is always the full parenthesised
        # shape (it is what `answer_text`/mcq options are graded against).
        assert _OPTION_RE.match(q.answer_text), (
            f"seed={seed}: answer_text {q.answer_text!r} is not in the mantissa/exponent/prefix shape"
        )


# --- answer-kind check: the skill's own declared kind (mcq) --------------


def test_exam_tier_answer_is_a_string_matching_answer_text():
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="exam", variant="sum_difference")
        assert isinstance(q.answer, str)
        assert q.answer == q.answer_text


# --- variety: at least 20 distinct rendered questions over 60 seeds ---------


def test_exam_tier_at_least_20_distinct_questions_over_60_seeds():
    texts = {make(TOPIC, seed=seed, difficulty="exam", variant="sum_difference").text for seed in SHAPE_SEEDS}
    assert len(texts) >= 20, f"only {len(texts)} distinct exam-tier questions over 60 seeds"


def test_exam_tier_clears_three_distinct_shapes():
    shapes = {prompt_shape(make(TOPIC, seed=seed, difficulty="exam", variant="sum_difference")) for seed in SHAPE_SEEDS}
    assert len(shapes) >= 3, f"only {len(shapes)} distinct prompt shapes over 60 seeds"


# --- real grader accept/reject (mcq) ---------------------------------------


def test_exam_tier_real_grader_accepts_canonical_and_rejects_distractors():
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="exam", variant="sum_difference")
        options = mcq_options(q)
        correct_idx = options.index(q.answer_text)
        assert grade_mcq(q, correct_idx).correct
        for i, opt in enumerate(options):
            if opt == q.answer_text:
                continue
            assert not grade_mcq(q, i).correct


def test_exam_tier_real_grader_rejects_a_deliberately_wrong_index():
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="exam", variant="sum_difference")
        options = mcq_options(q)
        wrong_idx = next(i for i, opt in enumerate(options) if opt != q.answer_text)
        assert not grade_mcq(q, wrong_idx).correct


# --- ignores-smaller-term misconception is genuinely exercisable -----------


def test_ignores_smaller_term_distractor_is_always_a_different_value_from_the_answer():
    """The generator's own resample loop guarantees the "take only the
    dominant term" misconception is a genuinely different VALUE from the
    correct answer on every seed (not merely a different string) -- re-
    derive the dominant-only value independently here (parse the prompt,
    take only the larger-magnitude term, regroup and round it exactly like
    `_oracle_exam` does for the true sum) and confirm it never equals the
    oracle's own combined-sum answer."""
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="exam", variant="sum_difference")
        match = _TEXT_RE.search(q.text)
        assert match
        a_str, exp_a_str, _op, _b_str, _exp_b_str = match.groups()
        dominant_value = Decimal(a_str).scaleb(int(exp_a_str))
        sci_exp = dominant_value.adjusted()
        eng_exp = 3 * (sci_exp // 3)
        mantissa = dominant_value.scaleb(-eng_exp)
        q_exp = mantissa.adjusted() - 4 + 1
        quantum = Decimal(1).scaleb(q_exp)
        rounded = mantissa.quantize(quantum, rounding=ROUND_HALF_UP)
        mantissa_str = format(rounded, "f")
        dominant_only_text = f"{mantissa_str} × 10^{eng_exp} ({mantissa_str} {_SI_INDEPENDENT[eng_exp]})"
        assert dominant_only_text != q.answer_text, (
            f"seed={seed}: taking only the dominant term reproduces the "
            f"correct answer -- the smaller term never moved it"
        )


# --- skill contract -----------------------------------------------------


def test_skill_contract_has_four_sources_including_exam():
    _, skills = load_skills(ROOT / "skills")
    by_id = {s.id: s for s in skills}
    skill = by_id["engineering-notation"]

    assert skill.answer == "mcq"
    assert skill.status == "active"
    assert skill.placement_eligible is False
    assert [source.difficulty for source in skill.sources] == ["easy", "medium", "hard", "exam"]
    assert {source.topic for source in skill.sources} == {TOPIC}

    exam_source = skill.sources[3]
    assert exam_source.stages == ["exam"]
    assert exam_source.params.get("variant") == "sum_difference"

    misconception_ids = {m.id for m in skill.misconceptions}
    assert "ignores-smaller-term" in misconception_ids


# default-params behaviour of the other tiers is untouched -


def test_other_variants_are_unaffected_by_the_new_variant():
    for seed in range(10):
        q_easy = make(TOPIC, seed=seed, difficulty="easy", variant="large_positive")
        q_medium = make(TOPIC, seed=seed, difficulty="medium", variant="small_value")
        q_hard = make(TOPIC, seed=seed, difficulty="hard", variant="wide_exponent")
        assert "significant figures" not in q_easy.text
        assert "significant figures" not in q_medium.text
        assert "significant figures" not in q_hard.text
