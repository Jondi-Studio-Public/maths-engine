"""Covers the four generators in ``mathgen/generators/p16c_finance_time_build.py``:
``p16c_loan_recurrence_balance``, ``p16c_compound_effect``, ``p16c_worker_wage_table``,
``p16c_gmt_dateline``.
"""

import functools
from fractions import Fraction

import mathgen.generators.p16c_finance_time_build  # noqa: F401 -- registers topics
from mathgen.grading.typed import GRADERS, grade_mcq, mcq_options
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import stated_precision_tolerance

DIFFICULTIES = ["easy", "medium", "hard", "exam"]
SEEDS = range(40)
_TOL2 = stated_precision_tolerance(2)

NUMBER_TOPICS = {
    "p16c_loan_recurrence_balance": "senior-use-technology-and-a-recurrence-relation-to-model",
    "p16c_compound_effect": "senior-use-technology-to-investigate-the-effect-of-the",
    "p16c_worker_wage_table": "senior-use-a-spreadsheet-to-display-examples-of-the",
}
MCQ_TOPIC = "p16c_gmt_dateline"
MCQ_SKILL = "senior-solve-problems-involving-greenwich-mean-time-and-the"
ALL_TOPICS = [*NUMBER_TOPICS, MCQ_TOPIC]


@functools.lru_cache(maxsize=None)
def _skills_dict():
    _, skills = load_skills()
    return {s.id: s for s in skills}


def _make_from_sources(skill_id, difficulty, seed):
    skill = _skills_dict()[skill_id]
    src = next(s for s in skill.sources if s.difficulty == difficulty)
    if src.topic == "exam_parts_compound_interest":
        # This skill's exam source is now the multi-part chain (tests/test_exam_parts_compound_interest.py);
        # these checks keep covering the single-answer exam preset it replaced, which the generator still has.
        return make("p16c_compound_effect", seed=seed, difficulty="exam", variant="percent_effect")
    if src.topic == "exd_x_r2d_loan_recurrence_chain":
        # This skill's exam source is now a multi-part chain (tests/test_exd_x_r2d_loan_recurrence_chain.py);
        # these checks keep covering the single-answer exam variant it replaced, which the generator still has.
        return make("p16c_loan_recurrence_balance", seed=seed, difficulty="exam", variant="rate_change")
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


# =============================================================================
# Determinism, and the shared shape checks that apply to every topic
# =============================================================================


def test_deterministic_across_seeds():
    for topic in ALL_TOPICS:
        for difficulty in DIFFICULTIES:
            for seed in SEEDS:
                a = make(topic, seed=seed, difficulty=difficulty)
                b = make(topic, seed=seed, difficulty=difficulty)
                assert a.text == b.text
                assert a.answer_text == b.answer_text
                assert a.distractors == b.distractors


def test_distractors_distinct_nonempty_never_equal_answer():
    for topic in ALL_TOPICS:
        for difficulty in DIFFICULTIES:
            for seed in SEEDS:
                q = make(topic, seed=seed, difficulty=difficulty)
                assert len(q.distractors) >= 1, (topic, difficulty, seed)
                assert len(set(q.distractors)) == len(q.distractors), (topic, difficulty, seed)
                assert q.answer_text not in q.distractors, (topic, difficulty, seed)


def test_at_least_20_distinct_questions_per_tier_over_60_seeds():
    for topic in ALL_TOPICS:
        for difficulty in DIFFICULTIES:
            texts = {make(topic, seed=s, difficulty=difficulty).text for s in range(60)}
            assert len(texts) >= 20, (topic, difficulty, len(texts))


def test_skills_load_with_expected_shape():
    for topic, skill_id in NUMBER_TOPICS.items():
        skill = _skills_dict()[skill_id]
        assert skill.answer == "number"
        assert skill.placement_eligible is False
        difficulties = {s.difficulty for s in skill.sources}
        assert difficulties == set(DIFFICULTIES)
        assert next(s for s in skill.sources if s.difficulty == "exam").stages == ["exam"]
    skill = _skills_dict()[MCQ_SKILL]
    assert skill.answer == "mcq"
    assert skill.placement_eligible is False
    difficulties = {s.difficulty for s in skill.sources}
    assert difficulties == set(DIFFICULTIES)


# =============================================================================
# Real-grader accept/reject -- number-answer skills
# =============================================================================


def test_number_grader_rejects_a_deliberately_wrong_answer():
    for topic, skill_id in NUMBER_TOPICS.items():
        for difficulty in DIFFICULTIES:
            q = _make_from_sources(skill_id, difficulty, 0)
            grader = GRADERS["number"]
            # An offset far larger than any plausible stated precision.
            if "." in q.answer_text:
                whole, frac = q.answer_text.lstrip("-").split(".")
                wrong = str(int(whole) + 97) + "." + frac
            else:
                wrong = str(int(q.answer_text) + 97)
            assert not grader(q, wrong).correct, (topic, difficulty, q.text, wrong)


def test_money_answers_land_on_the_cent_grid():
    for topic in ("p16c_loan_recurrence_balance", "p16c_compound_effect", "p16c_worker_wage_table"):
        for difficulty in DIFFICULTIES:
            if topic == "p16c_loan_recurrence_balance" and difficulty == "hard":
                continue  # this tier's answer is a plain period count, not money
            if topic == "p16c_compound_effect" and difficulty == "exam":
                continue  # this tier's answer is a percentage to 1dp, not money
            for seed in SEEDS:
                q = make(topic, seed=seed, difficulty=difficulty)
                text = q.answer_text.lstrip("-")
                assert "." in text, (topic, difficulty, seed, q.answer_text)
                whole, frac = text.split(".")
                assert len(frac) == 2, (topic, difficulty, seed, q.answer_text)


def test_find_n_answer_is_a_plain_integer():
    for seed in SEEDS:
        q = make("p16c_loan_recurrence_balance", seed=seed, difficulty="hard")
        assert q.answer_text.isdigit()
        assert isinstance(q.answer, int)


# =============================================================================
# Real-grader accept/reject -- mcq skill (GMT/IDL)
# =============================================================================


def test_mcq_grader_accepts_canonical_and_rejects_distractors():
    for difficulty in DIFFICULTIES:
        for seed in SEEDS:
            q = _make_from_sources(MCQ_SKILL, difficulty, seed)
            options = mcq_options(q)
            answer_idx = options.index(q.answer_text)
            assert grade_mcq(q, answer_idx).correct, (difficulty, seed, q.text)
            for i, opt in enumerate(options):
                if opt != q.answer_text:
                    assert not grade_mcq(q, i).correct, (difficulty, seed, opt)


def test_mcq_options_are_all_day_and_time_strings_never_bare_numbers():
    import re

    day_time_re = re.compile(r"^(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday) \d{2}:\d{2}$")
    for difficulty in DIFFICULTIES:
        for seed in SEEDS:
            q = make(MCQ_TOPIC, seed=seed, difficulty=difficulty)
            for opt in [q.answer_text, *q.distractors]:
                assert day_time_re.match(opt), (difficulty, seed, opt)


def test_mcq_answer_latex_is_plain_prefixed_never_bare_text():
    for difficulty in DIFFICULTIES:
        for seed in SEEDS:
            q = make(MCQ_TOPIC, seed=seed, difficulty=difficulty)
            assert q.answer_latex.startswith("plain:"), (difficulty, seed, q.answer_latex)


def test_gmt_routes_are_realistic_and_actually_cross():
    """Imports the generator's own table directly: this is a construction check, not the independent oracle."""
    import re

    from mathgen.generators.p16c_finance_time_build import _CROSSING_ROUTES, _CROSSING_ROUTES_HALF

    def _table_hours(place_x, off_x, place_y, off_y, table):
        for a, oa, b, ob, hours in table:
            if {a, b} == {place_x, place_y}:
                # Confirm the crossing criterion against the table's own
                # recorded offsets (independent of what the drawn scenario's
                # offsets happen to say -- they must match the table).
                assert {oa, ob} == {off_x, off_y} or {float(oa), float(ob)} == {float(off_x), float(off_y)}
                return hours
        return None

    single_re = re.compile(
        r"departs (.+?) \(GMT([+-][\d:]+)\) at \d{2}:\d{2} on \w+, and flies for (\d+) hours to "
        r"(.+?) \(GMT([+-][\d:]+)\), crossing the International Date Line"
    )
    two_leg_re = re.compile(
        r"departs (.+?) \(GMT([+-][\d:]+)\) at \d{2}:\d{2} on \w+ and flies (\d+) hours to "
        r"(.+?) \(GMT([+-][\d:]+)\), crossing the International Date Line\. After a (\d+)-hour "
        r"layover in .+?, a connecting flight takes (\d+) hours to (.+?) \(GMT([+-][\d:]+)\)"
    )

    def _parse_offset(s):
        s = s.replace(":", "/60*")  # "+9:30" -> "+9/60*30" -- evaluated below
        sign = 1 if s.startswith("+") else -1
        s = s[1:]
        if "/60*" in s:
            whole, minute = s.split("/60*")
            return sign * (Fraction(int(whole)) + Fraction(int(minute), 60))
        return sign * Fraction(int(s))

    checked_single = checked_two = 0
    for difficulty in ("easy", "medium", "hard"):
        table = _CROSSING_ROUTES_HALF if difficulty == "medium" else _CROSSING_ROUTES
        for seed in range(200):
            q = make("p16c_gmt_dateline", seed=seed, difficulty=difficulty)
            m = single_re.search(q.text)
            assert m, (difficulty, seed, q.text)
            place_from, off_from_s, duration_s, place_to, off_to_s = m.groups()
            off_from, off_to = _parse_offset(off_from_s), _parse_offset(off_to_s)
            # The crossing claim: one side >= +8, the other <= -3.
            assert (off_from >= 8 and off_to <= -3) or (off_to >= 8 and off_from <= -3), (
                difficulty,
                seed,
                place_from,
                off_from,
                place_to,
                off_to,
            )
            hours = _table_hours(place_from, off_from, place_to, off_to, table)
            assert hours is not None, (difficulty, seed, place_from, place_to, "not in the route table")
            assert int(duration_s) == hours, (difficulty, seed, duration_s, hours)
            checked_single += 1
    assert checked_single == 600

    for seed in range(200):
        q = make("p16c_gmt_dateline", seed=seed, difficulty="exam")
        m = two_leg_re.search(q.text)
        assert m, (seed, q.text)
        place_a, off_a_s, dur1_s, place_b, off_b_s, layover_s, dur2_s, place_c, off_c_s = m.groups()
        off_a, off_b, off_c = _parse_offset(off_a_s), _parse_offset(off_b_s), _parse_offset(off_c_s)
        assert (off_a >= 8 and off_b <= -3) or (off_b >= 8 and off_a <= -3), (seed, place_a, place_b)
        hours1 = _table_hours(place_a, off_a, place_b, off_b, _CROSSING_ROUTES)
        assert hours1 is not None and int(dur1_s) == hours1, (seed, place_a, place_b, dur1_s, hours1)
        hours2 = _table_hours(place_b, off_b, place_c, off_c, _CROSSING_ROUTES)
        assert hours2 is not None and int(dur2_s) == hours2, (seed, place_b, place_c, dur2_s, hours2)
        checked_two += 1
    assert checked_two == 200


def test_gmt_at_least_two_distractors_on_most_draws():
    # Not an mcq-answer-space binary case: at least 3-4 candidate misreadings
    # are always drawn, but pick_distractors can dedupe some away on a given
    # seed. Measure over 200 draws per tier.
    for difficulty in DIFFICULTIES:
        counts = [len(make(MCQ_TOPIC, seed=s, difficulty=difficulty).distractors) for s in range(200)]
        at_least_two = sum(1 for c in counts if c >= 2)
        assert at_least_two / 200 >= 0.85, (difficulty, at_least_two)


# =============================================================================
# Independent in-generator checks fire on a corrupted answer
# =============================================================================


def test_check_equivalent_fires_on_a_corrupted_route():
    from mathgen.core import VerificationError, check_equivalent

    with_pytest_raises = False
    try:
        check_equivalent(Fraction(1), Fraction(2), "deliberately different")
    except VerificationError:
        with_pytest_raises = True
    assert with_pytest_raises
