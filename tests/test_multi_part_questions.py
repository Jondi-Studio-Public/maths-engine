"""The multi-part question model: `Part`, `make()` building parts, and marking with follow-through.

See `docs/design/multi-part-questions.md`. The generators here are registered for one test at a time and
removed afterwards, so no library-wide sweep ever sees them.
"""

from __future__ import annotations

import json
from fractions import Fraction

import pytest

from mathgen import engine, make
from mathgen.answers import Distractor
from mathgen.core import PRESETS, REGISTRY, Part, VerificationError, generator
from mathgen.grading import GRADERS, deadline, grade
from mathgen.grading.parts import follow_through_target, reply_value, split_reply

TOPIC = "test_multi_part_circle"


def _circle(rng, overrides=None):
    # A circle of radius 6.2 cm: area 120.76 cm² (2 dp), circumference 38.96 -> 39.0 (1 dp).
    parts = [
        dict(
            prompt="Find the radius of the circle, correct to 1 decimal place.",
            answer_kind="number",
            answer=6.2,
            answer_text="6.2",
            marks=2,
            working=["r = √(120.76 ÷ π) = 6.2"],
            distractors=["12.4"],
        ),
        dict(
            prompt="Hence find the circumference, correct to 1 decimal place.",
            answer_kind="number",
            answer=39.0,
            answer_text="39.0",
            follow_through="2*pi*part_a",
            working=["C = 2πr = 39.0"],
            distractors=["19.5"],
        ),
        dict(
            prompt="Write an expression for the area of a ring of inner radius r and outer radius equal to your radius.",
            answer_kind="expression",
            answer="pi*(961/25 - r**2)",
            answer_text="pi*(38.44 - r^2)",
            marks=2,
            follow_through="pi*(part_a**2 - r**2)",
        ),
    ]
    for i, change in (overrides or {}).items():
        parts[i] = {**parts[i], **change}
    return dict(text="A circle has an area of 120.76 cm².", parts=parts)


@pytest.fixture
def register():
    added = []

    def _register(fn, topic=TOPIC):
        generator(topic, level=11, presets={"exam": {}})(fn)
        added.append(topic)
        return topic

    yield _register
    for topic in added:
        REGISTRY.pop(topic, None)
        PRESETS.pop(topic, None)


@pytest.fixture
def circle(register):
    register(lambda rng: _circle(rng))
    return make(TOPIC, seed=3)


# -- building ----------------------------------------------------------------------------------------------


def test_make_builds_parts_and_the_question_level_fields(circle):
    q = circle
    assert [p.label for p in q.parts] == ["a", "b", "c"]
    assert all(isinstance(p, Part) for p in q.parts)
    assert q.stem == "A circle has an area of 120.76 cm²."
    assert q.text.startswith(q.stem + " (a) Find the radius")
    assert "(b) Hence find the circumference" in q.text
    assert q.answer == {"a": 6.2, "b": 39.0, "c": "pi*(961/25 - r**2)"}
    assert q.answer_text == "(a) 6.2; (b) 39.0; (c) pi*(38.44 - r^2)"
    assert q.answer_latex.startswith("plain:")
    assert q.distractors == []
    assert q.working == ["(a) r = √(120.76 ÷ π) = 6.2", "(b) C = 2πr = 39.0"]
    assert [p.seed for p in q.parts] == ["3:a", "3:b", "3:c"]
    assert q.parts[0].prompt_latex.startswith(r"\text{")
    assert r"\text{(b) }" in q.latex


def test_to_dict_carries_parts_and_round_trips_as_json(circle):
    d = json.loads(circle.to_json())
    assert d["stem"] == circle.stem
    assert [p["label"] for p in d["parts"]] == ["a", "b", "c"]
    assert d["parts"][1]["follow_through"] == "2*pi*part_a"


def test_single_answer_questions_carry_no_parts_keys():
    d = make("addition", seed=1).to_dict()
    assert "parts" not in d and "stem" not in d


@pytest.mark.parametrize(
    "change, message",
    [
        ({1: {"follow_through": "2*pi*part_b"}}, "not an earlier part"),
        ({1: {"follow_through": "2*pi*part_z"}}, "not an earlier part"),
        ({1: {"follow_through": "3*pi*part_a"}}, "not the part's answer"),
        ({1: {"follow_through": "2*pi"}}, "names no earlier part"),
        ({0: {"answer_kind": "pair"}}, "not a single value"),
    ],
)
def test_make_refuses_a_wrong_follow_through(register, change, message):
    register(lambda rng: _circle(rng, change))
    with pytest.raises(VerificationError, match=message):
        make(TOPIC, seed=1)


@pytest.mark.parametrize(
    "build, message",
    [
        (lambda rng: {**_circle(rng), "answer": 1}, "on each part"),
        (lambda rng: {**_circle(rng), "parts": []}, "must not be empty"),
        (lambda rng: _circle(rng, {0: {"marks": 0}}), "marks"),
        (lambda rng: _circle(rng, {1: {"label": "a"}}), "unique"),
        (lambda rng: _circle(rng, {0: {"colour": "red"}}), "unknown keys"),
        (lambda rng: _circle(rng, {0: {"answer_kind": "nonsense"}}), "unknown answer_kind"),
        (lambda rng: _circle(rng, {0: {"distractors": [Distractor("6.2")]}}), "answer's own text"),
        (lambda rng: _circle(rng, {0: {"sensible_range": (7, None)}}), "outside sensible_range"),
        (lambda rng: _circle(rng, {0: {"sensible_range": (1, 0)}}), "sensible_range must be"),
        (lambda rng: _circle(rng, {2: {"sensible_range": (0, None)}}), "number or fraction part"),
        (lambda rng: _circle(rng, {0: {"distractors": [Distractor("12.4"), "12.4"]}}), "offered twice"),
    ],
)
def test_make_refuses_a_malformed_multi_part_question(register, build, message):
    register(build)
    with pytest.raises((ValueError, VerificationError), match=message):
        make(TOPIC, seed=1)


def test_follow_through_allows_one_unit_of_rounding(register):
    # (b) from the exact radius is 39.0; from the rounded radius 6.25 -> 39.3 it would be one unit off at most
    # when the radius is stated to fewer places. Here the stated (a) 6.2 gives 38.96 -> 39.0 exactly, so move
    # the canonical (b) one unit and the check must still pass, but not two units.
    register(lambda rng: _circle(rng, {1: {"answer": 39.1, "answer_text": "39.1"}}))
    make(TOPIC, seed=1)


def test_follow_through_two_units_off_is_refused(register):
    register(lambda rng: _circle(rng, {1: {"answer": 39.2, "answer_text": "39.2"}}))
    with pytest.raises(VerificationError):
        make(TOPIC, seed=1)


# -- reading replies -----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "given",
    [
        "(a) 6.2; (b) 39.0; (c) pi*(38.44 - r^2)",
        "(a) 6.2 (b) 39.0 (c) pi*(38.44 - r^2)",
        "a) 6.2, b) 39.0, c) pi*(38.44 - r^2)",
        "A: 6.2\nB: 39.0\nC: pi*(38.44 - r^2)",
        "6.2; 39.0; pi*(38.44 - r^2)",
        "6.2\n39.0\npi*(38.44 - r^2)",
        '{"a": "6.2", "b": "39.0", "c": "pi*(38.44 - r^2)"}',
        {"a": "6.2", "b": "39.0", "c": "pi*(38.44 - r^2)"},
    ],
)
def test_every_reply_shape_gets_full_marks(circle, given):
    result = grade("number", circle, given)
    assert result.correct
    assert (result.marks, result.max_marks) == (5, 5)


def test_split_reply_gives_missing_parts_an_empty_reply(circle):
    assert split_reply(circle.parts, {"a": "6.2"}) == {"a": "6.2", "b": "", "c": ""}
    assert split_reply(circle.parts, "6.2 and 39") == {"a": "", "b": "", "c": ""}


def test_labelled_replies_may_leave_parts_out_or_come_in_any_order(circle):
    assert split_reply(circle.parts, "(a) 6.2 (b) 39.0") == {"a": "6.2", "b": "39.0", "c": ""}
    assert split_reply(circle.parts, "(b) 39.0 (a) 6.2") == {"a": "6.2", "b": "39.0", "c": ""}
    assert grade("number", circle, "(a) 6.2 (b) 39.0").marks == 3
    assert grade("number", circle, "(c) pi*(38.44 - r^2); (a) 6.2; (b) 39.0").marks == 5


def test_reply_value_reads_one_value_or_none():
    assert reply_value("number", "r = 6.2 cm") == Fraction(31, 5)
    assert reply_value("fraction", "-3/4") == Fraction(-3, 4)
    assert str(reply_value("expression", "2x + 1")) == "2*x + 1"
    assert reply_value("number", "6 or 7") is None
    assert reply_value("number", "") is None
    assert reply_value("pair", "(1, 2)") is None


# -- marking -------------------------------------------------------------------------------------------------


def test_follow_through_gives_marks_for_using_a_wrong_earlier_answer(circle):
    # Radius 7 is wrong; 2π × 7 = 43.98 -> 44.0 and π(49 - r²) follow through from it.
    result = grade("number", circle, {"a": "7", "b": "44.0", "c": "pi*(49 - r^2)"})
    assert not result.correct
    assert (result.marks, result.max_marks) == (3, 5)
    a, b, c = (result.parts[k] for k in "abc")
    assert (a.correct, a.marks, a.follow_through) == (False, 0, False)
    assert (b.correct, b.marks, b.follow_through) == (True, 1, True)
    assert (c.correct, c.marks, c.follow_through) == (True, 2, True)


def test_follow_through_never_rescues_an_answer_wrong_both_ways(circle):
    result = grade("number", circle, {"a": "7", "b": "39.0", "c": "x"})
    # 39.0 is the canonical (b), right on its own; (c) is wrong both ways.
    assert result.parts["b"].correct and not result.parts["b"].follow_through
    assert result.parts["c"].marks == 0
    assert result.marks == 1


def test_a_correct_but_differently_rounded_earlier_answer_is_followed(register):
    # (a) asks for 2 dp: 6.20. A learner's (b) from 6.2 is exactly the canonical; from a correct (a) the
    # follow-through target and the canonical agree, so no follow_through flag is raised.
    register(lambda rng: _circle(rng))
    q = make(TOPIC, seed=1)
    result = grade("number", q, {"a": "6.2", "b": "39.0", "c": "pi*(38.44 - r^2)"})
    assert not any(p.follow_through for p in result.parts.values())


def test_blank_and_unreadable_earlier_answers_give_no_follow_through(circle):
    for a in ("", "six", "6 or 7"):
        result = grade("number", circle, {"a": a, "b": "44.0", "c": ""})
        assert result.marks == 0


def test_a_zero_earlier_answer_carries_no_follow_through(circle):
    # 2π × 0 = 0 and π(0 - r²) are "right" for a radius of 0, but a zero makes the later parts trivial.
    result = grade("number", circle, {"a": "0", "b": "0", "c": "pi*(0 - r^2)"})
    assert result.marks == 0


def test_an_earlier_answer_outside_its_sensible_range_carries_no_follow_through(register):
    register(lambda rng: _circle(rng, {0: {"sensible_range": (0, None)}}))
    q = make(TOPIC, seed=1)
    assert grade("number", q, {"a": "-7", "b": "-44.0"}).marks == 0
    assert grade("number", q, {"a": "7", "b": "44.0"}).marks == 1  # a wrong but sensible radius still carries


def test_repeating_an_earlier_answer_is_not_following_through(register):
    # (b) = part_a / 1 would make any repeated number "right"; a copy earns nothing unless it is canonical.
    register(
        lambda rng: _circle(
            rng,
            {1: {"answer": 6.2, "answer_text": "6.2", "follow_through": "part_a", "distractors": []}},
        )
    )
    q = make(TOPIC, seed=1)
    assert grade("number", q, {"a": "7", "b": "7"}).parts["b"].marks == 1  # the canonical answers agree


@pytest.mark.parametrize(
    "topic",
    [
        "exam_parts_circle_radius_arc",
        "exam_parts_square_pyramid",
        "exam_parts_implicit_tangent",
        "exam_parts_normal_distribution",
        "exam_parts_compound_interest",
    ],
)
@pytest.mark.parametrize("fill", ["0", "-3", "1", "-1", "0.5", "5000"])
def test_one_number_for_every_part_earns_no_follow_through_in_the_pilot_chains(topic, fill):
    for seed in range(40):
        q = make(topic, seed=seed, difficulty="exam")
        result = grade("number", q, {part.label: fill for part in q.parts})
        followed = [label for label, r in result.parts.items() if r.follow_through and r.marks]
        assert followed == [], (seed, followed, q.answer_text)


@pytest.mark.parametrize(
    "topic",
    [
        "exam_parts_circle_radius_arc",
        "exam_parts_square_pyramid",
        "exam_parts_implicit_tangent",
        "exam_parts_normal_distribution",
        "exam_parts_compound_interest",
    ],
)
@pytest.mark.parametrize("tiny", ["0.001", "0.0001", "0.01", "0.04"])
def test_a_tiny_earlier_answer_and_zeros_after_it_earn_no_follow_through(topic, tiny):
    for seed in range(40):
        q = make(topic, seed=seed, difficulty="exam")
        for source in q.parts:
            replies = {p.label: (tiny if p is source else "0") for p in q.parts}
            result = grade("number", q, replies)
            followed = [label for label, r in result.parts.items() if r.follow_through and r.marks]
            assert followed == [], (seed, source.label, followed, q.answer_text)


def test_a_tiny_radius_carries_nothing_but_a_wrong_sensible_one_still_does():
    q = make("exam_parts_circle_radius_arc", seed=0, difficulty="exam")
    assert grade("number", q, "(a) 0.001 (b) 0 (c) 0").marks == 0
    a = q.parts[0]
    wrong = str(Fraction(a.answer_text) + 1)
    b, c = (follow_through_target(p, {"a": reply_value("number", wrong)}) for p in q.parts[1:])
    result = grade("number", q, {"a": wrong, "b": b.answer_text, "c": c.answer_text})
    assert result.marks == 3 and result.parts["b"].follow_through and result.parts["c"].follow_through


def test_the_wrong_distractors_are_wrong_part_by_part(circle):
    canonical = {p.label: p.answer_text for p in circle.parts}
    for part in circle.parts:
        for distractor in part.distractors:
            result = grade("number", circle, {**canonical, part.label: distractor})
            assert not result.parts[part.label].correct
            assert result.marks == result.max_marks - part.marks


def test_every_grader_marks_a_multi_part_question_part_by_part(circle):
    # Library sweeps call `GRADERS[skill.answer]` directly; a multi-part question must not depend on which.
    for kind, grader in GRADERS.items():
        result = grader(circle, circle.answer_text)
        assert result.correct and result.max_marks == 5, kind


def test_partial_credit_comes_from_a_tagged_distractor(register):
    # A part's distractor can carry credit (mathgen/answers.py): a reply matching it earns that share of the
    # part's marks. 12.4 is the diameter given as the radius; say the scheme gives one of (a)'s two marks for it.
    slip = Distractor("12.4", misconception="squared-as-doubled", value=12.4, credit=Fraction(1, 2))
    register(lambda rng: _circle(rng, {0: {"distractors": [slip]}}))
    q = make(TOPIC, seed=1)
    assert q.parts[0].distractors == ["12.4"]
    assert q.parts[0].distractor_credit == {"12.4": "1/2"}
    result = grade("number", q, {"a": "12.40", "b": "39.0", "c": "x"})
    a = result.parts["a"]
    assert (a.correct, a.marks, a.matched_distractor, a.misconception) == (False, 1, "12.4", "squared-as-doubled")
    assert result.marks == 2


def test_an_oversized_reply_earns_nothing(circle):
    assert grade("number", circle, "9" * 200_000).marks == 0
    assert grade("number", circle, {"a": "9" * 200_000}).marks == 0


def test_follow_through_target_states_the_answer_as_the_part_does(circle):
    target = follow_through_target(circle.parts[1], {"a": Fraction(7)})
    assert target.answer_text == "44.0" and target.answer == 44.0
    assert target.distractors == []
    assert follow_through_target(circle.parts[1], {}) is None


def test_engine_grade_and_the_cli_shape(circle):
    class Skill:
        answer = "pair"
        misconceptions = near_misses = ()

    result = engine.grade(Skill(), circle, {"a": "6.2", "b": "39.0", "c": "pi*(38.44-r^2)"})
    assert result.correct and result.marks == 5


def test_a_chain_carried_from_a_wrong_first_part_is_follow_through_all_the_way(register):
    # (c) follows (b), which follows (a): a wrong (a) carried through (b) into (c) marks both as follow-through.
    def chain(rng):
        q = _circle(rng)
        q["parts"][2] = dict(
            prompt="Hence find the length of half the circumference, correct to 1 decimal place.",
            answer_kind="number",
            answer=19.5,
            answer_text="19.5",
            follow_through="part_b/2",
        )
        return q

    register(chain)
    q = make(TOPIC, seed=1)
    result = grade("number", q, {"a": "7", "b": "44.0", "c": "22.0"})
    assert [result.parts[k].follow_through for k in "abc"] == [False, True, True]
    assert result.marks == 2


def _tail_probability(rng):
    # z = 2.41; P(Z > z) = erfc(z/sqrt(2))/2 = 0.0080 (4 dp).
    return dict(
        text="A test statistic is standardised.",
        parts=[
            dict(prompt="Find z, correct to 2 decimal places.", answer_kind="number", answer=2.41, answer_text="2.41"),
            dict(
                prompt="Hence find P(Z > z), correct to 4 decimal places.",
                answer_kind="number",
                answer=0.008,
                answer_text="0.0080",
                follow_through="erfc(part_a/sqrt(2))/2",
            ),
            dict(
                prompt="Hence find e to the power of z, to 1 decimal place.",
                answer_kind="number",
                answer=11.1,
                answer_text="11.1",
                follow_through="exp(part_a)",
            ),
        ],
    )


@pytest.mark.parametrize("z", ["40", "4000", "999999", "-999999", "1e300"])
def test_a_huge_earlier_answer_cannot_stall_follow_through(register, z):
    # Writing erfc(999999/sqrt(2)), near 10**-217000000000, out as an exact fraction took minutes per reply.
    register(_tail_probability)
    q = make(TOPIC, seed=1)
    result = deadline.grade_with_deadline("number", q, f"(a) {z} (b) 0.0000 (c) 0.0", timeout=10)
    assert not result.timed_out
    assert result.marks == 0  # a zero tail and a value too large to state follow through to nothing


def test_follow_through_states_a_tiny_value_as_zero_and_refuses_a_huge_one(register):
    register(_tail_probability)
    q = make(TOPIC, seed=1)
    b, c = q.parts[1:]
    assert follow_through_target(b, {"a": reply_value("number", "999999")}).answer_text == "0.0000"
    assert follow_through_target(b, {"a": reply_value("number", "3")}).answer_text == "0.0013"
    assert follow_through_target(c, {"a": reply_value("number", "999999")}) is None
    assert follow_through_target(c, {"a": reply_value("number", "3")}).answer_text == "20.1"
