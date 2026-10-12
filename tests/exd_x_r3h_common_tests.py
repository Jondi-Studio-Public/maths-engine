"""The test functions every `test_exd_x_r3h_*` file shares, as one importable set.

A test file imports `make_tests(...)` and binds the returned functions at module level, so each file still names
its own oracle and its own follow-through case while the answer-space, grader and shape checks stay identical.
"""

from __future__ import annotations

from tests import exd_x_r3h_support as sup


def make_tests(topic: str, skill_id: str, oracle):
    def test_source_wired_and_shapes():
        assert sup.exam_source(skill_id).topic == topic
        assert sup.exam_source(skill_id).params == {"tier": "exam"}
        sup.shape_check(skill_id, topic)

    def test_keys_rederived_from_the_rendered_prompt():
        sup.run_oracle(topic, oracle, 300)

    def test_answer_space():
        counts = sup.answer_space(topic, 300)
        assert len(counts) >= 100

    def test_key_accepted_and_every_distractor_rejected():
        sup.full_marks_and_distractors(skill_id, topic, 120)

    def test_follow_through_from_a_wrong_first_answer():
        sup.follow_through_check(skill_id, topic, 80)

    return (
        test_source_wired_and_shapes,
        test_keys_rederived_from_the_rendered_prompt,
        test_answer_space,
        test_key_accepted_and_every_distractor_rejected,
        test_follow_through_from_a_wrong_first_answer,
    )
