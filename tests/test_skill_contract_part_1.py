"""Skill contract cases, part 1 of 8: see `tests/skill_contract_cases.py`."""

from tests.skill_contract_cases import family_test

test_shape = family_test("shape", 1)
test_determinism = family_test("determinism", 1)
test_distractors = family_test("distractors", 1)
test_options = family_test("options", 1)
test_variety = family_test("variety", 1)
test_escalation = family_test("escalation", 1)
test_grader_round_trip = family_test("grader", 1)
