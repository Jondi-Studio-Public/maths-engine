"""Skill contract cases, part 5 of 8: see `tests/skill_contract_cases.py`."""

from tests.skill_contract_cases import family_test

test_shape = family_test("shape", 5)
test_determinism = family_test("determinism", 5)
test_distractors = family_test("distractors", 5)
test_options = family_test("options", 5)
test_variety = family_test("variety", 5)
test_escalation = family_test("escalation", 5)
test_grader_round_trip = family_test("grader", 5)
