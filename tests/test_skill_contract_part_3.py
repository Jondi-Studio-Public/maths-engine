"""Skill contract cases, part 3 of 8: see `tests/skill_contract_cases.py`."""

from tests.skill_contract_cases import family_test

test_shape = family_test("shape", 3)
test_determinism = family_test("determinism", 3)
test_distractors = family_test("distractors", 3)
test_options = family_test("options", 3)
test_variety = family_test("variety", 3)
test_escalation = family_test("escalation", 3)
test_grader_round_trip = family_test("grader", 3)
