"""Skill contract cases, part 4 of 8: see `tests/skill_contract_cases.py`."""

from tests.skill_contract_cases import family_test

test_shape = family_test("shape", 4)
test_determinism = family_test("determinism", 4)
test_distractors = family_test("distractors", 4)
test_options = family_test("options", 4)
test_variety = family_test("variety", 4)
test_escalation = family_test("escalation", 4)
test_grader_round_trip = family_test("grader", 4)
