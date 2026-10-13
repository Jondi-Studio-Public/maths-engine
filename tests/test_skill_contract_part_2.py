"""Skill contract cases, part 2 of 8: see `tests/skill_contract_cases.py`."""

from tests.skill_contract_cases import family_test

test_shape = family_test("shape", 2)
test_determinism = family_test("determinism", 2)
test_distractors = family_test("distractors", 2)
test_options = family_test("options", 2)
test_variety = family_test("variety", 2)
test_escalation = family_test("escalation", 2)
test_grader_round_trip = family_test("grader", 2)
