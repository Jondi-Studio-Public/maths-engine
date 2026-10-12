"""Skill contract cases, part 6 of 8: see `tests/skill_contract_cases.py`."""

from tests.skill_contract_cases import family_test

test_shape = family_test("shape", 6)
test_determinism = family_test("determinism", 6)
test_distractors = family_test("distractors", 6)
test_options = family_test("options", 6)
test_variety = family_test("variety", 6)
test_escalation = family_test("escalation", 6)
test_grader_round_trip = family_test("grader", 6)
