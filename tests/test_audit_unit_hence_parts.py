"""The asked unit is read from parts that open "Hence", "Using your", "Round" and the like, and from "how many
more <unit>"; a wrongly stated unit is rejected, a bare number or the right unit is not."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mathgen.grading.units import unit_is_acceptable


def _q(text, answer_text="10"):
    return SimpleNamespace(text=text, answer_text=answer_text, answer=10)


@pytest.mark.parametrize(
    "text, right, wrong",
    [
        ("Hence express this energy in kilojoules.", "24624 kJ", "24624 kWh"),
        ("Using your estimate from (a), how many more kilograms can the vehicle carry?", "653 kg", "653 g"),
        ("Hence find the difference between the times, in minutes, to the nearest minute.", "141 min", "141 hours"),
        ("Using your height from (a), find how many millilitres the cylinder holds.", "2525 mL", "2525 L"),
        ("Hence find the area in hectares, correct to 2 decimal places.", "0.90 ha", "0.90 m²"),
        ("Hence find the length of the rod PQ, in centimetres, correct to 2 decimal places.", "3.97 cm", "3.97 m"),
        ("Hence find how long the flight takes, in hours, correct to 2 decimal places.", "2.50 hours", "2.50 min"),
        ("Round your answer to the nearest whole number, in metres.", "26 m", "26 km"),
        ("Then find how many fewer litres are left.", "5 L", "5 mL"),
        ("Find the distance after 4.5 seconds, in centimetres, correct to 1 decimal place.", "10.0 cm", "10.0 m"),
        ("Give your answer in dollars.", "30 dollars", "30 kg"),
    ],
)
def test_unit_named_in_a_follow_on_part_is_enforced(text, right, wrong):
    question = _q(text)
    assert unit_is_acceptable(question, right)
    assert not unit_is_acceptable(question, wrong)


@pytest.mark.parametrize(
    "text",
    [
        "Hence express this energy in kilojoules.",
        "Using your estimate from (a), how many more kilograms can the vehicle carry?",
        "Hence find the area in hectares, correct to 2 decimal places.",
    ],
)
def test_a_bare_number_is_still_accepted(text):
    assert unit_is_acceptable(_q(text), "24624")


def test_an_earlier_value_is_not_read_as_the_asked_unit():
    question = _q("Using your speed in m/s, how long would it take to ride a further 550 m, to the nearest second?")
    assert unit_is_acceptable(question, "100 s")
    assert unit_is_acceptable(question, "100 kN")
    assert unit_is_acceptable(question, "100")


@pytest.mark.parametrize(
    "text, part, other",
    [
        ("Hence find the bending moment, in kN·m.", "100 kN", "100 s"),
        ("Hence find the rate of change, in dollars per day.", "100 dollars", "100 s"),
        ("Hence find the average density, in kg/m³.", "100 kg", "100 s"),
        ("Hence find the gradient, in metres per hundred metres.", "100 m", "100 s"),
        ("Find the stiffness k in ω = √(k/m), in N/m.", "100 N", "100 kg"),
        ("Find the flow rate in kL per hour.", "100 kL", "100 kg"),
        ("By how many cents per 100 g is the cheaper buy cheaper?", "100 cents", "100 kg"),
    ],
)
def test_a_compound_unit_is_not_read_as_its_first_part_but_still_refuses_other_units(text, part, other):
    question = _q(text)
    assert unit_is_acceptable(question, part)
    assert unit_is_acceptable(question, "100")
    assert unit_is_acceptable(question, "100 km/h")  # a rate unit is never refused for a compound ask
    assert not unit_is_acceptable(question, other)


def test_asked_unit_through_real_questions():
    from mathgen import engine

    skills = {s.id: s for s in engine.load_skills()[1]}
    sk = skills["senior-convert-from-one-unit-of-energy-to-another"]
    source = next(i for i, s in enumerate(sk.sources) if s.difficulty == "exam")
    q = engine.make_for_skill(sk, 1, source=source)
    reply = {p.label: p.answer_text for p in q.parts}
    assert engine.grade(sk, q, reply).parts["b"].correct
    for unit, ok in (("kJ", True), ("kilojoules", True), ("kWh", False), ("MJ", False)):
        got = engine.grade(sk, q, {**reply, "b": f"{reply['b']} {unit}"}).parts["b"].correct
        assert got is ok, unit
