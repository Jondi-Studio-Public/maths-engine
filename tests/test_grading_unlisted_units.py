"""A reply in a different unit is refused when the question's unit is compound or was missing from the unit table."""

from __future__ import annotations

import pytest

from mathgen.core import Question
from mathgen.grading import grade


def _q(answer, answer_text, text):
    return Question(topic="t", level=0, seed=1, text=text, latex="", answer=answer, answer_text=answer_text)


CASES = [
    (
        "Find the acceleration in m/s².",
        9.8,
        "9.8 m/s²",
        ["9.8", "9.8 m/s²", "9.8 m/s^2", "9.8 metres per second squared"],
        ["9.8 kg", "9.8 m/s", "9.8 km/h", "9.8 N", "9.8 K"],
    ),
    (
        "Find the speed in km/h.",
        60,
        "60 km/h",
        ["60", "60 km/h", "60 kph", "60 kilometres per hour"],
        ["60 km/s", "60 km/min", "60 m/h", "60 km"],
    ),
    (
        "What is the temperature in °C?",
        30,
        "30 °C",
        ["30", "30 °C", "30°C", "30 C", "30 degrees Celsius"],
        ["30 F", "30 K", "30 °F"],
    ),
    ("How much force, in N?", 5, "5 N", ["5", "5 N", "5 newtons"], ["5 J", "5 kg", "5 W"]),
    ("Find the density in g/cm³.", 2, "2 g/cm³", ["2", "2 g/cm³", "2 g/cm^3"], ["2 g", "2 g/mL", "2 kg/m³"]),
    ("Find the energy in J.", 20, "20 J", ["20", "20 J", "20 joules"], ["20 kJ", "20 W", "20 N"]),
    ("Find the temperature in kelvin.", 300, "300 K", ["300", "300 K", "300 kelvin"], ["300 °C", "300 F", "300 kg"]),
    # a unit the table does not list: any listed unit is wrong, the question's own spelling and no unit are right
    ("Find the flow rate.", 15, "15 L/min", ["15", "15 L/min", "15 l/min"], ["15 kg", "15 m", "15 km/h"]),
    ("Find the cost.", 9, "$9", ["$9", "9", "9 dollars"], []),
]


@pytest.mark.parametrize("text, value, shown, good, bad", CASES)
def test_units(text, value, shown, good, bad):
    question = _q(value, shown, text)
    for reply in good:
        assert grade("number", question, reply).correct, reply
    for reply in bad:
        assert not grade("number", question, reply).correct, reply


def test_fraction_with_a_compound_unit():
    question = _q({"p": 39, "q": 4}, "39/4 m/s²", "Find the acceleration in m/s².")
    assert grade("fraction", question, "39/4 m/s^2").correct
    assert not grade("fraction", question, "39/4 kg").correct
    assert not grade("fraction", question, "39/4 m/s").correct


def test_a_long_odd_unit_text_is_read_in_linear_time():
    import time

    question = _q(5, "5 m/s²", "Find the speed in m/s².")
    for reply in ("5 " + "m/" * 50000, "5 " + "a " * 50000, "5 " + "/" * 100000, "5 " + "^" * 100000):
        started = time.perf_counter()
        grade("number", question, reply)
        assert time.perf_counter() - started < 5
