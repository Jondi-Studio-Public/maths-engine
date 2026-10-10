"""Independent oracle test for f10-scientific-notation-measurements.

Derives every answer purely by regex-parsing `question.text` and
recomputing from scratch -- imports nothing from
`mathgen.generators.p7e_scientific_measurements_build`. Generates through
the skill's own `sources` (topic + params + difficulty), never a bare
`make(topic, seed, difficulty=tier)`.
"""

import re
from fractions import Fraction

import pytest
import yaml

import mathgen

SKILL_PATH = "skills/secondary-number/f10-scientific-notation-measurements.yaml"

_SCI_RE = re.compile(r"(\d+(?:\.\d+)?) × 10\^(-?\d+)")


def _load_sources():
    with open(SKILL_PATH, encoding="utf-8") as f:
        skill = yaml.safe_load(f)
    return skill["sources"]


SOURCES = _load_sources()


def _sci_values(text: str) -> list[Fraction]:
    """Every `coefficient x 10^exponent` token in `text`, as exact Fractions."""
    return [Fraction(coef) * Fraction(10) ** int(exp) for coef, exp in _SCI_RE.findall(text)]


def _oracle_answer(text: str) -> Fraction:
    values = _sci_values(text)
    assert len(values) == 2, f"expected exactly 2 scientific-notation quantities: {text!r}"
    a, b = values
    if "distance it travels" in text:
        return a * b  # speed x time
    if "How many hours" in text:
        # "distance ... km" is stated first, "speed ... km/h" second
        return a / b
    if "How many times as massive" in text:
        return max(a, b) / min(a, b)
    raise AssertionError(f"unrecognised question shape: {text!r}")


@pytest.mark.parametrize("source_index", range(len(SOURCES)))
@pytest.mark.parametrize("seed", range(40))
def test_oracle(source_index, seed):
    spec = SOURCES[source_index]
    question = mathgen.make(spec["topic"], seed=seed, difficulty=spec["difficulty"], **(spec["params"] or {}))
    expected = _oracle_answer(question.text)
    assert expected.denominator == 1, (
        f"source {source_index} seed {seed}: expected a whole number, got {expected}\nQuestion: {question.text}"
    )
    actual = Fraction(question.answer_text)
    assert expected == actual, (
        f"source {source_index} seed {seed}: expected {expected}, got {actual}\nQuestion: {question.text}"
    )
    for distractor_text in question.distractors:
        distractor = Fraction(distractor_text)
        assert distractor != expected, (
            f"source {source_index} seed {seed}: distractor {distractor} matches expected "
            f"{expected}\nQuestion: {question.text}"
        )
