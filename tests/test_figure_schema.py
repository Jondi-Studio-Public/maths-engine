"""The published JSON Schema (`mathgen/figure_schema.json`) describes every figure the engine draws."""

from __future__ import annotations

import copy

import pytest
from jsonschema import Draft202012Validator

from mathgen import engine, figures

SEEDS = range(1, 5)


@pytest.fixture(scope="module")
def validator():
    schema = figures.schema()
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


CHUNKS = 8  # the generated-figure sweep is split so pytest-xdist can share it


EXAMPLES = {
    "base_ten": {"kind": "base_ten", "alt": "Base-ten blocks.", "hundreds": 1, "tens": 2, "ones": 3},
    "clock": {
        "kind": "clock",
        "alt": "A clock face.",
        "hour": 3,
        "minute": 0,
        "hour_angle": 90,
        "minute_angle": 0,
        "numerals": "all",
        "minute_ticks": True,
    },
    "venn": {
        "kind": "venn",
        "alt": "A Venn diagram.",
        "universe": "U",
        "sets": [{"id": "A"}, {"id": "B"}],
        "regions": {"A": "1", "B": "2", "AB": "3", "none": "4"},
    },
}


def test_schema_covers_every_kind():
    schema = figures.schema()
    assert schema["$id"] == "https://github.com/Jondi-Studio-Public/maths-engine/figure_schema.json"
    in_schema = set()
    for ref in schema["oneOf"]:
        branch = schema["$defs"][ref["$ref"].rsplit("/", 1)[1]]
        in_schema.add(branch["properties"]["kind"]["const"])
    assert in_schema == set(figures.kinds())


@pytest.mark.parametrize("kind", sorted(EXAMPLES))
def test_hand_made_examples_validate(validator, kind):
    assert validator.is_valid(EXAMPLES[kind])
    assert figures.validate(EXAMPLES[kind]) == []


@pytest.mark.parametrize("chunk", range(CHUNKS))
def test_every_generated_figure_validates(validator, chunk):
    """Every figure of every active skill source, seeds 1-4, validates against the schema."""
    _, skills = engine.load_skills()
    failures = []
    seen = set()
    for skill in skills[chunk::CHUNKS]:
        for i, _src in enumerate(skill.sources):
            for seed in SEEDS:
                try:
                    q = engine.make_for_skill(skill, seed, source=i)
                except Exception:  # a source that cannot generate is another test's concern
                    break
                if not q.figure:
                    break
                seen.add(q.figure["kind"])
                errors = list(validator.iter_errors(q.figure))
                if errors:
                    failures.append(f"{skill.id}[{i}]@{seed} {q.figure['kind']}: {errors[0].message[:200]}")
    assert seen <= set(figures.kinds())
    assert not failures, "\n".join(failures[:10])


def _example(kind):
    return copy.deepcopy(EXAMPLES[kind])


def test_unknown_kind_fails(validator):
    spec = _example("base_ten")
    spec["kind"] = "hexagon"
    assert not validator.is_valid(spec)


def test_missing_alt_fails(validator):
    spec = _example("base_ten")
    del spec["alt"]
    assert not validator.is_valid(spec)


def test_empty_alt_fails(validator):
    spec = _example("base_ten")
    spec["alt"] = "  "
    assert not validator.is_valid(spec)


def test_wrong_type_fails(validator):
    spec = _example("base_ten")
    spec["tens"] = "3"
    assert not validator.is_valid(spec)


def test_out_of_range_fails(validator):
    spec = _example("clock")
    spec["hour"] = 13
    assert not validator.is_valid(spec)


def test_unknown_key_fails(validator):
    spec = _example("venn")
    spec["colour"] = "red"
    assert not validator.is_valid(spec)


def test_not_an_object_fails(validator):
    assert not validator.is_valid([])
    assert not validator.is_valid({"alt": "x"})
