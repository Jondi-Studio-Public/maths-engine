"""The published JSON Schema (`mathgen/figure_schema.py`) describes every figure the engine draws."""

from __future__ import annotations

import copy

import pytest
from jsonschema import Draft202012Validator

from mathgen import engine, figures
from mathgen.figure_kinds import data

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
    "grid_square": {
        "kind": "grid_drawing",
        "alt": "A map on a square grid.",
        "grid": {"type": "square", "width": 8, "height": 6},
        "cells": [{"cells": [[0, 0], [1, 0]], "style": "hatched"}, {"cells": [[2, 2]], "style": "dotted"}],
        "polygons": [{"vertices": [[1, 1], [4, 1], [4, 3]], "style": "dashed", "label": "P"}],
        "lines": [{"points": [[0, 2.5], [8, 2.5]], "style": "road"}],
        "path": {"cells": [[5, 0], [5, 1], [6, 1]]},
        "labels": [{"at": [3, 0.5], "text": "Park"}],
        "symbols": [{"at": [2.5, 3.5], "symbol": "school"}],
        "legend": True,
        "column_labels": list("ABCDEFGH"),
        "row_labels": ["1", "2", "3", "4", "5", "6"],
        "scale": {"text": "1 cm : 50 m", "bar_cells": 2},
        "north": True,
        "canvas": {"width": 5, "height": 5},
    },
    "grid_cubes": {
        "kind": "grid_drawing",
        "alt": "A stack of cubes on isometric dot paper.",
        "grid": {"type": "isometric", "width": 3, "height": 3, "layers": 3},
        "cubes": [[0, 0, 0], [0, 0, 1], [1, 0, 0]],
        "canvas": {"width": 4, "height": 4},
    },
    "grid_shape": {
        "kind": "grid_drawing",
        "alt": "A shape on a square grid.",
        "grid": {"type": "square", "width": 6, "height": 6},
        "cells": [{"cells": [[1, 1], [2, 1]], "style": "shaded"}],
        "canvas": "grid",
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


def test_statistical_charts_check_their_shape(validator):
    pie = data.pie_chart(parts=4, sectors=[{"label": "Tea", "parts": 1}, {"label": "Milk", "parts": 3}], alt="A pie.")
    assert validator.is_valid(pie)
    assert not validator.is_valid(pie | {"parts": 7})
    bars = data.bar_chart(
        categories=["Mon", "Tue"], series=[{"name": "", "values": [2, 4]}], value_max=4, value_step=1, alt="Bars."
    )
    assert validator.is_valid(bars)
    assert not validator.is_valid(bars | {"value_axis": {**bars["value_axis"], "min": 1}})


def test_grid_drawing_enums_match_the_kind():
    from mathgen.figure_kinds import grid

    defs = figures.schema()["$defs"]["grid_drawing"]["properties"]
    assert set(defs["symbols"]["items"]["properties"]["symbol"]["enum"]) == set(grid.SYMBOLS)
    assert tuple(defs["cells"]["items"]["properties"]["style"]["enum"]) == grid.CELL_STYLES
    assert tuple(defs["lines"]["items"]["properties"]["style"]["enum"]) == grid.LINE_STYLES
    assert defs["cubes"]["maxItems"] == grid.MAX_CUBES
    assert defs["path"]["properties"]["cells"]["maxItems"] == grid.MAX_PATH_CELLS


def test_grid_drawing_checks_its_shape(validator):
    square = _example("grid_square")
    assert not validator.is_valid(square | {"grid": {"type": "square", "width": 31, "height": 4}})
    assert not validator.is_valid(_example("grid_cubes") | {"grid": {"type": "isometric", "width": 3, "height": 3}})
    assert not validator.is_valid(_example("grid_cubes") | {"cells": [{"cells": [[0, 0]], "style": "shaded"}]})
    assert not validator.is_valid(square | {"cubes": [[0, 0, 0]]})
    assert not validator.is_valid(square | {"cells": [{"cells": [[0, 0]], "style": "striped"}]})
    assert not validator.is_valid(square | {"symbols": [{"at": [1, 1], "symbol": "castle"}]})
    assert not validator.is_valid(square | {"lines": [{"points": [[0, 0]], "style": "road"}]})
    assert not validator.is_valid(square | {"canvas": {"width": 21, "height": 3}})
    assert validator.is_valid(square | {"canvas": "grid"})
