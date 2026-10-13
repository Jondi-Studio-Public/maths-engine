"""The ``grid_drawing`` figure kind (`mathgen.figure_kinds.grid`): its validator, its texts and the cube views."""

from __future__ import annotations

import pytest

from mathgen import figures
from mathgen.figure_kinds import grid as g

SQUARE = {"type": "square", "width": 6, "height": 4}
ISO = {"type": "isometric", "width": 3, "height": 3, "layers": 3}


def _spec(**keys) -> dict:
    return {"kind": "grid_drawing", "alt": "A shape drawn on a square grid.", "grid": dict(SQUARE), **keys}


def test_the_kind_is_registered():
    assert "grid_drawing" in figures.kinds()


def test_a_full_square_drawing_is_valid():
    spec = g.grid_drawing(
        alt="A map on a square grid.",
        grid=SQUARE,
        cells=[{"cells": [[0, 0], [1, 0]], "style": "shaded"}, {"cells": [[5, 3]], "style": "hatched"}],
        polygons=[{"vertices": [[1, 1], [4, 1], [4, 3]], "style": "dashed", "label": "P"}],
        lines=[{"points": [[0, 2.5], [6, 2.5]], "style": "road"}],
        labels=[{"at": [3, 0.5], "text": "Park"}],
        symbols=[{"at": [2.5, 3.5], "symbol": "school"}],
        legend=True,
        column_labels=list("ABCDEF"),
        row_labels=["1", "2", "3", "4"],
        scale={"text": "1 cm : 50 m", "bar_cells": 2},
        north=True,
        canvas={"width": 5, "height": 5},
    )
    assert figures.validate(spec) == []
    assert "school" in figures.labels(spec)
    assert {"Park", "P", "A", "4", "1 cm : 50 m"} <= set(figures.labels(spec))


def test_empty_groups_are_left_out():
    spec = g.grid_drawing(alt="An empty grid.", grid=SQUARE)
    assert set(spec) == {"kind", "alt", "grid"}


@pytest.mark.parametrize(
    "keys",
    [
        {"grid": {"type": "hex", "width": 3, "height": 3}},
        {"grid": {"type": "square", "width": 0, "height": 3}},
        {"grid": {"type": "square", "width": 31, "height": 3}},
        {"grid": {"type": "square", "width": 3.0, "height": 3}},
        {"cells": [{"cells": [[6, 0]], "style": "shaded"}]},  # off the grid
        {"cells": [{"cells": [[0, 0], [0, 0]], "style": "shaded"}]},
        {"cells": [{"cells": [[0, 0]], "style": "red"}]},  # style by colour alone
        {"cells": [{"cells": [[0.5, 0]], "style": "shaded"}]},
        {"polygons": [{"vertices": [[0, 0], [1.5, 0], [1, 1]]}]},  # off the lattice
        {"polygons": [{"vertices": [[0, 0], [7, 0], [1, 1]]}]},  # off the grid
        {"polygons": [{"vertices": [[0, 0], [1, 0]]}]},
        {"lines": [{"points": [[0, 0.25], [2, 0]], "style": "road"}]},
        {"lines": [{"points": [[0, 0], [2, 0]], "style": "motorway"}]},
        {"symbols": [{"at": [1, 1], "symbol": "castle"}]},
        {"legend": True},
        {"column_labels": ["A", "B"]},
        {"row_labels": ["1", "1", "2", "3"]},
        {"scale": {"text": "1 : 100", "bar_cells": 9}},
        {"north": False},
        {"canvas": {"width": 21, "height": 3}},
        {"canvas": "beside"},
        {"cubes": [[0, 0, 0]]},  # cubes on a square grid
        {"labels": [{"at": [9, 0], "text": "far"}]},
        {"labels": [{"at": [1, 0], "text": "x" * 25}]},
    ],
)
def test_a_bad_square_drawing_is_refused(keys):
    assert figures.validate(_spec(**keys)) != []


def test_a_cube_stack_is_valid():
    spec = g.grid_drawing(alt="A stack of cubes.", grid=ISO, cubes=[[0, 0, 0], [0, 0, 1], [1, 0, 0]])
    assert figures.validate(spec) == []


@pytest.mark.parametrize(
    "cubes",
    [
        [],
        [[0, 0, 1]],  # floating
        [[0, 0, 0], [0, 0, 0]],
        [[3, 0, 0]],
        [[0, 0, 3]],
        [[0, 0]],
        [[0, 0, True]],
    ],
)
def test_a_bad_cube_stack_is_refused(cubes):
    spec = {"kind": "grid_drawing", "alt": "Cubes.", "grid": dict(ISO), "cubes": cubes}
    assert figures.validate(spec) != []


def test_square_only_keys_are_refused_on_an_isometric_grid():
    spec = {"kind": "grid_drawing", "alt": "Cubes.", "grid": dict(ISO), "cubes": [[0, 0, 0]]}
    assert figures.validate({**spec, "cells": [{"cells": [[0, 0]], "style": "shaded"}]}) != []
    assert figures.validate({**spec, "canvas": "grid"}) != []
    assert figures.validate({**spec, "canvas": {"width": 3, "height": 3}}) == []


def test_the_leak_check_sees_labels_and_the_scale():
    spec = g.grid_drawing(
        alt="A floor plan.", grid=SQUARE, labels=[{"at": [1, 1], "text": "12 m"}], scale={"text": "1 : 200"}
    )
    with pytest.raises(AssertionError):
        figures.assert_no_leak(spec, ["12 m"])
    with pytest.raises(AssertionError):
        figures.assert_no_leak(spec, ["1:200"])
    figures.assert_no_leak(spec, ["24"])


def test_symbol_names_are_text_only_with_a_legend():
    plain = g.grid_drawing(alt="A map.", grid=SQUARE, symbols=[{"at": [1, 1], "symbol": "post_office"}])
    assert "post office" not in figures.labels(plain)
    keyed = g.grid_drawing(alt="A map.", grid=SQUARE, symbols=[{"at": [1, 1], "symbol": "post_office"}], legend=True)
    assert "post office" in figures.labels(keyed)


# ---------------------------------------------------------------- views

# An L of three columns: (0, 0) two high, (1, 0) one high, (0, 1) three high.
STACK = [[0, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 1, 1], [0, 1, 2]]


def test_count_and_plan():
    assert g.cube_count(STACK) == 6
    assert g.plan_view(STACK) == [[0, 0], [0, 1], [1, 0]]


def test_elevations():
    assert g.elevation(STACK, "front") == [[0, 0], [0, 1], [0, 2], [1, 0]]
    assert g.elevation(STACK, "right") == [[0, 0], [0, 1], [1, 0], [1, 1], [1, 2]]
    assert g.elevation(STACK, "back", depth=2) == [[0, 0], [1, 0], [1, 1], [1, 2]]
    assert g.elevation(STACK, "left", depth=2) == [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1]]
    with pytest.raises(ValueError):
        g.elevation(STACK, "top")


def test_back_and_front_are_mirror_images():
    width = 3
    front = g.elevation(STACK, "front")
    back = g.elevation(STACK, "back", depth=width)
    assert sorted([width - 1 - c, z] for c, z in front) == back


def test_tops_all_visible():
    assert g.tops_all_visible(STACK)  # the tall column is at the back
    hidden = [[0, 1, 0], [1, 0, 0], [1, 0, 1]]  # a taller column in front and to the right
    assert not g.tops_all_visible(hidden)
