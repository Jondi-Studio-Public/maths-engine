"""``grid_drawing``: the SVG renderer (`mathgen.render.grid`) and the describer (`mathgen.describe.grid`)."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import pytest

from mathgen import figures
from mathgen.describe import describe
from mathgen.figure_kinds import grid as g
from mathgen.render import figure_svg, renderable_kinds
from mathgen.render.grid import canvas_below

SVG = "{http://www.w3.org/2000/svg}"
_NON_FINITE = re.compile(r"\b(nan|inf|infinity)\b", re.I)

MAP = g.grid_drawing(
    alt="A map on a square grid.",
    title="Map of Kiama",
    grid={"type": "square", "width": 8, "height": 6},
    cells=[
        {"cells": [[0, 0], [1, 0], [1, 1]], "style": "shaded"},
        {"cells": [[5, 4], [6, 4]], "style": "hatched"},
        {"cells": [[7, 0], [7, 1]], "style": "dotted"},
    ],
    polygons=[{"vertices": [[1, 2], [4, 2], [4, 4]], "style": "dashed", "label": "Oval"}],
    lines=[
        {"points": [[0, 3], [3, 3], [3, 6]], "style": "road"},
        {"points": [[0, 5], [4, 5], [8, 4]], "style": "river"},
        {"points": [[4, 0], [4, 2]], "style": "rail"},
        {"points": [[5, 0], [8, 0]], "style": "wall"},
        {"points": [[5, 2], [8, 2]], "style": "fence"},
        {"points": [[0, 1.5], [3, 1.5]], "style": "mirror"},
    ],
    symbols=[{"at": [2.5, 2.5], "symbol": "school"}, {"at": [6.5, 3.5], "symbol": "post_office"}]
    + [{"at": [i + 0.5, 5.5], "symbol": name} for i, name in enumerate(sorted(g.SYMBOLS)[:8])],
    legend=True,
    column_labels=list("ABCDEFGH"),
    row_labels=["1", "2", "3", "4", "5", "6"],
    labels=[{"at": [6, 3], "text": "Beach"}],
    path=[[0, 4], [1, 4], [1, 5]],
    scale={"text": "1 cm : 2 m", "bar_cells": 2},
    north=True,
)
STACK = g.grid_drawing(
    alt="A stack of cubes on isometric dot paper.",
    grid={"type": "isometric", "width": 3, "height": 3, "layers": 3},
    cubes=[[0, 0, 0], [0, 0, 1], [0, 0, 2], [1, 2, 0], [1, 2, 1], [2, 0, 0], [2, 1, 0]],
    canvas={"width": 5, "height": 4},
)
SHAPE = g.grid_drawing(
    alt="A shape on a square grid.",
    grid={"type": "square", "width": 6, "height": 6},
    cells=[{"cells": [[1, 1], [2, 1], [2, 2]], "style": "shaded"}],
    canvas="grid",
)
SPECIMENS = {"map": MAP, "stack": STACK, "shape": SHAPE}


def _root(spec):
    return ET.fromstring(figure_svg(spec))


def _texts(root):
    return ["".join(n.itertext()).strip() for n in root.iter(f"{SVG}text")]


def test_the_kind_has_a_renderer():
    assert "grid_drawing" in renderable_kinds()


@pytest.mark.parametrize("name", sorted(SPECIMENS))
def test_renders_valid_svg(name):
    spec = SPECIMENS[name]
    svg = figure_svg(spec)
    root = ET.fromstring(svg)
    assert root.tag == f"{SVG}svg" and root.get("role") == "img"
    assert root.find(f"{SVG}title").text == spec["alt"]
    _, _, vw, vh = (float(v) for v in root.get("viewBox").split())
    assert 40 <= vw <= 1600 and 40 <= vh <= 1600
    for node in root.iter():
        for key, value in node.attrib.items():
            assert not _NON_FINITE.search(value), f"{node.tag} {key}={value!r}"
    assert "<script" not in svg and 'href="http' not in svg
    assert not re.search(r"#[0-9a-fA-F]{3,6}\b", svg.split("</style>", 1)[1]), "a hex colour in the drawing"
    assert figure_svg(spec) == svg
    assert figure_svg(spec, theme="dark") != svg


def test_every_label_is_drawn():
    drawn = set(_texts(_root(MAP)))
    for label in figures.labels(MAP)[1:]:
        assert label in drawn, label
    assert {"Map of Kiama", "Oval", "Beach", "1 cm : 2 m", "A", "H", "6", "post office", "school"} <= drawn
    # The only text drawn is the spec's own.
    assert drawn <= set(figures.labels(MAP)) | {"Map of Kiama"}


def test_the_legend_names_every_symbol_used_and_only_those():
    drawn = set(_texts(_root(MAP)))
    used = {g.SYMBOLS[s["symbol"]] for s in MAP["symbols"]}
    assert used <= drawn
    unused = {name for key, name in g.SYMBOLS.items() if name not in used}
    assert not unused & drawn
    plain = {k: v for k, v in MAP.items() if k != "legend"}
    assert "post office" not in _texts(_root(plain))


@pytest.mark.parametrize("symbol", sorted(g.SYMBOLS))
def test_every_symbol_draws(symbol):
    spec = g.grid_drawing(
        alt="A map.",
        grid={"type": "square", "width": 3, "height": 3},
        symbols=[{"at": [1.5, 1.5], "symbol": symbol}],
        legend=True,
    )
    root = _root(spec)
    assert len(list(root.iter(f"{SVG}g"))) > 2
    assert g.SYMBOLS[symbol] in _texts(root)


def test_hatched_and_dotted_cells_use_patterns_and_shaded_does_not():
    svg = figure_svg(MAP)
    root = ET.fromstring(svg)
    patterns = {p.get("id"): p for p in root.iter(f"{SVG}pattern")}
    assert len(patterns) == 2
    fills = {}
    for group in root.iter(f"{SVG}g"):
        cls = group.get("class", "")
        for style in ("hatched", "dotted", "shaded"):
            if f"mgf-gd-cells-{style}" in cls:
                fills[style] = {r.get("fill") for r in group.iter(f"{SVG}rect")}
    assert fills["hatched"] != fills["dotted"]
    for style in ("hatched", "dotted"):
        (ref,) = fills[style]
        assert ref.startswith("url(#") and ref[5:-1] in patterns
    assert fills["shaded"] == {None}
    kinds = {next(iter(p)).tag for p in patterns.values()}
    assert kinds == {f"{SVG}line", f"{SVG}circle"}
    # A style that is not used draws no pattern.
    assert not list(_root(SHAPE).iter(f"{SVG}pattern"))


def test_each_line_style_has_its_own_shape():
    root = _root(MAP)
    classes = {}
    for group in root.iter(f"{SVG}g"):
        cls = group.get("class", "")
        if cls.startswith("mgf-gd-feature "):
            classes[cls.split("-")[-1]] = [c.get("class") for c in group]
    assert classes["road"] == ["mgf-gd-road-edge", "mgf-gd-road-core"]
    assert "mgf-gd-rail-tick" in classes["rail"] and "mgf-gd-post" in classes["fence"]
    assert len({tuple(v) for v in classes.values()}) == 6


def test_polygon_dash_and_route_are_drawn():
    root = _root(MAP)
    polygon = next(p for p in root.iter(f"{SVG}polygon") if "mgf-gd-poly" in p.get("class", ""))
    assert "mgf-gd-poly-dashed" in polygon.get("class")
    assert [n.get("class") for n in root.iter(f"{SVG}polygon") if n.get("class") == "mgf-gd-path-head"] == [
        "mgf-gd-path-head"
    ] * 2
    assert len(list(root.iter(f"{SVG}circle"))) >= 1


def _cube_order(root):
    return [tuple(int(v) for v in n.get("data-cube").split(",")) for n in root.iter(f"{SVG}g") if n.get("data-cube")]


def test_cubes_are_painted_back_to_front():
    order = _cube_order(_root(STACK))
    assert sorted(order) == sorted(map(tuple, STACK["cubes"]))
    at = {c: i for i, c in enumerate(order)}
    # A cube nearer the viewer (the front, the right, above) is drawn after every cube it can hide.
    for a in order:
        for b in order:
            if a != b and b[0] >= a[0] and b[1] <= a[1] and b[2] >= a[2]:
                assert at[b] > at[a], f"{b} is in front of {a} but drawn first"


def test_a_cube_has_three_shaded_faces():
    root = _root(
        g.grid_drawing(
            alt="A cube.", grid={"type": "isometric", "width": 1, "height": 1, "layers": 1}, cubes=[[0, 0, 0]]
        )
    )
    (cube,) = [n for n in root.iter(f"{SVG}g") if n.get("data-cube")]
    assert [f.get("class") for f in cube] == ["mgf-gd-left", "mgf-gd-right", "mgf-gd-top"]
    assert len(list(root.iter(f"{SVG}circle"))) == 7  # the dot lattice


def test_canvas_is_drawn_beside_or_below():
    def extent(spec):
        _, _, vw, vh = (float(v) for v in _root(spec).get("viewBox").split())
        return vw, vh

    plain = {k: v for k, v in STACK.items() if k != "canvas"}
    sw, sh = extent(plain)
    cw, ch = extent(STACK)
    assert canvas_below(STACK) is False and cw > sw and ch >= sh
    wide = g.grid_drawing(
        alt="A grid.", grid={"type": "square", "width": 8, "height": 4}, canvas={"width": 5, "height": 5}
    )
    narrow = {k: v for k, v in wide.items() if k != "canvas"}
    assert canvas_below(wide) is True and extent(wide)[1] > extent(narrow)[1]
    # "grid" draws nothing extra.
    assert extent(SHAPE) == extent({k: v for k, v in SHAPE.items() if k != "canvas"})
    blank = [n for n in _root(STACK).iter(f"{SVG}g") if n.get("class") == "mgf-gd-canvas"]
    assert len(blank) == 1
    assert len(_root(SHAPE).findall(f".//{SVG}g[@class='mgf-gd-canvas']")) == 0


# ---------------------------------------------------------------- the describer


@pytest.mark.parametrize("name", sorted(SPECIMENS))
def test_description_follows_the_rules(name):
    spec = SPECIMENS[name]
    text = describe(spec)
    assert describe(spec) == text
    assert text.strip() == text and "  " not in text
    assert not re.search(r"[<>{}\\]|\*\*|\bNone\b|\bnan\b|\binf\b|\d\.0\b|\[|\]", text), text
    assert len(spec["alt"]) < len(text) <= 3000
    for sentence in re.split(r"(?<=[.])\s+", text):
        assert sentence[0].isupper() and sentence[-1] == ".", sentence
    for label in figures.labels(spec)[1:]:
        assert label in text, label


def test_the_map_is_described_in_full():
    text = describe(MAP)
    assert text.startswith("Map of Kiama. A square grid 8 squares wide and 6 squares high.")
    assert "The columns are labelled A, B, C, D, E, F, G and H from left to right" in text
    assert "Squares shaded: B2; A1 to B1." in text
    assert "Squares hatched with diagonal lines: F5 to G5." in text
    assert "Squares filled with dots: H2; H1." in text
    assert "A dashed shape with corners at (1, 2), (4, 2) and (4, 4), joined in that order, labelled Oval." in text
    assert "A road (a thick double line) runs from (0, 3) through (3, 3) to (3, 6)." in text
    assert "The school symbol is in square C3." in text
    assert "The label Beach is written at the point (6, 3)." in text
    assert (
        "A path, drawn as arrows from the middle of one square to the next, starts in square A5 and moves right to B5, then up to B6."
        in text
    )
    assert "A key below the grid names the symbols:" in text and "post office" in text
    assert "Scale: 1 cm : 2 m. A scale bar 2 squares long is drawn below the figure." in text
    assert "A north arrow points up the grid." in text


def test_squares_are_counted_when_there_are_no_labels():
    spec = g.grid_drawing(
        alt="A shape.",
        grid={"type": "square", "width": 5, "height": 4},
        cells=[{"cells": [[0, 0], [1, 0], [3, 0]], "style": "shaded"}],
        symbols=[{"at": [2.5, 1.5], "symbol": "tree"}],
        path=[[0, 0], [0, 1]],
    )
    text = describe(spec)
    assert "Squares shaded: row 1, columns 1 to 2 and 4." in text
    assert "The tree symbol is in square column 3, row 2." in text
    assert "starts in square column 1, row 1 and moves up to column 1, row 2." in text


def test_a_cube_stack_names_every_column_height_but_not_the_total():
    text = describe(STACK)
    assert "isometric dot paper, seen from the front right and above" in text
    for sentence in (
        "A column 2 cubes high in row 3 from the front, column 2 from the left.",
        "A column 1 cube high in row 1 from the front, column 3 from the left.",
        "A column 1 cube high in row 2 from the front, column 3 from the left.",
        "A column 3 cubes high in row 1 from the front, column 1 from the left.",
    ):
        assert sentence in text
    assert text.count("A column ") == 4
    total = len(STACK["cubes"])
    assert total == 7
    assert not re.search(rf"\b{total}\b|seven|total", text)
    assert "A blank grid 5 squares by 4 squares is beside the figure for your answer." in text


def test_canvas_wording_follows_the_drawing():
    wide = g.grid_drawing(
        alt="A grid.", grid={"type": "square", "width": 8, "height": 4}, canvas={"width": 5, "height": 5}
    )
    assert "A blank grid 5 squares by 5 squares is below the figure for your answer." in describe(wide)
    assert "Your answer is drawn on this same grid" in describe(SHAPE)
