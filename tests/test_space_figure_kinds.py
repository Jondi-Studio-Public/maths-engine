"""The `diagram`, `tree_diagram` and `venn` figure kinds: what they accept and what they reject."""

import copy
import re

import pytest

from mathgen import figures
from mathgen.figure_kinds import space


def _problems(spec):
    return figures.validate(spec)


def _triangle(**overrides):
    return dict(
        alt="A triangle.",
        points=[
            {"id": "A", "at": [10, 10], "label": "A"},
            {"id": "B", "at": [70, 10], "label": "B"},
            {"id": "C", "at": [10, 50], "label": "C"},
        ],
        segments=[{"from": "A", "to": "B", "label": "6 cm"}, {"from": "A", "to": "C", "label": "4 cm"}],
        polygons=[{"points": ["A", "B", "C"]}],
        angles=[{"at": "A", "from": "B", "to": "C", "right": True}],
        **overrides,
    )


def test_kinds_are_registered():
    assert {"diagram", "tree_diagram", "venn"} <= set(figures.kinds())


# ------------------------------------------------------------------ diagram


def test_a_to_scale_triangle_is_valid_and_has_no_leaking_texts():
    spec = space.diagram(**_triangle())
    assert spec["kind"] == "diagram" and figures.validate(spec) == []
    assert "6 cm" in figures.labels(spec)


def test_segment_labels_must_be_proportional_unless_not_to_scale():
    bad = _triangle()
    bad["segments"][0]["label"] = "9 cm"  # drawn 60 long against a 40-long segment labelled 4 cm
    with pytest.raises(ValueError, match="proportional"):
        space.diagram(**bad)
    space.diagram(**bad, not_to_scale=True)  # the order of the lengths is still right


def test_not_to_scale_still_rejects_a_longer_label_drawn_shorter():
    bad = _triangle()
    bad["segments"][0]["label"] = "2 cm"  # 6 cm segment now labelled shorter than the 4 cm one
    with pytest.raises(ValueError, match="drawn shorter"):
        space.diagram(**bad, not_to_scale=True)


def test_every_reference_must_exist():
    spec = space.diagram(**_triangle())
    for path, value in (
        (("segments", 0, "to"), "Z"),
        (("polygons", 0, "points", 2), "Z"),
        (("angles", 0, "at"), "Z"),
    ):
        broken = copy.deepcopy(spec)
        target = broken
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        assert _problems(broken), path


def test_points_outside_the_frame_and_duplicate_ids_are_rejected():
    spec = space.diagram(**_triangle())
    outside = copy.deepcopy(spec)
    outside["points"][0]["at"] = [-1, 10]
    assert any("outside the frame" in p for p in _problems(outside))
    repeated = copy.deepcopy(spec)
    repeated["points"][1]["id"] = "A"
    assert any("repeated" in p for p in _problems(repeated))


def test_a_right_angle_mark_must_be_drawn_square():
    spec = space.diagram(**_triangle())
    spec["points"][2]["at"] = [30, 50]  # the corner at A is no longer 90 degrees
    assert any("marked right" in p for p in _problems(spec))


def test_an_angle_label_must_match_the_drawing_unless_not_to_scale_and_never_the_wrong_kind():
    def at_point(label, not_to_scale=False):
        return space.angles_at_point(
            sweeps=[100, 100, 160], labels=[label, None, None], alt="Angles at a point.", not_to_scale=not_to_scale
        )

    at_point("100°")  # to scale and right
    with pytest.raises(ValueError, match="labelled 120°"):
        at_point("120°")
    with pytest.raises(ValueError, match="obtuse"):
        at_point("60°", not_to_scale=True)  # an acute label on an obtuse drawing is a contradiction
    at_point("120°", not_to_scale=True)  # same kind of angle: allowed when not to scale


def test_rays_must_lie_along_drawn_segments():
    spec = space.angles_at_point(sweeps=[100, 100, 160], labels=["100°", None, None], alt="a")
    spec["segments"] = spec["segments"][1:]  # ray R0 is no longer drawn
    assert any("not along a drawn segment" in p for p in _problems(spec))


def test_angles_at_one_vertex_must_not_overlap_and_labelled_ones_need_room():
    spec = space.angles_at_point(sweeps=[100, 100, 160], labels=["100°", "100°", "160°"], alt="a")
    spec["angles"].append({"at": "O", "from": "R0", "to": "R2", "label": "x"})
    assert any("overlap" in p for p in _problems(spec))
    with pytest.raises(ValueError, match="only"):
        space.angles_at_point(sweeps=[10, 190, 160], labels=["x", None, None], alt="a")


def test_tick_marks_and_arrows_are_claims():
    spec = space.diagram(**_triangle())
    spec["segments"][0]["ticks"] = 1
    assert any("only one segment has 1 ticks" in p for p in _problems(spec))
    spec["segments"][1]["ticks"] = 1  # 60 long against 40 long
    assert any("different lengths" in p for p in _problems(spec))
    # parallel lines
    ok = space.parallel_transversal(theta=70, labels={(0, 0): "x"}, alt="a")
    assert figures.validate(ok) == []
    crooked = copy.deepcopy(ok)
    crooked["points"][5]["at"] = [90, 40]  # B2 moves, the second arrowed line is no longer parallel
    assert any("not drawn parallel" in p for p in _problems(crooked))


def test_a_polygon_must_have_area_and_not_cross_itself():
    bow = _triangle()
    bow["points"].append({"id": "D", "at": [60, 45]})
    bow["polygons"] = [{"points": ["A", "B", "C", "D"]}]
    bow["angles"] = []
    spec = {"kind": "diagram", "frame": [100, 100], "not_to_scale": False, "segments": [], "angles": [], **bow}
    assert any("crosses itself" in p for p in _problems(spec))


def test_every_angle_builder_gives_a_valid_diagram():
    on_line = space.angles_on_line(sweeps=[50, 130], labels=["50°", "x"], alt="a", tilt=33, not_to_scale=False)
    assert figures.validate(on_line) == []
    crossing = space.crossing_lines(theta=70, labels=["70°", None, "x", None], alt="a", rotation=10, not_to_scale=False)
    assert figures.validate(crossing) == []
    for theta in (50, 90, 130):
        for rotation in (0, 25, 90, 200):
            fig = space.parallel_transversal(theta=theta, labels={(0, 1): "x"}, alt="a", rotation=rotation)
            assert figures.validate(fig) == []
    for orientation in range(12):
        fig = space.right_triangle(legs=(3, 4), labels={"a": "3", "b": "4", "c": "x"}, alt="a", orientation=orientation)
        assert figures.validate(fig) == []


def test_crossing_lines_label_vertically_opposite_angles_equal():
    fig = space.crossing_lines(theta=70, labels=["70°", None, "70°", None], alt="a", rotation=30, not_to_scale=False)
    assert figures.validate(fig) == []
    with pytest.raises(ValueError):
        space.crossing_lines(theta=70, labels=["110°", None, None, None], alt="a", not_to_scale=False)


# ------------------------------------------------------------------ tree_diagram


def _tree():
    return {
        "label": "Start",
        "children": [
            {
                "branch": "2/5",
                "label": "Red",
                "children": [
                    {"branch": "1/4", "label": "Red", "outcome": "Red, Red", "outcome_p": "1/10"},
                    {"branch": "3/4", "label": "Blue", "outcome": "Red, Blue", "outcome_p": "?"},
                ],
            },
            {
                "branch": "?",
                "label": "Blue",
                "children": [
                    {"branch": "1/2", "label": "Red", "outcome": "Blue, Red"},
                    {"branch": "1/2", "label": "Blue", "outcome": "Blue, Blue"},
                ],
            },
        ],
    }


def test_a_tree_with_blanks_and_a_marked_leaf_is_valid():
    spec = space.tree_diagram(alt="A probability tree.", root=_tree())
    assert figures.validate(spec) == []
    texts = figures.labels(spec)
    assert "Red, Blue" in texts and "?" in texts


@pytest.mark.parametrize(
    "edit,message",
    [
        (lambda t: t["children"][0]["children"][0].update(branch="1/2"), "total"),
        (lambda t: t["children"][0].update(branch="3/2"), "strictly between"),
        (lambda t: t["children"][0]["children"][0].update(outcome_p="1/3"), "product"),
        (lambda t: [k.update(branch="?") for k in t["children"][0]["children"]], "more than one blank"),
        (lambda t: t["children"][0]["children"].pop(), "2 to 4"),
        (lambda t: t["children"][1]["children"][0].pop("outcome"), "outcome"),
        (lambda t: t["children"][0]["children"][0].update(children=[{}, {}]), "stages|label|branch"),
    ],
)
def test_a_bad_tree_is_rejected(edit, message):
    root = _tree()
    edit(root)
    problems = _problems({"kind": "tree_diagram", "alt": "t", "root": root})
    assert problems and any(re.search(message, p) for p in problems), problems


def test_the_siblings_of_a_blank_cannot_already_total_one():
    root = _tree()
    root["children"][1]["children"][0]["branch"] = "?"
    root["children"][1]["children"][1]["branch"] = "1/1"
    assert _problems({"kind": "tree_diagram", "alt": "t", "root": root})


# ------------------------------------------------------------------ venn


def _venn(n=2, **regions):
    ids = list("ABC"[:n])
    base = {k: "3" for k in space.venn_region_keys(ids)}
    base.update(regions)
    return dict(alt="A Venn diagram.", sets=[{"id": i, "name": f"Set{i}"} for i in ids], regions=base)


def test_a_venn_diagram_needs_every_region():
    spec = space.venn(**_venn(2, AB="?"))
    assert figures.validate(spec) == []
    assert "?" in figures.labels(spec)
    assert figures.validate(space.venn(**_venn(3, ABC="?"))) == []
    missing = dict(spec)
    missing["regions"] = {k: v for k, v in spec["regions"].items() if k != "none"}
    assert any("exactly the keys" in p for p in _problems(missing))


def test_venn_region_text_must_be_a_count_a_short_list_or_a_question_mark():
    assert figures.validate(space.venn(**_venn(2, A="2, 4, 6", B="1, 3", AB="5", none="7"))) == []
    bad = space.venn(**_venn(2))
    bad["regions"]["A"] = "about ten"
    assert any("not a count" in p for p in _problems(bad))
    bad["regions"]["A"] = "1234"
    assert _problems(bad)


def test_venn_set_ids_are_unique_capitals():
    spec = space.venn(**_venn(2))
    spec["sets"][1]["id"] = "A"
    assert _problems(spec)
    assert _problems({**spec, "sets": spec["sets"][:1]})
