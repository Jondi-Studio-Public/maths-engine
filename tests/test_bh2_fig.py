"""Figure layout regressions from the 2026-10-19 bug hunt: labels sit beside what they name, text stays on the
canvas, drawing-board figures keep their frame faint, and numerals do not print over each other.

Everything is computed from the figure spec and the rendered SVG text; no browser is involved.
"""

from __future__ import annotations

import html
import re

import pytest

from mathgen import engine, svg
from mathgen.figure_kinds import placement

_SKILLS = {s.id: s for s in engine.load_skills()[1]}
_TEXT = re.compile(r"<text ([^>]*)>([^<]*)</text>")
_ATTR = re.compile(r'([\w-]+)="([^"]*)"')


def _figure(skill_id: str, source: int, seed: int) -> dict:
    return engine.make_for_skill(_SKILLS[skill_id], seed=seed, source=source).figure


def _texts(svg_text: str) -> list[dict]:
    """Each text element as {s, x, y, size, anchor}, in the SVG's own units (before any group translate)."""
    out = []
    for attrs, body in _TEXT.findall(svg_text):
        a = dict(_ATTR.findall(attrs))
        out.append(
            {
                "s": html.unescape(body),
                "x": float(a["x"]),
                "y": float(a["y"]),
                "size": float(a.get("font-size", 12)),
                "anchor": a.get("text-anchor", "start"),
                "bold": a.get("font-weight") == "bold",
            }
        )
    return out


def _box(t: dict) -> tuple[float, float, float, float]:
    w = len(t["s"]) * 0.52 * t["size"]
    return placement.text_box(t["x"], t["y"], w, t["anchor"], t["size"])


_CURVE_CASES = [
    ("senior-graph-translation-read-shift", 0),
    ("senior-graph-translation-read-shift", 1),
    ("senior-graph-translation-read-shift", 2),
    ("senior-graph-dilation-read-factor", 0),
    ("senior-graph-reciprocal-abs-compare", 0),
    ("f10-quadratic-parameter-effects", 0),
    ("f10-quadratic-parameter-effects", 2),
    ("senior-examine-the-reflection-property-of-the-graph-of", 0),
]


@pytest.mark.parametrize(("skill_id", "source"), _CURVE_CASES)
def test_curve_labels_are_nearer_their_own_curve(skill_id, source):
    for seed in range(8):
        fig = _figure(skill_id, source, seed)
        left, top, pw, ph, X, Y = svg._cartesian_frame(fig)
        plot = (left, top, left + pw, top + ph)
        curves = [
            (c.get("label"), placement.visible_samples([[(X(a), Y(b)) for a, b in pc] for pc in c["pieces"]], plot))
            for c in fig["curves"]
        ]
        texts = _texts(svg.figure_svg(fig))
        for label, own in curves:
            if not label:
                continue
            drawn = [t for t in texts if t["s"] == label and not t["bold"]]
            assert drawn, (skill_id, source, seed, label, "label not drawn")
            box = _box(drawn[0])
            d_own = placement.dist_to_box(own, box)
            others = [placement.dist_to_box(o, box) for lb, o in curves if lb != label]
            assert d_own < 12, (skill_id, source, seed, label, "label is far from its curve", d_own)
            assert all(d_own < d for d in others), (skill_id, source, seed, label, d_own, others)
            assert plot[0] - 2 <= box[0] and box[2] <= plot[2] + 2 and plot[1] - 2 <= box[1] and box[3] <= plot[3] + 2


def test_line_labels_stay_inside_the_plot_and_off_the_axis_name():
    for skill_id in ("solve-linear-equations-graphically", "f10-match-inequality-to-line"):
        for seed in range(8):
            fig = _figure(skill_id, 0, seed)
            left, top, pw, ph, _X, _Y = svg._cartesian_frame(fig)
            texts = _texts(svg.figure_svg(fig))
            names = [t for t in texts if t["bold"]]
            for ln in fig["lines"]:
                for t in (t for t in texts if t["s"] == ln.get("label") and not t["bold"]):
                    b = _box(t)
                    assert b[1] >= top - 0.5 and b[3] <= top + ph + 0.5, (skill_id, seed, t["s"])
                    for nm in names:
                        o = _box(nm)
                        assert not (b[0] < o[2] and o[0] < b[2] and b[1] < o[3] and o[1] < b[3]), (skill_id, seed)


_TICK_SKILLS = [
    "senior-graph-translation-read-shift",
    "senior-graph-dilation-read-factor",
    "senior-graph-reciprocal-abs-compare",
    "senior-examine-dilations-and-the-graphs-of-y-cf",
    "senior-examine-translations-and-the-graphs-of-y-f",
    "locus-circle-disc-pick-sketch",
]


@pytest.mark.parametrize("skill_id", _TICK_SKILLS)
def test_axis_numerals_do_not_overprint(skill_id):
    skill = _SKILLS[skill_id]
    for source in range(len(skill.sources)):
        for seed in range(3):
            fig = engine.make_for_skill(skill, seed=seed, source=source).figure
            if not fig or fig.get("kind") != "cartesian":
                continue
            numerals = [t for t in _texts(svg.figure_svg(fig)) if t["size"] == 11 and not t["bold"]]
            boxes = [_box(t) for t in numerals]
            for i, a in enumerate(boxes):
                for b in boxes[i + 1 :]:
                    inter_x = min(a[2], b[2]) - max(a[0], b[0])
                    inter_y = min(a[3], b[3]) - max(a[1], b[1])
                    assert not (inter_x > 1.0 and inter_y > 3.0), (skill_id, source, seed, numerals[boxes.index(a)])


def test_drawing_board_frame_is_faint():
    fig = _figure("f10-area-perimeter-count-squares", 0, 0)
    assert fig["tick_labels"] is False
    out = svg.figure_svg(fig)
    assert 'stroke-width="1.5" stroke="currentColor"' not in out.replace("stroke=", " stroke=").replace("  ", " ")
    axis_lines = re.findall(r"<line [^>]*>", out)
    heavy = [ln for ln in axis_lines if 'stroke="currentColor"' in ln and 'stroke-width="1.5"' in ln]
    assert not heavy, heavy


def test_labelled_axes_keep_a_strong_frame():
    out = svg.figure_svg(_figure("senior-graph-translation-read-shift", 0, 0))
    assert re.search(r'<line [^>]*stroke-width="1.5"', out)


def test_matching_view_title_fits_the_canvas():
    for source in range(len(_SKILLS["f10-pick-the-matching-view"].sources)):
        for seed in range(6):
            out = svg.figure_svg(_figure("f10-pick-the-matching-view", source, seed))
            for t in _texts(out):
                if t["size"] == 15:
                    b = _box(t)
                    assert b[0] >= 0 and b[2] <= 400, (source, seed, t["s"])


def test_plan_view_counts_have_no_marker_dots():
    out = svg.figure_svg(_figure("f10-plan-views-cube-stacks", 0, 0))
    assert "<circle" not in out
    counts = [t for t in _texts(out) if t["s"].isdigit()]
    assert len(counts) == 9


def test_matching_view_plan_counts_and_captions_are_plain_text():
    for source in range(len(_SKILLS["f10-pick-the-matching-view"].sources)):
        out = svg.figure_svg(_figure("f10-pick-the-matching-view", source, 0))
        assert "<circle" not in out
        caps = [t for t in _texts(out) if not t["s"].isdigit() and t["size"] < 15 and len(t["s"]) > 1]
        for i, a in enumerate(caps):
            for b in caps[i + 1 :]:
                if abs(a["y"] - b["y"]) < 8:
                    ba, bb = _box(a), _box(b)
                    assert ba[2] <= bb[0] or bb[2] <= ba[0], (a["s"], b["s"])


def test_growing_pattern_captions_have_no_dots():
    out = svg.figure_svg(_figure("f10-growing-visual-patterns", 0, 0))
    assert "<circle" not in out


def test_longitude_labels_do_not_touch():
    out = svg.figure_svg(_figure("senior-latitude-longitude-grid-places", 0, 0))
    row = sorted((t for t in _texts(out) if "W" in t["s"]), key=lambda t: t["x"])
    assert len(row) == 6
    for a, b in zip(row, row[1:]):
        assert _box(a)[2] < _box(b)[0], (a["s"], b["s"])


def test_triangle_height_and_side_labels_do_not_overlap():
    for source in range(3):
        for seed in range(8):
            out = svg.figure_svg(_figure("triangle-area-base-height", source, seed))
            labs = [t for t in _texts(out) if " " in t["s"] and t["size"] == 13]
            for i, a in enumerate(labs):
                for b in labs[i + 1 :]:
                    ba, bb = _box(a), _box(b)
                    overlap = min(ba[2], bb[2]) - max(ba[0], bb[0]) > 1 and min(ba[3], bb[3]) - max(ba[1], bb[1]) > 3
                    assert not overlap, (source, seed, a["s"], b["s"])


def test_no_text_runs_off_the_canvas_in_cartesian_figures():
    for skill_id in ("senior-recognise-features-of-the-graph-of-y-mx", "senior-find-the-line-of-best-fit-by-eye"):
        skill = _SKILLS[skill_id]
        for source in range(len(skill.sources)):
            for seed in range(8):
                fig = engine.make_for_skill(skill, seed=seed, source=source).figure
                if not fig or fig.get("kind") != "cartesian":
                    continue
                out = svg.figure_svg(fig)
                height = int(re.search(r'viewBox="0 0 400 (\d+)"', out).group(1))
                for t in _texts(out):
                    b = _box(t)
                    assert b[0] >= -1 and b[2] <= 401 and b[1] >= -1 and b[3] <= height + 1, (
                        skill_id,
                        source,
                        seed,
                        t["s"],
                    )
