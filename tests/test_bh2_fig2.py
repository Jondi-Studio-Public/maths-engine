"""Geometric regressions for the second figure round of the 2026-10-19 bug hunt.

Tick numerals sit under the curves, curves are clipped to the frame, axis names stay out of the data, the
protractor's numerals clear each other, a clock's hands pass behind its numerals, and a few grid-drawing labels stay
inside their frame. Everything is read from the rendered SVG text; no browser is involved.
"""

from __future__ import annotations

import html
import math
import re

import pytest

from mathgen import engine, svg
from mathgen.figure_kinds import placement

_SKILLS = {s.id: s for s in engine.load_skills()[1]}
_TEXT = re.compile(r"<text ([^>]*)>([^<]*)</text>")
_ATTR = re.compile(r'([\w-]+)="([^"]*)"')
SEEDS = range(12)


def _make(skill_id: str, source: int, seed: int) -> dict:
    return engine.make_for_skill(_SKILLS[skill_id], seed=seed, source=source).figure


def _svg(skill_id: str, source: int, seed: int) -> str:
    return svg.figure_svg(_make(skill_id, source, seed))


def _texts(svg_text: str) -> list[dict]:
    out = []
    for m in _TEXT.finditer(svg_text):
        a = dict(_ATTR.findall(m.group(1)))
        out.append(
            {
                "s": html.unescape(m.group(2)),
                "x": float(a["x"]),
                "y": float(a["y"]),
                "size": float(a.get("font-size", 12)),
                "anchor": a.get("text-anchor", "start"),
                "pos": m.start(),
                "halo": "stroke-width" in a,
                "bold": a.get("font-weight") == "bold",
            }
        )
    return out


def _box(t: dict, factor: float = 0.55) -> placement.Box:
    return placement.text_box(t["x"], t["y"], len(t["s"]) * factor * t["size"], t["anchor"], t["size"])


# ---------------------------------------------------------------- cartesian figures

_CARTESIAN = [
    ("senior-graph-translation-read-shift", 0),
    ("senior-graph-translation-read-shift", 1),
    ("graph-y-mx", 3),
    ("f10-describe-transformation-from-points", 2),
    ("senior-recognise-the-graphs-of-y-sin-x-y", 3),
    ("f10-quadratic-parameter-effects", 0),
    ("senior-position-time-tangent-velocity", 2),
]


@pytest.mark.parametrize("skill_id,source", _CARTESIAN)
@pytest.mark.parametrize("seed", SEEDS)
def test_tick_numerals_are_under_the_curves(skill_id, source, seed):
    """R2-F-01: a numeral's paper halo drawn after a curve cuts a gap in it, so numerals come first."""
    text = _svg(skill_id, source, seed)
    curves = [m.start() for m in re.finditer(r'<(?:polyline|path|line)[^>]*stroke-width="2"', text)]
    numerals = [t["pos"] for t in _texts(text) if t["size"] == 11 and t["halo"]]
    if not curves or not numerals:
        pytest.skip("no curve or numerals in this figure")
    assert max(numerals) < min(curves)


@pytest.mark.parametrize("skill_id,source", _CARTESIAN)
@pytest.mark.parametrize("seed", SEEDS)
def test_curves_are_clipped_to_the_frame(skill_id, source, seed):
    """R2-F-03: the clip rectangle is the plot frame, at most the width of a stroke larger."""
    text = _svg(skill_id, source, seed)
    clip = re.search(r'<clipPath[^>]*><rect x="([\d.-]+)" y="([\d.-]+)" width="([\d.]+)" height="([\d.]+)"', text)
    frame = re.search(r'<rect x="([\d.-]+)" y="([\d.-]+)" width="([\d.]+)" height="([\d.]+)" stroke="#[0-9a-f]+"', text)
    assert clip and frame
    cx, cy, cw, ch = (float(v) for v in clip.groups())
    fx, fy, fw, fh = (float(v) for v in frame.groups())
    assert 0 <= fx - cx <= 1.01 and 0 <= fy - cy <= 1.01
    assert 0 <= (cx + cw) - (fx + fw) <= 1.01 and 0 <= (cy + ch) - (fy + fh) <= 1.01


@pytest.mark.parametrize("source", [0, 1, 2])
@pytest.mark.parametrize("seed", SEEDS)
def test_two_curve_labels_stand_at_the_same_end(source, seed):
    """R2-F-02: f and g are both at the top or both at the bottom end of their curves, never one at each end."""
    text = _svg("senior-graph-translation-read-shift", source, seed)
    labels = {t["s"]: t for t in _texts(text) if t["s"] in ("f", "g") and t["halo"]}
    assert set(labels) == {"f", "g"}
    middle = 14 + 150
    assert (labels["f"]["y"] > middle) == (labels["g"]["y"] > middle)


def test_origin_numeral_is_beside_its_tick_when_the_axis_is_the_edge():
    """R2-F-04: with the y axis on the left edge the "0" sits level with the x axis, not 12 units below it."""
    for seed in SEEDS:
        fig = _make("senior-position-time-tangent-velocity", 2, seed)
        text = svg.figure_svg(fig)
        x_axis = re.search(r'<line x1="46" y1="([\d.]+)" x2="[\d.]+" y2="\1" stroke="currentColor"', text)
        zero = [t for t in _texts(text) if t["s"] == "0" and t["size"] == 11]
        if x_axis and zero and fig["y_range"][0] < 0 < fig["y_range"][1]:
            assert abs(zero[0]["y"] - float(x_axis.group(1))) < 0.01


@pytest.mark.parametrize(
    "skill_id,source", [("scatterplot-describe-association", 1), ("scatterplot-describe-association", 2)]
)
@pytest.mark.parametrize("seed", SEEDS)
def test_scatter_axis_names_are_outside_the_plot(skill_id, source, seed):
    """R2-F-06: axis names sit above and below the frame, so no point can lie under them."""
    text = _svg(skill_id, source, seed)
    names = [t for t in _texts(text) if t["bold"]]
    assert len(names) == 2
    top, bottom = 14.0, 14.0 + 300.0
    for t in names:
        box = _box(t, 0.6)
        assert box[3] <= top + 1 or box[1] >= bottom - 1


@pytest.mark.parametrize("seed", SEEDS)
def test_quadratic_y_name_is_not_beside_a_curve_label(seed):
    """R2-F-07: the y name does not sit next to another label so that the two read as one."""
    text = _svg("f10-quadratic-parameter-effects", 0, seed)
    ts = _texts(text)
    name = [t for t in ts if t["bold"] and t["s"] == "y"]
    assert name
    nb = _box(name[0])
    for t in ts:
        if t is name[0] or t["bold"] or t["size"] != 12 or not t["halo"]:
            continue
        b = _box(t)
        assert not placement._overlap(nb, b, 4.0), (t["s"], nb, b)


@pytest.mark.parametrize("seed", SEEDS)
def test_latitude_letters_do_not_shrink_the_latitude_marks(seed):
    """R2-F-05: a place letter on the grid does not shrink the numeral at the edge beside it."""
    text = _svg("senior-latitude-longitude-grid-places", 2, seed)
    marks = [t for t in _texts(text) if "°" in t["s"] and t["anchor"] == "middle"]
    assert marks
    assert all(t["size"] >= 11 for t in marks if t["x"] < 50)


# ---------------------------------------------------------------- protractor and clock


@pytest.mark.parametrize("source", [0, 1, 2])
@pytest.mark.parametrize("seed", SEEDS)
def test_protractor_numerals_do_not_overlap(source, seed):
    text = _svg("f10-measure-angle-protractor", source, seed)
    nums = [t for t in _texts(text) if t["s"].isdigit() and t["size"] <= 8]
    assert len(nums) >= 36
    boxes = [_box(t, 0.556) for t in nums]
    for i, a in enumerate(boxes):
        for b in boxes[i + 1 :]:
            assert not placement._overlap(a, b, 0.0), (a, b)


@pytest.mark.parametrize(
    "skill_id,source",
    [("tell-time-to-hour", 0), ("f10-clock-half-quarter-hour", 0), ("f10-read-time-to-the-minute", 0)],
)
@pytest.mark.parametrize("seed", SEEDS)
def test_clock_numerals_are_drawn_over_the_hands(skill_id, source, seed):
    """A hand that points at a numeral passes behind it, so the numeral is never struck through."""
    text = _svg(skill_id, source, seed)
    hands = [m.start() for m in re.finditer(r'stroke-width="(?:5|3)" stroke-linecap="round"', text)]
    numerals = [t["pos"] for t in _texts(text) if t["s"].isdigit() and t["bold"]]
    assert len(hands) >= 2 and numerals
    assert max(hands) < min(numerals)


# ---------------------------------------------------------------- grid drawings


@pytest.mark.parametrize("seed", SEEDS)
def test_last_step_arrowhead_is_not_under_the_target(seed):
    """N4: the final arrow of a count-the-steps path ends at the edge of the landmark's symbol, not inside it."""
    fig = _make("ey-count-steps-on-grid", 0, seed)
    text = svg.figure_svg(fig)
    heads = re.findall(r'<polygon points="([^"]+)" fill="currentColor"', text)
    assert heads
    tip = [tuple(float(v) for v in p.split(",")) for p in heads[-1].split()][0]
    target = fig["path"]["cells"][-1]
    # centre of the target cell, via the symbol drawn there
    sym = [s for s in fig["symbols"] if (int(s["at"][0]), int(s["at"][1])) == tuple(target)]
    if not sym:
        return
    w, h = fig["grid"]["width"], fig["grid"]["height"]
    frame = re.search(r'<rect x="([\d.]+)" y="([\d.]+)" width="([\d.]+)" height="([\d.]+)" stroke="currentColor"', text)
    assert frame
    gx, gy, gw, gh = (float(v) for v in frame.groups())
    c = gw / w
    cx, cy = gx + (target[0] + 0.5) * c, gy + (h - target[1] - 0.5) * c
    r_sym = max(7.0, min(14.0, c * 0.42))
    assert math.hypot(tip[0] - cx, tip[1] - cy) >= r_sym * 1.05


@pytest.mark.parametrize("source", [0, 1, 2])
@pytest.mark.parametrize("seed", SEEDS)
def test_start_label_stays_inside_the_frame(source, seed):
    """N3: the backing of the "Start" label must not run over the grid frame."""
    text = _svg("ey-follow-directions-grid", source, seed)
    frame = re.search(r'<rect x="([\d.]+)" y="([\d.]+)" width="([\d.]+)" height="([\d.]+)" stroke="currentColor"', text)
    assert frame
    fx, fy, fw, fh = (float(v) for v in frame.groups())
    for t in _texts(text):
        if t["s"] == "Start":
            half = len(t["s"]) * 0.55 * t["size"] / 2 + 3
            assert fx + half - 1 <= t["x"] <= fx + fw - half + 1


@pytest.mark.parametrize("seed", SEEDS)
def test_net_face_labels_stay_inside_the_frame(seed):
    """N5: "Face 1" in a one-square face is stacked, so it neither crosses the frame nor its neighbour."""
    text = _svg("f10-box-nets-on-a-grid", 2, seed)
    frame = re.search(r'<rect x="([\d.]+)" y="([\d.]+)" width="([\d.]+)" height="([\d.]+)" stroke="currentColor"', text)
    fx, fy, fw, fh = (float(v) for v in frame.groups())
    for t in _texts(text):
        if t["bold"] and ("Face" in t["s"] or t["s"].isdigit()):
            b = _box(t, 0.55)
            assert b[0] >= fx - 1 and b[2] <= fx + fw + 1, (t["s"], b, fx, fw)


def test_ruler_end_numerals_stay_inside_the_ruler():
    """N6: the numerals at the two ends of a ruler do not sit on its edge."""
    checked = 0
    for skill in _SKILLS.values():
        if not skill.needs_figure:
            continue
        for i in range(len(skill.sources)):
            try:
                fig = engine.make_for_skill(skill, seed=3, source=i).figure
            except Exception:
                continue
            if not fig or fig.get("kind") != "scale" or fig.get("variant") != "ruler":
                continue
            text = svg.figure_svg(fig)
            for t in _texts(text):
                if t["size"] == 12 and t["anchor"] == "middle" and t["y"] == 86:
                    b = _box(t, 0.55)
                    assert b[0] >= 30 + 2 and b[2] <= 370 - 2
                    checked += 1
    assert checked
