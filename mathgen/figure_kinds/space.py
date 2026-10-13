"""Geometry and diagram figure kinds: ``diagram``, ``tree_diagram`` and ``venn``.

``diagram`` draws labelled two-dimensional geometry (triangles, crossing and parallel lines, angles at a
point). A few builders return a ``diagram`` spec for the common angle pictures
(:func:`angles_at_point`, :func:`angles_on_line`, :func:`crossing_lines`, :func:`parallel_transversal`) and
for a right triangle (:func:`right_triangle`) so a generator never lays out coordinates by hand.

Every spec is plain JSON-able data. The front end draws it and never evaluates maths; it only places text
and arcs from the numbers given here.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction
from typing import Iterable

from mathgen.figures import _clean, _is_num, _xy, check, register_kind

Number = int | float

_EPS = 1e-6
MAX_POINTS = 16
MAX_SEGMENTS = 20
MAX_POLYGONS = 3
MAX_ANGLES = 10
MAX_LABEL = 12
#: A labelled angle arc needs room for its text.
MIN_LABELLED_SWEEP = 18.0
#: Drawn angles may differ from a numeric label by this much when the diagram is to scale.
ANGLE_TOLERANCE = 2.0
#: Drawn lengths may differ from a numeric label by this fraction when the diagram is to scale.
LENGTH_TOLERANCE = 0.06
#: Equal-length tick marks need equal drawn lengths to this fraction.
TICK_TOLERANCE = 0.04

_ANGLE_LABEL = re.compile(r"^(\d+(?:\.\d+)?)°$")
_LENGTH_LABEL = re.compile(r"^(\d+(?:\.\d+)?)(?: ?[a-zA-Z]{1,3})?$")


# ---------------------------------------------------------------- geometry helpers


def _dist(a, b) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _bearing(at, to) -> float:
    return math.degrees(math.atan2(to[1] - at[1], to[0] - at[0])) % 360


def sweep_degrees(at, frm, to) -> float:
    """Counter-clockwise sweep in degrees (frame y points up) from the ray at->frm to the ray at->to."""
    return (_bearing(at, to) - _bearing(at, frm)) % 360


def angle_class(deg: float) -> str:
    """acute / right / obtuse / straight / reflex; only (almost exactly) 90 and 180 are right and straight."""
    if abs(deg - 90) <= 0.5:
        return "right"
    if abs(deg - 180) <= 0.5:
        return "straight"
    if deg < 90:
        return "acute"
    if deg < 180:
        return "obtuse"
    return "reflex"


def _on_segment(p, a, b) -> bool:
    """Is ``p`` on the closed segment a-b (within a small tolerance)?"""
    ab = _dist(a, b)
    if ab < _EPS:
        return False
    cross = abs((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])) / ab
    if cross > 1e-4:
        return False
    dot = ((p[0] - a[0]) * (b[0] - a[0]) + (p[1] - a[1]) * (b[1] - a[1])) / ab
    return -1e-4 <= dot <= ab + 1e-4


def _ray_drawn(at, target, drawn) -> bool:
    """Is the ray from ``at`` toward ``target`` along a drawn segment that contains ``at``?"""
    d = (target[0] - at[0], target[1] - at[1])
    for a, b in drawn:
        if not _on_segment(at, a, b):
            continue
        e = (b[0] - a[0], b[1] - a[1])
        cross = d[0] * e[1] - d[1] * e[0]
        if abs(cross) > 1e-4 * math.hypot(*d) * math.hypot(*e):
            continue
        # the drawn segment must extend from ``at`` in the ray's direction
        if (b[0] - at[0]) * d[0] + (b[1] - at[1]) * d[1] > 1e-6 or (a[0] - at[0]) * d[0] + (a[1] - at[1]) * d[1] > 1e-6:
            return True
    return False


def _proper_cross(a, b, c, d) -> bool:
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    o1, o2, o3, o4 = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
    return o1 * o2 < -_EPS and o3 * o4 < -_EPS


def _polygon_area(pts) -> float:
    return abs(sum(p[0] * q[1] - q[0] * p[1] for p, q in zip(pts, pts[1:] + pts[:1]))) / 2


# ---------------------------------------------------------------- diagram


def diagram(
    *,
    alt: str,
    points: Iterable[dict],
    segments: Iterable[dict] = (),
    polygons: Iterable[dict] = (),
    angles: Iterable[dict] = (),
    frame: tuple[Number, Number] = (100, 100),
    not_to_scale: bool = False,
    title: str = "",
) -> dict:
    """A labelled two-dimensional geometry diagram.

    The drawing frame is an abstract box ``[0, W] x [0, H]`` with the origin at the BOTTOM-LEFT and ``y``
    pointing UP (so counter-clockwise is the usual mathematical direction). The renderer scales the frame to
    fit its box, keeping equal scale on both axes, and draws text at a fixed readable size.

    - ``frame``: ``[W, H]``, positive numbers. Every point lies inside it.
    - ``points``: ``{"id": str, "at": [x, y], "label"?: str}``. ``id`` is unique and is how every other item
      refers to the point. A point with a ``label`` (at most 4 characters) is named on the drawing: the label
      is placed just outside the figure, opposite the mean direction of the segments leaving that point. A
      point with no label is not drawn (it only anchors segments and angles).
    - ``segments``: ``{"from": id, "to": id, "label"?: str, "ticks"?: 1-3, "arrows"?: 1-2, "dashed"?: bool}``.
      A straight line from point to point. ``label`` (at most 12 characters: a length such as "5 cm", "13" or
      "x") is placed at the midpoint, offset perpendicular to the segment on the side away from the centroid
      of all points, as horizontal text. ``ticks`` draws that many short cross-marks at the midpoint
      (equal-length marking: every segment with the same tick count is the same drawn length). ``arrows``
      draws that many chevrons at the midpoint pointing along ``from -> to`` (parallel-line marking: every
      segment with the same arrow count is parallel and points the same way). ``dashed`` draws a dashed line
      (a perpendicular height, a construction line).
    - ``polygons``: ``{"points": [id, ...], "fill"?: bool}``. A closed polygon of 3 or more points joined in
      order, drawn as outline (and, with ``fill``, lightly shaded). A polygon is simple (no self-crossing).
      A ``segments`` entry on the same pair of points only adds a label, ticks, arrows or a dash style to
      that edge; the renderer draws one line, not two.
    - ``angles``: ``{"at": id, "from": id, "to": id, "label"?: str, "right"?: bool}``. The angle at vertex
      ``at`` measured COUNTER-CLOCKWISE from the ray ``at -> from`` to the ray ``at -> to``, so a reflex
      angle is simply a sweep over 180 degrees. Both rays lie along drawn segments or polygon edges that
      contain ``at``. The renderer draws an arc of constant radius inside that sweep and puts ``label``
      ("65°", "x", "(3x + 10)°") at the middle of the sweep just beyond the arc. With ``right: true`` it
      draws a small square in the corner instead of an arc, and the sweep is 90 degrees; a right angle
      carries no label. Sweeps at one vertex never overlap, and a labelled sweep is at least 18 degrees.
    - ``not_to_scale``: ``true`` makes the renderer print "Not drawn to scale" under the figure. When false
      every numeric angle label ("65°") equals the drawn sweep within 2 degrees and every numeric segment
      label ("5 cm") is proportional to the drawn length within 6%. When true they need only be the same
      kind of angle (acute, obtuse, reflex) and the same order of length when the numbers differ by a
      quarter or more, so the picture never contradicts the labels grossly.
    """
    spec = {
        "kind": "diagram",
        "alt": alt,
        "frame": [_clean(frame[0]), _clean(frame[1])],
        "not_to_scale": bool(not_to_scale),
        "points": [{**p, "at": [_clean(p["at"][0]), _clean(p["at"][1])]} for p in points],
        "segments": [dict(s) for s in segments],
        "polygons": [dict(p) for p in polygons],
        "angles": [dict(a) for a in angles],
    }
    if title:
        spec["title"] = title
    return check(spec)


def _validate_diagram(spec: dict) -> list[str]:
    problems: list[str] = []
    frame = spec.get("frame")
    if not (isinstance(frame, (list, tuple)) and len(frame) == 2 and all(_is_num(v) and v > 0 for v in frame)):
        return ["frame must be [W, H] with positive numbers"]
    if not isinstance(spec.get("not_to_scale"), bool):
        problems.append("not_to_scale must be a bool")
    raw_points = spec.get("points", [])
    if not isinstance(raw_points, list) or not raw_points:
        return problems + ["a diagram needs points"]
    if len(raw_points) > MAX_POINTS:
        problems.append(f"more than {MAX_POINTS} points")
    pos: dict[str, tuple[float, float]] = {}
    for i, p in enumerate(raw_points):
        pid = p.get("id")
        if not isinstance(pid, str) or not pid:
            problems.append(f"points[{i}].id must be a non-empty string")
            continue
        if pid in pos:
            problems.append(f"point id {pid!r} is repeated")
            continue
        if not _xy(p.get("at")):
            problems.append(f"point {pid!r}.at must be [x, y]")
            continue
        x, y = p["at"]
        if not (0 <= x <= frame[0] and 0 <= y <= frame[1]):
            problems.append(f"point {pid!r} at {p['at']} is outside the frame")
        label = p.get("label")
        if label is not None and (not isinstance(label, str) or not label or len(label) > 4):
            problems.append(f"point {pid!r}.label must be a short non-empty string")
        pos[pid] = (x, y)
    ids = list(pos)
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            if _dist(pos[a], pos[b]) < _EPS:
                problems.append(f"points {a!r} and {b!r} coincide")
    if problems:
        return problems

    # ---- segments
    drawn: list[tuple[tuple, tuple]] = []
    seg_pairs: set[frozenset] = set()
    segs = spec.get("segments", [])
    if len(segs) > MAX_SEGMENTS:
        problems.append(f"more than {MAX_SEGMENTS} segments")
    seg_info = []
    for i, s in enumerate(segs):
        a, b = s.get("from"), s.get("to")
        if a not in pos or b not in pos:
            problems.append(f"segments[{i}] refers to a point that does not exist")
            continue
        if a == b or _dist(pos[a], pos[b]) < _EPS:
            problems.append(f"segments[{i}] has no length")
            continue
        pair = frozenset((a, b))
        if pair in seg_pairs:
            problems.append(f"segments[{i}] repeats the segment {a}-{b}")
        seg_pairs.add(pair)
        drawn.append((pos[a], pos[b]))
        label = s.get("label")
        if label is not None and (not isinstance(label, str) or not label or len(label) > MAX_LABEL):
            problems.append(f"segments[{i}].label must be a non-empty string of at most {MAX_LABEL} characters")
        for key, top in (("ticks", 3), ("arrows", 2)):
            v = s.get(key)
            if v is not None and (not isinstance(v, int) or isinstance(v, bool) or not (1 <= v <= top)):
                problems.append(f"segments[{i}].{key} must be an integer from 1 to {top}")
        if s.get("ticks") and s.get("arrows"):
            problems.append(f"segments[{i}] has both ticks and arrows")
        if "dashed" in s and not isinstance(s["dashed"], bool):
            problems.append(f"segments[{i}].dashed must be a bool")
        seg_info.append((i, s, _dist(pos[a], pos[b])))

    # ---- polygons
    polys = spec.get("polygons", [])
    if len(polys) > MAX_POLYGONS:
        problems.append(f"more than {MAX_POLYGONS} polygons")
    for i, poly in enumerate(polys):
        pts = poly.get("points")
        if not (isinstance(pts, list) and len(pts) >= 3 and all(p in pos for p in pts)):
            problems.append(f"polygons[{i}] needs 3 or more existing points")
            continue
        if len(set(pts)) != len(pts):
            problems.append(f"polygons[{i}] repeats a point")
            continue
        coords = [pos[p] for p in pts]
        if _polygon_area(coords) < 1e-3:
            problems.append(f"polygons[{i}] has no area")
            continue
        n = len(coords)
        for e1 in range(n):
            for e2 in range(e1 + 1, n):
                if e2 == e1 + 1 or (e1 == 0 and e2 == n - 1):
                    continue
                if _proper_cross(coords[e1], coords[(e1 + 1) % n], coords[e2], coords[(e2 + 1) % n]):
                    problems.append(f"polygons[{i}] crosses itself")
        for p, q in zip(coords, coords[1:] + coords[:1]):
            drawn.append((p, q))

    # ---- ticks and arrows are claims about equality / parallelism
    for key in ("ticks", "arrows"):
        groups: dict[int, list] = {}
        for i, s, length in seg_info:
            if s.get(key):
                groups.setdefault(s[key], []).append((i, s, length))
        for count, members in groups.items():
            if len(members) < 2:
                problems.append(f"only one segment has {count} {key}; marks must come in matching sets")
                continue
            if key == "ticks":
                lengths = [m[2] for m in members]
                if max(lengths) > min(lengths) * (1 + TICK_TOLERANCE):
                    problems.append(f"segments marked with {count} ticks are drawn with different lengths")
            else:
                b0 = _bearing(pos[members[0][1]["from"]], pos[members[0][1]["to"]])
                for _i, s, _l in members[1:]:
                    delta = (_bearing(pos[s["from"]], pos[s["to"]]) - b0) % 360
                    if min(delta, 360 - delta) <= 1.0:
                        continue
                    if abs(delta - 180) <= 1.0:
                        problems.append(f"segments marked with {count} arrows point in opposite directions")
                    else:
                        problems.append(f"segments marked with {count} arrows are not drawn parallel")

    # ---- numeric segment labels against the drawn lengths
    numeric = []
    for _i, s, length in seg_info:
        m = _LENGTH_LABEL.match(s.get("label", "") or "")
        if m and float(m.group(1)) > 0:
            numeric.append((float(m.group(1)), length))
    if len(numeric) >= 2:
        if not spec.get("not_to_scale"):
            ratios = [length / value for value, length in numeric]
            if max(ratios) > min(ratios) * (1 + LENGTH_TOLERANCE):
                problems.append("segment labels are not proportional to the drawn lengths (set not_to_scale)")
        else:
            for k, (v1, l1) in enumerate(numeric):
                for v2, l2 in numeric[k + 1 :]:
                    if (v1 >= v2 * 1.25 and l1 < l2) or (v2 >= v1 * 1.25 and l2 < l1):
                        problems.append("a longer labelled segment is drawn shorter than a shorter one")

    # ---- angles
    angs = spec.get("angles", [])
    if len(angs) > MAX_ANGLES:
        problems.append(f"more than {MAX_ANGLES} angles")
    by_vertex: dict[str, list[tuple[float, float, int]]] = {}
    for i, a in enumerate(angs):
        at, frm, to = a.get("at"), a.get("from"), a.get("to")
        if at not in pos or frm not in pos or to not in pos:
            problems.append(f"angles[{i}] refers to a point that does not exist")
            continue
        if len({at, frm, to}) != 3:
            problems.append(f"angles[{i}] needs three different points")
            continue
        if not (_ray_drawn(pos[at], pos[frm], drawn) and _ray_drawn(pos[at], pos[to], drawn)):
            problems.append(f"angles[{i}] has a ray that is not along a drawn segment")
            continue
        sweep = sweep_degrees(pos[at], pos[frm], pos[to])
        if sweep < 1 or sweep > 359:
            problems.append(f"angles[{i}] has no opening")
            continue
        label = a.get("label")
        if label is not None and (not isinstance(label, str) or not label or len(label) > MAX_LABEL):
            problems.append(f"angles[{i}].label must be a non-empty string of at most {MAX_LABEL} characters")
            label = None
        if "right" in a and not isinstance(a["right"], bool):
            problems.append(f"angles[{i}].right must be a bool")
        if a.get("right"):
            if abs(sweep - 90) > ANGLE_TOLERANCE:
                problems.append(f"angles[{i}] is marked right but is drawn {sweep:.1f} degrees")
            if label:
                problems.append(f"angles[{i}] is a right angle and must not carry a label")
        elif label:
            if sweep < MIN_LABELLED_SWEEP:
                problems.append(f"angles[{i}] is labelled but drawn only {sweep:.1f} degrees wide")
            m = _ANGLE_LABEL.match(label)
            if m:
                value = float(m.group(1))
                if not spec.get("not_to_scale"):
                    if abs(value - sweep) > ANGLE_TOLERANCE:
                        problems.append(f"angles[{i}] is labelled {label} but drawn {sweep:.1f} degrees")
                elif angle_class(value) != angle_class(sweep):
                    problems.append(
                        f"angles[{i}] is labelled {label} ({angle_class(value)}) but drawn "
                        f"{angle_class(sweep)} ({sweep:.1f} degrees)"
                    )
        by_vertex.setdefault(at, []).append((_bearing(pos[at], pos[frm]), sweep, i))
    for arcs in by_vertex.values():
        for k, (s1, l1, i1) in enumerate(arcs):
            for s2, l2, i2 in arcs[k + 1 :]:
                if (s2 - s1) % 360 < l1 - 0.5 or (s1 - s2) % 360 < l2 - 0.5:
                    problems.append(f"angles[{i1}] and angles[{i2}] overlap at the same vertex")
    return problems


def _diagram_texts(spec: dict) -> list[str]:
    out = [a["label"] for a in spec.get("angles", []) if a.get("label")]
    if spec.get("not_to_scale"):
        out.append("Not drawn to scale")
    return out


register_kind("diagram", _validate_diagram, texts=_diagram_texts)


# ---------------------------------------------------------------- angle and triangle builders


def _polar(centre, radius, bearing_deg):
    t = math.radians(bearing_deg)
    return (centre[0] + radius * math.cos(t), centre[1] + radius * math.sin(t))


def angles_at_point(
    *, sweeps: list[Number], labels: list[str | None], alt: str, start: Number = 0, not_to_scale: bool = True
) -> dict:
    """Rays from one point, with the angle between each neighbouring pair marked.

    ``sweeps`` are the drawn sizes in degrees and must total 360; the rays start at bearing ``start`` and the
    angles run counter-clockwise, so ``labels[i]`` marks the arc between ray ``i`` and ray ``i + 1`` (a
    ``None`` label draws no arc for that angle). The
    point is ``O`` and the ray ends are ``R0``, ``R1``, ... (none labelled).
    """
    if abs(sum(sweeps) - 360) > 1e-6 or len(sweeps) != len(labels) or len(sweeps) < 3:
        raise ValueError("sweeps must total 360 and match labels, with at least 3 rays")
    centre = (50, 50)
    pts = [{"id": "O", "at": centre}]
    segs = []
    bearing = start
    for i, s in enumerate(sweeps):
        pts.append({"id": f"R{i}", "at": _polar(centre, 40, bearing)})
        segs.append({"from": "O", "to": f"R{i}"})
        bearing += s
    arcs = []
    for i, lab in enumerate(labels):
        if lab:
            arcs.append({"at": "O", "from": f"R{i}", "to": f"R{(i + 1) % len(sweeps)}", "label": lab})
    return diagram(alt=alt, points=pts, segments=segs, angles=arcs, not_to_scale=not_to_scale)


def angles_on_line(
    *, sweeps: list[Number], labels: list[str | None], alt: str, tilt: Number = 0, not_to_scale: bool = True
) -> dict:
    """A straight line through a point with rays on one side of it.

    ``sweeps`` are the drawn sizes in degrees and total 180; the line runs along bearing ``tilt`` (its right
    end, ``R0``) to ``tilt + 180`` (its left end, the last ray) and the angles are counted counter-clockwise
    from the right end, so with ``tilt = 0`` the rays rise above the line and ``labels[0]`` is the angle
    nearest its right end.
    """
    if abs(sum(sweeps) - 180) > 1e-6 or len(sweeps) != len(labels) or not sweeps:
        raise ValueError("sweeps must total 180 and match labels")
    centre = (50, 50)
    pts = [{"id": "O", "at": centre}, {"id": "R0", "at": _polar(centre, 42, tilt)}]
    segs = [{"from": "O", "to": "R0"}]
    bearing = tilt
    for i, s in enumerate(sweeps):
        bearing += s
        pts.append({"id": f"R{i + 1}", "at": _polar(centre, 42, bearing)})
        segs.append({"from": "O", "to": f"R{i + 1}"})
    arcs = []
    for i, lab in enumerate(labels):
        if lab:
            arcs.append({"at": "O", "from": f"R{i}", "to": f"R{i + 1}", "label": lab})
    return diagram(alt=alt, points=pts, segments=segs, angles=arcs, not_to_scale=not_to_scale)


def crossing_lines(
    *, theta: Number, labels: list[str | None], alt: str, rotation: Number = 0, not_to_scale: bool = True
) -> dict:
    """Two straight lines crossing at a point ``O``.

    One line runs along bearing ``rotation``, the other along ``rotation + theta`` (0 < theta < 180). Four
    angles are formed, counter-clockwise from the first line's right end: ``labels[0]`` is ``theta`` wide,
    ``labels[1]`` is ``180 - theta``, ``labels[2]`` is ``theta`` (vertically opposite ``labels[0]``) and
    ``labels[3]`` is ``180 - theta``.
    """
    if not (0 < theta < 180) or len(labels) != 4:
        raise ValueError("theta must be in (0, 180) and labels must have four entries")
    centre = (50, 50)
    rays = [rotation, rotation + theta, rotation + 180, rotation + 180 + theta]
    pts = [{"id": "O", "at": centre}] + [{"id": f"R{i}", "at": _polar(centre, 42, r)} for i, r in enumerate(rays)]
    segs = [{"from": "R0", "to": "R2"}, {"from": "R1", "to": "R3"}]
    arcs = []
    for i, lab in enumerate(labels):
        if lab:
            arcs.append({"at": "O", "from": f"R{i}", "to": f"R{(i + 1) % 4}", "label": lab})
    return diagram(alt=alt, points=pts, segments=segs, angles=arcs, not_to_scale=not_to_scale)


def parallel_transversal(
    *,
    theta: Number,
    labels: dict[tuple[int, int], str],
    alt: str,
    rotation: Number = 0,
    not_to_scale: bool = True,
) -> dict:
    """Two parallel lines (arrow-marked) cut by a transversal.

    In an unrotated picture the parallels are horizontal and the transversal leans at ``theta`` degrees
    (50 to 130) from the right-pointing direction. The upper parallel meets the transversal at intersection
    0 (point ``P``) and the lower at intersection 1 (point ``Q``). At each intersection four angles are
    numbered counter-clockwise starting between the right-pointing parallel and the transversal's upward
    end: ``k = 0`` is ``theta`` wide, ``k = 1`` is ``180 - theta`` (above-left), ``k = 2`` is ``theta``
    (below-left) and ``k = 3`` is ``180 - theta`` (below-right). ``labels`` maps ``(intersection, k)`` to the
    text of that angle. The whole picture is turned counter-clockwise by ``rotation`` degrees about the
    frame centre.
    """
    if not (50 <= theta <= 130):
        raise ValueError("theta must be between 50 and 130 degrees")
    cot = math.cos(math.radians(theta)) / math.sin(math.radians(theta))
    h, half, ext = 14, 38, 18
    local = {
        "P": (h * cot, h),
        "Q": (-h * cot, -h),
        "A1": (-half, h),
        "B1": (half, h),
        "A2": (-half, -h),
        "B2": (half, -h),
        "T1": (h * cot + ext * cot, h + ext),
        "T0": (-h * cot - ext * cot, -h - ext),
    }
    rot = math.radians(rotation)

    def place(p):
        x, y = p
        return (50 + x * math.cos(rot) - y * math.sin(rot), 50 + x * math.sin(rot) + y * math.cos(rot))

    pts = [{"id": k, "at": place(v)} for k, v in local.items()]
    segs = [
        {"from": "A1", "to": "B1", "arrows": 1},
        {"from": "A2", "to": "B2", "arrows": 1},
        {"from": "T0", "to": "T1"},
    ]
    # rays out of an intersection, counter-clockwise from the right-pointing parallel
    rays = {0: ["B1", "T1", "A1", "T0"], 1: ["B2", "T1", "A2", "T0"]}
    arcs = []
    for (inter, k), lab in sorted(labels.items()):
        r = rays[inter]
        arcs.append({"at": "PQ"[inter], "from": r[k], "to": r[(k + 1) % 4], "label": lab})
    return diagram(alt=alt, points=pts, segments=segs, angles=arcs, not_to_scale=not_to_scale)


def right_triangle(
    *,
    legs: tuple[Number, Number],
    labels: dict[str, str],
    alt: str,
    orientation: int = 0,
    not_to_scale: bool = True,
) -> dict:
    """A right triangle with the right angle marked and its three sides optionally labelled.

    ``legs`` are the DRAWN lengths of the two legs ``(a, b)`` (the generator chooses them, so a
    not-to-scale picture can keep honest proportions without giving the answer away). The right angle is at
    vertex ``C``; in orientation 0 leg ``a = CB`` runs along the bottom, leg ``b = CA`` rises from C and
    ``c = AB`` is the hypotenuse. ``labels`` maps ``"a"``, ``"b"`` and ``"c"`` to side labels.
    ``orientation`` is 0-11. 0-3 turn the triangle 0, 90, 180 and 270 degrees counter-clockwise, 4-7 do the
    same to its mirror image (legs along the axes, the hypotenuse slanting). 8 and 9 lay the hypotenuse
    horizontal, with the right angle below it (8) or above it (9); 10 and 11 are their mirror images, so the
    hypotenuse is not always at the bottom and not always slanted. Vertices are labelled A, B, C.
    """
    a, b = float(legs[0]), float(legs[1])
    scale = min(70 / a, 70 / b)
    a, b = a * scale, b * scale
    raw = {"C": (0.0, 0.0), "B": (a, 0.0), "A": (0.0, b)}
    cx, cy = a / 2, b / 2
    mirror = orientation in (4, 5, 6, 7, 10, 11)
    out = {}
    for k, (x, y) in raw.items():
        x, y = x - cx, y - cy
        out[k] = (-x if mirror else x, y)
    if orientation < 8:
        turns = orientation % 4
    else:
        # turn A -> B to point right (a whole number of quarter turns cannot do it), then half a turn more if the
        # right angle is on the wrong side of the hypotenuse
        ax, ay = out["A"]
        bx, by = out["B"]
        rho = -math.atan2(by - ay, bx - ax)
        cos, sin = math.cos(rho), math.sin(rho)
        out = {k: (x * cos - y * sin, x * sin + y * cos) for k, (x, y) in out.items()}
        below = out["C"][1] < out["A"][1]
        if below != (orientation in (8, 10)):
            out = {k: (-x, -y) for k, (x, y) in out.items()}
        turns = 0
    for _ in range(turns):
        out = {k: (-y, x) for k, (x, y) in out.items()}
    xs = [p[0] for p in out.values()]
    ys = [p[1] for p in out.values()]
    fit = min(1.0, 80 / (max(xs) - min(xs)), 80 / (max(ys) - min(ys)))
    mx, my = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    out = {k: (50 + (x - mx) * fit, 50 + (y - my) * fit) for k, (x, y) in out.items()}
    pts = [{"id": k, "at": v, "label": k} for k, v in out.items()]
    segs = []
    for key, (p, q) in {"a": ("C", "B"), "b": ("C", "A"), "c": ("A", "B")}.items():
        seg = {"from": p, "to": q}
        if labels.get(key):
            seg["label"] = labels[key]
        segs.append(seg)
    frm, to = ("B", "A") if sweep_degrees(out["C"], out["B"], out["A"]) < 180 else ("A", "B")
    return diagram(
        alt=alt,
        points=pts,
        segments=segs,
        polygons=[{"points": ["A", "B", "C"]}],
        angles=[{"at": "C", "from": frm, "to": to, "right": True}],
        not_to_scale=not_to_scale,
    )


# ---------------------------------------------------------------- tree_diagram

_PROB = re.compile(r"^(?:(\d+)/(\d+)|(0?\.\d+))$")
BLANK = "?"
MAX_TREE_LEAVES = 12


def parse_probability(text: str) -> Fraction | None:
    """The value of a branch label ("3/5", "0.4"), or ``None`` for a blank ("?") or anything else."""
    m = _PROB.match(text.strip()) if isinstance(text, str) else None
    if not m:
        return None
    if m.group(3) is not None:
        return Fraction(m.group(3))
    return Fraction(int(m.group(1)), int(m.group(2))) if int(m.group(2)) else None


def tree_diagram(*, alt: str, root: dict, title: str = "") -> dict:
    """A probability tree read left to right, two or three stages deep.

    ``root`` is ``{"label"?: str, "children": [node, ...]}``; the root is drawn at the left (its optional
    ``label``, e.g. "Start", beside it). Every other ``node`` is
    ``{"branch": str, "label": str, "children"?: [node, ...], "outcome"?: str, "outcome_p"?: str}``:

    - ``branch`` is the text written on the branch that leads INTO the node: a probability as a fraction
      ("3/5") or a decimal ("0.4"), or ``"?"`` for a branch the learner works out (drawn as an empty box).
    - ``label`` (at most 14 characters) is the event at the node ("Red", "Rain"), drawn at the node.
    - A leaf (no ``children``) must carry ``outcome``, the whole path as text ("Red, Blue"), drawn to the
      right of the leaf; ``outcome_p`` optionally gives the probability of that outcome ("9/25" or "?").
    - Every leaf is at the same depth, 2 or 3 stages below the root; a node has 2 to 4 children, and there
      are at most 12 leaves. Sibling branches that are all numbers total exactly 1; with one "?" among them
      the rest total less than 1. Nodes are laid out top to bottom in the order given.
    """
    spec = {"kind": "tree_diagram", "alt": alt, "root": root}
    if title:
        spec["title"] = title
    return check(spec)


def _walk_tree(node, path, depth, problems, leaves):
    kids = node.get("children")
    if not kids:
        leaves.append((depth, path, node))
        return
    if not isinstance(kids, list) or not (2 <= len(kids) <= 4):
        problems.append("every node with branches needs 2 to 4 of them")
        return
    values, blanks = [], 0
    for kid in kids:
        if not isinstance(kid, dict):
            problems.append("a tree node must be a dict")
            return
        branch = kid.get("branch")
        if not isinstance(branch, str) or not branch.strip():
            problems.append("every branch needs a label (a probability or '?')")
            continue
        if branch == BLANK:
            blanks += 1
        else:
            v = parse_probability(branch)
            if v is None or not (0 < v < 1):
                problems.append(f"branch label {branch!r} is not a probability strictly between 0 and 1")
            else:
                values.append(v)
        label = kid.get("label")
        if not isinstance(label, str) or not label or len(label) > 14:
            problems.append("every node needs a short label")
    if blanks > 1:
        problems.append("a set of sibling branches has more than one blank")
    elif len(values) == len(kids):
        if sum(values) != 1:
            problems.append(f"sibling branches total {sum(values)}, not 1")
    elif blanks == 1 and len(values) == len(kids) - 1 and sum(values) >= 1:
        problems.append("the siblings of a blank branch already total 1 or more")
    for kid in kids:
        _walk_tree(kid, path + [kid], depth + 1, problems, leaves)


def _validate_tree(spec: dict) -> list[str]:
    root = spec.get("root")
    if not isinstance(root, dict):
        return ["root must be a dict"]
    problems: list[str] = []
    leaves: list = []
    _walk_tree(root, [], 0, problems, leaves)
    if problems:
        return problems
    depths = {d for d, _p, _n in leaves}
    if len(depths) != 1 or not depths <= {2, 3}:
        problems.append("a tree must have 2 or 3 stages with every leaf at the same depth")
    if len(leaves) > MAX_TREE_LEAVES:
        problems.append(f"more than {MAX_TREE_LEAVES} leaves")
    for _d, path, leaf in leaves:
        outcome = leaf.get("outcome")
        if not isinstance(outcome, str) or not outcome or len(outcome) > 24:
            problems.append("every leaf needs a short outcome text")
        op = leaf.get("outcome_p")
        if op is not None and op != BLANK:
            v = parse_probability(op)
            if v is None or not (0 < v < 1):
                problems.append(f"outcome_p {op!r} is not a probability or '?'")
                continue
            branches = [parse_probability(n["branch"]) for n in path]
            if all(b is not None for b in branches):
                prod = Fraction(1)
                for b in branches:
                    prod *= b
                if prod != v:
                    problems.append(f"outcome_p {op!r} is not the product of its branches ({prod})")
    return problems


def _tree_texts(spec: dict) -> list[str]:
    out: list[str] = []

    def walk(node):
        for key in ("label", "branch", "outcome", "outcome_p"):
            if isinstance(node.get(key), str):
                out.append(node[key])
        for kid in node.get("children") or ():
            walk(kid)

    if isinstance(spec.get("root"), dict):
        walk(spec["root"])
    return out


register_kind("tree_diagram", _validate_tree, texts=_tree_texts)


# ---------------------------------------------------------------- venn

_REGION_COUNT = re.compile(r"^\d{1,3}$")
_REGION_ELEMENTS = re.compile(r"^[A-Za-z0-9]{1,3}(, [A-Za-z0-9]{1,3}){0,5}$")


def venn_region_keys(set_ids: list[str]) -> list[str]:
    """Every region key of a Venn diagram over ``set_ids`` (e.g. ``["A", "B"]``), including ``"none"``.

    A key is the ids of the sets a region lies in, joined in set order: ``"A"`` is inside A only, ``"AB"``
    inside both A and B only, ``"ABC"`` inside all three, ``"none"`` outside every circle.
    """
    n = len(set_ids)
    keys = ["".join(set_ids[i] for i in range(n) if mask >> i & 1) for mask in range(1, 1 << n)]
    return keys + ["none"]


def venn(*, alt: str, sets: list[dict], regions: dict[str, str], title: str = "", universe: str = "U") -> dict:
    """A Venn diagram of two or three sets inside a universal rectangle.

    - ``universe``: the label written in the corner of the rectangle ("U" or the name of the whole group).
    - ``sets``: two or three of ``{"id": "A", "name"?: str}``; ``id`` is a single capital letter, unique.
      ``name`` (a short word such as "Tennis", at most 12 characters) is written beside the circle,
      defaulting to the ``id``. Layout: two sets are overlapping circles, A left and B right; three sets are
      A top-left, B top-right and C below, each pair overlapping and all three overlapping in the middle.
      The rectangle surrounds them and the "none" region is the rectangle outside every circle.
    - ``regions``: one entry for EVERY region (see :func:`venn_region_keys`), mapping its key to the text
      written inside it: a whole-number count ("12"), a short comma-separated list of elements ("2, 4, 6")
      or ``"?"`` for a region the learner works out. Counts are exclusive: "A" counts only what is in A and
      in neither other set.
    """
    spec = {
        "kind": "venn",
        "alt": alt,
        "universe": universe,
        "sets": [dict(s) for s in sets],
        "regions": dict(regions),
    }
    if title:
        spec["title"] = title
    return check(spec)


def _validate_venn(spec: dict) -> list[str]:
    problems: list[str] = []
    sets = spec.get("sets")
    if not (isinstance(sets, list) and len(sets) in (2, 3)):
        return ["a Venn diagram has two or three sets"]
    ids = []
    for s in sets:
        sid = s.get("id") if isinstance(s, dict) else None
        if not (isinstance(sid, str) and re.fullmatch(r"[A-Z]", sid)) or sid in ids:
            problems.append("each set needs a unique single capital letter id")
        else:
            ids.append(sid)
        name = s.get("name") if isinstance(s, dict) else None
        if name is not None and (not isinstance(name, str) or not name or len(name) > 12):
            problems.append("a set name must be a short non-empty string")
    if problems:
        return problems
    if not isinstance(spec.get("universe"), str) or not spec["universe"] or len(spec["universe"]) > 14:
        problems.append("universe must be a short label")
    regions = spec.get("regions")
    keys = venn_region_keys(ids)
    if not isinstance(regions, dict) or set(regions) != set(keys):
        return problems + [f"regions must have exactly the keys {keys}"]
    for key, text in regions.items():
        if not isinstance(text, str):
            problems.append(f"region {key!r} must be text")
        elif not (text == BLANK or _REGION_COUNT.match(text) or _REGION_ELEMENTS.match(text)):
            problems.append(f"region {key!r} text {text!r} is not a count, a short element list or '?'")
    return problems


def _venn_texts(spec: dict) -> list[str]:
    out = [spec.get("universe", "")]
    out.extend(s.get("name") or s.get("id", "") for s in spec.get("sets", []))
    out.extend(spec.get("regions", {}).values())
    return out


register_kind("venn", _validate_venn, texts=_venn_texts)
