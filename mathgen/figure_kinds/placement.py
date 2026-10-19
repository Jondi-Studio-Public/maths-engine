"""Where to put a text label so it reads as belonging to the thing it names.

Used by the Cartesian renderer for curve, line and point labels. Everything is in drawing units (pixels of the
400-wide canvas). A label is a box; a candidate position is accepted only when its box

* stays inside the plot area,
* overlaps no tick numeral, axis name or label already placed,
* keeps clear of the curves and lines, and
* is nearer to its own item than to any other item.

Candidates walk back along the item from the end that leaves the plot, so two curves that leave through the same
edge get their labels at different heights on their own curves instead of being stacked in the margin.
"""

from __future__ import annotations

import math

Box = tuple[float, float, float, float]  # left, top, right, bottom
Pt = tuple[float, float]

STEP = 3.0  # spacing of the sample points that stand in for a curve when measuring distances
_GAP = 4.0  # distance from the item to the label box
_LABEL_H = 14.0
_OTHER_END = 100.0  # extra walk charged to the far end of an item when its label should stand at the last point


def text_box(x: float, y: float, width: float, anchor: str, height: float = _LABEL_H) -> Box:
    """The box of text anchored at ``(x, y)`` (vertically centred on ``y``)."""
    left = x - width if anchor == "end" else x - width / 2 if anchor == "middle" else x
    return (left, y - height / 2, left + width, y + height / 2)


def _seg_clip(p: Pt, q: Pt, box: Box) -> tuple[Pt, Pt] | None:
    """The part of segment pq inside ``box``, or None."""
    x0, y0, x1, y1 = box
    dx, dy = q[0] - p[0], q[1] - p[1]
    lo, hi = 0.0, 1.0
    for pp, qq in ((-dx, p[0] - x0), (dx, x1 - p[0]), (-dy, p[1] - y0), (dy, y1 - p[1])):
        if abs(pp) < 1e-12:
            if qq < 0:
                return None
            continue
        t = qq / pp
        if pp < 0:
            lo = max(lo, t)
        else:
            hi = min(hi, t)
        if lo > hi:
            return None
    return (p[0] + lo * dx, p[1] + lo * dy), (p[0] + hi * dx, p[1] + hi * dy)


def visible_samples(pieces: list[list[Pt]], box: Box) -> list[Pt]:
    """Points about ``STEP`` apart along the parts of ``pieces`` that lie inside ``box``, in drawing order.

    The last point is where the curve last leaves the box (or its end if it never does)."""
    out: list[Pt] = []
    for piece in pieces:
        for p, q in zip(piece, piece[1:]):
            seg = _seg_clip(p, q, box)
            if seg is None:
                continue
            (ax, ay), (bx, by) = seg
            n = max(1, math.ceil(math.hypot(bx - ax, by - ay) / STEP))
            for i in range(n + 1):
                t = i / n
                pt = (ax + t * (bx - ax), ay + t * (by - ay))
                if not out or math.hypot(pt[0] - out[-1][0], pt[1] - out[-1][1]) > 0.5:
                    out.append(pt)
    return out


def _dist_pt_box(pt: Pt, box: Box) -> float:
    dx = max(box[0] - pt[0], 0.0, pt[0] - box[2])
    dy = max(box[1] - pt[1], 0.0, pt[1] - box[3])
    return math.hypot(dx, dy)


_CELL = 16.0
_FAR = 14.0  # distances beyond this are all "far enough"


class SampleIndex:
    """Sample points bucketed on a grid, so a box's distance to them is cheap to find (capped at ``_FAR``)."""

    def __init__(self, groups: list[list[Pt]]) -> None:
        self.cells: dict[tuple[int, int], list[Pt]] = {}
        for g in groups:
            for p in g:
                self.cells.setdefault((int(p[0] // _CELL), int(p[1] // _CELL)), []).append(p)

    def distance(self, box: Box) -> float:
        best = _FAR
        c0, c1 = int((box[0] - _FAR) // _CELL), int((box[2] + _FAR) // _CELL)
        r0, r1 = int((box[1] - _FAR) // _CELL), int((box[3] + _FAR) // _CELL)
        for cx in range(c0, c1 + 1):
            for cy in range(r0, r1 + 1):
                for p in self.cells.get((cx, cy), ()):
                    d = _dist_pt_box(p, box)
                    if d < best:
                        best = d
        return best


def dist_to_box(samples: list[Pt], box: Box) -> float:
    return min((_dist_pt_box(p, box) for p in samples), default=1e9)


def _overlap(a: Box, b: Box, pad: float = 1.0) -> bool:
    return a[0] < b[2] + pad and b[0] < a[2] + pad and a[1] < b[3] + pad and b[1] < a[3] + pad


def _inside(a: Box, bounds: Box) -> bool:
    return a[0] >= bounds[0] and a[1] >= bounds[1] and a[2] <= bounds[2] and a[3] <= bounds[3]


# compass offsets (dx, dy) as multiples of half the box size
_COMPASS = [(1, 0), (-1, 0), (1, -1), (-1, -1), (1, 1), (-1, 1), (0, -1), (0, 1)]


def _candidate(p: Pt, d: tuple[int, int], w: float, h: float, gap: float) -> Box:
    cx = p[0] + d[0] * (w / 2 + gap)
    cy = p[1] + d[1] * (h / 2 + gap)
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def _walk_positions(own: list[Pt]) -> list[tuple[Pt, float, bool]]:
    """Points about a label height apart along ``own`` from both ends inwards, each with its walk from the
    nearer end and whether that end is the last point, nearest the last point first."""
    n = len(own)
    cum = [0.0]
    for i in range(1, n):
        cum.append(cum[-1] + math.hypot(own[i][0] - own[i - 1][0], own[i][1] - own[i - 1][1]))
    total = cum[-1]
    out: list[tuple[Pt, float, bool]] = []
    seen: set[int] = set()
    k = 0
    while k * 12.0 <= total:
        for from_last, target in ((True, total - k * 12.0), (False, k * 12.0)):
            i = _bisect(cum, target)
            if i not in seen:
                seen.add(i)
                out.append((own[i], k * 12.0, from_last))
        k += 1
    return out


def _bisect(cum: list[float], target: float) -> int:
    lo, hi = 0, len(cum) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if cum[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


def place(
    width: float,
    own: list[Pt],
    others: list[list[Pt]],
    avoid: list[list[Pt]],
    obstacles: list[Box],
    bounds: Box,
    *,
    order: list[tuple[int, int]] | None = None,
    gap: float = _GAP,
    ownership: bool = True,
    clear: list[list[Pt]] | None = None,
    same_end: bool = False,
) -> tuple[Box, bool]:
    """Choose a box of ``width`` for an item whose visible samples are ``own`` (in drawing order). ``others``
    are the samples of the other items, ``avoid`` of things to keep clear of without ownership (the axes).

    With ``ownership`` the best candidate is the one with the widest lead of its own item over every other
    item (ties to the nearest an end of the item); without it the first acceptable candidate in ``order`` wins.
    With ``same_end`` a label near the last point of its item is preferred, so that the labels of several curves
    all stand at the same end of their curves and none can be read as belonging to its neighbour.
    Returns the box and whether every rule was met."""
    h = _LABEL_H
    other_idx = SampleIndex(others)
    avoid_idx = SampleIndex(avoid)
    clear_idx = SampleIndex(clear or [])
    own_idx = SampleIndex([own])
    compass = order or _COMPASS
    spots = _walk_positions(own)
    rings = (gap,) if len(own) > 1 else (gap, gap * 2.2, gap * 3.6)  # a lone point's label may stand further off
    cands = [
        (_candidate(p, d, width, h, g), walked + (0.0 if from_last or not same_end else _OTHER_END))
        for g in rings
        for p, walked, from_last in spots
        for d in compass
    ]

    def judge(box: Box, walked: float, tier: int) -> float | None:
        """A score (higher is better) or None when the box breaks a rule at this tier."""
        if not _inside(box, bounds) or any(_overlap(box, o) for o in obstacles):
            return None
        if tier >= 3:
            return -walked
        d_other = other_idx.distance(box)
        if d_other < 2.0 or clear_idx.distance(box) < 2.0:
            return None
        if tier == 1 and avoid_idx.distance(box) < 1.5:
            return None
        lead = 10.0
        if ownership:
            d_own = own_idx.distance(box)
            lead = d_other - d_own
            if d_own < 2.0 or lead < 2.0:
                return None
        return min(lead, 10.0) - 0.03 * walked

    for tier in (1, 2, 3):
        best, best_score = None, None
        for box, walked in cands:
            sc = judge(box, walked, tier)
            if sc is None:
                continue
            if best_score is None or sc > best_score:
                best, best_score = box, sc
        if best is not None:
            return best, tier == 1
    return (cands[0][0] if cands else (0.0, 0.0, width, h)), False
