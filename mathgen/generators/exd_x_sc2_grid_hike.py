"""Exam-tier source for `f10-grid-reference-pathway`: a hike given in kilometres on a map grid, then the way back.

A map grid has lettered columns and numbered rows; each square is a stated number of kilometres wide, and the stem
says whether row 1 is at the northern or the southern edge. A hiker starts in a stated square and walks three or
four legs in compass directions, each leg given in kilometres (so the learner must convert to squares). Three
connected parts:

(a) the finishing column, as a number (A = 1, B = 2, ...);
(b) the finishing row;
(c) the length of the shortest way back to the start along grid lines (north-south and east-west only), in
    kilometres (follow-through from (a) and (b)).

The walk is drawn square by square and stays on the grid; an independent route recomputes the finish in screen
coordinates (x to the east, y to the south).
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..answers import Distractor, exact_value
from ..core import attempts, check, generator, latex_answer, pick_distractors, stated_precision_string

_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_DIRECTIONS = {"north": (0, -1), "south": (0, 1), "east": (1, 0), "west": (-1, 0)}
_SQUARE_KM = (Fraction(1, 2), Fraction(3, 2), Fraction(2), Fraction(5, 2), Fraction(3))
_PLACES = (
    "a national park in the Grampians",
    "a conservation reserve near Margaret River",
    "a state forest in the Blue Mountains",
    "a coastal reserve on the Eyre Peninsula",
    "a national park in the Flinders Ranges",
    "a rainforest reserve near Cairns",
)


def _km(value: Fraction) -> str:
    text = stated_precision_string(value, 1)
    return text[:-2] if text.endswith(".0") else text


def _ref(col: int, row: int) -> str:
    return f"{_LETTERS[col - 1]}{row}"


@generator("exd_x_sc2_grid_hike", level=9, presets={"exam": dict(tier="exam")})
def exd_x_sc2_grid_hike(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: finishing column, finishing row, and the shortest way back along grid lines in kilometres."""
    check(tier == "exam", f"unknown tier {tier!r}")
    cols, rows = rng.randint(8, 12), rng.randint(8, 12)
    north_first = rng.random() < 0.5  # row 1 at the northern edge, else at the southern edge
    side = rng.choice(_SQUARE_KM)

    for _ in attempts(3000, "an in-bounds hike that ends away from its start on both axes"):
        sc, sr = rng.randint(2, cols - 1), rng.randint(2, rows - 1)
        col, row = sc, sr
        legs = []
        previous_axis = None
        ok = True
        for _leg in range(rng.choice((3, 4))):
            options = []
            for word, (dx, dy) in _DIRECTIONS.items():
                axis = "h" if dx else "v"
                if axis == previous_axis:
                    continue
                for count in range(1, 6):
                    drow = dy * count if north_first else -dy * count
                    if 1 <= col + dx * count <= cols and 1 <= row + drow <= rows:
                        options.append((word, count))
            if not options:
                ok = False
                break
            word, count = rng.choice(options)
            dx, dy = _DIRECTIONS[word]
            col += dx * count
            row += dy * count if north_first else -dy * count
            legs.append((word, count))
            previous_axis = "h" if dx else "v"
        if ok and col != sc and row != sr and {("h" if _DIRECTIONS[w][0] else "v") for w, _ in legs} == {"h", "v"}:
            break

    # Independent route: screen coordinates, x east and y south, then convert the row back.
    x, y = sc, (sr - 1 if north_first else rows - sr)
    for word, count in legs:
        dx, dy = _DIRECTIONS[word]
        x, y = x + dx * count, y + dy * count
    final_row = y + 1 if north_first else rows - y
    check((x, final_row) == (col, row), "finish differs between the two routes")
    dcol, drow_abs = abs(col - sc), abs(row - sr)
    way_back = (dcol + drow_abs) * side
    walked = sum(count for _, count in legs) * side
    check(dcol > 0 and drow_abs > 0, "a hike that does not change both column and row")

    row_rule = "Row 1 is along the northern edge of the map." if north_first else "Row 1 is along the southern edge."
    steps = ", then ".join(f"{_km(count * side)} km {word}" for word, count in legs)
    stem = (
        f"The map of {rng.choice(_PLACES)} is a grid with columns A to {_LETTERS[cols - 1]} from west to east and "
        f"rows 1 to {rows}. {row_rule} Each grid square is {_km(side)} km wide. A hiker starts in square "
        f"{_ref(sc, sr)} and walks {steps}."
    )

    way_text = stated_precision_string(way_back, 1)

    def _dis_num(n: int, tag: str | None) -> Distractor:
        return Distractor(str(n), tag, Fraction(n))

    def _dis_km(v: Fraction, tag: str | None) -> Distractor:
        return Distractor(stated_precision_string(v, 1), tag, exact_value(v))

    # Column and row slips: a direction reversed, a number read from the wrong axis, a letter numbered from zero.
    a_cands = [
        _dis_num(2 * sc - col, "horizontal-direction-reversed"),
        _dis_num(col - 1, "letter-numbered-from-zero"),
        _dis_num(row, "swapped-column-row"),
        _dis_num(col + 1, None),
        _dis_num(col - 2, None),
        _dis_num(col + 2, None),
        _dis_num(sc, None),
    ]
    b_cands = [
        _dis_num(2 * sr - row, "vertical-direction-reversed"),
        _dis_num(col, "swapped-column-row"),
        _dis_num(row - 1 if row > 1 else row + 1, None),
        _dis_num(row + 1, None),
        _dis_num(row - 2, None),
        _dis_num(row + 2, None),
    ]
    c_cands = [
        _dis_km(Fraction(dcol + drow_abs), "squares-not-converted-to-km"),
        _dis_km(walked, "gave-distance-walked-not-way-back"),
        _dis_km(max(dcol, drow_abs) * side, "took-the-longer-side-only"),
        _dis_km((dcol + drow_abs + 1) * side, None),
    ]
    a_cands = [c for c in a_cands if 1 <= int(c.value) <= cols]  # a reply must be a column on the map
    b_cands = [c for c in b_cands if 1 <= int(c.value) <= rows]
    a_dis = pick_distractors(str(col), a_cands, 3)
    b_dis = pick_distractors(str(row), b_cands, 3)
    c_dis = pick_distractors(way_text, c_cands, 3)
    check(len(a_dis) >= 2 and len(b_dis) >= 2 and len(c_dis) >= 2, "too few distractors")

    return dict(
        text=stem,
        parts=[
            dict(
                prompt=f"In which column does the hiker finish? Give the column as a number (A = 1, B = 2, and so on).",
                answer_kind="number",
                answer=col,
                answer_text=str(col),
                answer_latex=latex_answer(str(col)),
                marks=2,
                sensible_range=(0, None),
                working=[
                    f"One square is {_km(side)} km, so the legs are "
                    + ", ".join(f"{count} squares {word}" for word, count in legs)
                    + ".",
                    f"Start in column {sc}: the east and west legs move it by {col - sc:+d}, so the column is {col}.",
                ],
                distractors=a_dis,
            ),
            dict(
                prompt="In which row does the hiker finish?",
                answer_kind="number",
                answer=row,
                answer_text=str(row),
                answer_latex=latex_answer(str(row)),
                marks=2,
                sensible_range=(0, None),
                working=[
                    f"Start in row {sr}. "
                    + (
                        "Row 1 is in the north, so moving south raises the row number."
                        if north_first
                        else "Row 1 is in the south, so moving north raises the row number."
                    ),
                    f"The finishing row is {row}.",
                ],
                distractors=b_dis,
            ),
            dict(
                prompt="The hiker returns to the start by the shortest route along grid lines (north-south and "
                "east-west only). How far does the hiker walk on the way back, in km, correct to 1 decimal place?",
                answer_kind="number",
                answer=float(way_back),
                answer_text=way_text,
                answer_latex=latex_answer(way_text),
                marks=2,
                sensible_range=(0, None),
                follow_through=f"(Abs(part_a - {sc}) + Abs(part_b - {sr}))*{side.numerator}/{side.denominator}",
                working=[
                    f"Columns apart: |{col} − {sc}| = {dcol}. Rows apart: |{row} − {sr}| = {drow_abs}.",
                    f"{dcol} + {drow_abs} = {dcol + drow_abs} squares, and {dcol + drow_abs} x {_km(side)} km = {way_text} km.",
                ],
                distractors=c_dis,
            ),
        ],
    )
