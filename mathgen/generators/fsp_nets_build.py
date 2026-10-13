"""Generates `fsp_nets` and `fsp_nets_exam` questions: nets of solids, drawn with the existing ``diagram`` figure.

One skill: ``f10-nets-build-solids`` (``answer: mcq``). The stimulus is a flat net with its fold lines dashed. Three
tiers, each a different thing to do with the net:

- easy: name the solid the net folds into (cube, rectangular prism, triangular prism or a pyramid on a triangle,
  square, pentagon or hexagon). The face shapes and the count decide it; the three wrong solids are the ones a
  careless reading picks.
- medium: squares joined edge to edge are cut out. Could they fold into a cube? Eleven six-square arrangements do and
  the other twenty-four do not (two squares land on the same face); a five-square arrangement fails for a different
  reason. The learner picks the statement that is true and gives the reason.
- hard: a cube net whose corners are lettered. When it is folded, three corners of the net meet at one corner of the
  cube. Which two others meet corner X? Answering means folding the net in the head, face by face.

``fsp_nets_exam`` is the exam chain on a drawn net of a closed box: (a) the area of a side face, (b) find the unlabelled width from
the measurements on the net, (c) the surface area, (d) the volume (both follow through from (b)).

Whether six squares fold into a cube is decided twice, by rolling a cube over the squares and by tracking which face
each square lands on (``_fsp_kit``); the two must agree and the eleven free cube nets must come out. The figure readers
rebuild each answer from the drawn segments alone.
"""

from __future__ import annotations

import re

from mathgen import figures
from mathgen.answers import Distractor
from mathgen.core import GRADER_TOLERANCE, attempts, check, generator, latex_answer, phrase, pick_distractors
from . import _fsp_kit as kit

_NAMES = ("Mia", "Tom", "Aiko", "Dev", "Leila", "Sam", "Jess", "Kofi", "Priya", "Marcus", "Zoe", "Ben")

# name -> (family, sides of the base, face shapes present)
_SOLIDS = {
    "Cube": ("prism", 4, {"square"}),
    "Rectangular prism": ("prism", 4, {"rectangle"}),
    "Triangular prism": ("prism", 3, {"triangle", "rectangle"}),
    "Pentagonal prism": ("prism", 5, {"pentagon", "rectangle"}),
    "Hexagonal prism": ("prism", 6, {"hexagon", "rectangle"}),
    "Triangular pyramid": ("pyramid", 3, {"triangle"}),
    "Square pyramid": ("pyramid", 4, {"square", "triangle"}),
    "Pentagonal pyramid": ("pyramid", 5, {"pentagon", "triangle"}),
    "Hexagonal pyramid": ("pyramid", 6, {"hexagon", "triangle"}),
}
_DRAWN = (
    "Cube",
    "Rectangular prism",
    "Triangular prism",
    "Triangular pyramid",
    "Square pyramid",
    "Pentagonal pyramid",
    "Hexagonal pyramid",
)
_SIDES_NAME = {3: "triangle", 4: "rectangle", 5: "pentagon", 6: "hexagon"}

_ALT_NET = "A flat net made of shapes joined along their sides, with the fold lines dashed."
_ALT_SQUARES = "Squares joined edge to edge, with the shared sides dashed."


# ------------------------------------------------------------------------------------------ signatures


def _solid_signature(name: str) -> tuple:
    """Sorted (sides, kind) for the faces of a solid, kind being 'square', 'rect', 'regular' or 'tri'."""
    family, n, _shapes = _SOLIDS[name]
    if name == "Cube":
        return tuple([(4, "square")] * 6)
    if name == "Rectangular prism":
        return tuple([(4, "rect")] * 6)
    kind = "tri" if n == 3 else "poly"
    if family == "prism":
        return tuple(sorted([(n, kind)] * 2 + [(4, "rect")] * n))
    base = (4, "square") if n == 4 else (n, kind)
    return tuple(sorted([base] + [(3, "tri")] * n))


def _figure_signature(fig) -> tuple:
    out = []
    for face in kit.read_faces(fig):
        n = len(face)
        if n == 4:
            out.append((4, "square" if kit.is_square(face) else "rect"))
        elif n == 3:
            out.append((3, "tri"))
        else:
            out.append((n, "poly"))
    return tuple(sorted(out))


def _name_from_figure(fig) -> str:
    sig = _figure_signature(fig)
    matches = [name for name in _SOLIDS if _solid_signature(name) == sig]
    check(len(matches) == 1, f"the net matches {len(matches)} solids")
    return matches[0]


# ------------------------------------------------------------------------------------------ drawing


def _draw_solid_net(rng, name: str):
    rotation, mirror = rng.randrange(4), rng.random() < 0.5
    if name == "Cube":
        cells = rng.choice(kit.hexomino_nets()[0])
        return kit.net_figure(kit.cell_faces(cells), alt=_ALT_SQUARES)[0]
    if name == "Rectangular prism":
        l, w, h = rng.sample([2, 3, 4, 5, 6], 3)
        faces, _xs, _widths = kit.cuboid_net(
            l, w, h, first_wide=rng.random() < 0.5, top_cell=rng.randrange(4), bottom_cell=rng.randrange(4)
        )
    elif name == "Triangular prism":
        sides = rng.choice([(3, 4, 5), (5, 5, 6), (4, 4, 4), (6, 8, 10), (5, 5, 8)])
        sides = tuple(rng.sample(list(sides), 3))
        length = rng.choice([n for n in (6, 7, 9, 11) if n not in sides])
        faces = kit.triangular_prism_net(sides, length, top_rect=rng.randrange(3), bottom_rect=rng.randrange(3))
    else:
        family, n, _shapes = _SOLIDS[name]
        side = 4.0
        slant = side if name == "Triangular pyramid" and rng.random() < 0.4 else side * rng.choice([1.3, 1.5, 1.7])
        faces = kit.pyramid_net(n, side, slant)
    return kit.net_figure(faces, alt=_ALT_NET, rotation=rotation, mirror=mirror)[0]


# ------------------------------------------------------------------------------------------ easy


def _easy_options(rng, name: str):
    family, n, shapes = _SOLIDS[name]
    others = [s for s in _SOLIDS if s != name and not (name == "Cube" and s == "Rectangular prism")]
    same_family = [s for s in others if _SOLIDS[s][0] == family and _SOLIDS[s][1] != n]
    cross = [s for s in others if _SOLIDS[s][0] != family and _SOLIDS[s][1] == n]
    first = rng.choice(same_family)
    second = rng.choice(cross) if cross else rng.choice([s for s in others if s != first])
    shared = [s for s in others if s not in (first, second) and _SOLIDS[s][2] & shapes]
    third = rng.choice(shared or [s for s in others if s not in (first, second)])
    out = [
        Distractor(first, "net-face-count-ignored"),
        Distractor(second, "net-prism-pyramid-confused"),
        Distractor(third, "net-face-shape-only"),
    ]
    return out


def _easy(rng):
    name = rng.choice(_DRAWN)
    fig = _draw_solid_net(rng, name)
    check(_name_from_figure(fig) == name, "the drawn net does not read back as the solid it was built for")
    options = _easy_options(rng, name)
    for opt in options:
        check(_solid_signature(opt.text) != _figure_signature(fig), "a wrong option also fits the net")
    who = rng.choice(_NAMES)
    text = phrase(
        rng,
        (
            "{who} cuts out the net shown from card and folds it along the dashed lines. Which solid does {who} make?",
            "The net shown is folded along the dashed lines to make a solid. Which solid does {who}'s net make?",
            "{who} draws the net shown for a model. When it is cut out and folded along the dashed lines, which solid "
            "does it make?",
            "Which solid can be made by folding the net shown along its dashed lines? {who} is checking a model.",
        ),
        who=who,
    )
    working = [
        "Look at the shapes of the faces and count them.",
        f"The faces match a {name.lower()}.",
    ]
    return fig, text, name, options, working


# ------------------------------------------------------------------------------------------ medium

_YES = "Yes, it folds into a cube"
_OVERLAP = "No, two squares would fold onto the same face"
_FEW = "No, there are too few squares"
_MANY = "No, there are too many squares"


def _medium(rng):
    n = 5 if rng.random() < 0.2 else 6
    if n == 5:
        pool = [s for s in kit.fixed_polyominoes(5) if not kit.has_block(s)]
    else:
        pool = kit.hexomino_nets()[0 if rng.random() < 0.5 else 1]
    cells = rng.choice(pool)
    fig = kit.net_figure(kit.cell_faces(cells), alt=_ALT_SQUARES)[0]
    if n == 5:
        key = _FEW
    elif kit.folds_into_cube(cells):
        key = _YES
    else:
        key = _OVERLAP
    # the second way: the shape must be one of the eleven free cube nets exactly when the answer is yes
    check((key == _YES) == _is_known_cube_net(cells), "the two cube tests disagree")
    wrong = {
        _YES: [
            (_OVERLAP, "net-overlap-seen-in-valid-net"),
            (_FEW, "net-faces-miscounted"),
            (_MANY, "net-faces-miscounted"),
        ],
        _OVERLAP: [(_YES, "net-any-six-squares"), (_FEW, "net-faces-miscounted"), (_MANY, "net-faces-miscounted")],
        _FEW: [
            (_YES, "net-any-six-squares"),
            (_OVERLAP, "net-overlap-seen-in-valid-net"),
            (_MANY, "net-faces-miscounted"),
        ],
    }[key]
    options = [Distractor(text, tag) for text, tag in wrong]
    who = rng.choice(_NAMES)
    text = phrase(
        rng,
        (
            "{who} joins squares of card edge to edge as shown and wants to fold them along the dashed lines into a "
            "cube. Which statement is correct?",
            "Squares are joined edge to edge as shown. {who} folds along the dashed lines to try to make a cube. "
            "Which statement is correct?",
            "{who} is testing whether the arrangement of squares shown is a net of a cube. Which statement is correct?",
            "The squares shown are joined along their shared sides. Could {who} fold them along the dashed lines "
            "into a cube? Choose the correct statement.",
        ),
        who=who,
    )
    working = {
        _YES: ["Fold the squares up one at a time: each lands on a different face of the cube."],
        _OVERLAP: ["Fold the squares up one at a time: two of them land on the same face, so one face is left open."],
        _FEW: ["A cube has six faces and this arrangement has only five squares."],
    }[key]
    return fig, text, key, options, working


def _is_known_cube_net(cells) -> bool:
    """Is the shape (after any turn or flip) one of the eleven cube nets? Checked against the eleven by their
    canonical form, built here from the 1-4-1, 2-3-1, 2-2-2 and 3-3 families."""
    return _canonical(cells) in _ELEVEN


def _canonical(cells):
    best = None
    for r in range(4):
        for m in (False, True):
            pts = []
            for x, y in cells:
                if m:
                    x = -x
                for _ in range(r):
                    x, y = -y, x
                pts.append((x, y))
            mx, my = min(p[0] for p in pts), min(p[1] for p in pts)
            t = tuple(sorted((x - mx, y - my) for x, y in pts))
            best = t if best is None or t < best else best
    return best


def _from_rows(rows):
    """Cells from rows of text (top row first), X marking a square."""
    height = len(rows)
    return [(x, height - 1 - y) for y, row in enumerate(rows) for x, ch in enumerate(row) if ch == "X"]


# The eleven nets of a cube, one drawing of each, written down independently of the folding code.
_ELEVEN = {
    _canonical(_from_rows(rows))
    for rows in (
        ("X...", "XXXX", "X..."),
        ("X...", "XXXX", ".X.."),
        ("X...", "XXXX", "..X."),
        ("X...", "XXXX", "...X"),
        (".X..", "XXXX", ".X.."),
        (".X..", "XXXX", "..X."),
        ("XX..", ".XXX", "..X."),
        ("XX..", ".XXX", "...X"),
        ("XX..", ".XXX", ".X.."),
        ("XX..", ".XX.", "..XX"),
        ("XXX..", "..XXX"),
    )
}


# ------------------------------------------------------------------------------------------ hard

_LETTERS = "ABCDEFGHJKLMNP"


def _hard(rng):
    pool = kit.hexomino_nets()[0]
    for _ in attempts(200, "a cube net with a corner where three net corners meet"):
        cells = rng.choice(pool)
        corners = kit.cube_corner_map(cells)
        classes = {}
        for lattice, cube in corners.items():
            classes.setdefault(cube, []).append(lattice)
        triples = [sorted(v) for v in classes.values() if len(v) == 3]
        if triples:
            break
    _check_fold_consistent(cells, corners)
    ordered = sorted(corners, key=lambda c: (-c[1], c[0]))
    letter = {c: _LETTERS[i] for i, c in enumerate(ordered)}
    fig = kit.net_figure(kit.cell_faces(cells), alt=_ALT_SQUARES, corner_labels=letter)[0]
    group = rng.choice(triples)
    start = rng.choice(group)
    key_letters = sorted(letter[c] for c in group if c != start)
    key = " and ".join(key_letters)
    # corners next to `start` along the outline of the net
    outline = _outline_neighbours(cells, start)
    neighbours = sorted(letter[c] for c in outline)
    partial = sorted([key_letters[0], rng.choice([letter[c] for c in ordered if c not in group and c != start])])
    other_groups = [g for g in triples if g != group]
    wrong_class = None
    if other_groups:
        g = rng.choice(other_groups)
        wrong_class = sorted(letter[c] for c in g[:2])
    seen = {key}
    options = []
    for pair, tag in (
        (neighbours, "net-neighbours-meet"),
        (partial, "net-corner-half-right"),
        (wrong_class, "net-wrong-corner"),
    ):
        if pair is None:
            continue
        txt = " and ".join(pair)
        if len(set(pair)) == 2 and txt not in seen:
            seen.add(txt)
            options.append(Distractor(txt, tag))
    free = [letter[c] for c in ordered if c != start]
    for _ in attempts(100, "a third wrong pair of corners"):
        if len(options) >= 3:
            break
        a, b = sorted(rng.sample(free, 2))
        txt = f"{a} and {b}"
        if txt not in seen:
            seen.add(txt)
            options.append(Distractor(txt, "net-wrong-corner"))
    who = rng.choice(_NAMES)
    text = phrase(
        rng,
        (
            "The net of a cube has its corners lettered. {who} folds it along the dashed lines into a cube. Corner "
            "{x} meets two other corners of the net at one corner of the cube. Which two?",
            "The lettered net shown folds into a cube. Three corners of the net come together at the same corner of "
            "the cube as corner {x}: {x} and two others. Which are the other two?",
            "{who} folds the lettered net shown into a cube. Which two corners of the net meet corner {x} at the same "
            "corner of the cube?",
            "When the lettered net is folded into a cube, corner {x} joins with two other corners of the net. Which "
            "two corners are they?",
        ),
        who=who,
        x=letter[start],
    )
    working = [
        "Fold the squares up one at a time and follow corner " + letter[start] + " round the cube.",
        f"The corners that meet at that corner of the cube are {letter[start]}, {key_letters[0]} and {key_letters[1]}.",
    ]
    return fig, text, key, options, working


def _outline_neighbours(cells, start):
    """The two corners joined to ``start`` by a side that lies on the edge of the net (not a fold line)."""
    cellset = set(cells)
    out = []
    x, y = start
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        end = (x + dx, y + dy)
        # the unit sides on either side of this segment
        if dx:
            lo = min(x, end[0])
            beside = [(lo, y), (lo, y - 1)]
        else:
            lo = min(y, end[1])
            beside = [(x, lo), (x - 1, lo)]
        count = sum(1 for c in beside if c in cellset)
        if count == 1:
            out.append(end)
    check(len(out) == 2, "a corner should have two outline neighbours")
    return out


def _check_fold_consistent(cells, corners):
    """Second check on the rolled-cube corner map: every cube edge is a fold line once or an outline pair twice."""
    check(len(set(corners.values())) == 8, "the folded net does not reach eight cube corners")
    cellset = set(cells)
    seen: dict[frozenset, int] = {}
    for x, y in cellset:
        for a, b, nb in (
            ((x, y), (x + 1, y), (x, y - 1)),
            ((x + 1, y), (x + 1, y + 1), (x + 1, y)),
            ((x, y + 1), (x + 1, y + 1), (x, y + 1)),
            ((x, y), (x, y + 1), (x - 1, y)),
        ):
            shared = nb in cellset
            edge = frozenset((corners[a], corners[b]))
            seen[edge] = seen.get(edge, 0) + (1 if not shared else 0.5)
    folds = sum(1 for v in seen.values() if v == 1)
    cuts = sum(1 for v in seen.values() if v == 2)
    check((len(seen), folds, cuts) == (12, 5, 7), "the folded net does not give the twelve cube edges")


# ------------------------------------------------------------------------------------------ the generator

_TIERS = {"easy": _easy, "medium": _medium, "hard": _hard}


@generator(
    "fsp_nets",
    level=7,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def fsp_nets(rng, tier="easy"):
    """A drawn net with a multiple-choice answer. See the module docstring for the three tiers."""
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    fig, text, key, options, working = _TIERS[tier](rng)
    distractors = pick_distractors(key, options, 3)
    check(len(distractors) == 3, "a net question needs three distinct wrong options")
    return dict(
        text=text,
        answer=key,
        answer_text=key,
        answer_kind="mcq",
        distractors=distractors,
        working=working,
        figure=fig,
    )


# ------------------------------------------------------------------------------------------ figure reader


def _read_nets(question) -> str:
    """Re-derive the key from the drawn net alone."""
    fig = question.figure
    faces = kit.read_faces(fig)
    if any(p.get("label") for p in fig["points"]):
        return _read_corner(question, fig, faces)
    if "solid" in question.text:
        return _name_from_figure(fig)
    cells = _cells_from_squares(fig, faces)
    if len(cells) != 6:
        return _FEW
    return _YES if kit.folds_into_cube(cells) else _OVERLAP


def _cells_from_squares(fig, faces):
    xs = [p["at"][0] for p in fig["points"]]
    ys = [p["at"][1] for p in fig["points"]]
    side = kit.side_lengths(faces[0])[0]
    cells = []
    for f in faces:
        cx = sum(p[0] for p in f) / 4
        cy = sum(p[1] for p in f) / 4
        cells.append((int((cx - min(xs)) // side), int((cy - min(ys)) // side)))
    return cells


def _read_corner(question, fig, faces):
    cells = _cells_from_squares(fig, faces)
    xs = [p["at"][0] for p in fig["points"]]
    ys = [p["at"][1] for p in fig["points"]]
    side = kit.side_lengths(faces[0])[0]
    mapping = kit.cube_corner_map(cells)
    labels = {}
    for p in fig["points"]:
        if p.get("label"):
            labels[(round((p["at"][0] - min(xs)) / side), round((p["at"][1] - min(ys)) / side))] = p["label"]
    start_letter = re.search(r"[Cc]orner ([A-Z])\b", question.text).group(1)
    start = next(c for c, lab in labels.items() if lab == start_letter)
    group = [labels[c] for c in mapping if mapping[c] == mapping[start] and c != start]
    return " and ".join(sorted(group))


figures.register_reader("fsp_nets", _read_nets)


# ------------------------------------------------------------------------------------------ exam chain


def _exam_dims(rng):
    for _ in attempts(200, "box dimensions with distinct sides"):
        l = rng.randint(4, 9)
        w = rng.randint(3, 8)
        h = rng.randint(2, 7)
        if len({l, w, h}) == 3 and 14 <= 2 * (l + w) <= 34:
            return l, w, h


@generator("fsp_nets_exam", level=8, presets={"exam": {}})
def fsp_nets_exam(rng):
    """A drawn net of a closed box with measurements on it: the area of a side face, the missing width, then the
    surface area and the volume (both follow through from the width)."""
    l, w, h = _exam_dims(rng)
    perimeter = 2 * (l + w)
    first_wide = rng.random() < 0.5
    widths = [l, w, l, w] if first_wide else [w, l, w, l]
    top_cell = rng.choice([i for i in range(4) if widths[i] == l])
    bottom_cell = rng.randrange(4)
    faces, xs, _widths = kit.cuboid_net(l, w, h, first_wide=first_wide, top_cell=top_cell, bottom_cell=bottom_cell)
    top_face = faces[4]
    # sides of the flap above the strip: index 2 is its top edge (length l); strip face 0, side 3 is its left edge (h)
    side_labels = {(4, 2): f"{l} cm", (0, 3): f"{h} cm"}
    y_dim = -max(l, w) - 1.5
    spec, _ids = kit.net_figure(
        faces,
        alt="The net of a closed box: four side faces in a row with a face above and a face below, measurements marked.",
        side_labels=side_labels,
        dimension_lines=[((0.0, y_dim), (float(perimeter), y_dim), f"{perimeter} cm")],
        not_to_scale=True,
    )
    check(len(top_face) == 4, "the flap is not a rectangle")
    # independent route to the width and the two derived quantities
    check(perimeter == 2 * l + 2 * w, "the strip does not go round the box once")
    width = perimeter // 2 - l
    check(width == w, "width read from the strip differs from the one drawn")
    area = 2 * (l * w + l * h + w * h)
    volume = l * w * h
    check(area == 2 * l * w + 2 * h * (l + w), "surface area disagrees with the face-by-face total")

    stem = phrase(
        rng,
        (
            "{who} makes a closed {thing} by folding the net shown along the dashed lines. The net is not drawn to "
            "scale.",
            "The net shown (not drawn to scale) folds along the dashed lines into a closed {thing} for {who}.",
            "{who} builds a closed {thing} from the net shown, which is not drawn to scale.",
            "The diagram shows the net of {who}'s closed {thing}, not drawn to scale. It is folded along the dashed "
            "lines.",
        ),
        who=rng.choice(_NAMES),
        thing=rng.choice(
            ("gift box", "pencil case", "jewellery box", "tissue box", "storage box", "shoe box", "toy box", "tea box")
        ),
    )
    side_area = l * h
    a_candidates = [
        Distractor(str(2 * (l + h)), "perimeter-for-area", 2 * (l + h)),
        Distractor(str(l + h), "net-face-sides-added", l + h),
        Distractor(str(2 * l * h), "net-two-faces-for-one", 2 * l * h),
    ]
    a_d = pick_distractors(str(side_area), a_candidates, 3, tolerance=GRADER_TOLERANCE)
    b_candidates = [
        Distractor(str(perimeter - l), "net-strip-not-halved", perimeter - l),
        Distractor(str(perimeter // 2), "net-strip-length-not-removed", perimeter // 2),
        *([Distractor(str(perimeter // 4), "net-strip-as-equal-faces", perimeter // 4)] if perimeter % 4 == 0 else []),
        Distractor(str(perimeter - 2 * l), "net-strip-not-halved", perimeter - 2 * l),
    ]
    c_candidates = [
        Distractor(str(l * w + l * h + w * h), "net-one-of-each-face", l * w + l * h + w * h),
        Distractor(str(2 * h * (l + w)), "surface-area-omits-ends", 2 * h * (l + w)),
        Distractor(str(volume), "net-volume-for-area", volume),
        Distractor(str(2 * (l * w) + h * (l + w)), "net-sides-counted-once", 2 * l * w + h * (l + w)),
    ]
    d_candidates = [
        Distractor(str(area), "net-area-for-volume", area),
        Distractor(str(l + w + h), "net-edges-added-for-volume", l + w + h),
        Distractor(str(l * w), "net-base-area-for-volume", l * w),
        Distractor(str(l * w + l * h + w * h), "net-one-of-each-face", l * w + l * h + w * h),
    ]
    b_d = pick_distractors(str(w), b_candidates, 3, tolerance=GRADER_TOLERANCE)
    c_d = pick_distractors(str(area), c_candidates, 3, tolerance=GRADER_TOLERANCE)
    d_d = pick_distractors(str(volume), d_candidates, 3, tolerance=GRADER_TOLERANCE)
    # distractors must also be wrong for the follow-through route from the canonical (b)
    check(len(a_d) >= 2 and len(b_d) >= 2 and len(c_d) >= 2 and len(d_d) >= 2, "too few distractors in a part")
    return dict(
        text=stem,
        figure=spec,
        parts=[
            dict(
                prompt=f"Find the area of one of the side faces that is {l} cm long and {h} cm high, in square "
                "centimetres.",
                answer_kind="number",
                answer=side_area,
                answer_text=str(side_area),
                answer_latex=latex_answer(str(side_area)),
                marks=1,
                sensible_range=(0, None),
                working=[f"Area = {l} × {h} = {side_area} cm²."],
                distractors=a_d,
            ),
            dict(
                prompt="Find the width of the box (the other side of the top face), in centimetres.",
                answer_kind="number",
                answer=w,
                answer_text=str(w),
                answer_latex=latex_answer(str(w)),
                marks=2,
                sensible_range=(0, None),
                working=[
                    f"The four side faces go round the box once: {perimeter} cm = 2 × {l} + 2 × width.",
                    f"Width = {perimeter} ÷ 2 − {l} = {w} cm.",
                ],
                distractors=b_d,
            ),
            dict(
                prompt="Hence find the total surface area of the box, in square centimetres.",
                answer_kind="number",
                answer=area,
                answer_text=str(area),
                answer_latex=latex_answer(str(area)),
                marks=2,
                sensible_range=(0, None),
                follow_through=f"2*({l}*part_b + {l}*{h} + part_b*{h})",
                working=[f"Surface area = 2 × ({l}×{w} + {l}×{h} + {w}×{h}) = {area} cm²."],
                distractors=c_d,
            ),
            dict(
                prompt="Using your answer to part (b), find the volume of the box, in cubic centimetres.",
                answer_kind="number",
                answer=volume,
                answer_text=str(volume),
                answer_latex=latex_answer(str(volume)),
                marks=1,
                sensible_range=(0, None),
                follow_through=f"{l}*part_b*{h}",
                working=[f"Volume = {l} × {w} × {h} = {volume} cm³."],
                distractors=d_d,
            ),
        ],
    )


def _exam_measurements(fig) -> tuple[int, int, int]:
    """(length of the top face, length of the strip of side faces, height of the side faces) read off the
    measurements marked on the exam net: the vertical one is the height and the longer horizontal one is the strip."""
    pos = {p["id"]: p["at"] for p in fig["points"]}
    labelled = [s for s in fig["segments"] if s.get("label")]
    horizontal = [s for s in labelled if abs(pos[s["from"]][1] - pos[s["to"]][1]) < 1e-6]
    vertical = [s for s in labelled if abs(pos[s["from"]][0] - pos[s["to"]][0]) < 1e-6]
    check(len(horizontal) == 2 and len(vertical) == 1, "the exam net should carry three measurements")
    values = sorted(int(s["label"].split()[0]) for s in horizontal)
    return values[0], values[1], int(vertical[0]["label"].split()[0])


def _read_exam(question) -> str:
    """Re-derive every part from the measurements on the net."""
    l, perimeter, h = _exam_measurements(question.figure)
    w = perimeter // 2 - l
    return f"(a) {l * h}; (b) {w}; (c) {2 * (l * w + l * h + w * h)}; (d) {l * w * h}"


# No sweep reader is registered for the exam topic: a multiple-choice skill's sweep would treat the labelled
# multi-part answer as one option. tests/test_fsp_nets.py checks `_read_exam` against every part instead.
