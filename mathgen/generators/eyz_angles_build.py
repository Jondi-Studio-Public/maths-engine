"""Comparing, naming and estimating the size of drawn angles (AC9M3M05, AC9M4M04), on the ``diagram`` figure.

Every angle is two straight arms from a vertex with a small arc between them. The figure never prints a size, so
the learner reads the picture. Three topics share the drawing helpers below:

- ``eyz_right_angle_compare`` (skill ``ey-compare-with-right-angle``, ``number``): five angles A to E; the learner
  counts how many are smaller than, the same as, or bigger than a right angle. Easy draws a dashed guide at a
  right angle from each angle's first arm, so each is judged on its own; medium draws one right angle R beside
  them to compare with; hard also draws angles close to a right angle (10 to 15 degrees off), straight angles and
  reflex angles.
- ``eyz_angle_kind`` (skill ``ey-name-angle-by-size``, ``set``): four angles A to D; the learner lists the letters of
  the angles of a named kind. Easy draws acute, right and obtuse angles and asks for one of those kinds; medium draws
  all five kinds (acute, right, obtuse, straight, reflex); hard asks for two kinds together or for every kind but
  one. A right angle is always drawn with the square corner, the convention a learner is taught.
- ``eyz_angle_estimate`` (skill ``ey-estimate-angle-size``, ``number``): a fan of dashed rays 30 degrees apart from a
  first arm, with the angle's second arm lying 10 or 20 degrees past one of them, so an estimate to the nearest
  10 degrees is a reading between two benchmarks and is graded exactly. Easy labels every ray, medium labels only
  the 90 and 180 degree rays, hard goes past 180 into reflex angles and labels 90, 180 and 270.

Each topic registers a figure reader that recomputes the answer from the drawn coordinates alone.
"""

from __future__ import annotations

import math

from .. import figures
from ..answers import Distractor
from ..core import attempts, generator, pick_distractors
from ..figure_kinds.space import angle_class, diagram, sweep_degrees
from ._ey2a_kit import tagged_numbers

_TIERS = ("easy", "medium", "hard")
_PRESETS = {t: {"tier": t} for t in _TIERS}
_KINDS = ("acute", "right", "obtuse", "straight", "reflex")


def _tier(tier: str) -> str:
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    return tier


def _at(centre, r: float, deg: float):
    rad = math.radians(deg)
    return [round(centre[0] + r * math.cos(rad), 3), round(centre[1] + r * math.sin(rad), 3)]


class _Drawing:
    """Points, segments and angles for a diagram, with unique point ids."""

    def __init__(self):
        self.points: list[dict] = []
        self.segments: list[dict] = []
        self.angles: list[dict] = []
        self._n = 0

    def point(self, at, label: str | None = None) -> str:
        self._n += 1
        pid = f"p{self._n}"
        entry = {"id": pid, "at": at}
        if label:
            entry["label"] = label
        self.points.append(entry)
        return pid

    def segment(self, a: str, b: str, *, dashed: bool = False) -> None:
        seg = {"from": a, "to": b}
        if dashed:
            seg["dashed"] = True
        self.segments.append(seg)

    def angle(
        self,
        centre,
        base_deg: float,
        alpha: float,
        ccw: bool,
        *,
        r: float,
        label: str | None = None,
        guide: bool = False,
        square: bool = False,
    ) -> None:
        """An angle of ``alpha`` degrees at ``centre``: a first arm along ``base_deg`` and a second arm ``alpha``
        degrees round from it, counter-clockwise when ``ccw`` and clockwise otherwise."""
        v = self.point(list(centre), label)
        b = self.point(_at(centre, r, base_deg))
        second = base_deg + (alpha if ccw else -alpha)
        a = self.point(_at(centre, r, second))
        self.segment(v, b)
        self.segment(v, a)
        spec = {"at": v, "from": b, "to": a} if ccw else {"at": v, "from": a, "to": b}
        if square:
            spec["right"] = True
        self.angles.append(spec)
        if guide:
            g = self.point(_at(centre, r + 3, base_deg + (90 if ccw else -90)))
            self.segment(v, g, dashed=True)

    def figure(self, frame, alt: str) -> dict:
        return diagram(
            alt=alt,
            points=self.points,
            segments=self.segments,
            angles=self.angles,
            frame=frame,
        )


def _sweep(fig: dict, spec: dict) -> float:
    pos = {p["id"]: p["at"] for p in fig["points"]}
    return sweep_degrees(pos[spec["at"]], pos[spec["from"]], pos[spec["to"]])


def _label_of(fig: dict, spec: dict) -> str | None:
    return next((p.get("label") for p in fig["points"] if p["id"] == spec["at"]), None)


# =============================================================================== compare with a right angle

# Cell centres for up to five angles, and the room each has (arm length 14 leaves a 6 unit margin).
_CELLS = ((22, 68), (62, 68), (102, 68), (22, 24), (62, 24), (102, 24))
_FRAME = (124, 92)
_LETTERS = "ABCD"

_SMALLER = {"easy": (20, 70), "medium": (20, 70), "hard": (20, 78)}
_BIGGER = {"easy": (110, 160), "medium": (110, 160), "hard": (102, 170)}
_ASKED = {"smaller": "smaller than", "bigger": "bigger than", "same": "the same as"}


def _relations(rng) -> list[str]:
    """Four angle relations to a right angle (``smaller``, ``same`` or ``bigger``), at least one of each, so every
    list of letters is a real choice."""
    kinds = ["smaller", "same", "bigger", rng.choice(("smaller", "bigger", "bigger", "same", "smaller"))]
    rng.shuffle(kinds)
    return kinds


def _size(rng, relation: str, tier: str) -> int:
    if relation == "same":
        return 90
    if relation == "smaller":
        return rng.choice(range(_SMALLER[tier][0], _SMALLER[tier][1] + 1, 5))
    if tier == "hard" and rng.random() < 0.3:
        return rng.choice((180, 200, 230, 260, 300))
    return rng.choice(range(_BIGGER[tier][0], _BIGGER[tier][1] + 1, 5))


def _set_text(letters) -> str:
    return "{" + ", ".join(sorted(letters)) + "}"


def _wrong_letter_sets(answer: set, all_letters: set, extras: list[tuple[set, str]]) -> list[Distractor]:
    """Wrong lists of letters, each the list a named slip reaches (``extras`` first, likeliest first), then a
    letter left out, a letter added and every letter."""
    out = list(extras)
    if len(answer) >= 2:
        for letter in sorted(answer):
            out.append((answer - {letter}, "eyz-leaves-out-an-angle"))
    for letter in sorted(all_letters - answer):
        out.append((answer | {letter}, "eyz-adds-an-extra-angle"))
    out.append((set(all_letters), "eyz-counts-every-angle"))
    seen, result = {frozenset(answer)}, []
    for letters, tag in out:
        key = frozenset(letters)
        if not letters or key in seen:
            continue
        seen.add(key)
        result.append(Distractor(_set_text(letters), tag, sorted(letters)))
    return result[:5]


def _letter_sets(sets: dict[str, set], asked: str, all_letters: set) -> list[Distractor]:
    """Wrong lists for the comparison skill: the other comparisons, the right angles taken the wrong way."""
    answer = sets[asked]
    extras: list[tuple[set, str]] = []
    for other in ("smaller", "bigger", "same"):
        if other != asked and sets[other]:
            extras.append((sets[other], "eyz-lists-the-other-comparison"))
    if asked == "bigger":
        extras.append((sets["bigger"] | sets["same"], "eyz-counts-right-angles-as-bigger"))
    elif asked == "smaller":
        extras.append((sets["smaller"] | sets["same"], "eyz-counts-right-angles-as-smaller"))
    else:
        extras.append((sets["same"] | sets["bigger"], "eyz-counts-bigger-angles-as-right"))
    return _wrong_letter_sets(answer, all_letters, extras)


def _right_compare(rng, tier: str) -> dict:
    relations = _relations(rng)
    asked = rng.choice(("smaller", "bigger", "bigger", "same"))
    cells = list(_CELLS)
    rng.shuffle(cells)
    draw = _Drawing()
    for i, relation in enumerate(relations):
        base = 0 if tier == "easy" else rng.choice(range(0, 360, 15))
        draw.angle(
            cells[i],
            base,
            _size(rng, relation, tier),
            rng.random() < 0.5,
            r=14,
            label=_LETTERS[i],
            guide=tier == "easy",
        )
    if tier != "easy":
        draw.angle(cells[4], rng.choice((0, 90, 180, 270)), 90, rng.random() < 0.5, r=14, label="R", square=True)
    alt = "four angles A to D" + (" with dashed square corners" if tier == "easy" else " and a right angle R")
    fig = draw.figure(_FRAME, alt)
    sets = {r: {_LETTERS[i] for i, rel in enumerate(relations) if rel == r} for r in ("smaller", "same", "bigger")}
    answer = sorted(sets[asked])
    word = _ASKED[asked]
    ref = "a right angle" if tier == "easy" else "the right angle R"
    text = {
        "easy": f"Which angles are {word} {ref}? Dashed lines are square corners.",
        "medium": f"Which of the angles A to D are {word} {ref}?",
        "hard": f"List the letters of the angles that are {word} {ref}.",
    }[tier]
    answer_text = _set_text(answer)
    working = [
        "Compare each angle with a right angle, a square corner.",
        f"Smaller: {_set_text(sets['smaller'])}. The same: {_set_text(sets['same'])}. Bigger: {_set_text(sets['bigger'])}.",
        f"The angles {word} a right angle are {answer_text}.",
    ]
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer,
        answer_text=answer_text,
        answer_latex=f"plain:{answer_text}",
        answer_kind="set",
        working=working,
        distractors=_letter_sets(sets, asked, set(_LETTERS)),
        figure=fig,
    )


@generator("eyz_right_angle_compare", level=3, presets=_PRESETS)
def eyz_right_angle_compare(rng, tier="easy"):
    _tier(tier)
    return _right_compare(rng, tier)


def _read_right_compare(q) -> str:
    fig = q.figure
    text = q.text
    key = "the same as" if "the same as" in text else "smaller than" if "smaller than" in text else "bigger than"
    hits = []
    for spec in fig["angles"]:
        label = _label_of(fig, spec)
        if label == "R":
            continue
        deg = _sweep(fig, spec)
        rel = "the same as" if abs(deg - 90) < 0.5 else "smaller than" if deg < 90 else "bigger than"
        if rel == key:
            hits.append(label)
    return _set_text(hits)


figures.register_reader("eyz_right_angle_compare", _read_right_compare)


# =============================================================================== name angles by their size

_SIZES = {
    "acute": (20, 70),
    "obtuse": (110, 160),
    "reflex": (200, 320),
}
_NEIGHBOURS = {
    "acute": ("right",),
    "right": ("acute", "obtuse"),
    "obtuse": ("right", "straight"),
    "straight": ("obtuse", "reflex"),
    "reflex": ("straight",),
}
# The hard tier asks for two kinds together, or for every kind but one.
_GROUPS = {
    "acute or right": ("acute", "right"),
    "obtuse or reflex": ("obtuse", "reflex"),
    "straight or reflex": ("straight", "reflex"),
    "not acute": ("right", "obtuse", "straight", "reflex"),
    "not obtuse": ("acute", "right", "straight", "reflex"),
}


def _kind_size(rng, kind: str) -> int:
    if kind == "right":
        return 90
    if kind == "straight":
        return 180
    lo, hi = _SIZES[kind]
    return rng.choice(range(lo, hi + 1, 5))


def _kind_working(kind: str) -> str:
    return {
        "acute": "An acute angle is smaller than a right angle.",
        "right": "A right angle is a square corner.",
        "obtuse": "An obtuse angle is bigger than a right angle but smaller than a straight line.",
        "straight": "A straight angle is one straight line.",
        "reflex": "A reflex angle is bigger than a straight line.",
    }[kind]


def _kind_question(rng, tier: str) -> dict:
    figure_kinds = ("acute", "right", "obtuse") if tier == "easy" else _KINDS
    if tier == "hard":
        asked_name = rng.choice(tuple(_GROUPS))
        asked = set(_GROUPS[asked_name])
    else:
        asked_name = rng.choice(figure_kinds)
        asked = {asked_name}
    for _ in attempts(500, "four angles with some but not all of the asked kind"):
        kinds = [rng.choice(figure_kinds) for _ in range(4)]
        hits = [i for i, k in enumerate(kinds) if k in asked]
        if 1 <= len(hits) <= 3 and (tier != "hard" or len(set(kinds)) >= 3):
            break
    cells = [(22, 68), (62, 68), (22, 24), (62, 24)]
    draw = _Drawing()
    for i, kind in enumerate(kinds):
        draw.angle(
            cells[i],
            rng.choice(range(0, 360, 15)),
            _kind_size(rng, kind),
            rng.random() < 0.5,
            r=14,
            label=_LETTERS[i],
            square=kind == "right",
        )
    fig = draw.figure((84, 92), "four angles A to D")
    letters = set(_LETTERS)
    answer = sorted(_LETTERS[i] for i in hits)
    sets = {k: {_LETTERS[i] for i, kk in enumerate(kinds) if kk == k} for k in _KINDS}
    text = {
        "easy": f"Which angles are {asked_name}?",
        "medium": f"Which of the angles A to D are {asked_name}?",
        "hard": f"Which angles are {asked_name}? List the letters.",
    }[tier]
    neighbour_kinds = sorted({n for k in asked for n in _NEIGHBOURS[k]} - asked)
    extras: list[tuple[set, str]] = []
    for n in neighbour_kinds:
        if sets[n]:
            extras.append((set(answer) | sets[n], "eyz-includes-the-neighbouring-kind"))
            extras.append((sets[n], "eyz-lists-the-neighbouring-kind"))
    rest = letters - set(answer)
    if rest:
        extras.append((rest, "eyz-lists-the-other-kinds"))
    answer_text = _set_text(answer)
    working = [_kind_working(k) for k in sorted(asked, key=_KINDS.index)] + [f"The angles that fit are {answer_text}."]
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer,
        answer_text=answer_text,
        answer_latex=f"plain:{answer_text}",
        answer_kind="set",
        working=working,
        distractors=_wrong_letter_sets(set(answer), letters, extras),
        figure=fig,
    )


@generator("eyz_angle_kind", level=4, presets=_PRESETS)
def eyz_angle_kind(rng, tier="easy"):
    _tier(tier)
    return _kind_question(rng, tier)


def _read_kind(q) -> str:
    fig = q.figure
    asked_name = q.text.removeprefix("Which of the angles A to D are ").removeprefix("Which angles are ")
    asked_name = asked_name.removesuffix(" List the letters.").removesuffix("?")
    asked = set(_GROUPS.get(asked_name, (asked_name,)))
    hits = [_label_of(fig, s) for s in fig["angles"] if angle_class(_sweep(fig, s)) in asked]
    return _set_text(hits)


figures.register_reader("eyz_angle_kind", _read_kind)


# =============================================================================== estimate an angle to the nearest 10

_FAN_FRAME = (112, 112)
_FAN_CENTRE = (56, 56)
_FAN_R = 34
# Which spokes carry their size, by tier.
_LABELLED = {"easy": (30, 60, 90, 120, 150, 180), "medium": (90, 180), "hard": (90, 180, 270)}


def _leaks(value: int, labelled) -> bool:
    """Would the number ``value`` appear inside a ray's label (20 inside 120)? The leak check forbids it."""
    return any(str(value) in f"{n}°" for n in labelled)


def _fan(rng, tier: str) -> tuple[dict, int, bool]:
    ccw = rng.random() < 0.5
    labelled = _LABELLED[tier]
    top = 180 if tier != "hard" else 360
    lo, hi = (10, 170) if tier != "hard" else (190, 350)
    spokes = list(range(30, top + 1, 30)) if top == 180 else list(range(30, 360, 30))
    options = [a for a in range(lo, hi + 1, 10) if a not in labelled and a not in (0, top) and not _leaks(a, labelled)]
    if tier == "easy":
        options = [a for a in options if a % 30 in (10, 20)]
    alpha = rng.choice(options)
    base = rng.choice(range(0, 360, 15))
    draw = _Drawing()
    v = draw.point(list(_FAN_CENTRE))
    b = draw.point(_at(_FAN_CENTRE, _FAN_R, base))
    draw.segment(v, b)
    direction = 1 if ccw else -1
    for s in spokes:
        if s == alpha:
            continue
        label = f"{s}°" if s in labelled else None
        e = draw.point(_at(_FAN_CENTRE, _FAN_R, base + direction * s), label)
        draw.segment(v, e, dashed=True)
    a = draw.point(_at(_FAN_CENTRE, _FAN_R, base + direction * alpha))
    draw.segment(v, a)
    draw.angles.append({"at": v, "from": b, "to": a} if ccw else {"at": v, "from": a, "to": b})
    fig = draw.figure(_FAN_FRAME, "an angle drawn against a fan of dashed rays")
    return fig, alpha, ccw


_ESTIMATE_TEXT = {
    "easy": (
        "Use the labelled rays. About how big is the angle, in degrees? Nearest 10.",
        "Read between the labelled rays. Estimate the angle in degrees, to the nearest 10.",
    ),
    "medium": (
        "Count the rays from 0. About how big is the angle, in degrees? Nearest 10.",
        "Count the dashed rays. Estimate the angle in degrees, to the nearest 10.",
    ),
    "hard": (
        "This angle is bigger than a straight line. Estimate it, to the nearest 10 degrees.",
        "The angle is past a straight line. About how big is it? Nearest 10 degrees.",
    ),
}


def _estimate_question(rng, tier: str) -> dict:
    fig, alpha, _ccw = _fan(rng, tier)
    labelled = {int(t.rstrip("°")) for t in figures.labels(fig) if t.endswith("°")}
    text = rng.choice(_ESTIMATE_TEXT[tier])
    cands: list[tuple[int, str | None]] = [
        (alpha + 10, "eyz-off-by-ten-degrees"),
        (alpha - 10, "eyz-off-by-ten-degrees"),
    ]
    if alpha < 180:
        cands.append((180 - alpha, "eyz-reads-from-the-wrong-arm"))
    else:
        cands.append((360 - alpha, "eyz-gives-the-angle-not-the-reflex"))
        cands.append((alpha - 180, "eyz-forgets-the-straight-angle"))
    cands += [
        (alpha + 20, "eyz-misjudges-how-far-past-the-ray"),
        (alpha - 20, "eyz-misjudges-how-far-past-the-ray"),
        (alpha + 30, "eyz-misjudges-how-far-past-the-ray"),
        (alpha - 30, "eyz-misjudges-how-far-past-the-ray"),
    ]
    cands = [(v, t) for v, t in cands if v not in labelled and not _leaks(v, labelled)]
    answer = alpha
    below = alpha // 30 * 30
    working = [
        f"The arm is past the {below} degree ray, by about {alpha - below} degrees.",
        f"{below} + {alpha - below} = {alpha}.",
    ]
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer,
        answer_text=str(answer),
        answer_latex=str(answer),
        answer_kind="number",
        working=working,
        distractors=tagged_numbers(answer, cands, low=5, high=355),
        figure=fig,
    )


@generator("eyz_angle_estimate", level=4, presets=_PRESETS)
def eyz_angle_estimate(rng, tier="easy"):
    _tier(tier)
    return _estimate_question(rng, tier)


def _read_estimate(q) -> str:
    fig = q.figure
    (spec,) = fig["angles"]
    return str(int(round(_sweep(fig, spec) / 10.0)) * 10)


figures.register_reader("eyz_angle_estimate", _read_estimate)
