"""Generates `fsp_solids` and `fsp_solids_exam` questions: faces, edges and vertices of a drawn solid, and Euler's formula.

One skill: ``f10-solid-faces-edges-vertices`` (``answer: number``). The stimulus is a ``diagram`` figure: a wireframe of a
prism or pyramid with the hidden edges dashed, or a flat net. Three tiers:

- easy: how many faces does the solid have? The learner has to count the faces at the back and underneath as well as
  the ones in view; the base of a pyramid and both ends of a prism count.
- medium: how many edges (or how many vertices) does the solid have? Dashed lines are real edges, and corners at the
  back are real corners.
- hard: a net is drawn and folds into a solid. How many vertices does the solid have? Counting the corners on the net
  gives the wrong number, so the learner works out the faces and edges of the solid (each edge is shared by two faces,
  so the sides of all the faces add to twice the edges) and then uses Euler's formula, V - E + F = 2.

``fsp_solids_exam`` is the exam chain on a drawn net: (a) faces, (b) edges, (c) vertices (follows through from (a)
and (b) by Euler's formula), (d) the total length of straw for a model with a straw along every edge (follows through
from (b)).

The counts come from the construction (a prism on an n-gon has 2n vertices, 3n edges and n + 2 faces; a pyramid has
n + 1, 2n and n + 1) and are checked by Euler's formula on the same numbers. The figure readers recount from the
drawing alone: points and segments for a wireframe, the faces and their corners for a net.
"""

from __future__ import annotations

import re

from mathgen import figures
from mathgen.answers import Distractor
from mathgen.core import attempts, check, generator, latex_answer, phrase, pick_distractors
from . import _fsp_kit as kit

_NAMES = ("Mia", "Tom", "Aiko", "Dev", "Leila", "Sam", "Jess", "Kofi", "Priya", "Marcus", "Zoe", "Ben")

# kind -> (vertices, edges, faces) for n sides on the base
_COUNTS = {
    "prism": lambda n: (2 * n, 3 * n, n + 2),
    "pyramid": lambda n: (n + 1, 2 * n, n + 1),
}
_ALT_WIRE = "A wireframe drawing of a solid, with the edges hidden at the back drawn dashed."
_ALT_NET = "A flat net made of shapes joined along their sides, with the fold lines dashed."


def _counts(kind: str, n: int) -> tuple[int, int, int]:
    v, e, f = _COUNTS[kind](n)
    check(v - e + f == 2, "Euler's formula fails for the solid")
    return v, e, f


# ------------------------------------------------------------------------------------------ drawing


def _draw_wire(rng, kind: str, n: int):
    """A readable wireframe of the solid, and the number of vertices, edges and faces in view."""
    for _ in attempts(300, "a readable drawing of the solid"):
        yaw = rng.uniform(20, 70) * rng.choice([-1, 1])
        elevation = rng.uniform(20, 36)
        if kind == "prism" and n == 4:
            l, w, h = rng.sample([3.0, 4.0, 5.0, 6.0], 3)
            verts, faces = kit.cuboid_solid(l, w, h)
        elif kind == "prism":
            verts, faces = kit.prism_solid(n, 3.0, rng.choice([4.0, 5.0, 6.0]))
        else:
            verts, faces = kit.pyramid_solid(n, 3.0, rng.choice([3.5, 4.0, 5.0]))
        fig = kit.solid_figure(verts, faces, alt=_ALT_WIRE, yaw_degrees=yaw, elevation_degrees=elevation)
        if kit.readable(fig):
            return fig
    raise AssertionError("unreachable")


def _draw_net(rng, kind: str, n: int):
    rotation, mirror = rng.randrange(4), rng.random() < 0.5
    if kind == "prism" and n == 3:
        sides = tuple(rng.sample(list(rng.choice([(3, 4, 5), (5, 5, 6), (4, 4, 4), (6, 8, 10)])), 3))
        length = rng.choice([v for v in (6, 7, 9, 11) if v not in sides])
        faces = kit.triangular_prism_net(sides, length, top_rect=rng.randrange(3), bottom_rect=rng.randrange(3))
    elif kind == "prism":
        l, w, h = rng.sample([2, 3, 4, 5, 6], 3)
        faces, _xs, _w = kit.cuboid_net(
            l, w, h, first_wide=rng.random() < 0.5, top_cell=rng.randrange(4), bottom_cell=rng.randrange(4)
        )
    else:
        faces = kit.pyramid_net(n, 4.0, 4.0 * rng.choice([1.3, 1.5, 1.7]))
    return kit.net_figure(faces, alt=_ALT_NET, rotation=rotation, mirror=mirror)[0]


# ------------------------------------------------------------------------------------------ tiers

_WIRE_SOLIDS = [("prism", n) for n in (3, 4, 5, 6)] + [("pyramid", n) for n in (3, 4, 5, 6)]
_NET_SOLIDS = [("prism", 3), ("prism", 4)] + [("pyramid", n) for n in (3, 4, 5, 6)]


def _visible_faces(fig) -> int:
    """Faces with at least one solid (not dashed) edge all round... counted from the drawing: the faces whose whole
    outline is solid."""
    return len(_solid_faces(fig))


def _solid_faces(fig):
    solid = [s for s in fig["segments"] if not s.get("dashed")]
    sub = {**fig, "segments": solid}
    return kit.read_faces(sub) if solid else []


def _hidden_vertices(fig) -> int:
    touching: dict[str, list[bool]] = {}
    for s in fig["segments"]:
        for end in (s["from"], s["to"]):
            touching.setdefault(end, []).append(bool(s.get("dashed")))
    return sum(1 for flags in touching.values() if all(flags))


def _easy(rng):
    kind, n = rng.choice(_WIRE_SOLIDS)
    fig = _draw_wire(rng, kind, n)
    v, e, f = _counts(kind, n)
    check(len(fig["points"]) == v and len(fig["segments"]) == e, "the drawing disagrees with the counts")
    in_view = _visible_faces(fig)
    candidates = [
        Distractor(str(in_view), "solid-hidden-faces-missed", in_view),
        Distractor(str(f - 1), "solid-ends-forgotten", f - 1),
        Distractor(str(e), "solid-faces-edges-vertices-mixed", e),
        Distractor(str(v), "solid-faces-edges-vertices-mixed", v),
        Distractor(str(n), "solid-ends-forgotten", n),
    ]
    text = phrase(
        rng,
        (
            "{who} draws the solid shown. The dashed lines are edges hidden at the back. How many faces does the "
            "solid have?",
            "How many faces does the solid shown have? The dashed lines show edges that are hidden from view, and "
            "{who} has drawn the solid.",
            "{who} sketches this solid with hidden edges dashed. How many faces does it have?",
            "The solid shown is drawn with its hidden edges dashed. {who} counts its faces, including the ones that "
            "cannot be seen. How many faces are there?",
        ),
        who=rng.choice(_NAMES),
    )
    working = [
        f"The base has {n} sides."
        + (
            " The solid has two matching ends and one face on each side."
            if kind == "prism"
            else " The solid has a base and one triangle on each side."
        ),
        f"Faces: {f}.",
    ]
    return fig, text, f, candidates, working


def _medium(rng):
    kind, n = rng.choice(_WIRE_SOLIDS)
    fig = _draw_wire(rng, kind, n)
    v, e, f = _counts(kind, n)
    ask_edges = rng.random() < 0.5
    dashed = sum(1 for s in fig["segments"] if s.get("dashed"))
    hidden_v = _hidden_vertices(fig)
    if ask_edges:
        answer = e
        candidates = [
            Distractor(str(e - dashed), "solid-hidden-edges-missed", e - dashed),
            Distractor(str(f), "solid-faces-edges-vertices-mixed", f),
            Distractor(str(v), "solid-faces-edges-vertices-mixed", v),
            Distractor(str(e - n), "solid-hidden-edges-missed", e - n),
        ]
        noun = "edges"
    else:
        answer = v
        candidates = [
            Distractor(str(v - hidden_v), "solid-hidden-vertices-missed", v - hidden_v),
            Distractor(str(f), "solid-faces-edges-vertices-mixed", f),
            Distractor(str(e), "solid-faces-edges-vertices-mixed", e),
            Distractor(str(n), "solid-ends-forgotten", n),
        ]
        noun = "vertices"
    text = phrase(
        rng,
        (
            "{who} draws the solid shown, with hidden edges dashed. How many {noun} does the solid have?",
            "How many {noun} does the solid shown have? Dashed lines are edges that are hidden at the back, drawn by "
            "{who}.",
            "The solid shown is drawn with its hidden edges dashed. How many {noun} does {who}'s solid have?",
            "{who} counts the {noun} of the solid shown, including the hidden ones. How many {noun} are there?",
        ),
        who=rng.choice(_NAMES),
        noun=noun,
    )
    working = [
        f"The base has {n} sides, so the solid has {v} vertices, {e} edges and {f} faces.",
        f"Check with Euler's formula: {v} - {e} + {f} = 2.",
    ]
    return fig, text, answer, candidates, working


def _net_counts(fig) -> tuple[int, int, int]:
    """(faces, edges, vertices) of the solid a drawn net folds into, from the net alone: the faces, half the sum of
    their sides, then Euler's formula."""
    faces = kit.read_faces(fig)
    total_sides = sum(len(f) for f in faces)
    check(total_sides % 2 == 0, "the sides of the faces cannot be paired into edges")
    edges = total_sides // 2
    return len(faces), edges, 2 + edges - len(faces)


def _hard(rng):
    kind, n = rng.choice(_NET_SOLIDS)
    fig = _draw_net(rng, kind, n)
    v, e, f = _counts(kind, n)
    check(_net_counts(fig) == (f, e, v), "the net does not give the solid's counts")
    candidates = [
        Distractor(str(len(fig["points"])), "solid-net-corners-counted", len(fig["points"])),
        Distractor(str(e - f), "solid-euler-sign", e - f),
        Distractor(str(e), "solid-faces-edges-vertices-mixed", e),
        Distractor(str(f), "solid-faces-edges-vertices-mixed", f),
        Distractor(str(f + e - 2), "solid-euler-sign", f + e - 2),
        Distractor(str(2 * e), "solid-edges-counted-twice", 2 * e),
    ]
    text = phrase(
        rng,
        (
            "{who} cuts out the net shown and folds it along the dashed lines into a solid. How many vertices does "
            "the solid have?",
            "The net shown folds along the dashed lines into a solid. How many vertices does {who}'s solid have?",
            "{who} folds the net shown into a solid. How many vertices (corners) does the finished solid have?",
            "How many vertices does the solid have that {who} makes by folding the net shown?",
        ),
        who=rng.choice(_NAMES),
    )
    working = [
        f"The net has {f} faces. Their sides add to {2 * e}, and each edge is shared by two faces, so there are {e} edges.",
        f"Euler's formula V - E + F = 2 gives V = 2 + {e} - {f} = {v}.",
    ]
    return fig, text, v, candidates, working


_TIERS = {"easy": _easy, "medium": _medium, "hard": _hard}


@generator(
    "fsp_solids",
    level=8,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def fsp_solids(rng, tier="easy"):
    """Faces, edges or vertices of a drawn solid, with a number answer. See the module docstring for the tiers."""
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    fig, text, answer, candidates, working = _TIERS[tier](rng)
    key = str(answer)
    distractors = pick_distractors(key, [c for c in candidates if int(c.text) > 0], 3)
    check(len(distractors) >= 2, "too few distractors")
    return dict(
        text=text,
        answer=answer,
        answer_text=key,
        answer_latex=latex_answer(key),
        answer_kind="number",
        distractors=distractors,
        working=working,
        figure=fig,
    )


# ------------------------------------------------------------------------------------------ reader


def _read_solid(question) -> str:
    """Recount from the drawing alone."""
    fig = question.figure
    text = question.text
    if "net shown" in text:
        return str(_net_counts(fig)[2])
    v, e = len(fig["points"]), len(fig["segments"])
    f = 2 - v + e  # Euler's formula, solved for the faces
    asked = re.search(r"[Hh]ow many (faces|edges|vertices)", text).group(1)
    return str({"faces": f, "edges": e, "vertices": v}[asked])


figures.register_reader("fsp_solids", _read_solid)


# ------------------------------------------------------------------------------------------ exam chain


@generator("fsp_solids_exam", level=9, presets={"exam": {}})
def fsp_solids_exam(rng):
    """A drawn net: count its faces and edges, find the vertices by Euler's formula, then the straw for a model."""
    kind, n = rng.choice(_NET_SOLIDS)
    fig = _draw_net(rng, kind, n)
    v, e, f = _counts(kind, n)
    f_net, e_net, v_net = _net_counts(fig)
    check((f_net, e_net, v_net) == (f, e, v), "the net does not give the solid's counts")
    straw = rng.choice([4, 5, 6, 8, 10])
    total = straw * e
    stem = phrase(
        rng,
        (
            "{who} cuts out the net shown and folds it along the dashed lines into a solid.",
            "The net shown folds along the dashed lines into a solid for {who}'s model.",
            "{who} is making a model of a solid. The net for it is shown, folded along the dashed lines.",
            "The diagram shows the net of {who}'s solid, which is folded along the dashed lines.",
        ),
        who=rng.choice(_NAMES),
    )
    corners = len(fig["points"])
    lines = len(fig["segments"])
    a_cands = [
        Distractor(str(f - 1), "solid-ends-forgotten", f - 1),
        Distractor(str(lines), "solid-net-lines-counted", lines),
        Distractor(str(corners), "solid-net-corners-counted", corners),
        Distractor(str(f + 1), "solid-faces-edges-vertices-mixed", f + 1),
    ]
    b_cands = [
        Distractor(str(2 * e), "solid-edges-counted-twice", 2 * e),
        Distractor(str(lines), "solid-net-lines-counted", lines),
        Distractor(str(v), "solid-faces-edges-vertices-mixed", v),
        Distractor(str(f), "solid-faces-edges-vertices-mixed", f),
    ]
    c_cands = [
        Distractor(str(e - f), "solid-euler-sign", e - f),
        Distractor(str(f + e - 2), "solid-euler-sign", f + e - 2),
        Distractor(str(corners), "solid-net-corners-counted", corners),
        Distractor(str(e), "solid-faces-edges-vertices-mixed", e),
    ]
    d_cands = [
        Distractor(str(straw * v), "solid-faces-edges-vertices-mixed", straw * v),
        Distractor(str(straw * f), "solid-faces-edges-vertices-mixed", straw * f),
        Distractor(str(2 * total), "solid-edges-counted-twice", 2 * total),
        Distractor(str(total - straw), "solid-hidden-edges-missed", total - straw),
    ]
    parts = [
        dict(
            prompt="How many faces does the solid have?",
            answer_kind="number",
            answer=f,
            answer_text=str(f),
            answer_latex=latex_answer(str(f)),
            marks=1,
            sensible_range=(0, None),
            working=[f"Count the shapes in the net: {f} faces."],
            distractors=pick_distractors(str(f), [c for c in a_cands if int(c.text) > 0], 3),
        ),
        dict(
            prompt="How many edges does the solid have?",
            answer_kind="number",
            answer=e,
            answer_text=str(e),
            answer_latex=latex_answer(str(e)),
            marks=2,
            sensible_range=(0, None),
            working=[
                f"The sides of all the faces add to {2 * e}. Each edge is shared by two faces, so there are {e} edges."
            ],
            distractors=pick_distractors(str(e), [c for c in b_cands if int(c.text) > 0], 3),
        ),
        dict(
            prompt="Use Euler's formula, V - E + F = 2, with your answers to parts (a) and (b) to find the number of "
            "vertices.",
            answer_kind="number",
            answer=v,
            answer_text=str(v),
            answer_latex=latex_answer(str(v)),
            marks=2,
            sensible_range=(0, None),
            follow_through="2 + part_b - part_a",
            working=[f"V = 2 + E - F = 2 + {e} - {f} = {v}."],
            distractors=pick_distractors(str(v), [c for c in c_cands if int(c.text) > 0], 3),
        ),
        dict(
            prompt=f"A model of the solid has a straw along every edge, and each straw is {straw} cm long. Using your "
            "answer to part (b), find the total length of straw needed, in centimetres.",
            answer_kind="number",
            answer=total,
            answer_text=str(total),
            answer_latex=latex_answer(str(total)),
            marks=1,
            sensible_range=(0, None),
            follow_through=f"{straw}*part_b",
            working=[f"{e} edges × {straw} cm = {total} cm."],
            distractors=pick_distractors(str(total), [c for c in d_cands if int(c.text) > 0], 3),
        ),
    ]
    check(all(len(p["distractors"]) >= 2 for p in parts), "too few distractors in a part")
    return dict(text=stem, figure=fig, parts=parts)
