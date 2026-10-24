"""Features of three-dimensional objects (AC9M3SP01), as typed counts.

Two skills, both with whole-number answers:

- ``eyz_named_solids`` (skill ``ey-features-of-named-solids``): a ``shape_scene`` picture of everyday solids (cube,
  rectangular prism, square pyramid, cylinder, cone, sphere). Easy asks one count about one solid (flat faces,
  curved surfaces, corners or edges); medium asks the count for a few different solids together; hard shows a
  mixed group with repeats and asks the count for every solid of one name, so the learner finds the solids,
  counts how many there are and multiplies.
- ``eyz_prism_pyramid_counts`` (skill ``ey-prism-pyramid-counts``): no picture. Easy gives a prism or pyramid by its
  base (a pentagonal prism) and asks for its faces, edges or vertices; medium gives the pieces of a net and asks
  for a count of the solid it folds into; hard works backwards from one count to another (a prism has 15 edges,
  how many faces?). ``eyz_prism_pyramid_counts_exam`` (exam tier) is a net in three parts: faces, edges, then
  the vertices from Euler's formula V - E + F = 2 using the first two.

Every count comes from building the solid's faces as lists of corners and counting the corners, the edges two faces
share and the faces, then checking Euler's formula, so no formula is trusted without a second route.
"""

from __future__ import annotations

from .. import figures
from ..core import check, generator
from ..figure_kinds.scene import scene_item, shape_scene
from ._spread_kit import tagged_numbers, whole_part, uses_rng


def _a(word: str) -> str:
    """The word with its indefinite article: "an octagon", "a square"."""
    return ("an " if word[:1] in "aeiou" else "a ") + word


_TIERS = ("easy", "medium", "hard")
_PRESETS = {t: {"tier": t} for t in _TIERS}


def _tier(tier: str) -> str:
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    return tier


# =============================================================================== counting a built solid


def solid_faces(kind: str, n: int) -> list[list[int]]:
    """The faces of a prism or pyramid on an n-sided base, each a list of corner numbers.

    A prism has corners 0 to n-1 on the base and n to 2n-1 on the top, so face i of its sides joins corners i, i+1,
    n+i+1 and n+i. A pyramid has corners 0 to n-1 on the base and the apex at n.
    """
    base = list(range(n))
    if kind == "prism":
        top = [n + i for i in range(n)]
        sides = [[i, (i + 1) % n, n + (i + 1) % n, n + i] for i in range(n)]
        return [base, top, *sides]
    if kind == "pyramid":
        return [base, *[[i, (i + 1) % n, n] for i in range(n)]]
    raise ValueError(kind)


def counts_of(faces: list[list[int]]) -> tuple[int, int, int]:
    """``(vertices, edges, faces)`` counted from the faces alone; Euler's formula is checked on the way out."""
    corners = {c for face in faces for c in face}
    edges = {frozenset((f[i], f[(i + 1) % len(f)])) for f in faces for i in range(len(f))}
    v, e, f = len(corners), len(edges), len(faces)
    check(v - e + f == 2, "Euler's formula fails for a built solid")
    return v, e, f


_BASE = {
    3: "triangular",
    4: "square",
    5: "pentagonal",
    6: "hexagonal",
    7: "heptagonal",
    8: "octagonal",
    9: "nonagonal",
    10: "decagonal",
    12: "dodecagonal",
}
_PRISM_NAME = {
    3: "triangular prism",
    5: "pentagonal prism",
    6: "hexagonal prism",
    7: "heptagonal prism",
    8: "octagonal prism",
    9: "nonagonal prism",
    10: "decagonal prism",
    12: "dodecagonal prism",
}
_BASE_SHAPE = {
    3: "triangle",
    4: "rectangle",
    5: "pentagon",
    6: "hexagon",
    7: "heptagon",
    8: "octagon",
    9: "nonagon",
    10: "decagon",
    12: "dodecagon",
}


def _prism_name(n: int) -> str:
    if n == 4:
        return "rectangular prism"
    return _PRISM_NAME[n]


def _pyramid_name(n: int) -> str:
    return "square pyramid" if n == 4 else f"{_BASE[n]} pyramid"


# =============================================================================== named solids in a picture

# solid -> (flat faces, curved surfaces, edges, corners); None where a young learner is not asked.
FACTS = {
    "cube": (6, 0, 12, 8),
    "rectangular prism": (6, 0, 12, 8),
    "square pyramid": (5, 0, 8, 5),
    "cylinder": (2, 1, None, None),
    "cone": (1, 1, None, None),
    "sphere": (0, 1, None, None),
}
_POLY = ("cube", "rectangular prism", "square pyramid")
_PLURAL = {
    "cube": "cubes",
    "rectangular prism": "rectangular prisms",
    "square pyramid": "square pyramids",
    "cylinder": "cylinders",
    "cone": "cones",
    "sphere": "spheres",
}
_ONE = {"flat faces": "flat face", "edges": "edge", "corners": "corner"}
_FEATURE_INDEX = {"flat faces": 0, "curved surfaces": 1, "edges": 2, "corners": 3}


def _verify_polyhedra() -> None:
    """The picture facts agree with a solid built corner by corner."""
    for name, (flat, _curved, edges, corners) in FACTS.items():
        if name == "square pyramid":
            built = counts_of(solid_faces("pyramid", 4))
        elif name in ("cube", "rectangular prism"):
            built = counts_of(solid_faces("prism", 4))
        else:
            continue
        check(built == (corners, edges, flat), f"{name} facts disagree with the built solid")


_verify_polyhedra()


def _scene(shapes: list[str], alt: str) -> dict:
    return shape_scene(items=[scene_item(s, size="medium") for s in shapes], alt=alt, layout="row")


def _feature_wrongs(name: str, feature: str) -> list[tuple[int, str | None]]:
    flat, curved, edges, corners = FACTS[name]
    value = FACTS[name][_FEATURE_INDEX[feature]]
    out: list[tuple[int, str | None]] = []
    if feature == "flat faces":
        if curved:
            out.append((flat + curved, "eyz-counts-the-curved-surface-as-a-face"))
        if edges:
            out += [(edges, "eyz-gives-edges-for-faces"), (corners, "eyz-gives-corners-for-faces")]
        if name in ("cube", "rectangular prism"):
            out.append((3, "eyz-counts-only-the-faces-you-can-see"))
    elif feature == "curved surfaces":
        out.append((flat, "eyz-gives-flat-faces-for-curved"))
        out.append((flat + curved, "eyz-gives-flat-faces-for-curved"))
    elif feature == "edges":
        out += [
            (flat, "eyz-gives-faces-for-edges"),
            (corners, "eyz-gives-corners-for-edges"),
            (edges - 4, "eyz-misses-the-hidden-edges"),
        ]
    else:
        out += [
            (flat, "eyz-gives-faces-for-corners"),
            (edges, "eyz-gives-edges-for-corners"),
            (corners - 1, "eyz-misses-a-hidden-corner"),
        ]
    return [(v, t) for v, t in out if v != value]


def _fact_question(text: str, answer: int, cands, working, fig) -> dict:
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer,
        answer_text=str(answer),
        answer_latex=str(answer),
        answer_kind="number",
        working=working,
        distractors=tagged_numbers(answer, cands, low=0, high=80),
        figure=fig,
    )


_ALT = "everyday solids in a row"


def _solid_easy(rng) -> dict:
    name = rng.choice(tuple(FACTS))
    options = ["flat faces", "curved surfaces"]
    if FACTS[name][2] is not None:
        options += ["edges", "corners"]
        if rng.random() < 0.6:
            options.remove("curved surfaces")  # 0 or 1 curved surface: almost nothing below the key to offer
    feature = rng.choice(options)
    if feature == "flat faces" and name not in _POLY and rng.random() < 0.6:
        name = rng.choice(
            _POLY
        )  # a sphere, cone or cylinder has 0 to 2 flat faces: few wrong counts can sit below that
    answer = FACTS[name][_FEATURE_INDEX[feature]]
    fig = _scene([name], "an everyday solid")
    stem = {
        "flat faces": ("How many flat faces does this solid have?", "Count the flat faces on this solid."),
        "curved surfaces": (
            "How many curved surfaces does this solid have?",
            "Count the curved surfaces on this solid.",
        ),
        "edges": ("How many edges does this solid have?", "Count the edges on this solid."),
        "corners": ("How many corners does this solid have?", "Count the corners on this solid."),
    }[feature]
    text = rng.choice(stem)
    singular = {"flat faces": "flat face", "curved surfaces": "curved surface", "edges": "edge", "corners": "corner"}[
        feature
    ]
    working = [f"A {name} has {answer} {singular if answer == 1 else feature}."]
    return _fact_question(text, answer, _feature_wrongs(name, feature), working, fig)


def _solid_medium(rng) -> dict:
    feature = rng.choice(("flat faces", "flat faces", "edges", "corners"))
    pool = tuple(FACTS) if feature == "flat faces" else _POLY
    k = rng.randint(2, 3)
    shapes = [rng.choice(pool) for _ in range(k)]
    if len(set(shapes)) == 1:
        shapes[-1] = rng.choice([s for s in pool if s != shapes[0]])
    answer = sum(FACTS[s][_FEATURE_INDEX[feature]] for s in shapes)
    fig = _scene(shapes, _ALT)
    text = rng.choice(
        (
            f"How many {feature} do these solids have in all?",
            f"Count the {feature} of all the solids together.",
        )
    )
    per = [FACTS[s][_FEATURE_INDEX[feature]] for s in shapes]
    cands: list[tuple[int, str | None]] = [
        (answer - per[0], "eyz-leaves-out-a-solid"),
        (answer - per[-1], "eyz-leaves-out-a-solid"),
        (max(per), "eyz-counts-one-solid-only"),
        (len(shapes) * min(per), "eyz-uses-the-same-count-for-every-solid"),
    ]
    if feature == "flat faces":
        cands.append((answer + sum(FACTS[s][1] for s in shapes), "eyz-counts-the-curved-surface-as-a-face"))
    working = [" + ".join(str(v) for v in per) + f" = {answer}."]
    return _fact_question(text, answer, cands, working, fig)


def _solid_hard(rng) -> dict:
    feature = rng.choice(("flat faces", "edges", "corners"))
    pool = tuple(FACTS) if feature == "flat faces" else _POLY
    target = rng.choice(pool)
    n_target = rng.randint(2, 4)
    # A cube is also a rectangular prism, so a prism question must not show cubes.
    others = [s for s in pool if s != target and not (target == "rectangular prism" and s == "cube")]
    shapes = [target] * n_target + [rng.choice(others) for _ in range(rng.randint(2, 4))]
    rng.shuffle(shapes)
    answer = n_target * FACTS[target][_FEATURE_INDEX[feature]]
    fig = _scene(shapes, _ALT)
    text = rng.choice(
        (
            f"How many {feature} do all the {_PLURAL[target]} have together?",
            f"Count the {feature} of every one of the {_PLURAL[target]}.",
        )
    )
    each = FACTS[target][_FEATURE_INDEX[feature]]
    total_others = sum(FACTS[s][_FEATURE_INDEX[feature]] for s in shapes if s != target)
    cands: list[tuple[int, str | None]] = [
        (each, "eyz-counts-one-solid-only"),
        (answer + total_others, "eyz-counts-every-solid-in-the-picture"),
        (n_target, "eyz-counts-the-solids-not-their-features"),
        ((n_target + 1) * each, "eyz-miscounts-the-solids"),
        ((n_target - 1) * each, "eyz-miscounts-the-solids"),
    ]
    working = [
        f"There are {n_target} {_PLURAL[target]}. Each has {each} {_ONE[feature] if each == 1 else feature}.",
        f"{n_target} × {each} = {answer}.",
    ]
    return _fact_question(text, answer, cands, working, fig)


@generator("eyz_named_solids", level=3, presets=_PRESETS)
@uses_rng
def eyz_named_solids(rng, tier="easy"):
    _tier(tier)
    return {"easy": _solid_easy, "medium": _solid_medium, "hard": _solid_hard}[tier](rng)


def _read_named_solids(q) -> str:
    shapes = [it["shape"] for it in q.figure["items"]]
    text = q.text
    feature = next(f for f in _FEATURE_INDEX if f in text)
    index = _FEATURE_INDEX[feature]
    named = [s for s in FACTS if _PLURAL[s] in text]
    chosen = [s for s in shapes if s in named] if named else shapes
    return str(sum(FACTS[s][index] for s in chosen))


figures.register_reader("eyz_named_solids", _read_named_solids)


# =============================================================================== prisms and pyramids from their base


# the fewest a solid can have (a tetrahedron has 4 faces, 6 edges and 4 vertices): a smaller count is no solid at all
_FEWEST = {"faces": 4, "edges": 6, "vertices": 4}


def _count_question(text: str, answer: int, cands, working, quantity: str) -> dict:
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer,
        answer_text=str(answer),
        answer_latex=str(answer),
        answer_kind="number",
        working=working,
        distractors=tagged_numbers(answer, cands, low=_FEWEST[quantity], high=60),
    )


def _wrong_formulas(kind: str, n: int, quantity: str) -> list[tuple[int, str | None]]:
    """The values other formulas give: the other solid's formula, and the base counted without the extras."""
    pv, pe, pf = counts_of(solid_faces("prism", n))
    yv, ye, yf = counts_of(solid_faces("pyramid", n))
    if quantity == "faces":
        mine = pf if kind == "prism" else yf
        return [
            (n, "eyz-counts-only-the-sides"),
            (yf if kind == "prism" else pf, "eyz-uses-the-other-solids-count"),
            (n + 1 if kind == "prism" else n + 2, "eyz-uses-the-other-solids-count"),
            (pv if kind == "prism" else yv, "eyz-gives-corners-for-faces"),
            (mine + 1, None),
        ]
    if quantity == "edges":
        return [
            (ye if kind == "prism" else pe, "eyz-uses-the-other-solids-count"),
            (2 * n if kind == "prism" else 3 * n, "eyz-uses-the-other-solids-count"),
            (n + (1 if kind == "prism" else 2), "eyz-gives-faces-for-edges"),
            (pe + n if kind == "prism" else ye + n, "eyz-counts-the-base-edges-twice"),
            (n, "eyz-counts-only-the-base-edges"),
        ]
    return [
        (yv if kind == "prism" else pv, "eyz-uses-the-other-solids-count"),
        (n, "eyz-counts-only-the-base-corners"),
        (n * 3 if kind == "prism" else 2 * n, "eyz-gives-edges-for-corners"),
        (n + 2 if kind == "prism" else n + 1, "eyz-gives-faces-for-corners"),
        (2 * n + 1 if kind == "prism" else n + 2, None),
    ]


def _prism_easy(rng) -> dict:
    n = rng.choice((3, 4, 5, 6, 8))
    quantity = rng.choice(("faces", "edges", "vertices"))
    v, e, f = counts_of(solid_faces("prism", n))
    answer = {"faces": f, "edges": e, "vertices": v}[quantity]
    shape = _BASE_SHAPE[n]
    name = _prism_name(n)
    text = rng.choice(
        (
            f"A prism has {_a(shape)} for its base. How many {quantity} does it have?",
            f"How many {quantity} does {_a(name)} have?",
        )
    )
    working = {
        "faces": [f"A prism has {n} side faces and 2 ends: {n} + 2 = {f}."],
        "edges": [f"{n} edges on the base, {n} on the top and {n} joining them: {3 * n}."],
        "vertices": [f"{n} corners on the base and {n} on the top: {2 * n}."],
    }[quantity]
    return _count_question(text, answer, _wrong_formulas("prism", n, quantity), working, quantity)


def _net_pieces(kind: str, n: int) -> str:
    base = _BASE_SHAPE[n]
    plural = base + "s" if base != "rectangle" else "rectangles"
    if kind == "prism":
        ends = f"2 {plural}" if n != 4 else "2 squares"
        sides = f"{n} rectangles" if n != 4 else "4 rectangles"
        return f"{ends} and {sides}"
    if n == 3:
        return "4 triangles"
    base_word = f"1 {base}" if n != 4 else "1 square"
    return f"{base_word} and {n} triangles"


def _prism_medium(rng) -> dict:
    kind = rng.choice(("prism", "pyramid"))
    n = rng.choice((3, 5, 6, 8)) if kind == "prism" else rng.choice((3, 4, 5, 6, 8))
    quantity = rng.choice(("faces", "edges", "vertices"))
    v, e, f = counts_of(solid_faces(kind, n))
    answer = {"faces": f, "edges": e, "vertices": v}[quantity]
    pieces = _net_pieces(kind, n)
    text = rng.choice(
        (
            f"A net has {pieces}. It folds into a solid. How many {quantity} does the solid have?",
            f"A net is made of {pieces}. How many {quantity} will the folded solid have?",
        )
    )
    working = [
        f"{pieces} make {_a(_prism_name(n) if kind == 'prism' else _pyramid_name(n))}.",
        f"It has {answer} {quantity}.",
    ]
    wrongs = _wrong_formulas(kind, n, quantity)
    if quantity == "vertices":
        wrongs.insert(0, (sum(len(fc) for fc in solid_faces(kind, n)), "eyz-counts-the-corners-of-the-flat-pieces"))
    if quantity == "faces":
        wrongs.insert(0, (n + 1 if kind == "prism" else n, "eyz-leaves-out-a-piece-of-the-net"))
    return _count_question(text, answer, wrongs, working, quantity)


def _prism_hard(rng) -> dict:
    kind = rng.choice(("prism", "pyramid"))
    n = rng.choice((5, 6, 7, 8))  # a triangular or square base gives 4 or 5 faces: too few for wrong counts below
    v, e, f = counts_of(solid_faces(kind, n))
    given, asked = rng.sample(("faces", "edges", "vertices"), 2)
    have = {"faces": f, "edges": e, "vertices": v}
    answer, value = have[asked], have[given]
    shape = "prism" if kind == "prism" else "pyramid"
    text = rng.choice(
        (
            f"A {shape} has {value} {given}. How many {asked} does it have?",
            f"A {shape} has {value} {given}. How many {asked} does that {shape} have?",
        )
    )
    other = "pyramid" if kind == "prism" else "prism"
    # Wrong base size: solving with the other solid's rule.
    wrong_n = {
        "faces": {"prism": value - 1, "pyramid": value - 2},
        "edges": {"prism": value // 2, "pyramid": value // 3},
        "vertices": {"prism": value - 1, "pyramid": value // 2},
    }[given][kind]
    other_have = {
        "faces": counts_of(solid_faces(other, wrong_n))[2] if wrong_n >= 3 else None,
        "edges": counts_of(solid_faces(other, wrong_n))[1] if wrong_n >= 3 else None,
        "vertices": counts_of(solid_faces(other, wrong_n))[0] if wrong_n >= 3 else None,
    }
    cands: list[tuple[int, str | None]] = []
    if other_have[asked] is not None:
        cands.append((other_have[asked], "eyz-uses-the-other-solids-rule"))
    cands += [
        (value, "eyz-repeats-the-given-count"),
        (n, "eyz-gives-the-base-sides-not-the-count"),
        (answer + 1, None),
        (answer - 1, None),
        (answer + 2, None),
    ]
    working = [
        f"The base has {n} sides.",
        f"{_a(_prism_name(n) if kind == 'prism' else _pyramid_name(n)).capitalize()} has {answer} {asked}.",
    ]
    return _count_question(text, answer, cands, working, asked)


@generator("eyz_prism_pyramid_counts", level=3, presets=_PRESETS)
@uses_rng
def eyz_prism_pyramid_counts(rng, tier="easy"):
    _tier(tier)
    return {"easy": _prism_easy, "medium": _prism_medium, "hard": _prism_hard}[tier](rng)


@generator("eyz_prism_pyramid_counts_exam", level=3, presets={"exam": dict(tier="exam")})
@uses_rng
def eyz_prism_pyramid_counts_exam(rng, tier="exam"):
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    kind = rng.choice(("prism", "pyramid"))
    n = rng.choice((3, 4, 5, 6, 7, 8, 9, 10, 12))
    v, e, f = counts_of(solid_faces(kind, n))
    pieces = _net_pieces(kind, n)
    k = rng.randint(5, 20)
    text = rng.choice(
        (
            f"A net is made of {pieces}. It folds into a solid.",
            f"{pieces[0].upper()}{pieces[1:]} are joined to make a net. It folds into a solid.",
        )
    )
    return dict(
        text=text,
        parts=[
            whole_part(
                "How many faces does the solid have?",
                f,
                1,
                [
                    (n, "eyz-leaves-out-a-piece-of-the-net"),
                    (n + 1 if kind == "prism" else n, "eyz-leaves-out-a-piece-of-the-net"),
                    (e, "eyz-gives-edges-for-faces"),
                    (v, "eyz-gives-corners-for-faces"),
                    (f + 2, None),
                    (f + 1, None),
                    (f - 1, None),
                ],
                lo=_FEWEST["faces"],
                working=[f"{pieces} make {f} faces."],
            ),
            whole_part(
                "How many edges does the solid have?",
                e,
                1,
                [
                    (f, "eyz-gives-faces-for-edges"),
                    (v, "eyz-gives-corners-for-edges"),
                    (e // 2 or 1, "eyz-counts-the-base-edges-twice"),
                    (e + n, "eyz-counts-the-base-edges-twice"),
                    (e + 1, None),
                    (e - 1, None),
                ],
                lo=_FEWEST["edges"],
                working=[
                    f"The faces have {sum(len(fc) for fc in solid_faces(kind, n))} sides in all. Two faces share each edge, so {sum(len(fc) for fc in solid_faces(kind, n))} ÷ 2 = {e}."
                ],
            ),
            whole_part(
                "Use V - E + F = 2 to find the vertices.",
                v,
                2,
                [
                    (f + e, "eyz-adds-instead-of-using-eulers-formula"),
                    (e - f, "eyz-forgets-the-two-in-eulers-formula"),
                    (e, "eyz-gives-edges-for-corners"),
                    (f, "eyz-gives-faces-for-corners"),
                    (f + e + 2, None),
                ],
                lo=_FEWEST["vertices"],
                follow_through="2 - part_a + part_b",
                working=[f"V = 2 + E - F = 2 + {e} - {f} = {v}."],
            ),
            whole_part(
                f"Each edge is a straw {k} cm long. How many cm of straw are used in all?",
                e * k,
                2,
                [
                    (f * k, "eyz-uses-faces-for-edges-in-the-total"),
                    (v * k, "eyz-uses-faces-for-edges-in-the-total"),
                    (e + k, "eyz-adds-the-length-instead-of-multiplying"),
                    (k, "eyz-adds-the-length-instead-of-multiplying"),
                    (e * k + k, None),
                    (e * k - k, None),
                ],
                follow_through=f"{k}*part_b",
                working=[f"{e} × {k} = {e * k} cm."],
            ),
        ],
    )
