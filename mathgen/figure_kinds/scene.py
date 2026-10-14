"""Early-years shape pictures: flat shapes and solids placed in a loose row or grid, or an everyday object.

Every kind here is a picture the front end draws and never evaluates. The spec carries every shape,
size, style and turn; the renderer decides only layout. Styles are told apart by more than colour
(plain is hollow, shaded is filled, dotted has a dotted outline), because learners may be colour blind.

- :func:`shape_scene`: up to 16 shapes (flat or solid), each with a size, a style and an optional
  turn, or one everyday object (a house, clock, ball, box, can or ice-cream) built from shapes.

The fact tables here (:data:`FACTS`, :data:`OBJECTS`) are the one place that says how many sides a
shape has, whether it is curved, and what an object is made of. Generators and their figure
readers both use them, so a question and its reader cannot disagree about a shape.
"""

from __future__ import annotations

from typing import Iterable

from mathgen.figures import check, register_kind

__all__ = [
    "DISPLAY",
    "FACTS",
    "ALSO",
    "FLAT",
    "MAX_ITEMS",
    "OBJECTS",
    "SIZES",
    "SOLIDS",
    "STYLES",
    "is_also",
    "object_part_shape",
    "scene_item",
    "shape_scene",
]

MAX_ITEMS = 16
MAX_COLS = 8
SIZES = ("small", "medium", "large")
STYLES = ("plain", "shaded", "dotted")

FLAT = (
    "circle",
    "triangle",
    "square",
    "rectangle",
    "oval",
    "hexagon",
    "pentagon",
    "rhombus",
    "kite",
    "semicircle",
)
SOLIDS = ("cube", "rectangular prism", "sphere", "cylinder", "cone", "square pyramid")
SHAPES = FLAT + SOLIDS

DISPLAY: dict[str, str] = {s: s.capitalize() for s in SHAPES}
PLURAL: dict[str, str] = {s: s + "s" for s in SHAPES}
PLURAL["rectangular prism"] = "rectangular prisms"
PLURAL["square pyramid"] = "square pyramids"

# shape -> facts a young learner is taught. ``sides`` and ``corners`` are None where the count is not
# a clean whole number (curved shapes, the semicircle's one straight edge, every solid).
# ``curved``: has a curved edge. ``straight_only``: every edge is straight. ``parallel``: has a pair of
# parallel sides.
FACTS: dict[str, dict] = {
    "circle": dict(sides=None, corners=0, curved=True, straight_only=False, parallel=False),
    "oval": dict(sides=None, corners=0, curved=True, straight_only=False, parallel=False),
    "semicircle": dict(sides=None, corners=2, curved=True, straight_only=False, parallel=False),
    "triangle": dict(sides=3, corners=3, curved=False, straight_only=True, parallel=False),
    "square": dict(sides=4, corners=4, curved=False, straight_only=True, parallel=True),
    "rectangle": dict(sides=4, corners=4, curved=False, straight_only=True, parallel=True),
    "rhombus": dict(sides=4, corners=4, curved=False, straight_only=True, parallel=True),
    "kite": dict(sides=4, corners=4, curved=False, straight_only=True, parallel=False),
    "pentagon": dict(sides=5, corners=5, curved=False, straight_only=True, parallel=False),
    "hexagon": dict(sides=6, corners=6, curved=False, straight_only=True, parallel=True),
}
for _s in SOLIDS:
    FACTS[_s] = dict(sides=None, corners=None, curved=None, straight_only=None, parallel=None)

# Inclusion: a square IS a rectangle and a rhombus, and a circle IS an oval. A question that asks for the
# larger family must not draw a member of the smaller one, and must not offer both names as options.
ALSO: dict[str, frozenset[str]] = {
    "rectangle": frozenset({"square"}),
    "rhombus": frozenset({"square"}),
    "oval": frozenset({"circle"}),
}


def is_also(shape: str, asked: str) -> bool:
    """True if a drawn ``shape`` is, by inclusion, also an ``asked`` shape (a square is also a rectangle)."""
    return shape == asked or shape in ALSO.get(asked, frozenset())


# object -> (alt, [(role, shape)]). The roles are the words a prompt uses ("the roof of this house").
OBJECTS: dict[str, dict] = {
    "house": {"alt": "a house", "parts": [("roof", "triangle"), ("walls", "square")]},
    "clock": {"alt": "a clock", "parts": [("face", "circle")]},
    "ball": {"alt": "a ball", "parts": [("ball", "sphere")]},
    "box": {"alt": "a box", "parts": [("box", "rectangular prism")]},
    "can": {"alt": "a can", "parts": [("can", "cylinder")]},
    "ice cream": {"alt": "an ice cream", "parts": [("bottom", "cone"), ("scoop", "semicircle")]},
}

# Turns that read as clearly turned (and, for a square, clearly not a diamond by accident of 90).
TURNS = tuple(range(0, 360, 15))


def scene_item(shape: str, *, size: str = "medium", style: str = "plain", turn: int = 0, label: str = "") -> dict:
    """One drawn shape: ``turn`` is degrees clockwise (flat shapes only), ``label`` 0-2 characters printed under it."""
    return {"shape": shape, "size": size, "style": style, "turn": turn, "label": label}


def shape_scene(
    *,
    items: Iterable[dict] = (),
    alt: str,
    layout: str = "row",
    cols: int | None = None,
    object: str | None = None,
) -> dict:
    """A picture of shapes, or of one everyday object built from shapes.

    Spec::

        {"kind": "shape_scene", "alt": str,
         "layout": "row" | "grid",
         "cols": int,                      # grid only, 2 to 8
         "items": [{"shape": str, "size": "small"|"medium"|"large",
                    "style": "plain"|"shaded"|"dotted",
                    "turn": int,           # degrees clockwise, a multiple of 15; solids stay 0
                    "label": str}],        # "" or 1-2 characters, printed under the shape
         "object": {"name": str,           # not drawn
                    "parts": [{"role": str, "shape": str}]} | None}   # not drawn: the drawing is the object

    Exactly one of ``items`` (1 to 16) and ``object`` is given. A row of more than 6 shapes
    wraps to two rows. Shapes sit loosely (a small deterministic wobble) in a row and neatly in a grid.
    The words of an object's name and roles, and its parts' shape names, are data for a
    figure reader and are never drawn.
    """
    item_list = [dict(i) for i in items]
    spec: dict = {"kind": "shape_scene", "alt": alt, "layout": layout, "items": item_list, "object": None}
    if cols is not None:
        spec["cols"] = cols
    if object is not None:
        spec["object"] = {
            "name": object,
            "parts": [{"role": r, "shape": s} for r, s in OBJECTS[object]["parts"]],
        }
    return check(spec)


def object_part_shape(spec: dict, role: str) -> str:
    """The shape of the named part of a scene's object, read from the drawn spec."""
    for part in spec["object"]["parts"]:
        if part["role"] == role:
            return part["shape"]
    raise KeyError(role)


def _validate_scene(spec: dict) -> list[str]:
    problems = []
    layout = spec.get("layout")
    if layout not in ("row", "grid"):
        problems.append("layout must be 'row' or 'grid'")
    cols = spec.get("cols")
    if layout == "grid":
        if not isinstance(cols, int) or isinstance(cols, bool) or not 2 <= cols <= MAX_COLS:
            problems.append(f"a grid needs cols from 2 to {MAX_COLS}")
    elif cols is not None:
        problems.append("cols belongs to a grid layout only")
    items, obj = spec.get("items"), spec.get("object")
    if not isinstance(items, list):
        return problems + ["items must be a list"]
    if obj is None:
        if not 1 <= len(items) <= MAX_ITEMS:
            problems.append(f"items must hold 1 to {MAX_ITEMS} shapes")
    else:
        if items:
            problems.append("a scene holds shapes or an object, not both")
        known = OBJECTS.get(obj.get("name")) if isinstance(obj, dict) else None
        if known is None:
            problems.append("object.name is not a known object")
        elif obj.get("parts") != [{"role": r, "shape": s} for r, s in known["parts"]]:
            problems.append("object.parts do not match the object")
    for i, it in enumerate(items):
        if not isinstance(it, dict) or set(it) != {"shape", "size", "style", "turn", "label"}:
            problems.append(f"items[{i}] must have exactly shape, size, style, turn and label")
            continue
        if it["shape"] not in SHAPES:
            problems.append(f"items[{i}].shape {it['shape']!r} is unknown")
        if it["size"] not in SIZES:
            problems.append(f"items[{i}].size must be one of {SIZES}")
        if it["style"] not in STYLES:
            problems.append(f"items[{i}].style must be one of {STYLES}")
        turn = it["turn"]
        if not isinstance(turn, int) or isinstance(turn, bool) or turn not in TURNS:
            problems.append("items[%d].turn must be a multiple of 15 from 0 to 345" % i)
        elif turn and it["shape"] in SOLIDS:
            problems.append(f"items[{i}] is a solid and cannot be turned")
        if not isinstance(it["label"], str) or len(it["label"]) > 2:
            problems.append(f"items[{i}].label must be a string of at most 2 characters")
    return problems


def _scene_texts(spec: dict) -> list[str]:
    return [it["label"] for it in spec["items"]]


register_kind("shape_scene", _validate_scene, texts=_scene_texts)
