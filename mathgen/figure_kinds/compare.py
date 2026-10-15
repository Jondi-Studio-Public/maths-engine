"""Comparing attributes by picture: lengths, a pan balance and containers.

One kind, ``compare_picture``, with three layouts. Every length, mass relation and fill level a
question depends on is carried by the spec; the renderer only lays them out, and each layout has a
pure ``read_*`` function that re-derives the relation from the spec alone, which a generator's
plausibility reader calls.

- ``lengths``: up to 5 objects drawn as bars or simple outlines, lined up at one end or deliberately
  not, optionally with a row of informal units (paper clips or blocks) laid under each object.
- ``balance``: one pan balance, or a chain of two, tipped the way the masses say. The validator
  refuses a tilt that disagrees with the stated masses.
- ``containers``: up to 4 containers with a capacity and a fill level, optionally with a row of
  small cups showing how many cups the contents were poured into.

Object names are the choices a question offers, so ``labels`` does not report them as shown text.
Nothing in a figure writes a number (a length, a mass, a capacity or a count is drawn, never
written), so a question can ask for it.
"""

from __future__ import annotations

from typing import Any

from mathgen.figures import _clean, _is_num, check, register_kind

LAYOUTS = ("lengths", "balance", "containers")
OBJECT_STYLES = ("bar", "pencil", "ribbon", "snake", "rope", "train")
UNIT_STYLES = ("paper_clip", "block")
SHAPES = ("jug", "cup", "bowl", "bottle")
FORMS = ("normal", "tall_thin", "short_wide")
TILTS = ("left_down", "right_down", "level")

MAX_OBJECTS = 5
MAX_EXTENT = 12  # unit lengths across the whole drawing
MAX_LABEL = 10
MAX_BALANCES = 2
MAX_PAN_OBJECTS = 2
MAX_BLOCKS = 8
MAX_MASS = 20
MAX_CONTAINERS = 4
MAX_CAPACITY = 12
MAX_CUPS = 8

#: Container height divided by width for each form. They are chosen so that ``capacity * ratio`` is
#: equal for (normal, c), (tall_thin, c / 2) and (short_wide, 2c): such containers stand equally
#: tall while holding different amounts, which is the trap a question can set.
FORM_RATIO = {"normal": 1.2, "tall_thin": 2.4, "short_wide": 0.6}
#: Drawn area per unit of capacity, in the renderer's 400-wide space.
AREA_PER_UNIT = 400


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _half(v: Any) -> bool:
    return _is_num(v) and abs(v * 2 - round(v * 2)) < 1e-9


def _label_ok(v: Any) -> bool:
    return isinstance(v, str) and 0 < len(v.strip()) <= MAX_LABEL


# ---------------------------------------------------------------- lengths


def compare_lengths(*, objects: list[dict], lined_up: bool, alt: str, unit: dict | None = None) -> dict:
    """Objects drawn as bars or outlines to compare lengths.

    - ``objects``: 1-5 of ``{"label", "style", "start", "length"}`` (+ ``"units"`` when ``unit`` is given),
      drawn top to bottom. ``style`` is one of ``OBJECT_STYLES``. ``start`` and ``length`` are in unit
      lengths (multiples of 0.5); an object runs from ``start`` to ``start + length`` and the whole
      drawing fits within ``MAX_EXTENT``.
    - ``lined_up``: ``True`` when every object starts at the same place (a guide line is drawn there);
      ``False`` when the objects are deliberately not lined up, so at least two starts differ.
    - ``unit``: optional ``{"name": str, "style": "paper_clip" | "block"}``. When given, every object
      carries ``units``, the whole number of units laid end to end under it, equal to its ``length``.
    """
    spec = {
        "kind": "compare_picture",
        "layout": "lengths",
        "alt": alt,
        "lined_up": bool(lined_up),
        "objects": [{k: (_clean(v) if _is_num(v) else v) for k, v in o.items()} for o in objects],
    }
    if unit:
        spec["unit"] = dict(unit)
    return check(spec)


def _validate_lengths(spec: dict) -> list[str]:
    problems = []
    objects = spec.get("objects")
    if not isinstance(objects, list) or not 1 <= len(objects) <= MAX_OBJECTS:
        return [f"objects must be a list of 1 to {MAX_OBJECTS}"]
    if not isinstance(spec.get("lined_up"), bool):
        problems.append("lined_up must be a bool")
    unit = spec.get("unit")
    if unit is not None:
        if not (
            isinstance(unit, dict)
            and isinstance(unit.get("name"), str)
            and unit["name"].strip()
            and unit.get("style") in UNIT_STYLES
        ):
            problems.append(f"unit must be {{name, style}} with style one of {UNIT_STYLES}")
    labels, starts = [], []
    for i, o in enumerate(objects):
        where = f"objects[{i}]"
        if not isinstance(o, dict):
            problems.append(f"{where} must be a dict")
            continue
        if not _label_ok(o.get("label")):
            problems.append(f"{where} needs a label of 1 to {MAX_LABEL} characters")
        labels.append(o.get("label"))
        if o.get("style") not in OBJECT_STYLES:
            problems.append(f"{where}.style must be one of {OBJECT_STYLES}")
        start, length = o.get("start"), o.get("length")
        if not (_half(start) and start >= 0):
            problems.append(f"{where}.start must be a non-negative multiple of 0.5")
            continue
        if not (_half(length) and length > 0):
            problems.append(f"{where}.length must be a positive multiple of 0.5")
            continue
        if start + length > MAX_EXTENT:
            problems.append(f"{where} reaches past {MAX_EXTENT} unit lengths")
        starts.append(start)
        if unit is not None:
            if not (_is_int(o.get("units")) and o["units"] == length):
                problems.append(f"{where}.units must be the whole number equal to its length")
        elif "units" in o:
            problems.append(f"{where} has units but the figure names no unit")
    if len(set(labels)) != len(labels):
        problems.append("object labels must be distinct")
    if isinstance(spec.get("lined_up"), bool) and len(starts) == len(objects):
        if spec["lined_up"] and len(set(starts)) != 1:
            problems.append("lined_up is true but the objects start in different places")
        if not spec["lined_up"] and len(objects) > 1 and len(set(starts)) < 2:
            problems.append("lined_up is false but every object starts in the same place")
    return problems


def read_lengths(spec: dict) -> dict[str, Any]:
    """label -> length in unit lengths, read from the spec alone (the unit count when units are drawn)."""
    return {o["label"]: (o["units"] if "units" in o else o["length"]) for o in spec["objects"]}


# ---------------------------------------------------------------- balance


def pan(*, objects: list[dict] = (), blocks: int = 0) -> dict:
    """One pan's load: ``objects`` (each ``{"label", "mass", "size"}``) and ``blocks`` identical blocks of mass 1."""
    return {"objects": [dict(o) for o in objects], "blocks": blocks}


def pan_mass(p: dict) -> int:
    return sum(o["mass"] for o in p["objects"]) + p["blocks"]


def tilt_for(left: dict, right: dict) -> str:
    """The tilt the masses force: the heavier pan goes down."""
    a, b = pan_mass(left), pan_mass(right)
    return "level" if a == b else ("left_down" if a > b else "right_down")


def balance(*, left: dict, right: dict, tilt: str | None = None) -> dict:
    """One balance: its two pans and the way it tips (derived from the masses unless given)."""
    return {"left": left, "right": right, "tilt": tilt_for(left, right) if tilt is None else tilt}


def compare_balance(*, balances: list[dict], alt: str) -> dict:
    """One pan balance, or a chain of two for ordering three objects indirectly.

    Each balance is ``{"left": pan, "right": pan, "tilt": TILTS}`` (see :func:`balance`). A pan holds up to
    two objects and up to eight blocks (a pan beside the other half of a chain holds exactly one object).
    ``size`` (1-3) is how big an object is drawn and has nothing to do with its mass. The tilt must agree
    with the masses. An object with the same label in two balances has the same mass in both.
    """
    return check({"kind": "compare_picture", "layout": "balance", "alt": alt, "balances": list(balances)})


def _validate_pan(p: Any, where: str, chained: bool) -> tuple[list[str], int]:
    if not isinstance(p, dict):
        return [f"{where} must be a dict"], 0
    problems = []
    objects, blocks = p.get("objects"), p.get("blocks")
    if not isinstance(objects, list) or len(objects) > MAX_PAN_OBJECTS:
        return [f"{where}.objects must be a list of at most {MAX_PAN_OBJECTS}"], 0
    if not (_is_int(blocks) and 0 <= blocks <= MAX_BLOCKS):
        return [f"{where}.blocks must be a whole number from 0 to {MAX_BLOCKS}"], 0
    if not objects and not blocks:
        problems.append(f"{where} is empty")
    if len(objects) == 2 and blocks:
        problems.append(f"{where} cannot hold two objects and blocks")
    if chained and (len(objects) != 1 or blocks):
        problems.append(f"{where} in a chain holds exactly one object and no blocks")
    for j, o in enumerate(objects):
        if not (isinstance(o, dict) and _label_ok(o.get("label"))):
            problems.append(f"{where}.objects[{j}] needs a label of 1 to {MAX_LABEL} characters")
        elif not (_is_int(o.get("mass")) and 1 <= o["mass"] <= MAX_MASS):
            problems.append(f"{where}.objects[{j}].mass must be a whole number from 1 to {MAX_MASS}")
        elif o.get("size") not in (1, 2, 3):
            problems.append(f"{where}.objects[{j}].size must be 1, 2 or 3")
    if problems:
        return problems, 0
    return [], pan_mass(p)


def _validate_balance(spec: dict) -> list[str]:
    balances = spec.get("balances")
    if not isinstance(balances, list) or not 1 <= len(balances) <= MAX_BALANCES:
        return [f"balances must be a list of 1 to {MAX_BALANCES}"]
    chained = len(balances) > 1
    problems, masses = [], {}
    for i, b in enumerate(balances):
        where = f"balances[{i}]"
        if not isinstance(b, dict):
            problems.append(f"{where} must be a dict")
            continue
        if b.get("tilt") not in TILTS:
            problems.append(f"{where}.tilt must be one of {TILTS}")
        got = []
        for side in ("left", "right"):
            ps, total = _validate_pan(b.get(side), f"{where}.{side}", chained)
            problems.extend(ps)
            got.append(total)
            if not ps:
                for o in b[side]["objects"]:
                    if masses.setdefault(o["label"], o["mass"]) != o["mass"]:
                        problems.append(f"{where}: {o['label']!r} has two different masses")
        if len(got) == 2 and all(got) and b.get("tilt") in TILTS:
            want = "level" if got[0] == got[1] else ("left_down" if got[0] > got[1] else "right_down")
            if b["tilt"] != want:
                problems.append(f"{where}.tilt is {b['tilt']!r} but the masses say {want!r}")
    return problems


def read_tilt(b: dict) -> str:
    """Which pan is lower, from the drawn tilt alone: ``left_down``, ``right_down`` or ``level``."""
    return b["tilt"]


# ---------------------------------------------------------------- containers


def compare_containers(*, containers: list[dict], alt: str) -> dict:
    """Containers with a capacity and a fill level.

    - ``containers``: 1-4 of ``{"label", "shape", "form", "capacity", "fill"}`` (+ ``"cups"``), drawn left to
      right. ``shape`` is one of ``SHAPES``; ``form`` one of ``FORMS`` (the proportions: tall and thin, short
      and wide). ``capacity`` is how many identical small cups the container holds, ``fill`` how many it
      holds now (0 to ``capacity``), both whole numbers. The renderer draws every container to scale: its
      area is proportional to ``capacity`` and the water to ``fill``.
    - ``cups``: optional, the number of small cups the contents were poured into, drawn as a row of full
      cups under the container. It equals ``fill``.
    """
    return check(
        {"kind": "compare_picture", "layout": "containers", "alt": alt, "containers": [dict(c) for c in containers]}
    )


def _validate_containers(spec: dict) -> list[str]:
    containers = spec.get("containers")
    if not isinstance(containers, list) or not 1 <= len(containers) <= MAX_CONTAINERS:
        return [f"containers must be a list of 1 to {MAX_CONTAINERS}"]
    problems, labels = [], []
    for i, c in enumerate(containers):
        where = f"containers[{i}]"
        if not isinstance(c, dict):
            problems.append(f"{where} must be a dict")
            continue
        if not _label_ok(c.get("label")):
            problems.append(f"{where} needs a label of 1 to {MAX_LABEL} characters")
        labels.append(c.get("label"))
        if c.get("shape") not in SHAPES:
            problems.append(f"{where}.shape must be one of {SHAPES}")
        if c.get("form") not in FORMS:
            problems.append(f"{where}.form must be one of {FORMS}")
        cap, fill = c.get("capacity"), c.get("fill")
        if not (_is_int(cap) and 1 <= cap <= MAX_CAPACITY):
            problems.append(f"{where}.capacity must be a whole number from 1 to {MAX_CAPACITY}")
        elif not (_is_int(fill) and 0 <= fill <= cap):
            problems.append(f"{where}.fill must be a whole number from 0 to its capacity")
        elif "cups" in c and not (_is_int(c["cups"]) and c["cups"] == fill and c["cups"] <= MAX_CUPS):
            problems.append(f"{where}.cups must equal fill and be at most {MAX_CUPS}")
    if len(set(labels)) != len(labels):
        problems.append("container labels must be distinct")
    return problems


def container_size(capacity: int, form: str) -> tuple[float, float]:
    """(width, height) of a container in the renderer's 400-wide space: area proportional to capacity."""
    ratio = FORM_RATIO[form]
    area = AREA_PER_UNIT * capacity
    height = (area * ratio) ** 0.5
    return area**0.5 / ratio**0.5, height


def read_capacities(spec: dict) -> dict[str, int]:
    """label -> capacity, from the spec alone."""
    return {c["label"]: c["capacity"] for c in spec["containers"]}


def read_amounts(spec: dict) -> dict[str, int]:
    """label -> how much is in the container now (the fill), from the spec alone."""
    return {c["label"]: c["fill"] for c in spec["containers"]}


# ---------------------------------------------------------------- the kind

_LAYOUT_VALIDATORS = {"lengths": _validate_lengths, "balance": _validate_balance, "containers": _validate_containers}


def _validate(spec: dict) -> list[str]:
    layout = spec.get("layout")
    if layout not in _LAYOUT_VALIDATORS:
        return [f"layout must be one of {LAYOUTS}"]
    return _LAYOUT_VALIDATORS[layout](spec)


def _texts(spec: dict) -> list[str]:
    # The names of the objects are the choices a question offers, so they are not reported as shown text.
    unit = spec.get("unit")
    return [unit["name"]] if unit else []


register_kind("compare_picture", _validate, texts=_texts)
