"""Descriptions for the space figure kinds: ``diagram``, ``tree_diagram`` and ``venn``."""

from __future__ import annotations

from mathgen.describe import count, join, num, register_describer, sentence
from mathgen.figure_kinds.space import BLANK, venn_region_keys

# ---------------------------------------------------------------- diagram


def _place(fx: float, fy: float) -> str:
    """Where a point sits within the drawn figure: ``top left``, ``bottom``, ``middle``..."""
    h = "left" if fx < 1 / 3 else "right" if fx > 2 / 3 else ""
    v = "top" if fy > 2 / 3 else "bottom" if fy < 1 / 3 else ""
    return f"{v} {h}".strip() or "middle"


def _where(name: str, fx: float, fy: float) -> str:
    return f"{name} at the {_place(fx, fy)} ({round(fx * 10)} across, {round(fy * 10)} up)"


def _describe_diagram(spec: dict) -> str:
    pts = spec["points"]
    xs = [p["at"][0] for p in pts]
    ys = [p["at"][1] for p in pts]
    x_lo, y_lo = min(xs), min(ys)
    w, h = max(xs) - x_lo, max(ys) - y_lo

    def frac(p) -> tuple[float, float]:
        return ((p["at"][0] - x_lo) / w if w else 0.5, (p["at"][1] - y_lo) / h if h else 0.5)

    def row(p) -> int:
        return 2 if frac(p)[1] > 2 / 3 else 0 if frac(p)[1] < 1 / 3 else 1

    ordered = sorted(pts, key=lambda p: (-row(p), p["at"][0], -p["at"][1]))
    name: dict[str, str] = {}
    k = 0
    for p in ordered:
        if p.get("label"):
            name[p["id"]] = p["label"]
        else:
            k += 1
            name[p["id"]] = f"unlabelled point {k}"
    by_id = {p["id"]: p for p in pts}

    out = ["A geometry diagram."]
    labelled = [p for p in ordered if p.get("label")]
    plain = [p for p in ordered if not p.get("label")]
    used = set()
    for s in spec["segments"]:
        used.update((s["from"], s["to"]))
    for poly in spec["polygons"]:
        used.update(poly["points"])
    for a in spec["angles"]:
        used.update((a["at"], a["from"], a["to"]))
    if labelled:
        out.append(sentence("Labelled points: " + join(_where(p["label"], *frac(p)) for p in labelled)))
    plain = [p for p in plain if p["id"] in used]
    if plain:
        out.append(
            sentence(
                "Corners and ends with no label (no dot is drawn there): "
                + join(_where(name[p["id"]], *frac(p)) for p in plain)
            )
        )

    if labelled or plain:
        out.append("Positions in parentheses are how far across and up the figure, each out of 10.")

    seg_of = {frozenset((s["from"], s["to"])): s for s in spec["segments"]}

    def edge(a: str, b: str, seg: dict | None) -> str:
        text = f"{name[a]} to {name[b]}"
        if not seg:
            return text
        if seg["from"] != a:
            a, b = seg["from"], seg["to"]
        bits = []
        if seg.get("label"):
            bits.append(f"labelled {seg['label']}")
        if seg.get("ticks"):
            bits.append(f"marked with {count(seg['ticks'], 'tick mark')}")
        if seg.get("arrows"):
            bits.append(
                f"marked with {count(seg['arrows'], 'arrow mark')} pointing from {name[seg['from']]} to {name[seg['to']]}"
            )
        if seg.get("dashed"):
            bits.append("dashed")
        return text + (", " + join(bits) if bits else "")

    poly_edges: set[frozenset] = set()
    for i, poly in enumerate(spec["polygons"]):
        ids = poly["points"]
        cyc = list(zip(ids, ids[1:] + ids[:1]))
        poly_edges.update(frozenset(e) for e in cyc)
        shape = (
            f"{'Shape' if len(spec['polygons']) == 1 else f'Shape {i + 1}'} with corners {join(name[p] for p in ids)}"
        )
        text = shape + ", joined in that order and back to the start" + (", lightly shaded" if poly.get("fill") else "")
        out.append(sentence(text))
        out.append(sentence("Its sides: " + "; ".join(edge(a, b, seg_of.get(frozenset((a, b)))) for a, b in cyc)))
    rest = [s for s in spec["segments"] if frozenset((s["from"], s["to"])) not in poly_edges]
    if rest:
        lead = "Other lines" if spec["polygons"] else "Lines"
        out.append(sentence(f"{lead}: " + "; ".join(edge(s["from"], s["to"], s) for s in rest)))

    for a in spec["angles"]:
        v, f, t = name[a["at"]], name[a["from"]], name[a["to"]]
        if a.get("right"):
            out.append(sentence(f"At {v} a small square is drawn in the corner between {v} to {f} and {v} to {t}"))
        else:
            text = f"At {v} an arc is drawn from the line {v} to {f}, anticlockwise to the line {v} to {t}"
            text += f", labelled {a['label']}" if a.get("label") else ", with no label"
            out.append(sentence(text))
    if spec.get("not_to_scale"):
        out.append("Not drawn to scale.")
    return " ".join(out)


# ---------------------------------------------------------------- tree


def _unknown(text: str) -> str:
    return f"an empty box ({BLANK})" if text == BLANK else text


def _describe_tree(spec: dict) -> str:
    root = spec["root"]

    def depth(n) -> int:
        kids = n.get("children")
        return 1 + depth(kids[0]) if kids else 0

    stages = depth(root)
    start = root.get("label")
    out = [
        f"A probability tree read from left to right, with {num(stages)} stages of branches.",
        sentence(f"It starts at {start}" if start else "It starts at a dot on the left"),
        "Branches are listed from top to bottom.",
    ]
    leaves: list[tuple[list[str], dict]] = []

    def walk(node, path):
        kids = node.get("children")
        if not kids:
            leaves.append((path, node))
            return
        lead = "From " + (start if start and not path else "the start") if not path else "After " + " then ".join(path)
        out.append(sentence(f"{lead}: " + "; ".join(f"{_unknown(k['branch'])} to {k['label']}" for k in kids)))
        for k in kids:
            walk(k, path + [k["label"]])

    walk(root, [])
    items = []
    for _path, leaf in leaves:
        text = leaf["outcome"]
        if leaf.get("outcome_p") is not None:
            op = leaf["outcome_p"]
            text += " and next to it P = " + (f"followed by an empty box ({BLANK})" if op == BLANK else op)
        items.append(text)
    out.append(sentence("Outcomes written at the right, top to bottom: " + "; ".join(items)))
    return " ".join(out)


# ---------------------------------------------------------------- venn


def _describe_venn(spec: dict) -> str:
    sets = spec["sets"]
    ids = [s["id"] for s in sets]
    nm = {s["id"]: s.get("name") or s["id"] for s in sets}
    n = len(ids)
    out = [f"A Venn diagram: a rectangle labelled {spec['universe']} holding {n} overlapping circles."]
    if n == 2:
        out.append(sentence(f"The left circle is labelled {nm[ids[0]]} and the right circle is labelled {nm[ids[1]]}"))
    else:
        out.append(
            sentence(
                f"The top left circle is labelled {nm[ids[0]]}, the top right circle {nm[ids[1]]} "
                f"and the lower circle {nm[ids[2]]}"
            )
        )
    for key in venn_region_keys(ids):
        text = spec["regions"][key]
        shown = f"an unknown ({BLANK})" if text == BLANK else text
        if key == "none":
            where = "Outside both circles" if n == 2 else "Outside all three circles"
        elif len(key) == n:
            where = "In both circles" if n == 2 else "In all three circles"
        elif len(key) == 1:
            where = f"In {nm[key]} only"
        else:
            other = next(i for i in ids if i not in key)
            where = f"In {join([nm[c] for c in key])} but not {nm[other]}"
        out.append(sentence(f"{where}: {shown}"))
    return " ".join(out)


register_describer("diagram", _describe_diagram)
register_describer("tree_diagram", _describe_tree)
register_describer("venn", _describe_venn)
