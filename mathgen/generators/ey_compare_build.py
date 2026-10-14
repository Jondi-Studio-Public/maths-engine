"""Comparing length, mass and capacity from a picture, drawn with the ``compare_picture`` figure kind.

Five topics, one per early-years skill. Every question is a picture plus a short prompt: the picture
carries the lengths, the tilt of the balance and the fill of the containers, and no figure writes a
number or says which is the answer. Each topic registers a figure reader that works from
``question.figure`` and the words of the prompt (never the key).

- ``compare_length_picture`` (skill ``compare-lengths``): which object is longer, longest or shortest.
  Easy lines two objects up at one end; medium lines up three; hard does NOT line up three to five
  objects, so the one reaching furthest right is not always the longest.
- ``compare_mass_capacity_picture`` (skill ``f10-compare-measurement-attributes``): direct comparison.
  Easy reads a balance tipped by two objects, or two containers filled to the brim; medium sets a
  trap (one object against several blocks, so more things is not always heavier; the tallest
  container is not always the one that holds most); hard is conservation of capacity (tall and thin
  against short and wide, so the higher water level is not always more water).
- ``compare_by_paper_clips`` (skill ``ey-compare-by-paper-clips``): objects NOT lined up, each measured
  with a row of informal units, so the learner counts instead of looking at where the ends are.
- ``order_by_two_balances`` (skill ``ey-order-by-two-balances``): two balances share one object, so
  three objects are ordered by mass indirectly.
- ``compare_capacity_by_cups`` (skill ``ey-compare-capacity-by-cups``): containers poured into the
  same small cups, so the cups count, not the look of the container, says which holds most.
"""

from __future__ import annotations

import itertools

from .. import figures
from ..core import attempts, check, generator, pick_distractors
from ..figure_kinds import compare as cmp

_TIERS = ("easy", "medium", "hard")
_PRESETS = {t: {"tier": t} for t in _TIERS}

_SAME = "They are the same"
_SAME_HOLD = "They hold the same"
_SAME_AMOUNT = "They have the same amount"

_LENGTH_OBJECTS = (
    ("pencil", "pencil"),
    ("ribbon", "ribbon"),
    ("snake", "snake"),
    ("rope", "rope"),
    ("train", "train"),
)
_MASS_OBJECTS = (
    "book",
    "apple",
    "shoe",
    "teddy",
    "ball",
    "hat",
    "toy car",
    "banana",
    "brick",
    "jar",
    "pear",
    "drum",
    "lunch box",
    "torch",
)
_SHAPES = ("jug", "cup", "bowl", "bottle")
_UNITS = (("paper clip", "paper_clip"), ("block", "block"))


def _the(label: str) -> str:
    return f"The {label}"


def _pair_text(a: str, b: str) -> str:
    return f"The {a} and the {b}"


def _finish(fig, text, answer_text, distractors, working, reader):
    """The question dict, after the figure, the leak check and the reader all agree with the key."""
    distractors = list(distractors)
    check(len(distractors) >= 2, "too few distractors")
    check(answer_text not in distractors, "a distractor equals the key")
    check(figures.validate(fig) == [], "the figure is invalid")
    figures.assert_no_leak(fig, [answer_text, *distractors])
    check(reader(fig, text) == answer_text, "the figure does not show the keyed answer")
    return dict(
        text=text,
        answer=answer_text,
        answer_text=answer_text,
        answer_latex=f"plain:{answer_text}",
        distractors=distractors,
        working=working,
        figure=fig,
    )


def _wants_more(text: str) -> bool:
    """True when the prompt asks for the greater of the attribute (longer, longest, heavier, holds more, most)."""
    t = text.lower()
    return any(w in t for w in ("longer", "longest", "heavier", "heaviest", "more", "most"))


def _extreme(values: dict, more: bool):
    """The label with the greatest (or least) value, or None when it is not unique."""
    best = max(values.values()) if more else min(values.values())
    hits = [k for k, v in values.items() if v == best]
    return hits[0] if len(hits) == 1 else None


# ====================================================================== lengths


_LENGTH_TEXT = {
    "pair_more": ("Which one is longer?", "Look at the two things. Which is longer?", "Which of these is longer?"),
    "pair_less": ("Which one is shorter?", "Look at the two things. Which is shorter?", "Which of these is shorter?"),
    "many_more": ("Which one is the longest?", "Which of these is the longest?", "Look at them all. Which is longest?"),
    "many_less": (
        "Which one is the shortest?",
        "Which of these is the shortest?",
        "Look at them all. Which is shortest?",
    ),
    # not lined up: the prompt says so, and a hard question reads differently from a medium one
    "loose_more": ("They are not lined up. Which is the longest?", "Not lined up. Which one is the longest?"),
    "loose_less": ("They are not lined up. Which is the shortest?", "Not lined up. Which one is the shortest?"),
}


def _length_objects(rng, n: int):
    picks = rng.sample(_LENGTH_OBJECTS, n)
    return [{"label": lbl, "style": style} for lbl, style in picks]


def _draw_lengths(rng, tier: str):
    """(objects with start and length, lined_up, more, key label or None when equal)."""
    more = rng.random() < 0.5
    for _ in attempts(3000, "a length comparison with a clear key"):
        n = {"easy": 2, "medium": 3, "hard": rng.randint(3, 5)}[tier]
        objs = _length_objects(rng, n)
        lined = tier != "hard"
        lens = [rng.randint(3, 10) for _ in range(n)]
        equal = tier == "easy" and rng.random() < 0.2
        if equal:
            lens[1] = lens[0]
        elif tier == "easy" and abs(lens[0] - lens[1]) < 2:
            continue
        elif tier != "easy" and (len(set(lens)) != n or sorted(lens)[-1] - sorted(lens)[-2] < 1):
            continue
        if not lined:
            starts = [rng.randint(0, 12 - length) for length in lens]
            if len(set(starts)) < 2:
                continue
        else:
            starts = [0] * n
        for o, s0, length in zip(objs, starts, lens):
            o["start"], o["length"] = s0, length
        if max(o["start"] + o["length"] for o in objs) > 12:
            continue
        values = {o["label"]: o["length"] for o in objs}
        key = None if equal else _extreme(values, more)
        if not equal and key is None:
            continue
        if tier == "hard":
            ends = {o["label"]: o["start"] + o["length"] for o in objs}
            # the object that reaches furthest right (or stops soonest) must not be the key
            if _extreme(ends, more) in (None, key) or len({v for v in ends.values()}) < n:
                continue
        return objs, lined, more, key
    raise AssertionError("unreachable")


def _read_length(fig: dict, text: str) -> str:
    lengths = cmp.read_lengths(fig)
    more = _wants_more(text)
    if len(lengths) == 2 and len(set(lengths.values())) == 1:
        return _SAME
    key = _extreme(lengths, more)
    check(key is not None, "no unique longest or shortest")
    return _the(key)


@generator("compare_length_picture", level=1, presets=_PRESETS)
def compare_length_picture(rng, tier="easy"):
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    objs, lined, more, key = _draw_lengths(rng, tier)
    fig = cmp.compare_lengths(
        objects=objs,
        lined_up=lined,
        alt="Objects of different lengths drawn one above the other"
        + (", all lined up at the left." if lined else ", not lined up."),
    )
    pair = len(objs) == 2
    pool = _LENGTH_TEXT[("pair_" if pair else "loose_" if not lined else "many_") + ("more" if more else "less")]
    text = rng.choice(pool)
    labels = [o["label"] for o in objs]
    if key is None:
        answer_text = _SAME
        cands = [_the(lbl) for lbl in labels]
    else:
        answer_text = _the(key)
        ends = {o["label"]: o["start"] + o["length"] for o in objs}
        far = _extreme(ends, more)
        cands = [_the(far)] if far != key else []
        cands += [_the(lbl) for lbl in labels if lbl != key]
        if pair:
            cands.append(_SAME)
    distractors = pick_distractors(answer_text, cands, n=2 if pair else len(labels) - 1)
    lens = ", ".join(f"{o['label']} {o['length']}" for o in objs)
    how = (
        "They all start at the same line, so the one that ends last is longest."
        if lined
        else "They do not start together, so compare how long each is, not where it ends."
    )
    return _finish(fig, text, answer_text, distractors, [f"{how} Lengths: {lens}."], _read_length)


figures.register_reader("compare_length_picture", lambda q: _read_length(q.figure, q.text))


# ====================================================================== mass and capacity, directly

_MASS_TEXT = {
    True: ("Which is heavier?", "Which one is heavier?", "Look at the balance. Which is heavier?"),
    False: ("Which is lighter?", "Which one is lighter?", "Look at the balance. Which is lighter?"),
}
_CAP_TEXT = {
    True: ("Which holds more?", "Which one holds more?", "Which holds the most?", "Which of these holds the most?"),
    False: ("Which holds less?", "Which one holds less?", "Which holds the least?", "Which of these holds the least?"),
}
_WATER_TEXT = ("Which has more water?", "Which one has more water in it?", "Look at the water. Which has more?")


def _sizes(rng, masses: list[int]) -> list[int]:
    """How big each object is drawn. Often the bigger-looking object is the lighter one."""
    order = sorted(range(len(masses)), key=lambda i: masses[i])
    sizes = [2] * len(masses)
    if len(masses) == 2 and masses[0] != masses[1] and rng.random() < 0.7:
        lo, hi = order
        sizes[lo], sizes[hi] = 3, 1
    else:
        sizes = [rng.randint(1, 3) for _ in masses]
    return sizes


def _mass_pair(rng):
    """(a, b, more) for a balance of one object each: distinct masses."""
    names = rng.sample(_MASS_OBJECTS, 2)
    for _ in attempts(200, "two distinct masses"):
        ma, mb = rng.randint(1, 12), rng.randint(1, 12)
        if abs(ma - mb) >= 2:
            break
    sizes = _sizes(rng, [ma, mb])
    return [{"label": n, "mass": m, "size": s} for n, m, s in zip(names, (ma, mb), sizes)]


def _object_vs_blocks(rng):
    """(object, blocks, tilt): the object balanced against several identical blocks."""
    label = rng.choice(_MASS_OBJECTS)
    level = rng.random() < 0.25
    for _ in attempts(500, "an object and a number of blocks"):
        blocks = rng.randint(2, 7)
        mass = blocks if level else rng.randint(2, 12)
        if not level and abs(mass - blocks) < 2:
            continue
        # more than half the time the object is heavier although the blocks are more things
        if not level and mass < blocks and rng.random() < 0.6:
            continue
        return {"label": label, "mass": mass, "size": rng.choice((1, 2, 3))}, blocks


def _containers_brimful(rng, caps: list[int], forms: list[str]):
    shapes = rng.sample(_SHAPES, len(caps))
    return [{"label": sh, "shape": sh, "form": fm, "capacity": c, "fill": c} for sh, fm, c in zip(shapes, forms, caps)]


def _height(c: dict) -> float:
    return cmp.container_size(c["capacity"], c["form"])[1]


def _draw_capacity(rng, tier: str):
    """Containers for a medium (three, forms mixed) or easy (two, same form) 'holds more' question."""
    more = rng.random() < 0.5
    for _ in attempts(3000, "a capacity comparison with a clear key"):
        if tier == "easy":
            caps = rng.sample(range(2, 13), 2)
            if abs(caps[0] - caps[1]) < 3:
                continue
            cs = _containers_brimful(rng, caps, ["normal", "normal"])
            return cs, more
        caps = [rng.randint(2, 12) for _ in range(3)]
        forms = [rng.choice(cmp.FORMS) for _ in range(3)]
        ordered = sorted(caps)
        near = ordered[-2] if more else ordered[1]
        best = ordered[-1] if more else ordered[0]
        if len(set(caps)) != 3 or abs(best - near) / max(best, near) < 0.25 or len(set(forms)) < 2:
            continue
        cs = _containers_brimful(rng, caps, forms)
        values = {c["label"]: c["capacity"] for c in cs}
        key = _extreme(values, more)
        heights = {c["label"]: _height(c) for c in cs}
        # the tallest (for most) or shortest (for least) container must not be the key
        if _extreme(heights, more) in (None, key) or len(set(round(h) for h in heights.values())) < 3:
            continue
        return cs, more
    raise AssertionError("unreachable")


# (tall jug capacity, short bowl capacity, jug water, bowl water): water is counted in small cups.
_CONSERVATION = {
    # the same amount of water, so the jug's water stands higher but is not more
    "same": ((4, 8, 2, 2), (4, 12, 3, 3), (6, 12, 4, 4), (6, 12, 3, 3), (8, 12, 5, 5), (8, 12, 4, 4)),
    # the water stands at the same height in both, so the wider bowl holds more of it
    "level": ((4, 8, 2, 4), (4, 8, 1, 2), (4, 8, 3, 6), (4, 12, 2, 6), (6, 12, 3, 6), (8, 12, 4, 6)),
    # the jug's water stands higher but the bowl still has more
    "taller": ((6, 12, 5, 7), (4, 12, 3, 5), (8, 12, 6, 8), (4, 8, 3, 4), (6, 12, 4, 6)),
    # the jug really does hold more water, so level and amount agree
    "jug_more": ((8, 12, 7, 3), (6, 12, 5, 2), (8, 12, 6, 3), (8, 12, 7, 4)),
}


def _draw_conservation(rng):
    """(jug capacity, bowl capacity, jug water, bowl water): a tall thin jug and a short wide bowl."""
    kind = rng.choice(("same", "same", "level", "level", "taller", "jug_more"))
    return rng.choice(_CONSERVATION[kind])


def _read_mass_capacity(fig: dict, text: str) -> str:
    more = _wants_more(text)
    if fig["layout"] == "balance":
        b = fig["balances"][0]
        tilt = cmp.read_tilt(b)
        if tilt == "level":
            return _SAME
        down, up = (b["left"], b["right"]) if tilt == "left_down" else (b["right"], b["left"])
        side = down if more else up

        def name(p):
            return _the(p["objects"][0]["label"]) if p["objects"] else "The blocks"

        return name(side)
    cs = fig["containers"]
    if any(c["fill"] < c["capacity"] for c in cs):
        amounts = {c["label"]: c["fill"] for c in cs}
        if len(set(amounts.values())) == 1:
            return _SAME_AMOUNT
        return _the(_extreme(amounts, True))
    key = _extreme(cmp.read_capacities(fig), more)
    check(key is not None, "no unique container")
    return _the(key)


@generator("compare_mass_capacity_picture", level=1, presets=_PRESETS)
def compare_mass_capacity_picture(rng, tier="easy"):
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    capacity = rng.random() < 0.5
    if tier == "hard":
        return _conservation_question(rng)
    if capacity:
        cs, more = _draw_capacity(rng, tier)
        fig = cmp.compare_containers(containers=cs, alt="Containers of different sizes, each filled to the top.")
        text = rng.choice(_CAP_TEXT[more])
        values = cmp.read_capacities(fig)
        key = _extreme(values, more)
        answer_text = _the(key)
        cands = []
        if tier == "medium":
            heights = {c["label"]: _height(c) for c in cs}
            cands.append(_the(_extreme(heights, True)))  # the tallest looks like it holds most
        cands += [_the(c["label"]) for c in cs if c["label"] != key]
        if len(cs) == 2:
            cands.append(_SAME_HOLD)
        distractors = pick_distractors(answer_text, cands, n=2)
        working = [f"{_the(key)} holds the {'most' if more else 'least'}: tall does not mean it holds more."]
        return _finish(fig, text, answer_text, distractors, working, _read_mass_capacity)
    more = rng.random() < 0.5
    if tier == "easy":
        a, b = _mass_pair(rng)
        left, right = cmp.pan(objects=[a]), cmp.pan(objects=[b])
        names = [a["label"], b["label"]]
        fig = cmp.compare_balance(
            balances=[cmp.balance(left=left, right=right)], alt="A pan balance with one object in each pan."
        )
        cands_extra = [_SAME]
    else:
        obj, blocks = _object_vs_blocks(rng)
        left, right = cmp.pan(objects=[obj]), cmp.pan(blocks=blocks)
        if rng.random() < 0.5:
            left, right = right, left
        fig = cmp.compare_balance(
            balances=[cmp.balance(left=left, right=right)],
            alt="A pan balance with an object in one pan and blocks in the other.",
        )
        names = [obj["label"]]
        cands_extra = [_SAME]
    text = rng.choice(_MASS_TEXT[more])
    answer_text = _read_mass_capacity(fig, text)
    cands = [_the(n) for n in names] + (["The blocks"] if tier == "medium" else []) + cands_extra
    # the object drawn biggest, or the side with the most things, is the tempting wrong answer
    distractors = pick_distractors(answer_text, cands, n=2)
    tilt = fig["balances"][0]["tilt"]
    working = [
        "The two sides are level, so they are the same."
        if tilt == "level"
        else f"The lower side is heavier. {answer_text} is on the {'lower' if more else 'higher'} side."
    ]
    return _finish(fig, text, answer_text, distractors, working, _read_mass_capacity)


def _conservation_question(rng):
    ta, wb, tall_fill, wide_fill = _draw_conservation(rng)
    jug = {"label": "jug", "shape": "jug", "form": "tall_thin", "capacity": ta, "fill": tall_fill}
    bowl = {"label": "bowl", "shape": "bowl", "form": "short_wide", "capacity": wb, "fill": wide_fill}
    cs = [jug, bowl] if rng.random() < 0.5 else [bowl, jug]
    fig = cmp.compare_containers(containers=cs, alt="A tall thin jug and a short wide bowl with water in each.")
    text = rng.choice(_WATER_TEXT)
    answer_text = _read_mass_capacity(fig, text)
    cands = [_the("jug"), _the("bowl"), _SAME_AMOUNT]
    # the jug's water is higher, which is the tempting but wrong reading, so it is always offered
    distractors = pick_distractors(answer_text, cands, n=2)
    working = ["Water level is not the same as amount. Look at how much space the water fills."]
    return _finish(fig, text, answer_text, distractors, working, _read_mass_capacity)


figures.register_reader("compare_mass_capacity_picture", lambda q: _read_mass_capacity(q.figure, q.text))


# ====================================================================== lengths in paper clips

_CLIP_TEXT = {
    "pair_more": (
        "Which is longer? Count the {u}s.",
        "Count the {u}s. Which one is longer?",
        "Which is longer? Use the {u}s.",
    ),
    "pair_less": (
        "Which is shorter? Count the {u}s.",
        "Count the {u}s. Which one is shorter?",
        "Which is shorter? Use the {u}s.",
    ),
    "many_more": ("Which is the longest? Count the {u}s.", "Count the {u}s. Which one is longest?"),
    "many_less": ("Which is the shortest? Count the {u}s.", "Count the {u}s. Which one is shortest?"),
    "same": (
        "Which two are the same length?",
        "Which two are as long as each other?",
        "Which two have the same length?",
    ),
}


def _draw_clips(rng, tier: str):
    """(objects with start, length and units, more, key) for easy and medium; hard returns the pair key."""
    n = {"easy": 2, "medium": 3, "hard": 4}[tier]
    more = rng.random() < 0.5
    for _ in attempts(6000, "a paper-clip comparison"):
        objs = _length_objects(rng, n)
        lens = [rng.randint(2, 8) for _ in range(n)]
        equal = tier == "easy" and rng.random() < 0.2
        if equal:
            lens[1] = lens[0]
        if tier == "hard":
            if len(set(lens)) != 3:
                continue
            dup = [v for v in lens if lens.count(v) == 2]
            if not dup:
                continue
        elif not equal and (len(set(lens)) != n or (n == 2 and abs(lens[0] - lens[1]) < 2)):
            continue
        starts = [rng.randint(0, 12 - length) for length in lens]
        if len(set(starts)) < 2:
            continue
        for o, s0, length in zip(objs, starts, lens):
            o["start"], o["length"], o["units"] = s0, length, length
        values = {o["label"]: o["length"] for o in objs}
        ends = {o["label"]: o["start"] + o["length"] for o in objs}
        if tier == "hard":
            key = tuple(o["label"] for o in objs if lens.count(o["length"]) == 2)
            # a different pair that stops at the same place, to tempt "same length" by where the ends are
            traps = [
                (a, b)
                for a, b in itertools.combinations([o["label"] for o in objs], 2)
                if ends[a] == ends[b] and values[a] != values[b]
            ]
            if not traps:
                continue
            return objs, more, key, traps[0]
        key = None if equal else _extreme(values, more)
        if not equal and key is None:
            continue
        if not equal and _extreme(ends, more) in (None, key):
            continue  # where the ends are must point at somebody else, and clearly
        return objs, more, key, None
    raise AssertionError("unreachable")


def _read_clips(fig: dict, text: str) -> str:
    units = cmp.read_lengths(fig)
    t = text.lower()
    if "same length" in t or "as long as each other" in t:
        by_len: dict[int, list[str]] = {}
        for o in fig["objects"]:
            by_len.setdefault(units[o["label"]], []).append(o["label"])
        pairs = [v for v in by_len.values() if len(v) == 2]
        check(len(pairs) == 1, "no single pair of equal lengths")
        return _pair_text(*pairs[0])
    if len(units) == 2 and len(set(units.values())) == 1:
        return _SAME
    key = _extreme(units, _wants_more(text))
    check(key is not None, "no unique longest or shortest")
    return _the(key)


@generator("compare_by_paper_clips", level=1, presets=_PRESETS)
def compare_by_paper_clips(rng, tier="easy"):
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    objs, more, key, trap = _draw_clips(rng, tier)
    uname, ustyle = rng.choice(_UNITS)
    fig = cmp.compare_lengths(
        objects=objs,
        lined_up=False,
        unit={"name": uname, "style": ustyle},
        alt=f"Objects that are not lined up, each with a row of {uname}s laid under it.",
    )
    labels = [o["label"] for o in objs]
    if tier == "hard":
        text = rng.choice(_CLIP_TEXT["same"])
        order = {lbl: i for i, lbl in enumerate(labels)}

        def pair(p):
            a, b = sorted(p, key=order.get)
            return _pair_text(a, b)

        answer_text = pair(key)
        others = [p for p in itertools.combinations(labels, 2) if set(p) != set(key) and set(p) != set(trap)]
        cands = [pair(trap), pair(rng.choice(others)), pair(rng.choice(others))]
        distractors = pick_distractors(answer_text, cands, n=2)
        working = [f"Count the {uname}s: two are the same. The ends are not at the same place for the same length."]
        return _finish(fig, text, answer_text, distractors, working, _read_clips)
    pair_q = len(objs) == 2
    text = rng.choice(_CLIP_TEXT[("pair_" if pair_q else "many_") + ("more" if more else "less")]).format(u=uname)
    if key is None:
        answer_text = _SAME
        cands = [_the(lbl) for lbl in labels]
    else:
        answer_text = _the(key)
        ends = {o["label"]: o["start"] + o["length"] for o in objs}
        cands = [_the(_extreme(ends, more))] + [_the(lbl) for lbl in labels if lbl != key]
        if pair_q:
            cands.append(_SAME)
    distractors = pick_distractors(answer_text, cands, n=2 if pair_q else len(labels) - 1)
    working = [f"Count the {uname}s under each. " + ", ".join(f"{o['label']} {o['units']}" for o in objs) + "."]
    return _finish(fig, text, answer_text, distractors, working, _read_clips)


figures.register_reader("compare_by_paper_clips", lambda q: _read_clips(q.figure, q.text))


# ====================================================================== order by two balances

_ORDER_TEXT = {
    "heaviest": ("Which is the heaviest?", "Look at both balances. Which is heaviest?", "Which one is the heaviest?"),
    "lightest": ("Which is the lightest?", "Look at both balances. Which is lightest?", "Which one is the lightest?"),
    "order": (
        "Which list goes from lightest to heaviest?",
        "Look at both balances. Which list goes lightest to heaviest?",
        "Put them in order, lightest first. Which list is right?",
        "Use both balances to put the three in order. Which list is right?",
    ),
}


def _chain_order(fig: dict) -> list[str]:
    """Labels from lightest to heaviest, from the tilts alone (never the masses)."""
    beats: dict[str, set[str]] = {}
    names: set[str] = set()
    for b in fig["balances"]:
        tilt = cmp.read_tilt(b)
        lo, hi = b["left"]["objects"][0]["label"], b["right"]["objects"][0]["label"]
        names |= {lo, hi}
        if tilt == "level":
            raise ValueError("a level balance cannot order the objects")
        heavy, light = (lo, hi) if tilt == "left_down" else (hi, lo)
        beats.setdefault(heavy, set()).add(light)
    wins = {n: len(_below(n, beats)) for n in names}
    check(len(set(wins.values())) == len(names), "the balances do not settle the order")
    return sorted(names, key=wins.get)


def _below(n: str, beats: dict, seen=None) -> set[str]:
    seen = set() if seen is None else seen
    for m in beats.get(n, ()):
        if m not in seen:
            seen.add(m)
            _below(m, beats, seen)
    return seen


def _read_order(fig: dict, text: str) -> str:
    order = _chain_order(fig)
    t = text.lower()
    if "list" in t:
        return ", ".join(order)
    return _the(order[0] if "light" in t else order[-1])


@generator("order_by_two_balances", level=1, presets=_PRESETS)
def order_by_two_balances(rng, tier="easy"):
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    names = rng.sample(_MASS_OBJECTS, 3)
    masses = sorted(rng.sample(range(2, 15), 3))
    for _ in attempts(500, "three objects with distinct drawn sizes"):
        sizes = [rng.randint(1, 3) for _ in range(3)]
        if len(set(sizes)) == 3 and sizes != [1, 2, 3] and sizes != [3, 2, 1]:
            break
    light, mid, heavy = ({"label": n, "mass": m, "size": s} for n, m, s in zip(names, masses, sizes))
    pans = [(heavy, mid), (mid, light)]
    rng.shuffle(pans)
    balances = []
    for a, b in pans:
        if rng.random() < 0.5:
            a, b = b, a
        balances.append(cmp.balance(left=cmp.pan(objects=[a]), right=cmp.pan(objects=[b])))
    fig = cmp.compare_balance(balances=balances, alt="Two pan balances side by side. One object is on both of them.")
    kind = {"easy": "heaviest", "medium": "lightest", "hard": "order"}[tier]
    text = rng.choice(_ORDER_TEXT[kind])
    answer_text = _read_order(fig, text)
    order = [light["label"], mid["label"], heavy["label"]]
    if kind == "order":
        l, m, h = order
        cands = [", ".join(order[::-1]), f"{m}, {l}, {h}", f"{l}, {h}, {m}"]
        distractors = pick_distractors(answer_text, cands, n=3)
    else:
        distractors = pick_distractors(answer_text, [_the(n) for n in order], n=2)
    working = [f"Lightest to heaviest: {', '.join(order)}. The lower side of each balance is heavier."]
    return _finish(fig, text, answer_text, distractors, working, _read_order)


figures.register_reader("order_by_two_balances", lambda q: _read_order(q.figure, q.text))


# ====================================================================== capacity by cups

_CUPS_TEXT = {
    "pair_more": ("Each is poured into small cups. Which holds more?", "Count the cups poured out. Which holds more?"),
    "pair_less": ("Each is poured into small cups. Which holds less?", "Count the cups poured out. Which holds less?"),
    "many_more": (
        "Each is poured into small cups. Which holds the most?",
        "Count the cups poured out. Which holds most?",
    ),
    "many_less": (
        "Each is poured into small cups. Which holds the least?",
        "Count the cups poured out. Which holds least?",
    ),
    "same": (
        "Each is poured into small cups. Which two hold the same?",
        "Which two hold the same? Count the cups.",
        "Two of these hold the same amount. Which two?",
        "Look at the cups. Which two are the same?",
    ),
}

# Containers that stand equally tall while holding different amounts: capacity * FORM_RATIO is equal.
_SAME_HEIGHT = (
    (("normal", 2), ("tall_thin", 1), ("short_wide", 4)),
    (("normal", 4), ("tall_thin", 2), ("short_wide", 8)),
)


def _poured(shape: str, form: str, cap: int) -> dict:
    return {"label": shape, "shape": shape, "form": form, "capacity": cap, "fill": cap, "cups": cap}


def _draw_cups(rng, tier: str):
    """(containers, more, key, trap)."""
    more = rng.random() < 0.5
    shapes = rng.sample(_SHAPES, {"easy": 2, "medium": 3, "hard": 4}[tier])
    for _ in attempts(4000, "a poured-cups comparison"):
        if tier == "hard":
            group = rng.choice(_SAME_HEIGHT)
            (fa, ca), (fb, cb) = rng.sample(group, 2)
            c0 = rng.choice([c for c in range(2, 8) if c not in (ca, cb)])
            f1, f2 = rng.sample(cmp.FORMS, 2)
            specs = [(fa, ca), (fb, cb), (f1, c0), (f2, c0)]
            rng.shuffle(specs)
            cs = [_poured(sh, f, c) for sh, (f, c) in zip(shapes, specs)]
            key = tuple(c["label"] for c in cs if c["capacity"] == c0)
            trap = tuple(
                c["label"]
                for c in cs
                if c["capacity"] in (ca, cb) and (c["form"], c["capacity"]) in ((fa, ca), (fb, cb))
            )
            if len({c["capacity"] for c in cs}) != 3 or len(key) != 2 or len(trap) != 2:
                continue
            return cs, more, key, trap
        n = len(shapes)
        caps = [rng.randint(1, 8) for _ in range(n)]
        forms = [rng.choice(cmp.FORMS) for _ in range(n)]
        if len(set(caps)) != n or len(set(forms)) < 2:
            continue
        if n == 2 and abs(caps[0] - caps[1]) < 2:
            continue
        cs = [_poured(sh, f, c) for sh, f, c in zip(shapes, forms, caps)]
        values = {c["label"]: c["capacity"] for c in cs}
        key = _extreme(values, more)
        heights = {c["label"]: _height(c) for c in cs}
        if key is None or _extreme(heights, more) in (None, key):
            continue  # the tallest (or shortest) must be a different container, and clearly
        return cs, more, key, None
    raise AssertionError("unreachable")


def _read_cups(fig: dict, text: str) -> str:
    cups = {c["label"]: c["cups"] for c in fig["containers"]}
    if "same" in text.lower():
        by: dict[int, list[str]] = {}
        for c in fig["containers"]:
            by.setdefault(cups[c["label"]], []).append(c["label"])
        pairs = [v for v in by.values() if len(v) == 2]
        check(len(pairs) == 1, "no single pair that hold the same")
        return _pair_text(*pairs[0])
    key = _extreme(cups, _wants_more(text))
    check(key is not None, "no unique container")
    return _the(key)


@generator("compare_capacity_by_cups", level=1, presets=_PRESETS)
def compare_capacity_by_cups(rng, tier="easy"):
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    cs, more, key, trap = _draw_cups(rng, tier)
    fig = cmp.compare_containers(
        containers=cs, alt="Containers of different shapes, each poured out into a row of small cups."
    )
    labels = [c["label"] for c in cs]
    if tier == "hard":
        text = rng.choice(_CUPS_TEXT["same"])
        order = {lbl: i for i, lbl in enumerate(labels)}

        def pair(p):
            a, b = sorted(p, key=order.get)
            return _pair_text(a, b)

        answer_text = pair(key)
        others = [p for p in itertools.combinations(labels, 2) if set(p) not in (set(key), set(trap))]
        cands = [pair(trap), pair(rng.choice(others)), pair(rng.choice(others))]
        distractors = pick_distractors(answer_text, cands, n=2)
        working = [
            "Count the cups under each. Two have the same number of cups. Standing equally tall is not the same."
        ]
        return _finish(fig, text, answer_text, distractors, working, _read_cups)
    text = rng.choice(_CUPS_TEXT[("pair_" if len(cs) == 2 else "many_") + ("more" if more else "less")])
    answer_text = _the(key)
    heights = {c["label"]: _height(c) for c in cs}
    cands = [_the(_extreme(heights, more))] + [_the(lbl) for lbl in labels if lbl != key]
    if len(cs) == 2:
        cands.append(_SAME_HOLD)
    distractors = pick_distractors(answer_text, cands, n=2)
    working = ["Count the cups poured out: " + ", ".join(f"{c['label']} {c['cups']}" for c in cs) + "."]
    return _finish(fig, text, answer_text, distractors, working, _read_cups)


figures.register_reader("compare_capacity_by_cups", lambda q: _read_cups(q.figure, q.text))
