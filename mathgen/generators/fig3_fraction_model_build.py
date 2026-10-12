"""Fractions from a picture: shaded wholes drawn with the ``fraction_model`` figure kind.

- ``fraction_shaded_name`` (skill ``f10-fraction-shaded-name``, ``answer: fraction``): name the
  fraction of a bar, rectangle grid or circle that is shaded. Easy uses halves, thirds and quarters
  of a bar or circle with the shaded parts together; medium adds five to eight parts, rectangle grids
  and shaded parts that are not next to each other; hard goes to ten and twelve parts and asks for the
  fraction NOT shaded. The key is written with the drawn number of parts (``2/4``); the fraction
  grader compares by value, so ``1/2`` is accepted too.
- ``fraction_compare_models`` (skill ``f10-fraction-compare-models``, ``answer: mcq``): two wholes of
  the same size, A and B; which has more shaded, or are they the same? Easy cuts both into the same
  number of parts; medium compares one shaded part of two different cuts (more parts, smaller parts);
  hard compares fractions with different numbers of parts where the one with more shaded parts is
  not the larger fraction, or the two are equal.

Prompts never state a fraction. Each topic registers a figure reader that works from the figure.
"""

from __future__ import annotations

from fractions import Fraction

from .. import figures
from ..core import attempts, check, generator, pick_distractors
from ..figure_kinds.early import fraction_model, fraction_shape
from ..slots import NAMES

_TIERS = ("easy", "medium", "hard")
_PRESETS = {t: {"tier": t} for t in _TIERS}

_DIMS = {2: (1, 2), 3: (1, 3), 4: (2, 2), 5: (1, 5), 6: (2, 3), 7: (1, 7), 8: (2, 4), 9: (3, 3), 10: (2, 5), 12: (3, 4)}
_CIRCLE = (2, 3, 4, 5, 6, 8, 10, 12)
_BAR = (2, 3, 4, 5, 6, 7, 8, 9, 10, 12)
_PARTS = {"bar": _BAR, "circle": _CIRCLE, "rectangle": tuple(_DIMS)}
_WORD = {"bar": "bar", "circle": "circle", "rectangle": "rectangle"}
_PLURAL = {"bar": "bars", "circle": "circles", "rectangle": "rectangles"}


def _shape(kind: str, parts: int, shaded, label: str = "") -> dict:
    rows, cols = _DIMS[parts] if kind == "rectangle" else (None, None)
    return fraction_shape(kind, parts, shaded, rows=rows, cols=cols, label=label)


def _frac_text(k: int, d: int) -> str:
    return f"{k}/{d}"


def _answer_dict(k: int, d: int) -> dict:
    return {"p": k, "q": d}


# ====================================================================== name the shaded fraction

_NAME_TEXT = {
    "easy": (
        "{n} shades part of a {w}. What fraction is shaded?",
        "{n} looks at a {w}. What fraction of it is shaded?",
        "Which fraction of this {w} is shaded? {n} is not sure.",
    ),
    "medium": (
        "{n} colours some parts of a {w}. What fraction is coloured in?",
        "{n} looks at the {w}. What fraction is shaded?",
        "What fraction of the {w} has {n} shaded?",
    ),
    "hard": (
        "{n} shades some of a {w}. What fraction is left unshaded?",
        "{n} looks at the {w}. What fraction has no shading?",
        "What fraction of the {w} is not shaded? {n} wants to know.",
    ),
}


def _draw_name(rng, tier: str):
    if tier == "easy":
        kind = rng.choice(("bar", "circle"))
        parts = rng.choice((2, 3, 4))
        k = rng.randint(1, parts - 1)
        shaded = list(range(k))
    else:
        kind = rng.choice(
            ("bar", "circle", "rectangle") if tier == "medium" else ("bar", "circle", "rectangle", "rectangle")
        )
        options = [p for p in _PARTS[kind] if (5 <= p <= 8 if tier == "medium" else p >= 8)]
        parts = rng.choice(options)
        k = rng.randint(1, parts - 1)
        shaded = sorted(rng.sample(range(parts), k))
    return kind, parts, k, shaded


def _name_distractors(k: int, d: int, asked: int):
    """``asked`` is the number of parts the key counts (shaded or unshaded); the rest are the slips."""
    other = d - asked
    cands = [
        (other, d),
        (asked, other),
        (asked + 1, d),
        (asked - 1, d),
        (asked, d + 1),
        (asked, d - 1),
        (asked, 2 * d),
        (asked + 1, d + 1),
    ]
    cands = [c for c in cands if c[1] in _DIMS]
    return [_frac_text(p, q) for p, q in cands if 0 < p < q and Fraction(p, q) != Fraction(asked, d)]


@generator("fraction_shaded_name", level=3, presets=_PRESETS)
def fraction_shaded_name(rng, tier="easy"):
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    kind, parts, k, shaded = _draw_name(rng, tier)
    fig = fraction_model(models=[_shape(kind, parts, shaded)], alt=f"a {_WORD[kind]} cut into equal parts, some shaded")
    asked = k if tier != "hard" else parts - k
    text = rng.choice(_NAME_TEXT[tier]).format(n=rng.choice(NAMES), w=_WORD[kind])
    answer_text = _frac_text(asked, parts)
    distractors = pick_distractors(answer_text, _name_distractors(k, parts, asked))
    check(len(distractors) >= 2, "too few distractors")
    figures.assert_no_leak(fig, [answer_text, *distractors])
    check(_read_name(fig, text) == answer_text, "the figure does not show the keyed fraction")
    what = "shaded" if tier != "hard" else "not shaded"
    return dict(
        text=text,
        answer=_answer_dict(asked, parts),
        answer_text=answer_text,
        answer_latex=answer_text,
        distractors=distractors,
        working=[f"The {_WORD[kind]} has {parts} equal parts. {asked} of them are {what}: {answer_text}."],
        figure=fig,
    )


def _read_name(fig: dict, text: str) -> str:
    m = fig["models"][0]
    n = len(m["shaded"])
    if "not shaded" in text or "unshaded" in text or "no shading" in text:
        n = m["parts"] - n
    return _frac_text(n, m["parts"])


figures.register_reader("fraction_shaded_name", lambda q: _read_name(q.figure, q.text))


# ====================================================================== compare two models

_SAME = "Same amount"
_A, _B = "Model A", "Model B"

_CMP_TEXT = {
    "easy": (
        "Model A and Model B are the same size. Which has more shaded, {n} asks?",
        "{n} shades two {ws} that are the same size. Which has more shaded?",
        "{n} looks at two models. Which has more shaded?",
    ),
    "medium": (
        "These two {ws} are the same size. Which model has more shaded, {n} asks?",
        "{n} compares two {ws} of the same size. Which has the larger shaded part?",
        "{n} wonders which model shows the greater amount shaded.",
    ),
    "hard": (
        "{n} sees two models of the same size. Which shows the greater fraction shaded, or are they equal?",
        "{n} compares two {ws} of the same size. Which has more shaded, or is it the same amount?",
        "{n} compares the shaded amounts of the two {ws}. Which is more, or are they the same?",
    ),
}


def _key(a: Fraction, b: Fraction) -> str:
    return _SAME if a == b else (_A if a > b else _B)


def _draw_compare(rng, tier: str):
    kind = rng.choice(("bar", "circle", "rectangle"))
    pool = _PARTS[kind]
    want = rng.choice((_A, _B, _SAME) if tier != "medium" else (_A, _B))
    for _ in attempts(3000, "a fraction comparison with the intended key"):
        if tier == "easy":
            d = rng.choice([p for p in pool if p <= 8])
            da = db = d
            ka, kb = rng.randint(1, d - 1), rng.randint(1, d - 1)
            if want == _SAME:
                kb = ka
        elif tier == "medium":
            da, db = rng.sample([p for p in pool if p <= 10], 2)
            ka = kb = 1
        else:
            da, db = rng.sample(pool, 2)
            ka, kb = rng.randint(1, da - 1), rng.randint(1, db - 1)
        va, vb = Fraction(ka, da), Fraction(kb, db)
        if _key(va, vb) != want:
            continue
        if tier == "easy" and want != _SAME and ka == kb:
            continue
        if tier == "hard":
            if want != _SAME and (abs(va - vb) < Fraction(1, 16) or not (ka == kb or (ka > kb) != (va > vb))):
                continue
        shapes = [
            _shape(kind, da, rng.sample(range(da), ka) if tier == "hard" else list(range(ka)), "A"),
            _shape(kind, db, rng.sample(range(db), kb) if tier == "hard" else list(range(kb)), "B"),
        ]
        return kind, shapes, want
    raise AssertionError("unreachable")


@generator("fraction_compare_models", level=4, presets=_PRESETS)
def fraction_compare_models(rng, tier="easy"):
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    kind, shapes, answer_text = _draw_compare(rng, tier)
    fig = fraction_model(
        models=shapes, alt=f"two {_PLURAL[kind]} of the same size, each cut into equal parts and partly shaded"
    )
    text = rng.choice(_CMP_TEXT[tier]).format(n=rng.choice(NAMES), ws=_PLURAL[kind])
    distractors = pick_distractors(answer_text, [_A, _B, _SAME], n=2)
    check(len(distractors) == 2)
    figures.assert_no_leak(fig, [answer_text, *distractors])
    check(_read_compare(fig) == answer_text, "the figure does not show the keyed model")
    parts = " and ".join(f"{len(m['shaded'])}/{m['parts']}" for m in fig["models"])
    return dict(
        text=text,
        answer=answer_text,
        answer_text=answer_text,
        answer_latex=f"plain:{answer_text}",
        distractors=distractors,
        working=[f"Model A shows {parts.split(' and ')[0]} and Model B shows {parts.split(' and ')[1]}."],
        figure=fig,
    )


def _read_compare(fig: dict) -> str:
    a, b = (Fraction(len(m["shaded"]), m["parts"]) for m in fig["models"])
    return _key(a, b)


figures.register_reader("fraction_compare_models", lambda q: _read_compare(q.figure))
