"""Small helpers shared by the `dg_*` data-display and graph builds: whole-or-half-number answers with tagged
distractors, bar-height replies, and a part builder. Nothing here draws random numbers or registers a topic."""

from __future__ import annotations

import random
from fractions import Fraction
from types import SimpleNamespace

from ..answers import Distractor
from ..grading import GRADERS
from ..grading.interactive import bars_answer, fraction_json
from ._spread_kit import _RNG, REQUIRED, check_bounds, mixed, round_extras, round_step

NEAR = "dg-near-miss-value"  # the plain near miss that fills a side of the options

__all__ = [
    "balanced",
    "draw_ranked",
    "bars_fields",
    "bars_dist",
    "bars_text",
    "numd",
    "numf",
    "numtext",
    "set_fields",
    "p_number",
    "p_bars",
]


def numtext(value) -> str:
    """A whole number, or a half-step number as a decimal: ``7`` or ``7.5`` (never ``15/2``)."""
    value = Fraction(value)
    if value.denominator == 1:
        return str(value.numerator)
    if value.denominator == 2:
        return f"{float(value):.1f}"
    raise ValueError(f"{value} is not a whole or half number")


def _py(value):
    value = Fraction(value)
    return value.numerator if value.denominator == 1 else float(value)


def numf(value) -> dict:
    """``answer``, ``answer_text`` and ``answer_latex`` for a whole or half-step number."""
    return dict(answer=_py(value), answer_text=numtext(value), answer_latex=numtext(value))


def numd(
    answer, candidates, n: int = 3, *, lo=REQUIRED, hi=REQUIRED, unbounded=None, nonzero=False, step=None
) -> list[Distractor]:
    """Tagged distractors for a whole or half-step number answer. ``candidates`` is ``[(value, tag), ...]``; a value
    equal to the answer, repeated, outside ``[lo, hi]`` or not a whole or half number is skipped. Every option has
    the same number of decimal places as the answer (``7.0`` stands for 7 beside an answer of 7.5)."""
    check_bounds(lo, hi, unbounded)
    answer = Fraction(answer)
    seen = {answer}
    out: list[Distractor] = []
    # The picks below take the first usable candidates; mix the order so the key has no fixed rank among the options.
    step = (
        Fraction(step)
        if step is not None  # a reading off an axis in unit steps: a neighbour is one unit away, not a gridline
        else Fraction(round_step(answer, always=True))
        if answer.denominator == 1
        else Fraction(1, 2)
    )
    # plain near misses stay positive unless the caller allows less (`lo=0` lets a difference or a count be 0)
    floor = 0 if lo is not None and lo <= 0 else None
    # a signed answer (a difference of two readings) has signed neighbours, on both sides of 0
    signed = (lo is not None and lo < 0) or (lo is None and answer < 0)
    near = [
        (answer + d * step, NEAR)
        for d in (-3, -2, -1, 1, 2, 3)
        if signed or (answer > 0 and (answer + d * step > 0 or (floor is not None and answer + d * step >= floor)))
    ]
    near += [(v, t) for v, t in round_extras(answer, lambda v: (Fraction(v), NEAR)) if answer.denominator == 1]

    def usable(value):
        value = Fraction(value)
        if value.denominator not in (1, 2) or (answer.denominator == 1 and value.denominator != 1):
            return False  # a half beside a whole-number answer would give the answer away by its shape
        return value != answer and not (nonzero and value == 0)

    # only options that will be shown take part in the draw, so the key's rank is the one the draw gave it
    candidates = mixed(
        answer,
        [(v, t) for v, t in candidates if v is not None and usable(v)],
        fillers=[f for f in near if usable(f[0])],
        lo=lo,
        hi=hi,
        unbounded=unbounded,
    )
    for value, tag in candidates:
        value = Fraction(value)
        if (
            value in seen
            or value.denominator not in (1, 2)
            or (lo is not None and value < lo)
            or (hi is not None and value > hi)
        ):
            continue
        seen.add(value)
        text = numtext(value) if answer.denominator == 1 and value.denominator == 1 else f"{float(value):.1f}"
        if answer.denominator == 1 and value.denominator != 1:
            continue  # a half beside a whole-number answer would give the answer away by its shape
        out.append(Distractor(text, tag, _py(value)))
        if len(out) == n:
            break
    return out


# -- bar heights (the `bars` answer) ---------------------------------------------------------------------------------

TOP = 40


def bars_text(categories, heights) -> str:
    return ", ".join(f"{c} = {numtext(h)}" for c, h in zip(categories, heights))


def bars_fields(categories, heights, snap=1) -> dict:
    text = bars_text(categories, heights)
    return dict(
        answer=bars_answer(categories, [Fraction(h) for h in heights], top=TOP, snap=snap),
        answer_text=text,
        answer_latex="plain:" + text,
    )


def balanced(usable, key_feature, n: int, rng) -> list[Distractor]:
    """`n` of the options in `usable`, a list of ``(feature, named, option)``, with a random number of them below the
    key's feature (a tuple read left to right): the key's place among the options is drawn, not left to the slips.

    The named slips come first on their side. Plain fillers would then be used only on a side the slips leave short, so a
    question with a filler would be one whose key sits at an end; three times in four a named slip gives its place to a
    filler on its side."""
    below = [(d, named) for f, named, d in usable if f < key_feature]
    above = [(d, named) for f, named, d in usable if f > key_feature]
    for side in (below, above):
        rng.shuffle(side)
        side.sort(key=lambda item: not item[1])  # named first (a stable sort keeps the shuffle within each kind)
    ranks = [r for r in range(n + 1) if r <= len(below) and n - r <= len(above)]
    if not ranks:
        return [d for _, _, d in usable][:n]
    r = rng.choice(ranks)
    chosen = [*below[:r], *above[: n - r]]
    if rng.random() < 0.75 and all(named for _, named in chosen):
        for side, taken in rng.sample([(below, below[:r]), (above, above[: n - r])], 2):
            spare = [item for item in side if not item[1] and item not in taken]
            if taken and spare:
                chosen[chosen.index(rng.choice(taken))] = spare[0]
                break
    return [d for d, _ in chosen]


def draw_ranked(usable, key_features: dict, n: int, rng) -> list[Distractor]:
    """`n` of `usable`, a list of ``(features, named, option)``, drawn so that the key's rank on EACH feature is random.

    `features` maps a name to a number (a count, a mean row, a mean column). Many draws of `n` options are made, named
    slips a little more likely than plain fillers; one feature and then one of the ranks (wrong options below the key,
    ties counting half) the draws reach on it are picked at random, and a draw with that rank is returned. Over many
    questions every feature's rank is spread, however the slips cluster."""
    if len(usable) <= n:
        return [d for _, _, d in usable]

    def rank_of(draw):
        out = {}
        for name, key in key_features.items():
            vals = [f[name] for f, _, _ in draw]
            out[name] = sum(v < key for v in vals) + sum(v == key for v in vals) / 2
        return out

    draws = []
    for _ in range(60):
        order = sorted(usable, key=lambda u: rng.random() * (0.5 if u[1] else 1.0))
        draw = order[:n]
        draws.append((rank_of(draw), draw))
    varying = [name for name in key_features if len({r[name] for r, _ in draws}) > 1]
    if not varying:
        return [d for _, _, d in draws[0][1]]
    name = rng.choice(varying)
    target = rng.choice(sorted({r[name] for r, _ in draws}))
    return [d for _, _, d in rng.choice([dr for r, dr in draws if r[name] == target])]


def bars_dist(categories, heights, candidates, snap=1, n: int = 3) -> list[Distractor]:
    """Tagged distractors for a bars answer: ``candidates`` is ``[(wrong heights, tag), ...]``. Kept when the real
    grader marks the typed text wrong for the answer and right for the heights it stands for.

    The key's place among the options is drawn, not left to the slips: a bar list is read left to right, so a wrong
    list is below the key where its first differing bar is lower. The usable slips are split into those below and
    those above the key, plus one-step bumps of a single bar (a plain near miss, which can go either way), and a random
    number of the `n` options come from below."""
    key = bars_fields(categories, heights, snap)
    probe = SimpleNamespace(answer=key["answer"], answer_text=key["answer_text"], answer_kind="bars", parts=None)
    seen = {key["answer_text"]}
    rng = _RNG.get() or random.Random(key["answer_text"])
    step = Fraction(snap)
    near = []
    for i in range(len(heights)):
        for d in (-1, 1):
            bumped = [Fraction(h) for h in heights]
            bumped[i] += d * step
            near.append((bumped, NEAR))
    usable = []
    key_heights = [Fraction(h) for h in heights]
    for named, group in ((0, candidates), (1, near)):
        for wrong, tag in group:
            if wrong is None or any(Fraction(h) < 0 or Fraction(h) > TOP for h in wrong):
                continue
            text = bars_text(categories, wrong)
            if text in seen:
                continue
            stands = bars_fields(categories, wrong, snap)
            if GRADERS["bars"](probe, text).correct:
                continue
            if not GRADERS["bars"](
                SimpleNamespace(answer=stands["answer"], answer_text=text, answer_kind="bars", parts=None), text
            ).correct:
                continue
            seen.add(text)
            # the named slips come first on their side, the plain bumps fill it
            usable.append(
                (
                    [Fraction(h) for h in wrong],
                    not named,
                    Distractor(text, tag, stands["answer"]),
                )
            )
    return balanced(usable, key_heights, n, rng)


def set_fields(items) -> dict:
    items = sorted(items)
    text = "{" + ", ".join(str(i) for i in items) + "}"
    return dict(answer=[str(i) for i in items], answer_text=text, answer_latex="plain:" + text)


def p_number(prompt, value, marks, working, distractors, **extra) -> dict:
    return dict(
        prompt=prompt,
        answer_kind="number",
        marks=marks,
        working=list(working),
        distractors=distractors,
        **numf(value),
        **extra,
    )


def p_bars(prompt, categories, heights, marks, working, distractors, snap=1, **extra) -> dict:
    return dict(
        prompt=prompt,
        answer_kind="bars",
        marks=marks,
        working=list(working),
        distractors=distractors,
        **bars_fields(categories, heights, snap),
        **extra,
    )


__all__ += ["fraction_json", "TOP"]
