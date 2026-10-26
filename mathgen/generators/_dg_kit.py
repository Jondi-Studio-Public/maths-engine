"""Small helpers shared by the `dg_*` data-display and graph builds: whole-or-half-number answers with tagged
distractors, bar-height replies, and a part builder. Nothing here draws random numbers or registers a topic."""

from __future__ import annotations

import functools
import random
import re
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

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
    equal to the answer, repeated, outside ``[lo, hi]`` or not a whole or half number is skipped. A whole value is written as a whole number (``7`` beside an answer of 7.5, never ``7.0``)."""
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
        text = numtext(value)  # a whole option is "4", not "4.0"
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


def _fitted_draw(usable, key_features: dict, n: int, rng) -> list[Distractor]:
    """The `joint` draw of `draw_ranked`, done on arrays: 2000 draws of `n` (named slips favoured) and, where the
    features hold `_items`, 300 clusters (one option and the ones nearest it), each draw's key rank on every feature,
    weights fitted (iterative proportional fitting) until each feature's rank is near uniform over the values the draws
    reach, and one draw taken by weight."""
    g = np.random.default_rng(rng.getrandbits(63))
    count = len(usable)
    weight = np.where([u[1] for u in usable], 3.0, 1.0)  # the favour a named slip has over a plain filler
    keys = np.log(weight) + g.gumbel(size=(2000, count))
    picks = np.argpartition(-keys, n - 1, axis=1)[:, :n]
    ranks = {}

    def add(name, below, tied):
        r = below + tied / 2
        if len(np.unique(r)) > 1:
            ranks[name] = np.unique(r, return_inverse=True)[1]

    if "_items" in key_features:
        names = sorted({i for u in usable for i in u[0]["_items"]} | set(key_features["_items"]))
        col = {name: k for k, name in enumerate(names)}
        mat = np.zeros((count, len(names)))
        for row, u in enumerate(usable):
            mat[row, [col[i] for i in u[0]["_items"]]] = 1
        key_row = np.zeros(len(names))
        key_row[[col[i] for i in key_features["_items"]]] = 1
        inter = mat @ mat.T
        size = mat.sum(axis=1)
        share = inter / np.maximum(size[:, None] + size[None, :] - inter, 1)  # how much two options share
        to_key = (mat @ key_row) / np.maximum(size + key_row.sum() - mat @ key_row, 1)
        # a cluster: one option and the options nearest it (a set of options the key is NOT the hub of; random draws
        # seldom give them, and without them the key is nearly always the list the others are a slip away from)
        seeds = g.integers(0, count, size=300)
        near = np.argsort(-share[seeds] + g.random((300, count)) * 0.3, axis=1)
        near = np.where(near == seeds[:, None], -1, near)
        clusters = np.concatenate([seeds[:, None], np.array([row[row >= 0][: n - 1] for row in near])], axis=1)
        picks = np.concatenate([picks, clusters])
        pair = share[picks[:, :, None], picks[:, None, :]]
        pair[:, np.arange(n), np.arange(n)] = 0
        agree = pair.sum(axis=2) + to_key[picks]  # each wrong option's sharing with the other wrong ones and the key
        key_agree = to_key[picks].sum(axis=1)
        add("agreement", (agree < key_agree[:, None]).sum(axis=1), (agree == key_agree[:, None]).sum(axis=1))
    for name, key in key_features.items():
        if name == "_items":
            continue
        sel = np.array([u[0][name] for u in usable], dtype=float)[picks]
        add(name, (sel < key).sum(axis=1), (sel == key).sum(axis=1))
    if not ranks:
        return [usable[i][2] for i in picks[0]]
    w = np.ones(len(picks))
    for _ in range(25):
        for inv in ranks.values():
            mass = np.bincount(inv, weights=w)
            w = w * (w.sum() / len(mass)) / mass[inv]
    return [usable[i][2] for i in picks[g.choice(len(picks), p=w / w.sum())]]


def draw_ranked(usable, key_features: dict, n: int, rng, joint: bool = False) -> list[Distractor]:
    """`n` of `usable`, a list of ``(features, named, option)``, drawn so that the key's rank on EACH feature is random.

    `features` maps a name to a number (a count, a mean row, a mean column). Many draws of `n` options are made, named
    slips a little more likely than plain fillers; one feature and then one of the ranks (wrong options below the key,
    ties counting half) the draws reach on it are picked at random, and a draw with that rank is returned. Over many
    questions every feature's rank is spread, however the slips cluster.

    With `joint` (numeric features only, and the name `_items`, a list's entries, which adds a rank on how much the list
    shares with the other options), thousands of draws are made and weighted (iterative proportional fitting) so that
    EVERY feature's rank is as near uniform over the values it reaches as they allow, however the features move
    together (a bigger list is also a later one), and one draw is taken by weight."""
    if len(usable) <= n:
        return [d for _, _, d in usable]
    if joint:
        return _fitted_draw(usable, key_features, n, rng)

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


# -- options that are an expression in x or an inequality, ranked by what they ARE ---------------------------------------

_INF = 1e9
_X = sympy.Symbol("x")


@functools.lru_cache(maxsize=16384)
def _expression_values(text: str):
    """(value at 0.37, value at 1.41) of an expression in x as the grader reads it, or None if it cannot be read."""
    body = re.sub(r"\s*\+\s*C$", "", text.strip()).replace("^", "**").replace("ln", "log")
    try:
        expr = parse_expr(
            body,
            local_dict={"x": _X, "e": sympy.E},
            transformations=standard_transformations + (implicit_multiplication_application,),
        )
        out = []
        for point in (0.37, 1.41):
            value = complex(expr.subs(_X, point))
            out.append(value.real if abs(value.imag) < 1e-9 else _INF)
        return tuple(out)
    except (SyntaxError, TypeError, ValueError, ZeroDivisionError, sympy.SympifyError, AttributeError):
        return None


def _number_of(text: str):
    try:
        return float(Fraction(text.strip()))
    except (ValueError, ZeroDivisionError):
        return None


def _inequality_features(text: str):
    """The first and the last bound as written (0 for `all real numbers` and `no solution`), the sum of the bounds, and
    the length of an inequality, else None if the text is not an inequality in x or y."""
    low = text.strip()
    if low not in ("all real numbers", "no solution") and not re.fullmatch(
        r"(-?[\d./]+ (≤|<) )?[xy]( (≤|<|≥|>) -?[\d./]+)?( or [xy] (≤|<|≥|>) -?[\d./]+)?", low
    ):
        return None
    nums = [_number_of(b) for b in re.findall(r"-?\d+(?:\.\d+)?(?:/\d+)?", low)]
    if any(n is None for n in nums):
        return None
    return {
        "first": nums[0] if nums else 0.0,
        "last": nums[-1] if nums else 0.0,
        "length": float(len(low)),
    }


def ranked_text(key_text, pool, kind: str, count: int = 3, evaluate=None, values=None) -> list[Distractor]:
    """`count` of `pool` (wrong expressions in x, or inequalities, as Distractors) drawn so the key's rank is spread on
    each thing a reader could rank them by: for an expression its value at two points and its length, for an
    inequality its first and last bound and its length. An option that cannot be read
    is kept out of the draw, so the draw ranks only what it can measure. `evaluate` is a function from an expression's
    text to a callable of x, when the caller has one already (it spares parsing every option again); `values` maps an
    option's text (and the key's) to its two values, when the caller has those."""

    def feats(text):
        if kind == "inequality":
            return _inequality_features(text)
        if values is not None and text in values:
            at = values[text]
        elif evaluate is not None:
            try:
                at = tuple(float(evaluate(text)(p)) for p in (0.37, 1.41))
            except (ValueError, ZeroDivisionError, OverflowError, TypeError, SyntaxError):
                at = None
            else:
                at = tuple(_INF if v != v or abs(v) == float("inf") else v for v in at)
        else:
            at = _expression_values(text)
        return None if at is None else {"at 0.37": at[0], "at 1.41": at[1], "length": float(len(text.strip()))}

    pool = list(pool)
    key = feats(str(key_text))
    usable = [
        (f, not (d.misconception or "").endswith("near-miss-value"), d)
        for d in pool
        if (f := feats(d.text)) is not None
    ]
    if key is None or len(usable) < count:
        return pool[:count]
    rng = _RNG.get() or random.Random(str(key_text))
    return draw_ranked(usable, key, count, rng, joint=True)
