"""Reading and using a hundred chart: ``hundred_chart_read`` (skill ``f10-hundred-chart-patterns``).

The chart is the whole stimulus: the number the learner must give is a blank cell, so it is never
printed. Tiers escalate by what the learner does with the chart:

- easy: one shaded blank square; say what number goes in it (find it by position).
- medium: one ringed number and a handful of blank squares; say what is one more, one less, ten more
  or ten less than the ringed number (the answer is among the blanks, so it cannot be copied).
- hard: four or five shaded squares make a skip-counting pattern with one of them blank; find the
  step from the numbers that show and say what is in the blank (steps of 2, 3, 4, 5, 6 or 10; the
  chart starts at 0 or 1).

Each question's reader re-derives the answer from ``question.figure`` (and, for medium, the wording
"one"/"ten" and "more"/"less" of the prompt).
"""

from __future__ import annotations

from .. import figures
from ..core import attempts, check, generator, pick_distractors
from ..figure_kinds.early import hundred_chart
from ..slots import NAMES

_TIERS = ("easy", "medium", "hard")
_PRESETS = {t: {"tier": t} for t in _TIERS}

_TEXT = {
    "easy": (
        "{n} is filling in a number chart. What number goes in the shaded square?",
        "{n} sees a shaded square with no number. What number is missing?",
        "{n} finds one shaded square that is empty. What number goes in it?",
    ),
    "medium": (
        "{n} looks at the ringed number. What is {q} it?",
        "{n} uses the chart. What number is {q} the ringed number?",
        "{n} checks the chart. What is {q} the ringed number?",
    ),
    "hard": (
        "{n} sees a counting pattern in the shaded squares. One is empty. What number goes in it?",
        "{n} spots a pattern in the shaded squares. What number is missing from the pattern?",
        "{n} colours some squares in a counting pattern. What number goes in the empty one?",
    ),
}

_QUANT = {(1, 1): "one more than", (1, -1): "one less than", (10, 1): "ten more than", (10, -1): "ten less than"}
_STEPS = (2, 3, 4, 5, 6, 10)


def _in_range(start: int, n: int) -> bool:
    return start <= n <= start + 99


def _build(rng, tier: str):
    for _ in attempts(500, "a hundred-chart question"):
        if tier == "easy":
            start = 1
            ans = rng.randint(start, start + 99)
            spec = dict(start=start, hidden=[ans], shaded=[ans], marked=[])
            info = {}
        elif tier == "medium":
            start = 1
            step, sign = rng.choice(list(_QUANT))
            ringed = rng.randint(start, start + 99)
            ans = ringed + step * sign
            if not _in_range(start, ans):
                continue
            decoys = {n for n in rng.sample(range(start, start + 100), 8) if n not in (ringed, ans)}
            spec = dict(start=start, hidden=sorted(decoys | {ans}), shaded=[], marked=[ringed])
            info = {"q": _QUANT[(step, sign)]}
        else:
            start = rng.choice((0, 1))
            step = rng.choice(_STEPS)
            terms = rng.randint(4, 5)
            first = rng.randint(start, start + 99 - step * (terms - 1))
            seq = [first + step * i for i in range(terms)]
            hole = rng.randint(1, terms - 1)
            ans = seq[hole]
            spec = dict(start=start, hidden=[ans], shaded=seq, marked=[])
            info = {"step": step, "seq": seq}
        fig = hundred_chart(alt="a hundred chart with some squares blank", **spec)
        try:
            figures.assert_no_leak(fig, [str(ans)])
        except AssertionError:
            continue
        return fig, ans, info
    raise AssertionError("unreachable")


def _distractors(tier: str, ans: int, info: dict, fig: dict):
    if tier == "easy":
        return [ans + 1, ans - 1, ans + 10, ans - 10]
    if tier == "medium":
        ringed = fig["marked"][0]
        step = 10 if "ten" in info["q"] else 1
        other = 1 if step == 10 else 10
        sign = 1 if ans > ringed else -1
        return [ringed - sign * step, ringed + sign * other, ans + 1, ans - 1]
    seq, step = info["seq"], info["step"]
    prev = ans - step
    return [ans + 1, ans - 1, prev + 1, prev + 10, ans + 10, ans - step * 2]


@generator("hundred_chart_read", level=1, presets=_PRESETS)
def hundred_chart_read(rng, tier="easy"):
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    fig, ans, info = _build(rng, tier)
    text = rng.choice(_TEXT[tier]).format(n=rng.choice(NAMES), **{k: v for k, v in info.items() if k == "q"})
    answer_text = str(ans)
    lo, hi = fig["start"], fig["start"] + 99
    cands = [str(c) for c in _distractors(tier, ans, info, fig) if lo <= c <= hi]
    distractors = pick_distractors(answer_text, cands)
    check(len(distractors) >= 2, "too few distractors")
    figures.assert_no_leak(fig, [answer_text])
    check(_read(fig, text) == answer_text, "the chart does not show the keyed number")
    working = {
        "easy": [f"Count along the row to the shaded square: it is {ans}."],
        "medium": [f"{info.get('q', '')} the ringed number is {ans}."],
        "hard": [f"The pattern goes up by {info.get('step')} each time, so the missing number is {ans}."],
    }[tier]
    return dict(
        text=text,
        answer=ans,
        answer_text=answer_text,
        answer_latex=answer_text,
        distractors=distractors,
        working=working,
        figure=fig,
    )


def _read(fig: dict, text: str) -> str:
    hidden, shaded, marked = set(fig["hidden"]), fig["shaded"], fig["marked"]
    if marked:
        check(len(marked) == 1 and not shaded, "a ringed number question rings exactly one square")
        step = 10 if " ten " in f" {text} " else 1
        sign = 1 if " more " in f" {text} " else -1
        ans = marked[0] + sign * step
        check(ans in hidden, "the answer square is not blank")
        return str(ans)
    blank = sorted(hidden & set(shaded))
    check(len(blank) == 1, "exactly one shaded square must be blank")
    if len(shaded) > 1:
        visible = sorted(set(shaded) - hidden)
        step = min(b - a for a, b in zip(visible, visible[1:]))
        whole = sorted(visible + blank)
        check(all(b - a == step for a, b in zip(whole, whole[1:])), "shaded squares are not an even counting pattern")
    return str(blank[0])


figures.register_reader("hundred_chart_read", lambda q: _read(q.figure, q.text))
