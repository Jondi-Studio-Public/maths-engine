"""Generates `investigation-step-order` questions: put the actions of a statistical investigation in order.

An investigation is described as a set of lettered actions in shuffled order. The learner types the letters in the
order the investigation would carry them out. The stages are fixed (pose the question, collect the data, organise it,
display it, analyse it, draw the conclusion) and every scenario gives two ways of wording each stage, so the stage
is recognised by what is done and not by a keyword.

Ladder (reasoning escalation, not size):
  easy:   five actions (no organising step); type the five letters in order.
  medium: six actions; type the order, then name the stage one chosen action belongs to (five options).
  hard:   seven actions, one of which is bad practice (changing the question after seeing the result, leaving out
          values that disagree, asking only people who agree) and is not part of a sound investigation; type the
          six letters that are, in order, then name the stage of one chosen action.
  exam:   six actions to order; the stage of one action; the mean of the first group's data; and the difference of the
          group means (follow-through).
"""

from __future__ import annotations

import random

from mathgen.core import attempts, check, generator

from ._ir2a_kit import LETTERS, NAMES, chain, mcq_part, number_part, ordered_part, single

_TIERS = ("easy", "medium", "hard")

POSE, COLLECT, ORGANISE, DISPLAY, ANALYSE, CONCLUDE = "pose", "collect", "organise", "display", "analyse", "conclude"
STAGE_ORDER = (POSE, COLLECT, ORGANISE, DISPLAY, ANALYSE, CONCLUDE)
STAGE_NAME = {
    POSE: "Pose the question",
    COLLECT: "Collect the data",
    ORGANISE: "Organise the data",
    DISPLAY: "Display the data",
    ANALYSE: "Analyse the data",
    CONCLUDE: "Draw a conclusion",
}
_STAGE_CONFUSION = {
    POSE: (COLLECT, ANALYSE),
    COLLECT: (POSE, ORGANISE),
    ORGANISE: (DISPLAY, COLLECT),
    DISPLAY: (ORGANISE, ANALYSE),
    ANALYSE: (CONCLUDE, DISPLAY),
    CONCLUDE: (ANALYSE, POSE),
}
_CONFUSION_TAG = {
    frozenset((ORGANISE, DISPLAY)): "ir2a-confuses-organising-with-displaying",
    frozenset((ANALYSE, CONCLUDE)): "ir2a-confuses-analysing-with-concluding",
    frozenset((POSE, COLLECT)): "ir2a-confuses-posing-with-collecting",
}
_SWAP_TAG = {
    (POSE, COLLECT): "ir2a-collects-before-posing-the-question",
    (COLLECT, ORGANISE): "ir2a-organises-before-collecting",
    (COLLECT, DISPLAY): "ir2a-displays-before-collecting",
    (ORGANISE, DISPLAY): "ir2a-displays-before-organising",
    (DISPLAY, ANALYSE): "ir2a-analyses-before-displaying",
    (ANALYSE, CONCLUDE): "ir2a-concludes-before-analysing",
}

# measure, group A, group B, unit of measurement, who, lo, hi, which is expected higher
_SCENARIOS = (
    dict(
        measure="reaction time",
        unit="milliseconds",
        a="students who slept 8 hours or more",
        b="students who slept less than 8 hours",
        who="students",
        lo=190,
        hi=330,
        how="with a ruler-drop test",
    ),
    dict(
        measure="height after six weeks",
        unit="centimetres",
        a="bean plants given fertiliser",
        b="bean plants given none",
        who="bean plants",
        lo=18,
        hi=48,
        how="with a ruler",
    ),
    dict(
        measure="number of words typed in one minute",
        unit="words",
        a="the morning class",
        b="the afternoon class",
        who="students",
        lo=28,
        hi=64,
        how="with a typing test",
    ),
    dict(
        measure="time the torch stays lit",
        unit="hours",
        a="brand A batteries",
        b="brand B batteries",
        who="batteries",
        lo=7,
        hi=19,
        how="by running each torch until it fails",
    ),
    dict(
        measure="time taken to walk 1 km",
        unit="minutes",
        a="students walking with a friend",
        b="students walking alone",
        who="students",
        lo=9,
        hi=21,
        how="with a stopwatch",
    ),
    dict(
        measure="time taken to finish a puzzle",
        unit="seconds",
        a="students who practised first",
        b="students who did not",
        who="students",
        lo=60,
        hi=150,
        how="with a stopwatch",
    ),
)

_BAD = (
    "Change the question after seeing which group came out ahead",
    "Leave out the results that disagree with what was expected",
    "Collect results only from people who already agree with the claim",
    "Make up a few extra results so that the sample looks bigger",
)


def _actions(rng, sc: dict, n: int) -> dict[str, list[str]]:
    a, b, m, who, how, unit = sc["a"], sc["b"], sc["measure"], sc["who"], sc["how"], sc["unit"]
    return {
        POSE: [
            f"Decide to find out whether the {m} differs between {a} and {b}",
            f"Write down the question: do {a} and {b} differ in {m}?",
        ],
        COLLECT: [
            f"Measure the {m} of {n} {who}, {how}, and note which group each belongs to",
            f"Record the {m} for {n} {who}, noting the group of each",
        ],
        ORGANISE: [
            "Sort the results into a table with one column for each group",
            "Check the results for slips and arrange them in order within each group",
        ],
        DISPLAY: [
            "Draw parallel box plots of the two groups on one scale",
            "Draw a dot plot for each group, one above the other, on the same scale",
        ],
        ANALYSE: [
            "Find the median and the interquartile range of each group",
            "Compare the centre and the spread of the two groups",
        ],
        CONCLUDE: [
            "State which group had the higher typical value, and say the result is for this sample",
            "Write what the data say about the original question, with a note on how far it can be trusted",
        ],
    }


def _scenario(rng, stages: tuple[str, ...], extra_bad: bool):
    sc = rng.choice(_SCENARIOS)
    n = rng.choice([20, 24, 30, 36, 40])
    acts = _actions(rng, sc, n)
    chosen = {s: rng.choice(acts[s]) for s in stages}
    items = list(chosen.values())
    bad = None
    if extra_bad:
        bad = rng.choice(_BAD)
        items.append(bad)
    order = list(items)
    rng.shuffle(order)
    letter_of = {text: LETTERS[i] for i, text in enumerate(order)}
    lines = " ".join(f"{LETTERS[i]}: {text}{'' if text.endswith('?') else '.'}" for i, text in enumerate(order))
    return sc, chosen, bad, letter_of, lines


def _order_part(stages, chosen, letter_of, marks):
    key = [letter_of[chosen[s]] for s in stages]
    wrongs = []
    for i in range(len(stages) - 1):
        pair = (stages[i], stages[i + 1])
        tag = _SWAP_TAG.get(pair, "ir2a-swaps-two-neighbouring-steps")
        w = key[:]
        w[i], w[i + 1] = w[i + 1], w[i]
        wrongs.append((w, tag))
    wrongs.append((key[::-1], "ir2a-reverses-the-whole-order"))
    # most-tempting swaps first
    wrongs.sort(
        key=lambda t: (
            0
            if t[1]
            in (
                _SWAP_TAG[(ORGANISE, DISPLAY)],
                _SWAP_TAG[(DISPLAY, ANALYSE)],
                _SWAP_TAG[(ANALYSE, CONCLUDE)],
                _SWAP_TAG[(POSE, COLLECT)],
            )
            else 1
        )
    )
    return ordered_part(
        "Write the letters of the actions in the order the investigation would carry them out, separated by commas.",
        key,
        wrongs,
        marks=marks,
        working=[f"{', '.join(STAGE_NAME[s] for s in stages)}: {', '.join(key)}."],
    )


def _stage_part(rng, stages, chosen, letter_of, marks):
    stage = rng.choice([s for s in stages if s in (ORGANISE, DISPLAY, ANALYSE, CONCLUDE, COLLECT)])
    letter = letter_of[chosen[stage]]
    others = [s for s in STAGE_ORDER if s != stage]
    confuse = [s for s in _STAGE_CONFUSION[stage]]
    ordered = confuse + [s for s in others if s not in confuse]
    wrongs = [(STAGE_NAME[s], _CONFUSION_TAG.get(frozenset((stage, s)), "ir2a-names-the-wrong-stage")) for s in ordered]
    return mcq_part(
        f"Which stage of the investigation is action {letter}?",
        STAGE_NAME[stage],
        wrongs,
        marks=marks,
        working=[f"Action {letter} is where the investigation does this: {STAGE_NAME[stage].lower()}."],
        keep=4,
    )


def _stem(sc, name, lines, tail=""):
    return f"{name} is carrying out a statistical investigation. These are the actions, in no particular order. {lines}{tail}"


def _easy(rng):
    stages = (POSE, COLLECT, DISPLAY, ANALYSE, CONCLUDE)
    sc, chosen, bad, letter_of, lines = _scenario(rng, stages, False)
    name = rng.choice(NAMES)
    return single(_stem(sc, name, lines), _order_part(stages, chosen, letter_of, 2))


def _medium(rng):
    sc, chosen, bad, letter_of, lines = _scenario(rng, STAGE_ORDER, False)
    name = rng.choice(NAMES)
    return chain(
        _stem(sc, name, lines),
        _order_part(STAGE_ORDER, chosen, letter_of, 3),
        _stage_part(rng, STAGE_ORDER, chosen, letter_of, 1),
    )


def _hard(rng):
    sc, chosen, bad, letter_of, lines = _scenario(rng, STAGE_ORDER, True)
    name = rng.choice(NAMES)
    tail = " One of these actions would make the investigation unsound and should not be part of it."
    a = _order_part(STAGE_ORDER, chosen, letter_of, 3)
    a["prompt"] = (
        "Write the letters of the sound actions in the order the investigation would carry them out, leaving out the unsound action."
    )
    keyl = a["answer"]
    wrongs = [
        (keyl[:i] + [letter_of[bad]] + keyl[i:], "ir2a-keeps-the-unsound-action")
        for i in rng.sample(range(len(keyl) + 1), 2)
    ]
    wrongs += [
        (keyl[:i] + [letter_of[bad]] + keyl[i + 1 :], "ir2a-keeps-the-unsound-action")
        for i in rng.sample(range(len(keyl)), 2)
    ]
    a = ordered_part(
        a["prompt"],
        a["answer"],
        wrongs + [(list(reversed(a["answer"])), "ir2a-reverses-the-whole-order")],
        marks=3,
        working=a["working"],
    )
    return chain(_stem(sc, name, lines, tail), a, _stage_part(rng, STAGE_ORDER, chosen, letter_of, 1))


@generator("ir2a_investigation_step_order", level=11, presets={t: dict(tier=t) for t in _TIERS})
def ir2a_investigation_step_order(rng: random.Random, tier: str = "easy") -> dict:
    check(tier in _TIERS, f"unknown tier {tier!r}")
    return {"easy": _easy, "medium": _medium, "hard": _hard}[tier](rng)


@generator("exd_x_ir2a_investigation_step_order", level=12, presets={"exam": dict(tier="exam")})
def exd_x_ir2a_investigation_step_order(rng: random.Random, tier: str = "exam") -> dict:
    """Order the stages, name one, then work out the first group's mean and the difference of the two group means."""
    check(tier == "exam", "only the exam tier exists")
    sc, chosen, bad, letter_of, lines = _scenario(rng, STAGE_ORDER, False)
    name = rng.choice(NAMES)
    n = 6
    lo, hi = sc["lo"], sc["hi"]
    for _ in attempts(500, "groups with whole, distinct means"):
        a = [rng.randint(lo, hi) for _ in range(n)]
        b = [rng.randint(lo, hi) for _ in range(n)]
        if sum(a) % n == 0 and sum(b) % n == 0 and sum(a) != sum(b):
            break
    ma, mb = sum(a) // n, sum(b) // n
    stem = _stem(
        sc,
        name,
        lines,
        f" The {n} results for each group, in {sc['unit']}, were: {sc['a']}: {', '.join(map(str, a))}; {sc['b']}: {', '.join(map(str, b))}.",
    )
    pa = _order_part(STAGE_ORDER, chosen, letter_of, 3)
    pb = _stage_part(rng, STAGE_ORDER, chosen, letter_of, 1)
    pc = number_part(
        prompt=f"What is the mean {sc['measure']} of the {sc['a']}?",
        exact=ma,
        places=0,
        marks=2,
        working=[f"{sum(a)} / {n} = {ma}."],
        wrong=[
            (sum(a), "ir2a-forgets-to-divide-by-the-count"),
            (sorted(a)[n // 2], "ir2a-takes-a-middle-value-for-the-mean"),
            (max(a), "ir2a-reports-the-largest-value"),
        ],
        sensible_range=(0, None),
    )
    pd = number_part(
        prompt="By how much do the two group means differ? Give the larger minus the smaller.",
        exact=abs(ma - mb),
        places=0,
        marks=2,
        working=[f"The other mean is {sum(b)} / {n} = {mb}.", f"{max(ma, mb)} - {min(ma, mb)} = {abs(ma - mb)}."],
        wrong=[
            (ma + mb, "ir2a-adds-instead-of-subtracting"),
            (abs(sum(a) - sum(b)), "ir2a-compares-totals-not-means"),
            (abs(max(a) - max(b)), "ir2a-compares-the-largest-values"),
        ],
        follow_through=f"Abs(part_c - {mb})",
        sensible_range=(0, None),
    )
    return chain(stem, pa, pb, pc, pd)
