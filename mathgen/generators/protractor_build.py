"""Measuring an angle with a protractor drawn on the figure.

``protractor_measure_angle`` shows a semicircular protractor laid on an angle AOB whose base
ray OA lies along the protractor's baseline, and asks for the size of the angle in whole
degrees (``answer: number``, exact). The protractor has an outer and an inner scale, each
0 to 180, and the learner must read the one whose 0 is on the base ray.

- easy: the base ray is on the outer scale's 0, the protractor is upright, and the angle is a
  multiple of 10, so the ray crosses a labelled tick.
- medium: the base ray may be on the inner scale's 0, where reading the other scale gives
  ``180 - angle``; the angle is a multiple of 5 so it can fall between two labelled ticks.
- hard: the protractor is turned (the baseline is not horizontal), and a third of the
  questions mark the reflex angle, whose size is ``360`` minus what the scale reads.

Distractors are the readings a named mistake gives: the other scale (``180 - x``), one tick out,
the smaller angle of a reflex pair, and the other scale applied to a reflex angle.
The registered reader works from the figure alone.
"""

from __future__ import annotations

from mathgen import figures
from mathgen.core import check, generator, pick_distractors
from mathgen.figure_kinds import measure

_TIERS = ("easy", "medium", "hard")
_PRESETS = {t: {"tier": t} for t in _TIERS}

_POOLS = {
    "easy": (
        (
            "A protractor is placed on the angle AOB.",
            "The diagram shows angle AOB with a protractor on it.",
            "Here is a protractor lying along the arm OA.",
            "Angle AOB is drawn with a protractor over it.",
            "A protractor sits on the corner of angle AOB.",
            "The arm OA runs along the bottom of a protractor.",
        ),
        (
            "Measure angle AOB.",
            "What is the size of angle AOB, in degrees?",
            "Use the protractor to measure angle AOB, in degrees.",
            "How many degrees is angle AOB?",
        ),
    ),
    "medium": (
        (
            "Angle AOB is shown with a protractor lined up on the arm OA.",
            "The arm OA lies along the base of the protractor.",
            "A protractor has been placed so that the arm OA is on its base line.",
            "Look at the protractor placed on angle AOB.",
            "A protractor is lined up with OA, one arm of angle AOB.",
            "The base line of the protractor lies along OA.",
        ),
        (
            "Read the size of angle AOB from the correct scale, in degrees.",
            "What is angle AOB, in degrees?",
            "Find the size of angle AOB in degrees.",
            "Measure angle AOB and give its size in degrees.",
        ),
    ),
    "hard": (
        (
            "The protractor has been turned to line up with the arm OA.",
            "A turned protractor has its base line along the arm OA.",
            "Here a protractor is set on a tilted angle AOB.",
            "The page is not level: the protractor sits along the arm OA.",
            "This protractor is tilted, with its base line on OA.",
            "The protractor is at a slant, lined up with the arm OA.",
        ),
        (
            "Find the size of angle AOB, in degrees.",
            "How many degrees is angle AOB?",
            "Use the protractor to find angle AOB in degrees.",
            "Read the size of angle AOB in degrees.",
        ),
    ),
}
_REFLEX_POOLS = (
    (
        "The reflex angle AOB is marked on the diagram, and a protractor lines up with the arm OA.",
        "A protractor lies along OA. The arc shows the reflex angle AOB.",
        "The protractor is placed on OA, and the marked angle AOB goes the long way round.",
    ),
    (
        "Find the size of the marked reflex angle AOB, in degrees.",
        "What is the size of the reflex angle AOB, in degrees?",
        "How many degrees is the reflex angle AOB marked on the diagram?",
    ),
)


def _pick(rng, pool):
    return pool[rng.randrange(len(pool))]


def _draw(rng, tier: str):
    if tier == "easy":
        angle = _pick(rng, [a for a in range(20, 161, 10) if a != 90])
        zero, rotation, reflex = "outer", 0, False
    elif tier == "medium":
        angle = _pick(rng, [a for a in range(15, 166, 5) if a != 90])
        zero, rotation, reflex = _pick(rng, ("outer", "inner")), 0, False
    else:
        angle = _pick(rng, [a for a in range(15, 166, 5) if a != 90])
        zero = _pick(rng, ("outer", "inner"))
        rotation = 15 * (1 + rng.randrange(23))  # 15..345, never level
        reflex = rng.randrange(3) == 0
    return angle, zero, rotation, reflex


def _distractors(reading: int, answer: int, tier: str, reflex: bool) -> list[int]:
    step = 10 if tier == "easy" else 5
    if reflex:
        cands = [reading, 180 + reading, answer + step, answer - step, 360 - answer + step]
    else:
        cands = [180 - reading, reading + step, reading - step, 180 - reading + step, 180 - reading - step]
    return [c for c in cands if 0 < c < 360 and c != answer]


def _ordered(rng, cands: list[int]) -> list[int]:
    """The wrong-scale (first) candidate stays; the others are shuffled so the key is not always
    the number between two neighbours."""
    rest = cands[1:]
    rng.shuffle(rest)
    return cands[:1] + rest


@generator("protractor_measure_angle", level=5, presets=_PRESETS)
def protractor_measure_angle(rng, tier="easy"):
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    reading, zero, rotation, reflex = _draw(rng, tier)
    answer = 360 - reading if reflex else reading
    fig = measure.protractor(
        angle=reading,
        rotation=rotation,
        zero_on=zero,
        reflex=reflex,
        alt="a protractor on an angle",
        names={"vertex": "O", "base": "A", "second": "B"},
    )
    check(measure.read_protractor(fig) == answer, "the figure does not show the keyed angle")
    answer_text = str(answer)
    distractors = pick_distractors(
        answer_text, [str(c) for c in _ordered(rng, _distractors(reading, answer, tier, reflex))], n=3
    )
    check(len(distractors) == 3, "the question needs three distinct distractors")
    figures.assert_no_leak(fig, [answer_text, *distractors])
    pools = _REFLEX_POOLS if reflex else _POOLS[tier]
    text = " ".join(_pick(rng, p) for p in pools)
    if reflex:
        working = [
            f"The scale that starts at 0 on OA reads {reading} degrees for the angle inside; "
            f"the reflex angle is 360 - {reading} = {answer} degrees."
        ]
    else:
        working = [f"Read the scale whose 0 is on the arm OA: the arm OB crosses it at {reading} degrees."]
    return dict(
        text=text,
        answer=answer,
        answer_text=answer_text,
        answer_latex=f"plain:{answer_text}",
        distractors=distractors,
        working=working,
        figure=fig,
    )


def _read_angle(question) -> str:
    return str(measure.read_protractor(question.figure))


figures.register_reader("protractor_measure_angle", _read_angle)
