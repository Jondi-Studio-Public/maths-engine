"""Classifying an angle read from a figure.

``classify_angle_protractor`` shows a protractor laid on an angle AOB and asks which type the marked angle is:
acute, right, obtuse, straight or reflex. The size is never printed, so the learner reads the scale and then
compares the reading with the 90 and 180 boundaries. All five names are always offered, so the key is never
the odd one out.

- easy: the protractor is upright and the angle a multiple of 10, so the ray crosses a labelled tick. The angle
  is well inside its class.
- medium: the angle is a multiple of 5 within 15 degrees of 90 or 180 (a quarter are the reflex angle just past 180),
  and either scale may start at the base ray.
- hard: the protractor is turned and the marked angle may be the reflex one.

The protractor figure cannot show exactly 90 or 180 degrees (both scales agree at 90, and 180 is a straight
line), so those two classes come from the number-only ``classify_angle`` source. The registered reader works
out the class from the figure alone.
"""

from __future__ import annotations

from mathgen import figures
from mathgen.core import check, generator, latex_answer, pick_distractors
from mathgen.figure_kinds import measure

from .curriculum_expansion import angle_type

_NAMES = ("Acute", "Right", "Obtuse", "Straight", "Reflex")
_PRESETS = {t: {"tier": t} for t in ("easy", "medium", "hard")}

_INTROS = (
    "A protractor is placed on the angle AOB.",
    "The diagram shows angle AOB with a protractor on it.",
    "Angle AOB is drawn with a protractor lined up on the arm OA.",
    "A protractor sits on the corner of angle AOB.",
)
_ASKS = (
    "What type of angle is the marked angle AOB?",
    "Is the marked angle AOB acute, right, obtuse, straight or reflex?",
    "Classify the marked angle AOB.",
    "Name the type of the marked angle AOB.",
)


def _draw(rng, tier):
    if tier == "easy":
        angle = rng.choice([a for a in range(20, 161, 10) if abs(a - 90) >= 20])
        return angle, "outer", 0, False
    if tier == "medium":
        if rng.randrange(4) == 0:  # the long way round a protractor's 170: 190 or 195, just past straight
            return rng.choice((165, 170)), rng.choice(("outer", "inner")), 0, True
        angle = rng.choice((75, 80, 85, 95, 100, 105, 160, 165, 170))
        return angle, rng.choice(("outer", "inner")), 0, False
    angle = rng.choice([a for a in range(15, 166, 5) if a != 90])
    reflex = rng.randrange(3) == 0
    return angle, rng.choice(("outer", "inner")), 15 * (1 + rng.randrange(23)), reflex


@generator("classify_angle_protractor", level=5, presets=_PRESETS)
def classify_angle_protractor(rng, tier="easy"):
    reading, zero, rotation, reflex = _draw(rng, tier)
    size = 360 - reading if reflex else reading
    fig = measure.protractor(
        angle=reading,
        rotation=rotation,
        zero_on=zero,
        reflex=reflex,
        alt="a protractor on an angle",
        names={"vertex": "O", "base": "A", "second": "B"},
    )
    check(measure.read_protractor(fig) == size, "the figure does not show the keyed angle")
    answer = angle_type(size)
    # 90 and 180 cannot be drawn, so the figure never has a Right or Straight key.
    check(answer in ("Acute", "Obtuse", "Reflex"), "a protractor figure cannot show a right or straight angle")
    distractors = pick_distractors(answer, [x for x in _NAMES if x != answer], n=4)
    check(len(distractors) == 4, "all five names must be offered")
    figures.assert_no_leak(fig, [answer, *distractors])
    return dict(
        text=f"{rng.choice(_INTROS)} {rng.choice(_ASKS)}",
        answer=answer,
        answer_text=answer,
        answer_latex=latex_answer(answer, plain=True),
        distractors=distractors,
        working=[f"The angle reads {size} degrees on the scale, so it is {answer.lower()}."],
        figure=fig,
    )


def _read_class(question) -> str:
    return angle_type(measure.read_protractor(question.figure))


figures.register_reader("classify_angle_protractor", _read_class)
