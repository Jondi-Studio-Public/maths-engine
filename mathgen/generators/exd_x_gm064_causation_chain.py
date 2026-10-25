"""Exam chain for causation versus correlation (``senior-distinguish-between-causality-and-correlation-through-examples``,
ACMGM064): the coefficient of determination, the variation it leaves unexplained, whether the study design supports
causation, and the conclusion the study justifies.

A study of two variables reports a correlation coefficient r. Set the way a school exam sets it:

(a) the coefficient of determination r^2 as a percentage            number, 2 marks
(b) the percentage of variation not explained by the explanatory variable
                                                                    number, 1 mark, follow-through (a)
(c) whether the way the data were collected supports causation       mcq,    1 mark
(d) which conclusion about the two variables the study justifies     mcq,    2 marks

## Construction

The design is drawn first: a controlled experiment (participants randomly assigned to levels of the explanatory
variable) or an observational survey that names a third variable associated with both. Each variable pair belongs to
one design only, because an experiment needs a variable a researcher can assign and a survey needs a pair whose
association is real but not causal, with the third variable that explains it. r has one decimal place so that
r^2 as a percentage is a whole number, and r = 0.5 is left out because then r as a percentage is exactly half of
(b)'s distractor set and collides with it. The sign of r follows the pair. Both verdicts occur, and the reader
registered in the tests re-derives all four answers from the rendered text.

## Distractors, each one named error of its own part

* (a): r read as a percentage, the unexplained percentage, the unexplained percentage taken straight from r.
* (b): the explained percentage, r as a percentage, the unexplained percentage taken straight from r.
* (c): a strong correlation settles causation, a perfect correlation is needed, no study can show causation,
  measuring the third variable turns a survey into an experiment.
* (d): the wrong verdict, the wrong percentage, r^2 read as a share of the causes.
"""

from __future__ import annotations

from fractions import Fraction

from mathgen.core import attempts, check, generator, phrase
from mathgen.generators._exd_x_w12a_kit import mcq_part, number_part

TOPIC = "exd_x_gm064_causation_chain"

# Pairs a researcher can set the level of: (explanatory variable, response, who took part, sign of r).
_EXPERIMENTS = (
    ("daily hours of study", "exam score", "Year 12 students", 1),
    ("hours of sleep each night", "reaction time", "volunteers", -1),
    ("hours of exercise each week", "resting heart rate", "adults", -1),
    ("daily typing practice time", "typing speed", "office workers", 1),
    ("the amount of fertiliser applied", "crop yield", "wheat plots", 1),
    ("weekly hours of tutoring", "maths test score", "Year 9 students", 1),
    ("the dose of a new medication", "recovery time", "patients", -1),
    ("weekly training hours", "100 m sprint time", "junior athletes", -1),
    ("minutes of reading each night", "reading speed", "Year 5 students", 1),
    ("hours of spaced revision", "number of vocabulary words recalled", "Year 8 students", 1),
    ("hours of daily sunlight", "height of tomato plants", "seedlings", 1),
    ("practice shots taken each week", "free-throw success rate", "netball players", 1),
    ("minutes of mindfulness each day", "stress score", "university students", -1),
    ("hours of screen time before bed", "time taken to fall asleep", "teenagers", 1),
)

# Pairs that can only be observed, each with the third variable that explains the association: (variable, response,
# the units observed, sign of r, third variable). Every association is real but neither variable causes the other.
_SURVEYS = (
    ("ice-cream sales", "the number of swimmers", "summer days", 1, "the daily temperature"),
    ("shoe size", "reading score", "primary school students", 1, "the child's age"),
    ("the number of fire trucks at a fire", "the amount of damage", "house fires", 1, "the size of the fire"),
    ("household heater use", "the number of colds reported", "suburbs", 1, "the time of year"),
    ("the number of lifeguards on duty", "the number of rescues", "beaches", 1, "the number of swimmers"),
    ("sunscreen sales", "the number of sunburn cases", "weeks", 1, "the UV index"),
    ("umbrella sales", "the number of traffic accidents", "days", 1, "the amount of rainfall"),
    ("arm length", "maths test score", "children aged 5 to 15", 1, "the child's age"),
    ("the number of books at home", "Year 9 maths score", "families", 1, "the parents' level of education"),
    ("the number of doctors in a town", "the number of illnesses reported", "towns", 1, "the town's population"),
    ("the number of pubs in a town", "the number of churches", "towns", 1, "the town's population"),
    ("jumper sales", "sales of cold drinks", "weeks", -1, "the temperature"),
    ("hot drink sales", "ice-cream sales", "days", -1, "the daily temperature"),
)

# r has one decimal place, so r^2 as a percentage is a whole number. 0.5 would make r as a percentage equal 2 x r^2.
_TENTHS = (3, 4, 6, 7, 8, 9)

_N = (24, 30, 36, 40, 48, 50, 60, 80, 120)

_LEAD_EXPERIMENT = (
    "A study of {n} {subj} looked at {x} and {y}. Researchers randomly assigned each one to a different level of "
    "{x}, then measured {y}. The correlation coefficient between {x} and {y} was r = {r}.",
    "{n} {subj} took part in an experiment. The level of {x} given to each was chosen at random, and {y} was then "
    "measured. The correlation coefficient was r = {r}.",
)
_LEAD_SURVEY = (
    "Researchers surveyed {n} {subj} and recorded {x} and {y} for each. They did not control either variable, and "
    "{z} is associated with both. The correlation coefficient between {x} and {y} was r = {r}.",
    "In an observational study of {n} {subj}, {x} and {y} were recorded and left as they were found. A third "
    "variable, {z}, is also associated with both. The correlation coefficient was r = {r}.",
)

_ASK_A = (
    "Take {x} as the explanatory variable. Find the coefficient of determination as a percentage.",
    "Calculate the coefficient of determination, r², as a percentage.",
)
_ASK_B = (
    "What percentage of the variation in {y} is not explained by the variation in {x}?",
    "Find the percentage of the variation in {y} that is not explained by {x}.",
)
_ASK_C = (
    "Does the way the data were collected support the claim that a change in {x} causes a change in {y}?",
    "Can the researchers conclude from the way this study was run that a change in {x} brings about a change in {y}?",
)
_ASK_D = (
    "Which conclusion is justified by this study?",
    "Which statement about {x} and {y} is best supported by the study?",
)


def cap(text: str) -> str:
    return text[0].upper() + text[1:]


@generator(TOPIC, level=13, presets={"exam": {"tier": "exam"}})
def exd_x_gm064_causation_chain(rng, tier: str = "exam"):
    """r², the unexplained variation, whether the design supports causation, and the justified conclusion."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(200, "a causation chain with three distractors per part"):
        experiment = rng.random() < 0.5
        tenths = rng.choice(_TENTHS)
        n = rng.choice(_N)
        if experiment:
            x, y, subj, sign = rng.choice(_EXPERIMENTS)
            z = ""
            lead = phrase(rng, _LEAD_EXPERIMENT, n=n, subj=subj, x=x, y=y, r="{r}")
        else:
            x, y, subj, sign, z = rng.choice(_SURVEYS)
            lead = phrase(rng, _LEAD_SURVEY, n=n, subj=subj, x=x, y=y, z=z, r="{r}")
        r_text = f"{'-' if sign < 0 else ''}0.{tenths}"
        lead = lead.replace("{r}", r_text)
        r_pct = tenths * 10  # |r| as a percentage
        explained = int(Fraction(tenths, 10) ** 2 * 100)
        unexplained = 100 - explained
        check(explained == tenths * tenths, "r squared as a percentage is not whole")

        part_a = number_part(
            phrase(rng, _ASK_A, x=x),
            explained,
            2,
            [
                f"r² = ({r_text})² = {tenths * tenths / 100:g}, so {explained}% of the variation in {y} is explained by {x}."
            ],
            [
                (r_pct, "cc-r-for-r-squared"),
                (unexplained, "cc-unexplained-for-explained"),
                (100 - r_pct, "cc-unexplained-from-r"),
            ],
        )
        part_b = number_part(
            phrase(rng, _ASK_B, x=x, y=y),
            unexplained,
            1,
            [f"The variation not explained is 100% - {explained}% = {unexplained}%."],
            [
                (explained, "cc-unexplained-for-explained"),
                (r_pct, "cc-r-for-r-squared"),
                (100 - r_pct, "cc-unexplained-from-r"),
            ],
            follow="100-part_a",
        )

        cx, cy = cap(x), cap(y)
        if experiment:
            c_key = "Yes: the levels of the variable were randomly assigned, so a causal claim is possible"
            c_wrong = [
                (
                    "No: a correlation coefficient smaller than 1 in size can never show causation",
                    "cc-no-cause-without-perfect-r",
                ),
                (
                    "No: no study design can ever show that one variable causes the other",
                    "cc-experiment-cannot-show-cause",
                ),
                (
                    "Yes: a strong correlation coefficient shows cause, whatever the design",
                    "cc-large-r-proves-cause",
                ),
            ]
            c_note = "Random assignment controls other variables, so the experiment can support a causal claim."
            d_key = f"Changes in {x} cause changes in {y}, and {explained}% of the variation in {y} is explained by {x}"
            d_wrong = [
                (
                    f"{cx} and {y} are only associated, because r is not 1 or -1",
                    "cc-no-cause-without-perfect-r",
                ),
                (
                    f"Changes in {x} cause changes in {y}, and {unexplained}% of the variation in {y} is explained by {x}",
                    "cc-unexplained-for-explained",
                ),
                (
                    f"{cx} and {y} are only associated, because a correlation never shows cause",
                    "cc-experiment-cannot-show-cause",
                ),
                (
                    f"Changes in {x} cause {explained}% of the changes in {y}, and the other {unexplained}% have no cause",
                    "cc-r-squared-as-causal-share",
                ),
            ]
            d_note = f"The design supports causation and r² = {explained}%, so the first statement is justified."
        else:
            c_key = f"No: the data were only observed, so this shows an association and {z} may explain it"
            c_wrong = [
                (
                    "Yes: a correlation coefficient this large shows that one variable causes the other",
                    "cc-large-r-proves-cause",
                ),
                (
                    "No: causation needs a correlation coefficient of exactly 1 or -1, which this is not",
                    "cc-no-cause-without-perfect-r",
                ),
                (
                    f"Yes: the researchers also measured {z}, so its effect has been controlled here",
                    "cc-measured-confounder-fixes-design",
                ),
            ]
            c_note = f"Nobody assigned the levels of {x} at random, and {z} could explain the association."
            d_key = (
                f"{cx} and {y} are associated, and {explained}% of the variation in {y} is explained by {x}, "
                f"but a causal link cannot be concluded because {z} may explain both"
            )
            d_wrong = [
                (
                    f"Changes in {x} cause changes in {y}, and {explained}% of the variation in {y} is explained by {x}",
                    "cc-large-r-proves-cause",
                ),
                (
                    f"{cx} and {y} are associated, and {unexplained}% of the variation in {y} is explained by {x}, "
                    f"but a causal link cannot be concluded because {z} may explain both",
                    "cc-unexplained-for-explained",
                ),
                (
                    f"Changes in {x} cause {explained}% of the changes in {y}, and {z} causes the rest",
                    "cc-r-squared-as-causal-share",
                ),
                (
                    f"Changes in {x} cause changes in {y}, because {z} was measured and so is controlled",
                    "cc-measured-confounder-fixes-design",
                ),
            ]
            d_note = f"The study is observational, so only an association can be concluded; r² = {explained}%."
        part_c = mcq_part(phrase(rng, _ASK_C, x=x, y=y), c_key, 1, [c_note], c_wrong)
        part_d = mcq_part(phrase(rng, _ASK_D, x=x, y=y), d_key, 2, [d_note], d_wrong)
        if None in (part_a, part_b, part_c, part_d):
            continue
        return dict(text=lead, parts=[part_a, part_b, part_c, part_d])
