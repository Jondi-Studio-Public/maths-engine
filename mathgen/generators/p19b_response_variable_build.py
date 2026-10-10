"""Generates questions that choose the explanatory and response variables from a study context, as a four-option
`mcq`.

Scope: given a stated study context that measures an association between two
variables and also names a third recorded quantity that is not one of the pair, choose
from offered ordered pairs the one that names the explanatory variable and the response
variable.

`answer: mcq`, ALWAYS four options (the key and three distractors), each an ordered pair
rendered as plain text `Explanatory variable: <E>; response variable: <R>`. Every draw
names three per-case variables of one scenario: E (explanatory), R (response) and Z (a
third recorded quantity). Z is only ever recorded neutrally (in the recorded list and, on
the exam tier, as the colour / label key of the points, and on the hard tier as the split of
a second analysis): the stimulus NEVER says Z is irrelevant, excluded or a label. The pair
the question is about is identified only by the aim / analysis sentence (and the question
sentence), so the learner must find which two variables that sentence is about.

The distractor classes are

    S = (R, E) role swap       A = (E, Z)     B = (Z, R)     C = (Z, E)     D = (R, Z)

A class set is drawn per question: S is present on `S_SHARE` of draws (with two of A-D),
otherwise three of A-D. The shares were tuned against three "consensus attacks" (score
each option by how many other options share its variables, its explanatory column value or
its response column value; the top scorer must be the key on <= 40% of draws), which the
build test measures at 200 seeds per tier. Name lengths of E, R and Z vary from scenario to
scenario (Z longer than both, shorter than both, or between) so that the key's length rank
is spread.

The words "explanatory" and "response" appear only in the tier's question sentence and in
the options; the stimulus never uses them, nor "dependent", "independent" or "depend*".
Direction is carried by verbs, the analysis or the axes:
    easy   -- the researcher sets the level of E and measures R (manipulation verbs);
    medium -- observational; an aim with an influence verb (E influences R / R is influenced by E);
    hard   -- two analyses: the sample is split by Z to compare the mean R of the subgroups, AND a
              line of best fit is used to predict R from E; the question sentence asks about the
              association the line of best fit describes, so the learner picks the right analysis
              and applies predicted quantity = response;
    exam   -- scatterplot axes (E horizontal, R vertical, points coloured or labelled by Z) and a
              stated purpose of fitting a least-squares line for making predictions, with NO
              statement of which quantity is predicted: the learner applies horizontal =
              explanatory, vertical = response.
The recorded-list sentence names the three variables in a uniformly random permutation. Where the
sentences sit varies so that neither the first- nor the last-mentioned variable is a tell:
easy and medium use (list, aim), (list, aim, neutral Z note) or (aim, list); exam puts the colour/label
sentence before the axes sentence on ~50% of draws and moves the list to the end on half the rest, so Z is
the last-mentioned variable on about a third of draws. Medium adds one template with no influence verb
("how R changes according to E").
"""

from __future__ import annotations

from ..core import check, generator, latex_answer, phrase

# Probability that the role-swap distractor S is among the three distractors.
S_SHARE = 0.55
# Exam: share of draws where the colour/label sentence comes before the axes sentence; otherwise, the share of
# draws where the recorded list is moved to the end (so Z is last-mentioned only ~1/3 of the time overall).
EXAM_Z_FIRST = 0.50
EXAM_LIST_LAST = 0.50


def _sc(e, r, z, unit, manipulable=False):
    return dict(e=e, r=r, z=z, unit=unit, manipulable=manipulable)


# Manipulable scenarios: E is something a researcher can set (used by easy, and by other tiers).
MANIPULABLE_SCENARIOS = (
    _sc("oven temperature", "risen cake height", "recipe followed for each cake", "cake", True),
    _sc("tyre pressure applied", "braking distance", "model of car under test", "car", True),
    _sc("amount of glue used", "joint breaking force", "species of timber in piece", "joint", True),
    _sc("fertiliser dose", "plant height after a month", "type of pot used", "plant", True),
    _sc("salt added to water", "boiling point", "material of the beaker", "beaker", True),
    _sc("length of a jump run-up", "distance jumped", "training squad", "jumper", True),
    _sc("number of tellers on duty", "queue waiting time", "location of the bank branch", "shift", True),
    _sc("monthly advertising spend", "monthly sales revenue", "region where shop trades", "shop", True),
    _sc("minutes of daily practice", "score in the skills test", "year level of each student", "student", True),
    _sc("speed of a conveyor belt", "parcels sorted per hour", "depot site", "shift", True),
    _sc("light intensity in the room", "seedling growth rate", "supplier of the seed batch", "seedling", True),
    _sc("amount of yeast added", "dough rise time", "flour brand", "loaf", True),
    _sc("volume of paint applied", "drying time", "colour of the paint", "wall", True),
    _sc("water temperature for brewing", "strength of the tea", "brand of tea bag", "cup", True),
    _sc("length of study session", "number of facts recalled", "time of day", "student", True),
    _sc("dose of plant food", "leaf count after three weeks", "window the plant sits by", "plant", True),
    _sc("wind speed in the tunnel", "drag force on the model", "shape of the model tested", "model", True),
    _sc("number of rest breaks given", "typing errors per page", "age group of each typist", "typist", True),
    _sc("amount of chlorine added", "bacteria count", "pool sampled from", "sample", True),
    _sc("angle of a ramp", "distance the trolley rolls", "trolley mass in grams", "trial", True),
    _sc("depth of seed planting", "days to sprout", "packet the seed came from", "seed", True),
    _sc("length of the pendulum string", "time of one swing", "mass of the bob", "pendulum", True),
    _sc("dose of caffeine", "reaction time in milliseconds", "volunteer's age in years", "volunteer", True),
    _sc("grinding time for coffee", "bitterness rating", "roast type", "batch", True),
    _sc("air pressure", "bounce height of the ball", "brand of the ball", "ball", True),
    _sc("heating time in the kiln", "hardness of the fired pot", "clay type used", "pot", True),
)

# Observational scenarios: E is observed, never set.
OBSERVATIONAL_SCENARIOS = (
    _sc("years of schooling completed", "annual income", "state of residence", "adult"),
    _sc("daily hours of sunshine", "ice cream sales", "beach location", "day"),
    _sc("engine capacity in litres", "fuel used per 100 km", "make of the car", "car"),
    _sc("age of a tree", "trunk circumference", "forest plot number", "tree"),
    _sc("number of bedrooms", "weekly rent", "postcode area of the flat", "flat"),
    _sc("weekly hours of paid work", "nightly hours of sleep", "school year level of student", "student"),
    _sc("distance from the city centre", "house sale price", "suburb the house stands in", "house"),
    _sc("daily bus travel time", "afternoon tiredness score", "class the student is in", "student"),
    _sc("daily fruit servings", "systolic blood pressure", "patient's age in completed years", "adult"),
    _sc("rainfall in the past month", "dam water level", "capacity of the dam", "dam"),
    _sc("hours of daily driving", "weekly petrol cost", "driver's age in years", "driver"),
    _sc("book length in pages", "printing cost per copy", "year of printing", "book"),
    _sc("hours of reading each day", "vocabulary test score", "age of child in months", "child"),
    _sc("weekly gym visits", "body fat percentage", "member's age", "member"),
    _sc("daily peak temperature", "household electricity use", "number of people in household", "household"),
    _sc("number of rostered staff", "orders filled per hour", "shift length in hours", "shift"),
    _sc("height of a sprinter", "race finishing time", "training group", "sprinter"),
    _sc("floor area of a house", "monthly heating bill", "number of people living there", "house"),
    _sc("hours of homework each night", "maths exam mark", "school they attend", "student"),
    _sc("age of the used car", "resale price", "colour of the car", "car"),
    _sc("steps walked each day", "resting heart rate", "occupation of each worker", "worker"),
    _sc("altitude of a town", "annual snowfall", "region of the state", "town"),
    _sc("number of years employed", "current salary", "industry of employment", "employee"),
    _sc("daily hours of screen use", "weekly headache count", "age band", "teenager"),
)

SCENARIOS = MANIPULABLE_SCENARIOS + OBSERVATIONAL_SCENARIOS

# One pool, sliced per tier: easy needs a manipulable E; medium and hard draw from different (but
# overlapping) slices; exam draws from a third.
TIER_POOLS = {
    "easy": MANIPULABLE_SCENARIOS,
    "medium": OBSERVATIONAL_SCENARIOS[:18] + MANIPULABLE_SCENARIOS[:8],
    "hard": OBSERVATIONAL_SCENARIOS[6:] + MANIPULABLE_SCENARIOS[8:16],
    "exam": OBSERVATIONAL_SCENARIOS + MANIPULABLE_SCENARIOS[20:],
}

_QUESTION = (
    "Which option correctly names the explanatory variable and the response variable "
    "for the association between the two variables the study is about?"
)
_QUESTION_HARD = (
    "Which option correctly names the explanatory variable and the response variable "
    "for the association that the line of best fit describes?"
)
_QUESTION_EXAM = (
    "Which option correctly names the explanatory variable and the response variable "
    "for the association shown by the two axes of the scatterplot?"
)
QUESTIONS = {"easy": _QUESTION, "medium": _QUESTION, "hard": _QUESTION_HARD, "exam": _QUESTION_EXAM}

_LIST_PATTERNS = (
    "A study recorded {a}, {b} and {c} for each {unit}.",
    "In a study, three quantities were recorded for each {unit}: {a}, {b} and {c}.",
    "The data set holds {a}, {b} and {c} for every {unit} in the study.",
)

_BANNED_ROOTS = ("explanat", "respon", "depend")
# Phrases that would label Z irrelevant / excluded; no stimulus may contain any of them.
Z_LABEL_PHRASES = (
    "plays no part",
    "only a",
    "excludes",
    "covariate",
    "background",
    "grouping label",
    "not part",
    "does not involve",
)

# A neutral note that Z was recorded (never labels Z irrelevant); used on easy and medium so that Z is sometimes
# the last-named variable and the recorded list does not always come first.
_Z_NOTE = (
    "The {Z} of each {unit} is also on record.",
    "{Zc} is noted for every {unit} as well.",
    "Each {unit}'s {Z} is written down too.",
)

# Easy: the researcher sets the level of E and measures R.
_EASY_E_FIRST = (
    "The researcher sets the level of {E} and measures {R}.",
    "The researcher chooses the level of {E} for each {unit}, then measures {R}.",
    "The researcher varies {E} on purpose and observes {R}.",
)
_EASY_R_FIRST = (
    "The researcher measures {R} after setting the level of {E}.",
    "The researcher measures {R} for each {unit}, having chosen the level of {E} in advance.",
    "The researcher observes {R} while deliberately varying {E}.",
)

# Medium: observational; the aim carries an influence verb.
_MED_R_FIRST = (
    "In an observational study, the researchers want to find out whether {R} is influenced by {E}.",
    "The researchers observe the {unit_pl} without changing anything, to see whether {R} is affected by {E}.",
    "In an observational study, the aim is to test whether {R} is influenced by {E}.",
    "In an observational study, the researchers describe how {R} changes according to {E}.",
)
_MED_E_FIRST = (
    "In an observational study, the researchers want to find out whether {E} influences {R}.",
    "The researchers observe the {unit_pl} without changing anything, to see whether {E} affects {R}.",
    "In an observational study, the aim is to test whether {E} has an influence on {R}.",
)

# Hard: two analyses. A line of best fit relates R and E; the sample is also split by Z.
_HARD_R_FIRST = (
    "A line of best fit is used to predict {R} from {E}.",
    "A line of best fit is fitted so that {R} can be estimated for a given value of {E}.",
    "A line of best fit gives {R} once the value of {E} is known.",
)
_HARD_E_FIRST = (
    "A line of best fit uses {E} to predict {R}.",
    "A line of best fit is fitted so that, given a value of {E}, {R} can be estimated.",
    "A line of best fit uses known values of {E} to estimate {R}.",
)
_HARD_GROUP = (
    "The {unit_pl} are also split into subgroups by {Z}, and the mean {R} of the subgroups is compared.",
    "The sample is also split by {Z} so that the mean {R} of each subgroup can be compared.",
    "Separately, the mean {R} is compared between the subgroups formed by {Z}.",
    "The mean {R} is also compared across the subgroups when the {unit_pl} are split by {Z}.",
    "The subgroups are also compared on their mean {R}, the {unit_pl} being split by {Z}.",
)

# Exam: scatterplot axis convention, a stated purpose of fitting a line, no statement of what is predicted.
_EXAM_E_FIRST = (
    "The scatterplot puts {E} on the horizontal axis and {R} on the vertical axis.",
    "The scatterplot has {E} along its horizontal axis and {R} along its vertical axis.",
    "The horizontal axis of the scatterplot shows {E}, and its vertical axis shows {R}.",
)
_EXAM_R_FIRST = (
    "The scatterplot puts {R} on the vertical axis and {E} on the horizontal axis.",
    "The scatterplot has {R} along its vertical axis and {E} along its horizontal axis.",
    "The vertical axis of the scatterplot shows {R}, and its horizontal axis shows {E}.",
)
_EXAM_Z = (
    "Each point of the scatterplot is {how} according to {Z}.",
    "In the scatterplot, points are {how} by {Z}.",
    "The points of the scatterplot are {how} according to {Z}.",
)
_EXAM_PURPOSE = (
    "A least-squares line will be fitted so that predictions can be made.",
    "The purpose is to fit a least-squares line and use it for making predictions.",
    "The points are to be used to fit a least-squares line for making predictions.",
)


def _cap(s: str) -> str:
    return s[0].upper() + s[1:] if s else s


def option_text(explanatory: str, response: str) -> str:
    return f"Explanatory variable: {explanatory}; response variable: {response}"


_IRREGULAR_PLURALS = {"person": "people", "child": "children", "loaf": "loaves"}


def _plural(unit: str) -> str:
    if unit in _IRREGULAR_PLURALS:
        return _IRREGULAR_PLURALS[unit]
    if unit.endswith(("ch", "sh", "s", "x")):
        return unit + "es"
    if unit.endswith("y") and unit[-2] not in "aeiou":
        return unit[:-1] + "ies"
    return unit + "s"


def _draw_classes(rng) -> list[str]:
    if rng.random() < S_SHARE:
        return ["S", *rng.sample("ABCD", 2)]
    return list(rng.sample("ABCD", 3))


def _class_pair(cls: str, e: str, r: str, z: str) -> tuple[str, str]:
    return {"S": (r, e), "A": (e, z), "B": (z, r), "C": (z, e), "D": (r, z)}[cls]


def _slots(rng, sc: dict) -> dict:
    a, b, c = rng.sample([sc["e"], sc["r"], sc["z"]], 3)
    return dict(
        E=sc["e"], R=sc["r"], Z=sc["z"], Zc=_cap(sc["z"]), unit=sc["unit"], unit_pl=_plural(sc["unit"]), a=a, b=b, c=c
    )


def _lead(rng, s: dict) -> str:
    return phrase(rng, _LIST_PATTERNS, **s)


# Layout draw for easy and medium: (recorded list, aim), (list, aim, Z note) or (aim, list).
LAYOUT_LIST_AIM = 0.30
LAYOUT_LIST_AIM_ZNOTE = 0.35


def _layout(rng, s, aim):
    lead = _lead(rng, s)
    note = phrase(rng, _Z_NOTE, **s)
    u = rng.random()
    if u < LAYOUT_LIST_AIM:
        parts = [lead, aim]
    elif u < LAYOUT_LIST_AIM + LAYOUT_LIST_AIM_ZNOTE:
        parts = [lead, aim, note]
    else:
        parts = [aim, lead]
    return " ".join(parts)


def _easy(rng, sc):
    s = _slots(rng, sc)
    direction = phrase(rng, _EASY_E_FIRST if rng.random() < 0.5 else _EASY_R_FIRST, **s)
    return _layout(rng, s, direction)


def _medium(rng, sc):
    s = _slots(rng, sc)
    direction = phrase(rng, _MED_R_FIRST if rng.random() < 0.5 else _MED_E_FIRST, **s)
    return _layout(rng, s, direction)


def _hard(rng, sc):
    s = _slots(rng, sc)
    line = phrase(rng, _HARD_R_FIRST if rng.random() < 0.5 else _HARD_E_FIRST, **s)
    group = phrase(rng, _HARD_GROUP, **s)
    analyses = [line, group] if rng.random() < 0.5 else [group, line]
    return " ".join([_lead(rng, s), *analyses])


def _exam(rng, sc):
    s = _slots(rng, sc)
    s["how"] = rng.choice(("coloured", "labelled"))
    axes = phrase(rng, _EXAM_E_FIRST if rng.random() < 0.5 else _EXAM_R_FIRST, **s)
    colour = phrase(rng, _EXAM_Z, **s)
    purpose = phrase(rng, _EXAM_PURPOSE, **s)
    lead = _lead(rng, s)
    if rng.random() < EXAM_Z_FIRST:
        parts = [lead, colour, axes, purpose]
    elif rng.random() < EXAM_LIST_LAST:
        parts = [axes, colour, purpose, lead]
    else:
        parts = [lead, axes, colour, purpose] if rng.random() < 0.5 else [lead, axes, purpose, colour]
    return " ".join(parts)


_BUILDERS = {"easy": _easy, "medium": _medium, "hard": _hard, "exam": _exam}


def _verify(text: str, sc: dict, key: str, distractors: list[str], tier: str = "easy") -> None:
    """In-generator check: the stimulus names all three variables verbatim, never uses the
    giveaway words or any phrase labelling Z irrelevant, the tier's question sentence ends the
    text, and exactly one of the four options is the (explanatory, response) pair the scenario
    defines."""
    question = QUESTIONS[tier]
    check(text.endswith(question), "the tier's question sentence must end the text")
    stimulus = text[: -len(" " + question)]
    for name in (sc["e"], sc["r"], sc["z"]):
        check(name in stimulus, f"variable {name!r} must appear verbatim in the stimulus")
    lowered = stimulus.lower()
    for root in _BANNED_ROOTS:
        check(root not in lowered, f"the stimulus must not contain {root!r}")
    check("independent" not in lowered, "the stimulus must not contain 'independent'")
    for phrase_ in Z_LABEL_PHRASES:
        check(phrase_ not in lowered, f"the stimulus must not label the third variable ({phrase_!r})")
    options = [key, *distractors]
    check(len(set(options)) == 4, "the four options must be pairwise distinct")
    matching = [o for o in options if o == option_text(sc["e"], sc["r"])]
    check(len(matching) == 1 and matching[0] == key, "exactly one option may be the true pair")
    check(len({sc["e"], sc["r"], sc["z"]}) == 3, "E, R and Z must be three different variables")


@generator(
    "p19b_response_variable",
    level=12,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p19b_response_variable(rng, tier="easy"):
    if tier not in _BUILDERS:
        raise ValueError(f"tier must be easy/medium/hard/exam; got {tier!r}")
    sc = rng.choice(TIER_POOLS[tier])
    stimulus = _BUILDERS[tier](rng, sc)
    text = f"{stimulus} {QUESTIONS[tier]}"

    classes = _draw_classes(rng)
    key = option_text(sc["e"], sc["r"])
    distractors = [option_text(*_class_pair(c, sc["e"], sc["r"], sc["z"])) for c in classes]
    _verify(text, sc, key, distractors, tier)

    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=key,
        answer_text=key,
        answer_latex=latex_answer(key, plain=True),
        working=[
            f"The question is about how {sc['r']} relates to {sc['e']}: {sc['e']} is the explanatory "
            f"variable and {sc['r']} is the response variable; {sc['z']} is not one of the two "
            f"variables the question is about."
        ],
        distractors=distractors,
    )
