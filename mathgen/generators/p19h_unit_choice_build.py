"""Generates `appropriate_activity_unit` and a mass-unit-choice skill: pick the appropriate unit for an activity
or a mass, as `mcq` questions.

Reworked: two unit-appropriateness mcq classifications -- picking the right unit for an activity's stated
measurement, and picking the right mass unit to avoid an awkward decimal or digit count.

``appropriate_activity_unit`` (``senior-identify-the-appropriate-units-for-
different-activities-such``, ACMEM085) -- ``mcq`` over the 4-key answer
space {m, km, km/h, m/s}, all four always offered. The correct key is drawn
UNIFORMLY first, then an item of that key's (type, scale) cell.

The m/s-vs-km/h key used to follow cue words in the measurement phrase ("burst", "gust", "sprint" -> m/s;
"average", "long", "cruising" -> km/h), and some keys contradicted real usage (a thrown cricket ball or tennis
serve keyed m/s). Now the SAME quantity phrase is asked with either speed unit, and every speed item names its
REPORTING CONVENTION, which alone settles the unit: a setting that works in SI base units -> m/s; a setting
written for a general audience -> km/h. Distance items are unambiguous by the object's real scale (a pool's
length in m, a marathon in km). Pools are widened, with three question frames, so a 200-seed sample holds
well over 150 distinct prompts per tier.

- easy: one quantity, no numbers; the object (distance) or the convention
  (speed) decides.
- medium: the activity name recurs across every cell, so it is uninformative
  (same shape as before, convention added for speed).
- hard: the quantity comes with a recorded NUMBER (no unit); the learner must
  judge which unit makes that number realistic (84 for a javelin throw is
  metres; 880 for a road trip is kilometres); speed still names its
  convention.
- exam: two recorded quantities of one activity (one distance, one speed),
  each with its number; the question says which of the two is graded.

``need_for_milligrams`` (``senior-recognise-the-need-for-milligrams``,
ACMEM099) -- ``mcq`` over {mg, g, kg, t}: which unit expresses the stated
mass as a whole number of no more than 3 digits, no decimal point. The
correct key K is drawn uniformly first; a mass whose ONLY such unit is K is
then built from a named real object whose mass is feasible.

- easy: one object, its mass stated in a unit ONE ladder step from K.
- medium: a COLLECTIVE -- a count of identical items, each
  with a stated mass; the total decides the unit. The per-item unit is the
  item's natural unit; the total either stays inside it (key equals the
  stated unit, genuinely) or crosses into the next unit up (12 tablets of
  250 mg is 3 g).
- hard: the collective plus a second, separately stated mass in a unit that
  must be judged, so the total needs both a multiplication and a common-unit
  sum.
- exam: two different objects, stated in units of their own, whose COMBINED
  mass is asked about.

The abstract "for consistency with a shared reporting system" sentence is
gone; every question names the real record it comes from (a pharmacy stock
sheet, a builder's docket, ...).
"""

from __future__ import annotations

from fractions import Fraction

from ..core import (
    check,
    check_conversion,
    check_pairwise_distinct,
    exact_decimal_string,
    generator,
    latex_answer,
)

# ===========================================================================
# Skill 1: appropriate_activity_unit (ACMEM085)
# ===========================================================================

_ACTIVITY_KEYS = ("m", "km", "km/h", "m/s")
_KEY_TYPE = {"m": "distance", "km": "distance", "km/h": "speed", "m/s": "speed"}
_KEY_SCALE = {"m": "fine", "km": "coarse", "km/h": "coarse", "m/s": "fine"}
# Independent reverse lookup -- constructed separately from _KEY_TYPE/_KEY_SCALE
# above so the in-generator check re-derives the key a second way rather than
# just reading the same two dicts back.
_TYPE_SCALE_TO_KEY = {
    ("distance", "coarse"): "km",
    ("distance", "fine"): "m",
    ("speed", "coarse"): "km/h",
    ("speed", "fine"): "m/s",
}

# The reporting convention that settles a speed item's unit. Neither list
# contains any cue word ("burst", "average", "long", ...) that could stand in
# for the convention.
_SI_CONVENTIONS = (
    "in a physics-class calculation that uses SI base units",
    "in a sports-science lab report that uses SI base units",
    "in a biomechanics paper that reports every quantity in SI base units",
    "in a university mechanics problem set that works in SI base units",
    "in a data table for a school physics experiment that uses SI base units",
    "in an engineering specification sheet that uses SI base units",
    "in a science-fair report that keeps every quantity in SI base units",
    "in a physics exam question where every quantity is in SI base units",
    "in a simulation's input file that expects SI base units",
)
_PUBLIC_CONVENTIONS = (
    "on a broadcast graphic aimed at general viewers",
    "in a news report written for the general public",
    "on a public information board for visitors",
    "in a club newsletter written for everyday readers",
    "on a tourist brochure for visitors",
    "in a magazine article for general readers",
    "in a match-day programme for spectators",
    "in a radio commentary for a general audience",
    "on a sports-app summary screen for everyday users",
)
_CONVENTIONS = {"m/s": _SI_CONVENTIONS, "km/h": _PUBLIC_CONVENTIONS}

_FRAMES = (
    "Which unit is the most appropriate for reporting {q}{c}?",
    "A report needs to state {q}{c}. Which unit is the most appropriate?",
    "Choose the most appropriate unit for {q}{c}.",
)

# easy: distance items, unambiguous from the object's real scale.
_EASY_DISTANCE = {
    "km": [
        "the total distance of a marathon",
        "the distance covered on a multi-day charity walk",
        "the length of a cross-country hiking trail",
        "the distance driven on a long outback road trip",
        "the distance flown between two capital cities on a domestic flight",
        "the distance covered on a week-long cycling tour",
        "the length of a coastal walking track",
        "the length of a cross-country skiing trail",
        "the length of a mountain bike race course",
        "the distance paddled on a multi-day canoe expedition",
        "the length of a river rafting run",
        "the distance covered on a horse-riding trek",
        "the length of an orienteering course",
        "the distance driven on a road cycling classic",
        "the length of a half-marathon course",
        "the length of one stage of a multi-day cycling race",
        "the distance between two country towns",
        "the length of a triathlon's cycling leg",
        "the length of a city bike-path loop",
        "the distance a marathon swimmer covers in an open-water crossing",
        "the length of a suburban train line",
        "the distance covered in a day of touring by car",
        "the length of an ultramarathon course",
        "the length of a national park's hiking circuit",
        "the distance flown by a long-haul airliner between Sydney and Singapore",
        "the length of a coastal cycling path",
        "the distance from a city to its regional airport",
        "the length of a school cross-country run",
        "the distance covered in a 24-hour relay running event",
        "the length of a scenic drive along a coastline",
    ],
    "m": [
        "the length of an Olympic swimming pool",
        "the distance of a long jump attempt",
        "the width of an athletics track lane",
        "the length of the home straight on an athletics track",
        "the distance a shot-putter throws the shot",
        "the height a pole vaulter clears",
        "the depth a scuba diver descends to",
        "the length of a skateboard half-pipe",
        "the distance a discus thrower's throw travels",
        "the length of a javelin throw",
        "the height a high jumper clears",
        "the length of a surfboard",
        "the width of a climbing wall's starting ledge",
        "the distance a hammer throw travels",
        "the length of a cricket pitch",
        "the length of a basketball court",
        "the width of a tennis court",
        "the height of a diving platform",
        "the length of a netball court",
        "the distance between hurdles on a hurdles track",
        "the length of a football field",
        "the length of a rowing eight boat",
        "the length of a kayak",
        "the length of a road bike",
        "the height of a basketball hoop",
        "the length of a sprint relay changeover zone",
        "the depth of a diving pool",
        "the length of a beach volleyball court",
        "the height of a climbing wall",
        "the length of a bowling green rink",
    ],
}

# easy: speed quantities (no convention here; the convention is drawn with them).
_EASY_SPEED = [
    "the speed of a thrown cricket ball",
    "the speed of a tennis serve at the moment of contact",
    "a sprinter's speed over the final stretch of a 100 metre race",
    "a swimmer's speed during a sprint lap",
    "the speed of a diver entering the water",
    "the speed of a downhill skier through a timed section",
    "the speed of a cyclist's sprint to the line",
    "the speed of a skateboarder leaving a half-pipe",
    "the speed of a paddler going through a short rapid",
    "the speed of a runner on the last lap of a mile race",
    "the speed of a rower in the final sprint of a race",
    "the speed of a baseball pitch",
    "the speed of a soccer ball in a penalty kick",
    "the speed of a gust of wind during a sailing race",
    "the wind speed recorded during an athletics meet",
    "the speed of a speedboat on a lake",
    "a cyclist's speed on a long touring ride",
    "the speed of a car on a highway",
    "the speed of an intercity train",
    "the speed of a hot-air balloon drifting with the wind",
    "the speed of a peloton on a road cycling classic",
    "the speed of a hiker on a flat trail",
    "the speed of a golf ball leaving the driver",
    "the average speed of a marathon's lead pack",
    "the top speed of a downhill skier",
    "the speed of a volleyball serve",
    "the speed of a mountain biker on a short, steep descent",
    "the speed of a hurdler clearing the final hurdle",
]

# medium/hard/exam: the SAME activity recurs across every cell. Each entry is
# (phrase, lo, hi, decimals): lo/hi are the realistic recorded values in the
# item's own unit (for a speed item, in m/s; km/h is derived as m/s x 3.6).
_ACTIVITIES = [
    {
        "name": "cycling",
        "km": [
            ("the distance a cyclist rides on a full day's tour", 80, 180, 0),
            ("the length of a stage in a multi-day cycling race", 120, 220, 0),
        ],
        "m": [("the width of a bike's handlebars", 0.4, 0.6, 1), ("the length of a road bike", 1.6, 1.8, 1)],
        "speed": [
            ("a cyclist's speed in a sprint finish", 12, 19, 0),
            ("a cyclist's speed climbing a mountain pass", 4, 6, 0),
        ],
    },
    {
        "name": "running",
        "km": [
            ("the distance a runner covers in a long training run", 15, 35, 0),
            ("the length of a city fun-run course", 5, 12, 0),
        ],
        "m": [
            ("the length of a single lap of an athletics track", 400, 400, 0),
            ("the length of a sprint relay changeover zone", 20, 20, 0),
        ],
        "speed": [
            ("a runner's speed over the final stretch of a 100 metre sprint", 9, 12, 0),
            ("a runner's speed during a steady training jog", 2.5, 3.5, 1),
        ],
    },
    {
        "name": "swimming",
        "km": [
            ("the distance a swimmer covers in an open-water race", 5, 10, 0),
            ("the distance swum in a cross-strait charity swim", 15, 40, 0),
        ],
        "m": [("the length of one lap of a swimming pool", 25, 50, 0), ("the depth of a diving pool", 4, 6, 0)],
        "speed": [
            ("a swimmer's speed in a sprint lap", 1.7, 2.3, 1),
            ("a swimmer's speed in a steady training set", 1.0, 1.5, 1),
        ],
    },
    {
        "name": "rowing",
        "km": [
            ("the distance a crew rows in a long training session", 12, 30, 0),
            ("the length of a river rowing race course", 4, 7, 0),
        ],
        "m": [("the length of a rowing eight boat", 17, 19, 0), ("the length of a single oar", 2.6, 2.9, 1)],
        "speed": [
            ("a rowing crew's speed in the final sprint", 5, 7, 0),
            ("a rowing crew's speed during a steady row", 3.5, 4.5, 1),
        ],
    },
    {
        "name": "driving",
        "km": [
            ("the distance covered on a long road trip", 300, 900, 0),
            ("the distance driven in a day's touring", 200, 500, 0),
        ],
        "m": [("the length of a car's parking space", 5, 6, 0), ("the length of a family car", 4.2, 4.9, 1)],
        "speed": [("a car's speed on a highway", 22, 33, 0), ("a car's speed on a suburban street", 11, 17, 0)],
    },
    {
        "name": "hiking",
        "km": [
            ("the distance covered on a multi-day hiking trip", 40, 120, 0),
            ("the length of a day-walk loop", 8, 20, 0),
        ],
        "m": [("the width of a narrow hiking trail", 0.5, 1.0, 1), ("the height of a boulder beside a trail", 1, 3, 0)],
        "speed": [
            ("a hiker's speed on a flat track", 1.2, 1.7, 1),
            ("a hiker's speed climbing a steep section", 0.5, 1.0, 1),
        ],
    },
    {
        "name": "skiing",
        "km": [
            ("the length of a cross-country ski race", 10, 50, 0),
            ("the distance a skier covers in a full day on the slopes", 20, 60, 0),
        ],
        "m": [("the width of a groomed ski run", 20, 50, 0), ("the vertical drop of a beginner slope", 60, 150, 0)],
        "speed": [
            ("a skier's speed through a timed downhill section", 20, 35, 0),
            ("a skier's speed on a gentle beginner slope", 4, 8, 0),
        ],
    },
    {
        "name": "kayaking",
        "km": [
            ("the distance a paddler covers on a multi-day kayaking trip", 60, 150, 0),
            ("the length of a sea-kayak day paddle", 12, 30, 0),
        ],
        "m": [("the length of a kayak", 3.0, 5.0, 1), ("the width of a narrow river channel", 5, 20, 0)],
        "speed": [
            ("a paddler's speed through a short rapid", 2, 5, 0),
            ("a paddler's speed on calm water", 1.0, 2.0, 1),
        ],
    },
    {
        "name": "climbing",
        "km": [
            ("the distance a climbing team walks approaching a base camp", 10, 40, 0),
            ("the length of a ridge traverse", 3, 9, 0),
        ],
        "m": [
            ("the height a climber ascends on a single pitch", 25, 60, 0),
            ("the length of a climbing rope", 50, 70, 0),
        ],
        "speed": [
            ("a climber's speed abseiling down a rope", 0.5, 1.5, 1),
            ("a climber's speed moving over an easy rocky section", 0.5, 1.2, 1),
        ],
    },
    {
        "name": "surfing",
        "km": [
            ("the length of a coastal drive between two surf breaks", 20, 90, 0),
            ("the distance from a beach car park to a remote surf break by track", 2, 6, 0),
        ],
        "m": [("the length of a surfboard", 1.8, 2.9, 1), ("the length of a single wave ride", 50, 200, 0)],
        "speed": [
            ("a surfer's speed riding down a wave face", 5, 9, 0),
            ("a surfer's speed paddling out", 1.0, 2.0, 1),
        ],
    },
    {
        "name": "mountain biking",
        "km": [
            ("the distance covered on a multi-day mountain-bike tour", 100, 300, 0),
            ("the length of a cross-country race course", 25, 45, 0),
        ],
        "m": [("the length of a mountain bike", 1.6, 1.9, 1), ("the width of a narrow singletrack", 0.4, 0.8, 1)],
        "speed": [
            ("a rider's speed on a steep descent", 8, 14, 0),
            ("a rider's speed on a rough uphill climb", 2, 4, 0),
        ],
    },
    {
        "name": "orienteering",
        "km": [
            ("the length of a long-distance orienteering course", 6, 15, 0),
            ("the total distance of a 24-hour orienteering rogaine", 30, 70, 0),
        ],
        "m": [
            ("the width of a narrow forest track", 1.0, 2.0, 1),
            ("the distance between two nearby checkpoints", 80, 300, 0),
        ],
        "speed": [
            ("a competitor's speed on a clear forest track", 3.0, 4.5, 1),
            ("a competitor's speed pushing through thick bush", 0.5, 1.2, 1),
        ],
    },
]

_UNIT_WORD_1 = {"m": "metres", "km": "kilometres", "km/h": "kilometres per hour", "m/s": "metres per second"}

_SPEED_KEYS = ("km/h", "m/s")


def _conv_suffix(rng, key: str) -> str:
    return " " + rng.choice(_CONVENTIONS[key])


def _draw_value(rng, lo, hi, decimals: int) -> Fraction:
    scale = 10**decimals
    return Fraction(rng.randint(round(lo * scale), round(hi * scale)), scale)


def _speed_value_text(rng, key: str, lo, hi, decimals: int) -> str:
    """A realistic recorded speed in `key`'s own unit. The m/s value is drawn
    from the item's realistic range; the km/h value is that same speed x 3.6
    rounded to a whole number, re-checked against the m/s value."""
    ms = _draw_value(rng, lo, hi, decimals)
    if key == "m/s":
        return exact_decimal_string(ms, 1)
    kmh = round(ms * Fraction(18, 5))
    check(abs(Fraction(kmh) - ms * Fraction(18, 5)) <= Fraction(1, 2), "km/h value drifted from the m/s value")
    check(kmh >= 1, "km/h value below 1")
    return str(kmh)


def _build_activity_easy(rng) -> tuple[str, str]:
    key = rng.choice(_ACTIVITY_KEYS)
    if _KEY_TYPE[key] == "distance":
        q, conv = rng.choice(_EASY_DISTANCE[key]), ""
    else:
        q, conv = rng.choice(_EASY_SPEED), _conv_suffix(rng, key)
    return key, rng.choice(_FRAMES).format(q=q, c=conv)


def _build_activity_medium(rng) -> tuple[str, str]:
    key = rng.choice(_ACTIVITY_KEYS)
    act = rng.choice(_ACTIVITIES)
    cell = act["km" if key == "km" else "m" if key == "m" else "speed"]
    q = rng.choice(cell)[0]
    conv = "" if _KEY_TYPE[key] == "distance" else _conv_suffix(rng, key)
    return key, rng.choice(_FRAMES).format(q=f"{q} (during {act['name']})", c=conv)


def _recorded_clause(rng, act: dict, key: str, verb: str) -> str:
    """`<phrase> <verb> <number> [<convention>]` for one item of `key`'s cell."""
    if key in ("m", "km"):
        phrase, lo, hi, d = rng.choice(act[key])
        v = exact_decimal_string(_draw_value(rng, lo, hi, d), 1)
        return f"{phrase} {verb} {v}"
    phrase, lo, hi, d = rng.choice(act["speed"])
    v = _speed_value_text(rng, key, lo, hi, d)
    return f"{phrase} {verb} {v}{_conv_suffix(rng, key)}"


def _build_activity_hard(rng) -> tuple[str, str]:
    key = rng.choice(_ACTIVITY_KEYS)
    act = rng.choice(_ACTIVITIES)
    clause = _recorded_clause(rng, act, key, "is recorded as")
    return key, f"During {act['name']}, {clause}. Which unit is the most appropriate for that quantity?"


def _build_activity_exam(rng) -> tuple[str, str]:
    key = rng.choice(_ACTIVITY_KEYS)
    act = rng.choice(_ACTIVITIES)
    graded_type = _KEY_TYPE[key]
    other_type = "speed" if graded_type == "distance" else "distance"
    other_key = rng.choice([k for k in _ACTIVITY_KEYS if _KEY_TYPE[k] == other_type])
    graded = _recorded_clause(rng, act, key, "is")
    other = _recorded_clause(rng, act, other_key, "is")
    if rng.choice([True, False]):
        first, second, label = graded, other, "first"
    else:
        first, second, label = other, graded, "second"
    text = (
        f"During {act['name']}, two quantities are recorded. First, {first}. Second, {second}. "
        f"Considering only the {label} of these two, which unit is the most appropriate?"
    )
    return key, text


def _build_appropriate_activity_unit(rng, tier: str) -> dict:
    if tier == "easy":
        key, text = _build_activity_easy(rng)
    elif tier == "medium":
        key, text = _build_activity_medium(rng)
    elif tier == "hard":
        key, text = _build_activity_hard(rng)
    elif tier == "exam":
        key, text = _build_activity_exam(rng)
    else:
        raise ValueError(f"unknown tier {tier!r}")
    options = list(_ACTIVITY_KEYS)

    # Independent re-derivation: recompute the key from (type, scale) via a
    # SEPARATE lookup table rather than trusting the pool's own labelling.
    expected_key = _TYPE_SCALE_TO_KEY[(_KEY_TYPE[key], _KEY_SCALE[key])]
    check(expected_key == key, f"activity-unit key mismatch: built {key!r}, re-derived {expected_key!r}")
    # A speed item always carries the convention that settles its unit; a
    # distance item never does.
    if key in _SPEED_KEYS:
        check(any(c in text for c in _CONVENTIONS[key]), "speed item without its own reporting convention")
        other = "km/h" if key == "m/s" else "m/s"
        check(not any(c in text for c in _CONVENTIONS[other]), "speed item carries the other unit's convention")

    distractors = [u for u in options if u != key]
    check(len(distractors) >= 3, "appropriate_activity_unit needs >= 3 distractors (never binary)")
    check_pairwise_distinct(key, *distractors, msg="appropriate_activity_unit options collided")

    latex = f"plain:{text}"
    working = [
        f"{key} is the unit whose type ({_KEY_TYPE[key]}) and scale ({_KEY_SCALE[key]}) match the measurement described."
    ]
    return dict(
        text=text,
        latex=latex,
        answer=key,
        answer_text=key,
        answer_latex=latex_answer(key, plain=True),
        working=working,
        distractors=distractors,
    )


@generator(
    "p19h_appropriate_activity_unit",
    level=12,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p19h_appropriate_activity_unit(rng, tier="easy"):
    """Choose the most appropriate unit (m, km, km/h or m/s) for a stated
    activity and described measurement -- see the module docstring for the
    per-tier reasoning escalation."""
    return _build_appropriate_activity_unit(rng, tier)


# ===========================================================================
# Skill 2: need_for_milligrams (ACMEM099)
# ===========================================================================

_MASS_UNITS = ("mg", "g", "kg", "t")
_RANK = {"mg": 0, "g": 1, "kg": 2, "t": 3}
_MG_PER_UNIT = {
    "mg": Fraction(1),
    "g": Fraction(1_000),
    "kg": Fraction(1_000_000),
    "t": Fraction(1_000_000_000),
}
_UP = {"mg": "g", "g": "kg", "kg": "t"}

# Single objects. `profiles[K] = (lo, hi)`: the REAL range of the object's
# mass, as a whole number of K. `stated` is the set of units the object's
# record could plausibly be written in. Every range is one a real object of
# that name can have.
_SINGLES = [
    dict(
        sentences=[
            "A pharmacy's stock sheet lists a vitamin tablet's mass as {v} {s}.",
            "A supplement label gives a vitamin tablet's mass as {v} {s}.",
        ],
        profiles={"mg": (100, 999), "g": (1, 4)},
        stated=("mg", "g"),
    ),
    dict(
        sentences=["A pharmacist's dispensing record lists one sachet of powdered medicine as {v} {s}."],
        profiles={"mg": (200, 999), "g": (1, 20)},
        stated=("mg", "g"),
    ),
    dict(
        sentences=["A chemistry student's practical notes record a sample of copper sulfate crystals as {v} {s}."],
        profiles={"mg": (50, 999), "g": (1, 50)},
        stated=("mg", "g"),
    ),
    dict(
        sentences=["A nutrition panel lists the caffeine in one cup of tea as {v} {s}."],
        profiles={"mg": (30, 100)},
        stated=("mg", "g"),
    ),
    dict(
        sentences=[
            "A supermarket shelf label states a bag of rice as {v} {s}.",
            "A pantry inventory lists a bag of rice as {v} {s}.",
        ],
        profiles={"g": (200, 999), "kg": (1, 25)},
        stated=("g", "kg"),
    ),
    dict(
        sentences=["A greengrocer's price ticket states a bag of potatoes as {v} {s}."],
        profiles={"g": (500, 999), "kg": (1, 10)},
        stated=("g", "kg"),
    ),
    dict(
        sentences=["An egg carton's label lists a hen's egg as {v} {s}."], profiles={"g": (50, 70)}, stated=("g", "kg")
    ),
    dict(sentences=["A wrapper lists a chocolate bar as {v} {s}."], profiles={"g": (40, 250)}, stated=("g", "kg")),
    dict(
        sentences=["A phone maker's specification sheet lists a smartphone as {v} {s}."],
        profiles={"g": (140, 250)},
        stated=("g", "kg"),
    ),
    dict(
        sentences=["A retailer's specification page lists a laptop as {v} {s}."],
        profiles={"g": (900, 999), "kg": (1, 3)},
        stated=("g", "kg"),
    ),
    dict(
        sentences=["A hospital record lists a newborn baby's mass as {v} {s}."],
        profiles={"kg": (2, 5)},
        stated=("g", "kg"),
    ),
    dict(
        sentences=["A bike shop's catalogue lists a road bicycle as {v} {s}."],
        profiles={"kg": (8, 15)},
        stated=("g", "kg"),
    ),
    dict(
        sentences=["A sports store's catalogue lists a bowling ball as {v} {s}."],
        profiles={"kg": (3, 7)},
        stated=("g", "kg"),
    ),
    dict(
        sentences=["A warehouse dispatch note lists a pallet of tiles as {v} {s}."],
        profiles={"kg": (300, 999), "t": (1, 2)},
        stated=("kg", "t"),
    ),
    dict(
        sentences=["A waste-collection docket lists a skip of building rubble as {v} {s}."],
        profiles={"kg": (300, 999), "t": (1, 5)},
        stated=("kg", "t"),
    ),
    dict(
        sentences=["A ute driver's delivery slip lists a load of firewood as {v} {s}."],
        profiles={"kg": (200, 999), "t": (1, 1)},
        stated=("kg", "t"),
    ),
    dict(
        sentences=["A steel supplier's quote lists a stack of steel beams as {v} {s}."],
        profiles={"kg": (400, 999), "t": (1, 30)},
        stated=("kg", "t"),
    ),
    dict(
        sentences=["A shipping manifest lists a shipping container's cargo as {v} {s}."],
        profiles={"t": (2, 28)},
        stated=("kg", "t"),
    ),
    dict(
        sentences=["A weighbridge ticket lists a semi-trailer's load as {v} {s}."],
        profiles={"t": (10, 40)},
        stated=("kg", "t"),
    ),
    dict(
        sentences=["A zoo's records list an African elephant's mass as {v} {s}."],
        profiles={"t": (3, 6)},
        stated=("kg", "t"),
    ),
    dict(
        sentences=["A marine museum's display panel lists a humpback whale's mass as {v} {s}."],
        profiles={"t": (25, 40)},
        stated=("kg", "t"),
    ),
    dict(
        sentences=["A port authority's record lists a barge's grain cargo as {v} {s}."],
        profiles={"t": (200, 999)},
        stated=("kg", "t"),
    ),
]
for _sp in _SINGLES:
    _sp["multi"] = len(_sp["profiles"]) >= 2

# Collectives: `count` identical items, each `p` of the item's natural unit
# `unit`. `cap` is the largest feasible total in `unit`; `extra` is a second,
# separately stated mass ((lo, hi) in `unit`) used by the hard tier.
_COLLECTIVES = [
    dict(
        unit="mg",
        p=[20, 25, 40, 50, 60, 80, 100, 125, 200, 250, 500],
        c=(4, 40),
        cap=30_000,
        sentence="A pharmacy's stock sheet for one bottle lists {c} vitamin tablets, each with a mass of {p} {u}.",
        extra="The same order includes a sachet of powdered medicine with a mass of {e} {E}.",
        e=(100, 999),
    ),
    dict(
        unit="g",
        p=[50, 55, 60, 65, 70],
        c=(6, 30),
        cap=2_000,
        sentence="A grocer's crate label lists {c} hens' eggs, each with a mass of {p} {u}.",
        extra="The crate also holds a jar of honey with a mass of {e} {E}.",
        e=(100, 999),
    ),
    dict(
        unit="g",
        p=[125, 200, 250, 500],
        c=(4, 40),
        cap=20_000,
        sentence="A dairy's delivery note lists {c} blocks of butter, each with a mass of {p} {u}.",
        extra="It also lists a tub of cream cheese with a mass of {e} {E}.",
        e=(100, 999),
    ),
    dict(
        unit="g",
        p=[250, 400, 500, 800],
        c=(2, 60),
        cap=30_000,
        sentence="A supermarket delivery note lists {c} tins of tomatoes, each with a mass of {p} {u}.",
        extra="The delivery also includes a box of pasta with a mass of {e} {E}.",
        e=(100, 999),
    ),
    dict(
        unit="g",
        p=[40, 50, 60, 75, 100, 150, 200],
        c=(5, 20),
        cap=5_000,
        sentence="A vending-machine refill sheet lists {c} chocolate bars, each with a mass of {p} {u}.",
        extra="It also lists a packet of biscuits with a mass of {e} {E}.",
        e=(100, 600),
    ),
    dict(
        unit="kg",
        p=[20, 25, 40],
        c=(8, 100),
        cap=3_000,
        sentence="A builder's supply docket lists {c} bags of cement, each with a mass of {p} {u}.",
        extra="The load also includes a bundle of timber with a mass of {e} {E}.",
        e=(10, 999),
    ),
    dict(
        unit="kg",
        p=[10, 20, 25, 50],
        c=(6, 120),
        cap=3_000,
        sentence="A landscaping supplier's docket lists {c} bags of sand, each with a mass of {p} {u}.",
        extra="The order also includes a stack of pavers with a mass of {e} {E}.",
        e=(10, 999),
    ),
    dict(
        unit="kg",
        p=[200, 250],
        c=(2, 60),
        cap=20_000,
        sentence="A workshop's chemical store lists {c} steel drums of lubricant, each with a mass of {p} {u}.",
        extra="The store also lists a gas cylinder with a mass of {e} {E}.",
        e=(10, 999),
    ),
    dict(
        unit="kg",
        p=[10, 20, 25],
        c=(3, 80),
        cap=2_000,
        sentence="A farm shed's dispatch sheet lists {c} sacks of potatoes, each with a mass of {p} {u}.",
        extra="It also lists a crate of onions with a mass of {e} {E}.",
        e=(10, 999),
    ),
    dict(
        unit="t",
        p=[40, 50, 60, 70, 80],
        c=(3, 14),
        cap=999,
        sentence="A rail operator's consist sheet lists {c} loaded grain wagons, each with a mass of {p} {u}.",
        extra="A crane wagon carries a load of steel with a mass of {e} {E}.",
        e=(1, 60),
    ),
    dict(
        unit="t",
        p=[20, 25, 30, 36],
        c=(2, 12),
        cap=999,
        sentence="A haulage firm's schedule lists {c} loaded trailers, each with a mass of {p} {u}.",
        extra="A lead truck carries a load of equipment with a mass of {e} {E}.",
        e=(1, 20),
    ),
]

_QUESTION_SINGLE = (
    "Which of these units would express this mass as a whole number of no more than 3 digits, with no decimal point?"
)
_QUESTION_TOTAL = (
    "Which of these units would let you write the total mass of everything listed as a whole number "
    "of no more than 3 digits, with no decimal point?"
)
_QUESTION_COMBINED = (
    "Which of these units would let you write their COMBINED mass as a whole number of no more than "
    "3 digits, with no decimal point?"
)
_MULTI_SHARE = 0.6


def _is_clean(value: Fraction) -> bool:
    """Whole number of no more than 3 digits, no decimal point."""
    return value.denominator == 1 and 0 < value <= 999


def _format_mass(value: Fraction) -> str:
    return exact_decimal_string(value, 12)


def _value_in(n: int, from_unit: str, to_unit: str) -> Fraction:
    pivot = Fraction(n) * _MG_PER_UNIT[from_unit]
    return pivot / _MG_PER_UNIT[to_unit]


def _clean_units(total_mg: Fraction) -> list[str]:
    return [u for u in _MASS_UNITS if _is_clean(total_mg / _MG_PER_UNIT[u])]


def _stated_mg(stated: list[tuple[Fraction, str]]) -> Fraction:
    """Sum of the STATED values, converted to mg from the numbers as printed."""
    return sum((v * _MG_PER_UNIT[u] for v, u in stated), Fraction(0))


def _prove_unique(key: str, total_mg: Fraction, label: str) -> None:
    """Prove the predicate is crisp: `key` is the ONLY unit in which the
    total is a whole number of no more than 3 digits, every draw."""
    for unit in _MASS_UNITS:
        candidate = total_mg / _MG_PER_UNIT[unit]
        if unit == key:
            check(_is_clean(candidate), f"{label}: total is not clean in its own unit {key}")
        else:
            check(not _is_clean(candidate), f"{label}: total is ALSO clean in {unit}, predicate not crisp")


def _pick_single(rng, cands: list):
    multi = [c for c in cands if c[0]["multi"]]
    if multi and rng.random() < _MULTI_SHARE:
        return rng.choice(multi)
    return rng.choice(cands)


def _build_need_easy(rng) -> tuple[str, str]:
    key = rng.choice(_MASS_UNITS)
    cands = []
    for sp in _SINGLES:
        if key in sp["profiles"]:
            s_opts = [u for u in sp["stated"] if u != key and abs(_RANK[u] - _RANK[key]) == 1]
            if s_opts:
                cands.append((sp, s_opts))
    sp, s_opts = _pick_single(rng, cands)
    s = rng.choice(s_opts)
    lo, hi = sp["profiles"][key]
    n = rng.randint(lo, hi)
    stated_value = _value_in(n, key, s)
    check_conversion(n, key, s, stated_value, _MG_PER_UNIT, label="need-for-milligrams stated value")
    _prove_unique(key, _stated_mg([(stated_value, s)]), "single")
    sentence = rng.choice(sp["sentences"]).format(v=_format_mass(stated_value), s=s)
    return key, f"{sentence} {_QUESTION_SINGLE}"


def _collective_units(rng, key: str):
    """The collectives whose total can land in `key`, split into 'stay' (the
    item's natural unit IS key) and 'cross' (key is the next unit up)."""
    stay = [c for c in _COLLECTIVES if c["unit"] == key]
    cross = [c for c in _COLLECTIVES if _UP.get(c["unit"]) == key]
    if stay and cross:
        return (stay, "stay") if rng.random() < 0.5 else (cross, "cross")
    return (stay, "stay") if stay else (cross, "cross")


def _build_collective(rng, key: str, with_extra: bool) -> str:
    pool, mode = _collective_units(rng, key)
    for _ in range(2000):
        sp = rng.choice(pool)
        u = sp["unit"]
        p = rng.choice(sp["p"])
        c = rng.randint(*sp["c"])
        total_u = p * c
        e = None
        if with_extra:
            elo, ehi = sp["e"]
            if mode == "stay":
                room = min(ehi, 999 - total_u)
                if room < elo:
                    continue
                e = rng.randint(elo, room)
            else:
                e = (-total_u) % 1000
                if e == 0 or not (elo <= e <= ehi):
                    continue
            total_u += e
        if mode == "stay":
            if not (1 <= total_u <= 999):
                continue
        else:
            if total_u % 1000 or total_u < 1000 or total_u > sp["cap"] or total_u // 1000 > 999:
                continue
        if total_u > sp["cap"]:
            continue
        # Build the printed values, then prove the key from the PRINTED numbers.
        stated = [(Fraction(p), u)]
        text = sp["sentence"].format(c=c, p=p, u=u)
        if with_extra:
            alt = u in _UP and rng.random() < 0.5
            e_unit = _UP[u] if alt else u
            e_val = Fraction(e) / 1000 if alt else Fraction(e)
            stated.append((e_val, e_unit))
            text += " " + sp["extra"].format(e=_format_mass(e_val), E=e_unit)
        total_mg = Fraction(c) * stated[0][0] * _MG_PER_UNIT[u] + sum(
            (v * _MG_PER_UNIT[su] for v, su in stated[1:]), Fraction(0)
        )
        if _clean_units(total_mg) != [key]:
            continue
        _prove_unique(key, total_mg, "collective")
        return f"{text} {_QUESTION_TOTAL}"
    raise RuntimeError(f"no collective found for key {key!r} (extra={with_extra})")


def _build_need_medium(rng) -> tuple[str, str]:
    key = rng.choice(_MASS_UNITS)
    return key, _build_collective(rng, key, with_extra=False)


def _build_need_hard(rng) -> tuple[str, str]:
    key = rng.choice(_MASS_UNITS)
    return key, _build_collective(rng, key, with_extra=True)


def _build_need_exam(rng) -> tuple[str, str]:
    key = rng.choice(_MASS_UNITS)
    cands = [sp for sp in _SINGLES if key in sp["profiles"]]
    for _ in range(500):
        a, b = rng.sample(cands, 2)
        n1 = rng.randint(*a["profiles"][key])
        n2 = rng.randint(*b["profiles"][key])
        if n1 + n2 > 999:
            continue
        s1 = rng.choice(a["stated"])
        s2 = rng.choice(b["stated"])
        if s1 == key and s2 == key:
            continue
        v1, v2 = _value_in(n1, key, s1), _value_in(n2, key, s2)
        check_conversion(n1, key, s1, v1, _MG_PER_UNIT, label="need-for-milligrams exam value 1")
        check_conversion(n2, key, s2, v2, _MG_PER_UNIT, label="need-for-milligrams exam value 2")
        total_mg = _stated_mg([(v1, s1), (v2, s2)])
        if _clean_units(total_mg) != [key]:
            continue
        _prove_unique(key, total_mg, "exam total")
        t1 = rng.choice(a["sentences"]).format(v=_format_mass(v1), s=s1)
        t2 = rng.choice(b["sentences"]).format(v=_format_mass(v2), s=s2)
        return key, f"{t1} {t2} {_QUESTION_COMBINED}"
    raise RuntimeError(f"no exam pair found for key {key!r}")


def _build_need_for_milligrams(rng, tier: str) -> dict:
    if tier == "easy":
        key, text = _build_need_easy(rng)
    elif tier == "medium":
        key, text = _build_need_medium(rng)
    elif tier == "hard":
        key, text = _build_need_hard(rng)
    elif tier == "exam":
        key, text = _build_need_exam(rng)
    else:
        raise ValueError(f"unknown tier {tier!r}")

    distractors = [u for u in _MASS_UNITS if u != key]
    check(len(distractors) == 3, "need_for_milligrams distractor count wrong")
    check_pairwise_distinct(key, *distractors, msg="need_for_milligrams options collided")

    latex = f"plain:{text}"
    working = [f"{key} is the only one of mg/g/kg/t in which this mass is a whole number of no more than 3 digits."]
    return dict(
        text=text,
        latex=latex,
        answer=key,
        answer_text=key,
        answer_latex=latex_answer(key, plain=True),
        working=working,
        distractors=distractors,
    )


@generator(
    "p19h_need_for_milligrams",
    level=12,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p19h_need_for_milligrams(rng, tier="easy"):
    """Choose the mass unit (mg, g, kg or t) that expresses a stated mass as
    a whole number of no more than 3 digits, avoiding an awkward decimal or
    an unwieldy digit count -- see the module docstring for the per-tier
    escalation and the distinction from the shipping eview skill's easy
    tier."""
    return _build_need_for_milligrams(rng, tier)
