"""Generates `rate_usage_classification` questions (ACMEM014): match a quantity in a named reporting context to
the rate unit conventionally used there, `answer: mcq`.

Conventional rate-unit usage mcq
classification (``senior-identify-common-usage-of-rates-for-example-km``, ACMEM014).

``rate_usage_classification`` -- ``mcq`` matching a quantity, reported in a
NAMED reporting context, to the rate unit that context conventionally uses.
Answer space: 5 keys spanning speed-type and non-speed-type
quantities:

- ``km/h``           -- a speed reported to the general public or on a road
  / TV / gym / news display (dashboard, broadcast graphic, scoreboard, public
  forecast, treadmill console ...)
- ``m/s``             -- the same kinds of speed reported in an SI-unit
  scientific or engineering document (lab table, physics-class problem,
  journal article, wind-tunnel or crash-test report)
- ``beats/minute``    -- a pulse / heart rate
- ``breaths/minute``  -- a breathing / respiratory rate
- ``L/minute``        -- a tap / shower / hose flow rate (WELS label, plumber's
  flow meter, bucket-fill timing)

1. **The context, not a cue word, settles km/h vs m/s.** Every speed subject
   (sprinter, cyclist, car, train, cricket ball, tennis serve, gust of wind,
   skydiver, skier, motorbike, racehorse, cheetah, treadmill runner, radar-gun
   readings ...) is offered with BOTH a public-facing km/h context and an
   SI-unit m/s context, so no subject or instrument word fixes the key. Items
   whose conventional unit is not km/h or m/s in real usage were dropped: a
   cyclist's leg-turnover (cadence is rpm), ferry dashboards and aeroplane
   flight computers (knots), pool / rainwater pumps and appliance inlet valves
   (often L/hour or m3/hour), fire-hose "discharge gauges" (pressure).
2. **No dead distractors.** Options are the key plus units someone could
   genuinely choose for THAT quantity (an explicit per-key plausible list):
   speed keys keep the other speed unit as the trap plus mph / knots / cm/s /
   m/minute; pulse and breathing offer the sibling vital sign and scale
   alternatives; flow offers L/hour, mL/minute, L/second, kL/hour. The option
   set varies per question (3 to 5 options), never a fixed five.
3. **Wide pools.** Easy and medium phrases are compositional (subject x head x
   context; actor x noun x setting; fixture x template x suffix); hard and
   exam are drawn from 7 scenarios with 3 clause variants per quantity.

Construct-to-target, every tier: the TYPE is drawn uniformly from
``{speed, non-speed}`` first, then the specific key uniformly from that type's
keys (so no key exceeds 25% by design), then the phrase / scenario.

Difficulty:

- **easy**: one clause naming an unambiguous quantity and its reporting
  context.
- **medium**: one clause, from a differently worded pool (a "figure quoted for
  ..." / "reading a meter shows" style) so the cue takes longer to place.
- **hard**: two quantities of one coherent scenario are named and the question
  picks one by ordinal position ("first"/"second").
- **exam**: three quantities of one scenario (always both speed conventions
  plus one non-speed quantity) are named; the question names the SOURCE it
  asks about, so the reader matches a named device to its own clause and then
  applies that clause's reporting convention.
"""

from __future__ import annotations

from ..core import check, check_pairwise_distinct, generator, latex_answer

# ===========================================================================
# Answer space
# ===========================================================================

_KEYS = ("km/h", "m/s", "beats/minute", "breaths/minute", "L/minute")
_TYPE_KEYS = {
    "speed": ("km/h", "m/s"),
    "non-speed": ("beats/minute", "breaths/minute", "L/minute"),
}
# Independent reverse lookup -- built separately from _TYPE_KEYS above so an
# in-generator check can re-derive the type a second way.
_KEY_TYPE = {
    "km/h": "speed",
    "m/s": "speed",
    "beats/minute": "non-speed",
    "breaths/minute": "non-speed",
    "L/minute": "non-speed",
}

# Per-quantity list of units someone could plausibly choose for that quantity
# (scale or convention alternatives). Every distractor offered is drawn from
# the list for the item's key; the other speed unit is always included for the
# two speed keys (it is the live trap).
_PLAUSIBLE_OPTIONS = {
    "km/h": ("m/s", "mph"),
    "m/s": ("km/h", "mph", "cm/s"),
    "beats/minute": ("breaths/minute", "beats/second"),
    "breaths/minute": ("beats/minute", "breaths/second"),
    "L/minute": ("L/hour", "mL/minute", "L/second"),
}
_MANDATORY_DISTRACTOR = {"km/h": "m/s", "m/s": "km/h"}
_MIN_DISTRACTORS = 2  # never binary: key + >= 2 distractors = 3 options at least

# ===========================================================================
# easy / medium speed pools: subject x head x context. Every subject carries
# BOTH a km/h context and an m/s context, so the subject word never fixes the
# key; the reporting context does.
# ===========================================================================

_SI = "using SI units"
_SPEED_SUBJECTS: list[dict] = [
    dict(
        S="a sprinter in a 100 m final",
        radar=True,
        km=[
            "on the television broadcast's on-screen speed graphic",
            "in a newspaper's race report written for the general public",
            "in the commentator's on-air remark to viewers",
        ],
        ms=[
            "in a biomechanics lab's table of results in SI units",
            "in a physics-class analysis of the race video, " + _SI,
            "in a sports-science journal article that reports in SI units",
        ],
    ),
    dict(
        S="a cyclist in a road race",
        radar=False,
        km=[
            "on the cyclist's bike computer",
            "on the race broadcast's on-screen graphic",
            "in a cycling magazine's feature for general readers",
        ],
        ms=[
            "in a wind-tunnel report that uses SI units",
            "in a physics-class problem about air resistance, " + _SI,
            "in a sports-science journal article that reports in SI units",
        ],
    ),
    dict(
        S="a car on a highway",
        radar=True,
        km=[
            "on the car's dashboard speedometer",
            "on a roadside speed-limit sign",
            "in a police media release about a speeding driver",
        ],
        ms=[
            "in a physics-class problem about braking, " + _SI,
            "in an engineering crash-test report that uses SI units",
            "in a physics-lab report on constant acceleration, " + _SI,
        ],
    ),
    dict(
        S="a train between two stations",
        radar=False,
        km=[
            "on the driver's onboard speed display",
            "on a station's public information board",
            "in a timetable's list of express services",
        ],
        ms=[
            "in a physics-class problem about constant acceleration, " + _SI,
            "in an engineering report on braking tests that uses SI units",
            "in a physics textbook's worked example, " + _SI,
        ],
    ),
    dict(
        S="a thrown cricket ball",
        radar=True,
        km=[
            "on the television broadcast's speed graphic",
            "in a newspaper's match report for general readers",
            "on the stadium's scoreboard display",
        ],
        ms=[
            "in a physics-class projectile-motion problem, " + _SI,
            "in a sports-science lab's SI-unit data table",
            "in a biomechanics journal article that reports in SI units",
        ],
    ),
    dict(
        S="a tennis serve",
        radar=True,
        km=[
            "on the tournament's scoreboard speed display",
            "on the television broadcast's on-screen graphic",
            "in a sports website's match report",
        ],
        ms=[
            "in a physics-class problem about projectiles, " + _SI,
            "in a sports-science lab's SI-unit data table",
            "in a biomechanics journal article that reports in SI units",
        ],
    ),
    dict(
        S="a gust of wind",
        radar=False,
        km=[
            "in the weather bureau's public forecast",
            "in a local news weather segment",
            "in a storm warning issued to the public",
        ],
        ms=[
            "in a meteorology research paper's data table, which uses SI units",
            "in a physics-class problem about drag, " + _SI,
            "in a wind-engineering report that uses SI units",
        ],
    ),
    dict(
        S="a skydiver in freefall",
        radar=False,
        km=[
            "in a skydiving company's brochure for customers",
            "in a television documentary's narration",
            "in a magazine feature for general readers",
        ],
        ms=[
            "in a physics textbook's terminal-velocity problem, " + _SI,
            "in a physics-class worked example, " + _SI,
            "in a sports-science journal article that reports in SI units",
        ],
    ),
    dict(
        S="a downhill skier",
        radar=True,
        km=[
            "on the race broadcast's timing graphic",
            "in a newspaper's race report for general readers",
            "in a ski magazine's article for general readers",
        ],
        ms=[
            "in a physics-class problem about slopes and friction, " + _SI,
            "in a sports-science lab's SI-unit data table",
            "in a biomechanics journal article that reports in SI units",
        ],
    ),
    dict(
        S="a motorbike on a highway",
        radar=True,
        km=[
            "on the motorbike's speedometer",
            "on a roadside speed-limit sign",
            "in a police media release about a speeding rider",
        ],
        ms=[
            "in a physics-class problem about circular motion, " + _SI,
            "in an engineering test report that uses SI units",
            "in a physics-lab report on acceleration, " + _SI,
        ],
    ),
    dict(
        S="a racehorse in a sprint race",
        radar=True,
        km=[
            "on the racing broadcast's on-screen graphic",
            "in a newspaper's race report for the general public",
            "in the commentator's on-air remark to viewers",
        ],
        ms=[
            "in an equine biomechanics lab's SI-unit data table",
            "in a veterinary sports-science journal article that reports in SI units",
            "in a physics-class problem about motion, " + _SI,
        ],
    ),
    dict(
        S="a cheetah in a chase",
        radar=True,
        km=[
            "in a wildlife documentary's narration",
            "on a zoo's information sign for visitors",
            "in a magazine feature for general readers",
        ],
        ms=[
            "in a zoology lab's SI-unit data table",
            "in a physics-class problem about motion, " + _SI,
            "in a biomechanics journal article that reports in SI units",
        ],
    ),
    dict(
        S="a runner on a treadmill",
        radar=False,
        km=["on the treadmill's console", "on the gym's display panel", "on a fitness app's summary screen"],
        ms=[
            "in an exercise-physiology lab's SI-unit data sheet",
            "in a physics-class problem about running, " + _SI,
            "in a biomechanics journal article that reports in SI units",
        ],
    ),
]

_EASY_SPEED_HEADS = ("the speed of {S}", "the top speed reached by {S}")
_EASY_RADAR_HEAD = "the speed a radar gun records for {S}"
_MEDIUM_SPEED_HEADS = (
    "the figure quoted for how fast {S} was moving",
    "the number given for the speed of {S} at its fastest",
)
_MEDIUM_RADAR_HEAD = "the value a radar gun logs for {S}"


def _speed_phrases(key: str, heads: tuple[str, ...], radar_head: str) -> list[str]:
    field = "km" if key == "km/h" else "ms"
    out = []
    for subj in _SPEED_SUBJECTS:
        head_list = list(heads) + ([radar_head] if subj["radar"] else [])
        for head in head_list:
            for ctx in subj[field]:
                out.append(f"{head.format(S=subj['S'])} {ctx}")
    return out


# ===========================================================================
# easy / medium non-speed pools
# ===========================================================================

_VITAL_READERS = (
    "a nurse records",
    "a paramedic checks",
    "a hospital monitor displays",
    "a smartwatch displays",
    "a fitness tracker reports",
    "a doctor counts",
    "a physiotherapist records",
    "an ambulance crew records",
    "a triage nurse notes",
    "a personal trainer checks",
    "a school nurse checks",
    "an anaesthetist monitors",
)
_VITAL_WHENS = (
    "during a fitness test",
    "at rest",
    "after light exercise",
    "before an operation",
    "after a short walk",
    "on arrival at the clinic",
    "during a routine check-up",
    "after climbing a flight of stairs",
)
_PULSE_EASY_NOUNS = ("the pulse", "the heart rate", "the pulse rate")
_BREATH_EASY_NOUNS = ("the respiratory rate", "the breathing rate", "the respiration rate")
_MEDIUM_FIGURES = ("the figure", "the number", "the reading")

_WELS_FIXTURES = ("a showerhead", "a bathroom tap", "a kitchen tap", "a laundry tap", "a basin mixer")
_GENERAL_FIXTURES = (
    "a garden tap",
    "a kitchen tap",
    "a bath tap",
    "a garden hose",
    "an outdoor shower",
    "a car-wash hose",
    "a laundry tap",
    "a hose reel",
)
_FLOW_SUFFIXES = (
    "",
    " when fully open",
    " on a normal household water supply",
    " in a family home",
    " at its standard setting",
)
_FLOW_EASY_WELS = ("the flow rate a WELS water-efficiency label states for {f}",)
_FLOW_EASY_GENERAL = (
    "how quickly water flows from {f}",
    "the rate at which {f} fills a bucket",
    "the flow rate a plumber's flow meter reads for {f}",
)
_FLOW_MEDIUM_WELS = ("the number a WELS water-efficiency rating gives for {f}",)
_FLOW_MEDIUM_GENERAL = (
    "the figure a flow meter fitted to {f} shows",
    "the reading a plumber notes for the water coming from {f}",
    "the number found by timing how fast {f} fills a bucket",
)


def _flow_phrases(wels: tuple[str, ...], general: tuple[str, ...]) -> list[str]:
    out = []
    for tmpl in wels:
        for f in _WELS_FIXTURES:
            for suf in _FLOW_SUFFIXES:
                out.append(tmpl.format(f=f) + suf)
    for tmpl in general:
        for f in _GENERAL_FIXTURES:
            for suf in _FLOW_SUFFIXES:
                out.append(tmpl.format(f=f) + suf)
    return out


def _easy_vital(nouns: tuple[str, ...]) -> list[str]:
    return [f"{n} {r} for someone {w}" for n in nouns for r in _VITAL_READERS for w in _VITAL_WHENS]


def _medium_vital(body: str) -> list[str]:
    return [
        f"{fig} {r} for someone's {body} {w}" for fig in _MEDIUM_FIGURES for r in _VITAL_READERS for w in _VITAL_WHENS
    ]


_EASY_POOL = {
    "km/h": _speed_phrases("km/h", _EASY_SPEED_HEADS, _EASY_RADAR_HEAD),
    "m/s": _speed_phrases("m/s", _EASY_SPEED_HEADS, _EASY_RADAR_HEAD),
    "beats/minute": _easy_vital(_PULSE_EASY_NOUNS),
    "breaths/minute": _easy_vital(_BREATH_EASY_NOUNS),
    "L/minute": _flow_phrases(_FLOW_EASY_WELS, _FLOW_EASY_GENERAL),
}
_MEDIUM_POOL = {
    "km/h": _speed_phrases("km/h", _MEDIUM_SPEED_HEADS, _MEDIUM_RADAR_HEAD),
    "m/s": _speed_phrases("m/s", _MEDIUM_SPEED_HEADS, _MEDIUM_RADAR_HEAD),
    "beats/minute": _medium_vital("heart"),
    "breaths/minute": _medium_vital("breathing"),
    "L/minute": _flow_phrases(_FLOW_MEDIUM_WELS, _FLOW_MEDIUM_GENERAL),
}

# ===========================================================================
# hard / exam scenarios: 3 (device, clause) variants per quantity. The device
# is a verbatim substring of exactly its own clause (checked at import) so the
# exam tier can point at a clause by its named source. Every speed quantity
# appears in a public-facing km/h source and an SI-unit m/s source of the SAME
# subject, so a subject or instrument word never fixes the key.
# ===========================================================================

_SCENARIOS: list[dict] = [
    dict(
        intro="A sprinter's 100 m race",
        variants={
            "km/h": [
                (
                    "the television broadcast's speed graphic",
                    "the speed of the sprinter on the television broadcast's speed graphic",
                ),
                (
                    "the newspaper's race report",
                    "the speed of the sprinter in the newspaper's race report for the general public",
                ),
                (
                    "the commentator's on-air remark",
                    "the speed of the sprinter in the commentator's on-air remark to viewers",
                ),
            ],
            "m/s": [
                (
                    "the biomechanics lab's SI-unit table",
                    "the speed of the sprinter in the biomechanics lab's SI-unit table of results",
                ),
                (
                    "the physics class's video analysis",
                    "the speed of the sprinter in the physics class's video analysis, done in SI units",
                ),
                (
                    "the sports-science journal article",
                    "the speed of the sprinter in the sports-science journal article, which reports in SI units",
                ),
            ],
            "beats/minute": [
                (
                    "the coach's heart-rate monitor",
                    "the heart rate the coach's heart-rate monitor records for the sprinter after the race",
                ),
                (
                    "the wrist fitness tracker",
                    "the pulse the wrist fitness tracker logs for the sprinter after the race",
                ),
                (
                    "the team doctor's pulse check",
                    "the pulse rate the team doctor's pulse check records for the sprinter after the race",
                ),
            ],
            "breaths/minute": [
                (
                    "the physiologist's respiratory monitor",
                    "the respiratory rate the physiologist's respiratory monitor records for the sprinter after the race",
                ),
                (
                    "the team doctor's breathing check",
                    "the breathing rate the team doctor's breathing check records for the sprinter after the race",
                ),
                (
                    "the recovery-room monitor",
                    "the respiration rate the recovery-room monitor records for the sprinter after the race",
                ),
            ],
        },
    ),
    dict(
        intro="A cyclist's time trial",
        variants={
            "km/h": [
                ("the handlebar bike computer", "the speed of the cyclist on the handlebar bike computer"),
                (
                    "the race broadcast's on-screen graphic",
                    "the speed of the cyclist on the race broadcast's on-screen graphic",
                ),
                (
                    "the cycling magazine's feature",
                    "the speed of the cyclist in the cycling magazine's feature for general readers",
                ),
            ],
            "m/s": [
                ("the wind-tunnel report", "the speed of the cyclist in the wind-tunnel report, which uses SI units"),
                (
                    "the physics class's air-resistance problem",
                    "the speed of the cyclist in the physics class's air-resistance problem, set in SI units",
                ),
                (
                    "the sports-science journal paper",
                    "the speed of the cyclist in the sports-science journal paper, which reports in SI units",
                ),
            ],
            "beats/minute": [
                (
                    "the chest-strap monitor",
                    "the heart rate the chest-strap monitor logs for the cyclist during the time trial",
                ),
                (
                    "the coach's pulse check",
                    "the pulse the coach's pulse check records for the cyclist after the time trial",
                ),
                (
                    "the team doctor's heart-rate monitor",
                    "the pulse rate the team doctor's heart-rate monitor records for the cyclist after the time trial",
                ),
            ],
            "breaths/minute": [
                ("the mask sensor", "the breathing rate the mask sensor records for the cyclist during the time trial"),
                (
                    "the lab technician's respiratory monitor",
                    "the respiratory rate the lab technician's respiratory monitor records for the cyclist during the time trial",
                ),
                (
                    "the team doctor's breathing check",
                    "the breathing rate the team doctor's breathing check records for the cyclist after the time trial",
                ),
            ],
        },
    ),
    dict(
        intro="A runner's treadmill test",
        variants={
            "km/h": [
                ("the treadmill's console", "the speed of the runner on the treadmill's console"),
                ("the gym's display panel", "the speed of the runner on the gym's display panel"),
                ("the fitness app's summary screen", "the speed of the runner on the fitness app's summary screen"),
            ],
            "m/s": [
                (
                    "the exercise-physiology lab's SI data sheet",
                    "the speed of the runner in the exercise-physiology lab's SI data sheet",
                ),
                (
                    "the physics class's running problem",
                    "the speed of the runner in the physics class's running problem, set in SI units",
                ),
                (
                    "the biomechanics journal article",
                    "the speed of the runner in the biomechanics journal article, which reports in SI units",
                ),
            ],
            "beats/minute": [
                ("the heart-rate strap", "the heart rate the heart-rate strap logs for the runner during the test"),
                ("the smartwatch", "the pulse the smartwatch displays for the runner during the test"),
                (
                    "the nurse's pulse check",
                    "the pulse rate the nurse's pulse check records for the runner after the test",
                ),
            ],
            "breaths/minute": [
                (
                    "the breathing-mask sensor",
                    "the breathing rate the breathing-mask sensor records for the runner during the test",
                ),
                (
                    "the physiologist's respiratory monitor",
                    "the respiratory rate the physiologist's respiratory monitor records for the runner during the test",
                ),
                (
                    "the nurse's breathing check",
                    "the breathing rate the nurse's breathing check records for the runner after the test",
                ),
            ],
        },
    ),
    dict(
        intro="A patient's ambulance transfer",
        variants={
            "km/h": [
                (
                    "the ambulance's dashboard speedometer",
                    "the speed of the ambulance on the ambulance's dashboard speedometer",
                ),
                (
                    "the driver's on-board trip display",
                    "the speed of the ambulance on the driver's on-board trip display",
                ),
                (
                    "the highway's electronic speed sign",
                    "the speed of the ambulance on the highway's electronic speed sign",
                ),
            ],
            "m/s": [
                (
                    "the physics class's braking problem",
                    "the speed of the ambulance in the physics class's braking problem, set in SI units",
                ),
                (
                    "the engineering crash-test report",
                    "the speed of the ambulance in the engineering crash-test report, which uses SI units",
                ),
                (
                    "the physics-lab report",
                    "the speed of the ambulance in the physics-lab report on constant acceleration, done in SI units",
                ),
            ],
            "beats/minute": [
                (
                    "the paramedic's cardiac monitor",
                    "the heart rate the paramedic's cardiac monitor records for the patient during the transfer",
                ),
                (
                    "the triage nurse's pulse check",
                    "the pulse rate the triage nurse's pulse check records for the patient on arrival",
                ),
                (
                    "the ward's heart-rate display",
                    "the pulse the ward's heart-rate display shows for the patient after the transfer",
                ),
            ],
            "breaths/minute": [
                (
                    "the paramedic's respiratory monitor",
                    "the respiratory rate the paramedic's respiratory monitor records for the patient during the transfer",
                ),
                (
                    "the triage nurse's breathing check",
                    "the breathing rate the triage nurse's breathing check records for the patient on arrival",
                ),
                (
                    "the ward's respiration display",
                    "the respiration rate the ward's respiration display shows for the patient after the transfer",
                ),
            ],
        },
    ),
    dict(
        intro="A fire crew's callout",
        variants={
            "km/h": [
                (
                    "the fire truck's dashboard speedometer",
                    "the speed of the fire truck on the fire truck's dashboard speedometer",
                ),
                (
                    "the fire truck's on-board trip display",
                    "the speed of the fire truck on the fire truck's on-board trip display",
                ),
                (
                    "the electronic speed sign on the highway",
                    "the speed of the fire truck on the electronic speed sign on the highway",
                ),
            ],
            "m/s": [
                (
                    "the physics class's braking problem",
                    "the speed of the fire truck in the physics class's braking problem, set in SI units",
                ),
                (
                    "the engineering crash-test report",
                    "the speed of the fire truck in the engineering crash-test report, which uses SI units",
                ),
                (
                    "the fire-engineering journal article",
                    "the speed of the fire truck in the fire-engineering journal article, which reports in SI units",
                ),
            ],
            "L/minute": [
                ("the hose nozzle's flow meter", "the flow rate the hose nozzle's flow meter records on scene"),
                ("the pump panel's flow meter", "the flow rate the pump panel's flow meter shows for the hose line"),
                (
                    "the brigade's hose-reel specification",
                    "the flow rate the brigade's hose-reel specification states for the hose reel",
                ),
            ],
            "beats/minute": [
                (
                    "the crew medic's pulse check",
                    "the pulse rate the crew medic's pulse check records for a firefighter after the callout",
                ),
                (
                    "the firefighter's heart-rate watch",
                    "the heart rate the firefighter's heart-rate watch logs during the callout",
                ),
                (
                    "the ambulance officer's cardiac monitor",
                    "the heart rate the ambulance officer's cardiac monitor records for a rescued resident",
                ),
            ],
            "breaths/minute": [
                (
                    "the crew medic's breathing check",
                    "the breathing rate the crew medic's breathing check records for a firefighter after the callout",
                ),
                (
                    "the breathing-apparatus monitor",
                    "the breathing rate the breathing-apparatus monitor logs for a firefighter during the callout",
                ),
                (
                    "the ambulance officer's respiratory monitor",
                    "the respiratory rate the ambulance officer's respiratory monitor records for a rescued resident",
                ),
            ],
        },
    ),
    dict(
        intro="A household's water and weather day",
        variants={
            "km/h": [
                (
                    "the weather bureau's public forecast",
                    "the speed of a gust of wind in the weather bureau's public forecast",
                ),
                ("the local news weather segment", "the speed of a gust of wind in the local news weather segment"),
                (
                    "the storm warning issued to the public",
                    "the speed of a gust of wind in the storm warning issued to the public",
                ),
            ],
            "m/s": [
                (
                    "the meteorology research paper's data table",
                    "the speed of a gust of wind in the meteorology research paper's data table, which uses SI units",
                ),
                (
                    "the physics class's drag problem",
                    "the speed of a gust of wind in the physics class's drag problem, set in SI units",
                ),
                (
                    "the wind-engineering report",
                    "the speed of a gust of wind in the wind-engineering report, which uses SI units",
                ),
            ],
            "L/minute": [
                ("the WELS label on the showerhead", "the flow rate the WELS label on the showerhead states"),
                ("the plumber's flow meter", "the flow rate the plumber's flow meter reads for the kitchen tap"),
                ("the garden hose's bucket-fill test", "the flow rate found from the garden hose's bucket-fill test"),
            ],
        },
    ),
    dict(
        intro="A racehorse's sprint race",
        variants={
            "km/h": [
                (
                    "the racing broadcast's on-screen graphic",
                    "the speed of the racehorse on the racing broadcast's on-screen graphic",
                ),
                (
                    "the newspaper's race report",
                    "the speed of the racehorse in the newspaper's race report for the general public",
                ),
                (
                    "the commentator's on-air remark",
                    "the speed of the racehorse in the commentator's on-air remark to viewers",
                ),
            ],
            "m/s": [
                (
                    "the equine biomechanics lab's SI-unit table",
                    "the speed of the racehorse in the equine biomechanics lab's SI-unit table",
                ),
                (
                    "the physics class's motion problem",
                    "the speed of the racehorse in the physics class's motion problem, set in SI units",
                ),
                (
                    "the veterinary sports-science journal article",
                    "the speed of the racehorse in the veterinary sports-science journal article, which reports in SI units",
                ),
            ],
            "beats/minute": [
                (
                    "the vet's pulse check",
                    "the pulse rate the vet's pulse check records for the racehorse after the race",
                ),
                (
                    "the girth-strap heart monitor",
                    "the heart rate the girth-strap heart monitor logs for the racehorse during the race",
                ),
                (
                    "the stable's cardiac monitor",
                    "the pulse the stable's cardiac monitor records for the racehorse after the race",
                ),
            ],
            "breaths/minute": [
                (
                    "the vet's breathing check",
                    "the breathing rate the vet's breathing check records for the racehorse after the race",
                ),
                (
                    "the trainer's respiratory monitor",
                    "the respiratory rate the trainer's respiratory monitor records for the racehorse after the race",
                ),
                (
                    "the stable's respiration sensor",
                    "the respiration rate the stable's respiration sensor records for the racehorse after the race",
                ),
            ],
        },
    ),
]

_EXAM_QUESTION_TEMPLATES = [
    "Considering only what {device} reports, which unit is conventionally used to report it?",
    "Focusing only on the reading given by {device}, which unit is conventionally used to report it?",
    "Taking only the figure {device} gives, which unit is conventionally used to report it?",
]


def _validate_scenarios() -> None:
    """Every device string must be a verbatim substring of exactly its own
    clause, and never of any other clause in the same scenario -- the exam
    tier's reference-by-name mechanism (and the independent oracle) depends on
    this being unambiguous. Every scenario carries both speed keys."""
    for scenario in _SCENARIOS:
        v = scenario["variants"]
        check("km/h" in v and "m/s" in v, f"scenario {scenario['intro']!r} lacks a speed key")
        for key, variants in v.items():
            check(key in _KEYS, f"unknown key {key!r}")
            check(len(variants) >= 3, f"scenario {scenario['intro']!r} key {key!r} needs >= 3 variants")
        for key, variants in v.items():
            for device, clause in variants:
                check(device in clause, f"device {device!r} not found in its own clause {clause!r}")
                for other_key, other_variants in v.items():
                    for other_device, other_clause in other_variants:
                        if (other_key, other_clause) == (key, clause):
                            continue
                        check(device not in other_clause, f"device {device!r} ambiguously also in {other_clause!r}")
                        check(other_key == key or device != other_device, f"device {device!r} reused across keys")


_validate_scenarios()


def _draw_key(rng) -> str:
    """Construct-to-target: draw {speed, non-speed} uniformly first, then
    the specific key uniformly within that type."""
    type_ = rng.choice(["speed", "non-speed"])
    key = rng.choice(_TYPE_KEYS[type_])
    # Independent re-derivation: recompute the type from the drawn key via
    # a SEPARATE lookup rather than trusting the draw path's own labelling.
    check(_KEY_TYPE[key] == type_, f"rate-usage type/key mismatch: drew {type_!r}, key {key!r}")
    return key


def _build_easy(rng) -> tuple[str, str]:
    key = _draw_key(rng)
    phrase = rng.choice(_EASY_POOL[key])
    return key, f"Which unit is conventionally used to report {phrase}?"


def _build_medium(rng) -> tuple[str, str]:
    key = _draw_key(rng)
    phrase = rng.choice(_MEDIUM_POOL[key])
    return key, f"Which unit is conventionally used to report {phrase}?"


def _build_hard(rng) -> tuple[str, str]:
    key = _draw_key(rng)
    eligible = [s for s in _SCENARIOS if key in s["variants"]]
    scenario = rng.choice(eligible)
    partner = rng.choice([k for k in scenario["variants"] if k != key])
    clause_key = rng.choice(scenario["variants"][key])[1]
    clause_partner = rng.choice(scenario["variants"][partner])[1]
    graded_first = rng.choice([True, False])
    if graded_first:
        a, b, label = clause_key, clause_partner, "first"
    else:
        a, b, label = clause_partner, clause_key, "second"
    text = (
        f"{scenario['intro']} is reported in two ways: {a}, and separately {b}."
        f" Considering only the {label} of these two, which unit is conventionally used to report it?"
    )
    return key, text


def _build_exam(rng) -> tuple[str, str]:
    key = _draw_key(rng)
    eligible = [s for s in _SCENARIOS if key in s["variants"]]
    scenario = rng.choice(eligible)
    variants = scenario["variants"]
    # Always both speed conventions plus one non-speed quantity.
    if _KEY_TYPE[key] == "non-speed":
        third = key
    else:
        third = rng.choice([k for k in variants if _KEY_TYPE[k] == "non-speed"])
    order = ["km/h", "m/s", third]
    rng.shuffle(order)
    picked = {k: rng.choice(variants[k]) for k in order}
    target_device = picked[key][0]
    joined = "; ".join(picked[k][1] for k in order)
    question = rng.choice(_EXAM_QUESTION_TEMPLATES).format(device=target_device)
    text = f"{scenario['intro']} is written up with three figures: {joined}. {question}"
    return key, text


def _pick_distractors(rng, key: str) -> list[str]:
    """Key's own plausible-unit list only; the other speed unit is always
    offered for a speed key. 2 to (list size) distractors, so 3-5 options."""
    plausible = list(_PLAUSIBLE_OPTIONS[key])
    mandatory = [_MANDATORY_DISTRACTOR[key]] if key in _MANDATORY_DISTRACTOR else []
    rest = [u for u in plausible if u not in mandatory]
    n = rng.randint(max(_MIN_DISTRACTORS, len(mandatory)), len(plausible))
    picks = mandatory + rng.sample(rest, n - len(mandatory))
    rng.shuffle(picks)
    return picks


def _build_rate_usage_classification(rng, tier: str) -> dict:
    if tier == "easy":
        key, text = _build_easy(rng)
    elif tier == "medium":
        key, text = _build_medium(rng)
    elif tier == "hard":
        key, text = _build_hard(rng)
    elif tier == "exam":
        key, text = _build_exam(rng)
    else:
        raise ValueError(f"unknown tier {tier!r}")

    check(key in _KEYS, f"rate-usage key {key!r} not in answer space")

    distractors = _pick_distractors(rng, key)
    check(len(distractors) >= _MIN_DISTRACTORS, "rate_usage_classification needs >= 3 options (never binary)")
    check(all(d in _PLAUSIBLE_OPTIONS[key] for d in distractors), "distractor not on the plausible list")
    check_pairwise_distinct(key, *distractors, msg="rate_usage_classification options collided")

    latex = f"plain:{text}"
    working = [f"In this reporting context, {key} is the unit conventionally used to describe this quantity."]
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
    "p19h_rate_usage_classification",
    level=12,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p19h_rate_usage_classification(rng, tier="easy"):
    """Match a quantity, reported in a named reporting context, to the rate
    unit conventionally used there (km/h, m/s, beats/minute, breaths/minute or
    L/minute) -- see the module docstring for the per-tier reasoning
    escalation."""
    return _build_rate_usage_classification(rng, tier)
