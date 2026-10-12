"""Short written-answer generators about simulation, chance variation, networks and induction.

Every question gives a concrete situation (a probability model, a set of repeated runs, a network written as an edge
list, a statement to prove by induction) and asks for a short written answer marked against a rubric. All numbers a
prompt states are computed in the generator. The `answer` of each question is a marking dict (see
`_written_answer_kit`), not a value.

Topics (one per skill):

- `wa_simulation_design`: setting up a simulation of a chance experiment with random numbers from technology,
  reading its output, and judging a simulation plan.
- `wa_simulation_factors`: factors in a real situation that a simple simulation leaves out.
- `wa_chance_variation`: why repeated runs of the same chance experiment differ, with the runs given as data.
- `wa_tree_spanning`: the meaning of a tree and a spanning tree for a network given as an edge list.
- `wa_induction_nature`: the initial statement and the inductive step, and what goes wrong when either is missing.
"""

from __future__ import annotations

from fractions import Fraction
from math import comb

from ..core import VerificationError, check
from ._written_answer_kit import (
    attempts,
    NAME_SLOTS,
    NAMES,
    Case,
    Tier,
    finish,
    marking,
    num_text,
    pct,
    register,
    register_custom,
    values_line,
)

# ---------------------------------------------------------------------------
# wa_simulation_design
# ---------------------------------------------------------------------------
SIM_CTX = (
    ("A netball player", "shot at goal", "shots at goal", "the shot scores a goal", "goals"),
    ("A bus company", "trip", "trips", "the bus arrives on time", "on-time arrivals"),
    ("A seed company", "seed planted", "seeds planted", "the seed germinates", "germinations"),
    ("A basketball player", "free throw", "free throws", "the shot goes in", "successful throws"),
    ("A factory machine", "item made", "items made", "the item passes inspection", "passes"),
)


def _prob_at_least(k10: int, m: int, j: int) -> Fraction:
    p = Fraction(k10, 10)
    return sum(comb(m, i) * p**i * (1 - p) ** (m - i) for i in range(j, m + 1))


def _sim_slots() -> dict:
    return {
        **NAME_SLOTS,
        "ctx": SIM_CTX,
        "actor": lambda r, v: v["ctx"][0],
        "att": lambda r, v: v["ctx"][1],
        "atts": lambda r, v: v["ctx"][2],
        "succ": lambda r, v: v["ctx"][3],
        "nouns": lambda r, v: v["ctx"][4],
        "k": [3, 4, 5, 6, 7, 8],
        "km1": lambda r, v: v["k"] - 1,
        "p": lambda r, v: f"0.{v['k']}",
        "pp": lambda r, v: f"{v['k'] * 10}%",
        "m": [3, 4, 5],
        "j": lambda r, v: r.randint(2, v["m"]),
        "N": [100, 200, 500],
        "theory": lambda r, v: num_text(_prob_at_least(v["k"], v["m"], v["j"]), 3),
        "hits": lambda r, v: (
            round(float(_prob_at_least(v["k"], v["m"], v["j"])) * v["N"]) + r.choice([-6, -5, -4, -3, 3, 4, 5, 6])
        ),
        "est": lambda r, v: num_text(Fraction(v["hits"], v["N"]), 3),
        "badk": lambda r, v: v["k"] + 1,
        "bad_p": lambda r, v: f"0.{v['k'] + 1}" if v["k"] < 9 else "1.0",
    }


SIM_FIELDS = dict(
    setup="{actor} has a probability of {p} that {succ} on each {att}, and the {atts} are independent. {person} wants to estimate, using technology, the probability that at least {j} of {m} {atts} are successful.",
    event="at least {j} of the {m} {atts} succeed",
    plan_ok="no",
)


def _plan(id_: str, sound: bool, plan: str, flaw: str, why: str, fix: str) -> Case:
    return Case(
        id_,
        slots=_sim_slots(),
        fields={**SIM_FIELDS, "plan": plan, "flaw": flaw, "why": why, "fix": fix, "sound": "yes" if sound else "no"},
        wrong=[
            (
                f"accepts_{id_}",
                "does it accept this plan without checking each step against the probability and the event?",
            )
        ],
    )


SIM_BASE = Case("base", slots=_sim_slots(), fields=dict(SIM_FIELDS), wrong=[])
SIM_PLANS = [
    _plan(
        "sound_plan",
        True,
        "Generate {m} random digits from 0 to 9, counting the digits 0 to {km1} as successes; count the successes in the trial; repeat for {N} trials and divide the number of trials with at least {j} successes by {N}.",
        "none: the plan is sound",
        "digits 0 to {km1} are {k} of the 10 equally likely digits, so each digit is a success with probability {p}; a trial is {m} digits, matching {m} {atts}; the event is counted in each trial; and {N} trials is a reasonable number",
        "no change needed, although more trials would reduce random variation",
    ),
    _plan(
        "wrong_mapping",
        False,
        "Generate {m} random digits from 0 to 9, counting the digits 0 to {k} as successes; count the successes in the trial; repeat for {N} trials and divide the number of trials with at least {j} successes by {N}.",
        "the digits are mapped to the wrong probability",
        "digits 0 to {k} are {badk} of the 10 digits, so each digit is a success with probability {bad_p}, not {p}, and the estimate will be too high",
        "count only the digits 0 to {km1} as successes, which is {k} of the 10 digits",
    ),
    _plan(
        "one_digit_trials",
        False,
        "Generate {N} random digits from 0 to 9, counting the digits 0 to {km1} as successes, and divide the number of successes by {N}.",
        "a trial is a single digit, so the event is never tested",
        "this estimates the probability of one success ({p}), not the probability of at least {j} successes in {m} {atts}",
        "make each trial {m} digits, count the successes in the trial, and count the trials with at least {j} successes",
    ),
    _plan(
        "too_few_trials",
        False,
        "Generate {m} random digits from 0 to 9, counting the digits 0 to {km1} as successes; count the successes in the trial; repeat for 8 trials and divide the number of trials with at least {j} successes by 8.",
        "far too few trials",
        "with only 8 trials the estimate can only be 0, 0.125, 0.25 and so on, and will vary a great deal from one run to the next",
        "run many more trials, such as {N} or more, so that the relative frequency settles near the true probability",
    ),
    _plan(
        "stops_early",
        False,
        "Generate {m} random digits from 0 to 9, counting the digits 0 to {km1} as successes. Repeat trials until one trial has at least {j} successes, then stop and report 1 divided by the number of trials used.",
        "the simulation stops as soon as the event occurs",
        "stopping at the first success measures how long it takes for the event to happen, not how often it happens, so the number is not an estimate of the probability",
        "decide the number of trials in advance, run them all, and divide the number of successful trials by the total",
    ),
]


_PLAN_BY_ID = {c.id: c for c in SIM_PLANS}
_FLAWED = ("wrong_mapping", "one_digit_trials", "too_few_trials", "stops_early")


def _plan_slot(key: str, field_: str):
    def f(rng, v):
        case = _PLAN_BY_ID["sound_plan" if key == "sound" else v["bad_id"]]
        return case.fields[field_].format(**v)

    return f


SIM_EXAM = Case(
    "exam",
    slots={
        **_sim_slots(),
        "bad_id": list(_FLAWED),
        "sound_first": [True, False],
        "sound_plan_text": _plan_slot("sound", "plan"),
        "sound_why": _plan_slot("sound", "why"),
        "bad_plan_text": _plan_slot("bad", "plan"),
        "flawed_flaw": _plan_slot("bad", "flaw"),
        "flawed_why": _plan_slot("bad", "why"),
        "flawed_fix": _plan_slot("bad", "fix"),
        "first": lambda r, v: v["person"],
        "second": lambda r, v: v["other"],
        "sound_who": lambda r, v: v["person"] if v["sound_first"] else v["other"],
        "flawed_who": lambda r, v: v["other"] if v["sound_first"] else v["person"],
        "first_plan": lambda r, v: v["sound_plan_text"] if v["sound_first"] else v["bad_plan_text"],
        "second_plan": lambda r, v: v["bad_plan_text"] if v["sound_first"] else v["sound_plan_text"],
    },
    fields=dict(SIM_FIELDS),
    wrong=[],
)

SIM_TIERS = {
    "easy": Tier(
        prompt="{setup} Explain how to simulate one {att}, using a random number generator that produces whole numbers from 0 to 9, each equally likely.",
        reference=[
            "Use the digits 0 to {km1} to mean the event happens ({k} of the 10 digits, a probability of {p}) and the other digits to mean it does not.",
            "Generate one random digit for each {att}; digits from 0 to {km1} give a success and any other digit gives a failure.",
        ],
        points=[
            (
                "correct_idea",
                "assigns exactly {k} of the 10 digits to success, such as 0 to {km1}, so the probability is {p}",
            ),
            ("correct_idea", "assigns the remaining digits to failure"),
            ("use_of_context", "describes one {att} in the situation as one random digit"),
            ("completeness", "says each digit is equally likely and each {att} uses a new random digit"),
            ("communication", "states the rule clearly enough that someone else could follow it"),
        ],
        wrong=[
            ("wrong_number_of_digits", "does it assign a number of digits other than {k} out of 10 to success?"),
            ("reuses_one_digit", "does it use the same digit for several attempts instead of a new digit each time?"),
        ],
        answer="digits 0 to {km1} are successes, the others failures",
        cases=[SIM_BASE],
    ),
    "medium": Tier(
        prompt="{setup} Describe how {person} could carry out the simulation using technology: say what one trial is, what is recorded for each trial, how many trials to use, and how the probability is estimated.",
        reference=[
            "One trial is {m} random digits from 0 to 9, with the digits 0 to {km1} counted as successes.",
            "For each trial, record the number of successes, and note whether it is at least {j}.",
            "Repeat for many trials, for example {N}, and estimate the probability as (the number of trials with at least {j} successes) divided by {N}.",
        ],
        points=[
            (
                "correct_idea",
                "defines a trial as {m} random digits, one for each {att}, with {k} of the 10 digits meaning success",
            ),
            ("correct_idea", "records the number of successes in each trial"),
            ("use_of_context", "counts the trials where there are at least {j} successes"),
            (
                "completeness",
                "uses a large number of trials (such as {N}) and gives the estimate as a relative frequency",
            ),
            ("communication", "describes the steps in order"),
        ],
        wrong=[
            (
                "estimates_single_success",
                "does it estimate the probability of a single success instead of the event 'at least {j} of {m}'?",
            ),
            ("one_trial_only", "does it use only one trial and treat the result as the probability?"),
        ],
        answer="{m} digits per trial, count trials with at least {j} successes, divide by the number of trials",
        cases=[SIM_BASE],
    ),
    "hard": Tier(
        prompt="{setup} {person} runs a simulation of {N} trials. In {hits} of the {N} trials, at least {j} of the {m} {atts} were successful. A calculation of the exact probability gives {theory}. Use these results to give the simulation's estimate, and explain why it is not exactly equal to {theory} and what could be done to get a closer estimate.",
        reference=[
            "The estimate is {hits} divided by {N}, which is {est}.",
            "It is not exactly {theory} because the simulation uses random numbers, so each run of {N} trials gives a slightly different relative frequency.",
            "Running many more trials, or repeating the simulation and combining the results, would give an estimate closer to {theory}.",
        ],
        points=[
            ("correct_idea", "gives the estimate {est} as {hits} out of {N}"),
            ("correct_idea", "explains that random variation between runs makes the estimate differ from {theory}"),
            ("use_of_context", "compares the estimate {est} with the exact value {theory}"),
            ("completeness", "suggests more trials or repeated simulations to get closer"),
            ("communication", "explains the difference without saying the simulation is wrong"),
        ],
        wrong=[
            (
                "simulation_is_wrong",
                "does it say the simulation must be faulty because the estimate is not exactly {theory}?",
            ),
            ("fewer_trials_better", "does it say fewer trials would improve the estimate?"),
        ],
        answer="{est}; random variation; run more trials",
        cases=[SIM_BASE],
    ),
    "exam": Tier(
        prompt=(
            "{setup}\n"
            "Two students describe how they would simulate this.\n"
            "{first}'s plan: {first_plan}\n"
            "{second}'s plan: {second_plan}\n"
            "Evaluate both plans. Say which one would give a sound estimate, and for the other say what is wrong and how to fix it."
        ),
        reference=[
            "The sound plan is {sound_who}'s: {sound_why}.",
            "{flawed_who}'s plan has a flaw: {flawed_flaw}, because {flawed_why}.",
            "Fix: {flawed_fix}.",
        ],
        points=[
            ("correct_idea", "identifies {sound_who}'s plan as sound and checks it step by step"),
            ("correct_idea", "identifies the flaw in {flawed_who}'s plan: {flawed_flaw}"),
            ("use_of_context", "explains why it matters for this situation: {flawed_why}"),
            ("completeness", "gives a fix: {flawed_fix}"),
            ("communication", "evaluates the two plans separately"),
        ],
        wrong=[
            ("approves_both", "does it approve both plans?"),
            ("wrong_plan_chosen", "does it call the flawed plan sound, or the sound plan flawed?"),
        ],
        answer="{sound_who}'s plan is sound; {flawed_who}'s plan: {flawed_flaw}",
        cases=[SIM_EXAM],
    ),
}
register(
    "wa_simulation_design",
    13,
    [SIM_BASE],
    SIM_TIERS,
    doc="setting up and judging a simulation.",
)

# ---------------------------------------------------------------------------
# wa_simulation_factors
# ---------------------------------------------------------------------------
FACTOR_CASES = [
    Case(
        "cafe_queue",
        slots={
            **NAME_SLOTS,
            "rate": [1, 2, 3],
            "svc": [2, 3, 4],
            "name": ["Corner Cafe", "Station Kiosk", "Harbour Bakery"],
        },
        fields=dict(
            setup="{person} simulates the queue at {name}. The simulation assumes one customer arrives every minute on average, at random, and each customer takes {svc} minutes to serve.",
            factor="the rate of arrivals changes through the day",
            effect="the simulation underestimates the queue in the lunchtime rush, when many more customers arrive, and overestimates it in quiet periods",
            change="use a higher arrival rate for the rush period and a lower one for quiet periods, instead of one constant rate",
            limit="the arrival rates in each period still have to be estimated from real counts",
            factor2="groups of friends arrive together, so arrivals are not independent",
        ),
        wrong=[("only_service_time", "does it only mention the serving time, which the simulation already includes?")],
    ),
    Case(
        "rain",
        slots={**NAME_SLOTS, "p": [20, 30, 40], "town": ["Bellmore", "Corrington", "Easton Bay"]},
        fields=dict(
            setup="{person} simulates the weather in {town} for 30 days. The simulation assumes that each day has a {p}% chance of rain, independent of every other day.",
            factor="wet days tend to follow wet days, so the days are not independent",
            effect="the simulation spreads the rainy days evenly, but real wet days come in runs, so it underestimates the chance of a long wet spell",
            change="make the chance of rain depend on whether it rained the day before, for example a higher chance after a wet day",
            limit="the new chances have to be estimated from past weather records",
            factor2="rainfall is seasonal, so a fixed {p}% is wrong for much of the year",
        ),
        wrong=[
            (
                "more_days_fix",
                "does it suggest simulating more days, when the problem is the assumption about each day?",
            )
        ],
    ),
    Case(
        "disease",
        slots={
            **NAME_SLOTS,
            "r": [2, 3],
            "school": ["Hillcrest High School", "Ridgeway College", "Bayview High School"],
        },
        fields=dict(
            setup="{person} simulates how a cold spreads through {school}. The simulation assumes that every infected student passes the cold to exactly {r} other students, who are chosen at random from the whole school.",
            factor="students mostly mix with their own class and friends, not randomly with the whole school",
            effect="the simulation spreads the cold faster across the school than it really would, and misses the way a class can be hit hard while others escape",
            change="group the students into classes and let students infect classmates with a higher chance than students from other classes",
            limit="the chances of infection within and between classes are not known exactly",
            factor2="students who are ill stay home, which removes them from contact",
        ),
        wrong=[("fixed_number_ok", "does it say the simulation is realistic as long as the number {r} is right?")],
    ),
    Case(
        "traffic",
        slots={**NAME_SLOTS, "cars": [10, 15, 20], "road": ["Bridge Road", "High Street", "Station Road"]},
        fields=dict(
            setup="{person} simulates the traffic arriving at the lights on {road}. The simulation assumes that {cars} cars arrive in each minute at random, at an even rate all day.",
            factor="traffic is heavier at rush hours, and the lights make cars arrive in bunches",
            effect="the simulation underestimates delays at the peak and overestimates them in the middle of the night",
            change="use different arrival rates for peak and off-peak times, and let cars arrive in groups after the lights change",
            limit="the groups and peaks must be estimated from counts taken on the road",
            factor2="accidents and roadworks block lanes at unpredictable times",
        ),
        wrong=[
            (
                "random_is_enough",
                "does it say that random arrivals are already realistic, without noting that the rate changes?",
            )
        ],
    ),
    Case(
        "machine",
        slots={**NAME_SLOTS, "pf": [1, 2, 5], "firm": ["Redhill Foods", "Mapleton Printing", "Coastline Packaging"]},
        fields=dict(
            setup="{person} simulates how often a machine at {firm} breaks down. The simulation assumes that on every day the machine has a {pf}% chance of breaking down, however old it is.",
            factor="machines wear out, so the chance of breaking down rises with age",
            effect="the simulation underestimates the number of breakdowns in the later years and overestimates them in the early years",
            change="make the chance of a breakdown increase with the age of the machine, and reset it after a repair",
            limit="the way the chance rises with age has to be estimated from the maintenance records",
            factor2="a breakdown stops production, which changes how much the machine is used afterwards",
        ),
        wrong=[("constant_chance_fine", "does it accept a constant daily chance for a machine that wears out?")],
    ),
    Case(
        "match",
        slots={**NAME_SLOTS, "pg": [55, 60, 65], "team": ["the Hawks", "the Wolves", "the Comets"]},
        fields=dict(
            setup="{person} simulates a season for {team}. The simulation assumes the team has a {pg}% chance of winning every game, whoever the opponent is.",
            factor="the chance of winning depends on the opponent, injuries and whether the game is at home",
            effect="the simulation makes every game equally likely to be won, so it gives too little chance to a bad run against strong teams and too much chance of an unbeaten run",
            change="use a different chance of winning for each opponent and for home and away games, and lower it when key players are injured",
            limit="the individual chances have to be estimated from limited past results",
            factor2="players tire during a long season, which changes the chances late in the year",
        ),
        wrong=[("one_chance_ok", "does it accept one fixed chance for every game?")],
    ),
]
FACTOR_TIERS = {
    "easy": Tier(
        prompt="{setup} Identify one real-world factor that this simulation leaves out.",
        reference=[
            "The simulation leaves out the fact that {factor}.",
            "Another factor it leaves out is that {factor2}.",
        ],
        points=[
            ("correct_idea", "names a real factor the simulation does not include, such as: {factor}"),
            ("use_of_context", "ties the factor to this situation rather than to simulations in general"),
            ("completeness", "explains in a sentence why this is not covered by the assumption stated"),
            ("communication", "names one factor clearly rather than listing many vague ones"),
        ],
        wrong=[("describes_an_assumption_already_made", "does it name a feature the simulation already includes?")],
        answer="{factor}",
    ),
    "medium": Tier(
        prompt="{setup} Identify a factor that the simulation leaves out, and explain how it could make the simulation's results wrong.",
        reference=["Factor: {factor}.", "Effect on the results: {effect}."],
        points=[
            ("correct_idea", "identifies the factor: {factor}"),
            ("use_of_context", "explains how it would change the results here: {effect}"),
            ("completeness", "says in which direction or at what time the results are wrong"),
            ("communication", "links the factor to its effect with a reason"),
        ],
        wrong=[("factor_without_effect", "does it name a factor but not say how the results would be wrong?")],
        answer="{factor}; {effect}",
    ),
    "hard": Tier(
        prompt="{setup} Explain one factor that makes this simulation unrealistic, describe how the simulation could be changed to include it, and state one limitation that would remain.",
        reference=["Factor: {factor}.", "Change to the simulation: {change}.", "A limitation that remains: {limit}."],
        points=[
            ("correct_idea", "explains the factor: {factor}"),
            ("correct_idea", "describes a change that builds it in: {change}"),
            ("use_of_context", "uses the details of the situation"),
            ("completeness", "gives a genuine remaining limitation: {limit}"),
            ("communication", "answers the three parts in order"),
        ],
        wrong=[
            ("change_does_not_address_factor", "does the proposed change actually deal with the factor named?"),
            ("no_limitation", "does it claim the modified simulation would be perfectly realistic?"),
        ],
        answer="{change}",
    ),
    "exam": Tier(
        prompt='{setup} {person} says: "If I run 10 000 trials the simulation will be accurate." {other} says: "More trials will not fix it, because the simulation leaves something out." Evaluate both claims, naming a factor that supports your answer.',
        reference=[
            "{other} is right: more trials only reduce random variation within the simulation; they do not correct a wrong assumption.",
            "A factor the simulation leaves out: {factor}; {effect}.",
            "{person} is wrong: the 10 000 trials would give a precise but still wrong answer.",
        ],
        points=[
            (
                "correct_idea",
                "judges {other} right and {person} wrong, explaining that more trials do not fix a wrong model",
            ),
            ("correct_idea", "names a factor that the simulation omits: {factor}"),
            ("use_of_context", "explains its effect on the results: {effect}"),
            (
                "completeness",
                "distinguishes random variation (reduced by more trials) from a wrong assumption (not reduced)",
            ),
            ("communication", "responds to each person in turn"),
        ],
        wrong=[
            ("agrees_with_more_trials", "does it accept that 10 000 trials makes the simulation accurate?"),
            (
                "dismisses_simulation",
                "does it say simulation is useless instead of saying what it can and cannot show?",
            ),
        ],
        answer="{other} is right: more trials cannot fix a wrong assumption",
    ),
}
register(
    "wa_simulation_factors",
    13,
    FACTOR_CASES,
    FACTOR_TIERS,
    doc="what a simple simulation leaves out.",
)

# ---------------------------------------------------------------------------
# wa_chance_variation
# ---------------------------------------------------------------------------
CHANCE_CTX = (
    dict(
        act="a fair coin is tossed", ev="lands heads", noun="heads", p=Fraction(1, 2), ns=(20, 30, 40, 50), ptext="1/2"
    ),
    dict(
        act="a fair six-sided die is rolled",
        ev="shows a six",
        noun="sixes",
        p=Fraction(1, 6),
        ns=(30, 60, 90),
        ptext="1/6",
    ),
    dict(
        act="a spinner with four equal sectors is spun",
        ev="lands on red",
        noun="reds",
        p=Fraction(1, 4),
        ns=(20, 40, 60),
        ptext="1/4",
    ),
    dict(
        act="a card is drawn at random from a full pack and replaced",
        ev="is a heart",
        noun="hearts",
        p=Fraction(1, 4),
        ns=(20, 40, 60),
        ptext="1/4",
    ),
)


_SINGULAR = {"heads": "head", "sixes": "six", "reds": "red", "hearts": "heart"}


def _runs(rng, ctx: dict, n: int, r: int) -> list[int]:
    p = float(ctx["p"])
    return [sum(rng.random() < p for _ in range(n)) for _ in range(r)]


def _run_lines(counts: list[int], noun: str) -> str:
    one = _SINGULAR.get(noun, noun)
    return "\n".join(f"Run {i + 1}: {c} {one if c == 1 else noun}" for i, c in enumerate(counts))


def _cv_easy(rng) -> dict:
    ctx = rng.choice(CHANCE_CTX)
    n, r = rng.choice(ctx["ns"]), rng.randint(5, 7)
    counts = _runs(rng, ctx, n, r)
    exp = num_text(ctx["p"] * n)
    text = (
        f"In an experiment, {ctx['act']} {n} times and the number of times it {ctx['ev']} is counted. The experiment is repeated {r} times with the same {'coin' if 'coin' in ctx['act'] else 'equipment'}. The results were:\n"
        f"{_run_lines(counts, ctx['noun'])}\n"
        f"The probability of each outcome counted is {ctx['ptext']}, so the expected number of {ctx['noun']} in {n} is {exp}. Explain why the {r} runs gave different numbers of {ctx['noun']}, even though the experiment was the same each time."
    )
    mark = marking(
        reference=[
            f"Each time the experiment is repeated, the outcomes are decided by chance, so the number of {ctx['noun']} changes from run to run.",
            f"The expected number {exp} is a long-run average, not what happens in each run; the counts ({values_line(counts)}) vary around it.",
            "Nothing is wrong with the equipment: this variation is what chance events produce.",
        ],
        points=[
            ("correct_idea", "explains that chance events give different results each time they are repeated"),
            (
                "correct_idea",
                f"says the expected value {exp} is an average over many runs, not a prediction for each run",
            ),
            ("use_of_context", f"refers to the actual counts ({values_line(counts)}) varying around {exp}"),
            ("completeness", "adds that the variation does not mean the equipment is faulty"),
            ("communication", "answers in terms of chance rather than blaming the experimenter"),
        ],
        wrong=[
            ("blames_a_fault", "does it say the equipment or the experimenter must be at fault?"),
            ("expects_the_same_each_time", f"does it expect every run to give exactly {exp}?"),
        ],
    )
    return finish(text, "each run is decided by chance, so the counts vary", mark)


def _cv_medium(rng) -> dict:
    ctx = rng.choice(CHANCE_CTX)
    n, r = rng.choice(ctx["ns"]), rng.randint(6, 8)
    counts = _runs(rng, ctx, n, r)
    lo, hi = min(counts), max(counts)
    exp = num_text(ctx["p"] * n)
    text = (
        f"{r} students each repeated an experiment in which {ctx['act']} {n} times, counting the number of times it {ctx['ev']}. The results were:\n"
        f"{_run_lines(counts, ctx['noun'])}\n"
        f"The expected number of {ctx['noun']} in {n} is {exp}. Find the range of the counts, and explain whether it is surprising that the results vary by this much."
    )
    mark = marking(
        reference=[
            f"The lowest count is {lo} and the highest is {hi}, so the range is {hi - lo}.",
            f"The counts are scattered on both sides of the expected {exp}. This is what chance produces, so it is not surprising that the runs differ by {hi - lo}.",
            "It would be surprising only if the counts were all exactly the same, or very far from the expected number in every run.",
        ],
        points=[
            ("correct_idea", f"finds the range correctly: {hi} - {lo} = {hi - lo}"),
            ("correct_idea", "judges that variation of this size is normal for chance events"),
            ("use_of_context", f"compares the counts with the expected number {exp}"),
            ("completeness", "says what would be surprising (all the same, or consistently far from expected)"),
            ("communication", "gives the range as a number and the judgement in words"),
        ],
        wrong=[
            ("range_wrong", f"is the range anything other than {hi - lo}?"),
            ("says_variation_means_bias", "does it say the variation shows the experiment is biased?"),
        ],
    )
    return finish(text, f"range {hi - lo}; not surprising", mark)


def _cv_hard(rng) -> dict:
    ctx = rng.choice(CHANCE_CTX)
    n, r = rng.choice(ctx["ns"]), rng.randint(6, 8)
    for _ in attempts(200, "a single highest run above the expected count"):
        counts = _runs(rng, ctx, n, r)
        if counts.count(max(counts)) == 1 and max(counts) > ctx["p"] * n:
            break
    i = counts.index(max(counts))
    top = counts[i]
    p = rng.choice(NAMES)
    text = (
        f"In an experiment, {ctx['act']} {n} times and the number of times it {ctx['ev']} is counted. This was repeated {r} times. The numbers of {ctx['noun']} were:\n"
        f"{_run_lines(counts, ctx['noun'])}\n"
        f'The probability of the outcome counted is {ctx["ptext"]}. {p} says: "Run {i + 1} gave {top} out of {n} ({pct(top, n)}), so the probability must be more than {ctx["ptext"]}." '
        f"Evaluate {p}'s claim, and say what you would expect if each run had ten times as many trials."
    )
    mark = marking(
        reference=[
            f"{p} is wrong to conclude this from one run: a single run of {n} can easily differ from {ctx['ptext']}, and the other runs ({values_line([c for j, c in enumerate(counts) if j != i])}) are lower.",
            f"The highest run is {pct(top, n)} but the other runs give different relative frequencies; chance variation explains this.",
            f"With ten times as many trials ({n * 10}), the relative frequencies of the runs would be closer together and closer to {ctx['ptext']}, although the counts themselves would vary by more.",
        ],
        points=[
            ("correct_idea", f"rejects {p}'s conclusion: one run is not enough to show the probability is different"),
            ("correct_idea", "explains that larger runs have relative frequencies closer to the true probability"),
            (
                "use_of_context",
                f"uses the figures: run {i + 1} gave {top} out of {n} ({pct(top, n)}), and the other runs gave different results",
            ),
            ("completeness", "separates the counts (which may vary more) from the relative frequencies (which settle)"),
            ("communication", f"responds directly to {p}'s statement"),
        ],
        wrong=[
            ("accepts_one_run", f"does it accept {p}'s claim based on one run?"),
            ("larger_runs_identical", "does it say larger runs would all give the same result?"),
        ],
    )
    return finish(text, f"{p} is wrong: one run varies by chance", mark)


def _cv_exam(rng) -> dict:
    ctx = rng.choice(CHANCE_CTX[:1] + CHANCE_CTX[2:3])
    n = rng.choice(ctx["ns"][:2])
    r = 6
    for _ in attempts(400, "two sets of runs with the wanted spreads"):
        a = _runs(rng, ctx, n, r)
        b = _runs(rng, ctx, n * 10, r)
        ra, rb = max(a) - min(a), max(b) - min(b)
        pa = Fraction(100 * (max(a) - min(a)), n)
        pb = Fraction(100 * (max(b) - min(b)), n * 10)
        if rb > ra and pb < pa and ra >= 2:
            break
    p, q = rng.sample(NAMES, 2)
    text = (
        f"Two groups repeated an experiment in which {ctx['act']} and the number of times it {ctx['ev']} was counted. Group A did {r} runs of {n} trials. Group B did {r} runs of {n * 10} trials. The numbers of {ctx['noun']} were:\n"
        f"Group A: {values_line(a)}\nGroup B: {values_line(b)}\n"
        f"The range of Group A's counts is {ra} and the range of Group B's counts is {rb}. {p} says: \"Group B's results vary more, so Group B's experiment is less reliable.\" "
        f'{q} says: "Group B\'s results are the better guide to the probability." Evaluate both claims by comparing the counts as proportions of the number of trials.'
    )
    sa, sb = num_text(pa, 1), num_text(pb, 1)
    mark = marking(
        reference=[
            f"The counts in B vary more in size (range {rb} against {ra}) because B has ten times as many trials.",
            f"As a percentage of the trials, the spread is {sa} percentage points for A and only {sb} for B, so B's relative frequencies are closer together.",
            f"{q} is right: more trials give relative frequencies closer to the true probability; {p} compares counts when the numbers of trials differ.",
        ],
        points=[
            (
                "correct_idea",
                f"explains that counts cannot be compared directly when the number of trials differs ({n} against {n * 10})",
            ),
            (
                "correct_idea",
                f"compares the spread as a proportion of trials: {sa} percentage points for A against {sb} for B",
            ),
            ("use_of_context", f"uses the ranges {ra} and {rb} and the group sizes"),
            (
                "completeness",
                f"judges {q} right and {p} wrong, because more trials give more reliable relative frequencies",
            ),
            ("communication", "evaluates each claim in turn"),
        ],
        wrong=[
            ("agrees_bigger_range_worse", f"does it agree with {p} because the range {rb} is bigger than {ra}?"),
            ("ignores_trial_numbers", "does it ignore that Group B did ten times as many trials in each run?"),
        ],
    )
    return finish(text, f"{q} is right: compare proportions, and B is more reliable", mark)


register_custom(
    "wa_chance_variation",
    13,
    dict(easy=_cv_easy, medium=_cv_medium, hard=_cv_hard, exam=_cv_exam),
    doc="repeated runs give different results.",
)

# ---------------------------------------------------------------------------
# wa_tree_spanning
# ---------------------------------------------------------------------------
VERTS = "ABCDEFGH"


def _components(n: int, edges: list) -> list[set]:
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        parent[find(VERTS.index(a))] = find(VERTS.index(b))
    groups: dict = {}
    for i in range(n):
        groups.setdefault(find(i), set()).add(VERTS[i])
    return sorted(groups.values(), key=lambda s: sorted(s))


def _is_tree(n: int, edges: list) -> bool:
    return len(edges) == n - 1 and len(_components(n, edges)) == 1


def _tree_edges(rng, n: int) -> list:
    return [(VERTS[rng.randrange(i)], VERTS[i]) for i in range(1, n)]


def _extra_edges(rng, n: int, edges: list, k: int) -> list:
    have = {frozenset(e) for e in edges}
    out = list(edges)
    for _ in attempts(200, "extra edges"):
        if len(out) - len(edges) >= k:
            break
        a, b = rng.sample(VERTS[:n], 2)
        if frozenset((a, b)) not in have:
            have.add(frozenset((a, b)))
            out.append((a, b))
    return out


def _find_cycle(n: int, edges: list):
    """A cycle as a list of vertices (first repeated at the end), found by removing each edge in turn."""
    for skip, (a, b) in enumerate(edges):
        rest = edges[:skip] + edges[skip + 1 :]
        path = _path(rest, a, b)
        if path:
            return path + [a]
    return None


def _path(edges: list, a: str, b: str):
    adj: dict = {}
    for x, y in edges:
        adj.setdefault(x, []).append(y)
        adj.setdefault(y, []).append(x)
    stack = [(a, [a])]
    seen = {a}
    while stack:
        v, path = stack.pop()
        if v == b:
            return path
        for w in adj.get(v, []):
            if w not in seen:
                seen.add(w)
                stack.append((w, path + [w]))
    return None


def _edge_text(edges: list, weights: dict | None = None, rng=None) -> str:
    items = list(edges)
    if rng is not None:
        rng.shuffle(items)
    if weights is None:
        return ", ".join(f"{a}{b}" for a, b in items)
    return ", ".join(f"{a}{b} ({weights[frozenset((a, b))]})" for a, b in items)


def _mst_weight(n: int, edges: list, weights: dict) -> int:
    chosen: list = []
    total = 0
    for a, b in sorted(edges, key=lambda e: weights[frozenset(e)]):
        if len(_components(n, chosen + [(a, b)])) < len(_components(n, chosen)):
            chosen.append((a, b))
            total += weights[frozenset((a, b))]
    return total


def _tree_easy(rng) -> dict:
    n = rng.randint(5, 8)
    kind = rng.choice(["tree", "cycle", "forest"])
    base = _tree_edges(rng, n)
    if kind == "tree":
        edges = base
    elif kind == "cycle":
        edges = _extra_edges(rng, n, base, 1)
    else:
        edges = base[: rng.randrange(1, n - 1)] + base[rng.randrange(1, n - 1) :]
        edges = list(dict.fromkeys(edges))
        edges.remove(edges[rng.randrange(len(edges))]) if len(_components(n, edges)) == 1 else None
    check(_is_tree(n, edges) == (kind == "tree"), "classification matches the construction")
    comps = _components(n, edges)
    names = ", ".join(VERTS[:n])
    text = (
        f"A network has {n} vertices ({names}) and {len(edges)} edges: {_edge_text(edges, rng=rng)}. "
        "Is this network a tree? Use the definition of a tree, and the numbers of vertices and edges, in your explanation."
    )
    if kind == "tree":
        ref = [
            f"Yes. The network is connected (every vertex can be reached from every other) and has no cycles.",
            f"It has {n} vertices and {len(edges)} edges, and a tree with {n} vertices always has {n} - 1 = {n - 1} edges.",
        ]
        pts = [
            ("correct_idea", "concludes that the network is a tree"),
            ("correct_idea", "gives the definition: connected with no cycles"),
            ("use_of_context", f"uses the counts: {n} vertices and {len(edges)} edges, which is {n} - 1"),
            ("completeness", "checks that the network is connected as well as counting edges"),
            ("communication", "states the conclusion and the reasons clearly"),
        ]
        ans = "yes, it is a tree"
        wrong = [
            ("counts_only", "does it rely on the edge count alone, without checking that the network is connected?")
        ]
    elif kind == "cycle":
        cyc = _find_cycle(n, edges)
        cyc_text = "-".join(cyc)
        ref = [
            f"No. The network is connected but contains a cycle, for example {cyc_text}, so it is not a tree.",
            f"It has {n} vertices and {len(edges)} edges, more than the {n - 1} edges a tree on {n} vertices has.",
        ]
        pts = [
            ("correct_idea", "concludes that the network is not a tree"),
            ("correct_idea", f"identifies a cycle, such as {cyc_text}, as the reason"),
            ("use_of_context", f"uses the counts: {len(edges)} edges is more than {n} - 1 = {n - 1}"),
            ("completeness", "notes that the network is connected but still not a tree because of the cycle"),
            ("communication", "names the vertices in the cycle"),
        ]
        ans = "no: it contains a cycle"
        wrong = [("says_tree_because_connected", "does it call the network a tree because it is connected?")]
    else:
        groups = " and ".join("{" + ", ".join(sorted(g)) + "}" for g in comps)
        ref = [
            f"No. The network is not connected: its vertices fall into separate pieces {groups}, so it is not a tree.",
            f"It has {n} vertices and {len(edges)} edges, fewer than the {n - 1} edges needed to connect {n} vertices.",
        ]
        pts = [
            ("correct_idea", "concludes that the network is not a tree"),
            ("correct_idea", f"identifies that it is not connected, with separate pieces {groups}"),
            ("use_of_context", f"uses the counts: {len(edges)} edges is fewer than {n} - 1 = {n - 1}"),
            ("completeness", "explains that a tree must be connected as well as having no cycles"),
            ("communication", "names the separate pieces"),
        ]
        ans = "no: it is not connected"
        wrong = [
            (
                "says_tree_because_no_cycles",
                "does it call the network a tree because there are no cycles, ignoring that it is in pieces?",
            )
        ]
    mark = marking(
        reference=ref,
        points=pts,
        wrong=wrong + [("no_definition", "does it give a verdict without using the definition of a tree?")],
    )
    return finish(text, ans, mark)


def _tree_medium(rng) -> dict:
    n = rng.randint(5, 7)
    base = _tree_edges(rng, n)
    edges = _extra_edges(rng, n, base, rng.randint(2, 3))
    kind = rng.choice(["valid", "cycle", "missing"])
    if kind == "valid":
        subset = list(base)
    elif kind == "cycle":
        for _ in attempts(300, "a cyclic subset"):
            drop = rng.randrange(len(base))
            others = [e for e in edges if e not in base]
            new = rng.choice(others)
            subset = base[:drop] + base[drop + 1 :] + [new]
            if not _is_tree(n, subset) and len(subset) == n - 1:
                break
    else:
        subset = list(base)
        subset.pop(rng.randrange(len(subset)))
    ok = _is_tree(n, subset)
    check(ok == (kind == "valid"), "verdict matches the construction")
    comps = _components(n, subset)
    names = ", ".join(VERTS[:n])
    text = (
        f"A network has {n} vertices ({names}) and {len(edges)} edges: {_edge_text(edges, rng=rng)}. "
        f"A spanning tree of a network is a tree that uses all of the vertices. Is the set of edges {_edge_text(subset, rng=rng)} a spanning tree of this network? "
        "Explain, and say how many edges any spanning tree of this network has."
    )
    if ok:
        ref = [
            f"Yes. Any spanning tree of a network with {n} vertices has {n} - 1 = {n - 1} edges, and this set has {len(subset)}.",
            "The set connects all the vertices and has no cycle, so it is a spanning tree.",
        ]
        pts = [
            ("correct_idea", "concludes that it is a spanning tree"),
            ("correct_idea", f"states that a spanning tree has {n - 1} edges (one fewer than the number of vertices)"),
            ("use_of_context", "checks that every vertex is reached and that there is no cycle"),
            ("completeness", f"notes that the set has {len(subset)} edges, matching {n - 1}"),
            ("communication", "gives the verdict and the checks clearly"),
        ]
        ans = f"yes; {n - 1} edges"
    elif kind == "cycle":
        cyc = _find_cycle(n, subset)
        ref = [
            f"No. A spanning tree has {n} - 1 = {n - 1} edges and this set does have {len(subset)}, but it contains the cycle {'-'.join(cyc)}.",
            "Because of the cycle, some vertex is left out and the set is not connected, so it is not a spanning tree.",
        ]
        pts = [
            ("correct_idea", "concludes that it is not a spanning tree"),
            ("correct_idea", f"states that a spanning tree has {n - 1} edges"),
            ("use_of_context", f"finds the cycle {'-'.join(cyc)} in the chosen edges, or a vertex that is not reached"),
            (
                "completeness",
                "explains that the right number of edges is not enough, because the set must be connected with no cycle",
            ),
            ("communication", "names the cycle or the vertex left out"),
        ]
        ans = f"no: it contains a cycle; a spanning tree has {n - 1} edges"
    else:
        groups = " and ".join("{" + ", ".join(sorted(g)) + "}" for g in comps)
        ref = [
            f"No. A spanning tree has {n} - 1 = {n - 1} edges, but this set has only {len(subset)}, and the vertices are in separate pieces {groups}.",
            "A spanning tree must connect every vertex, so this set is not a spanning tree.",
        ]
        pts = [
            ("correct_idea", "concludes that it is not a spanning tree"),
            ("correct_idea", f"states that a spanning tree has {n - 1} edges, and the set has {len(subset)}"),
            ("use_of_context", f"identifies that the vertices are not all connected: {groups}"),
            ("completeness", "explains that a spanning tree must reach every vertex"),
            ("communication", "names the pieces or the vertex left out"),
        ]
        ans = f"no: it does not connect every vertex; a spanning tree has {n - 1} edges"
    mark = marking(
        reference=ref,
        points=pts,
        wrong=[
            (
                "counts_edges_only",
                "does it decide only from the number of edges, without checking for a cycle or an unreached vertex?",
            ),
            ("wrong_edge_count", f"does it say a spanning tree has a number of edges other than {n - 1}?"),
        ],
    )
    return finish(text, ans, mark)


PRACTICAL = (
    ("buildings", "cable", "$ thousand", "A school wants to join {n} buildings with network cables."),
    ("towns", "road", "$ million", "A council wants to join {n} towns by new roads."),
    ("pumping stations", "pipe", "$ thousand", "A water company wants to join {n} pumping stations with pipes."),
)


def _weighted_graph(rng, n: int):
    base = _tree_edges(rng, n)
    edges = _extra_edges(rng, n, base, rng.randint(2, 4))
    for _ in attempts(200, "distinct weights"):
        w = {frozenset(e): rng.randint(2, 15) for e in edges}
        if len(set(w.values())) >= len(edges) - 1:
            return edges, w
    raise VerificationError("no weights")


def _tree_hard(rng) -> dict:
    n = rng.randint(5, 7)
    things, link, unit, lead = rng.choice(PRACTICAL)
    edges, w = _weighted_graph(rng, n)
    total = _mst_weight(n, edges, w)
    text = (
        f"{lead.format(n=n)} The possible {link}s and their costs ({unit}) are: {_edge_text(edges, w, rng)}. "
        f"The cheapest way to join all {n} {things} costs {total} ({unit}) in total. "
        f"Explain why the cheapest set of {link}s must be a spanning tree of this network, and how many {link}s it uses."
    )
    mark = marking(
        reference=[
            f"Every one of the {n} {things} must be joined, so the chosen {link}s must connect all the vertices of the network.",
            f"If the chosen {link}s contained a cycle, one {link} in the cycle could be removed without disconnecting anything, and the cost would fall; so the cheapest set has no cycle.",
            f"A connected network with no cycles that uses all {n} vertices is a spanning tree, and it uses {n} - 1 = {n - 1} {link}s.",
        ],
        points=[
            ("correct_idea", f"explains that the {link}s must connect all {n} {things} (spanning and connected)"),
            (
                "correct_idea",
                "explains that a cycle would contain a redundant, costly link, so the cheapest set has no cycle",
            ),
            (
                "use_of_context",
                f"states that the cheapest set uses {n - 1} {link}s, one fewer than the number of {things}",
            ),
            ("completeness", "names the result as a spanning tree: connected, all vertices, no cycles"),
            ("communication", "gives the reasoning in a logical order"),
        ],
        wrong=[
            ("uses_every_link", f"does it say all {len(edges)} {link}s are used?"),
            ("wrong_number_of_links", f"does it give a number of {link}s other than {n - 1}?"),
        ],
    )
    return finish(text, f"a spanning tree with {n - 1} {link}s", mark)


def _tree_exam(rng) -> dict:
    n = rng.randint(5, 7)
    things, link, unit, lead = rng.choice(PRACTICAL)
    base = _tree_edges(rng, n)
    edges = _extra_edges(rng, n, base, rng.randint(2, 4))
    remove = len(edges) - (n - 1)
    p, q = rng.sample(NAMES, 2)
    text = (
        f"{lead.format(n=n)} The possible {link}s are: {_edge_text(edges, rng=rng)}, which is {len(edges)} {link}s. "
        f"The aim is a network that joins every one of the {n} {things} using as few {link}s as possible. "
        f'{p} says: "Remove any {remove} {link}s and you will have it." {q} says: "Each removal must break a cycle without cutting any {things[:-1]} off." '
        f"Evaluate both claims, say how many {link}s the network will have, and give the name of the network you end with."
    )
    mark = marking(
        reference=[
            f"The final network must be a spanning tree, with {n} - 1 = {n - 1} {link}s, so {len(edges)} - {n - 1} = {remove} {link}s are removed.",
            f"{q} is right: each {link} removed must lie on a cycle, or the network is cut into pieces and some {things[:-1]} is left out.",
            f"{p} is wrong: removing any {remove} {link}s may disconnect the network or leave a cycle, so the choice of {link}s matters.",
        ],
        points=[
            ("correct_idea", f"states that the final network is a spanning tree with {n - 1} {link}s"),
            ("correct_idea", f"states the number removed correctly: {len(edges)} - {n - 1} = {remove}"),
            (
                "use_of_context",
                f"judges {q} right and {p} wrong, because each removal must keep the network connected and break a cycle",
            ),
            ("completeness", "explains that more than one spanning tree is usually possible"),
            ("communication", "evaluates each claim in turn"),
        ],
        wrong=[
            ("agrees_remove_any", f"does it agree with {p} that any {remove} {link}s can be removed?"),
            ("wrong_count", f"does it give a number of {link}s removed other than {remove}?"),
        ],
    )
    return finish(text, f"a spanning tree: remove {remove} {link}s, keeping the network connected", mark)


register_custom(
    "wa_tree_spanning",
    13,
    dict(easy=_tree_easy, medium=_tree_medium, hard=_tree_hard, exam=_tree_exam),
    doc="trees and spanning trees of a network.",
)

# ---------------------------------------------------------------------------
# wa_induction_nature
# ---------------------------------------------------------------------------


def _term(d: int, a: int) -> str:
    c = a - d
    if c == 0:
        return f"{d}n"
    return f"{d}n + {c}" if c > 0 else f"{d}n - {-c}"


def _sum_form(d: int, a: int) -> str:
    c = 2 * a - d
    return f"n({d}n + {c})/2" if c > 0 else f"n({d}n - {-c})/2"


def _family(rng) -> dict:
    """A statement P(n) with its initial statement P(1), P(k) and P(k+1), all written out."""
    kind = rng.choice(["odd", "even", "ap", "pow", "div"])
    if kind == "odd":
        return dict(
            stmt="1 + 3 + 5 + ... + (2n - 1) = n^2",
            p1="1 = 1^2, since both sides equal 1",
            pk="1 + 3 + ... + (2k - 1) = k^2",
            pk1="1 + 3 + ... + (2k - 1) + (2k + 1) = (k + 1)^2",
            key="add (2k + 1) to both sides of P(k): k^2 + 2k + 1 = (k + 1)^2",
        )
    if kind == "even":
        return dict(
            stmt="2 + 4 + 6 + ... + 2n = n(n + 1)",
            p1="2 = 1 × 2, since both sides equal 2",
            pk="2 + 4 + ... + 2k = k(k + 1)",
            pk1="2 + 4 + ... + 2k + (2k + 2) = (k + 1)(k + 2)",
            key="add (2k + 2) to both sides of P(k): k(k + 1) + 2(k + 1) = (k + 1)(k + 2)",
        )
    if kind == "ap":
        d = rng.randint(2, 6)
        a = d + rng.randint(0, 5)
        t1, t2 = a, a + d
        last = _term(d, a)
        nxt = f"{d}(k + 1) + {a - d}" if a > d else f"{d}(k + 1)"
        form = _sum_form(d, a)
        return dict(
            stmt=f"{t1} + {t2} + {a + 2 * d} + ... + ({last}) = {form}",
            p1=f"{t1} = 1({d} + {2 * a - d})/2 = {t1}, since both sides equal {t1}",
            pk=f"{t1} + {t2} + ... + ({_term(d, a).replace('n', 'k')}) = {form.replace('n', 'k')}",
            pk1=f"{t1} + {t2} + ... + ({_term(d, a).replace('n', 'k')}) + ({nxt}) = {form.replace('n', '(k + 1)')}",
            key=f"add the next term, {nxt}, to both sides of P(k) and simplify the right side",
        )
    if kind == "pow":
        m = rng.choice([3, 4, 5, 6, 7, 8, 9])
        return dict(
            stmt=f"{m}^n - 1 is divisible by {m - 1}",
            p1=f"{m}^1 - 1 = {m - 1}, which is divisible by {m - 1}",
            pk=f"{m}^k - 1 is divisible by {m - 1}",
            pk1=f"{m}^(k + 1) - 1 is divisible by {m - 1}",
            key=f"write {m}^(k + 1) - 1 = {m}({m}^k - 1) + {m - 1}, which is a multiple of {m - 1} plus {m - 1}",
        )
    triple = rng.choice([(4, 5, 3), (7, 5, 6), (10, 8, 9), (4, 2, 3), (7, 11, 6)])
    b, c, dd = triple
    check(all((b**n + c) % dd == 0 for n in range(1, 10)) and ((b - 1) * c) % dd == 0, "divisibility family is valid")
    return dict(
        stmt=f"{b}^n + {c} is divisible by {dd}",
        p1=f"{b}^1 + {c} = {b + c}, which is divisible by {dd}",
        pk=f"{b}^k + {c} is divisible by {dd}",
        pk1=f"{b}^(k + 1) + {c} is divisible by {dd}",
        key=f"write {b}^(k + 1) + {c} = {b}({b}^k + {c}) - {(b - 1) * c}, a multiple of {dd} minus a multiple of {dd}",
    )


def _ind_easy(rng) -> dict:
    f = _family(rng)
    text = (
        f"A student wants to prove by induction that, for every positive integer n: {f['stmt']}. "
        "State the initial statement (the statement for n = 1) and show that it is true, then say what is assumed in the inductive step."
    )
    mark = marking(
        reference=[
            f"The initial statement is P(1): {f['p1']}. So it is true for n = 1.",
            f"In the inductive step, assume P(k) is true for some positive integer k: {f['pk']}.",
            "Then show, using that assumption, that P(k + 1) is true.",
        ],
        points=[
            ("correct_idea", f"states and checks the initial statement: {f['p1']}"),
            ("correct_idea", f"says the inductive step assumes P(k), that is, {f['pk']}"),
            ("use_of_context", "writes the statements using the given formula, not a generic description"),
            ("completeness", "adds that the aim is then to show P(k + 1) follows from P(k)"),
            ("communication", "labels the initial statement and the assumption separately"),
        ],
        wrong=[
            ("assumes_p_k_plus_1", "does it assume the statement for k + 1, which is what has to be shown?"),
            ("assumes_all_n", "does it assume the statement is true for every n, which is the conclusion?"),
        ],
    )
    return finish(text, f"P(1): {f['p1']}; assume P(k)", mark)


def _ind_medium(rng) -> dict:
    f = _family(rng)
    text = (
        f"A student wants to prove by induction that, for every positive integer n: {f['stmt']}. "
        "Write out P(k) and P(k + 1) for this statement, and explain what the inductive step has to show and how P(k) is used in it."
    )
    mark = marking(
        reference=[
            f"P(k): {f['pk']}.",
            f"P(k + 1): {f['pk1']}.",
            f"The inductive step has to show that if P(k) is true then P(k + 1) is true. P(k) is used by working from it: {f['key']}.",
        ],
        points=[
            ("correct_idea", f"writes P(k) correctly: {f['pk']}"),
            ("correct_idea", f"writes P(k + 1) correctly: {f['pk1']}"),
            ("use_of_context", f"uses P(k) in the step: {f['key']}"),
            ("completeness", "states that the step proves 'if P(k) then P(k + 1)' and does not prove P(k) itself"),
            ("communication", "uses the notation P(k) and P(k + 1) consistently"),
        ],
        wrong=[
            ("proves_p_k_itself", "does it try to prove P(k) rather than assume it?"),
            ("p_k_plus_1_wrong", "is P(k + 1) written with n replaced by k + 1 correctly, including the extra term?"),
        ],
    )
    return finish(text, f"show P(k) implies P(k + 1): {f['key']}", mark)


def _ind_hard(rng) -> dict:
    mode = rng.choice(["no_base", "assumes_conclusion", "examples_only"])
    if mode == "no_base":
        a = rng.choice([2, 4, 6, 8])
        n1 = 1 + 1 + a
        text = (
            f"A student claims that n^2 + n + {a} is odd for every positive integer n, and writes this proof: "
            f'"Assume k^2 + k + {a} is odd. Then (k + 1)^2 + (k + 1) + {a} = (k^2 + k + {a}) + 2(k + 1). '
            'This is an odd number plus an even number, so it is odd. So the statement is true for all n by induction." '
            "What important part of an induction proof is missing, and does it matter here? Check the claim for n = 1 in your answer."
        )
        ref = [
            "The initial statement (the case n = 1) has not been checked, so the proof is incomplete.",
            f"It matters here: when n = 1, 1^2 + 1 + {a} = {n1}, which is even, so the claim is false.",
            "The inductive step is valid in itself (it shows that if the statement holds for k it holds for k + 1), but it has nothing to start from, so it proves nothing.",
        ]
        pts = [
            ("correct_idea", "identifies the missing initial statement (the n = 1 case)"),
            ("correct_idea", f"checks n = 1: 1 + 1 + {a} = {n1}, which is even, so the claim fails"),
            ("use_of_context", "refers to the student's step as logically valid but not enough on its own"),
            ("completeness", "explains that induction needs both the initial statement and the inductive step"),
            ("communication", "answers both questions, what is missing and whether it matters"),
        ]
        wrong = [
            ("blames_the_step", "does it say the inductive step algebra is wrong, when the step is valid?"),
            ("says_it_does_not_matter", "does it say the missing case does not matter because the step works?"),
        ]
        ans = f"the n = 1 case is missing, and it fails ({n1} is even)"
    elif mode == "assumes_conclusion":
        f = _family(rng)
        text = (
            f"To prove by induction that {f['stmt']} for every positive integer n, a student writes: "
            f'"Initial statement: {f["p1"]}. Inductive step: assume that the statement is true for n = k + 1, that is, {f["pk1"]}. '
            'Then check that this works for n = k." Explain what is wrong with this inductive step and what should be done instead.'
        )
        ref = [
            "The student assumes P(k + 1), which is what needs to be proved, and works backwards to P(k). That is circular.",
            f"The inductive step should assume P(k): {f['pk']}, and then show that P(k + 1) follows from it.",
            f"For this statement the step is: {f['key']}.",
        ]
        pts = [
            ("correct_idea", "identifies that the student assumes P(k + 1), the statement to be proved"),
            ("correct_idea", f"says the step should assume P(k): {f['pk']}"),
            ("use_of_context", f"describes the correct step for this statement: {f['key']}"),
            ("completeness", "explains the direction: from P(k) to P(k + 1)"),
            ("communication", "explains why the argument is circular"),
        ]
        wrong = [
            ("initial_statement_blamed", "does it say the initial statement is wrong, when it is fine?"),
            ("accepts_the_step", "does it accept the reversed argument as a valid induction step?"),
        ]
        ans = "the student assumes P(k + 1); the step must assume P(k) and prove P(k + 1)"
    else:
        f = _family(rng)
        text = (
            f'A student wants to prove that {f["stmt"]} for every positive integer n. The student checks n = 1, 2, 3, 4 and 5, finds it true each time, and writes "so it is proved". '
            "Explain why this is not a proof, and say what a proof by induction needs in addition."
        )
        ref = [
            "Checking some cases shows the statement holds for those cases only. It cannot show it holds for every positive integer, because there are infinitely many cases and a later case could fail.",
            f"An induction proof needs the initial statement ({f['p1']}) and an inductive step that shows P(k) implies P(k + 1) for every k.",
            "Together these give a chain: P(1) is true, P(1) gives P(2), P(2) gives P(3), and so on for every n.",
        ]
        pts = [
            ("correct_idea", "explains that checking some cases does not prove the statement for all n"),
            (
                "correct_idea",
                "says induction needs the initial statement and the inductive step (if P(k) then P(k + 1))",
            ),
            ("use_of_context", f"refers to the statement given: {f['stmt']}"),
            ("completeness", "explains how the two parts together give every n (a chain of statements)"),
            ("communication", "answers both parts of the question"),
        ]
        wrong = [
            ("five_cases_is_enough", "does it say five checked cases are enough to prove the statement?"),
            ("only_step_needed", "does it say only the inductive step is needed, without the initial statement?"),
        ]
        ans = "examples do not prove it; induction needs the initial statement and the inductive step"
    mark = marking(reference=ref, points=pts, wrong=wrong)
    return finish(text, ans, mark)


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return True


def _ind_exam(rng) -> dict:
    c = rng.choice([11, 17, 41])
    first_fail = c - 1
    check(all(_is_prime(n * n + n + c) for n in range(1, first_fail)), "prime for n below the failure")
    check(not _is_prime(first_fail * first_fail + first_fail + c), "fails at n = c - 1")
    p, q = rng.sample(NAMES, 2)
    last_ok = first_fail - 1
    bad = first_fail * first_fail + first_fail + c
    text = (
        f"Consider the statement P(n): n^2 + n + {c} is a prime number. It is true for each of n = 1, 2, ..., {last_ok}. "
        f'{p} says: "It is true for {last_ok} values, so it is true for every positive integer n." '
        f'{q} says: "A proof needs the initial statement and an inductive step. Checking cases is not enough, and this statement may fail." '
        f"Evaluate both claims. Test n = {first_fail} in your answer, and say what this tells you about trying to prove the inductive step."
    )
    mark = marking(
        reference=[
            f"{q} is right. For n = {first_fail}: {first_fail}^2 + {first_fail} + {c} = {bad} = {c}^2, which is not prime, so the statement is false.",
            f"{p} is wrong: {last_ok} true cases do not prove it for all n, and one counterexample is enough to disprove it.",
            f"Because P({first_fail}) fails, it is impossible to prove the inductive step for all k: if P({last_ok}) is true then P({first_fail}) would have to follow, and it does not. A proof by induction can only succeed for a true statement.",
        ],
        points=[
            ("correct_idea", f"tests n = {first_fail} and gets {bad} = {c}^2, which is not prime"),
            (
                "correct_idea",
                f"rejects {p}'s claim: finitely many checked cases do not prove a statement for all n, and one counterexample disproves it",
            ),
            ("use_of_context", f"judges {q} right: induction needs an initial statement and a step, not just examples"),
            (
                "completeness",
                f"explains that the inductive step cannot be proved here, because P({last_ok}) is true but P({first_fail}) is false",
            ),
            ("communication", "evaluates each person's claim in turn"),
        ],
        wrong=[
            ("accepts_pattern", f"does it accept {p}'s claim because the statement held for {last_ok} values?"),
            ("wrong_value", f"does it compute n = {first_fail} wrongly, or fail to say that {bad} is not prime?"),
        ],
    )
    return finish(text, f"{q} is right: n = {first_fail} gives {bad} = {c}^2, not prime", mark)


register_custom(
    "wa_induction_nature",
    13,
    dict(easy=_ind_easy, medium=_ind_medium, hard=_ind_hard, exam=_ind_exam),
    doc="the initial statement and the inductive step.",
)
