"""Generates `ds_census_sample_statements` questions (easy, medium, hard).

``f10-census-sample-experiment-practicalities`` (``statistics-probability``, answer ``set``): a study is described (a
census of a school, a sample of batteries tested until they go flat, an experiment on volunteers, birds counted at a
feeder) and the learner types the letters of the TRUE statements from a pool of lettered statements about the four ways
of collecting data: what each collects, what it changes, what it can and cannot show, how practical it is, and what its
result implies. ``choose-sampling-method`` names a method and works out an interval, and ``spot-bias`` finds the flaw in
a sample; neither asks for the practicalities or for experiment against observation. Written explanation stays a rubric
question and is not asked here.

Contract (every statement is true or false by construction, never by opinion):

* A census collects data from every member of the population; a random sample from part of it; an experiment CHANGES
  something for some of the units (a treatment, given to units put into groups at random); an observational study changes
  nothing. Only a well-designed experiment can show that one thing CAUSES a change in another. Every measurement is
  assumed to be recorded correctly, so a census gives the exact value for the population and a sample result can differ.
* "Collects from every unit" and "gives the exact value" are claimed only of a census or a sample, so an observation that
  happens to see everything cannot make a statement arguable.
* Whether checking a unit uses it up (a battery run until it goes flat, a tin opened and tasted) is a fact of the
  situation; so is whether the units were picked at random or chose themselves by volunteering; so are the numbers (how
  many times as many checks a census needs, what it costs, how many fall in each group, how many hours were watched).

Statements are drawn from slots, so the SAME frame is true in one situation and false in another and its wording never
gives the truth away: a property-and-method pair is worded one of two ways ("can" or "cannot", "every" or "some left
out"); a fact of the situation is worded as the fact or as its opposite; a number is the right figure or a plain slip.
Truth is chosen first (a random number of the statements are true), then the wording that has it.

* easy: six statements, each a property of one method (what it can show, what it changes, whom it covers).
* medium: seven statements; at least two are facts of THIS situation (does a check use the unit up, how were the units
  chosen and what that implies), one is a figure to work out, the rest are properties of a method.
* hard: eight statements; at least three are figures to work out (how many times as many checks, the total time or cost,
  the size of each group, the hours watched, the total counted) and at least one a fact of the situation.

Each figure is worked out two ways (by the formula and by an explicit count) before it is used.
"""

from __future__ import annotations

import random

from ..answers import Distractor
from ..core import attempts, check, generator
from ..grading.typed import set_answer_text

TOPIC = "ds_census_sample_statements"
AREA = "data-collection-census-sample"

MISCONCEPTIONS = [
    (
        "ds-cs-observation-shows-cause",
        "Takes a census, a sample survey or an observation as showing that one thing causes another.",
        AREA,
    ),
    (
        "ds-cs-experiment-shows-no-cause",
        "Thinks a well-designed experiment, like an observation, can show only a link and never a cause.",
        AREA,
    ),
    (
        "ds-cs-sample-covers-everyone",
        "Takes a sample, or any study of part of the group, as collecting data from every member.",
        AREA,
    ),
    ("ds-cs-census-leaves-out", "Thinks a census leaves some members of the population out.", AREA),
    (
        "ds-cs-observer-changes-things",
        "Counts an observation, a survey or a census as changing something for the units studied.",
        AREA,
    ),
    (
        "ds-cs-experiment-changes-nothing",
        "Thinks an experiment leaves the units as they were, like an observation.",
        AREA,
    ),
    ("ds-cs-sample-gives-exact", "Treats the result of a sample as the exact value for the whole population.", AREA),
    (
        "ds-cs-census-may-differ",
        "Thinks a census with every record correct can still differ from the value for the population.",
        AREA,
    ),
    (
        "ds-cs-ignores-destructive-test",
        "Overlooks that checking a unit can use it up, so that checking every unit is not practical.",
        AREA,
    ),
    ("ds-cs-assumes-testing-destroys", "Thinks a check that leaves the unit as it was uses the unit up.", AREA),
    (
        "ds-cs-volunteers-as-random",
        "Treats volunteers as if they had been picked at random, so bias from self-selection is missed.",
        AREA,
    ),
    ("ds-cs-random-as-volunteers", "Treats a random selection as if the units had chosen themselves.", AREA),
    (
        "ds-cs-ratio-as-difference",
        "Gives how many MORE units a census uses (the difference) for how many TIMES as many.",
        AREA,
    ),
    (
        "ds-cs-cost-of-sample",
        "Gives the time or cost for the sample, or for the part left out, for the whole census.",
        AREA,
    ),
    (
        "ds-cs-adds-instead-of-multiplies",
        "Adds the number of units to the time, cost or days each takes instead of multiplying.",
        AREA,
    ),
    (
        "ds-cs-ignores-the-split",
        "Gives the whole group as the size of each of two equal groups, or halves it twice.",
        AREA,
    ),
    (
        "ds-cs-total-from-one-day",
        "Gives one day's count, or the mean or the greatest count, for the total over all the days.",
        AREA,
    ),
    (
        "ds-cs-rejects-correct-figure",
        "Rejects a correct figure, usually after working it out a different, wrong way.",
        AREA,
    ),
]

# -- the four methods and their properties ------------------------------------------------------------------------------

METHODS = {
    "census": "A census",
    "sample": "A random sample",
    "experiment": "A well-designed experiment",
    "observation": "An observational study",
}
ALL = ("census", "sample", "experiment", "observation")
COVER = ("census", "sample")  # claimed of these two only


def _belief(prop: str, holds: bool) -> str:
    """The wrong belief behind an error on this fact: the property holds of the method (or does not)."""
    return {
        ("cause", True): "ds-cs-experiment-shows-no-cause",
        ("cause", False): "ds-cs-observation-shows-cause",
        ("every", True): "ds-cs-census-leaves-out",
        ("every", False): "ds-cs-sample-covers-everyone",
        ("changes", True): "ds-cs-experiment-changes-nothing",
        ("changes", False): "ds-cs-observer-changes-things",
        ("exact", True): "ds-cs-census-may-differ",
        ("exact", False): "ds-cs-sample-gives-exact",
    }[(prop, holds)]


# prop: (methods it is claimed of, wording A, methods where A is true, wording B (the opposite), why A holds / fails)
PROPS = {
    "cause": (
        ALL,
        "can show that {x} causes a change in {y}",
        {"experiment"},
        "cannot show that {x} causes a change in {y}",
    ),
    "changes": (
        ALL,
        "changes something for some of the {units} studied",
        {"experiment"},
        "changes nothing for the {units} studied",
    ),
    "every": (
        COVER,
        "collects data from every {unit} {inpop}",
        {"census"},
        "leaves out some of the {units} {inpop}",
    ),
    "exact": (
        COVER,
        "gives the exact value for {whole}",
        {"census"},
        "can give a value that differs from the value for {whole}",
    ),
}
WHY = {
    "cause": "Only a well-designed experiment, which gives some units the treatment and compares, can show a cause.",
    "changes": "An experiment applies a treatment to some units; a census, a sample and an observation change nothing.",
    "every": "A census collects data from every unit; a sample uses only part of the population.",
    "exact": "With every record correct, a census gives the exact value; a sample result can differ from it.",
}

# -- situations ---------------------------------------------------------------------------------------------------------
# cs: census or sample situations. unit/units, pop (after "in"/"of"), check (what one check does), destroys, work (what is
# counted), verb/fmt (time or cost of all the checks), cost sentence, ms (cost of one check), ns, ks (N = n * k)

CS = [
    dict(
        kinds=("census", "sample"),
        unit="student",
        units="students",
        pop="the school",
        inpop="at the school",
        whole="the whole school",
        lead="A school has {N} students.",
        what="how its students usually travel to school",
        x="whether a student is given a free bus pass",
        y="how the student travels",
        check="Asking a student the question",
        destroys=False,
        work="interviews",
        each="Each interview takes {m} minutes.",
        verb="take",
        fmt="{V} minutes",
        ms=(2, 3, 4, 5),
        ns=(30, 40, 50, 60),
        ks=(8, 10, 12, 15, 20),
        total="students",
    ),
    dict(
        kinds=("census", "sample"),
        unit="household",
        lead="A council area has {N} households.",
        units="households",
        pop="the council area",
        inpop="in the council area",
        whole="the whole council area",
        what="how its households dispose of garden waste",
        x="whether a household is given a free green-waste bin",
        y="how the household disposes of garden waste",
        check="Posting a survey form to a household",
        destroys=False,
        work="forms",
        each="Each form costs ${m} to print, post and process.",
        verb="cost",
        fmt="${V}",
        ms=(2, 3, 4, 6),
        ns=(40, 50, 60, 80),
        ks=(10, 12, 15, 20),
        total="households",
    ),
    dict(
        kinds=("census", "sample"),
        unit="tree",
        lead="A plantation has {N} trees.",
        units="trees",
        pop="the plantation",
        inpop="in the plantation",
        whole="the whole plantation",
        what="the trunk width of its trees",
        x="the soil under a tree",
        y="the width of the tree's trunk",
        check="Measuring the trunk of a tree",
        destroys=False,
        work="measurements",
        each="Each measurement takes {m} minutes.",
        verb="take",
        fmt="{V} minutes",
        ms=(3, 4, 5, 6),
        ns=(25, 40, 50, 60),
        ks=(8, 10, 12, 20),
        total="trees",
    ),
    dict(
        kinds=("census", "sample"),
        unit="battery",
        lead="A factory makes {N} batteries in a day.",
        units="batteries",
        pop="the day's batch",
        inpop="in the day's batch",
        whole="the whole day's batch",
        what="how long its batteries last",
        x="the brand of a battery",
        y="how long the battery lasts",
        check="Running a battery until it goes flat",
        destroys=True,
        work="tests",
        each="Each test takes {m} hours.",
        verb="take",
        fmt="{V} hours",
        ms=(3, 4, 5, 6),
        ns=(20, 30, 40, 50),
        ks=(10, 12, 15, 20),
        total="batteries",
    ),
    dict(
        kinds=("census", "sample"),
        unit="tin",
        lead="A cannery fills {N} tins in a shift.",
        units="tins",
        pop="the shift's output",
        inpop="in the shift's output",
        whole="the whole shift's output",
        what="whether its tinned soup is too salty",
        x="the recipe used for a tin",
        y="how salty the soup tastes",
        check="Opening a tin to taste the soup",
        destroys=True,
        work="tests",
        each="Each test costs ${m} in lost soup and labour.",
        verb="cost",
        fmt="${V}",
        ms=(2, 3, 4, 5),
        ns=(20, 30, 40, 50),
        ks=(10, 15, 20, 25),
        total="tins",
    ),
    dict(
        kinds=("census", "sample"),
        unit="light globe",
        lead="A factory makes {N} light globes in a production run.",
        units="light globes",
        pop="the production run",
        inpop="in the production run",
        whole="the whole production run",
        what="how long its light globes last",
        x="the filament used in a globe",
        y="how long the globe lasts",
        check="Leaving a globe on until it burns out",
        destroys=True,
        work="tests",
        each="Each test takes {m} days.",
        verb="take",
        fmt="{V} days",
        ms=(10, 20, 30, 40),
        ns=(10, 20, 25, 30),
        ks=(10, 12, 20, 30),
        total="light globes",
    ),
]

CS_SCENE = {
    "census": "{lead} To find out {what}, a team checks every one of them. {each} A random sample of {n} {units} could have been used instead.",
    "sample_random": "{lead} To find out {what}, a team picks {n} of them at random and checks only those. {each} Checking all {N} would be a census.",
    "sample_volunteer": "{lead} To find out {what}, a team asks for {n} volunteers and checks only those. {each} Checking all {N} would be a census.",
}
# ex: experiments. n (volunteers or units), groups at random, r records per unit
EX = [
    dict(
        unit="person",
        units="people",
        pop="the people on the club's list",
        inpop="on the club's list",
        whole="everyone on the club's list",
        x="which recipe a person is given",
        y="how highly the person rates the taste",
        what="tasting a new recipe",
        group_a="the new recipe",
        group_b="the current recipe",
        ns=(24, 30, 36, 40),
        rs=(2, 3, 4),
        record="ratings",
        each="Each person rates the taste {r} times, once for each serve.",
        hows=("volunteer", "random"),
        who_v="A bakery asks for {n} volunteers from the people on the club's list",
        who_r="A bakery picks {n} people at random from the people on the club's list",
    ),
    dict(
        unit="student",
        units="students",
        pop="the students in the year level",
        inpop="in the year level",
        whole="the whole year level",
        x="which study app a student uses",
        y="the student's score on a quiz",
        what="a new study app",
        group_a="the new app",
        group_b="the usual notes",
        ns=(24, 30, 36, 40),
        rs=(2, 3, 4),
        record="quiz scores",
        each="Each student sits {r} quizzes over the term.",
        hows=("volunteer", "random"),
        who_v="A teacher asks for {n} volunteers from the students in the year level",
        who_r="A teacher picks {n} students at random from the students in the year level",
    ),
    dict(
        unit="plot",
        units="plots",
        pop="the plots on the farm",
        inpop="on the farm",
        whole="all the plots on the farm",
        x="which fertiliser a plot gets",
        y="the crop yield from the plot",
        what="a new fertiliser",
        group_a="the new fertiliser",
        group_b="the usual fertiliser",
        ns=(20, 24, 30, 40),
        rs=(2, 3, 4),
        record="yield readings",
        each="Each plot is weighed at harvest in {r} separate seasons.",
        hows=(None,),
        who_v="",
        who_r="A farmer marks out {n} plots",
    ),
]

# ob: observations. d days, h hours a day, counts each day
OB = [
    dict(
        unit="bird",
        units="birds",
        pop="the birds that visit gardens in the suburb",
        inpop="that visit gardens in the suburb",
        whole="all the birds that visit gardens in the suburb",
        x="the type of seed in a feeder",
        y="the number of birds that visit",
        what="birds at a feeder",
        scene="A student sits beside one garden feeder for {h} hours each morning for {d} mornings and counts the birds that visit. No seed is changed. The counts were {counts}.",
        thing="birds",
        ds=(5, 6, 7),
        hs=(2, 3, 4),
        lo=8,
        hi=30,
    ),
    dict(
        unit="student",
        units="students",
        pop="the students at the school",
        inpop="at the school",
        whole="the whole school",
        x="the equipment in a playground",
        y="how many students use it",
        what="playground use",
        scene="A teacher watches the playground for {h} hours each lunchtime for {d} school days and counts the students on the climbing frame. Nothing about the playground is changed. The counts were {counts}.",
        thing="students",
        ds=(5, 6, 7),
        hs=(2, 3, 4),
        lo=10,
        hi=40,
    ),
    dict(
        unit="car",
        units="cars",
        pop="the cars on the town's roads",
        inpop="on the town's roads",
        whole="all the cars on the town's roads",
        x="the speed limit on a road",
        y="how many cars use it",
        what="traffic at a crossing",
        scene="A volunteer stands at a crossing for {h} hours each afternoon for {d} afternoons and counts the cars that pass. The road is left as it is. The counts were {counts}.",
        thing="cars",
        ds=(5, 6, 7),
        hs=(2, 3, 4),
        lo=30,
        hi=90,
    ),
]

EX_SCENE = ("{who}. They are put at random into two equal groups. One group is given {a} and the other {b}. {each}",)


# -- statements ---------------------------------------------------------------------------------------------------------


class Stmt:
    """One lettered statement: its text, whether it is true, the dimension it comes from, the error behind getting it wrong."""

    __slots__ = ("text", "truth", "dim", "tag", "why", "group")

    def __init__(self, text, truth, dim, tag, why, group):
        self.text, self.truth, self.dim, self.tag, self.why, self.group = text, truth, dim, tag, why, group


def _prop_statement(sit, prop, method, form_a, want):
    """A statement about `method` and `prop`: wording A (`form_a`) or B, the truth it has being `want`."""
    _, wa, a_true, wb = PROPS[prop]
    holds = method in a_true  # wording A is true of this method
    check((holds if form_a else not holds) == want, "the wording does not have the truth wanted")
    wording = wa if form_a else wb
    text = f"{METHODS[method]} {wording.format(**sit['slots'])}."
    return Stmt(text, want, "prop:" + prop + ":" + method, _belief(prop, holds), WHY[prop], "prop")


def _pick_prop(rng, used, count, want):
    """(prop, method, form_a) with the truth `want`: the property and the wording are drawn first, so the wording does
    not predict the truth, then a method that has the truth with that wording."""
    for _ in attempts(200, "a property and wording not yet used"):
        prop = rng.choice(sorted(PROPS))
        if count[prop] >= 2:
            continue
        methods, _, a_true, _ = PROPS[prop]
        form_a = rng.random() < 0.5
        # wording A is true of a method in a_true; wording B of one outside it
        eligible = [m for m in methods if (m in a_true) == (form_a == want) and (prop, m) not in used]
        if eligible:
            return prop, rng.choice(eligible), form_a
    raise AssertionError  # pragma: no cover


def _slips(true, options):
    """Distinct wrong figures (value, tag), none equal to the true one and all positive."""
    seen, out = {true}, []
    for v, tag in options:
        if v > 0 and v not in seen:
            seen.add(v)
            out.append((v, tag))
    return out


def _figure_statement(template, true, slips, want, rng, dim, why):
    if want:
        value, tag = true, "ds-cs-rejects-correct-figure"
    else:
        value, tag = rng.choice(_slips(true, slips))
    return Stmt(template.format(V=value), want, dim, tag, why, "figure")


def _build_cs(rng, kind):
    sit = rng.choice([s for s in CS if kind in s["kinds"]])
    n, k, m = rng.choice(sit["ns"]), rng.choice(sit["ks"]), rng.choice(sit["ms"])
    N = n * k
    how = (rng.choice(("random", "volunteer")) if sit["unit"] == "student" else "random") if kind == "sample" else None
    key = "census" if kind == "census" else f"sample_{how}"
    scene = CS_SCENE[key].format(
        lead=sit["lead"].format(N=N), N=N, n=n, units=sit["units"], what=sit["what"], each=sit["each"].format(m=m)
    )
    sit = dict(sit)
    sit["slots"] = dict(
        x=sit["x"], y=sit["y"], unit=sit["unit"], units=sit["units"], inpop=sit["inpop"], whole=sit["whole"]
    )
    # two ways: the ratio from the formula and from counting groups of n in N
    ratio = N // n
    check(ratio == k and ratio * n == N, "the ratio differs")
    check(len(range(0, N, n)) == ratio, "the ratio by counting differs")
    total = N * m
    check(total == sum(m for _ in range(N)), "the total differs")

    facts = []
    # does a check use the unit up (a fact of the situation)
    destroys = sit["destroys"]
    facts.append(("spoil", destroys))
    if kind == "sample" and sit["unit"] == "student":
        facts += [("how", how == "random"), ("impl", how == "random")]
    figs = []
    ratio_tpl = f"Checking all {N} {sit['units']} would need {{V}} times as many {sit['work']} as checking the {n} in the sample."
    figs.append(
        (
            "ratio",
            ratio_tpl,
            ratio,
            [
                (N - n, "ds-cs-ratio-as-difference"),
                (N, "ds-cs-ratio-as-difference"),
                (n, "ds-cs-ratio-as-difference"),
                (ratio + 1, "ds-cs-ratio-as-difference"),
            ],
            f"{N} / {n} = {ratio}.",
        )
    )
    cost_tpl = f"Checking all {N} {sit['units']} would {sit['verb']} {sit['fmt']} in all."
    figs.append(
        (
            "total",
            cost_tpl,
            total,
            [
                (n * m, "ds-cs-cost-of-sample"),
                ((N - n) * m, "ds-cs-cost-of-sample"),
                (N + m, "ds-cs-adds-instead-of-multiplies"),
                (n + m, "ds-cs-adds-instead-of-multiplies"),
            ],
            f"{N} x {m} = {total}.",
        )
    )
    return sit, scene, facts, figs, dict(destroys=destroys, how=how, N=N, n=n, kind=kind)


def _build_ex(rng, kind):
    sit = dict(rng.choice(EX))
    n, r = rng.choice(sit["ns"]), rng.choice(sit["rs"])
    how = rng.choice(sit["hows"])
    who = sit["who_v"] if how == "volunteer" else sit["who_r"]
    scene = EX_SCENE[0].format(who=who.format(n=n), a=sit["group_a"], b=sit["group_b"], each=sit["each"].format(r=r))
    sit["slots"] = dict(
        x=sit["x"], y=sit["y"], unit=sit["unit"], units=sit["units"], inpop=sit["inpop"], whole=sit["whole"]
    )
    half = n // 2
    check(2 * half == n and half + half == n, "the groups differ")
    records = n * r
    check(records == sum(r for _ in range(n)), "the records differ")
    facts = []
    if how is not None:
        facts += [("how", how == "random"), ("impl", how == "random")]
    figs = [
        (
            "group",
            f"Each of the two groups has {{V}} {sit['units']}.",
            half,
            [(n, "ds-cs-ignores-the-split"), (n // 4, "ds-cs-ignores-the-split"), (n - 2, "ds-cs-ignores-the-split")],
            f"{n} / 2 = {half}.",
        ),
        (
            "records",
            f"The experiment collects {{V}} {sit['record']} in all.",
            records,
            [
                (n + r, "ds-cs-adds-instead-of-multiplies"),
                (n, "ds-cs-total-from-one-day"),
                (r * 2, "ds-cs-total-from-one-day"),
                (half * r, "ds-cs-ignores-the-split"),
            ],
            f"{n} x {r} = {records}.",
        ),
    ]
    return sit, scene, facts, figs, dict(destroys=False, how=how, N=None, n=n, kind=kind)


def _build_ob(rng, kind):
    sit = dict(rng.choice(OB))
    d, h = rng.choice(sit["ds"]), rng.choice(sit["hs"])
    for _ in attempts(500, "daily counts that add to a multiple of the number of days"):
        counts = [rng.randint(sit["lo"], sit["hi"]) for _ in range(d)]
        if sum(counts) % d == 0 and len(set(counts)) >= d - 1:
            break
    total, mean = sum(counts), sum(counts) // d
    acc = 0
    for c in counts:
        acc += c
    check(acc == total and total == mean * d, "the counts differ")
    hours = d * h
    check(hours == sum(h for _ in range(d)), "the hours differ")
    scene = sit["scene"].format(h=h, d=d, counts=", ".join(str(c) for c in counts))
    sit["slots"] = dict(
        x=sit["x"], y=sit["y"], unit=sit["unit"], units=sit["units"], inpop=sit["inpop"], whole=sit["whole"]
    )
    figs = [
        (
            "hours",
            "The student watched for {V} hours in all.".replace("The student", "The observer"),
            hours,
            [
                (d + h, "ds-cs-adds-instead-of-multiplies"),
                (h, "ds-cs-total-from-one-day"),
                (d, "ds-cs-total-from-one-day"),
                (hours + h, "ds-cs-adds-instead-of-multiplies"),
            ],
            f"{d} x {h} = {hours}.",
        ),
        (
            "counts",
            f"The {sit['thing']} counted over the {d} days add up to {{V}}.",
            total,
            [
                (mean, "ds-cs-total-from-one-day"),
                (max(counts), "ds-cs-total-from-one-day"),
                (total - counts[0], "ds-cs-total-from-one-day"),
                (total + h, "ds-cs-adds-instead-of-multiplies"),
            ],
            f"The counts add to {total}.",
        ),
    ]
    return sit, scene, [], figs, dict(destroys=False, how=None, N=None, n=None, kind=kind)


# -- the situation facts as statements ----------------------------------------------------------------------------------


def _fact_statement(sit, info, name, want):
    """A statement about a fact of the situation with the truth `want`."""
    u, us = sit["unit"], sit["units"]
    if name == "spoil":
        yes = info["destroys"]
        # A: the check uses the unit up (true iff destroys); B: the opposite
        use_a = yes == want
        a = f"{sit['check']} uses up the {u}."
        b = f"{sit['check']} leaves the {u} as it was."
        return Stmt(
            a if use_a else b,
            want,
            "fact:spoil",
            "ds-cs-ignores-destructive-test" if yes else "ds-cs-assumes-testing-destroys",
            "Whether a check uses the unit up is a fact of the situation; when it does, checking every unit would leave none.",
            "fact",
        )
    if name == "how":
        is_random = info["how"] == "random"
        use_a = is_random == want
        a = f"The {us} in this study were picked at random from {sit['whole']}."
        b = f"The {us} in this study chose themselves by volunteering."
        return Stmt(
            a if use_a else b,
            want,
            "fact:how",
            "ds-cs-volunteers-as-random" if not is_random else "ds-cs-random-as-volunteers",
            "The description says whether the units were picked at random or volunteered.",
            "fact",
        )
    if name == "impl":
        is_random = info["how"] == "random"
        # A: biased because they chose themselves (true iff volunteers); B: not biased by choosing themselves (true iff random)
        use_a = (not is_random) == want
        a = f"These results may be biased because the {us} chose themselves."
        b = f"These results cannot be biased by the {us} choosing themselves."
        return Stmt(
            a if use_a else b,
            want,
            "fact:impl",
            "ds-cs-volunteers-as-random" if not is_random else "ds-cs-random-as-volunteers",
            "Volunteers choose themselves, so their answers can differ from the rest; a random pick gives every unit the same chance.",
            "fact",
        )
    raise ValueError(name)


INTROS = {
    "easy": (
        "{scene} Assume every measurement is recorded correctly. Which of these statements are true? Type the letters of all the true statements, separated by commas.",
        "{scene} Every measurement is recorded correctly. Choose the true statements and type their letters, separated by commas.",
        "{scene} Assume no measurement is recorded wrongly. Type the letters of the statements that are true, separated by commas.",
    ),
    "medium": (
        "{scene} Assume every measurement is recorded correctly. Which statements about this study and the other ways of collecting the data are true? Type the letters of all the true statements, separated by commas.",
        "{scene} All the measurements are recorded correctly. Decide which statements are true of this study and of the other methods, and type their letters, separated by commas.",
        "{scene} Suppose no measurement is recorded wrongly. Pick out the true statements and type their letters, separated by commas.",
    ),
    "hard": (
        "{scene} Assume every measurement is recorded correctly. Work out any figures you need, then type the letters of all the true statements, separated by commas.",
        "{scene} All the measurements are recorded correctly. Check each statement against the facts and figures above, then type the letters of the true ones, separated by commas.",
        "{scene} Suppose no measurement is recorded wrongly. Work out what you need and type the letters of every true statement, separated by commas.",
    ),
}
SIZE = {"easy": 6, "medium": 7, "hard": 8}
LETTERS = "ABCDEFGH"


@generator(
    TOPIC,
    level=8,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def ds_census_sample_statements(rng: random.Random, tier: str = "easy") -> dict:
    """Tick the true statements about census, sample, experiment and observation for a stated study."""
    if tier not in SIZE:
        raise ValueError(f"unknown tier {tier!r}")
    size = SIZE[tier]
    kind = rng.choice(("census", "sample", "experiment", "observation"))
    build = {"census": _build_cs, "sample": _build_cs, "experiment": _build_ex, "observation": _build_ob}[kind]
    sit, scene, facts, figs, info = build(rng, kind)

    # the plan: how many statements of each family
    if tier == "easy":
        n_fig, n_fact = 0, 0
    elif tier == "medium":
        n_fig = 1 if facts else 2
        n_fact = min(len(facts), 2)
        if n_fact + n_fig < 3:
            n_fig = 3 - n_fact
        n_fig = min(n_fig, len(figs))
    else:
        n_fact = min(len(facts), 2 if len(facts) >= 2 else 1)
        n_fig = len(figs)
    # a medium or hard question for an observation (no facts, two figures) fills with properties
    n_prop = size - n_fig - n_fact

    # truth: a random number of true statements, 2 to size - 2
    k = rng.randint(2, size - 2)
    truths = [True] * k + [False] * (size - k)
    rng.shuffle(truths)

    fam = ["prop"] * n_prop + ["fig"] * n_fig + ["fact"] * n_fact
    rng.shuffle(fam)

    fact_names = list(facts)
    rng.shuffle(fact_names)
    fig_pool = list(figs)
    rng.shuffle(fig_pool)
    fact_pool = [name for name, _ in fact_names]
    used_props: set = set()
    stmts: list[Stmt] = []
    prop_count = {p: 0 for p in PROPS}
    fact_i = fig_i = 0
    for family, want in zip(fam, truths):
        if family == "prop":
            prop, method, form_a = _pick_prop(rng, used_props, prop_count, want)
            used_props.add((prop, method))
            prop_count[prop] += 1
            stmts.append(_prop_statement(sit, prop, method, form_a, want))
        elif family == "fig":
            name, tpl, true, slips, why = fig_pool[fig_i]
            fig_i += 1
            stmts.append(_figure_statement(tpl, true, slips, want, rng, "fig:" + name, why))
        else:
            name = fact_pool[fact_i]
            fact_i += 1
            stmts.append(_fact_statement(sit, info, name, want))
    check(len(stmts) == size, "the pool is the wrong size")
    check(len({s.text for s in stmts}) == size, "two statements are the same")

    letters = LETTERS[:size]
    key_letters = [letters[i] for i, s in enumerate(stmts) if s.truth]
    check(len(key_letters) == k, "the number of true statements differs")

    text_lines = [f"({letters[i]}) {s.text}" for i, s in enumerate(stmts)]
    intro = rng.choice(INTROS[tier]).format(scene=scene)
    text = intro + " " + " ".join(text_lines)

    key_text = set_answer_text(key_letters)
    # wrong sets: the key with one statement ticked wrongly or one true statement missed; each stands for the error behind it
    order = list(range(size))
    rng.shuffle(order)
    distractors = []
    seen = {key_text}
    for i in order:
        s = stmts[i]
        letter = letters[i]
        wrong = [l for l in key_letters if l != letter] if s.truth else sorted(key_letters + [letter])
        txt = set_answer_text(wrong)
        if txt in seen:
            continue
        seen.add(txt)
        distractors.append(Distractor(txt, s.tag))
        if len(distractors) == 4:
            break
    check(len(distractors) >= 3, "too few wrong sets")

    working = []
    for i, s in enumerate(stmts):
        working.append(f"{letters[i]} is {'true' if s.truth else 'false'}. {s.why}")
    return dict(
        text=text,
        answer_kind="set",
        answer=key_letters,
        answer_text=key_text,
        working=working,
        distractors=distractors,
    )
