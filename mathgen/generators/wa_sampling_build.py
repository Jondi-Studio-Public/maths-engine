"""Short written-answer generators about samples, censuses and the purpose of sampling.

Every question gives a concrete survey situation and asks for a short written answer, marked against a rubric. The
`answer` of each question is a marking dict (see `_written_answer_kit`), not a value.

Topics (one per skill):

- `wa_sample_kinds`: name the kind of sample used, challenge a wrong label, weigh a strength against a weakness,
  then compare two surveys.
- `wa_sample_pros_cons`: advantages and disadvantages of a sampling method under a stated constraint (no list,
  little time, groups that must all be represented, an important decision), up to a recommendation between methods.
- `wa_census_pros_cons`: when a census is sensible and when it is not, with the cost and time figures stated.
- `wa_census_procedure`: the steps of a census, then spotting the step a plan leaves out or does badly.
- `wa_sampling_purpose`: why a sample estimates a population value, with the estimate worked out from the sample.
- `wa_sampling_bias`: sources of bias in how a sample is chosen and the procedure that makes the choice random.

Ladder: name or identify one thing -> explain its effect or challenge a claim -> justify a choice or fix ->
evaluate competing claims or plans.
"""

from __future__ import annotations

from ..core import check
from ._written_answer_kit import (
    DATA_TAILS,
    NAME_SLOTS,
    TAILS,
    Case,
    Tier,
    ctx_slots,
    finish,
    marking,
    pct,
    register,
    register_custom,
    ri,
)

# ---------------------------------------------------------------------------
# wa_sample_kinds
# ---------------------------------------------------------------------------
_SCHOOLS = ("Hillcrest High School", "Ridgeway College", "Maple Grove Secondary College", "Bayview High School")
_STRAT_TOPICS = ("how students travel to school", "whether students want a later start time", "canteen prices")


def _strat_sizes(r, v):
    f = v["f"]
    return tuple(f * r.randint(6, 18) for _ in range(3))


KIND_CASES = [
    Case(
        "srs",
        slots={**ctx_slots(), **NAME_SLOTS, "N": ri(400, 1500, 50), "n": [30, 40, 50, 60]},
        fields=dict(
            setup="{org} has {N} {members}. Each {member} is given a number from 1 to {N}, and a computer picks {n} different numbers at random. The {members} with those numbers are asked about {topic}.",
            kind="simple random sample",
            why="every {member} had the same chance of being chosen and chance alone decided who was picked",
            mislabel="systematic sample",
            mislabel_why="the chosen {members} were not picked at a fixed interval; each number was picked independently by chance",
            strength="every {member} has an equal chance of selection, so no group is favoured",
            weakness="it needs a complete numbered list of all {N} {members}, and by chance it can still miss a small group",
            represent="all {N} {members} of {org}",
            fix="use a complete numbered list of the {members} and a random number generator",
            rank=4,
        ),
        wrong=[
            ("calls_it_systematic", "does it call this a systematic sample? {mislabel_why}"),
        ],
    ),
    Case(
        "systematic",
        slots={
            **ctx_slots(),
            **NAME_SLOTS,
            "k": [5, 8, 10, 12, 15, 20],
            "n": [30, 40, 50, 60],
            "N": lambda r, v: v["k"] * v["n"],
        },
        fields=dict(
            setup="{org} keeps an alphabetical list of its {N} {members}. {person} picks a random starting place among the first {k} names and then asks every {k}th {member} on the list, {n} {members} in all, about {topic}.",
            kind="systematic sample",
            why="after a random start the {members} were chosen at a fixed interval of {k} down the list",
            mislabel="simple random sample",
            mislabel_why="once the starting place is fixed, the rest of the sample is fixed too, so the {members} were not chosen independently of each other",
            strength="it is quick to carry out down a list and spreads the sample evenly through it",
            weakness="if the list has a pattern that repeats every {k} names, the sample could be biased, and it still needs a full list",
            represent="all {N} {members} of {org}",
            fix="check that the list has no pattern that repeats at the interval of {k}, or use random numbers instead",
            rank=3,
        ),
        wrong=[
            ("calls_it_simple_random", "does it call this a simple random sample? {mislabel_why}"),
        ],
    ),
    Case(
        "self_selected",
        slots={
            **ctx_slots(),
            **NAME_SLOTS,
            "N": ri(300, 2000, 100),
            "r": ri(15, 90, 5),
            "rate": lambda r, v: pct(v["r"], v["N"]),
        },
        fields=dict(
            setup="{org} puts a link to a survey about {topic} on its website, where all {N} {members} can see it. {r} of the {N} {members} ({rate}) choose to complete it.",
            kind="self-selected sample",
            why="the {members} decided for themselves whether to take part, instead of being chosen",
            mislabel="simple random sample",
            mislabel_why="the {members} were not chosen by chance; only those who saw the link and chose to answer are included",
            strength="it is cheap and easy and reaches many {members} quickly",
            weakness="people with strong opinions are more likely to respond, so the sample is usually biased, and only {rate} replied",
            represent="only the {members} who chose to respond, not all {N} {members}",
            fix="contact a randomly chosen group of {members} directly and follow up those who do not reply",
            rank=0,
        ),
        wrong=[
            ("calls_it_simple_random", "does it call this a simple random sample? {mislabel_why}"),
        ],
    ),
    Case(
        "convenience",
        slots={**ctx_slots(), **NAME_SLOTS, "n": [20, 25, 30, 40]},
        fields=dict(
            setup="{person} waits at the main entrance of {org} on one weekday morning and asks the first {n} {members} who walk in about {topic}.",
            kind="convenience sample",
            why="{person} chose whoever was easiest to reach, so {members} who come at other times or by other entrances had no chance of being asked",
            mislabel="simple random sample",
            mislabel_why="being first through the door is not a chance process, and {members} who come at other times could never be chosen",
            strength="it is very quick and costs almost nothing",
            weakness="the {members} who are easiest to reach may differ from the others, for example in when they arrive, so the sample is likely to be biased",
            represent="only the {members} who arrive at that entrance at that time",
            fix="choose {members} from a full list using random numbers instead of by who is nearby",
            rank=1,
        ),
        wrong=[
            ("calls_it_simple_random", "does it call this a simple random sample? {mislabel_why}"),
        ],
    ),
    Case(
        "stratified",
        slots={
            **NAME_SLOTS,
            "school": _SCHOOLS,
            "topic": _STRAT_TOPICS,
            "org": lambda r, v: v["school"],
            "members": lambda r, v: "students",
            "member": lambda r, v: "student",
            "f": [5, 10, 20],
            "sizes": _strat_sizes,
            "a": lambda r, v: v["sizes"][0],
            "b": lambda r, v: v["sizes"][1],
            "c": lambda r, v: v["sizes"][2],
            "sa": lambda r, v: v["a"] // v["f"],
            "sb": lambda r, v: v["b"] // v["f"],
            "sc": lambda r, v: v["c"] // v["f"],
            "N": lambda r, v: v["a"] + v["b"] + v["c"],
        },
        fields=dict(
            setup="{school} has {a} Year 7, {b} Year 8 and {c} Year 9 students. {person} randomly selects {sa} of the {a} Year 7 students, {sb} of the {b} Year 8 students and {sc} of the {c} Year 9 students, and asks them about {topic}.",
            kind="stratified random sample",
            why="the students were split into year groups and a random sample was taken from each group in proportion to its size",
            mislabel="simple random sample",
            mislabel_why="the sample was built from separate year groups, with each group's share fixed in advance, instead of being drawn from all the students at once",
            strength="every year group is represented in proportion to its size, so the sample mirrors the make-up of the school",
            weakness="it needs a list showing each student's year group and takes more effort to organise than a simple random sample",
            represent="all {N} students at {school}",
            fix="sample each year group separately in proportion to its size",
            rank=4,
        ),
        wrong=[
            ("calls_it_simple_random", "does it call this a simple random sample? {mislabel_why}"),
        ],
    ),
]


def _rank_gap(da: dict, db: dict) -> bool:
    return abs(da["rank"] - db["rank"]) >= 2


def _kinds_derive(v: dict) -> dict:
    better, worse = ("a", "b") if v["a_rank"] > v["b_rank"] else ("b", "a")
    label = {"a": "Survey A", "b": "Survey B"}
    return dict(
        better_label=label[better],
        worse_label=label[worse],
        better_kind=v[f"{better}_kind"],
        worse_kind=v[f"{worse}_kind"],
        better_why=v[f"{better}_why"],
        worse_weakness=v[f"{worse}_weakness"],
        worse_fix=v[f"{worse}_fix"],
    )


KIND_TIERS = {
    "easy": Tier(
        prompt="{setup} What kind of sample is this? Give a reason that refers to how the {members} were chosen.",
        reference=["This is a {kind}.", "Reason: {why}."],
        points=[
            ("correct_idea", "names the sample as a {kind}"),
            ("use_of_context", "links the name to how the {members} of {org} were chosen: {why}"),
            ("completeness", "gives a reason as well as the name"),
            ("communication", "uses the term {kind} accurately, in a full sentence"),
        ],
        wrong=[
            ("name_only", "does the answer give a name only, with no reason from the way the {members} were chosen?")
        ],
        answer="a {kind}",
    ),
    "medium": Tier(
        prompt="{setup} {other} says this is a {mislabel}. Is {other} correct? Explain, and say what kind of sample it is.",
        reference=[
            "{other} is not correct. This is a {kind}.",
            "It is not a {mislabel} because {mislabel_why}.",
            "It is a {kind} because {why}.",
        ],
        points=[
            ("correct_idea", "says {other} is wrong and names the sample as a {kind}"),
            ("use_of_context", "explains why the {mislabel} label does not fit: {mislabel_why}"),
            ("completeness", "also says why the sample is a {kind}: {why}"),
            ("communication", "gives a clear verdict first, then the reasons"),
        ],
        wrong=[
            ("agrees_with_other", "does it agree that this is a {mislabel}?"),
            ("verdict_only", "does it say 'no' without explaining why the {mislabel} label does not fit?"),
        ],
        answer="{other} is wrong: it is a {kind}",
    ),
    "hard": Tier(
        prompt="{setup} Explain one strength and one weakness of choosing the sample this way for this survey, and say which group of people the results can fairly be said to represent.",
        reference=[
            "Strength: {strength}.",
            "Weakness: {weakness}.",
            "The results can fairly be said to represent {represent}.",
        ],
        points=[
            ("correct_idea", "gives a genuine strength of a {kind}: {strength}"),
            ("correct_idea", "gives a genuine weakness of a {kind}: {weakness}"),
            ("use_of_context", "ties both points to the survey of {org} about {topic}, not to samples in general"),
            ("completeness", "names the group the results represent: {represent}"),
            ("communication", "separates the strength, the weakness and the group clearly"),
        ],
        wrong=[
            (
                "generic_points",
                "are the strength and weakness generic ('it is easy', 'it is hard') instead of tied to this survey?",
            ),
            ("generalises_too_far", "does it say the results apply to everyone, not just {represent}?"),
        ],
        answer="strength: {strength}; weakness: {weakness}",
    ),
    "exam": Tier(
        prompt=(
            "Two surveys were carried out.\n"
            "Survey A: {a_setup}\n"
            "Survey B: {b_setup}\n"
            "Name the kind of sample in each survey. Decide which survey is more likely to give results that represent the group it was about, and justify your decision. Suggest one change that would improve the weaker survey."
        ),
        reference=[
            "Survey A is a {a_kind}; Survey B is a {b_kind}.",
            "{better_label} is more likely to be representative: {better_why}.",
            "{worse_label} is weaker because {worse_weakness}.",
            "An improvement for {worse_label}: {worse_fix}.",
        ],
        points=[
            ("correct_idea", "names Survey A as a {a_kind} and Survey B as a {b_kind}"),
            ("correct_idea", "judges {better_label} (a {better_kind}) the more representative"),
            (
                "use_of_context",
                "justifies the judgement from how the people were chosen in each survey: {better_why}, against {worse_weakness}",
            ),
            ("completeness", "suggests a workable improvement for {worse_label}: {worse_fix}"),
            ("communication", "compares the two surveys directly rather than describing them one after the other"),
        ],
        wrong=[
            (
                "prefers_bigger_sample",
                "does it prefer a survey because more people were asked, without looking at how they were chosen?",
            ),
        ],
        answer="{better_label} is more representative; {worse_label} should change: {worse_fix}",
        two=True,
        pair_ok=_rank_gap,
        derive=_kinds_derive,
    ),
}

register(
    "wa_sample_kinds",
    12,
    KIND_CASES,
    KIND_TIERS,
    doc="name a sampling method from a description.",
)

# ---------------------------------------------------------------------------
# wa_sample_pros_cons: a method judged under a constraint
# ---------------------------------------------------------------------------
METHODS = {
    "srs": dict(
        name="simple random sample",
        adv="every {member} has an equal chance of being chosen, so no group is favoured",
        dis="it needs a complete list of all the {members}, and by chance it can still leave out a small group",
    ),
    "systematic": dict(
        name="systematic sample",
        adv="it is quick to carry out down a list and spreads the sample evenly through it",
        dis="it needs a full list, and a pattern in the list that repeats at the interval could bias the sample",
    ),
    "stratified": dict(
        name="stratified sample",
        adv="every group is represented in proportion to its size",
        dis="each {member}'s group must be known beforehand and it takes more effort to organise",
    ),
    "cluster": dict(
        name="cluster sample",
        adv="no list of individual {members} is needed and it is cheap because the chosen {members} are close together",
        dis="{members} in the same cluster tend to be alike, so the sample may be less representative",
    ),
    "convenience": dict(
        name="convenience sample",
        adv="it is quick and costs almost nothing",
        dis="only the {members} who are easy to reach are included, so it is likely to be biased",
    ),
    "self": dict(
        name="self-selected sample",
        adv="it is cheap and easy and can reach many {members}",
        dis="people with strong views are the most likely to respond, so it is biased",
    ),
}

#: constraint id -> (sentence, {method: (score, reason)}) ; score 2 suitable, 1 partly, 0 unsuitable
CONSTRAINTS = {
    "nolist": (
        "There is no list of the {members}, and {person} cannot get one.",
        {
            "srs": (0, "a simple random sample needs every {member} to be listed and numbered, and there is no list"),
            "systematic": (0, "a systematic sample needs a list to step down, and there is no list"),
            "stratified": (0, "a stratified sample needs to know each {member}'s group, which needs a list"),
            "cluster": (
                2,
                "whole clusters such as sessions or places can be chosen at random without a list of individuals",
            ),
            "convenience": (1, "it needs no list, but it is likely to be biased"),
            "self": (1, "it needs no list, but it is biased towards those with strong views"),
        },
    ),
    "quick": (
        "The results are needed by tomorrow and there is almost no money to spend.",
        {
            "srs": (1, "it can be done quickly with a list, but contacting the chosen {members} takes time"),
            "systematic": (2, "it is quick to carry out down a list and cheap"),
            "stratified": (0, "it takes the most effort to organise, which does not fit one day and no money"),
            "cluster": (2, "the chosen {members} are close together, so it is cheap and quick"),
            "convenience": (1, "it is quick and almost free, but it is likely to be biased"),
            "self": (1, "it is almost free, but replies cannot be guaranteed by tomorrow and it is biased"),
        },
    ),
    "groups": (
        "The {members} fall into groups with very different views, and every group must be represented.",
        {
            "srs": (1, "by chance it can still miss or under-represent a group"),
            "systematic": (1, "it can miss a small group, because the groups are not taken into account"),
            "stratified": (2, "each group is sampled in proportion to its size, so every group is represented"),
            "cluster": (0, "whole clusters may all come from one group, so some groups could be missed"),
            "convenience": (0, "the easiest-to-reach {members} may all come from one group"),
            "self": (0, "the groups with strong views reply and the others may not reply at all"),
        },
    ),
    "decision": (
        "The results will be used to make an important decision about all the {members}.",
        {
            "srs": (2, "it gives every {member} an equal chance, so the results can be trusted for all {members}"),
            "systematic": (2, "it spreads the sample evenly through the list and is close to random"),
            "stratified": (2, "it is random and keeps every group in proportion, so the results can be trusted"),
            "cluster": (1, "it is random, but {members} in a cluster are alike, so the results are less reliable"),
            "convenience": (0, "it is likely to be biased, so it is not safe for an important decision"),
            "self": (0, "it is biased towards people with strong views, so it is not safe for an important decision"),
        },
    ),
}
_VERDICT = {2: "suitable", 1: "only partly suitable", 0: "not suitable"}


def _ctx_values(rng) -> dict:
    d = {}
    for name, spec in {**ctx_slots(), **NAME_SLOTS, "N": ri(300, 1800, 100)}.items():
        d[name] = spec(rng, d) if callable(spec) else rng.choice(spec)
    return d


def _f(s: str, d: dict) -> str:
    return s.format(**d)


def _pc_easy(rng) -> dict:
    d = _ctx_values(rng)
    mid = rng.choice(list(METHODS))
    m = METHODS[mid]
    d["m"] = m["name"]
    adv, dis = _f(m["adv"], d), _f(m["dis"], d)
    text = (
        f"{d['person']} plans to use a {d['m']} to find out what the {d['N']} {d['members']} of {d['org']} think about {d['topic']}. "
        f"Give one advantage and one disadvantage of using a {d['m']} for this survey."
    )
    mark = marking(
        reference=[f"Advantage: {adv}.", f"Disadvantage: {dis}."],
        points=[
            ("correct_idea", f"gives a genuine advantage of a {d['m']}: {adv}"),
            ("correct_idea", f"gives a genuine disadvantage of a {d['m']}: {dis}"),
            (
                "use_of_context",
                f"applies the points to the {d['members']} of {d['org']} rather than listing generic words",
            ),
            ("completeness", "gives both an advantage and a disadvantage, each explained"),
            ("communication", "writes in sentences and names the method correctly"),
        ],
        wrong=[
            (
                "swaps_advantage_and_disadvantage",
                f"does it present the weakness of a {d['m']} as its advantage, or the reverse?",
            ),
            ("vague_point", "is either point only a word such as 'easy' or 'hard' with no reason?"),
        ],
    )
    return finish(text, f"advantage: {adv}; disadvantage: {dis}", mark)


def _pc_medium(rng) -> dict:
    d = _ctx_values(rng)
    cid = rng.choice(list(CONSTRAINTS))
    mid = rng.choice(list(METHODS))
    sentence, table = CONSTRAINTS[cid]
    score, reason = table[mid]
    m = METHODS[mid]["name"]
    constraint = _f(sentence, d)
    reason = _f(reason, d)
    text = (
        f"{d['person']} is surveying the {d['N']} {d['members']} of {d['org']} about {d['topic']}. {constraint} "
        f"{d['person']} suggests using a {m}. Is a {m} a suitable choice here? Give a reason."
    )
    verdict = _VERDICT[score]
    mark = marking(
        reference=[f"A {m} is {verdict} here.", f"Reason: {reason}."],
        points=[
            ("correct_idea", f"judges the {m} {verdict} for this survey"),
            ("use_of_context", f"links the judgement to the stated constraint: {reason}"),
            ("completeness", "gives a reason for the verdict, not just yes or no"),
            ("communication", "states the verdict clearly before the reason"),
        ],
        wrong=[
            (
                "ignores_the_constraint",
                "does the answer ignore the constraint and describe the method in general terms?",
            ),
            (
                "wrong_verdict",
                f"does it call the {m} {'suitable' if score == 0 else 'unsuitable' if score == 2 else 'fully suitable or fully unsuitable'}?",
            ),
        ],
    )
    return finish(text, f"{m}: {verdict}", mark)


def _pc_hard(rng) -> dict:
    d = _ctx_values(rng)
    cid = rng.choice(list(CONSTRAINTS))
    sentence, table = CONSTRAINTS[cid]
    for _ in range(100):
        m1, m2 = rng.sample(list(METHODS), 2)
        if table[m1][0] != table[m2][0]:
            break
    else:  # pragma: no cover
        raise AssertionError("no pair with different verdicts")
    best, other = (m1, m2) if table[m1][0] > table[m2][0] else (m2, m1)
    nb, no = METHODS[best]["name"], METHODS[other]["name"]
    constraint = _f(sentence, d)
    text = (
        f"{d['person']} is surveying the {d['N']} {d['members']} of {d['org']} about {d['topic']}. {constraint} "
        f"{d['person']} cannot decide between a {METHODS[m1]['name']} and a {METHODS[m2]['name']}. "
        f"Which would you recommend? Justify your choice by giving an advantage of the method you choose and a disadvantage of the other."
    )
    mark = marking(
        reference=[
            f"Recommend the {nb}.",
            f"Advantage in this situation: {_f(table[best][1], d)}.",
            f"The {no} is weaker here: {_f(table[other][1], d)}.",
        ],
        points=[
            ("correct_idea", f"recommends the {nb} rather than the {no}"),
            ("correct_idea", f"gives a correct advantage of the {nb}: {_f(table[best][1], d)}"),
            (
                "use_of_context",
                f"uses the constraint to show why the {no} is the weaker choice: {_f(table[other][1], d)}",
            ),
            ("completeness", "gives both parts, an advantage of the choice and a disadvantage of the other"),
            ("communication", "recommends first, then justifies"),
        ],
        wrong=[
            ("advantages_only", "does it list advantages of both methods without choosing one?"),
            (
                "ignores_the_constraint",
                "does it choose by general features and ignore the constraint stated in the question?",
            ),
        ],
    )
    return finish(text, f"recommend the {nb}", mark)


def _pc_exam(rng) -> dict:
    d = _ctx_values(rng)
    for _ in range(400):
        c1, c2 = rng.sample(list(CONSTRAINTS), 2)
        s1, t1 = CONSTRAINTS[c1]
        s2, t2 = CONSTRAINTS[c2]
        m1, m2 = rng.sample(list(METHODS), 2)
        a, b = t1[m1][0] + t2[m1][0], t1[m2][0] + t2[m2][0]
        if a != b:
            top = m1 if a > b else m2
            if min(t1[top][0], t2[top][0]) >= 1:
                break
    else:  # pragma: no cover
        raise AssertionError("no pair with different totals")
    win, lose = (m1, m2) if a > b else (m2, m1)
    nw, nl = METHODS[win]["name"], METHODS[lose]["name"]
    text = (
        f"The {d['N']} {d['members']} of {d['org']} are to be surveyed about {d['topic']}. Two things are true of the survey.\n"
        f"First: {_f(s1, d)}\n"
        f"Second: {_f(s2, d)}\n"
        f"{d['person']} says a {METHODS[m1]['name']} is the best choice. {d['other']} says a {METHODS[m2]['name']} is the best choice. "
        f"Evaluate both claims against the two facts above and say which method you would use."
    )

    def r(t, mid, _c):
        return _f(t[mid][1], d)

    mark = marking(
        reference=[
            f"Use the {nw}.",
            f"First fact, for the {nw}: {r(t1, win, 1)}. Second fact: {r(t2, win, 2)}.",
            f"The {nl} is weaker: first fact, {r(t1, lose, 1)}; second fact, {r(t2, lose, 2)}.",
        ],
        points=[
            ("correct_idea", f"prefers the {nw} to the {nl}"),
            ("correct_idea", f"tests the {nw} against both facts: {r(t1, win, 1)}; {r(t2, win, 2)}"),
            ("use_of_context", f"explains where the {nl} falls short: {r(t1, lose, 1)}; {r(t2, lose, 2)}"),
            ("completeness", "answers both claims, not just one, and makes a final choice"),
            ("communication", "organises the answer around the two facts"),
        ],
        wrong=[
            ("one_fact_only", "does it use only one of the two facts to decide?"),
            ("accepts_both", "does it accept both claims as equally good without choosing?"),
        ],
    )
    return finish(text, f"use the {nw}", mark)


register_custom(
    "wa_sample_pros_cons",
    12,
    dict(easy=_pc_easy, medium=_pc_medium, hard=_pc_hard, exam=_pc_exam),
    doc="weigh a sampling method against a constraint.",
)

# ---------------------------------------------------------------------------
# wa_census_pros_cons
# ---------------------------------------------------------------------------
_HOURS_T = {6: "6 minutes", 12: "12 minutes", 15: "15 minutes", 20: "20 minutes", 30: "30 minutes"}


def _census_people(rng, v):
    return rng.choice([240, 300, 480, 600, 900, 1200, 1800])


CENSUS_CASES = [
    Case(
        "large_costly",
        slots={
            **ctx_slots(),
            **NAME_SLOTS,
            "N": _census_people,
            "t": [6, 12, 15, 20, 30],
            "c": [4, 5, 6, 8, 10, 12],
            "n": [60, 80, 100],
            "cost": lambda r, v: v["N"] * v["c"],
            "hours": lambda r, v: v["N"] * v["t"] // 60,
            "scost": lambda r, v: v["n"] * v["c"],
        },
        fields=dict(
            setup="{org} wants the views of its {N} {members} on {topic}. Each interview costs ${c} and takes {t} minutes. A census would mean interviewing all {N} {members}, which would cost ${cost} and take {hours} hours of interviewing. A sample of {n} {members} would cost ${scost}.",
            adv="every {member} is asked, so there is no sampling error and the views of small groups are included",
            dis="it is costly (${cost}) and slow ({hours} hours), and some {members} may not be reached",
            verdict="a sample",
            reason="a census would cost ${cost} against ${scost} for a sample, and a well-chosen sample of {n} would give a good estimate of the views of all {N} {members}",
            census_ok="no",
        ),
        wrong=[
            (
                "census_always_best",
                "does it say a census is always better because it asks everyone, without weighing the cost of ${cost} and {hours} hours?",
            )
        ],
    ),
    Case(
        "tiny_population",
        slots={
            **NAME_SLOTS,
            "N": [24, 30, 36, 45],
            "group": [
                "a Year 9 class",
                "the staff of a small business",
                "a netball team squad and its supporters' committee",
                "a local book club",
            ],
            "topic": ["which day to meet", "the best fundraising idea", "whether to change the meeting time"],
        },
        fields=dict(
            setup="{person} wants to know the views of every one of the {N} people in {group} on {topic}. All {N} can be reached at their next meeting, and asking each takes only about a minute.",
            adv="with so few people everyone can be asked quickly, so the result is the actual view of the whole group, not an estimate",
            dis="it still depends on everyone replying honestly, and absent people would be missed",
            verdict="a census",
            reason="the group is small and everyone can be reached easily, so a census is quick and gives the true answer for all {N} people",
            census_ok="yes",
        ),
        wrong=[
            (
                "sample_of_a_tiny_group",
                "does it recommend a sample even though the whole group of {N} can easily be asked?",
            )
        ],
    ),
    Case(
        "destructive",
        slots={
            **NAME_SLOTS,
            "N": [2000, 5000, 8000, 12000],
            "item": ["batteries", "light globes", "fireworks", "bicycle tyres"],
            "maker": ["Brightcell", "Lumo", "Skyburst", "Treadwell"],
            "n": [40, 50, 60],
        },
        fields=dict(
            setup="{maker} makes {N} {item} in a batch. To find how long they last, each one must be used until it fails. {person} says every one in the batch should be tested.",
            adv="testing every item would give the exact lifetime of the whole batch",
            dis="each item tested is used up, so testing all {N} would leave nothing to sell",
            verdict="a sample",
            reason="testing destroys the {item}, so only a random sample can be tested, and it estimates the lifetime of the whole batch",
            census_ok="no",
        ),
        wrong=[
            (
                "census_possible_here",
                "does it recommend testing all {N} {item} without noting that every one tested is used up?",
            )
        ],
    ),
    Case(
        "hard_to_reach",
        slots={
            **NAME_SLOTS,
            "N": [800, 1200, 1500, 2400],
            "back": lambda r, v: r.choice([k for k in range(30, 70, 5)]),
            "rate": lambda r, v: f"{v['back']}%",
            "count": lambda r, v: v["N"] * v["back"] // 100,
            "group": [
                "casual workers in a regional town",
                "visitors to a national park over a summer",
                "tenants in a large block of rental flats",
            ],
            "topic": ["how safe they feel", "what facilities they would use", "how satisfied they are with services"],
        },
        fields=dict(
            setup="A council tries to survey all {N} {group} about {topic} by posting a form to each. Only {back}% of the forms are returned, which is {count} forms.",
            adv="a census aims to include everyone, so no group should be left out by design",
            dis="in practice many forms are not returned ({count} of {N}), and those who do not reply may differ from those who do, so the result may still be biased",
            verdict="a sample",
            reason="a census does not guarantee full replies; a smaller random sample could be followed up thoroughly, giving a better response from the people chosen",
            census_ok="no",
        ),
        wrong=[("assumes_full_response", "does it assume a census means everyone replies?")],
    ),
]

CENSUS_TIERS = {
    "easy": Tier(
        prompt="{setup} Give one advantage and one disadvantage of using a census in this situation.",
        reference=["Advantage: {adv}.", "Disadvantage: {dis}."],
        points=[
            ("correct_idea", "gives a genuine advantage of a census here: {adv}"),
            ("correct_idea", "gives a genuine disadvantage of a census here: {dis}"),
            (
                "use_of_context",
                "uses details from the situation (the numbers or the type of item), not generic wording",
            ),
            ("completeness", "gives one advantage and one disadvantage, each with a reason"),
            ("communication", "writes in clear sentences"),
        ],
        wrong=[
            (
                "advantage_is_a_disadvantage",
                "does it list a feature of a census as an advantage when in this situation it is a disadvantage, or the reverse?",
            )
        ],
        answer="advantage: {adv}; disadvantage: {dis}",
    ),
    "medium": Tier(
        prompt="{setup} Would you recommend a census or a sample? Give two reasons that refer to the situation.",
        reference=["Recommend {verdict}.", "Because {reason}."],
        points=[
            ("correct_idea", "recommends {verdict}"),
            ("use_of_context", "uses the details of the situation: {reason}"),
            ("completeness", "gives two separate reasons, not one reason repeated"),
            ("communication", "states the recommendation first"),
        ],
        wrong=[("recommends_without_reasons", "does it recommend one option without two reasons from the situation?")],
        answer="{verdict}",
    ),
    "hard": Tier(
        prompt="{setup} {person} argues that a census is always better than a sample because it asks everyone. Explain whether {person} is right in this situation, and what a census can and cannot guarantee.",
        reference=[
            "{person} is only partly right: in a census {adv}.",
            "In this situation, though, {dis}.",
            "A census removes sampling error but cannot guarantee full, honest replies, and it can be too costly or impossible.",
        ],
        points=[
            ("correct_idea", "accepts the genuine strength of a census: {adv}"),
            ("correct_idea", "explains why it is not always better here: {dis}"),
            ("use_of_context", "uses the details of this situation to show the limit"),
            ("completeness", "says what a census cannot guarantee, such as full response or accuracy of answers"),
            ("communication", "reaches a clear verdict on {person}'s claim"),
        ],
        wrong=[
            ("agrees_always_better", "does it agree that a census is always better?"),
            ("rejects_census_entirely", "does it dismiss a census entirely without crediting that it asks everyone?"),
        ],
        answer="{person} is only partly right",
    ),
    "exam": Tier(
        prompt=(
            "Two organisations are deciding whether to run a census.\n"
            "Situation A: {a_setup}\n"
            "Situation B: {b_setup}\n"
            "For which situation is a census more appropriate? Justify your answer using both situations, and explain one reason why the census might still not give a completely accurate result."
        ),
        reference=[
            "A census is more appropriate for {pick_label}: {pick_reason}.",
            "In {other_label}, {other_dis}.",
            "Even a census can be imperfect because some people do not reply and some answers are wrong.",
        ],
        points=[
            ("correct_idea", "chooses {pick_label} as the better situation for a census"),
            ("use_of_context", "justifies from the details: {pick_reason}"),
            ("use_of_context", "explains why the other situation is a poorer fit: {other_dis}"),
            ("completeness", "gives a reason a census may still be imperfect (non-response, mistakes in answers)"),
            ("communication", "compares the two situations directly"),
        ],
        wrong=[("one_situation_only", "does it discuss only one of the two situations?")],
        two=True,
        pair_ok=lambda a, b: a["census_ok"] != b["census_ok"],
        derive=lambda v: (
            dict(pick_label="Situation A", pick_reason=v["a_reason"], other_label="Situation B", other_dis=v["b_dis"])
            if v["a_census_ok"] == "yes"
            else dict(
                pick_label="Situation B", pick_reason=v["b_reason"], other_label="Situation A", other_dis=v["a_dis"]
            )
        ),
        answer="{pick_label}",
    ),
}
register(
    "wa_census_pros_cons",
    12,
    CENSUS_CASES,
    CENSUS_TIERS,
    tails=TAILS,
    doc="when a census is sensible.",
)

# ---------------------------------------------------------------------------
# wa_census_procedure
# ---------------------------------------------------------------------------
PLAIN_CENSUS = [
    Case(
        "plain",
        slots={**ctx_slots(), **NAME_SLOTS, "N": ri(120, 900, 20)},
        fields=dict(
            setup="{org} plans a census of all of its {N} {members} about {topic}. {person} is in charge.",
            steps="define exactly who is in the population and what will be asked; get a complete list of all {N} {members}; write and trial a clear questionnaire; give it to every {member} and collect the replies; follow up those who have not replied; check and record the data; report the results",
        ),
        wrong=[
            ("only_some_asked", "does it describe asking only some of the {members}, which is a sample, not a census?")
        ],
    )
]

FLAW_CASES = [
    Case(
        "no_follow_up",
        slots={
            **ctx_slots(),
            **NAME_SLOTS,
            "N": ri(200, 1000, 100),
            "back": lambda r, v: r.choice([k for k in range(55, 90, 5)]),
            "got": lambda r, v: v["N"] * v["back"] // 100,
            "missing": lambda r, v: v["N"] - v["N"] * v["back"] // 100,
        },
        fields=dict(
            setup="{org} gives a form about {topic} to each of its {N} {members}. By the due date {got} forms ({back}%) are back, and {person} writes the report from those forms straight away.",
            step="following up the {members} who have not replied",
            why="{missing} {members} have not replied, and they may differ from those who did, so the census no longer covers everyone",
            fix="send reminders and contact the {members} who have not replied until nearly everyone is included",
        ),
        wrong=[
            (
                "blames_questionnaire",
                "does it blame the wording of the form, when the problem is the {missing} missing replies?",
            )
        ],
    ),
    Case(
        "old_list",
        slots={**ctx_slots(), **NAME_SLOTS, "N": ri(200, 1000, 50), "new": ri(20, 90, 10)},
        fields=dict(
            setup="{org} sends a form about {topic} to every {member} on last year's list of {N} {members}. Since then {new} new {members} have joined, and some on the list have left.",
            step="getting a complete and up-to-date list of the population",
            why="the {new} new {members} are not on the list, so they are never asked, and a census must include every {member}",
            fix="update the list before the census so that it matches the people who are in {org} now",
        ),
        wrong=[("blames_sample_size", "does it talk about sample size, when a census should have no sample at all?")],
    ),
    Case(
        "only_saturday",
        slots={
            **ctx_slots(),
            **NAME_SLOTS,
            "N": ri(200, 1000, 50),
            "present": lambda r, v: r.choice([k for k in range(40, 75, 5)]),
        },
        fields=dict(
            setup="{org} has {N} {members}. {person} hands out the form about {topic} at the Saturday session, which about {present}% of the {members} attend, and collects them there.",
            step="giving the form to every member of the population",
            why="the {members} who do not attend on Saturday are never given a form, so this is not a census and it may miss {members} with different views",
            fix="deliver the form to every {member} by other means as well, such as mail or email",
        ),
        wrong=[("calls_it_a_census", "does it accept that handing forms out at one session is still a census?")],
    ),
    Case(
        "no_privacy",
        slots={**ctx_slots(), **NAME_SLOTS, "N": ri(200, 1000, 50)},
        fields=dict(
            setup="{org} asks each of its {N} {members} to write their name and their answers about {topic} on a form and to put it in an open box in the front office, where anyone can read the forms.",
            step="keeping the replies private and confidential",
            why="the {members} can be identified and their answers read by others, so people may not answer honestly or at all",
            fix="make the forms anonymous or store them securely, and tell the {members} how their answers will be used",
        ),
        wrong=[("ignores_privacy", "does it overlook the privacy problem and suggest only a change to the questions?")],
    ),
    Case(
        "no_checking",
        slots={**ctx_slots(), **NAME_SLOTS, "N": ri(200, 1000, 50), "volunteers": [3, 4, 5], "dup": ri(6, 24, 2)},
        fields=dict(
            setup="{org} collects forms from all {N} {members} about {topic}. {volunteers} volunteers type the answers into a spreadsheet and the totals are published without anyone checking the entries. Later {dup} forms turn out to have been entered twice.",
            step="checking and cleaning the data before analysing it",
            why="the {dup} duplicated forms count some {members} twice, so the totals are wrong, and nobody checked for other typing mistakes",
            fix="check the entries against the forms, remove duplicates and look for impossible values before totalling",
        ),
        wrong=[("blames_the_volunteers", "does it blame the volunteers instead of the missing checking step?")],
    ),
]

CENSUS_PROC_TIERS = {
    "easy": Tier(
        cases=PLAIN_CENSUS,
        prompt="{setup} Describe, in order, the main steps involved in carrying out this census.",
        reference=["The steps are: {steps}.", "Every one of the {N} {members} must be included, not a sample."],
        points=[
            (
                "correct_idea",
                "puts the steps in a sensible order: define the population, list it, collect from every member, check, report",
            ),
            ("correct_idea", "makes clear that every {member} is included, not a sample"),
            ("use_of_context", "refers to the {members} of {org} and the survey about {topic}"),
            ("completeness", "includes following up non-replies and checking the data, not only handing out forms"),
            ("communication", "numbers or orders the steps clearly"),
        ],
        wrong=[
            ("describes_a_sample", "does it choose some {members} at random, which is a sample rather than a census?")
        ],
    ),
    "medium": Tier(
        cases=FLAW_CASES,
        prompt="{setup} One step of a good census procedure has been left out or done badly. Which step is it, and why does it matter?",
        reference=["The problem is with: {step}.", "It matters because {why}."],
        points=[
            ("correct_idea", "identifies the step: {step}"),
            ("use_of_context", "explains why it matters using the details given: {why}"),
            ("completeness", "gives a reason that links the step to the census result, not just its name"),
            ("communication", "names the step and the reason in separate clear sentences"),
        ],
        wrong=[
            (
                "names_a_different_step",
                "does it name a step that is not the one in question, such as the wording of the questions?",
            )
        ],
    ),
    "hard": Tier(
        cases=FLAW_CASES,
        prompt="{setup} Explain what is wrong with this procedure, how it could change the results, and what the organisers should do to fix it.",
        reference=["The step missing or done badly is {step}.", "Effect: {why}.", "Fix: {fix}."],
        points=[
            ("correct_idea", "identifies the faulty step: {step}"),
            ("use_of_context", "explains the effect on this census: {why}"),
            ("completeness", "proposes a workable fix: {fix}"),
            ("communication", "answers the three parts (problem, effect, fix) in order"),
        ],
        wrong=[
            (
                "fix_does_not_address_problem",
                "does the fix actually deal with the faulty step, not a different problem?",
            ),
            ("effect_missing", "does it describe the fault without saying how the results could be affected?"),
        ],
    ),
    "exam": Tier(
        cases=FLAW_CASES,
        two=True,
        pair_ok=lambda a, b: a["case_id"] != b["case_id"],
        prompt=(
            "Two organisations each ran a census.\n"
            "Census A: {a_setup}\n"
            "Census B: {b_setup}\n"
            "Identify the weakness in each census procedure. Which weakness is more likely to leave out people whose views matter, and so make the census results unreliable? Justify your answer and say how the other could be put right."
        ),
        reference=[
            "Census A: {a_step} is the weakness; {a_why}.",
            "Census B: {b_step} is the weakness; {b_why}.",
            "A weakness that leaves out whole groups of people (a missing list, forms given to only some people, or unreplied forms) damages reliability most; a fault in checking or privacy needs fixing but leaves the coverage intact.",
            "Fixes: for A, {a_fix}; for B, {b_fix}.",
        ],
        points=[
            ("correct_idea", "identifies the weakness in A: {a_step}"),
            ("correct_idea", "identifies the weakness in B: {b_step}"),
            ("use_of_context", "judges which weakness leaves out people and explains using the details given"),
            ("completeness", "says how the other procedure could be put right (A: {a_fix}; B: {b_fix})"),
            ("communication", "compares the two censuses and reaches a clear decision"),
        ],
        wrong=[("one_census_only", "does it discuss only one of the censuses?")],
        answer="A: {a_step}. B: {b_step}.",
    ),
}
register(
    "wa_census_procedure",
    12,
    PLAIN_CENSUS,
    CENSUS_PROC_TIERS,
    doc="steps of a census and where a plan fails.",
)

# ---------------------------------------------------------------------------
# wa_sampling_purpose
# ---------------------------------------------------------------------------
_PURPOSE_CTX = (
    ("Hillcrest High School", "students", "student", "walk to school"),
    ("Eastside Tennis Club", "members", "member", "play at least twice a week"),
    ("Bayview Library", "borrowers", "borrower", "borrowed an e-book last month"),
    ("Northgate Council", "ratepayers", "ratepayer", "support the new bike path"),
    ("Greenacre Retirement Village", "residents", "resident", "attend the weekly games afternoon"),
    ("Lakeside Fitness Centre", "members", "member", "use the pool at least once a week"),
)


def _pc_slots():
    return {
        **NAME_SLOTS,
        "ctx": _PURPOSE_CTX,
        "org": lambda r, v: v["ctx"][0],
        "members": lambda r, v: v["ctx"][1],
        "member": lambda r, v: v["ctx"][2],
        "yes": lambda r, v: v["ctx"][3],
        "n": [40, 50, 60, 80, 100, 120],
        "m": ri(8, 30),
        "N": lambda r, v: v["n"] * v["m"],
        "k": lambda r, v: r.randrange(v["n"] * 15 // 100, v["n"] * 70 // 100),
        "k2": lambda r, v: v["k"] + r.choice([-4, -3, -2, 2, 3, 4]),
        "rate": lambda r, v: pct(v["k"], v["n"]),
        "rate2": lambda r, v: pct(v["k2"], v["n"]),
        "est": lambda r, v: v["k"] * v["m"],
        "est2": lambda r, v: v["k2"] * v["m"],
        "frac": lambda r, v: pct(v["n"], v["N"]),
    }


PURPOSE_CASES = [
    Case(
        "purpose",
        slots=_pc_slots(),
        fields=dict(
            setup="{org} has {N} {members}. Asking all of them whether they {yes} would take too long and cost too much, so {person} asks a random sample of {n} {members} instead. {k} of the {n} ({rate}) say they {yes}.",
        ),
        wrong=[],
    )
]

PURPOSE_TIERS = {
    "easy": Tier(
        prompt="{setup} Why is a sample used here instead of a census, and what is the sample used to find out?",
        reference=[
            "A census of all {N} {members} would take too long and cost too much, so a sample is used.",
            "The sample is used to estimate a value for the whole population: the proportion of all {N} {members} who {yes}.",
        ],
        points=[
            (
                "correct_idea",
                "explains that a sample is used because a census of all {N} {members} is too costly or slow",
            ),
            ("correct_idea", "says the sample is used to estimate a population value, here the proportion who {yes}"),
            ("use_of_context", "refers to {org}, the {n} sampled and the {N} in the population"),
            ("completeness", "gives both the reason for sampling and what the sample is used for"),
            ("communication", "uses the words sample and population correctly"),
        ],
        wrong=[
            ("sample_gives_exact_answer", "does it say the sample tells exactly how many of the {N} {members} {yes}?"),
            ("only_cost", "does it give a reason for sampling but not say what the sample estimates?"),
        ],
        answer="a sample is cheaper and quicker, and estimates the population proportion",
    ),
    "medium": Tier(
        prompt="{setup} Use the sample to estimate how many of the {N} {members} {yes}. Explain how you got your estimate and why it is only an estimate.",
        reference=[
            "The sample proportion is {k} out of {n}, which is {rate}.",
            "Applying this to all {N} {members} gives {est} {members} who {yes}.",
            "It is only an estimate because another random sample of {n} would probably give a slightly different proportion.",
        ],
        points=[
            ("correct_idea", "uses the sample proportion {k} out of {n} ({rate}) and applies it to the population"),
            ("correct_idea", "gives an estimate of {est} {members} who {yes}"),
            ("use_of_context", "scales the sample to the {N} {members} of {org}"),
            ("completeness", "says why it is only an estimate: a different sample would give a different result"),
            ("communication", "shows the working in words"),
        ],
        wrong=[
            ("states_k_as_the_answer", "does it say {k} {members} {yes}, without scaling up to all {N}?"),
            ("claims_exact", "does it describe {est} as the exact number rather than an estimate?"),
        ],
        answer="about {est} {members}",
    ),
    "hard": Tier(
        prompt="{setup} {other} takes a second random sample of {n} {members} and finds {k2} who {yes}. Explain why the two samples give different estimates, say whether either must be wrong, and describe one way to get a more reliable estimate.",
        reference=[
            "{person}'s sample gives {rate} and an estimate of {est}; {other}'s gives {rate2} and an estimate of {est2}.",
            "The samples differ because different {members} were chosen by chance, and this sampling variation is expected.",
            "Neither is necessarily wrong; both estimate the same population value.",
            "A larger random sample would vary less and give a more reliable estimate.",
        ],
        points=[
            ("correct_idea", "explains that different random samples give different results through chance variation"),
            ("correct_idea", "says neither estimate has to be wrong (they are {est} and {est2})"),
            ("use_of_context", "refers to the two proportions, {rate} and {rate2}, from the {n} sampled each time"),
            ("completeness", "suggests a way to improve reliability, such as a larger sample or combining samples"),
            ("communication", "answers each of the three parts"),
        ],
        wrong=[
            (
                "one_must_be_wrong",
                "does it say one of the two samples must have been done badly because they disagree?",
            ),
            ("more_samples_no_help", "does it say taking a bigger sample would not help?"),
        ],
        answer="chance variation between samples; use a larger sample",
    ),
    "exam": Tier(
        prompt='{setup} {person} says: "The sample shows that exactly {est} {members} {yes}." {other} says: "The sample is too small to tell us anything, because it is only {frac} of the {members}." Evaluate both claims.',
        reference=[
            "{person} is wrong: {est} is an estimate from the sample proportion {rate}, not an exact count.",
            "{other} is also wrong: a random sample of {n} can estimate the population proportion well even if it is a small fraction of {N}; what matters most is that it is random and not tiny in absolute terms.",
            "The sensible view: the sample supports an estimate of about {est}, with some uncertainty, and a larger sample would reduce that uncertainty.",
        ],
        points=[
            ("correct_idea", "rejects {person}'s claim: {est} is an estimate, not an exact count"),
            (
                "correct_idea",
                "rejects {other}'s claim: a small fraction of the population can still give a useful estimate if the sample is random",
            ),
            ("use_of_context", "uses the figures: {rate} of {n}, scaled to {N}, and the sample fraction {frac}"),
            ("completeness", "gives a balanced conclusion about what the sample can and cannot show"),
            ("communication", "treats each claim in turn"),
        ],
        wrong=[
            ("agrees_with_exactness", "does it accept that the sample proves the exact number?"),
            (
                "agrees_fraction_matters",
                "does it accept that a sample must be a large fraction of the population to be useful?",
            ),
        ],
        answer="both claims are wrong; the sample gives an estimate of about {est}",
    ),
}
register(
    "wa_sampling_purpose",
    12,
    PURPOSE_CASES,
    PURPOSE_TIERS,
    tails=DATA_TAILS,
    doc="a sample estimates a population value.",
)

# ---------------------------------------------------------------------------
# wa_sampling_bias
# ---------------------------------------------------------------------------
_RANDOM_PROC = "give every {member} a number from 1 to {N}, use a random number generator to pick {n} different numbers, ignore repeats, and contact each {member} with a chosen number, following up non-replies"

BIAS_CASES = [
    Case(
        "phone_in",
        slots={
            **ctx_slots(),
            **NAME_SLOTS,
            "N": ri(400, 2000, 100),
            "n": [40, 50, 60],
            "r": ri(20, 80, 5),
            "rate": lambda r, v: pct(v["r"], v["N"]),
        },
        fields=dict(
            setup="{org} asks its {N} {members} to phone a number to give their views on {topic}. {r} {members} ({rate}) phone in.",
            source="self-selection (voluntary response)",
            over="{members} with strong opinions",
            under="{members} who are not bothered or who do not see the request",
            direction="results that show stronger views than the {members} as a whole hold",
            fix="choose {members} by random numbers from a list and contact them directly",
            proc=_RANDOM_PROC,
            rank=0,
        ),
        wrong=[
            (
                "blames_sample_size",
                "does it say the problem is only that {r} is too few, instead of who chose to reply?",
            )
        ],
    ),
    Case(
        "wrong_list",
        slots={
            **ctx_slots(),
            **NAME_SLOTS,
            "N": ri(400, 2000, 100),
            "n": [40, 50, 60],
            "list": [
                "the landline phone directory",
                "a list of people who bought tickets online",
                "the list of people who attend the Monday session",
            ],
        },
        fields=dict(
            setup="To survey all {N} {members} of {org} about {topic}, {person} chooses {n} names at random from {list}. Not every {member} appears on that list.",
            source="a sampling frame that leaves out part of the population (undercoverage)",
            over="{members} who appear on {list}",
            under="{members} who are not on {list}",
            direction="results that describe only the {members} on the list and may differ from all {members}",
            fix="build a list that includes every {member} before choosing at random",
            proc=_RANDOM_PROC,
            rank=1,
        ),
        wrong=[
            (
                "random_so_unbiased",
                "does it say the sample is unbiased because the names were chosen at random from the list?",
            )
        ],
    ),
    Case(
        "non_response",
        slots={
            **ctx_slots(),
            **NAME_SLOTS,
            "N": ri(500, 2000, 100),
            "back": lambda r, v: r.choice([k for k in range(10, 40, 5)]),
            "got": lambda r, v: v["N"] * v["back"] // 100,
            "n": [50, 60],
        },
        fields=dict(
            setup="{org} posts a survey about {topic} to {N} randomly chosen {members}. Only {got} of the {N} ({back}%) reply, and {person} writes up the answers of those {got} as the views of all {members}.",
            source="non-response",
            over="{members} who bother to reply",
            under="{members} who do not reply",
            direction="results that may differ from the views of those who did not reply",
            fix="follow up the {members} who did not reply, so that nearly all of the chosen sample is included",
            proc=_RANDOM_PROC,
            rank=1,
        ),
        wrong=[
            (
                "random_selection_enough",
                "does it say random selection alone removes the bias, ignoring the {back}% reply rate?",
            )
        ],
    ),
    Case(
        "location",
        slots={
            **ctx_slots(),
            **NAME_SLOTS,
            "N": ri(400, 2000, 100),
            "n": [30, 40, 50],
            "place": [
                "the sports store on the corner",
                "the front gate at 8 am",
                "the cafe next door",
                "the car park near the gym",
            ],
        },
        fields=dict(
            setup="{person} surveys {n} {members} of {org} about {topic} by approaching people outside {place} on one morning.",
            source="convenience sampling at one place and time",
            over="{members} who pass {place} at that time",
            under="{members} who are elsewhere at that time",
            direction="results that describe only the {members} who pass that place",
            fix="select {members} from a full list using random numbers rather than asking whoever is nearby",
            proc=_RANDOM_PROC,
            rank=1,
        ),
        wrong=[("unbiased_if_enough_asked", "does it say the sample is fine as long as enough people are asked?")],
    ),
    Case(
        "looks_friendly",
        slots={**ctx_slots(), **NAME_SLOTS, "N": ri(400, 2000, 100), "n": [20, 25, 30]},
        fields=dict(
            setup="To choose {n} of the {N} {members} of {org} to ask about {topic}, {person} looks around the room and picks those who seem friendly and willing to talk.",
            source="selection by the surveyor's personal choice",
            over="{members} who seem friendly and approachable to {person}",
            under="{members} who seem shy, busy or unfriendly",
            direction="results that favour the views of approachable {members}",
            fix="let random numbers, not {person}, decide who is chosen",
            proc=_RANDOM_PROC,
            rank=0,
        ),
        wrong=[("treats_as_random", "does it treat picking by eye as the same as choosing at random?")],
    ),
]


def _bias_derive(v: dict) -> dict:
    better, worse = ("a", "b") if v["a_rank"] > v["b_rank"] else ("b", "a")
    label = {"a": "Plan A", "b": "Plan B"}
    return dict(
        better_label=label[better],
        worse_label=label[worse],
        better_source=v[f"{better}_source"],
        worse_source=v[f"{worse}_source"],
        worse_fix=v[f"{worse}_fix"],
        worse_over=v[f"{worse}_over"],
        worse_under=v[f"{worse}_under"],
    )


BIAS_TIERS = {
    "easy": Tier(
        prompt="{setup} Describe the main source of bias in how this sample was chosen, and say which {members} are likely to be over-represented.",
        reference=["The source of bias is {source}.", "Over-represented: {over}."],
        points=[
            ("correct_idea", "identifies the source of bias: {source}"),
            ("use_of_context", "says who is over-represented in this survey: {over}"),
            ("completeness", "also says who is left out or under-represented: {under}"),
            ("communication", "links the source of bias to the way the sample was chosen"),
        ],
        wrong=[("names_no_source", "does it say the sample is 'biased' without saying how it was chosen?")],
        answer="{source}",
    ),
    "medium": Tier(
        prompt="{setup} Explain how the way the sample was chosen could make the results misleading about all the {members}.",
        reference=[
            "The source of bias is {source}.",
            "Over-represented: {over}; under-represented: {under}.",
            "So the results could be {direction}.",
        ],
        points=[
            ("correct_idea", "names the source of bias: {source}"),
            ("correct_idea", "explains who is over- and under-represented: {over}; {under}"),
            ("use_of_context", "concludes the results may be misleading about all the {members}: {direction}"),
            ("completeness", "connects the bias to the conclusion, not only the cause"),
            ("communication", "uses the words biased and representative correctly"),
        ],
        wrong=[("bias_without_effect", "does it name the bias but not say how the results could be misleading?")],
        answer="{direction}",
    ),
    "hard": Tier(
        prompt="{setup} Describe a procedure that {person} could use to choose a random sample of {n} of the {N} {members}, and explain why it removes the bias in this survey.",
        reference=[
            "Procedure: {proc}.",
            "It removes the bias because chance, not {person} or the {members}' own choices, decides who is included, so every {member} has the same chance.",
            "This deals with the problem: {source}.",
        ],
        points=[
            (
                "correct_idea",
                "describes a random method: numbering the {members} 1 to {N} and choosing {n} different random numbers",
            ),
            ("correct_idea", "explains that every {member} then has an equal chance of selection"),
            ("use_of_context", "links the procedure to the bias in this survey: {source}"),
            ("completeness", "mentions ignoring repeated numbers and following up the {members} who do not reply"),
            ("communication", "sets the procedure out as ordered steps"),
        ],
        wrong=[
            (
                "not_random",
                "is the procedure really random, or does it still depend on who is easy to reach or who volunteers?",
            ),
            ("no_list", "does it forget that every {member} must be listed or numbered first?"),
        ],
        answer="number the {members}, pick {n} random numbers",
    ),
    "exam": Tier(
        two=True,
        pair_ok=lambda a, b: a["rank"] != b["rank"] and a["source"] != b["source"],
        derive=_bias_derive,
        prompt=(
            "Two organisations each used a sample to survey their members.\n"
            "Plan A: {a_setup}\n"
            "Plan B: {b_setup}\n"
            "Identify the source of bias in each plan. Which plan is the less random, and so the more likely to be biased? Justify your answer, and say how the less random plan could be improved."
        ),
        reference=[
            "Plan A: {a_source}. Plan B: {b_source}.",
            "{worse_label} is less random: {worse_source}, which over-represents {worse_over} and under-represents {worse_under}.",
            "Improvement for {worse_label}: {worse_fix}.",
        ],
        points=[
            ("correct_idea", "identifies Plan A's bias: {a_source}"),
            ("correct_idea", "identifies Plan B's bias: {b_source}"),
            (
                "use_of_context",
                "judges that {worse_label} is the less random and explains who is over- and under-represented",
            ),
            ("completeness", "improves {worse_label}: {worse_fix}"),
            ("communication", "compares the two plans directly"),
        ],
        wrong=[("larger_sample_is_better", "does it choose the plan only because more people were asked?")],
        answer="{worse_label} is less random; {worse_fix}",
    ),
}
register(
    "wa_sampling_bias",
    12,
    BIAS_CASES,
    BIAS_TIERS,
    doc="bias in selection and procedures that give randomness.",
)

check(len(KIND_CASES) == 5, "five sampling methods")
