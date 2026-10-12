"""Generates `sampling_techniques` items: classify a described way of collecting data (census, random sample,
non-random sample, experiment, observation) and give a practical, ethical or bias implication for the question
being asked.

Every plan is built from a scenario (who is studied, what is asked, how a convenient or volunteer sample would
skew, whether an experiment is ethical) and a plan family; the numbers (population size, sample size, every k-th,
stratum counts, response rate) are computed so they are consistent. The answer is marked against a rubric, so
`answer` is a marking dict (see `_crit_kit`).

Ladder:

- easy: census or simple random sample; say which and give one practical reason for the choice.
- medium: random or non-random plan (simple random, systematic, stratified, convenience, volunteer); name the
  method, say whether it is random, and describe the bias or lack of it.
- hard: experiment or observation as well; classify, and give the ethical or practical implication, including
  whether treatments were randomly assigned.
- exam: two plans for the same question with different sizes and response rates; classify both, compare their
  bias, practicality and ethics, and recommend one.
"""

from __future__ import annotations

from ..core import check, generator
from ._crit_kit import attempts, Prompt, SHORT_TAILS, finish, short_marking

_PRESETS = {d: {"variant": d} for d in ("easy", "medium", "hard", "exam")}

# who, N choices, question, convenient place (+ the skew), volunteer skew, can_volunteer, experiment, ethics
SCENARIOS = [
    dict(
        causal=False,
        who="students at a secondary school",
        sizes=(480, 600, 640, 720, 800, 960),
        place="the school library at lunch",
        q="how many hours a day students spend on screens",
        conv="students found in the library at lunch are more likely to be keen readers and studiers, so heavy screen users are probably under-represented",
        vol="students who feel strongly about screens are the most likely to reply, so the replies lean to the extremes",
        volunteer=True,
        treat="limiting one group to one hour of screen time a day for a month and measuring sleep in both groups",
        ethic_ok=True,
        ethic="needs consent from the students and their parents, and the limit must not harm anyone",
        census_note="asking every student is possible, but it takes a lot of class time",
    ),
    dict(
        causal=False,
        who="residents of a town",
        sizes=(1200, 1500, 2400, 3000, 4800),
        place="the railway station at the morning peak",
        q="how people usually travel to work",
        conv="people at the station in the morning peak are mostly rail commuters, so car and bus travellers are under-represented",
        vol="residents who are unhappy with the services are more likely to answer, so the replies over-state complaints",
        volunteer=True,
        treat="giving one group free train tickets for a month and recording how both groups travel",
        ethic_ok=True,
        ethic="the free tickets are a benefit, so the unequal treatment needs to be explained and the residents' consent obtained",
        census_note="visiting every household is too slow and too costly for one team",
    ),
    dict(
        causal=False,
        people=False,
        who="batteries from one factory batch",
        sizes=(1200, 1800, 2400, 3600),
        place="the top box, which was packed last",
        q="how many hours a battery lasts",
        conv="the top box was packed last, so it may hold batteries from a different production run from the rest of the batch",
        vol="",
        volunteer=False,
        treat="running one group in a new device and the other group in the old one and timing both",
        ethic_ok=True,
        ethic="there is no ethical problem for objects, but a test that uses a battery up cannot be run on the whole batch",
        census_note="testing every battery until it runs flat would destroy the whole batch, so a census is not practical",
    ),
    dict(
        causal=True,
        who="patients at a clinic with high blood pressure",
        sizes=(240, 300, 360, 480),
        place="the waiting room on Monday morning",
        q="whether a new tablet lowers blood pressure",
        conv="patients who attend on Monday morning may be older or more severe cases, so the sample does not represent all patients",
        vol="patients who feel their treatment is not working are more likely to volunteer, so the replies are biased towards poor results",
        volunteer=True,
        treat="giving one group the new tablet and the other group the usual treatment, then measuring blood pressure in both",
        ethic_ok=False,
        ethic="every patient must give informed consent, and withholding treatment from patients who need it is unethical, so the comparison group must get the usual treatment",
        census_note="measuring every patient is possible, but each extra measurement costs clinic time",
    ),
    dict(
        causal=False,
        who="households in a suburb",
        sizes=(600, 800, 1000, 1200, 1500),
        place="the street where the surveyor lives",
        q="how much electricity a household uses in a month",
        conv="households in one street are likely to be similar in age, size and income, so they do not represent the suburb",
        vol="households that are interested in saving energy are more likely to fill in an online form, so the replies under-state the average use",
        volunteer=True,
        treat="giving one group a display showing their use and comparing both groups' bills",
        ethic_ok=True,
        ethic="the households must agree to share their bills, and their details must be kept private",
        census_note="reading every household's meter is possible but costly",
    ),
    dict(
        causal=False,
        who="members of a sports club",
        sizes=(240, 360, 480, 600),
        place="the club room after Saturday matches",
        q="how many training sessions a member attends each week",
        conv="members who stay after a match are the keenest, so the number of sessions is probably over-stated",
        vol="members who train a lot are more likely to answer a club-wide email, so the replies over-state attendance",
        volunteer=True,
        treat="asking one group to add one extra session a week and comparing fitness tests in both groups",
        ethic_ok=False,
        ethic="asking members to train more than is safe could cause injury, so the extra training must be within safe limits and each member must be free to say no",
        census_note="asking every member is possible because the club keeps a full list of them",
    ),
]

OBS = (
    "follows {n} of the students for a month, recording their daily screen hours and leaving their habits unchanged",
    "follows {n} of the residents for a month, recording how each travels to work and changing nothing",
    "records how long {n} batteries last in ordinary use in customers' devices, without changing how they are used",
    "compares the blood pressure of {n} patients who chose to take the tablet with that of patients who chose not to, and changes nothing",
    "records the monthly bills of {n} households and changes nothing about how they use power",
    "records the weekly attendance of {n} members for a month and changes nothing about their training",
)
for _sc, _o in zip(SCENARIOS, OBS):
    _sc["obs"] = _o
    _sc["act"] = "tests" if _sc["who"].startswith("batteries") else "asks"


def _plan(rng, sc: dict, kind: str, N: int | None = None) -> dict:
    """One described plan. `group` is the broad type (census, random sample, non-random sample, experiment,
    observation); `label` names the method; `random` says whether who is chosen (or assigned) is random."""
    N = N or rng.choice(sc["sizes"])
    who, act = sc["who"], sc["act"]
    out = dict(kind=kind, N=N, ethics=sc["ethic"])
    if kind == "census":
        return dict(
            out,
            n=N,
            group="census",
            label="a census",
            random=None,
            text=f"{act} all {N} of the {who}.",
            practical=sc["census_note"],
            bias="everyone is included, so who is chosen cannot bias it, though people who do not answer still could",
        )
    if kind == "simple_random":
        n = rng.choice([x for x in (40, 50, 60, 80, 100, 120) if x * 4 <= N])
        return dict(
            out,
            n=n,
            group="random sample",
            label="a simple random sample",
            random=True,
            text=f"numbers the {N} {who} from 1 to {N} and uses a random number generator to pick {n} of them to be {'asked' if act == 'asks' else 'tested'}.",
            practical=f"{n} of {N} is only {100 * n / N:g}% of them, far quicker than {act[:-1]}ing all {N}",
            bias="every one of them has the same chance of being chosen, so the choice of who is asked does not bias it",
        )
    if kind == "systematic":
        k = rng.choice([x for x in (5, 8, 10, 12, 15, 20) if N % x == 0])
        return dict(
            out,
            n=N // k,
            group="random sample",
            label="a systematic sample",
            random=True,
            text=f"takes the list of all {N} of the {who}, picks a starting place at random from the first {k}, and then {act} every {k}th one on the list, which gives {N // k} of them.",
            practical=f"every {k}th of {N} gives {N // k}, which is much quicker than all {N}",
            bias=f"the starting place is random so each has the same chance, but it would be biased if the list had a pattern that repeats every {k}",
        )
    if kind == "stratified":
        f = rng.choice((10, 20, 25))
        counts = [20 * rng.randint(2, 12) for _ in range(rng.choice((2, 3)))]
        N = sum(counts)
        picks = [c * f // 100 for c in counts]
        names = (
            ["younger", "middle-aged", "older"][: len(counts)]
            if not who.startswith("batteries")
            else ["line 1", "line 2", "line 3"][: len(counts)]
        )
        desc = ", ".join(f"{c} {nm}" for c, nm in zip(counts, names))
        pk = ", ".join(f"{p} {nm}" for p, nm in zip(picks, names))
        return dict(
            out,
            N=N,
            n=sum(picks),
            group="random sample",
            label="a stratified random sample",
            random=True,
            text=f"splits the {N} {who} into groups ({desc}) and picks {f}% of each group at random, which gives {pk}, and {sum(picks)} in all.",
            practical=f"{f}% of {N} is {sum(picks)}, far fewer than all {N}",
            bias="every group is represented in proportion to its size and chosen at random inside the group, so the choice of who is asked does not bias it",
        )
    if kind == "convenience":
        n = rng.choice((30, 40, 50, 60, 80))
        return dict(
            out,
            n=n,
            group="non-random sample",
            label="a convenience sample",
            random=False,
            text=f"{act} the first {n} of the {who} {'met at' if sc.get('people', True) else 'taken from'} {sc['place']}.",
            practical=f"{n} people close by is quick and cheap",
            bias=sc["conv"],
        )
    if kind == "volunteer":
        check(sc["volunteer"], "this scenario has no volunteers")
        pct = rng.choice((10, 15, 20, 25))
        got = N * pct // 100
        check(got * 100 == N * pct, "response is whole")
        return dict(
            out,
            n=got,
            pct=pct,
            group="non-random sample",
            label="a self-selected (volunteer) sample",
            random=False,
            text=f"invites all {N} of the {who} to answer an online poll, and {got} answer ({pct}%).",
            practical=f"the poll costs almost nothing, but only {pct}% answer",
            bias=sc["vol"] + f", and a bigger number of replies ({got}) does not remove this",
        )
    if kind in ("experiment_random", "experiment_chosen"):
        N -= N % 2
        h = N // 2
        if kind == "experiment_random":
            how = f"randomly splits {N} of the {who} into two groups of {h} using a random number generator, and then runs the study by {sc['treat']}."
            return dict(
                out,
                N=N,
                n=N,
                group="experiment",
                label="an experiment with random assignment",
                random=True,
                text=how,
                practical="random assignment makes the two groups alike apart from the treatment, so a difference can be put down to it",
                bias="because the groups were assigned at random, other differences between them are spread evenly",
            )
        how = f"lets each of {N} of the {who} choose which of two groups to join ({h} end up in each), and then runs the study by {sc['treat']}."
        return dict(
            out,
            N=N,
            n=N,
            group="experiment",
            label="an experiment without random assignment",
            random=False,
            text=how,
            practical="the treatment is applied, so it is an experiment, but the groups were not formed at random",
            bias="people who choose a group may differ in other ways (for example, motivation), so a difference may not be due to the treatment",
        )
    if kind == "observation":
        n = rng.choice((60, 80, 100, 120, 150))
        return dict(
            out,
            n=n,
            group="observation",
            label="an observational study",
            random=None,
            text=sc["obs"].format(n=n) + ".",
            practical="nothing is imposed on anyone, which is cheaper and avoids any harm from a treatment",
            bias="because nothing is assigned, groups may differ in other ways, so the study can show a link but not that one thing causes another",
        )
    raise ValueError(kind)


def _stem(sc: dict) -> str:
    return f"A researcher wants to find out {sc['q']}. "


def _plan_marking(sc: dict, pl: dict, ask: str) -> dict:
    ref = [
        f"The plan is {pl['label']}, which is a kind of {pl['group']}.",
        f"Why: {pl['text'].rstrip('.')}.",
        f"Bias: {pl['bias']}.",
        f"Practical: {pl['practical']}. Ethics: {pl['ethics']}.",
    ]
    steps = [
        ("correct_idea", f"names the type of plan correctly: {_name(pl).lower()}"),
        ("completeness", f"gives the feature that decides it: {pl['text'].rstrip('.')}"),
        (
            "use_of_context",
            {
                "bias": f"states the bias or the lack of it: {pl['bias']}",
                "practical": f"states the practical point: {pl['practical']}",
                "ethics": f"states the ethical point: {pl['ethics']}",
            }[ask],
        ),
        (
            "communication",
            "uses the right word for each idea (census, sample, random, experiment, observation) in a short, clear answer",
        ),
    ]
    traps = [
        (
            "random_vs_non_random",
            "does it call a plan random when who is chosen is decided by convenience or by who volunteers, or non-random when every one has an equal chance?",
        ),
        ("bigger_fixes_bias", "does it say that a larger number of replies removes bias from a non-random sample?"),
        (
            "experiment_vs_observation",
            "does it call a study an experiment when nothing is applied to the groups, or an observation when a treatment is applied?",
        ),
    ]
    return dict(ref=ref, steps=steps, traps=traps)


_ASK = {
    "bias": "Say what bias, if any, this plan could have and why.",
    "practical": "Say what is practical about this plan and what limits it.",
    "ethics": "Say what ethical point has to be dealt with.",
}


def _name(pl: dict) -> str:
    lab = pl["label"][0].upper() + pl["label"][1:]
    return lab if pl["group"] in pl["label"] or pl["label"] == "a census" else f"{lab} ({pl['group']})"


def _single(rng, kinds: tuple[str, ...], ask_choices: tuple[str, ...], head: str) -> dict:
    for _ in attempts(500, "a plan"):
        sc = rng.choice(SCENARIOS)
        kind = rng.choice(kinds)
        if kind == "volunteer" and not sc["volunteer"]:
            continue
        pl = _plan(rng, sc, kind)
        break
    ask = rng.choice([a for a in ask_choices if a != "ethics" or sc.get("people", True)])
    p = (
        Prompt()
        .t(_stem(sc))
        .t(f"The researcher {pl['text']} ")
        .t(head + " " + _ASK[ask] + " ")
        .t(rng.choice(SHORT_TAILS))
    )
    m = _plan_marking(sc, pl, ask)
    return finish(p, _name(pl) + ".", short_marking(reference=m["ref"], steps=m["steps"], traps=m["traps"]))


def _easy(rng) -> dict:
    return _single(
        rng,
        ("census", "simple_random", "census", "simple_random", "systematic"),
        ("practical",),
        "Is this a census or a sample? Name the method and say what that means.",
    )


def _medium(rng) -> dict:
    return _single(
        rng,
        ("simple_random", "systematic", "stratified", "convenience", "volunteer"),
        ("bias",),
        "Name the method, and say whether it is a random or a non-random sample and why.",
    )


def _hard(rng) -> dict:
    return _single(
        rng,
        ("experiment_random", "experiment_chosen", "observation", "volunteer", "convenience"),
        ("bias", "ethics"),
        "Say whether this is an experiment, an observational study or a sample, and whether anything is random.",
    )


def _exam(rng) -> dict:
    sc = rng.choice(SCENARIOS)
    if sc["causal"]:
        a, b = _plan(rng, sc, "observation"), _plan(rng, sc, "experiment_random")
        good, why = (
            b,
            "only the experiment with random assignment can show that the tablet causes a change, as long as the comparison group is not denied the usual treatment",
        )
    else:
        weak = rng.choice(("convenience", "volunteer") if sc["volunteer"] else ("convenience",))
        b = _plan(rng, sc, rng.choice(("simple_random", "systematic", "stratified")))
        a = _plan(rng, sc, weak, b["N"])
        if weak == "convenience":
            a["n"] = max(a["n"], 30)
        good, why = (
            b,
            "its sample is random, so it is not biased by who is asked, even though it asks fewer than the other plan"
            if b["n"] < a["n"]
            else "its sample is random, so it is not biased by who is asked",
        )
    pair = [a, b]
    rng.shuffle(pair)
    names = ("Plan A", "Plan B")
    p = Prompt().t(_stem(sc))
    for nm, pl in zip(names, pair):
        p.t(f"{nm}: the researcher {pl['text']} ")
    p.t(
        "For each plan, name the method and say whether who is chosen or assigned is random. Compare the plans for bias, for practicality and for ethics, and recommend one plan with a reason. "
    ).t(rng.choice(SHORT_TAILS))
    rec = names[pair.index(good)]
    ref = [
        *(f"{nm} is {_name(pl).lower()}: {pl['text'].rstrip('.')}." for nm, pl in zip(names, pair)),
        *(f"{nm}, bias: {pl['bias']}." for nm, pl in zip(names, pair)),
        f"Practical and ethical points: {pair[0]['practical']} (Plan A); {pair[1]['practical']} (Plan B). {sc['ethic'][0].upper()}{sc['ethic'][1:]}.",
        f"Recommend {rec}: {why}.",
    ]
    steps = [
        ("correct_idea", "; ".join(f"{nm} is {_name(pl).lower()}" for nm, pl in zip(names, pair))),
        ("completeness", f"recommends {rec}, because {why}"),
        (
            "use_of_context",
            "uses the numbers given (sample sizes, response rates) and compares bias, practicality and ethics; "
            + "; ".join(f"{nm}: {pl['bias']}" for nm, pl in zip(names, pair)),
        ),
        (
            "communication",
            "keeps the two plans apart, compares them point by point and ends with a clear recommendation",
        ),
    ]
    traps = [
        (
            "bigger_is_better",
            "does it prefer the plan with more replies or a larger sample without saying whether who is chosen is random?",
        ),
        ("random_vs_non_random", "does it call a convenience or volunteer sample random?"),
        ("no_ethics", "does it leave out the ethical point, or the effect on who is asked?"),
        ("causation_from_observation", "does it say an observational study shows that one thing causes another?"),
    ]
    return finish(p, f"Recommend {rec}.", short_marking(reference=ref, steps=steps, traps=traps))


FAMILIES = {"easy": [_easy], "medium": [_medium], "hard": [_hard], "exam": [_exam]}


@generator("sampling_techniques", level=14, presets=_PRESETS)
def sampling_techniques(rng, variant="easy"):
    """`f10-sampling-techniques`. Rubric-assessed (`answer: rubric`): every question returned carries a
    per-item marking dict (`Question.answer`), never a plain value."""
    if variant not in FAMILIES:
        raise ValueError(f"unknown variant: {variant}")
    return rng.choice(FAMILIES[variant])(rng)
