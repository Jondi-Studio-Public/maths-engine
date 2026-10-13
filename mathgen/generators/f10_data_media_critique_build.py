"""Generates `data_media_critique` items: read a short statistical claim from the media and pick out its data,
its display, its method and its conclusion.

A report is assembled from a topic (who was asked and what they said), a counted result (the counts and the
share are computed), a way of collecting (online poll, random phone survey, interviews at a shopping centre,
...), a described graph (its feature is stated in words; no picture is shown) and a headline conclusion. The
answer is marked against a rubric, so `answer` is a marking dict (see `_crit_kit`).

Ladder:

- easy: four sentences in the order data, method, display, conclusion; identify each part.
- medium: the sentences are mixed up and one more (a remark by someone quoted) belongs to none of the four; identify
  each part and say which sentence belongs to none.
- hard: the data is given only as a percentage and a total; identify the four parts and work out the count that
  said yes.
- exam: two reports on the same question that differ in method, display and in whom the conclusion is about;
  identify the four parts of each and say which parts differ.
"""

from __future__ import annotations

from ..core import check, generator
from ._crit_kit import attempts, Prompt, SHORT_TAILS, finish, short_marking

_PRESETS = {d: {"variant": d} for d in ("easy", "medium", "hard", "exam")}

TOPICS = [
    dict(group="students", did="said they walk to school", short="walk to school", wide="students in the country"),
    dict(
        group="drivers",
        did="said they have used a phone while driving",
        short="use a phone while driving",
        wide="drivers in the state",
    ),
    dict(
        group="households",
        did="said they recycle every week",
        short="recycle every week",
        wide="households in the city",
    ),
    dict(
        group="teenagers",
        did="said they sleep less than eight hours on school nights",
        short="sleep too little on school nights",
        wide="teenagers in the country",
    ),
    dict(
        group="shoppers",
        did="said they would buy the new flavour",
        short="would buy the new flavour",
        wide="shoppers in the country",
    ),
    dict(
        group="residents",
        did="said they support the new bike path",
        short="support the new bike path",
        wide="people in the region",
    ),
    dict(
        group="workers",
        did="said they work from home at least one day a week",
        short="work from home",
        wide="workers in the country",
    ),
]

METHODS = [
    dict(
        key="online",
        kind="an online poll that anyone could answer (self-selected, so not random)",
        text="the figures come from an online poll on the website of a company that sells related products, which anyone who visited the site could answer",
    ),
    dict(
        key="phone",
        kind="a phone survey of people chosen at random from a full list (a random sample)",
        text="the people were chosen at random from a full list of {pop} and rung by phone",
    ),
    dict(
        key="mall",
        kind="interviews with people the reporter happened to meet (a convenience sample, not random)",
        text="the people were the first ones that a reporter met and interviewed in a shopping centre on a Saturday morning",
    ),
    dict(
        key="class",
        kind="a questionnaire given to one class (a small, convenient group, not random)",
        text="the people were from a single class who were given the questionnaire by their teacher",
    ),
]

SPEAKERS = (
    "a spokesperson for the company that paid for the report",
    "a local councillor",
    "a retired teacher who was interviewed",
    "the editor of the newspaper",
)
REMARKS = (
    "said the result was no surprise to anyone.",
    "said that people should make up their own minds.",
    "said that more work was needed on the topic.",
    "said that this was a good start.",
)


def _share(rng, total: int) -> tuple[int, int]:
    for _ in attempts(200, "a share"):
        pct = rng.choice(range(10, 100, 5))
        if total * pct % 100 == 0:
            return total * pct // 100, pct
    raise AssertionError


def _strength(pct: int) -> str:
    if pct >= 90:
        return "almost all"
    if pct >= 70:
        return "most"
    if pct > 50:
        return "more than half of"
    if pct >= 45:
        return "about half of"
    if pct >= 25:
        return "fewer than half of"
    return "only a few"


def _of(strength: str) -> str:
    return strength if strength.endswith(" of") else strength + " of"


def _display(rng, yes: int, no: int, total: int, pct: int, group: str) -> tuple[str, str]:
    """(sentence about the graph, short description)."""
    k = rng.choice(("axis", "pie", "wide", "picto"))
    if k == "axis":
        start = max(0, min(yes, no) // 10 * 10 - 10)
        where = f"starts at {start} instead of 0" if start else "starts at 0"
        return (
            f"The results are shown in a bar graph with two bars, one for yes and one for no, and the vertical axis {where}.",
            f"a bar graph of the yes and no counts, with the vertical axis starting at {start}",
        )
    if k == "pie":
        return (
            f"The results are shown in a pie chart with a yes slice labelled {pct}% and a no slice labelled {100 - pct}%.",
            f"a pie chart with slices of {pct}% (yes) and {100 - pct}% (no)",
        )
    if k == "wide":
        return (
            "The results are shown in a column graph in which the yes column is drawn twice as wide as the no column.",
            "a column graph in which the yes column is drawn twice as wide as the no column",
        )
    return (
        "The results are shown in a pictograph in which each person symbol stands for 10 people.",
        "a pictograph in which each person symbol stands for 10 people",
    )


def _report(rng, topic: dict, total: int, method: dict, *, as_percent: bool, pop: int) -> dict:
    yes, pct = _share(rng, total)
    if as_percent:
        data_s = f"In a survey of {total} {topic['group']}, {pct}% {topic['did']}."
        data_k = f"a survey of {total} {topic['group']} in which {pct}% ({yes}) {topic['did']}"
    else:
        data_s = f"In a survey of {total} {topic['group']}, {yes} {topic['did']}."
        data_k = f"a survey of {total} {topic['group']} in which {yes} ({pct}%) {topic['did']}"
    m_text = method["text"].format(pop=f"{pop} {topic['group']}")
    method_s = f"{m_text[0].upper()}{m_text[1:]}."
    disp_s, disp_k = _display(rng, yes, total - yes, total, pct, topic["group"])
    concl_s = f"The headline says: {_strength(pct).capitalize()} {topic['group']} {topic['short']}."
    return dict(
        data=data_s,
        data_key=data_k,
        method=method_s,
        method_key=method["kind"],
        display=disp_s,
        display_key=disp_k,
        conclusion=concl_s,
        concl_key=f"{_strength(pct)} {topic['group']} {topic['short']}",
        yes=yes,
        pct=pct,
        total=total,
        topic=topic,
    )


def _marking(r: dict, extra: list[str], dist: str | None, ask_count: bool) -> dict:
    ref = [
        f"Data: {r['data_key']}.",
        f"Display: {r['display_key']}.",
        f"Method: {r['method_key']}.",
        f"Conclusion: the headline claims that {r['concl_key']}.",
    ] + extra
    steps = [
        ("correct_idea", f"identifies the data as {r['data_key']}"),
        ("correct_idea", f"identifies the conclusion as the headline claim: {r['concl_key']}"),
        (
            "completeness",
            f"keeps the method (how the people were chosen and asked: {r['method_key']}) apart from the display ({r['display_key']})",
        ),
        ("use_of_context", "quotes or refers to the sentence in the report for each of the four parts"),
        ("communication", "labels each part clearly (data, display, method, conclusion) in a short answer"),
    ]
    if ask_count:
        steps.append(("correct_idea", f"works out the count: {r['pct']}% of {r['total']} is {r['yes']}"))
    if dist:
        steps.append(
            (
                "completeness",
                f"says that the remark by {dist} is none of the four parts: it is an opinion, not data, method, display or conclusion",
            )
        )
    traps = [
        (
            "method_vs_display",
            "does it give the graph or chart as the method, or the way the people were chosen as the display?",
        ),
        ("data_vs_conclusion", "does it give the headline claim as the data, or the counts as the conclusion?"),
        ("opinion_as_part", "does it count a remark by someone quoted as the data or the conclusion?"),
    ]
    if ask_count:
        traps.append(("wrong_count", f"is the count right (the share of the total, here {r['pct']}% of {r['total']})?"))
    return short_marking(reference=ref, steps=steps, traps=traps)


def _build(rng, *, mix: bool, remark: bool, as_percent: bool, ask_count: bool) -> dict:
    topic = rng.choice(TOPICS)
    total = rng.choice((80, 100, 120, 150, 200, 250, 400, 500))
    method = rng.choice(METHODS)
    r = _report(rng, topic, total, method, as_percent=as_percent, pop=total * rng.choice((10, 20, 50)))
    sents = [r["data"], r["method"], r["display"], r["conclusion"]]
    dist = None
    if remark:
        dist = rng.choice(SPEAKERS)
        sents.append(f"{dist[0].upper()}{dist[1:]} {rng.choice(REMARKS)}")
    if mix:
        rng.shuffle(sents)
    p = Prompt().t("Read this report from the media. " + " ".join(f'"{s}"' if False else s for s in sents) + " ")
    q = "Identify the data, the display, the method and the conclusion in the report."
    if remark:
        q += " Say which sentence belongs to none of these four."
    if ask_count:
        q += " Work out how many people said yes."
    p.t(q + " ").t(rng.choice(SHORT_TAILS))
    ans = (
        f"Data: {r['data_key']}. Display: {r['display_key']}. Method: {r['method_key']}. Conclusion: {r['concl_key']}."
    )
    return finish(p, ans, _marking(r, [], dist, ask_count))


def _easy(rng) -> dict:
    return _build(rng, mix=False, remark=False, as_percent=False, ask_count=False)


def _medium(rng) -> dict:
    return _build(rng, mix=True, remark=True, as_percent=False, ask_count=False)


def _hard(rng) -> dict:
    return _build(rng, mix=True, remark=True, as_percent=True, ask_count=True)


def _exam(rng) -> dict:
    topic = rng.choice(TOPICS)
    for _ in attempts(100, "two methods"):
        ma, mb = rng.sample(METHODS, 2)
        if {ma["key"], mb["key"]} != {"phone", "class"}:
            break
    ta, tb = rng.sample((80, 100, 120, 150, 200, 250, 400, 500), 2)
    ra = _report(rng, topic, ta, ma, as_percent=bool(rng.getrandbits(1)), pop=ta * 20)
    rb = _report(rng, topic, tb, mb, as_percent=bool(rng.getrandbits(1)), pop=tb * 20)
    # report A widens its claim to everyone, report B states it for those surveyed
    ra["conclusion"] = f"The headline says: {_strength(ra['pct']).capitalize()} {topic['wide']} {topic['short']}."
    ra["concl_key"] = (
        f"{_strength(ra['pct'])} {topic['wide']} {topic['short']}, a wider group than the {ra['total']} surveyed"
    )
    rb["conclusion"] = f"The headline says: {_of(_strength(rb['pct'])).capitalize()} those surveyed {topic['short']}."
    rb["concl_key"] = f"{_of(_strength(rb['pct']))} the {rb['total']} surveyed {topic['short']}"
    p = Prompt()
    for nm, r in (("Report A", ra), ("Report B", rb)):
        p.t(f"{nm}: " + " ".join((r["data"], r["method"], r["display"], r["conclusion"])) + " ")
    p.t(
        "For each report, identify the data, the display, the method and the conclusion. Then say which of the four parts are different in the two reports. "
    ).t(rng.choice(SHORT_TAILS))
    diffs = []
    if ra["method_key"] != rb["method_key"]:
        diffs.append("the method")
    diffs.append("the data (the totals and shares differ)")
    diffs.append("the conclusion (A is about a wider group than those surveyed; B is about those surveyed)")
    ref = []
    for nm, r in (("Report A", ra), ("Report B", rb)):
        ref += [
            f"{nm} data: {r['data_key']}.",
            f"{nm} display: {r['display_key']}.",
            f"{nm} method: {r['method_key']}.",
            f"{nm} conclusion: {r['concl_key']}.",
        ]
    ref.append("The parts that differ: " + "; ".join(diffs) + ".")
    steps = [
        ("correct_idea", f"Report A: {ra['data_key']}; {ra['method_key']}; {ra['display_key']}"),
        ("correct_idea", f"Report B: {rb['data_key']}; {rb['method_key']}; {rb['display_key']}"),
        ("completeness", f"separates the conclusions: A claims {ra['concl_key']}; B claims {rb['concl_key']}"),
        (
            "use_of_context",
            "refers to the sentences in each report and says which of the four parts differ: " + "; ".join(diffs),
        ),
        ("communication", "keeps the two reports apart and labels each part"),
    ]
    traps = [
        ("method_vs_display", "does it give a graph or chart as a method, or the way people were chosen as a display?"),
        ("data_vs_conclusion", "does it give a headline claim as data, or the counts as the conclusion?"),
        ("same_population", "does it treat the conclusion of Report A as only about the people surveyed?"),
    ]
    return finish(p, " ".join(ref[:4]), short_marking(reference=ref, steps=steps, traps=traps))


FAMILIES = {"easy": [_easy], "medium": [_medium], "hard": [_hard], "exam": [_exam]}


@generator("data_media_critique", level=13, presets=_PRESETS)
def data_media_critique(rng, variant="easy"):
    """`f10-data-media-critique`. Rubric-assessed (`answer: rubric`): every question returned carries a
    per-item marking dict (`Question.answer`), never a plain value."""
    if variant not in FAMILIES:
        raise ValueError(f"unknown variant: {variant}")
    return rng.choice(FAMILIES[variant])(rng)
