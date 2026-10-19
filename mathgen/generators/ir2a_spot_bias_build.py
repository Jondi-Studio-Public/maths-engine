"""Generates `spot-bias` questions: find what is wrong with how a survey was run, then estimate from the right data.

A survey is described in short numbered statements, each one about a different part of the survey: which list the
people were chosen from, how they were picked, the wording of the question, how many replied, and how they were
asked. A statement is either sound or it carries one named source of bias:

  undercoverage       the list leaves out part of the population;
  voluntary response  people chose themselves;
  convenience         only people who were easy to reach were asked;
  leading question    the wording pushes towards one answer;
  non-response        most people chosen did not reply and the report uses only the replies;
  being observed      people answered in front of others.

Ladder (reasoning escalation, not size):
  easy:   four numbered statements; list the numbers of the ones that bias the survey (a list answer).
  medium: five statements, then the first estimate: the percentage who agreed, applied to the whole population.
  hard:   one flaw described in prose with the survey's result; name it from five options in the scenario's own
          words, then estimate the population count from a second survey that was run properly, not from the first.
  exam:   five statements; list the biased ones; the estimate the biased survey would give; the estimate from a
          properly drawn sample; and how much the biased survey overstates the count (follow-through).
"""

from __future__ import annotations

import random

from mathgen.core import attempts, check, generator

from ._ir2a_kit import NAMES, chain, join_and, list_part, mcq_part, number_part, single

_TIERS = ("easy", "medium", "hard")

_CONTEXTS = (
    dict(
        pl="residents",
        sg="resident",
        body="the Shire of Marlow",
        measure="support for a skate park",
        neutral_q="Should the council build a skate park?",
        leading_q="Wouldn't you agree that a skate park is the best way to keep young people out of trouble?",
        partial="the list of residents who have registered an email address with the council, although about a third of residents have not",
        easy_to_reach="the residents who were walking past the library at lunchtime",
        invite="an open invitation printed in the local paper, asking anyone who wished to ring in",
        agree="agree that the skate park should be built",
    ),
    dict(
        pl="students",
        sg="student",
        body="Banksia Secondary College",
        measure="a later start to the school day",
        neutral_q="Would you prefer the school day to start later?",
        leading_q="Don't you agree that a later start would help everyone concentrate and get better marks?",
        partial="the roll of students who catch the school bus, although many students walk or are driven",
        easy_to_reach="the students sitting near the canteen at recess",
        invite="a link on the school app asking any student who wanted to take part to click it",
        agree="would prefer a later start",
    ),
    dict(
        pl="customers",
        sg="customer",
        body="the Gum Tree Cafe",
        measure="longer evening opening hours",
        neutral_q="Would you like the cafe to open later in the evening?",
        leading_q="Our cheerful team works hard for you, so would you like us to stay open later in the evening?",
        partial="the loyalty-card list, which only includes customers who come in at least once a week",
        easy_to_reach="the customers waiting at the counter on Saturday morning",
        invite="a sign at the register asking any customer who felt strongly to fill in a card",
        agree="would like longer opening hours",
    ),
    dict(
        pl="members",
        sg="member",
        body="the Wattle Park Sports Club",
        measure="a rise in yearly fees",
        neutral_q="Would you accept a $20 rise in the yearly fee?",
        leading_q="Given how hard the volunteers work to keep our wonderful grounds running, would you accept a small rise in the yearly fee?",
        partial="the list of members who play in Saturday teams, although many members only use the gym",
        easy_to_reach="the members at the bar after Saturday's final",
        invite="a notice on the clubhouse door asking any member who wanted a say to put a card in the box",
        agree="would accept the fee rise",
    ),
)

# aspect -> (sound statement, biased statement, bias id)
BIAS_TAG = {
    "frame": "ir2a-blames-undercoverage",
    "pick": "ir2a-blames-voluntary-response",
    "easy": "ir2a-blames-easy-to-reach-sample",
    "question": "ir2a-blames-leading-question",
    "reply": "ir2a-blames-non-response",
}
SOUND_TAG = "ir2a-flags-a-sound-feature-as-biased"
MISSED_TAG = "ir2a-misses-a-biased-feature"


def _statements(rng, ctx: dict, N: int, aspects: list[str], biased: set[str]) -> dict[str, str]:
    pl, sg = ctx["pl"], ctx["sg"]
    reply_pct = rng.choice([85, 90, 92])
    low_pct = rng.choice([18, 22, 25, 28, 30, 35])
    pick = rng.choice(["invite", "easy"])
    out: dict[str, str] = {}
    out["frame"] = (
        f"The {pl} were chosen from {ctx['partial']}."
        if "frame" in biased
        else f"The {pl} were chosen from the complete list of all {N:,} {pl}."
    )
    if "pick" in biased:
        out["pick"] = f"The {pl} who were surveyed were those who responded to {ctx['invite']}."
    elif "easy" in biased:
        out["pick"] = f"The {pl} who were surveyed were {ctx['easy_to_reach']}."
    else:
        out["pick"] = f"A computer picked the {pl} who were surveyed at random from the list."
    out["question"] = (
        f"The question asked was: “{ctx['leading_q']}”"
        if "question" in biased
        else f"The question asked was: “{ctx['neutral_q']}”"
    )
    out["reply"] = (
        f"Only {low_pct}% of the {pl} chosen replied, and the results use only those replies."
        if "reply" in biased
        else f"Reminders were sent until {reply_pct}% of the {pl} chosen had replied."
    )
    return out


def _bias_kind(biased: set[str]) -> dict[str, str]:
    return {a: BIAS_TAG[a] for a in biased}


def _numbered(rng, statements: dict[str, str], aspects: list[str]):
    order = list(aspects)
    rng.shuffle(order)
    return order, {a: i + 1 for i, a in enumerate(order)}


def _list_part_for(rng, aspects, order, number_of, biased, marks):
    key = [number_of[a] for a in aspects if a in biased]
    sound = [number_of[a] for a in aspects if a not in biased]
    wrongs = []
    # a sound statement added to the list
    if sound:
        wrongs.append((sorted(key + [rng.choice(sound)]), SOUND_TAG))
    # one biased statement left out
    if len(key) > 1:
        gone = rng.choice(key)
        wrongs.append(([k for k in key if k != gone], MISSED_TAG))
    # the sound statements instead
    if sound and sorted(sound) != sorted(key):
        wrongs.append((sound, SOUND_TAG))
    # one swap
    if sound:
        swap = key[:]
        swap[rng.randrange(len(swap))] = rng.choice(sound)
        wrongs.append((sorted(set(swap)), SOUND_TAG))
    wrongs.append((sorted(set(key) ^ {rng.choice(list(number_of.values()))}), MISSED_TAG))
    return list_part(
        "Write the numbers of the statements that make the survey biased, separated by commas.",
        key,
        wrongs,
        marks=marks,
        working=[f"Statement {number_of[a]} is a source of bias." for a in aspects if a in biased]
        + [f"Statement {number_of[a]} is sound." for a in aspects if a not in biased],
    )


def _bias_set(rng, aspects: list[str], low: int, high: int) -> set[str]:
    k = rng.randint(low, high)
    return set(rng.sample(aspects, k))


def _stem(rng, ctx, name, N, statements, order, number_of, opening=None):
    lines = " ".join(f"({number_of[a]}) {statements[a]}" for a in sorted(order, key=lambda a: number_of[a]))
    head = opening or (
        f"{name} surveyed the {ctx['pl']} of {ctx['body']} on {ctx['measure']}. The survey is described in these statements."
    )
    return f"{head} {lines}"


def _easy(rng):
    ctx = rng.choice(_CONTEXTS)
    name = rng.choice(NAMES)
    N = rng.choice([840, 1200, 1500, 2400, 3600])
    aspects = ["frame", "pick", "question", "reply"]
    biased = _bias_set(rng, aspects, 1, 2)
    st = _statements(rng, ctx, N, aspects, biased)
    # the "easy" aspect never appears here; keep the pick statement sound-or-voluntary only
    order, number_of = _numbered(rng, st, aspects)
    part = _list_part_for(rng, aspects, order, number_of, biased, 2)
    return single(_stem(rng, ctx, name, N, st, order, number_of), part)


def _estimate_numbers(rng):
    p_bad = rng.choice([60, 65, 70, 75, 80, 85])
    p_true = p_bad - rng.choice([15, 20, 25, 30])
    p_true = max(p_true, 20)
    check(p_bad > p_true, "biased result must exceed the true result")
    N = rng.choice([1200, 1500, 2000, 2400, 3000, 4000, 5000, 6000, 8000])
    m = rng.choice([100, 120, 160, 200, 240, 300, 400])
    k = m * p_true // 100
    check(m * p_true % 100 == 0 and N * p_true % 100 == 0 and N * p_bad % 100 == 0, "estimates must be whole")
    return p_bad, p_true, N, m, k


def _five_statements(rng, ctx, N, low, high):
    """(aspects, biased aspects, statements) for a survey described in five statements."""
    aspects = ["frame", "pick", "question", "reply"]
    biased = _bias_set(rng, aspects, low, high)
    st = _statements(rng, ctx, N, aspects, biased)
    asked_bad = rng.random() < 0.5
    st["asked"] = (
        f"Each {ctx['sg']} gave an answer aloud in front of the other {ctx['pl']} waiting."
        if asked_bad
        else f"Each {ctx['sg']} gave an answer in private, on a form that did not ask for a name."
    )
    if asked_bad:
        biased = biased | {"asked"}
    return aspects + ["asked"], biased, st


def _medium(rng):
    ctx = rng.choice(_CONTEXTS)
    name = rng.choice(NAMES)
    p_bad, p_true, N, m, k = _estimate_numbers(rng)
    aspects, biased, st = _five_statements(rng, ctx, N, 1, 2)
    order, number_of = _numbered(rng, st, aspects)
    a = _list_part_for(rng, aspects, order, number_of, biased, 2)
    b = number_part(
        prompt=f"The survey found that {p_bad}% of those surveyed {ctx['agree']}. If that percentage held for all "
        f"{N:,} {ctx['pl']}, estimate the number of them who {ctx['agree']}.",
        exact=N * p_bad // 100,
        places=0,
        marks=2,
        working=[f"{p_bad}% of {N:,} = {N * p_bad // 100:,}."],
        wrong=[
            (p_bad, "ir2a-reports-the-percentage-as-the-count"),
            (N - N * p_bad // 100, "ir2a-gives-those-who-disagree"),
            (N * p_bad // 1000, "ir2a-slips-a-power-of-ten"),
        ],
        sensible_range=(0, N),
    )
    return chain(_stem(rng, ctx, name, N, st, order, number_of), a, b)


_FLAW_SENTENCE = {
    "frame": "The list the {pl} were chosen from leaves out some of the {pl}.",
    "pick": "The {pl} chose themselves, so those with strong views are more likely to take part.",
    "easy": "Only the {pl} who were easy to reach were asked.",
    "question": "The wording of the question pushes the {pl} towards one answer.",
    "reply": "Most of the {pl} chosen did not reply, and they may think differently from those who did.",
}


def _hard(rng):
    ctx = rng.choice(_CONTEXTS)
    name = rng.choice(NAMES)
    p_bad, p_true, N, m, k = _estimate_numbers(rng)
    flaw = rng.choice(["frame", "pick", "easy", "question", "reply"])
    st = _statements(rng, ctx, N, ["frame", "pick", "question", "reply"], {flaw} if flaw != "easy" else {"easy"})
    body = {
        "frame": st["frame"],
        "pick": st["pick"],
        "easy": st["pick"],
        "question": st["question"],
        "reply": st["reply"],
    }[flaw]
    n1 = rng.choice([100, 120, 160, 200, 240, 300])
    k1 = n1 * p_bad // 100
    check(n1 * p_bad % 100 == 0, "first survey count must be whole")
    stem = (
        f"{name} surveyed the {ctx['pl']} of {ctx['body']} on {ctx['measure']}. {body} In that survey {k1} of the {n1} "
        f"{ctx['pl']} surveyed ({p_bad}%) {ctx['agree']}. {name} then took a second survey, drawing {m} of the {N:,} "
        f"{ctx['pl']} by computer from the complete list and asking each in private with a neutral question; "
        f"{k} of them {ctx['agree']}."
    )
    wrongs = [
        (_FLAW_SENTENCE[a].format(pl=ctx["pl"]), BIAS_TAG[a])
        for a in ("frame", "pick", "easy", "question", "reply")
        if a != flaw
    ]
    a = mcq_part(
        "What is the main flaw in the first survey?",
        _FLAW_SENTENCE[flaw].format(pl=ctx["pl"]),
        wrongs,
        marks=2,
        working=[f"{_FLAW_SENTENCE[flaw].format(pl=ctx['pl']).rstrip('.')}, so the first survey is biased."],
        keep=4,
        bare=True,
    )
    b = number_part(
        prompt=f"Use the second survey to estimate the number of the {N:,} {ctx['pl']} who {ctx['agree']}.",
        exact=N * k // m,
        places=0,
        marks=3,
        working=[f"{k}/{m} = {p_true}%.", f"{p_true}% of {N:,} = {N * p_true // 100:,}."],
        wrong=[
            (N * p_bad // 100, "ir2a-uses-the-biased-survey-percentage"),
            (k, "ir2a-reports-the-sample-count-not-the-population-estimate"),
            (N - N * p_true // 100, "ir2a-gives-those-who-disagree"),
        ],
        sensible_range=(0, N),
    )
    return chain(stem, a, b)


@generator("ir2a_spot_bias", level=11, presets={t: dict(tier=t) for t in _TIERS})
def ir2a_spot_bias(rng: random.Random, tier: str = "easy") -> dict:
    check(tier in _TIERS, f"unknown tier {tier!r}")
    return {"easy": _easy, "medium": _medium, "hard": _hard}[tier](rng)


@generator("exd_x_ir2a_spot_bias", level=12, presets={"exam": dict(tier="exam")})
def exd_x_ir2a_spot_bias(rng: random.Random, tier: str = "exam") -> dict:
    """List the biased statements, the biased estimate, the estimate from a sound sample, and the overstatement."""
    check(tier == "exam", "only the exam tier exists")
    ctx = rng.choice(_CONTEXTS)
    name = rng.choice(NAMES)
    p_bad, p_true, N, m, k = _estimate_numbers(rng)
    aspects, biased, st = _five_statements(rng, ctx, N, 2, 3)
    order, number_of = _numbered(rng, st, aspects)
    n1 = rng.choice([100, 120, 160, 200, 240, 300])
    k1 = n1 * p_bad // 100
    stem = (
        _stem(rng, ctx, name, N, st, order, number_of)
        + f" In this survey {k1} of the {n1} {ctx['pl']} surveyed ({p_bad}%) {ctx['agree']}. "
        f"{name} then repeated the survey properly: {m} of the {N:,} {ctx['pl']} were drawn at random from the complete "
        f"list and asked in private with a neutral question, and {k} of them {ctx['agree']}."
    )
    a = _list_part_for(rng, aspects, order, number_of, biased, 2)
    est_bad = N * p_bad // 100
    est_true = N * k // m
    b = number_part(
        prompt=f"Use the first survey's percentage to estimate the number of the {N:,} {ctx['pl']} who {ctx['agree']}.",
        exact=est_bad,
        places=0,
        marks=1,
        working=[f"{p_bad}% of {N:,} = {est_bad:,}."],
        wrong=[
            (k1, "ir2a-reports-the-sample-count-not-the-population-estimate"),
            (p_bad, "ir2a-reports-the-percentage-as-the-count"),
            (N - est_bad, "ir2a-gives-those-who-disagree"),
        ],
        sensible_range=(0, N),
    )
    c = number_part(
        prompt=f"Use the second survey to estimate the number of the {N:,} {ctx['pl']} who {ctx['agree']}.",
        exact=est_true,
        places=0,
        marks=2,
        working=[f"{k}/{m} = {p_true}%.", f"{p_true}% of {N:,} = {est_true:,}."],
        wrong=[
            (k, "ir2a-reports-the-sample-count-not-the-population-estimate"),
            (est_bad, "ir2a-uses-the-biased-survey-percentage"),
            (N - est_true, "ir2a-gives-those-who-disagree"),
        ],
        sensible_range=(0, N),
    )
    d = number_part(
        prompt="By how many is the first survey's estimate higher than the second survey's estimate?",
        exact=est_bad - est_true,
        places=0,
        marks=2,
        working=[f"{est_bad:,} - {est_true:,} = {est_bad - est_true:,}."],
        wrong=[
            (est_bad + est_true, "ir2a-adds-instead-of-subtracting"),
            (p_bad - p_true, "ir2a-compares-percentages-not-counts"),
            (est_true, "ir2a-reports-the-better-estimate"),
        ],
        follow_through="part_b - part_c",
        sensible_range=(0, N),
    )
    return chain(stem, a, b, c, d)
