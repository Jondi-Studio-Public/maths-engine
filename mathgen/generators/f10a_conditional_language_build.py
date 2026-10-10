"""Generates `f10-conditional-probability-language` questions (AC9M10P01, Year 10): read a sentence that uses
"given", "knowing that", "if ... then" or "of" and pick the conditional-event description it states, `answer: mcq`.

Scope: a plain-language sentence follows exactly one of four fixed templates and names exactly two events. The
learner finds which event is the condition and which is the target, then picks "the probability of [target], given
[condition]". Nothing is calculated and no notation is used; interpreting the result in context is outside this skill.

The four templates (the marker word fixes which event is the condition):

    given        "What is the probability that <target>, given that <condition>?"      condition after "given"
    knowing that "Knowing that <condition>, what is the probability that <target>?"    condition after "knowing that"
    if ... then  "If <condition>, then the probability that <target> is p."            condition in the "if" clause
    of           "Of the c <group> who <condition>, k <target>."                        the named group is the condition

Options are always one key and three distractors, each a named misconception:

    key        the probability of <target>, given <condition>
    swapped    the probability of <condition>, given <target>           (swaps-condition-and-target)
    joint      the probability of <target> and <condition> together     (reads-conditional-as-both-events)
    alone      the probability of <target>      (drops-the-condition) or
               the probability of <condition>   (reports-the-condition-as-the-target)

The swapped option and the key are the only two that use the word "given", so a learner who has learnt that "given"
signals a conditional still has to decide the direction; that decision is the skill. Nothing else in the question
can be used to guess it: which event is named first in the sentence, and which has the larger count, are both
randomised independently of which is the condition.

Ladder (reasoning escalation, not size):
  easy:   "given" and "knowing that" sentences, with no numbers.
  medium: "if ... then" and "of" sentences. The marker no longer names the target first, and "of" carries counts.
  hard:   all four templates, each preceded by a counts line that includes how many are in both groups (so a
          joint reading has a tempting number), and "of" sentences say "also" ("12 also play netball"), which invites
          the joint reading.
"""

from __future__ import annotations

import random

from mathgen.core import check, generator, latex_answer, phrase

TIERS = ("easy", "medium", "hard")

# Each context: group noun (singular, plural) and events as (name, third person singular, plural/base).
# The name is a gerund phrase used inside the option; sg/pl complete "the <group> ..." and "<group> who ...".
CONTEXTS = [
    dict(
        sg="student",
        pl="students",
        events=[
            ("swimming", "swims", "swim"),
            ("playing netball", "plays netball", "play netball"),
            ("walking to school", "walks to school", "walk to school"),
            ("owning a bike", "owns a bike", "own a bike"),
            ("learning the piano", "learns the piano", "learn the piano"),
            ("studying French", "studies French", "study French"),
        ],
    ),
    dict(
        sg="shopper",
        pl="shoppers",
        events=[
            ("paying by card", "pays by card", "pay by card"),
            ("buying fruit", "buys fruit", "buy fruit"),
            ("using a loyalty card", "uses a loyalty card", "use a loyalty card"),
            ("shopping online", "shops online", "shop online"),
            ("bringing their own bag", "brings their own bag", "bring their own bag"),
        ],
    ),
    dict(
        sg="resident",
        pl="residents",
        events=[
            ("owning a dog", "owns a dog", "own a dog"),
            ("owning a cat", "owns a cat", "own a cat"),
            ("recycling", "recycles", "recycle"),
            ("having solar panels", "has solar panels", "have solar panels"),
            ("growing vegetables", "grows vegetables", "grow vegetables"),
        ],
    ),
    dict(
        sg="library member",
        pl="library members",
        events=[
            ("borrowing a novel", "borrows a novel", "borrow a novel"),
            ("borrowing a DVD", "borrows a DVD", "borrow a DVD"),
            ("using the study room", "uses the study room", "use the study room"),
            ("attending story time", "attends story time", "attend story time"),
        ],
    ),
]

STEMS = [
    "Which option states the conditional event that this sentence describes?",
    "Which option correctly describes the conditional event in the sentence?",
    "Which of these descriptions matches the sentence?",
]

MISCONCEPTIONS = {
    "swapped": "swaps-condition-and-target",
    "joint": "reads-conditional-as-both-events",
    "target_alone": "drops-the-condition",
    "condition_alone": "reports-the-condition-as-the-target",
}


def _key(target: str, cond: str) -> str:
    return f"the probability of {target}, given {cond}"


def _options(rng: random.Random, target: str, cond: str):
    key = _key(target, cond)
    third_kind = rng.choice(["target_alone", "condition_alone"])
    third_text = {
        "target_alone": f"the probability of {target}",
        "condition_alone": f"the probability of {cond}",
    }[third_kind]
    wrong = [
        (_key(cond, target), MISCONCEPTIONS["swapped"]),
        (f"the probability of {target} and {cond} together", MISCONCEPTIONS["joint"]),
        (third_text, MISCONCEPTIONS[third_kind]),
    ]
    return key, wrong


def _counts(rng: random.Random):
    """(N, c, t, b): total, condition group, target group, both. b is strictly below c and t and c != t, so no
    count can be mistaken for another and the 'both' number is always a real, smaller one."""
    while True:
        b = rng.randint(6, 20)
        c = b + rng.randint(4, 25)
        t = b + rng.randint(4, 25)
        if c != t:
            break
    total = c + t - b + rng.randint(5, 40)
    return total, c, t, b


def _sentence(template: str, ctx: dict, cond, target, c: int, b: int, hard: bool, pct: int) -> str:
    sg, pl = ctx["sg"], ctx["pl"]
    art = "an" if sg[0] in "aeiou" else "a"
    if template == "given":
        return f"What is the probability that {art} randomly chosen {sg} {target[1]}, given that the {sg} {cond[1]}?"
    if template == "knowing":
        return f"Knowing that {art} randomly chosen {sg} {cond[1]}, what is the probability that the {sg} {target[1]}?"
    if template == "ifthen":
        value = f"{b}/{c}" if hard else f"{pct}%"
        return f"If {art} randomly chosen {sg} {cond[1]}, then the probability that the {sg} {target[1]} is {value}."
    if template == "of":
        mid = "also " if hard else ""
        return f"Of the {c} {pl} who {cond[2]}, {b} {mid}{target[2]}."
    raise ValueError(template)


@generator(
    "f10a_conditional_language",
    level=10,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def f10a_conditional_language(rng: random.Random, tier: str = "easy") -> dict:
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    templates = {"easy": ("given", "knowing"), "medium": ("ifthen", "of"), "hard": ("given", "knowing", "ifthen", "of")}
    template = rng.choice(templates[tier])
    ctx = rng.choice(CONTEXTS)
    cond, target = rng.sample(ctx["events"], 2)
    hard = tier == "hard"

    total, c, t, b = _counts(rng)
    pct = rng.choice([10, 15, 20, 25, 30, 35, 40, 45, 55, 60, 65, 70, 75, 80, 85, 90])
    sentence = _sentence(template, ctx, cond, target, c, b, hard, pct)

    key, wrong = _options(rng, target[0], cond[0])
    distractors = [w for w, _m in wrong]
    options = [key, *distractors]
    check(len(set(options)) == 4, "f10-conditional-probability-language: options must be pairwise distinct")
    check(target[0] != cond[0], "f10-conditional-probability-language: two different events required")
    # Independent check: read the condition straight off the finished sentence by its marker, and confirm it is
    # the keyed condition and not the target.
    marker_cond = {
        "given": lambda: sentence.split(", given that ")[1],
        "knowing": lambda: sentence.split("Knowing that ")[1].split(", what")[0],
        "ifthen": lambda: sentence.split("If ")[1].split(", then")[0],
        "of": lambda: sentence.split(" who ")[1].split(",")[0],
    }[template]()
    cond_form = cond[2] if template == "of" else cond[1]
    target_form = target[2] if template == "of" else target[1]
    check(
        marker_cond.rstrip("?").endswith(cond_form),
        "f10-conditional-probability-language: the condition read from the marker is not the keyed condition",
    )
    check(target_form not in marker_cond, "f10-conditional-probability-language: the target appears in the condition")

    parts = []
    if hard:
        parts.append(f"In a survey of {total} {ctx['pl']}, {c} {cond[2]}, {t} {target[2]} and {b} do both.")
    parts.append(f'Read this sentence: "{sentence}"')
    parts.append(phrase(rng, STEMS))
    text = " ".join(parts)

    marker_name = {"given": '"given"', "knowing": '"knowing that"', "ifthen": '"if ... then"', "of": '"of"'}[template]
    where = {
        "given": 'The event after the word "given" is the condition.',
        "knowing": 'The event after "knowing that" is the condition.',
        "ifthen": 'The event in the "if" clause is the condition.',
        "of": 'The group named after "Of the" is the condition: the count that follows is out of that group.',
    }[template]
    working = [
        f"The sentence uses {marker_name}. {where}",
        f"Condition: {cond[0]}. Target: {target[0]}.",
        f"So the description is: {key}.",
    ]
    for wrong_text, m in wrong:
        working.append(f'"{wrong_text}" is not it ({m.replace("-", " ")}).')

    return dict(
        text=text,
        answer=key,
        answer_text=key,
        answer_latex=latex_answer(key, plain=True),
        working=working,
        distractors=distractors,
        option_notes={w: m for w, m in wrong},
    )
