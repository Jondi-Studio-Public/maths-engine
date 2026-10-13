"""Short written-answer generators about questionnaires, errors in surveys, faults in collected data and the way
survey results are reported.

Every question gives a concrete situation (a draft question, a news report, a short table of recorded values) and asks
for a short written answer marked against a rubric. The `answer` of each question is a marking dict (see
`_written_answer_kit`), not a value. Any number in a prompt is computed, never typed in.

Topics (one per skill):

- `wa_questionnaire_design`: spot the design principle a draft question breaks, explain its effect, rewrite it.
- `wa_survey_misrepresentation`: a media report of a survey that misleads (small base, "most", non-response,
  cause from association), up to judging two readers' opposite verdicts.
- `wa_results_misrepresentation`: survey findings generalised beyond the people asked, or misread because the
  question measured something different.
- `wa_survey_error_sources`: sampling error against measurement error, and what a larger sample can and cannot fix.
- `wa_collection_faults`: faults in a short table of recorded values (a unit slip, an impossible value, a duplicate,
  a missing value) and their effect on the mean.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import check
from ._written_answer_kit import (
    DATA_TAILS,
    NAME_SLOTS,
    Case,
    Tier,
    finish,
    marking,
    pct,
    register,
    register_custom,
    ri,
)

PAPERS = ("The Daily Courier", "Metro News", "Channel 5 News", "The Weekend Herald", "Radio 3AB", "The Evening Post")
TOWNS = ("Bellmore", "Corrington", "Easton Bay", "Halcombe", "Northdale", "Ravenswood")

# ---------------------------------------------------------------------------
# wa_questionnaire_design
# ---------------------------------------------------------------------------
_LEAD = (
    ("the new canteen menu", "a big improvement"),
    ("the school's new uniform", "much smarter than the old one"),
    ("the library's longer opening hours", "a great success"),
    ("the new bus timetable", "a clear improvement"),
)
_DOUBLE = (
    ("the price and the quality of the canteen lunches", "satisfied"),
    ("the cleanliness and the opening hours of the pool", "happy"),
    ("the teachers and the facilities at the school", "pleased"),
    ("the speed and the friendliness of the checkout staff", "satisfied"),
)
_VAGUE = (
    (
        "Do you exercise regularly?",
        "regularly",
        "On how many days in a typical week do you exercise for at least 30 minutes?",
    ),
    (
        "Do you spend a lot of time on your phone?",
        "a lot of time",
        "About how many hours do you spend on your phone on a typical school day?",
    ),
    ("Do you eat healthy food?", "healthy", "How many serves of vegetables did you eat yesterday?"),
    ("Do you live close to school?", "close", "About how many minutes does it take you to get to school?"),
)
_JARGON = (
    (
        "To what extent do you consider the provision of recreational infrastructure in your locality adequate?",
        "Are there enough places near you to play sport and exercise?",
    ),
    (
        "Do you perceive the remuneration structure at your workplace as equitable?",
        "Do you think people at your workplace are paid fairly?",
    ),
    (
        "How would you characterise the efficacy of the municipal refuse collection regime?",
        "How well does the council collect your rubbish?",
    ),
    ("Is your utilisation of public transportation infrequent?", "How often do you catch a bus or train?"),
)
_OVERLAP = (
    (
        "How many hours do you sleep on a school night?",
        "0-4 hours, 4-8 hours, 8-12 hours",
        "less than 4 hours, 4 hours to less than 8 hours, 8 hours to less than 12 hours, 12 hours or more",
        "the options 4 and 8 hours each appear in two choices, so a person who sleeps 4 or 8 hours does not know which to tick, and anyone who sleeps more than 12 hours has no choice",
    ),
    (
        "How old are you?",
        "10-20, 20-30, 30-40",
        "under 20, 20 to 29, 30 to 39, 40 or over",
        "20 and 30 each appear in two choices, and anyone aged 40 or over, or under 10, has no choice",
    ),
    (
        "How much do you spend at the canteen each week?",
        "$0-$5, $5-$10, $10-$15",
        "less than $5, $5 to $9.99, $10 to $14.99, $15 or more",
        "$5 and $10 each appear in two choices, and anyone spending $15 or more has no choice",
    ),
    (
        "How many books did you read last year?",
        "1-5, 5-10, 10-15",
        "none, 1 to 4, 5 to 9, 10 or more",
        "5 and 10 each appear in two choices, and a person who read none has no choice",
    ),
)
_PRIVATE = (
    (
        "What is your exact household income and your home address?",
        "ask only for an income range, or not at all if it is not needed, and do not ask for an address; make answers anonymous and optional",
    ),
    (
        "Write your full name and describe any health problems you have.",
        "do not ask for names or health details unless they are needed; make the survey anonymous and say how answers will be used",
    ),
    (
        "Please give your phone number and tell us who you voted for.",
        "do not ask for a phone number, make voting questions optional and anonymous, and say how answers will be kept private",
    ),
    (
        "State your age, your address and how much money you owe.",
        "ask for an age group only, leave out the address, make questions about debts optional and anonymous",
    ),
)
_LOADED = (
    (
        "Do you support wasting thousands of dollars on a new skate park?",
        "Do you support, oppose or have no opinion on spending council money on a new skate park?",
    ),
    (
        "Do you agree that the ridiculous new homework rules should be scrapped?",
        "How do you feel about the new homework rules: strongly support, support, neutral, oppose, strongly oppose?",
    ),
    (
        "Would you back the sensible plan to ban cars from the main street?",
        "Would you support, oppose or have no opinion on banning cars from the main street?",
    ),
    (
        "Are you tired of the council's useless bin collection?",
        "How satisfied are you with the council's bin collection: very satisfied, satisfied, neutral, dissatisfied, very dissatisfied?",
    ),
)

QUESTION_CASES = [
    Case(
        "leading",
        slots={**NAME_SLOTS, "pair": _LEAD, "thing": lambda r, v: v["pair"][0], "praise": lambda r, v: v["pair"][1]},
        fields=dict(
            question="Don't you agree that {thing} is {praise}?",
            principle="freedom from bias (it is a leading question)",
            short="a leading question",
            why="the wording tells the person what answer is expected, so more of them will agree than really think so",
            effect="the results will show more agreement than the true level",
            better="How would you rate {thing}: much worse, worse, the same, better or much better?",
            better_req="uses neutral wording and offers balanced choices",
        ),
        wrong=[
            (
                "ambiguity_instead",
                "does it say the problem is that the question is unclear, when the real problem is that it pushes people towards one answer?",
            )
        ],
    ),
    Case(
        "double",
        slots={**NAME_SLOTS, "pair": _DOUBLE, "things": lambda r, v: v["pair"][0], "adj": lambda r, v: v["pair"][1]},
        fields=dict(
            question="Are you {adj} with {things}?",
            principle="unambiguous questions (it asks about two things at once)",
            short="a double-barrelled question",
            why="a person who is happy with one of the two things and not the other cannot give a true answer, and the answer cannot be matched to either part",
            effect="the answers cannot be interpreted, because we do not know which part each person was answering",
            better="ask two separate questions, one about each thing",
            better_req="splits the question so that each question asks about one thing only",
        ),
        wrong=[("too_many_choices", "does it say the problem is the number of answer choices?")],
    ),
    Case(
        "vague",
        slots={
            **NAME_SLOTS,
            "triple": _VAGUE,
            "question": lambda r, v: v["triple"][0],
            "word": lambda r, v: v["triple"][1],
            "bq": lambda r, v: v["triple"][2],
        },
        fields=dict(
            question="{question}",
            principle="unambiguous questions (the word '{word}' means different things to different people)",
            short="an unclear, ambiguous question",
            why="'{word}' can mean different things to different people, so two people with the same habits could answer differently",
            effect="the answers are not comparable, because people are not answering the same question",
            better="{bq}",
            better_req="replaces the vague word with something that can be counted or measured",
        ),
        wrong=[
            (
                "leading_instead",
                "does it say the question pushes people to a particular answer, when the real problem is that the key word is vague?",
            )
        ],
    ),
    Case(
        "jargon",
        slots={**NAME_SLOTS, "pair": _JARGON, "question": lambda r, v: v["pair"][0], "bq": lambda r, v: v["pair"][1]},
        fields=dict(
            question="{question}",
            principle="simple language (the wording is too complicated)",
            short="a question that uses complicated language",
            why="many people will not understand the long, formal words, so they may guess, skip the question or answer something different",
            effect="answers from people who did not understand the question are unreliable, and some people will not answer at all",
            better="{bq}",
            better_req="uses short, everyday words that every respondent understands",
        ),
        wrong=[("privacy_instead", "does it call this a privacy problem?")],
    ),
    Case(
        "overlap",
        slots={
            **NAME_SLOTS,
            "quad": _OVERLAP,
            "question": lambda r, v: v["quad"][0],
            "opts": lambda r, v: v["quad"][1],
            "bopts": lambda r, v: v["quad"][2],
            "fault": lambda r, v: v["quad"][3],
        },
        fields=dict(
            question="{question} Choices: {opts}.",
            principle="a sensible number and choice of answer options (the choices overlap and do not cover every answer)",
            short="answer choices that overlap and leave gaps",
            why="{fault}",
            effect="people on the boundaries answer inconsistently, so the counts in each choice are unreliable",
            better="use the choices: {bopts}",
            better_req="gives choices that do not overlap and cover every possible answer",
        ),
        wrong=[("leading_instead", "does it call this a leading question?")],
    ),
    Case(
        "privacy",
        slots={
            **NAME_SLOTS,
            "pair": _PRIVATE,
            "question": lambda r, v: v["pair"][0],
            "fixline": lambda r, v: v["pair"][1],
        },
        fields=dict(
            question="{question}",
            principle="privacy and ethics (it asks for personal information that is not needed)",
            short="a question that invades privacy",
            why="people have a right to keep private details to themselves, and many will refuse to answer or will not answer truthfully",
            effect="many people will skip the question or give untrue answers, so the results are unreliable and the survey may be unethical",
            better="{fixline}",
            better_req="removes or limits the personal details and makes answers anonymous or optional",
        ),
        wrong=[("language_instead", "does it say the only problem is complicated wording?")],
    ),
    Case(
        "loaded",
        slots={**NAME_SLOTS, "pair": _LOADED, "question": lambda r, v: v["pair"][0], "bq": lambda r, v: v["pair"][1]},
        fields=dict(
            question="{question}",
            principle="freedom from bias (the question uses emotive, loaded words)",
            short="a loaded question",
            why="words such as 'wasting', 'ridiculous', 'sensible' or 'useless' push people towards agreeing with the writer",
            effect="the results will exaggerate support for the writer's view",
            better="{bq}",
            better_req="removes the emotive words and gives balanced answer choices",
        ),
        wrong=[("jargon_instead", "does it say the problem is that the words are too difficult?")],
    ),
]

QUESTION_TIERS = {
    "easy": Tier(
        prompt='A survey draft contains this question: "{question}" Which principle of good questionnaire design does it break? Explain why.',
        reference=["It breaks the principle of {principle}.", "Why: {why}."],
        points=[
            ("correct_idea", "identifies the principle broken: {principle}"),
            ("use_of_context", "refers to the actual wording of the question when explaining why it is a problem"),
            ("completeness", "gives a reason, not only the name of the principle: {why}"),
            ("communication", "uses terms such as bias, ambiguous or privacy correctly"),
        ],
        wrong=[("names_principle_only", "does it name a principle but not explain what is wrong with this question?")],
        answer="{principle}",
    ),
    "medium": Tier(
        prompt='A survey draft contains this question: "{question}" Explain what is wrong with it and how it could affect the results of the survey.',
        reference=["The problem: {short}; {why}.", "Effect on the results: {effect}."],
        points=[
            ("correct_idea", "identifies the problem: {short}"),
            ("use_of_context", "explains it using the wording of this question: {why}"),
            ("completeness", "states the effect on the results: {effect}"),
            ("communication", "separates the problem from its effect"),
        ],
        wrong=[
            (
                "problem_without_effect",
                "does it describe the fault but not say how the survey results would be affected?",
            )
        ],
        answer="{short}; {effect}",
    ),
    "hard": Tier(
        prompt='A survey draft contains this question: "{question}" Identify the problem, rewrite the question (or the part that is wrong) so that it is fixed, and explain why your version is better.',
        reference=[
            "The problem: {short}.",
            "Improved version: {better}.",
            "It is better because it {better_req}, so it avoids the problem that {why}.",
        ],
        points=[
            ("correct_idea", "identifies the problem: {short}"),
            ("correct_idea", "gives a rewrite that really removes it (it {better_req})"),
            ("use_of_context", "keeps the same topic as the original question"),
            ("completeness", "explains why the new version is better, not only what it is"),
            ("communication", "presents the new wording clearly"),
        ],
        wrong=[
            (
                "rewrite_keeps_the_fault",
                "does the rewritten question still contain the same problem (a lead, a vague word, overlapping choices)?",
            ),
            ("changes_the_topic", "does the rewrite ask about a different subject?"),
        ],
        answer="{better}",
    ),
    "exam": Tier(
        two=True,
        pair_ok=lambda a, b: a["short"] != b["short"],
        prompt=(
            "A draft questionnaire has two questions.\n"
            'Question 1: "{a_question}"\n'
            'Question 2: "{b_question}"\n'
            "Identify the problem with each question. Which problem is more likely to make the results of the survey misleading, rather than just harder to answer? Justify your choice, and rewrite that question."
        ),
        reference=[
            "Question 1: {a_short}. Question 2: {b_short}.",
            "A problem that pushes answers one way (bias) or makes answers uninterpretable misleads the results most; one that only confuses or annoys some people mainly lowers the response.",
            "Question 1 effect: {a_effect}. Question 2 effect: {b_effect}.",
            'A rewrite of the question judged more damaging, for example Question 1 becomes "{a_better}" and Question 2 becomes "{b_better}".',
        ],
        points=[
            ("correct_idea", "identifies the problem with Question 1: {a_short}"),
            ("correct_idea", "identifies the problem with Question 2: {b_short}"),
            (
                "use_of_context",
                "compares the effects on the results (Question 1: {a_effect}; Question 2: {b_effect}) and chooses one",
            ),
            ("completeness", "rewrites the chosen question so the problem is removed"),
            ("communication", "justifies the choice instead of asserting it"),
        ],
        wrong=[
            (
                "treats_both_as_equal",
                "does it treat the two problems as equally damaging without comparing their effects?",
            )
        ],
        answer="Question 1: {a_short}. Question 2: {b_short}.",
    ),
}
register(
    "wa_questionnaire_design",
    12,
    QUESTION_CASES,
    QUESTION_TIERS,
    doc="faults in draft survey questions.",
)

# ---------------------------------------------------------------------------
# wa_survey_misrepresentation: media reports of surveys
# ---------------------------------------------------------------------------
_PROS = (
    ("dentists", "Sparkle toothpaste"),
    ("vets", "PawPro dog food"),
    ("teachers", "Zing pens"),
    ("mechanics", "TorqueMax engine oil"),
)
_CAUSES = ("the new ferry service", "a four-day school week", "the council's recycling scheme", "longer library hours")
_OPTS = (
    ("teenagers", "chose pizza as their favourite meal", "pizza"),
    ("students", "chose the bus as their usual way to school", "the bus"),
    ("members", "chose Saturday as their preferred training day", "Saturday"),
)


def _small_n(r, v):
    return r.choice([10, 12, 15, 20])


def _pct_change_n(r, v):
    return r.choice([25, 50, 75, 100])


MEDIA_CASES = [
    Case(
        "tiny_sample",
        slots={
            "paper": PAPERS,
            "pair": _PROS,
            "pros": lambda r, v: v["pair"][0],
            "product": lambda r, v: v["pair"][1],
            "n": _small_n,
            "k": lambda r, v: v["n"] - 1,
            "p": lambda r, v: pct(v["k"], v["n"]),
            "one": lambda r, v: pct(1, v["n"]),
            **NAME_SLOTS,
        },
        fields=dict(
            setup='{paper} reports: "{k} out of {n} {pros} recommend {product}." The survey was paid for by the makers of {product} and asked {n} {pros}; {k} of them, or {p}, said yes.',
            flaw="the sample of only {n} {pros} is far too small to speak for all {pros}, and the survey was paid for by the makers of the product",
            figures="only {n} {pros} were asked, so one person changing their answer moves the percentage by {one}",
            better_headline="In a small survey of {n} {pros} paid for by its maker, {k} said they would recommend {product}",
            needs="how the {n} {pros} were chosen, how the question was worded and who paid for the survey",
            claim_good="the report is misleading because {n} {pros} is too few, and the maker paid for the survey, so the result cannot be trusted for all {pros}",
            claim_bad="{k} out of {n} is {p}, which is nearly everyone, so the result must be reliable",
            bad_flaw="a high percentage from a tiny, interested-party sample is not reliable; the percentage says nothing about how the {n} were chosen",
        ),
        wrong=[
            (
                "percentage_is_enough",
                "does it accept the high percentage ({p}) as proof, without questioning the sample size or who paid?",
            )
        ],
    ),
    Case(
        "small_base",
        slots={
            "paper": PAPERS,
            "cause": _CAUSES,
            "n": _pct_change_n,
            "k1": lambda r, v: v["n"] * 4 // 100,
            "k2": lambda r, v: v["n"] * 8 // 100,
            "diff": lambda r, v: v["k2"] - v["k1"],
            **NAME_SLOTS,
        },
        fields=dict(
            setup='{paper} runs the headline "Support for {cause} doubles!" The report says 4% of the {n} people surveyed last year supported it and 8% of the {n} people surveyed this year did, which is {k1} people and {k2} people.',
            flaw="'doubles' hides that the actual increase is only {diff} people out of {n}; a change from {k1} to {k2} people is within what chance variation between two samples could produce",
            figures="the change is from {k1} people to {k2} people, an increase of {diff}",
            better_headline="Support for {cause} rose from {k1} to {k2} people in surveys of {n} people each year, a change that may be due to chance",
            needs="how the people were chosen each year and whether the change is more than chance variation",
            claim_good="the headline is misleading: the percentages doubled, but only {diff} more people said yes, which could easily be chance",
            claim_bad="4% to 8% is exactly twice as much, so the headline is accurate and support really has doubled",
            bad_flaw="the doubling is arithmetically right, but it is a change in counts of only {diff} people, too small to show real change",
        ),
        wrong=[
            (
                "trusts_the_doubling",
                "does it accept that support has doubled without looking at the small counts of {k1} and {k2}?",
            )
        ],
    ),
    Case(
        "most_claim",
        slots={
            "paper": PAPERS,
            "trip": _OPTS,
            "group": lambda r, v: v["trip"][0],
            "action": lambda r, v: v["trip"][1],
            "choice": lambda r, v: v["trip"][2],
            "n": [200, 250, 300, 400, 500],
            "k": lambda r, v: r.choice([x for x in range(v["n"] * 35 // 100, v["n"] * 49 // 100 + 1) if x % 5 == 0]),
            "p": lambda r, v: pct(v["k"], v["n"]),
            **NAME_SLOTS,
        },
        fields=dict(
            setup='{paper} runs the headline "Most {group} choose {choice}." In the survey, {k} of the {n} {group} ({p}) {action}, and the rest chose one of three other options.',
            flaw="'most' means more than half, but only {p} chose {choice}; it was the largest group, not a majority",
            figures="{k} of {n} is {p}, which is less than half",
            better_headline="{choice} was the most popular choice among {group}, chosen by {p} of those surveyed",
            needs="the share who chose each of the other options",
            claim_good="the headline overstates the result: {p} is less than half, so 'most' is wrong, although {choice} was the most popular single choice",
            claim_bad="{choice} got more votes than any other option, so most {group} must choose it",
            bad_flaw="getting more votes than any other single option is not the same as getting more than half of all votes",
        ),
        wrong=[("accepts_most", "does it accept 'most' even though {p} is less than half?")],
    ),
    Case(
        "non_response",
        slots={
            "paper": PAPERS,
            "town": TOWNS,
            "N": [1000, 2000, 2500, 4000],
            "back": lambda r, v: r.choice([15, 20, 25, 30]),
            "got": lambda r, v: v["N"] * v["back"] // 100,
            "yes": lambda r, v: v["got"] * r.choice([60, 65, 70, 75]) // 100,
            "py": lambda r, v: pct(v["yes"], v["got"]),
            "rate": lambda r, v: pct(v["got"], v["N"]),
            "nonreply": lambda r, v: v["N"] - v["got"],
            **NAME_SLOTS,
        },
        fields=dict(
            setup='{paper} reports: "{py} of {town} residents support the new toll road." The survey was posted to {N} residents. {got} of the {N} ({rate}) replied, and {yes} of those {got} ({py}) supported the toll road.',
            flaw="only {rate} of the people surveyed replied, so the {py} describes only those who chose to reply, not all {town} residents",
            figures="{yes} of the {got} who replied ({py}) supported it, but {got} of the {N} surveyed is only {rate}",
            better_headline="Of the {rate} of {town} residents who replied to a postal survey, {py} supported the toll road",
            needs="why the other {nonreply} residents did not reply and whether they differ from those who did",
            claim_good="the report is misleading because most of those surveyed did not reply, so {py} cannot be assumed to hold for all {town} residents",
            claim_bad="{py} is a clear majority, and the survey was sent to {N} residents, so it must be right for the town",
            bad_flaw="the survey was sent to {N} residents, but only {got} answered; a large sample sent does not mean a large, representative sample returned",
        ),
        wrong=[("sent_equals_sampled", "does it treat all {N} people the survey was sent to as having answered?")],
    ),
    Case(
        "causation",
        slots={
            "paper": PAPERS,
            "a": [60, 80, 100, 120],
            "b": [40, 50, 60],
            "x": ri(68, 80),
            "y": lambda r, v: v["x"] - r.randint(5, 12),
            "d": lambda r, v: v["x"] - v["y"],
            **NAME_SLOTS,
        },
        fields=dict(
            setup='{paper} runs the headline "Eating breakfast lifts test marks, study shows." In a survey of students, the {a} students who ate breakfast every day had a mean test mark of {x}, and the {b} students who did not had a mean mark of {y}, a difference of {d} marks.',
            flaw="the survey only shows an association between breakfast and marks; it does not show that breakfast caused the difference, since other factors such as sleep or home life could explain both",
            figures="the mean marks are {x} and {y}, a difference of {d}, but the students chose whether to eat breakfast",
            better_headline="Students who ate breakfast had a mean mark {d} higher in a survey, but the survey cannot show that breakfast caused this",
            needs="whether other differences between the two groups, such as sleep or study time, could explain the gap",
            claim_good="the headline claims a cause that the survey does not show: students who eat breakfast may differ in other ways that affect marks",
            claim_bad="the breakfast group scored {d} marks higher on average, so breakfast must be causing the higher marks",
            bad_flaw="a difference between groups that chose their own habits shows an association only; a cause would need other explanations to be ruled out",
        ),
        wrong=[("accepts_cause", "does it accept that eating breakfast causes the higher marks?")],
    ),
    Case(
        "sponsor",
        slots={
            "paper": PAPERS,
            "company": ["Swiftnet", "BrightPower", "GreenLeaf Energy", "Cloudline"],
            "n": [200, 300, 400, 500],
            "k": lambda r, v: v["n"] * r.choice([80, 85, 90]) // 100,
            "p": lambda r, v: pct(v["k"], v["n"]),
            **NAME_SLOTS,
        },
        fields=dict(
            setup='{paper} reports: "{p} of Australians are happy with their internet provider." The survey was run by {company} among {n} of its own customers; {k} of them said they were happy.',
            flaw="only {company}'s own customers were asked, so the {p} describes those customers, not all Australians, and the company has an interest in a good result",
            figures="{k} of {n} is {p}, but all {n} were customers of {company}",
            better_headline="{p} of {n} {company} customers surveyed by the company said they were happy with its service",
            needs="how the customers were chosen and whether customers of other providers were asked",
            claim_good="the report is misleading because it generalises from one company's customers, surveyed by that company, to all Australians",
            claim_bad="{k} out of {n} is a large number of happy people, so most Australians must be happy with their provider",
            bad_flaw="the number of people asked does not matter if they are all customers of the company that ran the survey",
        ),
        wrong=[
            ("big_sample_hides_bias", "does it say the sample of {n} is large enough to trust, ignoring who was asked?")
        ],
    ),
]


def _derive_media(v: dict) -> dict:
    return {}


MEDIA_TIERS = {
    "easy": Tier(
        prompt="{setup} Describe what is misleading about the way the survey result is reported.",
        reference=["What is misleading: {flaw}.", "Evidence from the report: {figures}."],
        points=[
            ("correct_idea", "identifies the real misrepresentation: {flaw}"),
            ("use_of_context", "uses the figures in the report: {figures}"),
            ("completeness", "explains why the report misleads, not just that it does"),
            ("communication", "refers to the headline and to the survey separately"),
        ],
        wrong=[
            (
                "vague_criticism",
                "does it only say the survey is 'not good' or 'biased' without saying how it is reported misleadingly?",
            )
        ],
        answer="{flaw}",
    ),
    "medium": Tier(
        prompt="{setup} Using the figures in the report, explain why the claim is not justified by the survey.",
        reference=["The figures: {figures}.", "So the claim is not justified because {flaw}."],
        points=[
            ("correct_idea", "identifies why the claim is unjustified: {flaw}"),
            ("use_of_context", "uses the actual figures: {figures}"),
            ("completeness", "links the figures to the weakness in the claim"),
            ("communication", "reasons from the figures to the conclusion"),
        ],
        wrong=[("figures_ignored", "does it criticise the report without quoting any figure from it?")],
        answer="{flaw}",
    ),
    "hard": Tier(
        prompt="{setup} Write a more accurate version of the headline that the survey results do support, and say what extra information a reader would need to judge how far the survey can be trusted.",
        reference=[
            'A fairer headline: "{better_headline}".',
            "A reader would need to know {needs}.",
            "The original misleads because {flaw}.",
        ],
        points=[
            ("correct_idea", "writes a headline that the data really support: {better_headline}"),
            ("use_of_context", "keeps the figures from the report accurate"),
            ("completeness", "names useful extra information: {needs}"),
            ("communication", "the new headline is short, clear and not just a copy of the original"),
        ],
        wrong=[
            ("headline_still_overclaims", "does the new headline still claim more than the figures allow?"),
            ("no_extra_information", "does it give a headline but no information a reader would need?"),
        ],
        answer="{better_headline}",
    ),
    "exam": Tier(
        prompt='{setup} Two readers react to the report. {person} says: "{claim_bad}." {other} says: "{claim_good}." Evaluate both readers\' claims and say what the survey does and does not show.',
        reference=[
            "{other} is right: {claim_good}.",
            "{person} is wrong: {bad_flaw}.",
            "The survey shows: {figures}; it does not show what the headline claims.",
        ],
        points=[
            ("correct_idea", "judges {other} to be right: {claim_good}"),
            ("correct_idea", "judges {person} to be wrong: {bad_flaw}"),
            ("use_of_context", "uses the figures: {figures}"),
            ("completeness", "says what the survey does show as well as what it does not"),
            ("communication", "treats each reader's claim separately"),
        ],
        wrong=[
            ("splits_the_difference", "does it say both readers are partly right without deciding which is correct?"),
            ("sides_with_person", "does it side with {person}?"),
        ],
        answer="{other} is right and {person} is wrong",
    ),
}
register(
    "wa_survey_misrepresentation",
    12,
    MEDIA_CASES,
    MEDIA_TIERS,
    tails=DATA_TAILS,
    doc="misleading reports of surveys.",
)

# ---------------------------------------------------------------------------
# wa_results_misrepresentation: generalising beyond the people asked
# ---------------------------------------------------------------------------
GEN_CASES = [
    Case(
        "club_to_town",
        slots={
            "paper": PAPERS,
            "town": TOWNS,
            "club": ["the Bellmore Swimming Club", "the Parkside Aquatic Club", "the Town Pool Supporters' Group"],
            "n": [60, 80, 100, 120],
            "k": lambda r, v: v["n"] * r.choice([60, 65, 70, 75]) // 100,
            "p": lambda r, v: pct(v["k"], v["n"]),
            "N": [18000, 24000, 30000],
            **NAME_SLOTS,
        },
        fields=dict(
            setup='A survey of {n} members of {club} found that {k} ({p}) want the {town} pool to open longer. {paper} reports: "Most {town} residents want longer pool hours." {town} has {N} residents.',
            measured="the views of {n} members of {club}",
            claimed="the views of all {N} residents of {town}",
            gap="swimming-club members use the pool far more than most residents and are likely to want longer hours, so they are not typical of the town",
            fair="Most of the {n} members of {club} who were surveyed want longer pool hours.",
            need="a sample of {town} residents chosen at random from all {N} residents",
            claim_good="the survey asked club members only, so it cannot show what residents in general think",
            claim_bad="{p} is a big majority, so the town must want longer hours too",
            bad_flaw="a big majority within the wrong group says nothing about the whole town",
        ),
        wrong=[("generalises_from_members", "does it accept the result for all {N} residents of {town}?")],
    ),
    Case(
        "time_and_place",
        slots={
            "paper": PAPERS,
            "place": ["outside a city office tower", "at the main railway station", "outside a city bank"],
            "when": ["between 8 am and 9 am on a weekday", "from 7 am to 9 am on Tuesday and Wednesday"],
            "n": [100, 150, 200],
            "k": lambda r, v: v["n"] * r.choice([55, 60, 65, 70]) // 100,
            "p": lambda r, v: pct(v["k"], v["n"]),
            **NAME_SLOTS,
        },
        fields=dict(
            setup='Of {n} people stopped {place} {when}, {k} ({p}) said they travel to work by train. {paper} reports: "Most people in the state travel to work by train."',
            measured="the travel habits of {n} people who were {place} {when}",
            claimed="the travel habits of everyone in the state",
            gap="people {place} on a weekday morning are mostly city workers who are more likely to use trains than people who work from home, work locally or work at other times",
            fair="Most of the {n} people stopped {place} {when} said they travel to work by train.",
            need="a sample drawn from workers across the whole state at different places and times",
            claim_good="the survey covered one place and one time, so it cannot speak for the whole state",
            claim_bad="{n} people is plenty, and {p} said train, so the claim for the whole state is fine",
            bad_flaw="the number surveyed is not the problem; the problem is that everyone was asked in the same place at the same time",
        ),
        wrong=[
            (
                "sample_size_only",
                "does it say the only issue is that {n} is a small number, instead of where and when they were asked?",
            )
        ],
    ),
    Case(
        "interest_not_action",
        slots={
            "paper": PAPERS,
            "item": ["an electric bike", "a smart watch", "solar panels for their house", "a robot vacuum cleaner"],
            "n": [200, 300, 400, 500],
            "k": lambda r, v: v["n"] * r.choice([30, 35, 40, 45]) // 100,
            "p": lambda r, v: pct(v["k"], v["n"]),
            **NAME_SLOTS,
        },
        fields=dict(
            setup='{n} shoppers were asked, "Would you be interested in buying {item}?" {k} of them ({p}) said yes. {paper} reports: "{p} of shoppers will buy {item} this year."',
            measured="how many shoppers said they were interested in {item}",
            claimed="how many shoppers will actually buy {item} this year",
            gap="saying you are interested is not the same as buying; many interested people will not buy, and the question said nothing about this year",
            fair="{p} of the {n} shoppers asked said they would be interested in buying {item}.",
            need="data on what interested shoppers actually bought, or a question about firm plans to buy this year",
            claim_good="the report confuses interest with buying: the question asked about interest only",
            claim_bad="{k} people said yes, so {k} of them will buy {item}",
            bad_flaw="a yes to 'interested' is not a promise to buy, and the survey did not ask about buying",
        ),
        wrong=[("interest_equals_purchase", "does it treat saying 'interested' as the same as buying?")],
    ),
    Case(
        "ever_vs_regular",
        slots={
            "paper": PAPERS,
            "activity": [
                "played a team sport",
                "visited the library",
                "used a skate park",
                "volunteered in the community",
            ],
            "n": [200, 250, 300, 400],
            "k": lambda r, v: v["n"] * r.choice([55, 60, 65, 70]) // 100,
            "p": lambda r, v: pct(v["k"], v["n"]),
            **NAME_SLOTS,
        },
        fields=dict(
            setup='{n} students were asked, "Have you ever {activity}?" {k} of them ({p}) said yes. {paper} reports: "{p} of students {activity} regularly."',
            measured="whether each student had ever {activity}, at any time",
            claimed="how many students {activity} regularly",
            gap="'ever' includes a single occasion years ago, so many of those who said yes may do it rarely or never now",
            fair="{p} of the {n} students asked had ever {activity}.",
            need="a question about how often students do it, such as the number of times in the past month",
            claim_good="the question asked 'ever', but the report says 'regularly', so it describes something the survey did not measure",
            claim_bad="{p} said yes, so about {p} of students must do it regularly",
            bad_flaw="'ever' and 'regularly' are different, so a count of 'ever' cannot be treated as a count of 'regularly'",
        ),
        wrong=[("ever_equals_regularly", "does it treat 'ever' as meaning 'regularly'?")],
    ),
    Case(
        "age_mix",
        slots={
            "paper": PAPERS,
            "town": TOWNS,
            "n": [400, 500, 600],
            "k": lambda r, v: v["n"] * r.choice([45, 50, 55, 60]) // 100,
            "pk": lambda r, v: pct(v["k"], v["n"]),
            "b": [20, 25, 30],
            "k2": lambda r, v: v["n"] * r.choice([55, 60, 65, 70]) // 100,
            "p2": lambda r, v: pct(v["k2"], v["n"]),
            **NAME_SLOTS,
        },
        fields=dict(
            setup="An online survey got {n} replies about a proposed skate park in {town}. {k2} of the {n} respondents ({p2}) were under 25, though only about {b}% of {town} residents are under 25. {paper} reports the survey as showing what {town} residents think.",
            measured="the views of {n} online respondents, most of whom ({p2}) were under 25",
            claimed="the views of all {town} residents",
            gap="young people are over-represented ({p2} of respondents against about {b}% of residents) and are the most likely to want a skate park",
            fair="Among {n} online respondents, most of whom were under 25, views on the skate park were as reported.",
            need="a sample that matches the town's age mix, or results reported separately for each age group",
            claim_good="the sample has far more young people than the town does, so its result cannot stand for all residents",
            claim_bad="{n} replies is a big sample, so it represents the town",
            bad_flaw="a large sample can still be unrepresentative if some groups are over-represented",
        ),
        wrong=[("big_means_representative", "does it assume a sample of {n} must be representative?")],
    ),
]
GEN_TIERS = {
    "easy": Tier(
        prompt="{setup} Explain why the report's conclusion goes beyond what the survey can show.",
        reference=["The survey measured {measured}.", "The report claims {claimed}.", "The gap: {gap}."],
        points=[
            ("correct_idea", "says what the survey measured: {measured}"),
            ("correct_idea", "says what the report claims instead: {claimed}"),
            ("use_of_context", "explains why they differ: {gap}"),
            ("completeness", "concludes that the report goes beyond what the survey can show"),
            ("communication", "contrasts the measured group with the claimed group"),
        ],
        wrong=[
            ("asks_for_more_people_only", "does it say the only trouble is that more people should have been asked?")
        ],
        answer="the survey measured {measured}, not {claimed}",
    ),
    "medium": Tier(
        prompt="{setup} Identify the misunderstanding in the way the result is reported, and explain how it makes the finding look more reliable than it is.",
        reference=[
            "Misunderstanding: the report treats {measured} as if it were {claimed}.",
            "Why this is wrong: {gap}.",
        ],
        points=[
            ("correct_idea", "identifies the misunderstanding: {measured} is not {claimed}"),
            ("use_of_context", "explains using the situation: {gap}"),
            ("completeness", "says how the finding is made to look more reliable than it is"),
            ("communication", "states the misunderstanding in a single clear sentence"),
        ],
        wrong=[("restates_the_report", "does it repeat the report's claim without explaining the misunderstanding?")],
        answer="{gap}",
    ),
    "hard": Tier(
        prompt="{setup} Rewrite the report's claim so that it states only what the survey supports, and describe what would be needed before the original claim could be made.",
        reference=[
            'A supported statement: "{fair}"',
            "To make the original claim, {need} would be needed.",
            "The original goes too far because {gap}.",
        ],
        points=[
            ("correct_idea", "writes a statement the survey supports: {fair}"),
            ("use_of_context", "keeps the numbers and the group asked accurate"),
            ("completeness", "says what would be needed for the broader claim: {need}"),
            ("communication", "the rewrite is brief and clear"),
        ],
        wrong=[
            ("rewrite_still_overclaims", "does the rewritten claim still speak for people who were not surveyed?"),
            ("no_requirement_given", "does it fail to say what would be needed for the broader claim?"),
        ],
        answer="{fair}",
    ),
    "exam": Tier(
        prompt='{setup} {person} says: "{claim_bad}." {other} says: "{claim_good}." Evaluate both claims and explain how much the survey can tell us.',
        reference=[
            "{other} is right: {claim_good}.",
            "{person} is wrong: {bad_flaw}.",
            "The survey tells us only about {measured}; to know about {claimed}, you would need {need}.",
        ],
        points=[
            ("correct_idea", "judges {other}'s claim correct: {claim_good}"),
            ("correct_idea", "judges {person}'s claim wrong: {bad_flaw}"),
            ("use_of_context", "says what the survey does tell us: {measured}"),
            ("completeness", "says what would be needed to go further: {need}"),
            ("communication", "responds to each person in turn"),
        ],
        wrong=[
            ("agrees_with_person", "does it agree with {person}?"),
            (
                "dismisses_everything",
                "does it say the survey tells us nothing at all, instead of what it does show about the people asked?",
            ),
        ],
        answer="{other} is right; the survey shows {measured}",
    ),
}
register(
    "wa_results_misrepresentation",
    12,
    GEN_CASES,
    GEN_TIERS,
    tails=DATA_TAILS,
    doc="generalising survey findings.",
)

# ---------------------------------------------------------------------------
# wa_survey_error_sources: sampling error against measurement error
# ---------------------------------------------------------------------------
ERR_CASES = [
    Case(
        "sampling_two_samples",
        slots={
            **NAME_SLOTS,
            "org": ["Hillcrest High School", "Ridgeway College", "Maple Grove Secondary College"],
            "behaviour": ["walk to school", "bring lunch from home", "play a musical instrument"],
            "n": [100, 200, 300, 400, 500],
            "pop": [30, 35, 40, 45],
            "d1": [-4, -3, 3, 5],
            "d2": [-5, 2, 4, 6],
            "p1": lambda r, v: v["pop"] + v["d1"],
            "p2": lambda r, v: v["pop"] + v["d2"],
            "k1": lambda r, v: v["n"] * (v["pop"] + v["d1"]) // 100,
            "k2": lambda r, v: v["n"] * (v["pop"] + v["d2"]) // 100,
            "diff": lambda r, v: abs(v["p1"] - v["p2"]),
        },
        fields=dict(
            setup="At {org}, {person} takes a random sample of {n} students and finds that {k1} of them ({p1}%) {behaviour}. {other} takes a different random sample of {n} students and finds {k2} ({p2}%). A census of every student later shows that {pop}% {behaviour}.",
            kind="sampling error",
            why="each random sample includes only some of the students, so its percentage differs by chance from the true {pop}%; the two samples differ from each other by {diff} percentage points",
            bigger="yes, a larger random sample would usually be closer to the true value, so the sampling error would tend to shrink",
            bigger_yes="yes",
            reduce="take a larger random sample, and report that the result is an estimate",
            nature="a measurement problem would have the same fault in every sample, whatever its size",
        ),
        wrong=[
            (
                "blames_measurement",
                "does it blame faulty measuring or a badly worded question, when the two samples simply differ by chance?",
            )
        ],
    ),
    Case(
        "measurement_ruler",
        slots={
            **NAME_SLOTS,
            "org": ["Hillcrest High School", "Bayview High School"],
            "off": [2, 3],
            "n": [60, 100, 200],
        },
        fields=dict(
            setup="At {org}, the heights of {n} students are measured against a wall chart that was put up {off} cm too low on the wall, so every height is recorded {off} cm too high.",
            kind="measurement error",
            why="the instrument is faulty, so every recorded height is {off} cm too high, however many students are measured",
            bigger="no, measuring more students would repeat the same {off} cm fault in every measurement",
            bigger_yes="no",
            reduce="check and calibrate the measuring instrument before using it",
            nature="a sampling problem would shrink as the sample grew; this fault does not",
        ),
        wrong=[("blames_sample_size", "does it say the heights are wrong because too few students were measured?")],
    ),
    Case(
        "measurement_rounding",
        slots={
            **NAME_SLOTS,
            "org": ["Ridgeway College", "Maple Grove Secondary College"],
            "n": [100, 150, 200],
            "k": lambda r, v: v["n"] * r.choice([40, 50, 60]) // 100,
            "pk": lambda r, v: pct(v["k"], v["n"]),
        },
        fields=dict(
            setup="At {org}, {n} students are asked how many minutes of homework they did last night. {k} of them ({pk}) say exactly 30, 60 or 90 minutes, and nobody gives an answer such as 47 or 73.",
            kind="measurement error",
            why="students are estimating from memory and rounding to convenient numbers, so the recorded times are not the true times",
            bigger="no, asking more students would repeat the same rounding in every answer",
            bigger_yes="no",
            reduce="ask students to record their homework time during the evening, or give a diary to fill in",
            nature="a sampling problem would shrink as the sample grew; this one does not",
        ),
        wrong=[
            (
                "blames_random_choice",
                "does it blame the way students were chosen, when the problem is how the answers were given and recorded?",
            )
        ],
    ),
    Case(
        "measurement_interviewer",
        slots={
            **NAME_SLOTS,
            "town": TOWNS,
            "n": [120, 200, 300],
            "k": lambda r, v: v["n"] * r.choice([5, 8, 10]) // 100,
            "pk": lambda r, v: pct(v["k"], v["n"]),
        },
        fields=dict(
            setup="In {town}, {n} drivers are asked by a uniformed police officer, face to face, whether they have ever driven over the speed limit. Only {k} of them ({pk}) say yes.",
            kind="measurement error",
            why="drivers are unlikely to admit to an offence to a police officer, so the answers are untruthful, however carefully the drivers were chosen",
            bigger="no, a larger sample would be asked in the same way and would give the same untruthful answers",
            bigger_yes="no",
            reduce="make the survey anonymous and have it run by someone who is not a police officer",
            nature="a sampling problem would shrink as the sample grew; this one does not",
        ),
        wrong=[("blames_small_sample", "does it say the answer would be fixed by asking more drivers the same way?")],
    ),
    Case(
        "sampling_small_sample",
        slots={
            **NAME_SLOTS,
            "club": ["the Eastside Tennis Club", "the Lakeside Fitness Centre", "Bayview Library"],
            "N": [1200, 1500, 2000],
            "n": [20, 25, 30],
            "k": lambda r, v: v["n"] * r.choice([20, 40, 60]) // 100,
            "pk": lambda r, v: pct(v["k"], v["n"]),
            "pop": [30, 50, 55],
        },
        fields=dict(
            setup="{person} chooses {n} of the {N} members of {club} using random numbers. {k} of the {n} ({pk}) say they would use a new cafe. A census of all {N} members later shows that {pop}% would use it.",
            kind="sampling error",
            why="only {n} of the {N} members were asked, so the sample percentage ({pk}) differs by chance from the true {pop}%",
            bigger="yes, a larger random sample would usually be closer to the true {pop}%",
            bigger_yes="yes",
            reduce="increase the number of members in the random sample",
            nature="a measurement problem would remain the same however many members were asked",
        ),
        wrong=[
            (
                "blames_wording",
                "does it blame the wording of the question, when the difference arises from asking only {n} of {N}?",
            )
        ],
    ),
    Case(
        "processing_entry",
        slots={
            **NAME_SLOTS,
            "org": ["Westfield Community Centre", "Southbank Swimming Club"],
            "N": [150, 240, 320],
            "bad": [4, 6, 9, 12],
        },
        fields=dict(
            setup="All {N} members of {org} complete a paper form. When the answers are typed into a spreadsheet, {bad} forms are typed in twice and some ages are mistyped, for example 61 typed as 16.",
            kind="measurement error (an error in recording the data)",
            why="the data have been wrongly recorded, so the spreadsheet no longer matches what the members wrote, and this happens in a census as well as in a sample",
            bigger="no, collecting more forms would only mean more chances for typing mistakes",
            bigger_yes="no",
            reduce="have a second person check the typed entries against the forms, and look for duplicates and impossible values",
            nature="a sampling problem would shrink as the sample grew; this one does not",
        ),
        wrong=[
            ("calls_it_sampling_error", "does it call this sampling error, when every member's form was collected?")
        ],
    ),
]
ERR_TIERS = {
    "easy": Tier(
        prompt="{setup} Is the main source of error here sampling error or measurement error? Explain your choice.",
        reference=["This is {kind}.", "Why: {why}."],
        points=[
            ("correct_idea", "classifies the error as {kind}"),
            ("use_of_context", "explains using this situation: {why}"),
            ("completeness", "gives a reason, not just the label"),
            ("communication", "uses the terms sampling error and measurement error correctly"),
        ],
        wrong=[("label_swapped", "does it call a sampling error a measurement error, or the reverse?")],
        answer="{kind}",
    ),
    "medium": Tier(
        prompt="{setup} Name the source of error and say whether collecting a larger sample would fix it. Give a reason.",
        reference=["The source of error is {kind}.", "Would a larger sample fix it? {bigger}."],
        points=[
            ("correct_idea", "names the source of error: {kind}"),
            ("correct_idea", "answers the larger-sample question correctly: {bigger_yes}"),
            ("use_of_context", "gives a reason from the situation: {bigger}"),
            ("completeness", "answers both parts, the type of error and the effect of a larger sample"),
            ("communication", "reasons in clear sentences"),
        ],
        wrong=[("bigger_sample_fixes_everything", "does it say a larger sample fixes every kind of error?")],
        answer="{kind}; larger sample: {bigger_yes}",
    ),
    "hard": Tier(
        prompt="{setup} Explain how this error arises, and describe one change that would reduce it.",
        reference=["The error is {kind}: {why}.", "To reduce it: {reduce}."],
        points=[
            ("correct_idea", "explains how the error arises: {why}"),
            ("use_of_context", "refers to the details of this situation"),
            ("completeness", "gives a change that really reduces this error: {reduce}"),
            ("communication", "separates the cause from the remedy"),
        ],
        wrong=[
            (
                "remedy_for_the_other_kind",
                "does the proposed change deal with a different kind of error (for example a bigger sample for a faulty instrument)?",
            ),
            ("no_remedy", "does it explain the error but give no change?"),
        ],
        answer="{reduce}",
    ),
    "exam": Tier(
        two=True,
        pair_ok=lambda a, b: a["bigger_yes"] != b["bigger_yes"],
        prompt=(
            'A newsletter says: "All errors in a survey can be fixed by asking more people." Two surveys are described.\n'
            "Survey A: {a_setup}\n"
            "Survey B: {b_setup}\n"
            "Identify the main source of error in each survey. Use the two surveys to evaluate the newsletter's claim, and say what each survey should do about its error."
        ),
        reference=[
            "Survey A has {a_kind}; Survey B has {b_kind}.",
            "The claim is false: for A, {a_bigger}; for B, {b_bigger}.",
            "Fixes: A, {a_reduce}; B, {b_reduce}.",
        ],
        points=[
            ("correct_idea", "names the error in A ({a_kind}) and in B ({b_kind})"),
            (
                "correct_idea",
                "judges the claim false: asking more people helps with sampling error but not measurement error",
            ),
            ("use_of_context", "uses both surveys: A, {a_bigger}; B, {b_bigger}"),
            ("completeness", "gives a fix for each: A, {a_reduce}; B, {b_reduce}"),
            ("communication", "answers the evaluation of the claim before the fixes"),
        ],
        wrong=[("accepts_the_claim", "does it accept that asking more people fixes both surveys?")],
        answer="the claim is false",
    ),
}
register(
    "wa_survey_error_sources",
    12,
    ERR_CASES,
    ERR_TIERS,
    doc="sampling error against measurement error.",
)


# ---------------------------------------------------------------------------
# wa_collection_faults: faults in a short table of recorded values
# ---------------------------------------------------------------------------
def _mean_text(values: list, places: int = 1) -> str:
    m = Fraction(sum(Fraction(str(v)) for v in values), len(values))
    text = f"{round(float(m), places):.{places}f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


COLLECT_SETS = (
    # label, who, unit, lo, hi, unit-slip entry fn, unit-slip note, impossible fn
    dict(
        label="Heights of Year 8 students",
        who="Student",
        unit="cm",
        lo=148,
        hi=176,
        slip=lambda v: f"{v / 100:.2f}",
        slip_note="written in metres instead of centimetres",
        typo=lambda v: v * 10,
        method="measured with a tape measure by six different teams",
    ),
    dict(
        label="Travel times to school",
        who="Student",
        unit="minutes",
        lo=8,
        hi=40,
        slip=lambda v: f"{v * 60}",
        slip_note="written in seconds instead of minutes",
        typo=lambda v: v * 100,
        method="reported by students on one form",
    ),
    dict(
        label="Weekly pocket money amounts",
        who="Student",
        unit="dollars",
        lo=5,
        hi=30,
        slip=lambda v: f"{v * 100}",
        slip_note="written in cents instead of dollars",
        typo=lambda v: v * 100,
        method="reported by students on one form",
    ),
    dict(
        label="Resting heart rates",
        who="Student",
        unit="beats per minute",
        lo=58,
        hi=90,
        slip=lambda v: f"{round(v / 4)}",
        slip_note="counted over 15 seconds instead of one minute",
        typo=lambda v: v * 10,
        method="counted by students in pairs",
    ),
)
FAULTS = ("slip", "typo", "dup", "missing")


def _table(rng, spec: dict, n: int, faults: list[str]):
    """The recorded table: lines, the list of recorded numeric values, the corrected list, and fault details."""
    true = [rng.randint(spec["lo"], spec["hi"]) for _ in range(n)]
    rows = [(i + 1, str(v), v) for i, v in enumerate(true)]  # (id, shown, recorded numeric or None)
    corrected = list(true)
    details = {}
    free = list(range(n))
    rng.shuffle(free)
    for f in sorted(faults, key=lambda x: x == "dup"):
        i = free.pop()
        sid, _shown, v = rows[i]
        if f == "slip":
            shown = spec["slip"](v)
            rows[i] = (sid, shown, float(shown))
            details[f] = dict(row=sid, shown=shown, true=v)
        elif f == "typo":
            shown = str(spec["typo"](v))
            rows[i] = (sid, shown, float(shown))
            details[f] = dict(row=sid, shown=shown, true=v)
        elif f == "missing":
            rows[i] = (sid, "blank", None)
            details[f] = dict(row=sid, shown="blank", true=v)
            corrected[i] = None
        elif f == "dup":
            rows.insert(i + 1, (sid, str(v), v))
            details[f] = dict(row=sid, shown=str(v), true=v)
    recorded = [r[2] for r in rows if r[2] is not None]
    fixed = [c for c in corrected if c is not None]
    return rows, recorded, fixed, details


def _render_rows(spec: dict, rows) -> str:
    return "\n".join(f"{spec['who']} {sid}: {shown}" for sid, shown, _ in rows)


FAULT_WORDS = {
    "slip": lambda d, s: (
        f"{s['who']} {d['row']}'s value ({d['shown']}) is {s['slip_note']}, so it is on a different scale from the other values"
    ),
    "typo": lambda d, s: (
        f"{s['who']} {d['row']}'s value ({d['shown']}) is impossibly large for these data; it is probably a typing slip for {d['true']}"
    ),
    "dup": lambda d, s: (
        f"{s['who']} {d['row']} appears twice with the same value, so one person has been counted twice"
    ),
    "missing": lambda d, s: f"{s['who']} {d['row']} has no value recorded, so that person is missing from the data",
}
FAULT_NAMES = {
    "slip": "a unit slip (a value written in different units)",
    "typo": "an impossible value (a typing slip)",
    "dup": "a duplicated record",
    "missing": "a missing value",
}
FAULT_FIX = {
    "slip": "convert it to the same unit as the others",
    "typo": "check it against the original record and correct it",
    "dup": "delete the repeated record",
    "missing": "go back to the person to collect the value, and not to treat it as zero",
}


def _prompt_head(spec, rows, n_people, recorded):
    return (
        f"{spec['label']} ({spec['unit']}) were {spec['method']}. The recorded values were entered like this:\n"
        f"{_render_rows(spec, rows)}"
    )


def _cf(rng, faults: list[str], ask: str, tier: str) -> dict:
    spec = rng.choice(COLLECT_SETS)
    n = rng.randint(8, 12)
    rows, recorded, fixed, det = _table(rng, spec, n, faults)
    mean_rec = _mean_text(recorded)
    head = _prompt_head(spec, rows, n, recorded)
    words = [FAULT_WORDS[f](det[f], spec) for f in faults]
    names = [FAULT_NAMES[f] for f in faults]
    fixes = [FAULT_FIX[f] for f in faults]
    return spec, rows, recorded, fixed, det, mean_rec, head, words, names, fixes


def _cf_easy(rng) -> dict:
    f = rng.choice(FAULTS)
    spec, rows, recorded, fixed, det, mean_rec, head, words, names, fixes = _cf(rng, [f], "", "easy")
    text = f"{head}\nThere is one fault in the way these data were collected or recorded. Identify the fault and explain why it is a problem."
    mark = marking(
        reference=[f"The fault is {names[0]}.", f"{words[0]}."],
        points=[
            ("correct_idea", f"identifies the fault: {names[0]}"),
            ("use_of_context", f"points to the specific record: {words[0]}"),
            ("completeness", "explains why it is a problem for the data, not just that it looks odd"),
            ("communication", "refers to the record by its number"),
        ],
        wrong=[
            ("picks_a_normal_value", f"does it pick a different record, not {spec['who']} {det[f]['row']}?"),
            (
                "calls_it_a_sampling_fault",
                "does it blame the way the group was sampled, when the fault is in a single recorded value?",
            ),
        ],
    )
    return finish(text, names[0], mark)


def _cf_medium(rng) -> dict:
    f = rng.choice(FAULTS)
    spec, rows, recorded, fixed, det, mean_rec, head, words, names, fixes = _cf(rng, [f], "", "medium")
    text = (
        f"{head}\nThe mean of the recorded values is {mean_rec} {spec['unit']}. One of the records has a fault. "
        "Identify the fault and explain how it affects the mean."
    )
    effect = {
        "slip": f"the value {det[f]['shown']} is on the wrong scale and drags the mean far away from the typical value",
        "typo": f"the value {det[f]['shown']} is far too large and pulls the mean up well above the typical value",
        "dup": f"the repeated value of {det[f]['shown']} is counted twice and pulls the mean towards it",
        "missing": "the person with no value is left out, so the mean is based on one fewer person and may not represent the group",
    }[f]
    mark = marking(
        reference=[f"The fault is {names[0]}.", f"{words[0]}.", f"Effect on the mean of {mean_rec}: {effect}."],
        points=[
            ("correct_idea", f"identifies the fault: {names[0]}"),
            ("use_of_context", f"refers to the record and the stated mean of {mean_rec}: {words[0]}"),
            ("completeness", f"explains the effect on the mean: {effect}"),
            ("communication", "separates the fault from its effect"),
        ],
        wrong=[
            ("no_effect_on_mean", "does it identify the fault but say nothing about how the mean is affected?"),
            ("says_mean_unaffected", "does it say the fault does not change the mean?"),
        ],
    )
    return finish(text, f"{names[0]}; {effect}", mark)


def _cf_hard(rng) -> dict:
    fs = rng.sample(FAULTS, 2)
    spec, rows, recorded, fixed, det, mean_rec, head, words, names, fixes = _cf(rng, fs, "", "hard")
    text = (
        f"{head}\nThere are two faults in these data. Identify both faults, say how you would deal with each one, "
        "and suggest one change to the way the data were collected that would prevent such faults."
    )
    mark = marking(
        reference=[
            f"Fault 1: {names[0]}: {words[0]}. Deal with it: {fixes[0]}.",
            f"Fault 2: {names[1]}: {words[1]}. Deal with it: {fixes[1]}.",
            "Prevention: use one agreed unit and a single recording sheet, enter data from the original record, and have a second person check the entries.",
        ],
        points=[
            ("correct_idea", f"identifies both faults: {names[0]} and {names[1]}"),
            ("use_of_context", f"points to the specific records: {words[0]}; {words[1]}"),
            ("correct_idea", f"gives a sensible treatment for each: {fixes[0]}; {fixes[1]}"),
            (
                "completeness",
                "suggests a change to collection that prevents such faults (agreed units, a standard sheet, checking entries)",
            ),
            ("communication", "deals with the two faults separately"),
        ],
        wrong=[
            (
                "deletes_everything_odd",
                "does it simply delete every odd-looking value rather than correcting what can be corrected?",
            ),
            ("treats_missing_as_zero", "does it enter zero for a missing value?"),
        ],
    )
    return finish(text, f"{names[0]} and {names[1]}", mark)


def _cf_exam(rng) -> dict:
    f = rng.choice(["slip", "typo"])
    spec, rows, recorded, fixed, det, mean_rec, head, words, names, fixes = _cf(rng, [f, "dup"], "", "exam")
    fixed_vals = [v for v in fixed]
    mean_fixed = _mean_text(fixed_vals)
    p, q = rng.sample(["Amir", "Priya", "Noah", "Olivia", "Liam", "Mia"], 2)
    true_v = det[f]["true"]
    text = (
        f"{head}\nThe mean of the recorded values is {mean_rec} {spec['unit']}. The original record shows that "
        f'{spec["who"]} {det[f]["row"]}\'s true value was {true_v}. {p} says: "Delete every value that looks odd and recalculate." '
        f'{q} says: "Leave the table as it is, because every value was recorded by someone." '
        "Evaluate both views, say how you would treat the faults in this table, and say what the mean becomes."
    )
    mark = marking(
        reference=[
            f"Neither is right. The faults are {names[0]}: {words[0]}; and a duplicated record: {words[1]}.",
            f"Correct {spec['who']} {det[f]['row']} to {true_v} and delete the repeated record; do not just delete or ignore them.",
            f"The corrected mean is {mean_fixed} {spec['unit']}, compared with the recorded {mean_rec}.",
        ],
        points=[
            ("correct_idea", f"rejects both views and identifies both faults: {names[0]} and a duplicated record"),
            (
                "correct_idea",
                f"corrects {spec['who']} {det[f]['row']} to {true_v} and removes the repeat, keeping the other values",
            ),
            ("use_of_context", f"states the corrected mean of {mean_fixed} {spec['unit']} against {mean_rec} recorded"),
            (
                "completeness",
                f"explains why deleting everything odd loses data ({p}) and why leaving it distorts the mean ({q})",
            ),
            ("communication", "evaluates each person's view in turn before the treatment"),
        ],
        wrong=[
            ("agrees_with_delete_all", f"does it agree with {p} and delete values that can be corrected?"),
            ("agrees_with_leave", f"does it agree with {q} and keep the faulty values?"),
        ],
    )
    return finish(text, f"correct the faults; mean {mean_fixed} {spec['unit']}", mark)


register_custom(
    "wa_collection_faults",
    12,
    dict(easy=_cf_easy, medium=_cf_medium, hard=_cf_hard, exam=_cf_exam),
    doc="faults in a table of recorded values.",
)

check(len(COLLECT_SETS) == 4, "four recorded-value contexts")
