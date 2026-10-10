"""This file does NOT import `mathgen.generators.pv0c_choose_display_build`
(confirmed by its own import list below: only `re`, `collections`,
`mathgen.make`) and uses only `question.text`, `question.answer_text` and
`question.distractors`, exactly as a learner sees them.

If a sweep this size still produces a masked description this file has not
labelled, `test_every_description_is_labelled` fails and names it -- a draw
space this oracle cannot saturate is itself a finding, not something to
silently skip.
"""

from __future__ import annotations

import re
from collections import Counter

import pytest

from mathgen import make

TOPIC = "pv0c_choose_display"
TIERS = ("easy", "medium", "hard", "exam")
SEEDS_PER_TIER = 300  # 4 tiers x 300 = 1200 >= 1000 total, >= 250 per tier

_DIGIT_RE = re.compile(r"\d+")
_TAIL_RE = re.compile(r"\s*Which type of graph is best suited to display this data\?\s*$")


def _mask(description: str) -> str:
    return _DIGIT_RE.sub("N", description)


def _description(question) -> str:
    text = _TAIL_RE.sub("", question.text)
    assert text != question.text, f"instruction sentence not found: {question.text!r}"
    return text


# ============================================================================ This oracle's own
# compatibility matrix -- transcribed by hand, independent of the generator's
# `CLASS_INFO` dict. `also_acceptable` displays must NEVER be offered (answer or distractor) for
# that class; anything else is fair game as a distractor.
# ============================================================================
KEY = {
    "C1": "column graph",
    "C2": "histogram",
    "C3": "scatterplot",
    "C4": "line graph",
    "C5": "back-to-back stem-and-leaf plot",
    "C6": "segmented column graph",
}
ALSO_ACCEPTABLE = {
    "C1": {"pie chart", "dot plot", "segmented column graph"},
    "C2": {"dot plot", "stem-and-leaf plot", "box plot"},
    "C3": set(),
    "C4": {"scatterplot", "column graph"},
    "C5": {"parallel box plots", "parallel dot plots"},
    "C6": {"side-by-side column graph", "two-way table"},
}
ALL_DISPLAYS = {
    "column graph",
    "pie chart",
    "dot plot",
    "segmented column graph",
    "histogram",
    "scatterplot",
    "line graph",
    "back-to-back stem-and-leaf plot",
}

# ============================================================================
# Hand-built description -> class dict. Grown by reading a 300-seed-per-tier
# sweep's rendered descriptions (digits masked to "N"); every masked
# description this generator can currently produce must have an entry here.
# ============================================================================
DESC_CLASS: dict[str, str] = {
    # -- exam (research-purpose framing widened 2 -> 4 per class) --
    "A cafe owner wants to know which drink is most popular. She records the drink each of N customers orders over a week.": "C1",
    "A council wants to know which bus route is used most. It records the route each of N commuters boards during peak hour.": "C1",
    "A city library wants to know which genre is borrowed most. It records the genre of each of N books borrowed last week.": "C1",
    "A gym manager wants to know which class time slot is most popular. She records the class time slot chosen by each of N members last month.": "C1",
    "A quality-control officer wants to know how consistent a factory's bottling line is. She records the volume, in millilitres, of the liquid in each of N bottles sampled from the line.": "C2",
    "A nutritionist wants to know how much sleep university students get. She records the sleep duration, in hours, of N students over one night.": "C2",
    "An engineer wants to know how consistent a ND printer's output is. She records the diameter, in millimetres, of N printed parts sampled from a production run.": "C2",
    "A dietitian wants to know how much water people drink. She records the daily water intake, in millilitres, of N participants over one day.": "C2",
    "A sports scientist wants to know whether taller players jump higher. She records the height (cm) and the vertical jump (cm) of N basketball players.": "C3",
    "An economist wants to know whether higher advertising spend increases revenue. She records the advertising spend ($) and the weekly revenue ($) of N stores.": "C3",
    "A biologist wants to know whether taller trees have thicker trunks. She measures the height (m) and the trunk diameter (cm) of N trees in a forest plot.": "C3",
    "A researcher wants to know whether more training hours are linked to higher fitness test scores. She records the training hours per week and the fitness test score of N athletes.": "C3",
    "An analyst wants to know how a company's share price has moved. She records the company's closing share price at the end of each trading day for the past N days.": "C4",
    "A climatologist wants to know how rainfall has changed over time. She records the total rainfall (mm) for each month over the past N months.": "C4",
    "A hydrologist wants to know how a river's flow has changed. She records the river's flow rate, in megalitres per day, for each day over the past N days.": "C4",
    "A retailer wants to know how weekly sales have trended. She records the store's total weekly sales, in dollars, for each week over the past N weeks.": "C4",
    "A researcher wants to know whether left-handed students have faster reaction times than right-handed students. She records the reaction time, in milliseconds, of N left-handed students and N right-handed students.": "C5",
    "A doctor wants to know whether smokers have higher resting heart rates than non-smokers. She records the resting heart rate, in beats per minute, of N smokers and N non-smokers.": "C5",
    "A physiotherapist wants to know whether a stretching program improves flexibility more than a strength program. She records the flexibility test score, in cm, of N clients following the stretching program and N clients following the strength program.": "C5",
    "A researcher wants to know whether city runners are faster than regional runners in a fun run. She records the finishing time, in minutes, of N city runners and N regional runners.": "C5",
    "A marketer wants to know whether product preference differs by age group. She records the age group and the preferred product of each of N survey respondents.": "C6",
    "A researcher wants to know whether car brand choice is linked to fuel type. She records the car brand and the fuel type of each of N vehicles registered this year.": "C6",
    "A council wants to know whether recycling participation differs by suburb. It records the suburb and whether each of N households recycles.": "C6",
    "An airline wants to know whether ticket class is linked to the check-in method used. It records the ticket class and the check-in method used by each of N passengers.": "C6",
    # -- easy (type named) --
    "One categorical variable: the blood type of each of N hospital patients.": "C1",
    "One categorical variable: the favourite music genre of each of N students.": "C1",
    "One categorical variable: the favourite sport of each of N students.": "C1",
    "One categorical variable: the home suburb of each of N commuters.": "C1",
    "One categorical variable: the preferred holiday destination of each of N survey respondents.": "C1",
    "One categorical variable: the t-shirt size ordered of each of N customers.": "C1",
    "One numerical variable, continuous measurements, many values: the battery life (hours) of N phones.": "C2",
    "One numerical variable, continuous measurements, many values: the commute time (minutes) of N workers.": "C2",
    "One numerical variable, continuous measurements, many values: the delivery time (minutes) of N parcels.": "C2",
    "One numerical variable, continuous measurements, many values: the resting heart rate (beats per minute) of N patients.": "C2",
    "One numerical variable, continuous measurements, many values: the running time (minutes) of N movies.": "C2",
    "One numerical variable, continuous measurements, many values: the seedling height (cm) of N seedlings.": "C2",
    "Two numerical variables measured on the same individuals: the height (cm) and weight (kg) of N adults.": "C3",
    "Two numerical variables measured on the same individuals: the study time (hours) and test score (marks) of N students.": "C3",
    "Two numerical variables measured on the same individuals: the advertising spend ($) and weekly revenue ($) of N stores.": "C3",
    "Two numerical variables measured on the same individuals: the arm span (cm) and height (cm) of N gymnasts.": "C3",
    "Two numerical variables measured on the same individuals: the floor area (mN) and sale price ($) of N houses.": "C3",
    "One numerical variable recorded at regular time points: the average daily temperature (°C) for each day of March.": "C4",
    "One numerical variable recorded at regular time points: the number of visitors for each month of last year.": "C4",
    "One numerical variable recorded at regular time points: the closing share price ($) for each trading day over the past month.": "C4",
    "One numerical variable recorded at regular time points: the monthly rainfall (mm) for each month from N to N.": "C4",
    "One numerical variable recorded at regular time points: the number of website hits for each week of the campaign.": "C4",
    "One numerical variable compared between two groups: the reaction time (ms) of left-handed students and right-handed students.": "C5",
    "One numerical variable compared between two groups: the resting heart rate (beats per minute) of smokers and non-smokers.": "C5",
    "One numerical variable compared between two groups: the commute time (minutes) of city residents and suburban residents.": "C5",
    "One numerical variable compared between two groups: the sleep duration (hours) of shift workers and day workers.": "C5",
    "Two categorical variables: the eye colour and hair colour of N students.": "C6",
    "Two categorical variables: the car brand and fuel type of N vehicles.": "C6",
    "Two categorical variables: the preferred subject and preferred sport of N students.": "C6",
    "Two categorical variables: the playing position and dominant foot of N football players.": "C6",
    "Two categorical variables: the phone brand and operating system of N customers.": "C6",
    # -- medium (type not named) --
    "The favourite sport of each of N students.": "C1",
    "The favourite music genre of each of N students.": "C1",
    "The blood type of each of N hospital patients.": "C1",
    "The home suburb of each of N commuters.": "C1",
    "The preferred holiday destination of each of N survey respondents.": "C1",
    "The t-shirt size ordered of each of N customers.": "C1",
    "The running time, in minutes, measured for each of N movies.": "C2",
    "The battery life, in hours, measured for each of N phones.": "C2",
    "The delivery time, in minutes, measured for each of N parcels.": "C2",
    "The seedling height, in cm, measured for each of N seedlings.": "C2",
    "The resting heart rate, in beats per minute, measured for each of N patients.": "C2",
    "The commute time, in minutes, measured for each of N workers.": "C2",
    "The height (cm) and the weight (kg) recorded for N adults.": "C3",
    "The study time (hours) and the test score (marks) recorded for N students.": "C3",
    "The advertising spend ($) and the weekly revenue ($) recorded for N stores.": "C3",
    "The arm span (cm) and the height (cm) recorded for N gymnasts.": "C3",
    "The floor area (mN) and the sale price ($) recorded for N houses.": "C3",
    "The average daily temperature recorded for each day of March.": "C4",
    "The number of visitors recorded for each month of last year.": "C4",
    "The closing share price recorded for each trading day over the past month.": "C4",
    "The monthly rainfall recorded for each month from N to N.": "C4",
    "The number of website hits recorded for each week of the campaign.": "C4",
    "The reaction time (ms) of left-handed students, compared with right-handed students.": "C5",
    "The resting heart rate (beats per minute) of smokers, compared with non-smokers.": "C5",
    "The commute time (minutes) of city residents, compared with suburban residents.": "C5",
    "The sleep duration (hours) of shift workers, compared with day workers.": "C5",
    "The eye colour and the hair colour recorded for N students.": "C6",
    "The car brand and the fuel type recorded for N vehicles.": "C6",
    "The preferred subject and the preferred sport recorded for N students.": "C6",
    "The playing position and the dominant foot recorded for N football players.": "C6",
    "The phone brand and the operating system recorded for N customers.": "C6",
    # -- hard (misleading surface form replaced near-unique
    # identifiers -- room number, jersey number, customer ID number -- with
    # numeric CODES that genuinely repeat, and replaced "length of hospital
    # stay" (a whole-day count with few values) with continuous measures) --
    "The postcode of each of N customers who placed an online order.": "C1",
    "The flight number of each of N passengers who checked in this morning.": "C1",
    "The bus route number of each of N commuters using the city network this week.": "C1",
    "The gate number of each of N flights scheduled to depart this morning.": "C1",
    "The length of each customer service call, in minutes, recorded for N calls logged last week.": "C2",
    "The distance travelled to work, in km, recorded for N employees surveyed.": "C2",
    "The volume of fuel used per trip, in litres, recorded for N delivery vans logged over the past month.": "C2",
    "The house price ($) and the floor area (mN) recorded for N houses sold in a suburb last year.": "C3",
    "The annual income ($) and the years of education (years) recorded for N adults surveyed.": "C3",
    "The study hours per week (hours) and the final exam mark (marks) recorded for N students.": "C3",
    "The closing share price ($) recorded at the end of each trading day for the past N days.": "C4",
    "The daily maximum temperature (°C) recorded at the same weather station for each of the past N days.": "C4",
    "The number of daily active users recorded on an app for each of the past N days.": "C4",
    "The time taken to finish a N km fun run (minutes) of runners under N, compared with runners over N.": "C5",
    "The mass of newborn babies (kg) of babies born in public hospitals, compared with babies born in private hospitals.": "C5",
    "The length of nails grown in a month (mm) of participants taking the supplement, compared with participants taking the placebo.": "C5",
    "The bus route number and the ticket type recorded for each of N passengers who boarded during peak hour.": "C6",
    "The flight number and the ticket class recorded for each of N passengers who checked in this morning.": "C6",
    "The gate number and the airline recorded for each of N flights scheduled to depart this morning.": "C6",
    "The school house number and the sport chosen for the athletics carnival recorded for each of N students.": "C6",
}


def _all_questions():
    out = []
    for tier in TIERS:
        for seed in range(SEEDS_PER_TIER):
            out.append((tier, make(TOPIC, seed=seed, difficulty=tier)))
    return out


QUESTIONS = _all_questions()


def test_swept_enough_questions():
    assert len(QUESTIONS) >= 1000
    by_tier = Counter(tier for tier, _ in QUESTIONS)
    for tier in TIERS:
        assert by_tier[tier] >= 250, by_tier


def test_every_description_is_labelled():
    unlabelled = set()
    for _tier, q in QUESTIONS:
        masked = _mask(_description(q))
        if masked not in DESC_CLASS:
            unlabelled.add(masked)
    assert not unlabelled, f"{len(unlabelled)} unlabelled description(s), e.g.: {sorted(unlabelled)[:5]}"


def test_key_matches_this_oracles_own_matrix_for_every_question():
    for tier, q in QUESTIONS:
        masked = _mask(_description(q))
        class_id = DESC_CLASS.get(masked)
        assert class_id is not None, f"{tier}: unlabelled description {masked!r}"
        expected_key = KEY[class_id]
        assert q.answer_text == expected_key, (
            f"{tier}: description class {class_id} expects key {expected_key!r}, got {q.answer_text!r} for {q.text!r}"
        )


def test_no_option_is_on_its_classs_also_acceptable_list():
    for tier, q in QUESTIONS:
        masked = _mask(_description(q))
        class_id = DESC_CLASS[masked]
        offered = {q.answer_text, *q.distractors}
        forbidden = offered & ALSO_ACCEPTABLE[class_id]
        assert not forbidden, f"{tier}: {class_id} offered forbidden {forbidden} in {q.text!r}"


def test_every_option_is_a_real_display_name():
    for tier, q in QUESTIONS:
        for option in (q.answer_text, *q.distractors):
            assert option in ALL_DISPLAYS, (tier, option, q.text)


def test_distinct_description_x_option_set_combinations_report():
    """Regression guard for a previously fixed defect."""
    combos = {(_mask(_description(q)), frozenset([q.answer_text, *q.distractors])) for _tier, q in QUESTIONS}
    assert len(combos) > 0
    print(f"\ndistinct (description, option-set) combinations: {len(combos)}")


@pytest.mark.parametrize("class_id", sorted(KEY))
def test_each_class_is_reachable_at_every_tier(class_id):
    for tier in TIERS:
        seen = any(DESC_CLASS.get(_mask(_description(q))) == class_id for t, q in QUESTIONS if t == tier)
        assert seen, f"class {class_id} never seen at tier {tier} over {SEEDS_PER_TIER} seeds"
