"""Exam tier for `f10-percentage-equivalence-comparison`: a percentage as a fraction, used to compare two teams.

Two teams played different numbers of games. One team's record is given as wins out of games; the other's as a
percentage of its games. The learner converts the percentage to a fraction in simplest form, uses it to find the
number of wins, and compares the two teams' wins.

* (a) the second team's win rate as a fraction in simplest form (1 mark);
* (b) the number of games that team won: the fraction of its games (2 marks, follow-through from (a));
* (c) how many more games the first team won (1 mark, follow-through from (b)).

Built backwards: the second team's games are a multiple of the simplified denominator, so its wins are whole.
"""

from __future__ import annotations

import random
from fractions import Fraction

from mathgen.answers import Distractor
from mathgen.core import attempts, check, generator, latex_answer, numeric_answer, pick_distractors, rational_text

# (opening sentence naming the sport and season, what a win is called)
_SETTINGS = [
    "In the winter netball season in Bendigo,",
    "In the junior soccer competition in Newcastle,",
    "In a school basketball carnival in Geelong,",
    "In the summer cricket season in Toowoomba,",
    "In a lawn bowls club's pennant season in Hobart,",
]
_TEAMS = [("Eagles", "Hawks"), ("Rockets", "Lions"), ("Stingrays", "Wombats"), ("Magpies", "Kookaburras")]
_BENCHMARKS = {10, 20, 25, 50, 75}


class _Retry(Exception):
    pass


def _whole(pairs):
    """Distractors for the (value, misconception) pairs that are positive whole numbers."""
    return [Distractor(str(int(v)), m, v) for v, m in pairs if v.denominator == 1 and v > 0]


def _attempt(rng: random.Random) -> dict:
    for _ in attempts(1000, "a percentage whose fraction divides the second team's games"):
        pct = rng.randint(4, 96)
        rate = Fraction(pct, 100)
        if pct in _BENCHMARKS or rate.denominator > 20:
            continue
        games2 = rate.denominator * rng.randint(1, 40 // rate.denominator)
        if games2 < 12:
            continue
        wins2 = rate * games2
        games1 = rng.randint(12, 30)
        if wins2 + 1 >= games1:
            continue
        wins1 = rng.randint(int(wins2) + 1, games1 - 1)
        if wins1 <= wins2 or Fraction(wins1, games1) == rate:
            continue
        diff = wins1 - wins2
        shown = {pct, games1, games2, wins1}
        if {int(wins2), int(diff)} & shown or int(wins2) == int(diff):
            continue
        break

    opening = rng.choice(_SETTINGS)
    team1, team2 = rng.choice(_TEAMS)
    stem = f"{opening} the {team1} won {wins1} of their {games1} games. The {team2} won {pct}% of their {games2} games."
    fraction_wrong_power = Fraction(pct, 1000)
    d_a = pick_distractors(
        rational_text(rate),
        [
            Distractor(rational_text(Fraction(pct)), "drops-the-percent-divisor", Fraction(pct)),
            Distractor(rational_text(fraction_wrong_power), "wrong-power-of-ten-denominator", fraction_wrong_power),
            Distractor(rational_text(1 - rate), "complement-instead-of-value", 1 - rate),
            Distractor(rational_text(Fraction(pct, 10)), "wrong-power-of-ten-denominator", Fraction(pct, 10)),
        ],
        3,
    )
    d_b = pick_distractors(
        str(int(wins2)),
        _whole(
            [
                (Fraction(games2) - wins2, "complement-instead-of-value"),
                (Fraction(pct), "percent-as-absolute"),
                (Fraction(games2, rate.denominator), "ignores-numerator"),
                (Fraction(games2 * pct, 10), "decimal-place-shift"),
            ]
        ),
        3,
    )
    d_c = pick_distractors(
        str(int(diff)),
        _whole(
            [
                (wins1 - Fraction(pct), "percent-as-absolute"),
                (wins1 - (games2 - wins2), "complement-instead-of-value"),
                (wins1 + wins2, "story-operation-reversed"),
                (wins1 - Fraction(games2, rate.denominator), "ignores-numerator"),
            ]
        ),
        3,
    )
    if min(len(d_a), len(d_b), len(d_c)) < 3:
        raise _Retry
    check(wins2 + (games2 - wins2) == games2 and diff == wins1 - wins2, "counts do not add up")

    return dict(
        text=stem,
        parts=[
            dict(
                prompt=f"Write the {team2}' win rate, {pct}%, as a fraction in simplest form.",
                answer_kind="fraction",
                answer={"p": rate.numerator, "q": rate.denominator},
                answer_text=rational_text(rate),
                answer_latex=latex_answer(rational_text(rate)),
                marks=1,
                sensible_range=(0, 1),
                working=[f"{pct}% = {pct}/100 = {rational_text(rate)}."],
                distractors=d_a,
            ),
            dict(
                prompt=f"How many games did the {team2} win?",
                answer_kind="number",
                answer=numeric_answer(wins2),
                answer_text=str(int(wins2)),
                answer_latex=latex_answer(str(int(wins2))),
                marks=2,
                sensible_range=(0, None),
                follow_through=f"part_a*{games2}",
                working=[f"{rational_text(rate)} of {games2} = {int(wins2)}."],
                distractors=d_b,
            ),
            dict(
                prompt=f"How many more games did the {team1} win than the {team2}?",
                answer_kind="number",
                answer=numeric_answer(diff),
                answer_text=str(int(diff)),
                answer_latex=latex_answer(str(int(diff))),
                marks=1,
                sensible_range=(0, None),
                follow_through=f"{wins1} - part_b",
                working=[f"{wins1} - {int(wins2)} = {int(diff)}."],
                distractors=d_c,
            ),
        ],
    )


def _build(rng: random.Random) -> dict:
    for _ in attempts(200, "a draw with three distinct distractors per part"):
        try:
            return _attempt(rng)
        except _Retry:
            continue


@generator("exd_x_pr5_season_rates", level=6, presets={"exam": dict(tier="exam")})
def exd_x_pr5_season_rates(rng: random.Random, tier: str = "exam") -> dict:
    """A win rate given as a percentage, converted to a fraction and used to compare two teams' wins."""
    check(tier == "exam", "exd_x_pr5_season_rates: only the exam tier exists")
    return _build(rng)
