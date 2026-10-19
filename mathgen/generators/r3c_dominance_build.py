"""Generates `dominance-matrix-ranking` questions (easy, medium, hard) and the exam chain `r3c_dominance_chain`.

In a round robin every team plays every other team once and there are no draws. The one-step dominance matrix ``D`` has
``D[i][j] = 1`` when team ``i`` beat team ``j`` and 0 otherwise (zeros on the diagonal), so a row sum is a team's wins.
``D squared`` counts the two-step routes (team ``i`` beat someone who beat team ``j``), and a team is ranked by its
row sum of ``D + D squared``, which separates teams that have the same number of wins.

* easy: the dominance matrix is given; rank the teams by one-step dominance (wins), ties in listed order;
* medium: the dominance matrix is given; rank the teams by one-step plus two-step dominance;
* hard: the results are listed in sentences, the two best teams have the same number of wins, and the teams must be
  ranked by one-step plus two-step dominance;
* exam (``r3c_dominance_chain``): the matrix written from the results, one team's one-step dominance, its two-step dominance and the combined
  score (carried forward).

Tournaments are drawn at random, then kept only when the ranking has the feature the tier needs (no tied final
ranking; a different order from the wins alone; a tie in wins at the top).
"""

from __future__ import annotations

from fractions import Fraction

from ..core import attempts, check, generator
from ._r3c_kit import mat_mul, matrix_part, matrix_single, names_text, number_part, ordered_single, transpose

_CLUBS = (
    "Kookaburras",
    "Wombats",
    "Emus",
    "Dingoes",
    "Cockatoos",
    "Echidnas",
    "Magpies",
    "Brolgas",
    "Wallabies",
)

_SPORTS = ("netball", "basketball", "hockey", "volleyball")


def _tournament(rng, n):
    D = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.5:
                D[i][j] = 1
            else:
                D[j][i] = 1
    return D


def _square(D):
    return [[int(v) for v in row] for row in mat_mul(D, D)]


def _scores(D):
    n = len(D)
    D2 = _square(D)
    wins = [sum(D[i]) for i in range(n)]
    two = [sum(D2[i]) for i in range(n)]
    return wins, two, [w + t for w, t in zip(wins, two)]


def _ranking(names, scores):
    """Names from highest score to lowest; None when two teams tie."""
    if len(set(scores)) < len(scores):
        return None
    return [names[i] for i in sorted(range(len(names)), key=lambda i: -scores[i])]


def _stable_ranking(names, scores):
    return [names[i] for i in sorted(range(len(names)), key=lambda i: (-scores[i], i))]


def _results_text(rng, D, names):
    n = len(names)
    games = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rng.shuffle(games)
    out = []
    for i, j in games:
        w, l = (i, j) if D[i][j] else (j, i)
        if rng.random() < 0.35:
            out.append(f"{names[l]} lost to {names[w]}.")
        else:
            out.append(f"{names[w]} beat {names[l]}.")
    return " ".join(out)


def _setup(rng, n):
    sport = rng.choice(_SPORTS)
    names = sorted(rng.sample(_CLUBS, n))
    return sport, names


def _matrix_slips(rng, D):
    n = len(D)
    flipped = [[int(v) for v in row] for row in transpose(D)]
    complement = [[(1 - D[i][j]) if i != j else 0 for j in range(n)] for i in range(n)]
    diag = [[1 if i == j else D[i][j] for j in range(n)] for i in range(n)]
    out = [
        (flipped, "r3c-dom-rows-columns-swapped"),
        (complement, "r3c-dom-wins-losses-swapped"),
        (diag, "r3c-dom-diagonal-ones"),
    ]
    for _ in range(3):
        i, j = rng.sample(range(n), 2)
        slip = [list(r) for r in D]
        slip[i][j] = 1 - slip[i][j]
        out.append((slip, "r3c-dom-entry-slip"))
    swapped = [list(r) for r in D]
    a, b = rng.sample(range(n), 2)
    swapped[a], swapped[b] = swapped[b], swapped[a]
    out.append((swapped, "r3c-dom-row-order-slip"))
    return out


def _order_note(names):
    return (
        f"Write the matrix with a row and a column for each of {names_text(names)}, in that order, with 1 when the "
        "team in the row beat the team in the column and 0 otherwise. Write it as rows, for example [[0, 1], [0, 0]]."
    )


def _matrix_text(D):
    return "[" + ", ".join("[" + ", ".join(str(v) for v in row) + "]" for row in D) + "]"


def _easy(rng):
    for _ in attempts(500, "an easy dominance question"):
        n = 5
        sport, names = _setup(rng, n)
        D = _tournament(rng, n)
        wins, two, _both = _scores(D)
        correct = _stable_ranking(names, wins)
        if len(set(wins)) == n or len(set(wins)) < 3 or correct == _stable_ranking(names, two):
            continue
        text = (
            f"Each of {n} {sport} teams plays each of the others once, with no draws. The one-step dominance matrix D "
            f"has a row and a column for each of {names_text(names)}, in that order, with 1 when the team in the row "
            f"beat the team in the column and 0 otherwise. D = {_matrix_text(D)}. "
            "Rank the teams from strongest to weakest by their one-step dominance (the number of teams each beat). "
            "Teams with the same one-step dominance are listed in the order they appear above. "
            "Write the teams in order, separated by commas."
        )
        Dt = [[int(v) for v in row] for row in transpose(D)]
        adjacent = list(correct)
        k = rng.randrange(n - 1)
        adjacent[k], adjacent[k + 1] = adjacent[k + 1], adjacent[k]
        slips = [
            (correct[::-1], "r3c-dom-order-reversed"),
            (_stable_ranking(names, [-sum(r) for r in Dt]), "r3c-dom-rows-columns-swapped"),
            (_stable_ranking(names, two), "r3c-dom-two-step-only"),
            (adjacent, "r3c-dom-ranking-slip"),
            (_stable_ranking(names, [-w for w in wins]), "r3c-dom-order-reversed"),
        ]
        working = [
            "The one-step dominance of a team is its row sum in D.",
            *[f"{names[i]}: {wins[i]}" for i in range(n)],
        ]
        q = ordered_single(text, correct, working, slips)
        if q:
            return q
    raise AssertionError  # pragma: no cover


def _rank_slips(names, D):
    n = len(names)
    wins, two, both = _scores(D)
    Dt = [[int(v) for v in row] for row in transpose(D)]
    _, _, both_t = _scores(Dt)
    correct = _ranking(names, both)
    out = [
        (_stable_ranking(names, wins), "r3c-dom-one-step-only"),
        (_stable_ranking(names, two), "r3c-dom-two-step-only"),
        (_stable_ranking(names, [-b for b in both_t]), "r3c-dom-rows-columns-swapped"),
        (correct[::-1], "r3c-dom-order-reversed"),
        (_stable_ranking(names, [-w for w in wins]), "r3c-dom-order-reversed"),
    ]
    return out


def _medium(rng):
    for _ in attempts(1500, "a medium dominance question"):
        n = 5
        sport, names = _setup(rng, n)
        D = _tournament(rng, n)
        wins, two, both = _scores(D)
        correct = _ranking(names, both)
        if correct is None:
            continue
        if _stable_ranking(names, wins) == correct or _stable_ranking(names, two) == correct:
            continue
        text = (
            f"Each of {n} {sport} teams plays each of the others once, with no draws. The one-step dominance matrix D "
            f"has a row and a column for each of {names_text(names)}, in that order, with 1 when the team in the row "
            f"beat the team in the column and 0 otherwise. D = {_matrix_text(D)}. "
            "Rank the teams from strongest to weakest by adding each team's one-step dominance and its two-step "
            "dominance. Write the teams in order, separated by commas."
        )
        working = [
            "The one-step dominance of a team is its row sum in D; the two-step dominance is its row sum in D squared.",
            *[f"{names[i]}: {wins[i]} + {two[i]} = {both[i]}" for i in range(n)],
        ]
        q = ordered_single(text, correct, working, _rank_slips(names, D))
        if q:
            return q
    raise AssertionError  # pragma: no cover


def _hard(rng):
    for _ in attempts(4000, "a hard dominance question"):
        n = rng.randint(5, 6)
        sport, names = _setup(rng, n)
        D = _tournament(rng, n)
        wins, two, both = _scores(D)
        correct = _ranking(names, both)
        top = sorted(wins, reverse=True)
        if correct is None or top[0] != top[1] or top[1] == top[2]:
            continue
        if _stable_ranking(names, wins) == correct:
            continue
        text = (
            f"Each of {n} {sport} teams plays each of the others once, with no draws. "
            + _results_text(rng, D, names)
            + " Rank the teams from strongest to weakest by adding each team's one-step dominance and its two-step "
            "dominance (the number of teams it beat directly plus the number of routes through one beaten team). "
            "Write the teams in order, separated by commas."
        )
        working = [
            "Build the dominance matrix D from the results, then D squared.",
            *[f"{names[i]}: {wins[i]} + {two[i]} = {both[i]}" for i in range(n)],
            f"Two teams have {top[0]} wins, so the two-step dominance separates them.",
        ]
        q = ordered_single(text, correct, working, _rank_slips(names, D))
        if q:
            return q
    raise AssertionError  # pragma: no cover


@generator(
    "r3c_dominance",
    level=12,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r3c_dominance(rng, tier="easy"):
    """Dominance matrices and rankings from round-robin results; the tiers are in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


@generator("r3c_dominance_chain", level=12, presets={"exam": dict(tier="exam")})
def r3c_dominance_chain(rng, tier="exam"):
    """Exam chain: the matrix, one team's one-step and two-step dominance, and their sum."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(2000, "a dominance chain"):
        n = 5
        sport, names = _setup(rng, n)
        D = _tournament(rng, n)
        wins, two, both = _scores(D)
        k = rng.randrange(n)
        if not (1 <= wins[k] <= 3) or two[k] < 2 or len(set(both)) < 3 or wins[k] == two[k]:
            continue
        stem = f"Each of {n} {sport} teams plays each of the others once, with no draws. " + _results_text(
            rng, D, names
        )
        D2 = _square(D)
        part_a = matrix_part(
            _order_note(names),
            D,
            3,
            ["Put a 1 in the row of the winner and the column of the loser of each game."],
            _matrix_slips(rng, D),
        )
        losses = n - 1 - wins[k]
        part_b = number_part(
            f"What is the one-step dominance of the {names[k]}, the number of teams they beat directly?",
            wins[k],
            1,
            [f"The row sum of the {names[k]} in D is {wins[k]}."],
            [
                (losses, "r3c-dom-wins-losses-swapped"),
                (wins[k] + 1, "r3c-dom-count-slip"),
                (wins[k] - 1, "r3c-dom-count-slip"),
                (n, "r3c-dom-counts-all-teams"),
            ],
            lo=0,
            sensible=(-1, None),
        )
        part_c = number_part(
            f"What is the two-step dominance of the {names[k]}, the number of routes of the form: they beat a team that beat a third team?",
            two[k],
            2,
            [f"The row sum of the {names[k]} in D squared is {two[k]}."],
            [
                (sum(D2[i][k] for i in range(n)), "r3c-dom-rows-columns-swapped"),
                (wins[k] * wins[k], "r3c-dom-squares-wins"),
                (two[k] + 1, "r3c-dom-count-slip"),
                (two[k] - 1, "r3c-dom-count-slip"),
                (sum(D2[k][j] > 0 for j in range(n)), "r3c-dom-counts-teams-not-routes"),
            ],
            lo=0,
            sensible=(-1, None),
        )
        part_d = number_part(
            f"What is the combined dominance of the {names[k]}, the one-step and two-step dominance added?",
            both[k],
            1,
            [f"{wins[k]} + {two[k]} = {both[k]}."],
            [
                (wins[k] * two[k], "r3c-dom-multiplies"),
                (two[k], "r3c-dom-two-step-only"),
                (wins[k], "r3c-dom-one-step-only"),
                (both[k] + 1, "r3c-dom-count-slip"),
            ],
            follow="part_b + part_c",
            lo=1,
            sensible=(0, None),
        )
        if None in (part_a, part_b, part_c, part_d):
            continue
        return dict(text=stem, parts=[part_a, part_b, part_c, part_d])
    raise AssertionError  # pragma: no cover
