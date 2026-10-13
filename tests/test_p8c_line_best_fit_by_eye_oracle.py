"""Independent oracle for line-of-best-fit-by-eye.

Enumerates: parses the dataset and the four candidate equations out of
`question.text` with a regex, INDEPENDENTLY recomputes each candidate's ssr
and sad from the parsed numbers, and confirms the parsed `answer_text` is
the one with the smallest ssr AND the smallest sad, each with the required
margin. Never imports anything from `mathgen.generators.p8c_line_best_fit_by_eye_build`.
"""

import re

from mathgen import make

TOPIC = "p8c_line_best_fit_by_eye"
TIERS = ["easy", "medium", "hard"]
SEEDS_PER_TIER = 60

_EQ_RE = re.compile(r"y = (-?\d+)x ([+-]) (\d+)")


def _ssr(points, m, b):
    return sum((y - (m * x + b)) ** 2 for x, y in points)


def _sad(points, m, b):
    return sum(abs(y - (m * x + b)) for x, y in points)


def _parse(question):
    question_text = question.text
    points = [tuple(p["at"]) for p in question.figure["points"]]
    eqs = []
    for m_str, sign, b_str in _EQ_RE.findall(question_text):
        m = int(m_str)
        b = int(b_str) * (1 if sign == "+" else -1)
        eqs.append((m, b))
    return points, eqs


def test_oracle_rederives_the_winner_every_seed_every_tier():
    challenged = []
    rederived = 0
    for tier in TIERS:
        for seed in range(SEEDS_PER_TIER):
            q = make(TOPIC, seed=seed, difficulty=tier)
            points, eqs = _parse(q)
            assert len(points) == 5
            assert len(eqs) == 4
            ssrs = [_ssr(points, m, b) for m, b in eqs]
            sads = [_sad(points, m, b) for m, b in eqs]
            ssr_order = sorted(range(4), key=lambda i: ssrs[i])
            sad_order = sorted(range(4), key=lambda i: sads[i])

            def eq_text(m, b):
                sign = "+" if b >= 0 else "-"
                return f"y = {m}x {sign} {abs(b)}"

            winner_by_ssr = eq_text(*eqs[ssr_order[0]])
            winner_by_sad = eq_text(*eqs[sad_order[0]])
            rederived += 1
            ok = (
                winner_by_ssr == winner_by_sad == q.answer_text
                and 2 * ssrs[ssr_order[1]] >= 3 * ssrs[ssr_order[0]]
                and sads[sad_order[1]] - sads[sad_order[0]] >= 4
            )
            if not ok:
                challenged.append((tier, seed, q.text, q.answer_text, winner_by_ssr, winner_by_sad))
    assert not challenged, f"oracle challenges {len(challenged)} rows: {challenged[:3]}"
    # Report how many questions the oracle re-derived.
    assert rederived == len(TIERS) * SEEDS_PER_TIER
