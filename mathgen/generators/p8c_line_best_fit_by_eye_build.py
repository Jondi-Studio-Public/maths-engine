"""Generates `p8c_line_best_fit_by_eye` questions: pick the best of four candidate lines for a small stated
scatter dataset.

Line of best fit by eye.

- ``p8c_line_best_fit_by_eye`` (``senior-find-the-line-of-best-fit-by-eye``,
  ACMEM141, domain statistics-probability, answer: mcq) -- four candidate
  lines are proposed against a small integer scatter dataset, drawn as a
  scatter plot (a ``cartesian`` figure; the prompt does not list the points). "Best fit" is never a judgement
  call: it is the candidate with the strictly smallest sum of squared
  residuals (ssr) AND the strictly smallest sum of absolute residuals (sad)
  against the *drawn* points, each separated from its own runner-up by a
  real margin -- never a hidden "true" generating line the learner cannot
  see. Every data point and every candidate (m, b) is an integer, so every
  residual sum is an exact integer; no floats, no rounding ambiguity.
"""

from __future__ import annotations

import re

from .. import figures
from ..core import check, generator, join_terms, pick_distractors

# Every candidate slope is drawn (or, for the outlier-drag candidate,
# adjusted) so |m| >= 2 -- every rendered equation then shows an explicit
# numeric coefficient ("y = 3x + 2", never a bare "y = x + 2"), so the four
# options in one question are always the identical "y = <int>x +/- <int>"
# shape; none is identifiable by format alone.
_M_CHOICES = (-5, -4, -3, -2, 2, 3, 4, 5)
_X_VALUES = (1, 2, 3, 4, 5)

# Difficulty controls how far the three losing candidates sit from the winner, and how large the
# single outlier's noise is -- not the dataset's size or the arithmetic shape, which is genuine
# reasoning escalation.
_TIER_PARAMS = {
    "easy": dict(
        noise_choices=(-1, 0, 1),
        outlier_noise_choices=(-3, -2, 2, 3),
        offset_choices=(6, 7, 8, 9, 10),
        sign_tweak_choices=(-1, 0, 1),
    ),
    "medium": dict(
        noise_choices=(-2, -1, 0, 1, 2),
        outlier_noise_choices=(-5, -4, 4, 5),
        offset_choices=(3, 4, 5),
        sign_tweak_choices=(-1, 0, 1),
    ),
    "hard": dict(
        noise_choices=(-2, -1, 0, 1, 2),
        outlier_noise_choices=(-6, -5, 5, 6),
        offset_choices=(1, 2),
        sign_tweak_choices=(-2, -1, 1, 2),
    ),
}


def _line_text(m: int, b: int) -> str:
    return "y = " + join_terms([(m, "x"), (b, "")])


def _ssr(points, m, b):
    """Sum of squared residuals, direct route."""
    return sum((y - (m * x + b)) ** 2 for x, y in points)


def _sad(points, m, b):
    """Sum of absolute residuals, direct route (generator-expression loop)."""
    return sum(abs(y - (m * x + b)) for x, y in points)


@generator(
    "p8c_line_best_fit_by_eye",
    level=12,
    presets={
        "easy": dict(variant="easy"),
        "medium": dict(variant="medium"),
        "hard": dict(variant="hard"),
    },
)
def p8c_line_best_fit_by_eye(rng, variant="easy"):
    """Which of four stated candidate lines best fits a stated integer
    scatter, judged by ssr AND sad against the stated points -- never by how
    a free-hand line happens to look."""
    if variant not in _TIER_PARAMS:
        raise ValueError(f"p8c_line_best_fit_by_eye: unknown variant {variant!r}")
    params = _TIER_PARAMS[variant]

    for _attempt in range(500):
        m0 = rng.choice(_M_CHOICES)
        b0 = rng.choice([v for v in range(-6, 7) if v != 0])
        outlier_idx = rng.randrange(len(_X_VALUES))
        points = []
        for i, x in enumerate(_X_VALUES):
            noise = (
                rng.choice(params["outlier_noise_choices"]) if i == outlier_idx else rng.choice(params["noise_choices"])
            )
            points.append((x, m0 * x + b0 + noise))

        # Candidate 0 (the answer): the actual generating line.
        # Candidate 1: wrong-sign gradient -- fits a line sloping the wrong way.
        m1 = -m0
        b1 = b0 + rng.choice(params["sign_tweak_choices"])
        # Candidate 2: right gradient, wrong intercept -- a parallel line
        # offset from the data.
        sign = rng.choice((-1, 1))
        m2 = m0
        b2 = b0 + sign * rng.choice(params["offset_choices"])
        # Candidate 3: overreacts to the single outlier point -- the line
        # forced exactly through the outlier and one anchor point, dragged
        # away from the rest of the data at the other points' expense.
        anchor_idx = 0 if outlier_idx != 0 else len(_X_VALUES) - 1
        xo, yo = points[outlier_idx]
        xa, ya = points[anchor_idx]
        m3 = round((yo - ya) / (xo - xa))
        if abs(m3) < 2:
            m3 = 2 if m3 >= 0 else -2
        b3 = yo - m3 * xo

        candidates = [(m0, b0), (m1, b1), (m2, b2), (m3, b3)]
        if len(set(candidates)) != 4:
            continue
        if any(m == 0 or b == 0 for m, b in candidates):
            continue
        texts = [_line_text(m, b) for m, b in candidates]
        if len(set(texts)) != 4:
            continue

        ssrs = [_ssr(points, m, b) for m, b in candidates]
        sads = [_sad(points, m, b) for m, b in candidates]
        if len(set(ssrs)) != 4 or len(set(sads)) != 4:
            continue

        ssr_order = sorted(range(4), key=lambda i: ssrs[i])
        sad_order = sorted(range(4), key=lambda i: sads[i])
        if ssr_order[0] != 0 or sad_order[0] != 0:
            continue  # the stated generating line must win under both metrics

        smallest_ssr, second_ssr = ssrs[0], ssrs[ssr_order[1]]
        smallest_sad, second_sad = sads[0], sads[sad_order[1]]
        # Margin rule, one per metric:
        # ssr (squared-y-unit, scales with the data range) -- relative:
        # the runner-up is at least 50% larger than the winner.
        if not (2 * second_ssr >= 3 * smallest_ssr):
            continue
        # sad (linear-y-unit, comparable scale to the data spread) --
        # absolute: at least 4 units clear.
        if not (second_sad - smallest_sad >= 4):
            continue
        break
    else:
        raise RuntimeError(
            "p8c_line_best_fit_by_eye: no draw produced a double-metric winner with margin in 500 attempts"
        )

    m_win, b_win = candidates[0]

    # Assert directly (not just by construction) that the winner is
    # strictly best under both metrics, with the required margin on both,
    # and that no two candidates ever tie on residual sum or equation text.
    check(all(ssrs[0] < ssrs[i] for i in (1, 2, 3)), "winner's ssr not strictly smallest")
    check(all(sads[0] < sads[i] for i in (1, 2, 3)), "winner's sad not strictly smallest")
    check(2 * second_ssr >= 3 * smallest_ssr, "ssr margin below the 50%-relative rule")
    check(second_sad - smallest_sad >= 4, "sad margin below the absolute-4 rule")
    check(ssr_order[0] == sad_order[0], "ssr and sad disagree on the winning candidate")
    check(len(set(ssrs)) == 4, "two candidates share a ssr total")
    check(len(set(sads)) == 4, "two candidates share a sad total")
    check(len(set(texts)) == 4, "two candidates share an equation string")

    # Independent check 1: recompute the winner's ssr via a distinct
    # algebraic identity (expand the square), never the direct residual
    # loop, and confirm the two routes agree.
    n = len(points)
    sum_x = sum(x for x, _ in points)
    sum_y = sum(y for _, y in points)
    sum_x2 = sum(x * x for x, _ in points)
    sum_y2 = sum(y * y for _, y in points)
    sum_xy = sum(x * y for x, y in points)
    ssr_algebraic = (
        sum_y2
        - 2 * m_win * sum_xy
        - 2 * b_win * sum_y
        + m_win * m_win * sum_x2
        + 2 * m_win * b_win * sum_x
        + b_win * b_win * n
    )
    check(ssr_algebraic == ssrs[0], f"ssr algebraic identity mismatch: {ssr_algebraic} != {ssrs[0]}")

    # Independent check 2: recompute the winner's sad via a second,
    # independently written route (explicit for-loop, not the generator
    # expression `_sad` used to construct it) and confirm they agree.
    sad_alt = 0
    for x, y in points:
        sad_alt += abs(y - (m_win * x + b_win))
    check(sad_alt == sads[0], f"sad alternate-loop mismatch: {sad_alt} != {sads[0]}")

    # The four candidates' in-text listing order is its own rng-shuffle,
    # independent of the mcq_options() shuffle that decides answer-option
    # order -- otherwise the winner's always being listed first would be a
    # free positional shortcut.
    display_order = list(range(4))
    rng.shuffle(display_order)
    lines_text = ", ".join(texts[i] for i in display_order)
    question_text = (
        "The scatter plot shows five data points. "
        f"Four candidate lines are proposed as models for the data: {lines_text}. "
        "Which line is the best fit for this data?"
    )
    ys = [y for _, y in points]
    figure = figures.cartesian(
        x_range=(0, max(_X_VALUES) + 1),
        y_range=(min(ys) - 1, max(ys) + 1),
        grid=1,
        equal_scale=False,
        points=[{"at": [x, y]} for x, y in points],
        alt="a scatter plot",
    )

    answer_text = texts[0]
    distractor_texts = [texts[1], texts[2], texts[3]]
    distractors = pick_distractors(answer_text, distractor_texts)

    working = [
        f"Sum of squared residuals: {answer_text} scores {smallest_ssr}; the next-best candidate scores {second_ssr}.",
        f"Sum of absolute residuals: {answer_text} scores {smallest_sad}; the next-best candidate scores {second_sad}.",
        f"{answer_text} has the smallest total residual under both measures, so it is the best fit.",
    ]

    return {
        "text": question_text,
        "latex": f"plain:{question_text}",
        "answer": answer_text,
        "answer_text": answer_text,
        "answer_latex": answer_text,
        "distractors": distractors,
        "working": working,
        "figure": figure,
    }


_EQ_RE = re.compile(r"y = (-?\d+)x ([+-]) (\d+)")


def _read_best_line(question) -> str:
    """Re-derive the best candidate from the drawn points and the equations in the prompt alone."""
    points = [tuple(p["at"]) for p in question.figure["points"]]
    best = None
    for m_str, sign, b_str in _EQ_RE.findall(question.text):
        m, b = int(m_str), int(b_str) * (1 if sign == "+" else -1)
        score = (_ssr(points, m, b), _sad(points, m, b))
        if best is None or score < best[0]:
            best = (score, _line_text(m, b))
    check(best is not None, "no candidate lines in the prompt")
    return best[1]


figures.register_reader("p8c_line_best_fit_by_eye", _read_best_line)
