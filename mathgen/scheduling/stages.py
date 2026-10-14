"""Which of a skill's sources a review draws from at each stage.

A review ladder has `n_stages` numbered stages (0 to n_stages - 1) and then
maintenance, which callers pass as stage `n_stages`. A source lists the stages
it serves in `stages` (ints, "maintenance", or "exam"); no list means every
stage. Exam sources are reserved for exam-style tests and never serve a review.
A numbered stage past the end of the ladder counts as maintenance.
"""

from __future__ import annotations

# The public names: see docs/api.md. Anything else here may change in any release.
__all__ = [
    "MAINTENANCE",
    "EXAM",
    "is_exam_source",
    "non_exam_sources",
    "eligible_sources",
    "pick_sources",
]

MAINTENANCE = "maintenance"
EXAM = "exam"


def is_exam_source(src) -> bool:
    """True for a source reserved for the exam tier (`stages: [exam]`). Keyed on
    `stages`, the routing label, not `difficulty`, which is a generator preset."""
    return bool(src.stages) and EXAM in src.stages


def non_exam_sources(skill) -> list:
    """`skill.sources` without exam sources, in their original order."""
    return [src for src in skill.sources if not is_exam_source(src)]


def _label(stage: int, n_stages: int):
    return MAINTENANCE if stage >= n_stages else stage


def _serves(src, stage: int, n_stages: int) -> bool:
    if is_exam_source(src):
        return False
    if not src.stages:
        return True
    label = _label(stage, n_stages)
    return any(_label(s, n_stages) == label if isinstance(s, int) else s == label for s in src.stages)


def eligible_sources(skill, stage: int, n_stages: int) -> list:
    """The non-exam sources a review at `stage` draws from, in YAML order.

    When no source names `stage`, the sources of the nearest stage that has some
    are used, the easier side first on a tie. A skill whose ladder starts at
    stage 1 therefore serves its stage-1 sources at stage 0, rather than every
    source at once (which handed a new learner the hardest item on day one).
    Returns [] only when the skill has no non-exam source at all."""
    stage = min(max(stage, 0), n_stages)
    direct = [src for src in skill.sources if _serves(src, stage, n_stages)]
    if direct:
        return direct
    for distance in range(1, n_stages + 1):
        for nearby in (stage - distance, stage + distance):
            if 0 <= nearby <= n_stages:
                found = [src for src in skill.sources if _serves(src, nearby, n_stages)]
                if found:
                    return found
    return []


def pick_sources(sources: list, need: int, start: int = 0) -> list:
    """`need` sources taken round-robin from `sources`, beginning at index
    `start` (wrapped). With a varying start (a random one, say) every eligible
    source is served over time, even when there are more sources than questions."""
    if not sources or need <= 0:
        return []
    return [sources[(start + i) % len(sources)] for i in range(need)]
