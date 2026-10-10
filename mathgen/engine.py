"""The small public API for working with skills: load, generate a question, grade an answer.

    from mathgen import engine

    skills = {s.id: s for s in engine.load_skills()[1]}
    skill = skills["simplify-surds"]
    q = engine.make_for_skill(skill, seed=7)
    print(q.text)                              # Simplify √756. Give your answer in the form a√b.
    print(q.answer_text)                       # 6*sqrt(21)
    print(engine.grade(skill, q, "6√21").correct)   # True
    print(engine.grade(skill, q, "2√3").correct)    # False

A service that grades replies from people it does not trust should call `grade_with_deadline` instead: same
arguments, but a reply that takes the grader longer than a time limit is graded wrong rather than hanging the call.

Nothing here touches a database, the network or any app code.

For a multiple-choice skill (`skill.answer == "mcq"`), `grade` takes the 0-based index of the chosen option in
`mathgen.core.mcq_options(q)`, the seed-shuffled list `[answer, *distractors]`; the command line shows the
options and accepts an option's label or its text instead.
"""

from __future__ import annotations

from . import make
from .grading import GradeResult, grade as _grade_kind
from .skills import Domain, Skill, Source, load_skills  # noqa: F401  (re-exported)


def _tuplify(params: dict | None) -> dict:
    """Generators expect e.g. `kinds=(...)`; YAML and JSON only have lists."""
    return {k: (tuple(v) if isinstance(v, list) else v) for k, v in (params or {}).items()}


def pick_source(skill: Skill, source: Source | int | str | None = None) -> Source:
    """A skill's source: a `Source`, an index into `skill.sources`, a difficulty label, or the first."""
    if isinstance(source, Source):
        return source
    if not skill.sources:
        raise ValueError(f"skill {skill.id!r} has no question sources (status: {skill.status})")
    if source is None:
        return skill.sources[0]
    if isinstance(source, int):
        return skill.sources[source]
    for src in skill.sources:
        if src.difficulty == source:
            return src
    labels = sorted({src.difficulty for src in skill.sources if src.difficulty})
    raise KeyError(
        f"skill {skill.id!r} has no source with difficulty {source!r}"
        + (f"; its difficulties are {', '.join(labels)}" if labels else "; none of its sources is labelled")
    )


def make_for_skill(skill: Skill, seed: int | None = None, *, source: Source | int | str | None = None):
    """Build one verified question for `skill`. Same (skill, source, seed) gives the same question."""
    src = pick_source(skill, source)
    return make(src.topic, seed=seed, difficulty=src.difficulty, **_tuplify(src.params))


def grade(skill: Skill, question, given: str) -> GradeResult:
    """Grade `given` against `question` with the grader the skill's answer kind names."""
    return _grade_kind(skill.answer, question, given, skill=skill)


def grade_with_deadline(skill: Skill, question, given: str, *, timeout: float | None = None) -> GradeResult:
    """`grade`, in a worker process that is stopped after `timeout` seconds (2 by default); a reply that runs out of
    time is graded wrong. Use this for replies from people you do not trust: see `mathgen.grading.deadline`."""
    from .grading import deadline

    limit = deadline.DEFAULT_TIMEOUT if timeout is None else timeout
    return deadline.grade_with_deadline(skill.answer, question, given, skill=skill, timeout=limit)
