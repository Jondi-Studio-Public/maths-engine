"""The prerequisite graph: what a skill builds on, and what a learner is ready for.

Prerequisites are advice, not gates. These helpers say which skills a learner's
mastered set leaves ready to start and which earlier skills a struggling learner
could revisit; nothing here stops a host serving any skill. Skills are the
loader's `Skill` objects (anything with `id`, `status` and `prerequisites`).
Deferred skills are never offered, and the validator keeps an active skill from
depending on one.
"""

from __future__ import annotations

from collections.abc import Iterable


def _by_id(skills) -> dict:
    return skills if isinstance(skills, dict) else {s.id: s for s in skills}


def prerequisite_closure(skill_id: str, skills) -> set[str]:
    """Every skill `skill_id` builds on, directly or through its own
    prerequisites; not the skill itself. `skills` is a list of skills or a dict
    keyed by id. Raises KeyError for an unknown id."""
    by_id = _by_id(skills)
    seen: set[str] = set()
    stack = list(by_id[skill_id].prerequisites)
    while stack:
        pre = stack.pop()
        if pre in seen or pre == skill_id:
            continue
        seen.add(pre)
        stack.extend(by_id[pre].prerequisites)
    return seen


def ready_to_start(skills, mastered: Iterable[str]) -> list:
    """The active skills not yet mastered whose direct prerequisites are all
    mastered, in the order `skills` gives them. A skill with no prerequisites is
    always ready."""
    by_id = _by_id(skills)
    done = set(mastered)
    return [
        s
        for s in by_id.values()
        if s.status == "active" and s.id not in done and all(p in done for p in s.prerequisites)
    ]


def gaps_before(skill_id: str, skills, mastered: Iterable[str]) -> list[str]:
    """The active skills in `skill_id`'s prerequisite closure that are not
    mastered, earliest first: a skill comes after everything it builds on, and
    ties keep id order. What a learner stuck on `skill_id` could revisit."""
    by_id = _by_id(skills)
    done = set(mastered)
    todo = {sid for sid in prerequisite_closure(skill_id, by_id) if sid not in done and by_id[sid].status == "active"}
    ordered: list[str] = []
    placed: set[str] = set()
    while todo:
        ready = sorted(sid for sid in todo if all(p in placed or p not in todo for p in by_id[sid].prerequisites))
        if not ready:
            raise ValueError(f"prerequisite cycle among {sorted(todo)}")
        ordered.extend(ready)
        placed.update(ready)
        todo.difference_update(ready)
    return ordered
