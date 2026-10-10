"""Regression test that the medium and hard tiers of `pythagorean-hypotenuse`, `pythagorean-leg` and `right-
triangle-angle` no longer collide.

`pythagorean-hypotenuse`, `pythagorean-leg` and `right-triangle-angle` each
pin the `trigonometry` topic's `kinds` param to a single value in their
`sources`, but before this fix the generator's `medium` and `hard`
presets both drew `k` from `randint(1, 3)` -- once `kinds` is pinned, medium
and hard sample the identical 18-outcome space (6 `TRIPLES` x k in 1..3),
independently seeded, so they could render the exact same question text at
the ~1/18 birthday-paradox rate. Confirmed empirically before the fix: 4/60
collisions per skill at seeds {0, 16, 27, 44}.

This test reads the topic/params for each skill's own `medium` and `hard`
`sources` entries off the loaded skill (never hardcoded), so a future
skill-YAML change cannot silently desync the test from what the skill
actually renders.
"""

import pytest

from mathgen.skills import load_skills
from mathgen import make

SKILL_IDS = ["pythagorean-hypotenuse", "pythagorean-leg", "right-triangle-angle"]
SEEDS = range(60)


def _source_for(skill, difficulty):
    matches = [s for s in skill.sources if s.difficulty == difficulty]
    assert len(matches) == 1, f"{skill.id}: expected exactly one {difficulty!r} source, found {len(matches)}"
    return matches[0]


@pytest.mark.parametrize("skill_id", SKILL_IDS)
def test_medium_hard_never_collide(skill_id):
    _, skills = load_skills()
    skill = next(s for s in skills if s.id == skill_id)
    medium_source = _source_for(skill, "medium")
    hard_source = _source_for(skill, "hard")

    collisions = []
    for seed in SEEDS:
        medium_text = make(medium_source.topic, seed=seed, difficulty="medium", **medium_source.params).text
        hard_text = make(hard_source.topic, seed=seed, difficulty="hard", **hard_source.params).text
        if medium_text == hard_text:
            collisions.append(seed)

    assert collisions == [], f"{skill_id}: {len(collisions)}/{len(SEEDS)} medium/hard collisions at seeds {collisions}"
