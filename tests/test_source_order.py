"""Source-order invariant across the real skill tree.

A `ladder`/`cycle`-style level walks a skill's `sources` in YAML order and expects difficulty to be
non-decreasing across that order, for any skill whose sources are all labelled with one of the three ordinary
difficulty tiers. This is a static, load-bearing property of the skill library itself, not a generator
behaviour, so it belongs in its own test module rather than a generator test module.

A skill using any difficulty label outside `{"easy", "medium", "hard"}`
(including a source with no `difficulty` at all, i.e. `None`) is exempt and not constrained by this invariant.

The test loads the real tree and applies the exemption rule (every source's difficulty in {easy, medium, hard},
otherwise exempt) to every loaded skill.
"""

from __future__ import annotations

from pathlib import Path

from mathgen.skills import load_skills

REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_SKILLS = REPO_ROOT / "skills"

_ORDINAL = {"easy": 0, "medium": 1, "hard": 2, "exam": 3}


def test_source_difficulty_order_is_non_decreasing_for_ordinary_skills():
    # Pass the real-tree root explicitly rather than relying on
    # MATHS_SKILLS_DIR being unset -- self-verifying regardless of what
    # other fixtures in this directory do with that env var.
    _, skills = load_skills(PRODUCTION_SKILLS)

    # Confirm this really is the production tree, not some fixture that
    # happens to load without error (a `sine-rule` sanity check, and a
    # population-size check on the constrained set below).
    assert any(s.id == "sine-rule" for s in skills), (
        "sine-rule not found -- load_skills(PRODUCTION_SKILLS) did not load the real skill tree"
    )

    violations = []
    constrained_ids = set()
    for skill in skills:
        difficulties = [src.difficulty for src in skill.sources]
        if not difficulties:
            continue
        if any(d not in _ORDINAL for d in difficulties):
            continue  # exempt: uses a label outside easy/medium/hard (or None)
        constrained_ids.add(skill.id)
        ordinals = [_ORDINAL[d] for d in difficulties]
        if ordinals != sorted(ordinals):
            violations.append((skill.id, difficulties))

    # Guard against a loader regression that silently exempted everything
    # (e.g. every `difficulty` coming back `None`), which would make the
    # assertion below pass vacuously: `add-within-10` itself uses only the
    # standard labels, so it must land in the constrained set, and the
    # constrained set as a whole must not be empty or suspiciously tiny.
    # (An exam source ranks after hard, so a skill with one is checked too: it must come last.)
    assert "add-within-10" in constrained_ids, (
        "add-within-10 was exempted from the order check -- its sources should "
        "all carry easy/medium/hard labels; a loader regression may be "
        "dropping difficulty values"
    )
    assert len(constrained_ids) >= 300, (
        f"only {len(constrained_ids)} skills had all-ordinary-tier sources -- "
        f"suspiciously low, a loader regression may be exempting skills it "
        f"shouldn't"
    )

    assert violations == [], "skills whose sources are not in non-decreasing difficulty order: " + ", ".join(
        f"{sid} {seq}" for sid, seq in violations
    )
