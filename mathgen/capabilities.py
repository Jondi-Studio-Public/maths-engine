"""Capability-gate registry loader and checker.

Gates are first-class, named, owned dependencies with a pass/fail state (see
``skills/capability-gates.yaml``). Any skill citing an unpassed or unknown gate must be blocked from going
live -- fail closed.

Follows the load style of ``mathgen/skills.py`` (``skills_root()``): a
``MATHS_CAPABILITY_GATES_FILE`` environment override is supported but not
required.

Nothing here talks to the database or HTTP layer.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from ._paths import default_skills_dir

# Lives with the skills: in `skills/` of a checkout, or inside the installed package.
DEFAULT_CAPABILITY_GATES_FILE = default_skills_dir() / "capability-gates.yaml"

# Spellings of a gate id that a skill file may use, mapped to the registry id (see `canonical_gate_id`).
GATE_ALIASES = {"technology": "technology_use", "visual": "visual_construction"}


def canonical_gate_id(raw: str) -> str:
    """Resolve a gate id as spelled in a skill file to its canonical registry id.

    Unrecognized ids pass through unchanged (the caller decides whether an
    unrecognized id is an error -- this function only resolves aliases).
    """
    return GATE_ALIASES.get(raw, raw)


def capability_gates_file() -> Path:
    return Path(os.environ.get("MATHS_CAPABILITY_GATES_FILE", str(DEFAULT_CAPABILITY_GATES_FILE)))


def load_capability_gates(root: Path | None = None) -> dict[str, dict]:
    """Parse the capability-gate registry YAML into a dict keyed by gate id.

    ``root``, if given, is the path to the registry YAML file directly
    (not a directory), matching the single-file nature of the registry.
    """
    path = Path(root) if root else capability_gates_file()
    if not root and not path.exists():
        # No registry at this path (say, a wrong MATHS_CAPABILITY_GATES_FILE): every gate is unknown, so gated
        # skills fail closed in the validator instead of the loader raising.
        return {}
    with open(path, encoding="utf-8") as fh:
        rows = yaml.safe_load(fh) or []

    registry: dict[str, dict] = {}
    for row in rows:
        gate_id = row.get("id")
        if not gate_id:
            continue
        registry[gate_id] = row
    return registry


def gate_passed(gate_id: str, registry: dict[str, dict]) -> bool:
    """True only if ``gate_id`` is a known registry entry with status "passed".

    Never raises for an unknown or malformed gate id -- fail closed.
    """
    entry = registry.get(gate_id)
    if not isinstance(entry, dict):
        return False
    return entry.get("status") == "passed"


def skill_gates_satisfied(gate_ids: list[str], registry: dict[str, dict]) -> bool:
    """True only if every id in ``gate_ids`` passes ``gate_passed``.

    An empty list is vacuously True (no gates required).
    """
    return all(gate_passed(gate_id, registry) for gate_id in gate_ids)
