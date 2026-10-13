"""Server settings, read from environment variables so a container is configured without files."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .pool import default_workers

ROLES = ("author", "learner")


def _float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value in (None, "") else float(value)


def _int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value in (None, "") else int(value)


def parse_keys(text: str) -> list[tuple[str, str]]:
    """`role:key,role:key` -> [(key, role), ...]. Roles: `author` (may see answers) and `learner` (may not)."""
    keys = []
    for entry in (e.strip() for e in text.split(",")):
        if not entry:
            continue
        role, sep, key = entry.partition(":")
        if not sep or role not in ROLES or len(key) < 16:
            raise ValueError(f"MATHS_API_KEYS entries are `author:<key>` or `learner:<key>`, keys 16+ characters")
        keys.append((key, role))
    return keys


@dataclass
class Config:
    """See `docs/server.md` for each setting's environment variable."""

    keys: list[tuple[str, str]] = field(default_factory=list)  # (key, role)
    open_role: str = "author"  # the role every caller has when no keys are set
    workers: int = field(default_factory=default_workers)
    timeout: float = 5.0  # seconds for one question or one grade
    paper_timeout: float = 60.0
    queue_timeout: float = 30.0  # seconds a call may wait for a free worker before a 503
    max_body_bytes: int = 256 * 1024
    cors_origins: list[str] = field(default_factory=list)
    # None: the library's own (MATHS_SKILLS_DIR and MATHS_EXTRA_SKILLS_DIRS still apply, see mathgen/skills.py).
    skills_dir: Path | None = None
    extra_skills_dirs: list[Path] | None = None

    @classmethod
    def from_env(cls) -> Config:
        open_role = os.environ.get("MATHS_API_OPEN_ROLE", "author")
        if open_role not in ROLES:
            raise ValueError(f"MATHS_API_OPEN_ROLE must be one of {ROLES}")
        return cls(
            keys=parse_keys(os.environ.get("MATHS_API_KEYS", "")),
            open_role=open_role,
            workers=_int("MATHS_API_WORKERS", default_workers()),
            timeout=_float("MATHS_API_TIMEOUT", 5.0),
            paper_timeout=_float("MATHS_API_PAPER_TIMEOUT", 60.0),
            queue_timeout=_float("MATHS_API_QUEUE_TIMEOUT", 30.0),
            max_body_bytes=_int("MATHS_API_MAX_BODY", 256 * 1024),
            cors_origins=[o.strip() for o in os.environ.get("MATHS_API_CORS_ORIGINS", "").split(",") if o.strip()],
        )
