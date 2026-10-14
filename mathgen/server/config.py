"""Server settings, read from environment variables so a container is configured without files."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path

from .. import service
from .pool import default_workers

ROLES = ("author", "learner")


def _float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value in (None, "") else float(value)


def _int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value in (None, "") else int(value)


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def parse_keys(text: str) -> list[tuple[str, str]]:
    """`role:key,role:key` -> [(key, role), ...]. Roles: `author` (may see answers) and `learner` (may not).

    Text with something in it must give at least one key: a stray `,` is a typo, and an empty key list would
    otherwise be read as "no authentication"."""
    keys = []
    for entry in (e.strip() for e in text.split(",")):
        if not entry:
            continue
        role, sep, key = entry.partition(":")
        if not sep or role not in ROLES or len(key) < 16:
            raise ValueError("MATHS_API_KEYS entries are `author:<key>` or `learner:<key>`, keys 16+ characters")
        keys.append((key, role))
    if text.strip() and not keys:
        raise ValueError("MATHS_API_KEYS is set but holds no key; write `author:<key>,learner:<key>` or unset it")
    return keys


@dataclass
class Config:
    """See `docs/server.md` for each setting's environment variable."""

    keys: list[tuple[str, str]] = field(default_factory=list)  # (key, role)
    #: True (MATHS_API_OPEN=1) lets the server start with no keys at all. Without it, no keys is a refusal to start.
    open: bool = False
    open_role: str = "learner"  # the role every caller has when no keys are set
    workers: int = field(default_factory=default_workers)
    timeout: float = 5.0  # seconds for one question or one grade
    paper_timeout: float = 60.0
    queue_timeout: float = 30.0  # seconds a call may wait for a free worker before a 503
    #: One total deadline for a whole grade/batch call. None: `timeout * 10`.
    batch_timeout: float | None = None
    #: Requests running or queued at once; more get a 503. None: twice the workers.
    max_in_flight: int | None = None
    #: After `cooldown_strikes` timeouts on one (skill, source) within `cooldown` seconds, further calls on it
    #: fail fast with a 503 for `cooldown` seconds, rather than each killing another worker.
    cooldown: float = 30.0
    cooldown_strikes: int = 3
    #: Address-space cap per worker process in MB (POSIX only). 0: no cap.
    worker_memory_mb: int = 1024
    max_body_bytes: int = 256 * 1024
    cors_origins: list[str] = field(default_factory=list)
    #: Optional secret that makes question ids unforgeable; 16+ characters. See `service.Library.id_secret`.
    id_secret: str | None = field(default=None, repr=False)
    #: Serve /docs, /redoc and /openapi.json (MATHS_API_DOCS=1). They are open to anyone who can reach the server.
    docs: bool = False
    # None: the library's own (MATHS_SKILLS_DIR and MATHS_EXTRA_SKILLS_DIRS still apply, see mathgen/skills.py).
    skills_dir: Path | None = None
    extra_skills_dirs: list[Path] | None = None

    def __post_init__(self):
        if self.open_role not in ROLES:
            raise ValueError(f"open_role must be one of {ROLES}")
        if not isinstance(self.workers, int) or self.workers < 1:
            raise ValueError("workers must be a whole number, 1 or more")
        for name in ("timeout", "paper_timeout", "queue_timeout", "cooldown"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite number of seconds above 0, got {value!r}")
        if self.batch_timeout is None:
            self.batch_timeout = self.timeout * 10
        if not math.isfinite(self.batch_timeout) or self.batch_timeout <= 0:
            raise ValueError(f"batch_timeout must be a finite number of seconds above 0, got {self.batch_timeout!r}")
        if self.max_in_flight is None:
            self.max_in_flight = self.workers * 2
        if self.max_in_flight < 1 or self.cooldown_strikes < 1 or self.max_body_bytes < 1:
            raise ValueError("max_in_flight, cooldown_strikes and max_body_bytes must be 1 or more")
        if self.worker_memory_mb < 0:
            raise ValueError("worker_memory_mb must be 0 (no cap) or more")
        if self.id_secret is not None and len(self.id_secret) < service.MIN_ID_SECRET_CHARS:
            raise ValueError(f"MATHS_API_ID_SECRET must be at least {service.MIN_ID_SECRET_CHARS} characters")

    def check_auth(self):
        """Refuse a server nobody can authenticate to, unless it was asked for (`open`)."""
        if not self.keys and not self.open:
            raise ValueError(
                "no API keys are set. Set MATHS_API_KEYS=author:<key>,learner:<key> (keys 16+ characters), "
                "or MATHS_API_OPEN=1 to serve every caller as role "
                f"{self.open_role!r} with no key"
            )

    @classmethod
    def from_env(cls) -> Config:
        workers = _int("MATHS_API_WORKERS", default_workers())
        timeout = _float("MATHS_API_TIMEOUT", 5.0)
        batch = os.environ.get("MATHS_API_BATCH_TIMEOUT")
        in_flight = os.environ.get("MATHS_API_MAX_IN_FLIGHT")
        return cls(
            keys=parse_keys(os.environ.get("MATHS_API_KEYS", "")),
            open=_flag("MATHS_API_OPEN"),
            open_role=os.environ.get("MATHS_API_OPEN_ROLE", "learner"),
            workers=workers,
            timeout=timeout,
            paper_timeout=_float("MATHS_API_PAPER_TIMEOUT", 60.0),
            queue_timeout=_float("MATHS_API_QUEUE_TIMEOUT", 30.0),
            batch_timeout=float(batch) if batch else None,
            max_in_flight=int(in_flight) if in_flight else None,
            cooldown=_float("MATHS_API_COOLDOWN", 30.0),
            cooldown_strikes=_int("MATHS_API_COOLDOWN_STRIKES", 3),
            worker_memory_mb=_int("MATHS_API_WORKER_MEMORY_MB", 1024),
            max_body_bytes=_int("MATHS_API_MAX_BODY", 256 * 1024),
            cors_origins=[o.strip() for o in os.environ.get("MATHS_API_CORS_ORIGINS", "").split(",") if o.strip()],
            id_secret=os.environ.get("MATHS_API_ID_SECRET") or None,
            docs=_flag("MATHS_API_DOCS"),
        )
