"""Where the skill library lives on disk.

Two layouts exist:

* a repository checkout (or an editable install of one) keeps the library in ``skills/`` next to the
  ``mathgen/`` package;
* an installed wheel carries the same files inside the package as ``mathgen/_skills/`` (``pyproject.toml``
  maps that package onto ``skills/``, so there is one copy in the repository).

A ``skills/`` folder beside the imported ``mathgen`` package wins over the in-package copy. So Python
sees edits to ``skills/`` whenever it imports ``mathgen`` from a clone (running from the clone's root,
or after ``pip install -e .``). A plain ``pip install .`` imports the installed copy instead, which holds
the library as it was at install time; reinstall after editing ``skills/``.
"""

from __future__ import annotations

import atexit
from contextlib import ExitStack
from importlib import resources
from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent
# Name of the in-package copy; it must match `package-dir` and `packages` in pyproject.toml.
PACKAGED_DIRNAME = "_skills"
# A real library has this file (the domain list); it keeps an unrelated `skills/` folder that
# happens to sit next to a site-packages install from being mistaken for the library.
_MARKER = "domains.yaml"


def checkout_skills_dir() -> Path:
    """``skills/`` beside the package, as in a repository checkout."""
    return _PACKAGE_DIR.parent / "skills"


def packaged_skills_dir() -> Path | None:
    """The ``mathgen/_skills`` folder of an installed wheel, or None when this is not a wheel install."""
    ref = resources.files("mathgen") / PACKAGED_DIRNAME
    if not ref.is_dir():
        return None
    if isinstance(ref, Path):
        return ref
    # Not a plain directory on disk (a zipped install): unpack once, kept for the life of the process.
    stack = ExitStack()
    atexit.register(stack.close)
    return Path(stack.enter_context(resources.as_file(ref)))


def default_skills_dir() -> Path:
    """The library to use when ``MATHS_SKILLS_DIR`` is not set.

    If neither layout exists, the checkout path comes back, so an error can say where it looked.
    """
    checkout = checkout_skills_dir()
    if (checkout / _MARKER).is_file():
        return checkout
    packaged = packaged_skills_dir()
    if packaged is not None:
        return packaged
    return checkout
