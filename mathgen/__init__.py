"""mathgen mass-generates maths questions with verified answers."""

import logging
from importlib import metadata as _metadata

# A library never prints its own log records: without a handler on the "mathgen" logger, Python's last-resort
# handler writes any warning or error from `mathgen.*` to stderr (a traceback for a reply that could not be
# graded, say). An application that wants them attaches its own handler or configures the root logger, and
# they reach it as usual.
logging.getLogger("mathgen").addHandler(logging.NullHandler())

from .core import Question, REGISTRY, make, topics  # noqa: E402

# Importing the package imports every generator module, registering each topic.
from . import generators  # noqa: E402,F401


def _version() -> str:
    """The installed distribution's version; a checkout that was never installed has none to report."""
    try:
        return _metadata.version("maths-engine")
    except _metadata.PackageNotFoundError:
        return "0+unknown"


__version__ = _version()

__all__ = ["Question", "REGISTRY", "make", "topics", "__version__"]
