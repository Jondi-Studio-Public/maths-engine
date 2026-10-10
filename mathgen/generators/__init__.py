"""Generator modules — the single registration point.

Importing this package imports every module in it, which runs the
`@generator(...)` decorators and populates `core.REGISTRY`. Dropping a new
`my_topic.py` in this directory is all that's needed; nothing else lists the
modules by name. Modules are imported in sorted order so registration (and
any duplicate-topic error) is deterministic.
"""

import importlib
import pkgutil

__all__ = sorted(m.name for m in pkgutil.iter_modules(__path__) if not m.name.startswith("_"))

for _name in __all__:
    importlib.import_module(f"{__name__}.{_name}")
del _name
