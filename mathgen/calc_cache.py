"""Per-process memo of the slow SymPy calculus calls the generators make as an independent check.

A generator that builds its answer from a known antiderivative then calls ``integrate`` (or ``dsolve``) a second
time, on purpose, to prove the answer. Those calls are pure functions of their arguments and the argument space
of a generator is small, so across seeds the very same call recurs. These wrappers compute each distinct call once
per process and return the same (immutable) SymPy result afterwards. The check itself is unchanged: it still runs
SymPy's own routine on the original arguments the first time it sees them.

The key is ``srepr`` of the arguments, not the arguments themselves, so ``1`` and ``1.0`` (equal and equally hashed
in Python, different to SymPy) and two symbols that differ only in their assumptions never share an entry. A call
whose arguments cannot be keyed, that passes keyword options, or that raises is simply passed through uncached.
"""

from __future__ import annotations

from collections import OrderedDict
from threading import Lock

import sympy

_MAX = 4096
_LOCK = Lock()


class _Memo:
    def __init__(self, fn, name: str) -> None:
        self._fn = fn
        self._name = name
        self._data: OrderedDict[str, object] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def __call__(self, *args, **kwargs):
        if kwargs:
            return self._fn(*args, **kwargs)
        try:
            key = sympy.srepr(args)
        except Exception:  # an argument SymPy cannot print: just compute it
            return self._fn(*args)
        with _LOCK:
            if key in self._data:
                self._data.move_to_end(key)
                self.hits += 1
                return self._data[key]
        value = self._fn(*args)
        with _LOCK:
            self.misses += 1
            self._data[key] = value
            if len(self._data) > _MAX:
                self._data.popitem(last=False)
        return value

    def cache_clear(self) -> None:
        with _LOCK:
            self._data.clear()
            self.hits = self.misses = 0


integrate = _Memo(sympy.integrate, "integrate")
dsolve = _Memo(sympy.dsolve, "dsolve")
