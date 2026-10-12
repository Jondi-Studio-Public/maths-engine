"""Every generator returns or raises on small parameter values; none hangs.

A generator built backwards from its answer often redraws until a value passes a rule (nonzero,
distinct, a positive result). When a caller narrows a range so far that no value can pass, for
example `make("simultaneous", coef_hi=0)`, an unbounded redraw spins forever and logs nothing.
This test sets each integer keyword of each registered generator to 0, 1 and 2 in turn and
requires every call to finish within LIMIT_S: a question, or an exception (a ValueError naming
the parameter, or the VerificationError `core.draw_until` raises when its tries run out). It does
not judge which exception; it only rules out the hang.
"""

import inspect
import signal

import pytest

from mathgen import make
from mathgen.core import REGISTRY

LIMIT_S = 10  # a hang never returns; a slow but finite call stays far below this
VALUES = (0, 1, 2)


class _Hang(Exception):
    pass


def _int_params(fn):
    return [
        name
        for name, p in inspect.signature(fn).parameters.items()
        if isinstance(p.default, int) and not isinstance(p.default, bool)
    ]


TOPICS = sorted(t for t, (_level, fn) in REGISTRY.items() if _int_params(fn))


def _returns(topic: str, params: dict) -> bool:
    def on_alarm(_signum, _frame):
        raise _Hang

    previous = signal.signal(signal.SIGALRM, on_alarm)
    signal.setitimer(signal.ITIMER_REAL, LIMIT_S)
    try:
        make(topic, seed=0, **params)
    except _Hang:
        return False
    except Exception:  # noqa: BLE001 -- refusing bad parameters is fine; hanging is not
        pass
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    return True


@pytest.mark.skipif(not hasattr(signal, "setitimer"), reason="needs POSIX interval timers")
@pytest.mark.parametrize("topic", TOPICS)
def test_small_integer_parameters_never_hang(topic):
    fn = REGISTRY[topic][1]
    hung = [f"{name}={value}" for name in _int_params(fn) for value in VALUES if not _returns(topic, {name: value})]
    assert not hung, (
        f"make({topic!r}, ...) did not return within {LIMIT_S} s for {', '.join(hung)}. Raise ValueError "
        "for a parameter no draw can satisfy, and bound the redraw with mathgen.core.draw_until."
    )


@pytest.mark.skipif(not hasattr(signal, "setitimer"), reason="needs POSIX interval timers")
def test_fractions_subtraction_with_halves_only_does_not_hang():
    # Only 1/2 - 1/2 can be drawn, and the generator wants a positive difference.
    assert _returns("fractions", {"max_den": 2, "ops": ("-",)})
