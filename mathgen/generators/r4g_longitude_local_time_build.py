"""Local clock time from longitude: `r4g_longitude_local_time` (easy, medium, hard).

``longitude-and-local-time`` (``senior-mathematics``, answer ``number``). Local time depends only on longitude:
15 degrees of longitude is 1 hour, and places further east are later. Places are fictional ports so no real time
zone is implied. Times are four-digit 24-hour numbers (1430 is 2:30 pm) kept between 1000 and 2359 on the same day.

* easy: the local time at a second port when both ports are on the same side of the prime meridian.
* medium: the same, with the ports on opposite sides of the prime meridian (the longitudes are added).
* hard: the local arrival time of a flight between two ports, given the departure time and the flight time.

The existing time-difference skill finds the size of the difference; this one applies it to a clock.
"""

from __future__ import annotations

from ..core import attempts, check, generator
from ._r4g_kit import clock_text, hhmm, minutes, single_number, retry

PORTS = ["Ardent", "Belmar", "Corvale", "Dunmoor", "Eskwith", "Farrow", "Garrick", "Halloran", "Ilsby", "Jorvik"]
RULE = "Local time depends only on longitude: it is 1 hour later for every 15° further east."


def lon_text(deg: int) -> str:
    return f"{abs(deg)}°{'E' if deg >= 0 else 'W'}"


def shift_minutes(lon_from: int, lon_to: int) -> int:
    """Minutes by which the clock at `lon_to` is ahead of the clock at `lon_from` (15° is 60 minutes)."""
    check((lon_to - lon_from) % 15 == 0, "longitudes must differ by a multiple of 15")
    return (lon_to - lon_from) * 4


def local_time(minute_of_day: int, lon_from: int, lon_to: int) -> int:
    return minute_of_day + shift_minutes(lon_from, lon_to)


def draw_places(rng, *, opposite: bool):
    for _ in attempts(500, "two ports with a clean time difference"):
        a, b = rng.sample(PORTS, 2)
        if opposite:
            east, west = rng.randrange(15, 106, 15), rng.randrange(15, 106, 15)
            la, lb = (east, -west) if rng.random() < 0.5 else (-west, east)
        else:
            sign = rng.choice((1, -1))
            la, lb = sign * rng.randrange(15, 166, 15), sign * rng.randrange(15, 166, 15)
        if la == lb or abs(la - lb) > 90 + (30 if opposite else 0) or abs(la - lb) < 30:
            continue
        return a, b, la, lb
    raise AssertionError("unreachable")


def draw_clock(rng, lo=1000, hi=2359):
    return rng.randrange(lo // 100 * 60, hi // 100 * 60 + 1, 5)


def wrap(total_minutes) -> int:
    """A clock time as a four-digit 24-hour number, wrapped into one day."""
    return hhmm(int(total_minutes) % 1440)


def _fits(t):
    return 1000 <= hhmm(t) <= 2359 and 600 <= t < 1440


def _ports_text(a, b, la, lb):
    return f"Port {a} is at longitude {lon_text(la)} and Port {b} is at longitude {lon_text(lb)}."


def _clock_question(rng, opposite):
    for _ in attempts(2000, "a clock time that stays inside the day"):
        a, b, la, lb = draw_places(rng, opposite=opposite)
        t = draw_clock(rng, 1000, 2000)
        ans = local_time(t, la, lb)
        if _fits(ans) and ans != t:
            break
    text = (
        f"{_ports_text(a, b, la, lb)} {RULE} When the local time at Port {a} is {clock_text(t)}, "
        f"what is the local time at Port {b}? Give the time in 24-hour form as a four-digit number, such as 1342."
    )
    hours = abs(lb - la) // 15
    sign = 1 if lb > la else -1
    same_side_sum = (abs(la) + abs(lb)) // 15
    other = local_time(t, lb, la)  # direction reversed
    slips = [
        (wrap(other), "r4g-east-west-direction-reversed"),
        (wrap(t + sign * same_side_sum * 60), "r4g-longitudes-added-same-side")
        if not opposite
        else (wrap(t + sign * abs(abs(la) - abs(lb)) // 15 * 60), "r4g-longitudes-subtracted-opposite-sides"),
        (wrap(t), "r4g-no-time-shift"),
        (wrap(t + sign * (abs(lb - la) // 30) * 60), "r4g-half-the-hours"),
    ]
    slips = [(v, tag) for v, tag in slips if 1000 <= v <= 2359]
    direction = "east" if sign > 0 else "west"
    working = [
        f"The longitudes differ by {abs(lb - la)}°, which is {abs(lb - la)} / 15 = {hours} hours.",
        f"Port {b} is to the {direction} of Port {a}, so its clock is {hours} hours {'ahead' if sign > 0 else 'behind'}: {clock_text(ans)}.",
    ]
    check(minutes(hhmm(ans)) == ans, "answer is not a valid clock time")
    return single_number(text, hhmm(ans), 0, working, slips)


def _easy(rng):
    return _clock_question(rng, opposite=False)


def _medium(rng):
    return _clock_question(rng, opposite=True)


def _hard(rng):
    for _ in attempts(3000, "a flight whose local arrival stays inside the day"):
        a, b, la, lb = draw_places(rng, opposite=rng.random() < 0.5)
        t = draw_clock(rng, 1000, 1900)
        flight = rng.randrange(120, 721, 10)
        ans = t + flight + shift_minutes(la, lb)
        if _fits(ans) and abs(lb - la) >= 45:
            break
    fh, fm = divmod(flight, 60)
    flight_text = f"{fh} hours" if fm == 0 else f"{fh} hours {fm} minutes"
    text = (
        f"{_ports_text(a, b, la, lb)} {RULE} A flight leaves Port {a} at {clock_text(t)} local time and takes "
        f"{flight_text}. What is the local time at Port {b} when it lands? "
        f"Give the time in 24-hour form as a four-digit number, such as 1342."
    )
    shift = shift_minutes(la, lb)
    slips = [
        (wrap(t + flight - shift), "r4g-east-west-direction-reversed"),
        (wrap(t + flight), "r4g-no-time-shift"),
        (wrap(t + shift), "r4g-flight-time-left-out"),
        (wrap(t - flight + shift), "r4g-flight-time-subtracted"),
        (wrap(t + flight + shift + 60 * (1 if shift > 0 else -1)), "r4g-time-zone-off-by-one-hour"),
    ]
    slips = [(v, tag) for v, tag in slips if 1000 <= v <= 2359]
    working = [
        f"The flight lands at {clock_text(t + flight)} Port {a} time.",
        f"The longitudes differ by {abs(lb - la)}°, so Port {b}'s clock is {abs(shift) // 60} hours {'ahead' if shift > 0 else 'behind'}: {clock_text(ans)}.",
    ]
    return single_number(text, hhmm(ans), 0, working, slips)


_TIER = {"easy": _easy, "medium": _medium, "hard": _hard}


@generator(
    "r4g_longitude_local_time",
    level=11,
    presets={"easy": {"tier": "easy"}, "medium": {"tier": "medium"}, "hard": {"tier": "hard"}},
)
def r4g_longitude_local_time(rng, tier="easy"):
    return retry(_TIER[tier], rng)
