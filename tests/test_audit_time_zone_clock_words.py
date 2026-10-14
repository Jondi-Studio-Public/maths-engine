"""am/pm in the time-zone working must follow the clock when a time runs past midnight."""

from __future__ import annotations

import re

import pytest

from mathgen import make

TOPICS = ["exd_x_sc6_travel_zones", "exd_x_sc6_longitude", "exd_x_sc6_aus_zones", "exd_x_sc6_call_window"]


def _minutes(text):
    h, m, ap = re.match(r"(\d+):(\d+) (am|pm)", text).groups()
    return (int(h) % 12 + (12 if ap == "pm" else 0)) * 60 + int(m)


@pytest.mark.parametrize("seed", range(100))
def test_travel_working_landing_time_on_origin_clock(seed):
    q = make("exd_x_sc6_travel_zones", seed=seed)
    dep = re.search(r"leaves \w[\w ]* at (\d+:\d+ [ap]m)", q.stem).group(1)
    dur = float(re.search(r"takes ([\d.]+) hours", q.stem).group(1))
    shown = re.match(r"Landing time on .*? clock = (\d+:\d+ [ap]m);", q.parts[0].working[0]).group(1)
    assert _minutes(shown) == round(_minutes(dep) + dur * 60) % 1440


@pytest.mark.parametrize("topic", TOPICS)
@pytest.mark.parametrize("seed", range(40))
def test_every_clock_in_working_is_a_real_time(topic, seed):
    q = make(topic, seed=seed)
    for p in q.parts:
        for line in p.working:
            for h, m in re.findall(r"(\d+):(\d+) [ap]m", line):
                assert 1 <= int(h) <= 12 and int(m) < 60
