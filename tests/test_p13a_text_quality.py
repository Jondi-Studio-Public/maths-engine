"""Tests the rendered text of the p13a questions for wording defects, sampling every tier explicitly.

`difficulty=tier` is passed explicitly on every call (never relying on a default variant): sampling the
default variant misses this class of defect entirely.
"""

import re

import pytest

from mathgen import make

SEEDS_PER_TIER = 200

# One (topic, [tiers]) pair per p13a skill -- topic names and declared tiers come from
# the `@generator(` decorators in mathgen/generators/p13a_*.py and each generator's own
# `presets={...}` dict.
_TOPICS_AND_TIERS = (
    ("p13a_bracketed_number", ("easy", "medium", "hard")),
    ("p13a_convert_rate_units", ("easy", "medium", "hard", "exam")),
    ("p13a_direct_proportion_scale", ("easy", "medium", "hard", "exam")),
    ("p13a_divide_ratio", ("easy", "medium", "hard", "exam")),
    ("p13a_polygon_angle_sum", ("easy", "medium", "hard", "exam")),
)

_SINGULAR_PLURAL_NOUN_RE = re.compile(r"\b1\s+(metres|litres|days|hours)\b")


def _offending_strings(text: str):
    return _SINGULAR_PLURAL_NOUN_RE.findall(text)


@pytest.mark.parametrize("topic,tiers", _TOPICS_AND_TIERS)
def test_no_singular_value_with_plural_noun(topic, tiers):
    offences = []
    for tier in tiers:
        for seed in range(SEEDS_PER_TIER):
            q = make(topic, seed=seed, difficulty=tier)
            fields = [("text", q.text), ("answer_text", q.answer_text)]
            for i, d in enumerate(q.distractors):
                fields.append((f"distractor[{i}]", d))
            for field_name, value in fields:
                if value is None:
                    continue
                for noun in _offending_strings(value):
                    offences.append((topic, tier, seed, field_name, noun, value))

    assert not offences, (
        f"{len(offences)} offending (singular count + plural noun) string(s) found "
        f"in topic {topic!r}:\n"
        + "\n".join(
            f"  tier={tier!r} seed={seed} field={field_name} noun={noun!r} full_text={value!r}"
            for (t, tier, seed, field_name, noun, value) in offences
        )
    )
