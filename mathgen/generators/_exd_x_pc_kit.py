"""Small shared builder for the whole-number exam chains in the ``exd_x_pc1_*`` and ``exd_x_tg1_*`` modules.

Not a generator module (the leading underscore keeps ``mathgen.generators`` from importing it as one).
"""

from __future__ import annotations

from ..answers import Distractor
from ..core import GRADER_TOLERANCE, latex_answer, pick_distractors
from ._spread_kit import mixed


def count_part(prompt, key, marks, working, wrong, *, follow=None, want=3, signed=False, hi=None, exclude=()):
    """A ``number`` part whose key is a positive whole number (any non-zero whole number when ``signed``).

    The caller's generator must be wrapped in ``uses_rng`` so the option split varies by seed.
    ``wrong`` lists ``(value, misconception id)`` in priority order. A value that is not a whole number (or not positive, unless ``signed``), or
    that the grader would accept for ``key``, is dropped. Returns None when fewer than ``want`` survive, so the
    caller draws again.
    """
    key = int(key)
    text = str(key)
    cands = [
        Distractor(str(int(v)), tag, int(v))
        for v, tag in wrong
        if v is not None and (v > 0 or signed) and v == int(v) and int(v) not in exclude
    ]
    # Spread the key's rank among the options: a random number of the named slips sit below it, the rest above it, with
    # plain near misses filling a side that has too few (`hi` is the greatest value the part can take, if it has one).
    near = [
        Distractor(str(key + d), None, key + d)
        for d in (-3, -2, -1, 1, 2, 3)
        if (signed or key + d > 0) and key + d not in exclude
    ]
    cands = mixed(key, cands, want, near, lo=-(10**6) if signed else 1, hi=hi)
    picked = pick_distractors(text, cands, want, tolerance=GRADER_TOLERANCE)
    if len(picked) < want:
        return None
    picked = _keep_round_key_company(key, picked, hi=hi, signed=signed, exclude=exclude)
    part = dict(
        prompt=prompt,
        answer_kind="number",
        answer=key,
        answer_text=text,
        answer_latex=latex_answer(text),
        marks=marks,
        working=working,
        distractors=picked,
    )
    if not signed:
        part["sensible_range"] = (0, None)
    if follow:
        part["follow_through"] = follow
    return part


def _keep_round_key_company(key, picked, *, hi, signed, exclude):
    """A round key (a non-zero multiple of 10) must not be the lone round option: a learner would pick it unseen.
    When no wrong option is round, the option nearest to the key's neighbour 10 away on its own side is replaced by it."""
    if key == 0 or key % 10 or any(int(d.value) % 10 == 0 and d.value != 0 for d in picked):
        return picked
    taken = {int(d.value) for d in picked}
    for step in (10, -10):
        value = key + step
        if value in taken or value in exclude or (value <= 0 and not signed) or (hi is not None and value > hi):
            continue
        same_side = [i for i, d in enumerate(picked) if (d.value < key) == (step < 0)]
        if same_side:
            nearest = min(same_side, key=lambda i: abs(picked[i].value - value))
            return [*picked[:nearest], Distractor(str(value), None, value), *picked[nearest + 1 :]]
    return picked
