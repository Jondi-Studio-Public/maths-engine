"""The same seed must give the same question in every process.

`p15d_shape_sorting` (skill `f10-shape-sorting-algorithm`) built its prompt by iterating a `set` of strings
into `rng.choice`. The iteration order of a set of strings depends on `PYTHONHASHSEED`, which Python picks
afresh for every process, so one seed gave a different question in different processes: a question served
by one worker could not be regenerated, graded or reported from another, and 36 of 36 (preset, seed) keys
checked differed between two hash seeds. The tests below run the generators in child processes under
different hash seeds and compare every question byte for byte.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]

_DIGEST_SCRIPT = """\
import hashlib, json, sys
sys.path.insert(0, {root!r})
from mathgen import REGISTRY, make
from mathgen.core import PRESETS

out = {{}}
for topic in sorted(REGISTRY):
    if not topic.startswith({prefix!r}):
        continue
    for tier in [None] + sorted(PRESETS.get(topic, {{}})):
        for seed in range(1, {seeds} + 1):
            question = make(topic, seed=seed, difficulty=tier)
            out[f"{{topic}}/{{tier}}/{{seed}}"] = hashlib.sha1(question.to_json().encode()).hexdigest()
print(json.dumps(out, sort_keys=True))
"""


def _digests(tmp_path: pathlib.Path, hash_seed: str, prefix: str, seeds: int) -> dict:
    script = tmp_path / "digest_questions.py"
    script.write_text(_DIGEST_SCRIPT.format(root=str(_ROOT), prefix=prefix, seeds=seeds), encoding="utf-8")
    env = {**os.environ, "PYTHONHASHSEED": hash_seed, "PYTHONPATH": str(_ROOT)}
    done = subprocess.run(
        [sys.executable, str(script)],
        env=env,
        cwd=_ROOT,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    return json.loads(done.stdout)


def test_the_control_really_changes_set_order_between_processes(tmp_path):
    # If PYTHONHASHSEED did not reach the child, the checks below would pass for any generator.
    script = tmp_path / "order.py"
    script.write_text("print(list({'circle', 'square', 'triangle', 'pentagon', 'hexagon', 'kite', 'rhombus'}))\n")
    orders = set()
    for hash_seed in ("0", "1", "2", "3", "4242"):
        done = subprocess.run(
            [sys.executable, str(script)],
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=60,
        )
        orders.add(done.stdout)
    assert len(orders) > 1


def test_shape_sorting_questions_do_not_depend_on_the_hash_seed(tmp_path):
    first = _digests(tmp_path, "0", "p15d_shape_sorting", seeds=12)
    second = _digests(tmp_path, "4242", "p15d_shape_sorting", seeds=12)
    third = _digests(tmp_path, "7", "p15d_shape_sorting", seeds=12)
    assert len(first) == 5 * 12  # no preset, and easy / medium / hard / exam
    assert first == second == third


def test_every_p15d_question_is_independent_of_the_hash_seed(tmp_path):
    # The same builder module, the same helpers: the whole batch is checked, not only the one that was found.
    first = _digests(tmp_path, "0", "p15d_", seeds=4)
    second = _digests(tmp_path, "4242", "p15d_", seeds=4)
    assert first == second
    assert len(first) >= 6 * 4


def test_shape_sorting_asks_the_same_question_twice_in_one_process():
    from mathgen import make

    for tier in ("easy", "medium", "hard", "exam"):
        for seed in range(1, 9):
            assert (
                make("p15d_shape_sorting", seed=seed, difficulty=tier).to_json()
                == make("p15d_shape_sorting", seed=seed, difficulty=tier).to_json()
            )


@pytest.mark.parametrize("module", sorted((_ROOT / "mathgen" / "generators").glob("p15d_*.py")))
def test_p15d_generators_do_not_choose_from_a_set(module):
    # A set of strings has no stable order. `rng.choice(sorted(...))` is fine; `rng.choice(list(a_set))` is not.
    import ast

    tree = ast.parse(module.read_text(encoding="utf-8"))
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") in {"choice", "sample", "shuffle"}:
            for arg in node.args:
                if isinstance(arg, ast.Call) and getattr(arg.func, "id", "") in {"set", "frozenset"}:
                    offenders.append(node.lineno)
                if isinstance(arg, (ast.Set, ast.SetComp)):
                    offenders.append(node.lineno)
                if (
                    isinstance(arg, ast.Call)
                    and getattr(arg.func, "id", "") == "list"
                    and arg.args
                    and isinstance(arg.args[0], (ast.Set, ast.SetComp))
                ):
                    offenders.append(node.lineno)
    assert offenders == [], f"{module.name}: rng draws from a set at lines {offenders}"
