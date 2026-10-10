"""
Independent oracle for project network skills.

Parses rendered questions without reading mathgen/generators/p6b_project_network_build.py
and verifies answers using standard Critical Path Method (CPM) algorithm.
"""

import re
import pytest
from mathgen import make


class ProjectNetwork:
    """Represents a project network and computes CPM values independently."""

    def __init__(self):
        self.activities = {}  # {label: (duration, predecessors)}
        self.es = {}  # earliest start times
        self.ef = {}  # earliest finish times
        self.ls = {}  # latest start times
        self.lf = {}  # latest finish times
        self.successors = {}  # reverse of predecessor relation

    def add_activity(self, label, duration, predecessors):
        """Add an activity to the network."""
        self.activities[label] = (duration, predecessors)
        self.successors[label] = []

    def build_successors(self):
        """Build the successors map from predecessors."""
        for label, (duration, preds) in self.activities.items():
            for pred in preds:
                if pred not in self.successors:
                    self.successors[pred] = []
                self.successors[pred].append(label)

    def compute_forward_pass(self):
        """Compute earliest start (ES) and earliest finish (EF) times."""
        # Process activities in topological order
        processed = set()
        while len(processed) < len(self.activities):
            before = len(processed)
            for label, (duration, preds) in self.activities.items():
                if label in processed:
                    continue
                # Check if all predecessors are processed
                if all(p in processed for p in preds):
                    if not preds:
                        self.es[label] = 0
                    else:
                        self.es[label] = max(self.ef[p] for p in preds)
                    self.ef[label] = self.es[label] + duration
                    processed.add(label)
            assert len(processed) > before, (
                f"No progress in forward pass; unresolved activities: {set(self.activities) - processed}"
            )

    def compute_backward_pass(self):
        """Compute latest finish (LF) and latest start (LS) times."""
        self.build_successors()
        # Completion time = max EF over all activities
        completion_time = max(self.ef.values())

        # Process activities in reverse topological order
        processed = set()
        while len(processed) < len(self.activities):
            before = len(processed)
            for label, (duration, preds) in self.activities.items():
                if label in processed:
                    continue
                # Check if all successors are processed
                succs = self.successors.get(label, [])
                if all(s in processed for s in succs):
                    if not succs:
                        self.lf[label] = completion_time
                    else:
                        self.lf[label] = min(self.ls[s] for s in succs)
                    self.ls[label] = self.lf[label] - duration
                    processed.add(label)
            assert len(processed) > before, (
                f"No progress in backward pass; unresolved activities: {set(self.activities) - processed}"
            )


def parse_project_network(question_text):
    """Parse question text to extract activities and query type."""
    lines = question_text.strip().split(". ")

    network = ProjectNetwork()
    query_types = {"est": False, "lst": False, "completion": False}
    query_activity = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Parse "The project starts at time 0"
        if "project starts at time" in line.lower():
            continue

        # Parse activity definitions: "Activity X: duration Y days, <tail>"
        activity_match = re.match(r"Activity\s+(\w+):\s+duration\s+(\d+)\s+days?,\s+(.+)$", line)
        if activity_match:
            label = activity_match.group(1)
            duration = int(activity_match.group(2))
            tail = activity_match.group(3).strip()

            # Parse the tail: either "no predecessors" or "predecessors: A, B, ..."
            if tail.lower() == "no predecessors":
                preds = []
            elif tail.lower().startswith("predecessors:"):
                preds_str = tail.split(":", 1)[1].strip()
                preds = [p.strip() for p in preds_str.split(",")]
            else:
                raise ValueError(f"Unparseable activity tail in line: {line}")

            network.add_activity(label, duration, preds)
            continue

        # Parse query - check all three types
        has_earliest = "earliest starting time" in line.lower()
        has_latest = "latest starting time" in line.lower()
        has_completion = "minimum time to complete the project" in line.lower()

        query_count = sum([has_earliest, has_latest, has_completion])
        if query_count > 1:
            raise ValueError(f"Ambiguous query in line: {line}")

        if has_earliest:
            query_types["est"] = True
            matches = list(re.finditer(r"activity\s+(\w+)", line, re.IGNORECASE))
            assert len(matches) == 1, f"Expected exactly one activity reference in: {line}"
            query_activity = matches[0].group(1)

        if has_latest:
            query_types["lst"] = True
            matches = list(re.finditer(r"activity\s+(\w+)", line, re.IGNORECASE))
            assert len(matches) == 1, f"Expected exactly one activity reference in: {line}"
            query_activity = matches[0].group(1)

        if has_completion:
            query_types["completion"] = True

    # Validate that we found activities and a single query type
    if not network.activities:
        raise ValueError(f"No activities found in question: {question_text}")

    query_type_count = sum(query_types.values())
    assert query_type_count == 1, f"Expected exactly one query type, found {query_type_count}"

    # Validate that activity labels in queries exist in the network
    if query_activity is not None:
        assert query_activity in network.activities, f"Query references activity {query_activity} not found in network"

    # Validate that all predecessor labels exist
    for label, (duration, preds) in network.activities.items():
        for pred in preds:
            assert pred in network.activities, f"Activity {label} references predecessor {pred} not found in network"

    # Determine which query type was found
    if query_types["est"]:
        query_type = "est"
    elif query_types["lst"]:
        query_type = "lst"
    else:
        query_type = "completion"

    return network, query_type, query_activity


def compute_answer(network, query_type, query_activity):
    """Compute the answer using CPM algorithm."""
    network.compute_forward_pass()
    network.compute_backward_pass()

    if query_type == "est":
        return network.es[query_activity]
    elif query_type == "lst":
        return network.ls[query_activity]
    elif query_type == "completion":
        return max(network.ef.values())
    else:
        raise ValueError(f"Unknown query type: {query_type}")


@pytest.mark.parametrize(
    "topic,difficulty,seed",
    [
        (topic, diff, s)
        for topic in [
            "use_forward_and_backward_scanning_to_determine_the",
            "use_the_critical_path_to_determine_the_minimum",
        ]
        for diff in ["easy", "medium", "hard"]
        for s in range(60)
    ],
)
def test_project_network_oracle(topic, difficulty, seed):
    """Test oracle for project network skills."""
    question = make(topic, seed=seed, difficulty=difficulty)

    # Parse the question text independently
    network, query_type, query_activity = parse_project_network(question.text)

    # Compute answer using our own CPM implementation
    computed_answer = compute_answer(network, query_type, query_activity)

    # Verify against the generator's answer
    expected_answer = int(question.answer_text)
    assert computed_answer == expected_answer, (
        f"Seed {seed}, {difficulty}, {topic}: computed {computed_answer} but expected {expected_answer}"
    )

    # Verify answer is not a distractor
    distractors = [int(d) for d in question.distractors]
    assert computed_answer not in distractors, (
        f"Seed {seed}, {difficulty}, {topic}: computed answer {computed_answer} matches a distractor"
    )
