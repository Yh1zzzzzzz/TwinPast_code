"""Exact-letter branch and paired metrics with fixed denominators."""
from .data import TASKS, json_lines, require


def score_predictions(dataset, predictions):
    by_id = {row["pair_id"]: row for row in dataset.pairs}
    answers = {}
    invalid = 0
    for row in json_lines(predictions):
        pid, branch = row.get("pair_id"), row.get("branch")
        require(isinstance(pid, str) and pid in by_id, "Unknown prediction pair ID")
        require(isinstance(branch, str) and branch in ("A", "B"), "Unknown prediction branch")
        key = pid, branch
        require(key not in answers, "Duplicate prediction branch")
        value = row.get("answer")
        valid = isinstance(value, str) and len(value) == 1 and "A" <= value <= chr(64 + len(by_id[pid]["options"]))
        answers[key] = value if valid else None
        invalid += not valid

    def summarize(pairs):
        count = len(pairs)
        correct = both = different = reverse = same = incomplete = 0
        for pair in pairs:
            a, b = (answers.get((pair["pair_id"], branch)) for branch in ("A", "B"))
            valid = a is not None and b is not None
            correct += int(a == pair["answer_a"]) + int(b == pair["answer_b"])
            both += a == pair["answer_a"] and b == pair["answer_b"]
            different += valid and a != b
            reverse += a == pair["answer_b"] and b == pair["answer_a"]
            same += valid and a == b
            incomplete += not valid
        return {"pairs": count, "branches": 2 * count, "branch_correct": correct, "both_correct": both,
                "branch_accuracy": correct / (2 * count), "pair_accuracy": both / count,
                "different_valid_pairs": different, "reversed_pairs": reverse,
                "same_valid_pairs": same, "invalid_or_missing_pairs": incomplete,
                "D": different / count, "C": both / different if different else None,
                "R": reverse / count, "S": (both - reverse) / count}

    overall = summarize(dataset.pairs)
    return {"overall": overall,
            "by_task": {task: summarize([p for p in dataset.pairs if p["task"] == task])
                        for task in TASKS if any(p["task"] == task for p in dataset.pairs)},
            "prediction_rows": len(answers), "invalid_predictions": invalid,
            "missing_predictions": 2 * len(dataset.pairs) - len(answers),
            "fixed_denominator": True, "metric_units": "fractions"}

