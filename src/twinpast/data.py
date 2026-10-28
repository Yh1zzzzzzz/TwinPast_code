"""Standard-library loading of the public paired-video schema."""
from dataclasses import dataclass
import json
import math
import re
from pathlib import Path, PurePosixPath, PureWindowsPath

TASKS = ("AIR", "HSR", "EOR", "HAC", "MEI")
FIELDS = {"pair_id", "video_a", "video_b", "question", "options", "answer_a", "answer_b",
          "t_query", "evidence_intervals", "last_decisive_evidence_time",
          "target_event_rank", "position_bin", "task"}


class DataError(ValueError):
    """Invalid data, prediction, or output destination."""


def require(condition, message):
    if not condition:
        raise DataError(message)


def _unique_keys(items):
    result = {}
    for key, value in items:
        require(key not in result, "Duplicate JSON key: " + key)
        result[key] = value
    return result


def _reject_constant(value):
    raise DataError("Non-finite JSON number")


def json_lines(path):
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line, object_pairs_hook=_unique_keys, parse_constant=_reject_constant)
            except (ValueError, TypeError) as exc:
                raise DataError("{}:{}: {}".format(path, number, exc)) from exc
            require(isinstance(value, dict), "{}:{}: expected an object".format(path, number))
            yield value


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def regular_file(root, relative):
    require(isinstance(relative, str) and bool(relative), "Expected a relative file path")
    require(not PurePosixPath(relative).is_absolute() and not PureWindowsPath(relative).drive
            and "\\" not in relative and all(p not in ("", ".", "..") for p in relative.split("/")),
            "Unsafe relative file path")
    target = root
    for part in relative.split("/"):
        target = target / part
        require(not target.is_symlink(), "Dataset files must be regular local files: " + relative)
    require(target.is_file(), "Missing regular file: " + relative)
    return target


@dataclass(frozen=True)
class Dataset:
    root: Path
    pairs: tuple

    @property
    def qa(self):
        return tuple(dict(pair_id=p["pair_id"], branch=branch,
                          video=p["video_" + branch.lower()], question=p["question"],
                          options=p["options"], answer=p["answer_" + branch.lower()],
                          task=p["task"], t_query=p["t_query"])
                     for p in self.pairs for branch in ("A", "B"))


def load_dataset(root, expected_pairs=350):
    root = Path(root).expanduser().resolve()
    pairs = tuple(json_lines(regular_file(root, "data.jsonl")))
    require(len(pairs) == expected_pairs and expected_pairs > 0, "Unexpected pair count")
    seen = set()
    identities = set()
    for row in pairs:
        require(set(row) == FIELDS, "Unexpected or missing data fields")
        pid = row["pair_id"]
        require(isinstance(pid, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", pid)
                and pid not in seen, "Duplicate or invalid pair ID")
        seen.add(pid)
        require(isinstance(row["task"], str) and row["task"] in TASKS, "Unknown task")
        require(isinstance(row["question"], str) and bool(row["question"]), "Empty question")
        options = row["options"]
        require(isinstance(options, list) and 2 <= len(options) <= 26
                and all(isinstance(v, str) and bool(v) for v in options), "Invalid options")
        letters = {chr(65 + i) for i in range(len(options))}
        answers = row["answer_a"], row["answer_b"]
        require(all(isinstance(a, str) and a in letters for a in answers) and answers[0] != answers[1],
                "Pair answers must be valid and different")
        query = row["t_query"]
        require(finite(query) and query >= 0, "Invalid query time")
        evidence = row["evidence_intervals"]
        if evidence is not None:
            require(isinstance(evidence, dict) and set(evidence) == {"A", "B"}, "Invalid evidence branches")
            for intervals in evidence.values():
                require(isinstance(intervals, list), "Expected evidence intervals")
                for interval in intervals:
                    require(isinstance(interval, list) and len(interval) == 2 and all(finite(v) for v in interval),
                            "Invalid evidence interval")
                    require(0 <= interval[0] <= interval[1] <= query + 1e-6, "Evidence outside query prefix")
        decisive = row["last_decisive_evidence_time"]
        require(decisive is None or (finite(decisive) and 0 <= decisive <= query + 1e-6), "Invalid decisive time")
        rank = row["target_event_rank"]
        require(rank is None or (isinstance(rank, int) and not isinstance(rank, bool) and rank >= 0), "Invalid event rank")
        require(row["position_bin"] in (None, "Early", "Middle", "Late"), "Invalid positional label")
        for branch in ("a", "b"):
            video = regular_file(root, row["video_" + branch])
            identity = video.stat()
            require(identity.st_size > 0, "Empty video")
            key = identity.st_dev, identity.st_ino
            require(key not in identities, "A media file is shared by multiple branches")
            identities.add(key)
    return Dataset(root, pairs)


def new_output(root, requested):
    """Require a new destination outside the dataset, including through aliases."""
    requested = Path(requested).expanduser().absolute()
    require(not requested.is_symlink(), "Output cannot be a symlink")
    path = requested.resolve()
    root = Path(root).resolve()
    require(path != root and root not in path.parents, "Output must be outside the dataset")
    require(not path.exists(), "Output already exists")
    return path

