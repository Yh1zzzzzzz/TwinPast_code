"""Create model inputs without answer labels or evidence annotations."""
import json
from .data import require


def build_jobs(dataset, limit_pairs=None):
    if limit_pairs is not None:
        require(isinstance(limit_pairs, int) and not isinstance(limit_pairs, bool)
                and 1 <= limit_pairs <= len(dataset.pairs), "Invalid pair limit")
    pairs = dataset.pairs if limit_pairs is None else dataset.pairs[:limit_pairs]
    jobs = []
    for pair in pairs:
        for branch in ("A", "B"):
            path = dataset.root / pair["video_" + branch.lower()]
            st = path.stat()
            jobs.append({"job_id": pair["pair_id"] + "__" + branch,
                         "pair_id": pair["pair_id"], "branch": branch,
                         "video": str(path),
                         "video_stat": {"size": st.st_size, "inode": st.st_ino, "mtime_ns": st.st_mtime_ns},
                         "t_query": pair["t_query"], "question": pair["question"], "options": pair["options"]})
    return jobs


def adapter_input(job):
    return {key: job[key] for key in ("job_id", "video", "video_stat", "t_query", "question", "options")}


def prompt(job):
    return job["question"] + "\n" + "\n".join(chr(65 + i) + ". " + option
                                              for i, option in enumerate(job["options"])) + \
        "\nAnswer with only the letter of the correct option."


def write_jobs(path, jobs):
    with path.open("x", encoding="utf-8") as stream:
        for job in jobs:
            stream.write(json.dumps(job, ensure_ascii=False, allow_nan=False) + "\n")

