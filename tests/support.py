import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def fixture(root, count=2):
    root.mkdir()
    rows = []
    for index in range(count):
        pid = "twinpast_" + str(index + 1).zfill(4)
        directory = root / "videos" / pid
        directory.mkdir(parents=True)
        (directory / "A.mp4").write_bytes(b"fixture video A")
        (directory / "B.mp4").write_bytes(b"fixture video B")
        rows.append({"pair_id": pid, "video_a": "videos/" + pid + "/A.mp4", "video_b": "videos/" + pid + "/B.mp4",
                     "question": "Which object appeared first?", "options": ["Cup", "Book", "Ball"],
                     "answer_a": "A", "answer_b": "B", "t_query": 10.0,
                     "evidence_intervals": {"A": [[1, 2]], "B": [[3, 4]]},
                     "last_decisive_evidence_time": 4.0, "target_event_rank": 1,
                     "position_bin": "Early", "task": ("AIR", "HSR", "EOR", "HAC", "MEI")[index % 5]})
    save(root, rows)
    return rows


def save(root, rows):
    (root / "data.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

