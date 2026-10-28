import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from support import fixture, save
from twinpast.data import DataError, load_dataset, new_output
from twinpast.inputs import adapter_input, build_jobs
from twinpast.metrics import score_predictions
from twinpast.runtime import parse_answer, run


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root = self.base / "data"
        self.rows = fixture(self.root)

    def load(self):
        return load_dataset(self.root, expected_pairs=2)

    def score(self, rows):
        path = self.base / "predictions.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        return score_predictions(self.load(), path)

    def predictions(self, a="A", b="B"):
        return [{"pair_id": p["pair_id"], "branch": branch, "answer": answer}
                for p in self.rows for branch, answer in (("A", a), ("B", b))]

    def test_public_schema_loads_after_relocation(self):
        moved = self.base / "moved data"
        shutil.copytree(self.root, moved)
        shutil.rmtree(self.root)
        self.assertEqual(len(load_dataset(moved, expected_pairs=2).qa), 4)

    def test_rejects_extra_fields_and_duplicate_json_keys(self):
        self.rows[0]["extra"] = "unexpected"
        save(self.root, self.rows)
        with self.assertRaises(DataError):
            self.load()
        path = self.root / "data.jsonl"
        text = path.read_text().replace('"extra": "unexpected"', '"task": "AIR"')
        path.write_text(text)
        with self.assertRaisesRegex(DataError, "Duplicate JSON key"):
            self.load()

    def test_rejects_path_traversal_and_symbolic_media(self):
        self.rows[0]["video_a"] = "../video.mp4"
        save(self.root, self.rows)
        with self.assertRaises(DataError):
            self.load()
        self.rows[0]["video_a"] = "videos/twinpast_0001/A.mp4"
        save(self.root, self.rows)
        video = self.root / self.rows[0]["video_a"]
        video.unlink()
        video.symlink_to(self.root / self.rows[0]["video_b"])
        with self.assertRaises(DataError):
            self.load()

    def test_rejects_unsafe_pair_id(self):
        self.rows[0]["pair_id"] = "../../outside"
        save(self.root, self.rows)
        with self.assertRaises(DataError):
            self.load()

    def test_rejects_out_of_bounds_or_nonnumeric_evidence(self):
        for interval in ([0, 11], [0, "note"], [0, True]):
            self.rows[0]["evidence_intervals"]["A"] = [interval]
            save(self.root, self.rows)
            with self.assertRaises(DataError):
                self.load()

    def test_null_annotations_are_preserved(self):
        for key in ("evidence_intervals", "last_decisive_evidence_time", "target_event_rank", "position_bin"):
            self.rows[0][key] = None
        save(self.root, self.rows)
        self.assertIsNone(self.load().pairs[0]["evidence_intervals"])

    def test_perfect_scores_and_task_metrics(self):
        scored = self.score(self.predictions())
        overall = scored["overall"]
        self.assertEqual([overall[k] for k in ("branch_accuracy", "pair_accuracy", "D", "C", "R", "S")], [1, 1, 1, 1, 0, 1])
        self.assertEqual(scored["by_task"]["AIR"]["pairs"], 1)

    def test_same_valid_answers_have_half_branch_accuracy(self):
        scores = self.score(self.predictions("A", "A"))["overall"]
        self.assertEqual(scores["branch_accuracy"], .5)
        self.assertEqual(scores["pair_accuracy"], 0)
        self.assertIsNone(scores["C"])

    def test_reversed_predictions_have_negative_signal(self):
        scores = self.score(self.predictions("B", "A"))["overall"]
        self.assertEqual([scores[k] for k in ("D", "C", "R", "S")], [1, 0, 1, -1])

    def test_missing_predictions_keep_fixed_denominators(self):
        scores = self.score(self.predictions()[:1])
        self.assertEqual(scores["overall"]["branches"], 4)
        self.assertEqual(scores["overall"]["pairs"], 2)
        self.assertEqual(scores["missing_predictions"], 3)

    def test_invalid_unhashable_and_lowercase_answers_count_as_wrong(self):
        for answer in (None, "a", "D", "Answer A", {"answer": "A"}, ["A"]):
            scores = self.score(self.predictions(answer, "B"))
            self.assertEqual(scores["invalid_predictions"], 2)
            self.assertEqual(scores["overall"]["branch_accuracy"], .5)

    def test_duplicate_and_unknown_predictions_rejected(self):
        for rows in (self.predictions() + self.predictions()[:1],
                     [{"pair_id": "unknown", "branch": "A", "answer": "A"}],
                     [{"pair_id": "twinpast_0001", "branch": "C", "answer": "A"}]):
            with self.assertRaises(DataError):
                self.score(rows)

    def test_jobs_and_adapter_inputs_exclude_all_answers_and_evidence(self):
        jobs = build_jobs(self.load())
        self.assertEqual(len(jobs), 4)
        self.assertEqual(set(adapter_input(jobs[0])), {"job_id", "video", "video_stat", "t_query", "question", "options"})
        serialized = json.dumps(jobs)
        for key in ("answer_a", "answer_b", "evidence_intervals", "target_event_rank", "position_bin", "task"):
            self.assertNotIn('"' + key + '"', serialized)

    def test_output_alias_back_into_data_is_rejected(self):
        alias = self.base / "alias"
        alias.symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(DataError):
            new_output(self.root, alias / "output.json")

    def test_readout_is_explicit_and_independent_of_gold(self):
        self.assertIsNone(parse_answer("The answer is A", ["x", "y"]))
        self.assertEqual(parse_answer("The answer is A", ["x", "y"], "first_standalone_letter"), "A")
        self.assertIsNone(parse_answer("a", ["x", "y"]))


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root = self.base / "data"
        fixture(self.root, 350)

    def test_prepare_only_never_constructs_adapter(self):
        with patch("twinpast.runtime.load_adapter", side_effect=AssertionError("should not import a model")):
            state = run(self.root, "online/flash", self.base / "prepared", prepare_only=True)
        self.assertEqual(state["job_count"], 700)
        self.assertFalse((self.base / "prepared/predictions.jsonl").exists())

    def test_run_through_fake_adapter_and_score_partial_predictions(self):
        observed = []
        class Fake:
            def answer_stream(self, job, prompt, emit):
                observed.append(job)
                emit({"type": "observed"})
                return {"response": "A" if job["video"].endswith("A.mp4") else "B"}
        with patch("twinpast.runtime.load_adapter", return_value=Fake()):
            state = run(self.root, "api/chat_video", self.base / "run", limit_pairs=1)
        self.assertEqual(state["state"], "complete")
        self.assertEqual(len(observed), 2)
        self.assertNotIn("answer", observed[0])
        result = score_predictions(load_dataset(self.root), self.base / "run/predictions.jsonl")
        self.assertEqual(result["overall"]["branch_correct"], 2)
        self.assertEqual(result["missing_predictions"], 698)

    def test_technical_failure_is_recorded_without_fabricating_answer(self):
        class Failure:
            def answer_stream(self, job, prompt, emit):
                raise RuntimeError("transport unavailable")
        with patch("twinpast.runtime.load_adapter", return_value=Failure()):
            with self.assertRaises(DataError):
                run(self.root, "api/chat_video", self.base / "failure", limit_pairs=1)
        state = json.loads((self.base / "failure/run.json").read_text())
        self.assertEqual(state["state"], "failed")
        self.assertEqual((self.base / "failure/predictions.jsonl").read_text(), "")

    def test_secret_is_redacted_from_failure(self):
        class Failure:
            def answer_stream(self, job, prompt, emit):
                raise RuntimeError("not-a-live-key")
        with patch.dict("os.environ", {"TWINPAST_API_KEY": "not-a-live-key"}), patch("twinpast.runtime.load_adapter", return_value=Failure()):
            with self.assertRaises(DataError):
                run(self.root, "api/chat_video", self.base / "redacted", limit_pairs=1)
        self.assertNotIn("not-a-live-key", (self.base / "redacted/run.json").read_text())

    def test_run_rejects_existing_output_and_data_output(self):
        with self.assertRaises(DataError):
            run(self.root, "api/chat_video", self.root / "run", prepare_only=True)
        output = self.base / "existing"
        output.mkdir()
        with self.assertRaises(DataError):
            run(self.root, "api/chat_video", output, prepare_only=True)
