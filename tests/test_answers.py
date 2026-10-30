"""Answer-format and ambiguity regressions, independent of video answer keys."""
import contextlib
import io
import unittest
from unittest.mock import patch

import support
from twinpast.answers import parse_answer
from twinpast.cli import main
from twinpast.data import DataError


class AnswerTests(unittest.TestCase):
    options = ["Red", "Blue", "Green", "Yellow", "Black", "White"]

    def test_explicit_letters_and_wrappers(self):
        for text in ("B", "b", " B. ", "(B)", "B).", "[B]", "**B**", "`B`",
                     "Ｂ。", "```text\nB\n```", r"\boxed{B}",
                     "B<|im_end|>", "B.<|eot_id|>", "<think>A or C?</think>\nB",
                     "<|response|>B.<|im_end|><|im_end|>", "</response> B<|im_end|>",
                     "<response>B</response>", "(Video Time = 103.83s) Assistant:B"):
            with self.subTest(text=text):
                self.assertEqual(parse_answer(text, self.options), "B")

    def test_explicit_answer_statements_and_final_selection(self):
        for text in ("The answer is B.", "Option B", "Choice: (b)",
                     "The correct option is B", "Final answer: Blue",
                     "I considered A and C.\nFinal answer: B.",
                     "A. Red\nB. Blue\nC. Green\nAnswer: B",
                     "Answer: A\nFinal answer: B", "Answer: B\nAnswer: B",
                     "## Final answer\n**B**", "Therefore, the answer is B.",
                     "The object is blue.\n\nB", "I compared A and C.\n\n**B**"):
            with self.subTest(text=text):
                self.assertEqual(parse_answer(text, self.options), "B")

    def test_complete_option_content_and_explicit_explanations(self):
        for text in ("blue", '"Blue."', "B. Blue", "(B): Blue", "B — Blue",
                     "B. Blue)", "The correct answer is Blue.", "B because the object is blue.",
                     "B. Blue\nExplanation: A and C describe other colors."):
            with self.subTest(text=text):
                self.assertEqual(parse_answer(text, self.options), "B")
        self.assertEqual(parse_answer("Blue", ["Blue", "Red"]), "A")

    def test_no_letter_mining_from_prose(self):
        for text in ("It is a blue object, so B.", "A blue object is visible.",
                     "I compared B with C.", "The answer is not B.", "Not option B.",
                     "Could the answer be B?", "B is one possibility.", "Maybe B."):
            with self.subTest(text=text):
                self.assertIsNone(parse_answer(text, self.options))

    def test_ambiguous_contradictory_and_out_of_range_answers(self):
        for text in ("A or B", "A/B", "A, B", "A and B", "B. Red", "(B): Red",
                     "Answer: B or C", "Answer: A\nAnswer: B", "Final answer: B\nFinal answer: C",
                     "Final answer: B\nAnswer: C", "A. Red\nB. Blue\n\nC. Green",
                     "Final answer: B\nFinal answer: unsure", "B\nC", "G", "Option Z", "B?",
                     "Blue?", "A\n\nB", "A. Red\n\nB. Blue", "B. Blue or Green",
                     "B, probably", "<think>B", "<|silent|>", "", None):
            with self.subTest(text=text):
                self.assertIsNone(parse_answer(text, self.options))
        self.assertIsNone(parse_answer("C.", ["Red", "Blue"]))

    def test_partial_reordered_and_duplicate_options_are_not_guessed(self):
        options = ["Red, then blue", "Blue, then red"]
        self.assertEqual(parse_answer("red,  then blue.", options), "A")
        self.assertEqual(parse_answer("A. red,then blue.", options), "A")
        self.assertIsNone(parse_answer("Red", options))
        self.assertIsNone(parse_answer("A. Blue, then red", options))
        self.assertIsNone(parse_answer("Blue", ["Blue", "blue."]))
        self.assertIsNone(parse_answer("A. Blue", ["Blue", "blue."]))
        self.assertEqual(parse_answer("-1", ["1", "-1"]), "B")
        self.assertEqual(parse_answer("1.5", ["15", "1.5"]), "B")
        self.assertEqual(parse_answer(".5", ["5", ".5"]), "B")

    def test_legacy_modes_are_explicit_and_unchanged(self):
        self.assertEqual(parse_answer("B", self.options, "exact_letter"), "B")
        self.assertIsNone(parse_answer("B.", self.options, "exact_letter"))
        self.assertEqual(parse_answer("It is a blue object, so B.", self.options, "first_standalone_letter"), "A")
        with self.assertRaises(DataError):
            parse_answer("B", self.options, "unknown")

    def test_cli_defaults_to_new_parser(self):
        with patch("twinpast.cli.run", return_value={}) as run, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["run", "/unused/data", "--condition", "online/flash",
                                   "--output-dir", "/unused/output", "--prepare-only"]), 0)
        self.assertEqual(run.call_args.args[5], "explicit_option")


if __name__ == "__main__":
    unittest.main()
