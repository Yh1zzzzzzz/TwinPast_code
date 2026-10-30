"""Command line interface for public TwinPast data and model evaluation."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

from .data import DataError, load_dataset, new_output
from .inputs import build_jobs, write_jobs
from .metrics import score_predictions
from .runtime import catalog, run
from .answers import READOUTS


def main(argv=None):
    parser = argparse.ArgumentParser(prog="twinpast")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="validate public data without decoding or hashing videos")
    validate.add_argument("dataset", type=Path)
    score = commands.add_parser("score", help="score exact-letter branch predictions")
    score.add_argument("dataset", type=Path)
    score.add_argument("predictions", type=Path)
    score.add_argument("--output", type=Path)
    prepare = commands.add_parser("prepare", help="freeze label-free inference inputs")
    prepare.add_argument("dataset", type=Path)
    prepare.add_argument("--output", type=Path, required=True)
    commands.add_parser("models", help="list model conditions")
    evaluate = commands.add_parser("run", help="run one configured model condition")
    evaluate.add_argument("dataset", type=Path)
    evaluate.add_argument("--condition", choices=sorted(catalog()), required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--model-root", type=Path)
    evaluate.add_argument("--config", type=Path, help="model/API settings; credentials are read from the environment")
    evaluate.add_argument("--readout", choices=READOUTS, default="explicit_option",
                          help="answer extraction (default: explicit_option); other modes replay legacy rules")
    evaluate.add_argument("--limit-pairs", type=int)
    evaluate.add_argument("--prepare-only", action="store_true")
    evaluate.add_argument("--seed", type=int, default=0)
    evaluate.add_argument("--adapter", help="custom module:Class implementing the adapter interface")
    upstream = commands.add_parser("upstream", help="inspect registered upstream sources")
    upstream.add_argument("--model")
    upstream.add_argument("--fetch", action="store_true", help="download missing source code at its registered revision")
    args = parser.parse_args(argv)
    try:
        if args.command == "models":
            result = catalog()
        elif args.command == "upstream":
            from .upstream import setup
            result = setup(args.model, args.fetch)
        elif args.command == "run":
            configuration = json.loads(args.config.read_text()) if args.config else {}
            if not isinstance(configuration, dict):
                raise DataError("Configuration must be a JSON object")
            result = run(args.dataset, args.condition, args.output_dir, args.model_root,
                         configuration, args.readout, args.limit_pairs, args.prepare_only, args.seed, args.adapter)
        else:
            dataset = load_dataset(args.dataset)
            if args.command == "validate":
                result = {"pairs": len(dataset.pairs), "videos": 2 * len(dataset.pairs),
                          "tasks": dict(Counter(p["task"] for p in dataset.pairs)), "valid": True}
            elif args.command == "prepare":
                path = new_output(dataset.root, args.output)
                write_jobs(path, build_jobs(dataset))
                result = {"jobs": 2 * len(dataset.pairs), "output": str(path)}
            else:
                result = score_predictions(dataset, args.predictions)
                if args.output:
                    with new_output(dataset.root, args.output).open("x", encoding="utf-8") as stream:
                        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (DataError, OSError, ValueError) as exc:
        print("error: " + str(exc), file=sys.stderr)
        return 2
