"""Independent per-condition runs; optional model packages are imported lazily."""
import importlib
from contextlib import suppress
import json
import os
from pathlib import Path
import sys
import time
import zlib

from .data import DataError, load_dataset, new_output, require
from .answers import PARSER_VERSION, READOUTS, parse_answer
from .inputs import adapter_input, build_jobs, prompt, write_jobs
from .security import redact as _redact, validate_configuration

PACKAGE = Path(__file__).resolve().parent


def catalog():
    return json.loads((PACKAGE / "catalog.json").read_text())


def _json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_adapter(spec, configuration, custom=None):
    shared = PACKAGE / "adapters/eval_streaming_shared"
    sys.path.insert(0, str(shared))
    if custom or spec.get("factory"):
        module, name = (custom or spec["factory"]).split(":", 1)
        return getattr(importlib.import_module(module), name)(configuration)
    folder = PACKAGE / "adapters" / spec["folder"]
    shared = PACKAGE / "adapters/eval_streaming_shared"
    sys.path[:0] = [str(folder / relative) for relative in spec.get("python_paths", [])] + [str(folder), str(shared)]
    # One model per process avoids collisions among upstream modules with common names.
    return importlib.import_module(spec["module"]).Adapter()


def run(dataset_root, condition, output_dir, model_root=None, configuration=None,
        readout="explicit_option", limit_pairs=None, prepare_only=False, seed=0, custom_adapter=None):
    conditions = catalog()
    require(condition in conditions, "Unknown condition")
    require(readout in READOUTS, "Unknown readout")
    dataset = load_dataset(dataset_root)
    output = new_output(dataset.root, output_dir)
    require(output != PACKAGE and PACKAGE not in output.parents, "Output cannot be inside the installed package")
    jobs = build_jobs(dataset, limit_pairs)
    spec = conditions[condition]
    configuration = dict(configuration or {})
    validate_configuration(configuration)
    output.mkdir(parents=True, exist_ok=False)
    state = {"condition": condition, "readout": readout, "seed": seed,
             "parser_version": PARSER_VERSION if readout == "explicit_option" else "legacy_" + readout,
             "job_count": len(jobs), "dataset_pairs": len(dataset.pairs),
             "state": "prepared", "full_collection": len(jobs) == 2 * len(dataset.pairs)}
    _json(output / "run.json", state)
    write_jobs(output / "jobs.jsonl", jobs)
    if prepare_only:
        return state
    configuration.update(spec.get("configuration", {}))
    configuration["output_dir"] = str(output)
    configuration["seed"] = seed
    configuration["model_root"] = str(Path(model_root).expanduser().resolve()) if model_root else str(Path.cwd() / "models")
    os.environ["TWINPAST_RUN_ROOT"] = str(output)
    os.environ["TWINPAST_MODEL_ROOT"] = configuration["model_root"]
    os.environ.update(spec.get("environment", {}))
    adapter = None
    try:
        if not custom_adapter and ("folder" in spec or spec.get("requires_torch")):
            import torch
            import numpy as np
            torch.set_num_threads(4)
            torch.manual_seed(seed)
            np.random.seed(seed)
        adapter = load_adapter(spec, configuration, custom_adapter)
        state["state"] = "running"
        _json(output / "run.json", state)
        (output / "traces").mkdir()
        with (output / "predictions.jsonl").open("x", encoding="utf-8", buffering=1) as stream:
            for job in jobs:
                pair_seed = (int(seed) + zlib.crc32(job["pair_id"].encode())) % (2 ** 32)
                # Local models use matching A/B seeds. API sampling is configured by its adapter.
                if "torch" in sys.modules:
                    torch = sys.modules["torch"]
                    torch.manual_seed(pair_seed)
                if "numpy" in sys.modules:
                    sys.modules["numpy"].random.seed(pair_seed)
                started = time.monotonic()
                with (output / "traces" / (job["job_id"] + ".jsonl")).open("x", encoding="utf-8") as trace:
                    def emit(event):
                        trace.write(json.dumps(_redact(event), ensure_ascii=False, allow_nan=False) + "\n")
                        trace.flush()
                    if spec["mode"] == "offline":
                        from common import sample
                        frames, metadata = sample(job)
                        result = getattr(adapter, spec.get("method", "answer"))(frames, metadata, prompt(job), job["video"])
                    else:
                        result = adapter.answer_stream(adapter_input(job), prompt(job), emit)
                if "torch" in sys.modules:
                    torch = sys.modules["torch"]
                    for device in range(torch.cuda.device_count()):
                        torch.cuda.synchronize(device)
                require(isinstance(result, dict) and isinstance(result.get("response"), str), "Adapter must return a response string")
                record = {"pair_id": job["pair_id"], "branch": job["branch"],
                          "answer": parse_answer(result["response"], job["options"], readout),
                          "response": result["response"], "wall_s": time.monotonic() - started,
                          "seed": pair_seed, "details": result}
                stream.write(json.dumps(_redact(record), ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                state["completed"] = state.get("completed", 0) + 1
                _json(output / "run.json", state)
        state["state"] = "complete"
        _json(output / "run.json", state)
        return state
    except Exception as exc:
        state.update(state="failed", error=_redact(type(exc).__name__ + ": " + str(exc)))
        _json(output / "run.json", state)
        raise DataError(state["error"]) from None
    finally:
        if adapter is not None and callable(getattr(adapter, "close", None)):
            with suppress(Exception):
                adapter.close()

