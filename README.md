# TwinPast

Evaluation code for **TwinPast: Benchmarking Historical Evidence Use in Streaming Video Understanding with Evidence-Contrastive Video Pairs**.

The benchmark contains **350 video pairs (700 videos)**. Each pair shares a question and ordered options but has different correct answers. Five tasks cover historical evidence use: AIR (Attribute and Identity Recall), HSR (Historical State Recall), EOR (Event Order Reasoning), HAC (Historical Action Counting), and MEI (Multi-Event Integration).

## Installation

Data validation and scoring require Python 3.8+ with no extra dependencies. Model inference requires Linux, Python 3.10+, and a separate environment for each model family.

```bash
python -m pip install -e .
twinpast --help
```

## Data and scoring

Download the dataset separately. Set `DATASET` to the directory containing `data.jsonl` and `videos/`.

```bash
DATASET=/path/to/TwinPast
twinpast validate "$DATASET"
twinpast prepare "$DATASET" --output /tmp/twinpast-jobs.jsonl
twinpast score "$DATASET" /path/to/predictions.jsonl --output /tmp/twinpast-scores.json
```

Prepared jobs exclude reference answers, evidence annotations, and task labels. Output paths must be new and outside the dataset.

Predictions use one JSON object per video branch, with a public pair ID, branch `A` or `B`, and an uppercase option letter:

```json
{"pair_id":"twinpast_0001","branch":"A","answer":"B"}
```

| Metric | Definition |
| --- | --- |
| `branch_accuracy` (Acc.) | Correct branches / 700 |
| `pair_accuracy` (PairAcc) | Both-correct pairs / 350 |
| `D` | Pairs with valid, different predictions / 350 |
| `C` | Both-correct / valid-different pairs; null if none |
| `R` | Fully reversed pairs (pA=yB and pB=yA) / 350 |
| `S` | PairAcc minus R |

Scores include overall and per-task results, expressed as fractions. Missing or invalid answers count as wrong; partial runs retain the full benchmark denominators.

## Model evaluation

List supported conditions and fetch the selected model's registered upstream source:

```bash
twinpast models
twinpast upstream --model eval_flash_vstream --fetch
```

Install the upstream requirements and a CUDA-compatible PyTorch build in the model's own environment, then install the adapter dependencies. For example, Flash-VStream expects weights in `/path/to/models/Flash-VStream-Qwen-7b`:

```bash
python -m pip install -r src/twinpast/adapters/eval_flash_vstream/requirements.txt
python -m pip install -e '.[video]'
twinpast run /path/to/TwinPast --condition online/flash \
  --model-root /path/to/models --output-dir /tmp/twinpast-flash
```

Weights are downloaded separately. Dependency files live beside the [adapters](src/twinpast/adapters); conditions and upstream revisions are listed in [catalog.json](src/twinpast/catalog.json) and [upstreams.json](src/twinpast/upstreams.json). FFmpeg/ffprobe is required for codec, chunked, and whole-video inputs.

Use `--prepare-only` to prepare inputs without inference, or `--limit-pairs 1` for a two-video smoke run. Run one condition per process. Streaming models ingest history before receiving the question and options at the query; sessions reset between branches.

The default readout accepts only an exact uppercase option letter. The scorer evaluates supplied letters; model-specific semantic answer extraction is not automatic. Report the checkpoint, environment, input mode, generation settings, readout, and seed with results.

## Hosted models

Install `python -m pip install -e '.[http,video]'`. Put endpoint settings in a local JSON file and provide credentials through `TWINPAST_API_KEY`; credentials must not appear in configuration files or URLs.

```json
{
  "base_url": "https://api.example.invalid/v1",
  "model": "your-model-id",
  "generation": {"temperature": 0, "max_tokens": 256}
}
```

Replace the example values with your provider's settings:

```bash
twinpast run /path/to/TwinPast --condition offline/http64 \
  --config /path/to/local-config.json --output-dir /tmp/twinpast-http64
```

Other hosted conditions include native video, Google Files, and realtime interfaces; their protocols and generation parameters are provider-specific. Custom adapters can be selected with `--adapter package.module:Class`.

## Checks and license

```bash
python -B tools/check_release.py
python -B -m unittest discover -s tests -v
```

Keep credentials, local configurations, run outputs, weights, and downloaded upstream code outside the published source. Model inference executes upstream or checkpoint code; use trusted sources and inspect outputs before sharing. Tests use mocked model/API interfaces and do not establish numerical reproduction.

Our code is licensed under [Apache-2.0](LICENSE). Third-party terms are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Videos, model weights, and hosted services retain their respective terms.

## Citation

```bibtex
```
