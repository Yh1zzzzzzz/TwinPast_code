# TwinPast

Code for **TwinPast: Benchmarking Historical Evidence Use in Streaming Video Understanding with Evidence-Contrastive Video Pairs**.

TwinPast contains **350 video pairs (700 videos)** across five tasks: AIR, HSR, EOR, HAC, and MEI. Each pair shares a question and options but has different correct answers.

## Quick start

Use Python 3.10+. Download the dataset separately; its root should contain `data.jsonl` and `videos/`.

```bash
python -m pip install -e .
twinpast validate /path/to/TwinPast
twinpast score /path/to/TwinPast /path/to/predictions.jsonl --output /tmp/scores.json
```

Predictions are JSONL records such as `{"pair_id":"twinpast_0001","branch":"A","answer":"B"}`. Use new output paths outside the dataset. Run `twinpast --help` for all commands.

## Model evaluation

Use a separate environment per model, with its upstream dependencies, CUDA-compatible PyTorch, and downloaded weights. Flash-VStream example:

```bash
twinpast models
twinpast upstream --model eval_flash_vstream --fetch
python -m pip install -r src/twinpast/adapters/eval_flash_vstream/requirements.txt
python -m pip install -e '.[video]'
twinpast run /path/to/TwinPast --condition online/flash \
  --model-root /path/to/models --output-dir /tmp/twinpast-flash
```

Place Flash-VStream weights under `/path/to/models/Flash-VStream-Qwen-7b`. See the [model catalog](src/twinpast/catalog.json) and [adapters](src/twinpast/adapters) for other models. API credentials use `TWINPAST_API_KEY`.

## Answer extraction

`twinpast run` defaults to `--readout explicit_option`. It accepts bare letters,
punctuation and Markdown wrappers (`A.`, `(B)`, `**B**`), explicit answer statements
(`The answer is B`), and a unique complete option text, ignoring case, surrounding
punctuation and whitespace. A letter plus option text must agree. A marked final
answer can follow discussion or an option list; a complete answer in a separate
final paragraph is also supported. Conflicting final statements,
alternatives, partial options and unrecognized prose remain unresolved. A rationale
introduced by `because` or a separate `Explanation:` / `Reason:` line may follow
a complete answer. Known adapter response/end-of-turn markers and a completed leading `<think>`
block are removed; an unfinished thinking block is unresolved.

The parser uses only the response and ordered options, never the reference answer.
It does not infer semantic paraphrases or replace the paper's reviewed answer
mappings. The run records the parser version and preserves each raw response.
`--readout exact_letter` and `--readout first_standalone_letter` retain the old rules
for historical replay; the latter can mistake prose for a choice. `twinpast score`
continues to consume already-extracted uppercase letters; missing or invalid
answers count as incorrect with fixed denominators.

## License

Code: [Apache-2.0](LICENSE). See [third-party notices](THIRD_PARTY_NOTICES.md); datasets and model weights retain their own terms.

## Citation

```bibtex
```
