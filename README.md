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

## License

Code: [Apache-2.0](LICENSE). See [third-party notices](THIRD_PARTY_NOTICES.md); datasets and model weights retain their own terms.

## Citation

```bibtex
```
