# TwinPast

Evaluation code for **TwinPast: Benchmarking Historical Evidence Use in Streaming Video Understanding with Evidence-Contrastive Video Pairs**.

TwinPast contains 350 video pairs (700 branches). Each pair shares a question and ordered options but has different correct answers. The tasks are AIR (Attribute and Identity Recall), HSR (Historical State Recall), EOR (Event Order Reasoning), HAC (Historical Action Counting), and MEI (Multi-Event Integration).

## Quick start

Validation, input preparation and scoring use Python 3.8+ and the standard library. Use a maintained Python version for deployment. Model inference requires Linux, Python 3.10+ and a separate environment for each model family.

```bash
python -m pip install -e .
twinpast --help
```

Download the dataset separately. Its root must contain `data.jsonl` and `videos/`. In a combined release checkout the dataset is `../huggingface`; replace that argument with your own dataset path as needed.

```bash
twinpast validate ../huggingface
twinpast prepare ../huggingface --output /tmp/twinpast-jobs.jsonl
twinpast score ../huggingface /path/to/predictions.jsonl --output /tmp/twinpast-scores.json
```

Outputs require new paths outside the dataset. The loader checks the exact public schema, relative paths and file identity without decoding or hashing media. Prepared inference jobs exclude reference answers, evidence annotations and task labels. Null timing/position annotations are preserved.

## Model inference

List the available conditions, obtain the selected model's upstream source and install its dependencies in a separate environment:

```bash
twinpast models
twinpast upstream --model eval_flash_vstream
twinpast upstream --model eval_flash_vstream --fetch
# First install the upstream's requirements and a CUDA-compatible PyTorch build.
python -m pip install -r src/twinpast/adapters/eval_flash_vstream/requirements.txt
python -m pip install -e '.[video]'
twinpast run ../huggingface --condition online/flash \
  --model-root /path/to/models --output-dir /tmp/twinpast-flash
```

Model-specific `requirements.txt` files live beside their adapters. They preserve compatibility constraints and are not complete GPU environment locks. Upstream requirements may add dependencies. Do not combine incompatible Transformers versions. FFmpeg/ffprobe must be on PATH for codec, chunked and whole-video inputs. Weights and upstream repositories are obtained separately; no weights are bundled here.

Use `--prepare-only` to prepare a run without loading weights, decoding media or calling an API. Use `--limit-pairs 1` for a two-branch smoke run; scoring still uses the full dataset denominator. Run one condition per process. Model sessions reset between branches; streaming models receive the question and options at the query, after history ingestion.

<details>
<summary>Checkpoints and model compatibility</summary>

`--model-root` selects the parent of these weight directories. `src/twinpast/upstreams.json` records public repository revisions. An existing checkout with a different revision is reported and left unchanged.

| Model | Weight subdirectories | Upstream registry name | Transformers compatibility |
| --- | --- | --- | --- |
| Dispider | `Dispider` | `eval_dispider` | [4.41.2](src/twinpast/adapters/eval_dispider/requirements.txt) |
| MOSS-VL Realtime | `MOSS-VL-Realtime` | Checkpoint code | Checkpoint requirements |
| VideoLLM-online | `videollm-online-8b-v1plus`, `Meta-Llama-3-8B-Instruct`, `siglip-large-patch16-384` | `eval_videollm_online` | Upstream requirements |
| Flash-VStream | `Flash-VStream-Qwen-7b` | `eval_flash_vstream` | [4.45.0](src/twinpast/adapters/eval_flash_vstream/requirements.txt) |
| MiniCPM-o 4.5 | `MiniCPM-o-4_5` | Checkpoint code | [4.51.0](src/twinpast/adapters/eval_minicpm_o45/requirements.txt) |
| Mage-VL | `Mage-VL` | Checkpoint code | Checkpoint requirements |
| JoyAI-VL Interaction | `JoyAI-VL-Interaction`; streaming also `Qwen3-VL-4B` | `eval_joyai_vl` | Qwen3-VL-compatible checkpoint requirements |
| StreamForest | `StreamForest-Qwen2-7B`, `siglip-so400m-patch14-384` | `eval_streamforest` | [4.46.0](src/twinpast/adapters/eval_streamforest/requirements.txt) |
| ReKV | `llava-onevision-qwen2-7b-ov-hf` | `eval_rekv` | Upstream's pinned Transformers source |
| OneStreamer | `OneStreamer-4B` | `eval_onestreamer` | [4.57.6](src/twinpast/adapters/eval_onestreamer/requirements.txt) |
| AURA | `AURA-8B` | `eval_aura` | [4.57.1](src/twinpast/adapters/eval_aura/requirements.txt) |
| Qwen3-VL controls | `Qwen3-VL-8B` or configuration `checkpoint` | Transformers | Qwen3-VL-compatible release |

Dispider and MiniCPM include two-GPU placement logic; other adapters use automatic placement or their upstream engines. Most local adapters use BF16 and SDPA; Dispider retains its native FP16 path. Hardware and checkpoint compatibility must be checked for the selected model.

</details>

<details>
<summary>Observation, memory and readout settings</summary>

| Streaming condition | Observation and memory | Final readout |
| --- | --- | --- |
| `online/dispider` | 16-second windows, 16 frames/window, eight scheduler-history windows | Native reaction, free generation |
| `online/moss` | Acknowledged 1-fps realtime session, full-context grouped-query attention | Free generation |
| `online/videollm_mcq` | 2-fps LiveInfer history | Legal-option logits |
| `online/videollm_free` | Native LiveInfer history | Official free generation |
| `online/flash` | 1-fps visual-memory updates | Free generation, 128 tokens |
| `online/minicpm_mcq` | 1-fps duplex observation, persistent KV | Legal-option logits |
| `online/mage_codec_gate` | Eight-second codec segments and current-segment gate | Free generation |
| `online/joyai` | 1-fps proactive controller, summary memory, generation-call KV caching | Free generation, 128 tokens |
| `online/streamforest` | 1-fps compressed memory, 5,120 long-term tokens | Free generation, 1,024 tokens |
| `online/rekv` | 0.5-fps encoding, 15,000 local tokens, internal retrieval of 64 frames | Free generation, 256 tokens |
| `online/onestreamer` | 4-fps captions, one-second updates, 128-frame visual context, 12,000 text tokens; 16 recent QA frames at 1 fps | Free generation, 16 tokens |
| `online/aura` | Origin-anchored 2-fps one-second chunks; 30 retained video turns, trimming above 45 | Greedy generation, 128 tokens |

Native local replay expects 24-fps constant-frame-rate inputs and admits only source frames at or before `t_query`. Incremental observations have no global frame cap; model-specific processors retain their own resizing and temporal grouping. AURA uses lossless causal chunks and recomputes the active context, without claiming cross-turn KV reuse.

Local `offline/dispider`, `offline/moss`, `offline/videollm_mcq`, `offline/flash`, `offline/minicpm` and `offline/joyai` use up to 64 uniform causal frames capped at 151,200 pixels before model processing. Dispider requires 64 frames for complete 16-frame clips. Offline free generation is greedy with 128 new tokens; VideoLLM uses constrained MCQ readout.

`offline/mage_codec` uses the sparse codec interface with canvas 64 and patch size 16. `offline/aura_tail30` uses end-aligned one-second chunks from up to the last 30 seconds, at 1 fps/minimum two temporal frames, with silent assistant turns and only a final answer. Streaming/offline AURA retain their respective released system prompts.

The generic Qwen and HTTP controls use `video.sampled_images`:

| Conditions | Input |
| --- | --- |
| `offline/qwen`, `offline/http64` | Up to 64 uniform causal frames |
| `recent/qwen4`, `recent/http4` | Four evenly spaced targets from max(0, query-4) to query, mapped to causal source indices |
| `recent/http1` | Latest causal frame |
| `blind/qwen`, `blind/http` | Question and options only |

Generic recent-four sampling is explicitly defined here and need not match other recent-window implementations. The default pixel cap is 151,200. Qwen controls use BF16, SDPA and greedy decoding with a configurable new-token limit (default 128). Model input/memory budgets differ across conditions.

</details>

## Hosted and custom models

Install `python -m pip install -e '.[http,video]'`. Supply API settings in a local JSON file outside the dataset, and the credential through `TWINPAST_API_KEY`. Credentials cannot appear in configuration JSON or API URLs; configuration files are not copied to outputs.

```json
{
  "base_url": "https://api.example.invalid/v1",
  "model": "your-model-id",
  "generation": {"temperature": 0, "max_tokens": 256}
}
```

Replace the example address/model with your provider's documented values:

```bash
twinpast run ../huggingface --condition offline/http64 \
  --config /path/to/local-config.json --output-dir /tmp/twinpast-http64
```

<details>
<summary>HTTP, Google Files and realtime interfaces</summary>

| Condition | Input or protocol |
| --- | --- |
| `blind/http`, `recent/http1`, `recent/http4`, `offline/http64` | Controls defined above; chronological JPEGs at quality 95 |
| `api/chat_video` | Original bytes as a chat `video_url` part |
| `api/responses_video` | Original bytes as an `input_video` part |
| `api/google_files` | Resumable Files upload, then generateContent |
| `online/realtime` | Qwen-compatible native WebSocket video/audio-buffer protocol |

Configure generation parameters supported by your endpoint. HTTP image controls accept `pixel_cap`; the paper's DeepSeek recent-frame settings use 602112. Responses-style services use `max_output_tokens` (default 128), rather than `max_tokens`. Native video content types are provider extensions and are not supported by every compatible endpoint. Whole-video clients reject audio tracks, variable frame rate and post-query frames. The inline-video limit defaults to 512 MiB (`max_video_bytes`). There is no automatic transcoding or paid-request retry.

For Google Files, set `base_url` to `https://generativelanguage.googleapis.com`, choose the model ID, and use generateContent parameter names such as `maxOutputTokens`. Uploads are polled until ACTIVE (processing timeout 900 seconds) and deleted afterward unless `keep_upload` is true. Only the file created by that request is deleted. See the upstream [video guide](https://ai.google.dev/gemini-api/docs/video-understanding) and [Files API](https://ai.google.dev/api/files).

Realtime requires `python -m pip install -e '.[realtime]'` and a configuration containing `endpoint` (the caller's complete credential-free WSS/workspace address) and `model`. The client appends the model query parameter and creates a new socket for each branch. It sends causal frames at 1 fps, padded to 640 by 360 as quality-95 JPEGs, alongside one second of zero-valued 16-kHz PCM per frame. Source audio is omitted. Every ten frames, including a final shorter segment, it commits and generates text observations at temperature 0.7 with a 20-token limit. The final question arrives after all frames, at temperature 0 with a 128-token limit. The default system instruction is `Streaming Omni Conversation.` Search, tools, transcription, audio output, VAD and video compaction are disabled. Intermediate model text stays in the native conversation; the provider's context limits apply. See the provider's [realtime documentation](https://www.alibabacloud.com/help/en/model-studio/realtime).

</details>

For custom adapters, use `--adapter package.module:Class`. The constructor receives configuration; `answer_stream(job, prompt, emit)` returns a dictionary containing a string `response`. This is the video-job interface; whole-video HTTP inference remains an offline setting. Adapters must not read the answer key.

## Scoring

Predictions contain one JSON object per branch:

```json
{"pair_id":"twinpast_0001","branch":"A","answer":"B"}
```

The scorer requires public pair IDs, branch `A`/`B`, and an uppercase option letter. Extra prediction fields are ignored. Unknown IDs/branches, duplicate records/JSON keys and nonstandard JSON numbers are rejected. Invalid or missing answers count as wrong, with fixed denominators of 700 branches and 350 pairs, plus the corresponding task denominators.

| Metric | Definition for N pairs |
| --- | --- |
| `branch_accuracy` (Acc.) | Correct branches / 2N |
| `pair_accuracy` (PairAcc) | Both-correct pairs / N |
| `D` | Pairs with valid, different predictions / N |
| `C` | Both-correct / valid-different pairs; null when no such pairs exist |
| `R` | Fully reversed pairs (pA=yB and pB=yA) / N |
| `S` | PairAcc minus R |

Results include counts, overall and per-task metrics. Values are fractions; multiply by 100 for percentages. Score each repetition separately and average its ratios. Partial runs are smoke tests.

The runner's default `exact_letter` parser accepts only one legal uppercase letter after stripping whitespace. `--readout first_standalone_letter` explicitly selects a more permissive parser that can misinterpret prose. Raw final responses are retained for separately declared extraction rules. Constrained MCQ selects a legal option token directly. Paper free-generation results use model-specific extraction, including option-content matching where supported; the public scorer evaluates supplied letters and does not reproduce semantic extraction automatically. Never choose a parsing rule using an example's reference answer.

## Checks and runtime boundaries

```bash
python -B tools/check_release.py
python -B -m unittest discover -s tests -v
```

`public-files.txt` is the explicit publication list. The release checker rejects unlisted files, credentials it detects, local artifacts, links, binary files and personal filesystem paths. It checks the current files and Git index, including already tracked files that `.gitignore` cannot remove; past commits need separate review. Console output omits credential values and potentially sensitive filenames. Keep configs, runs, weights and downloaded upstream code outside the published source. Static checks cannot prove that every possible secret format is absent.

Core installation, validation and scoring do not load model code or call providers. Fetching upstream code and running inference are explicit commands. Model adapters and custom adapters execute Python with the process's permissions, including checkpoint remote code and upstream history definitions. Use trusted sources and separate model environments. Compatibility pins can lag security fixes and should not be used as public-service environments. Use maintained Python and patched media/HTTP packages for deployment; the HTTP extra requires Requests 2.32.4+ ([upstream advisory](https://github.com/psf/requests/security/advisories/GHSA-9hjg-9r4m-mvj7)).

HTTP uses HTTPS except explicitly configured loopback HTTP; realtime uses WSS. Redirects are disabled and Google upload URLs must retain the API origin. Environment proxy settings are disabled unless `trust_env` is true; for HTTP that also opts into Requests' netrc handling. Run outputs contain caller runtime paths, sampling metadata, predictions and model observations. The runner redacts credential fields and common forms of the current API key; inspect third-party console logs before sharing them. The release checker does not read account credential stores.

Tests cover the data interface, metrics, label exclusion, failure handling, credential handling and mocked model/API protocols. They do not download weights, make paid calls or establish numerical reproduction. Local model A/B branches share a base seed plus CRC32 of the public pair ID; realtime also uses a paired public-ID seed. Changing IDs, nondeterministic kernels or hosted service behavior can change results. Report checkpoint, environment, input mode, generation parameters, readout and seed alongside scores.

## Layout and license

- `src/twinpast/`: data tools, inference, scoring and model adapters; adapter dependencies live beside the corresponding code.
- `tests/`: core and mocked integration checks.
- `tools/`: the publication checker.

Our code is licensed under [Apache-2.0](LICENSE). Third-party software retains its own terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). This code license does not grant rights to videos, checkpoints or hosted services.
