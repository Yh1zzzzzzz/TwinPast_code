# Third-party software and model terms

The Apache-2.0 license at the repository root applies to our evaluation integration code. It does not relicense third-party software, checkpoint remote code, weights, datasets or hosted services. Upstream repositories and model weights are downloaded separately by the user. Their own LICENSE/NOTICE files and model cards remain authoritative.

The public source registry is `src/twinpast/upstreams.json`; it records repository URLs and pinned revisions. Registered software includes Dispider, Flash-VStream, JoyAI-VL Interaction, VideoLLM-online, StreamForest, ReKV, OneStreamer and AURA. Check the terms at the registered revision before installing or redistributing any upstream component. For ReKV, do not infer a redistribution grant from this project's license; inspect the upstream terms separately.

Checkpoint integrations additionally use MOSS-VL Realtime, MiniCPM-o 4.5, Mage-VL, Qwen3-VL, Llama and SigLIP publishers' software and weights. Their licenses and access conditions are separate. Runtime dependencies include Transformers, PyTorch, Accelerate, NumPy, Pillow, Decord, Requests, websocket-client and FFmpeg, when selected by an adapter. Their notices accompany the respective installed distributions. FFmpeg build-specific codec terms continue to apply.

OneStreamer's upstream notice identifies original OneStreamer code as Apache-2.0, copyright 2026 OneStreamer contributors. Its evaluation tree also retains VLMEvalKit and ViSpeak-Bench under their original terms. Those trees are not bundled in this source distribution. Preserve their applicable notices if you redistribute them after fetching upstream code.

No third-party attribution in this document extends the code license to benchmark videos or model checkpoints.
