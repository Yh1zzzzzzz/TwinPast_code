# SPDX-License-Identifier: Apache-2.0
"""AURA tail-window and sequential generated-history conditions."""
from .backend import Backend
from .media import prepare, benchmark_messages
from .official import StreamingHistory


class Adapter:
    def __init__(self, configuration):
        self.condition = configuration.get("input_mode", "streaming")
        if self.condition not in ("tail30", "streaming"):
            raise ValueError("Unknown AURA input mode")
        self.backend = Backend()

    def answer_stream(self, job, prompt, emit):
        prepared = prepare(job, self.condition)
        self.backend.reset_session(self.condition)
        if self.condition == "tail30":
            result = self.backend.generate(benchmark_messages(prepared, prompt))
            result["sampling"] = prepared["sampling"]
            return result
        history = StreamingHistory()
        result = None
        segments = prepared["segments"]
        for index, segment in enumerate(segments):
            question = prompt if index == len(segments) - 1 else ""
            history.add_chunk((segment["path"],), question)
            result = self.backend.generate(history.messages)
            history.add_response(result["response"])
            emit({"type": "turn", "turn": index, "question_present": bool(question),
                  "source_timestamps": segment["source_timestamps"], "history": history.state,
                  "response": result["response"]})
        if result is None:
            raise ValueError("No causal video segments")
        return {"response": result["response"], "turns": len(segments),
                "history": history.state, "cross_step_kv_cache": False,
                "sampling": prepared["sampling"]}

