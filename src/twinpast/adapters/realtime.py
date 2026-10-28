"""Manual-turn video replay through the Qwen-compatible realtime protocol."""
import base64
import io
import json
import os
import time
from urllib.parse import quote, urlsplit
import uuid
import zlib

from twinpast.data import DataError, require
from twinpast.video import check_identity


class Adapter:
    def __init__(self, configuration):
        self.config = configuration
        self.endpoint = configuration.get("endpoint", "")
        try:
            parsed = urlsplit(self.endpoint)
            parsed.port
        except ValueError:
            raise DataError("Invalid realtime endpoint") from None
        require(parsed.scheme == "wss" and parsed.hostname and not parsed.query and not parsed.fragment
                and not parsed.username and not parsed.password, "Set a credential-free wss endpoint")
        self.model = configuration.get("model")
        require(isinstance(self.model, str) and bool(self.model), "Set the realtime model")
        self.key = os.environ.get("TWINPAST_API_KEY")
        require(bool(self.key), "Set TWINPAST_API_KEY")
        require(isinstance(configuration.get("trust_env", False), bool), "trust_env must be a boolean")
        self.timeout = configuration.get("timeout", 180)

    def send(self, event):
        event["event_id"] = "event_" + uuid.uuid4().hex
        self.socket.send(json.dumps(event))

    def wait(self, kind):
        import websocket
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            try:
                raw = self.socket.recv()
            except websocket.WebSocketTimeoutException:
                self.socket.ping()
                continue
            require(bool(raw), "Realtime connection closed")
            event = json.loads(raw)
            require(event.get("type") != "error" and not event.get("code"), "Realtime provider returned an error")
            if event.get("type") == kind:
                return event
            require(event.get("type") != "response.done", "Unexpected unsolicited realtime response")
        raise DataError("Realtime response timed out; no request was retried")

    def generate(self):
        self.send({"type": "response.create"})
        response = self.wait("response.done")["response"]
        require(response.get("status") in ("completed", "incomplete"), "Realtime generation failed")
        parts = []
        for item in response.get("output", []):
            require(item.get("type") == "message" and item.get("role") == "assistant", "Unexpected realtime output type")
            for part in item.get("content", []):
                require(part.get("type") in ("text", "output_text"), "Unexpected realtime content type")
                parts.append(part.get("text", ""))
        return {"response": "".join(parts), "usage": response.get("usage"), "status": response["status"]}

    def answer_stream(self, job, prompt, emit):
        import websocket
        from PIL import ImageOps
        from native import CausalFrames
        check_identity(job)
        proxy_options = {} if self.config.get("trust_env", False) else {"http_no_proxy": ["*"]}
        self.socket = websocket.create_connection(self.endpoint + "?model=" + quote(self.model, safe=""),
                        header=["Authorization: Bearer " + self.key], timeout=20, suppress_origin=True,
                        redirect_limit=0, **proxy_options)
        self.socket.settimeout(10)
        try:
            created = self.wait("session.created")["session"]
            require(created.get("model") == self.model, "Realtime model mismatch")
            seed = (self.config.get("seed", 0) + zlib.crc32(job["job_id"].rsplit("__", 1)[0].encode())) & 0x7fffffff
            session = {"modalities": ["text"], "turn_detection": None, "input_audio_transcription": None,
                       "instructions": self.config.get("system", "Streaming Omni Conversation."),
                       "temperature": 0.7, "max_tokens": 20, "seed": seed, "enable_search": False,
                       "audio": {"input": {"format": {"type": "pcm", "sample_rate": 16000,
                                  "sample_format": "s16le", "channels": 1, "packing": "interleaved", "channel_layout": "mono"}}},
                       "video": {"input": {"representation_compact": "none"}}}
            self.send({"type": "session.update", "session": session})
            updated = self.wait("session.updated")["session"]
            require(updated.get("turn_detection") is None and updated.get("enable_search") is not True
                    and updated.get("modalities") == ["text"] and updated.get("model") == self.model,
                    "Manual text-only session was not acknowledged")
            require(updated.get("temperature") == 0.7 and updated.get("max_tokens") == 20,
                    "Observation parameters were not acknowledged")
            silence = base64.b64encode(bytes(32000)).decode("ascii")
            count = 0
            last_sent = None
            observation_responses = []
            for image, timestamp in CausalFrames(job, fps=1.0):
                if last_sent is not None:
                    time.sleep(max(0, last_sent + 1 - time.monotonic()))
                frame = ImageOps.pad(image, (640, 360), color=(0, 0, 0))
                buffer = io.BytesIO()
                frame.save(buffer, format="JPEG", quality=95)
                self.send({"type": "input_audio_buffer.append", "audio": silence})
                self.send({"type": "input_image_buffer.append", "image": base64.b64encode(buffer.getvalue()).decode("ascii")})
                last_sent = time.monotonic()
                count += 1
                emit({"type": "frame", "timestamp_s": timestamp, "question_present": False})
                if count % 10 == 0:
                    self.send({"type": "input_audio_buffer.commit"})
                    self.wait("input_audio_buffer.committed")
                    observation_responses.append(self.generate())
            require(count > 0, "No realtime video frames")
            if count % 10:
                self.send({"type": "input_audio_buffer.commit"})
                self.wait("input_audio_buffer.committed")
                observation_responses.append(self.generate())
            self.send({"type": "session.update", "session": {"temperature": 0, "max_tokens": 128}})
            updated = self.wait("session.updated")["session"]
            require(updated.get("temperature") == 0 and updated.get("max_tokens") == 128,
                    "Final answer parameters were not acknowledged")
            self.send({"type": "conversation.item.create", "item": {"type": "message", "role": "user",
                       "content": [{"type": "input_text", "text": prompt}]}})
            self.wait("conversation.item.created")
            result = self.generate()
            return dict(result, frames_received=count, observation_responses=observation_responses,
                        source_audio=False, silence_seconds=count, replay_fps=1.0)
        finally:
            self.socket.close()
