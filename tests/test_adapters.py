import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import support
from twinpast.data import DataError
from twinpast.adapters.http_video import Adapter as HTTP, endpoint
from twinpast.adapters.google_video import Adapter as Google
from twinpast.video import complete_video


class Response:
    def __init__(self, value=None, headers=None, status=200):
        self.value = value or {}
        self.headers = headers or {}
        self.status_code = status
    def json(self):
        return self.value


class HTTPTests(unittest.TestCase):
    def adapter(self, config=None):
        obj = HTTP.__new__(HTTP)
        obj.config = config or {}
        obj.model, obj.key, obj.base, obj.timeout = "example-model", "not-a-live-key", "https://api.example.invalid", 20
        return obj

    def test_endpoint_rejects_credential_urls(self):
        for value in ("https://user:pass@example.invalid", "https://api.example.invalid?key=value", "http://remote.example.invalid"):
            with self.assertRaises(DataError):
                endpoint(value)
        self.assertEqual(endpoint("https://api.example.invalid/v1/"), "https://api.example.invalid/v1")

    def test_invalid_url_error_omits_supplied_port_value(self):
        with self.assertRaises(DataError) as caught:
            endpoint("https://api.example.invalid:not-a-live-key")
        self.assertNotIn("not-a-live-key", str(caught.exception))

    def test_chat_query_only_sends_no_media_or_labels(self):
        adapter = self.adapter({"input_mode": "query_only"})
        requests = []
        def request(method, url, **kwargs):
            requests.append(kwargs)
            return Response({"choices": [{"message": {"content": "B"}}]})
        adapter.request = request
        result = adapter.answer_stream({}, "Which?\nA. Cup\nB. Book", lambda event: None)
        payload = requests[0]["json"]
        self.assertEqual(result["response"], "B")
        self.assertEqual(payload["messages"][0]["content"], [{"type": "text", "text": "Which?\nA. Cup\nB. Book"}])
        self.assertNotIn("not-a-live-key", json.dumps(payload))

    def test_native_video_uses_original_bytes_and_responses_transport(self):
        adapter = self.adapter({"transport": "responses"})
        calls = []
        adapter.request = lambda method, url, **kw: (calls.append((url, kw)) or Response({"output": [{"type": "message", "content": [{"type": "output_text", "text": "A"}]}]}))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "video.mp4"
            path.write_bytes(b"video bytes")
            with patch("twinpast.adapters.http_video.complete_video", return_value=path):
                result = adapter.answer_stream({}, "question", lambda event: None)
        self.assertEqual(result["response"], "A")
        self.assertTrue(calls[0][0].endswith("/responses"))
        self.assertEqual(calls[0][1]["json"]["max_output_tokens"], 128)
        self.assertNotIn("max_tokens", calls[0][1]["json"])
        self.assertEqual([v["type"] for v in calls[0][1]["json"]["input"][0]["content"]], ["input_text", "input_video"])

    def test_unknown_transports_and_input_modes_fail_before_requests(self):
        for config in ({"transport": "unknown"}, {"input_mode": "unknown"}):
            adapter = self.adapter(config)
            adapter.request = lambda *args, **kwargs: self.fail("Unexpected request")
            with self.assertRaises(DataError):
                adapter.answer_stream({}, "question", lambda event: None)

    def test_http_errors_do_not_expose_provider_error_body_or_retry(self):
        adapter = self.adapter()
        calls = []
        adapter.session = SimpleNamespace(request=lambda *args, **kwargs: (calls.append(args) or Response({"secret": "should not appear"}, status=403)))
        with self.assertRaisesRegex(DataError, "HTTP 403") as caught:
            adapter.request("POST", adapter.base)
        self.assertNotIn("should not appear", str(caught.exception))
        self.assertEqual(len(calls), 1)

    def test_google_upload_poll_generate_and_cleanup(self):
        adapter = Google.__new__(Google)
        adapter.config, adapter.key, adapter.base, adapter.model = {}, "not-a-live-key", "https://api.example.invalid", "example-model"
        calls = []
        def request(method, url, **kwargs):
            calls.append((method, url, kwargs))
            if url.endswith("/upload/v1beta/files"):
                return Response(headers={"X-Goog-Upload-URL": adapter.base + "/upload?upload_id=example"})
            if "upload_id=" in url:
                return Response({"file": {"name": "files/example", "state": "ACTIVE", "uri": "https://files.example.invalid/video"}})
            if method == "DELETE":
                return Response()
            return Response({"candidates": [{"content": {"parts": [{"text": "private reasoning", "thought": True}, {"text": "A"}]}}]})
        adapter.request = request
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "video.mp4"
            path.write_bytes(b"video bytes")
            with patch("twinpast.adapters.google_video.complete_video", return_value=path):
                result = adapter.answer_stream({}, "question", lambda event: None)
        self.assertEqual(result["response"], "A")
        self.assertEqual(calls[-1][0], "DELETE")
        self.assertEqual(calls[2][2]["json"]["contents"][0]["parts"][0], {"text": "question"})

    def test_whole_video_rejects_post_query_frames_and_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "video.mp4"
            path.write_bytes(b"video bytes")
            st = path.stat()
            job = {"video": str(path), "video_stat": {"size": st.st_size, "inode": st.st_ino, "mtime_ns": st.st_mtime_ns}, "t_query": 1}
            streams = [{"codec_type": "video", "nb_frames": "100", "avg_frame_rate": "24/1", "r_frame_rate": "24/1"}]
            with patch("twinpast.video.subprocess.run", return_value=SimpleNamespace(stdout=json.dumps({"streams": streams}))):
                with self.assertRaises(DataError):
                    complete_video(job)
            streams[0]["nb_frames"] = "24"
            streams.append({"codec_type": "audio"})
            with patch("twinpast.video.subprocess.run", return_value=SimpleNamespace(stdout=json.dumps({"streams": streams}))):
                with self.assertRaises(DataError):
                    complete_video(job)

    def test_google_upload_rejects_changed_origin_before_sending_file(self):
        adapter = Google.__new__(Google)
        adapter.config, adapter.key, adapter.base = {}, "not-a-live-key", "https://api.example.invalid"
        for url in ("https://different.example.invalid/upload", "https://api.example.invalid:444/upload"):
            calls = []
            adapter.request = lambda *args, **kwargs: (calls.append(args) or Response(headers={"X-Goog-Upload-URL": url}))
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "video.mp4"
                path.write_bytes(b"video bytes")
                with patch("twinpast.adapters.google_video.complete_video", return_value=path):
                    with self.assertRaises(DataError):
                        adapter.answer_stream({}, "question", lambda event: None)
            self.assertEqual(len(calls), 1)


class AURATests(unittest.TestCase):
    def test_question_is_sent_only_on_final_turn(self):
        from twinpast.adapters.eval_aura import adapter as aura
        calls = []
        class History:
            def __init__(self): self.messages = []
            def add_chunk(self, video, text): self.messages.append({"video": video, "text": text})
            def add_response(self, text): self.messages.append({"response": text})
            @property
            def state(self): return {"rounds": len(self.messages) // 2}
        class Backend:
            def reset_session(self, condition): pass
            def generate(self, messages):
                calls.append(list(messages))
                return {"response": "A"}
        obj = aura.Adapter.__new__(aura.Adapter)
        obj.condition, obj.backend = "streaming", Backend()
        prepared = {"segments": [{"path": "first.mp4", "source_timestamps": [0]}, {"path": "second.mp4", "source_timestamps": [1]}], "sampling": "causal"}
        with patch.object(aura, "prepare", return_value=prepared), patch.object(aura, "StreamingHistory", History):
            result = obj.answer_stream({}, "final question", lambda event: None)
        self.assertEqual(calls[0][0]["text"], "")
        self.assertEqual(calls[1][-1]["text"], "final question")
        self.assertEqual(result["turns"], 2)


@unittest.skipUnless(importlib.util.find_spec("PIL"), "Optional realtime image dependency is absent")
class RealtimeTests(unittest.TestCase):
    def test_manual_protocol_withholds_question_until_all_frames(self):
        from twinpast.adapters import realtime
        sent = []
        updates = {"model": "example-model", "modalities": ["text"], "turn_detection": None, "enable_search": False, "temperature": .7, "max_tokens": 20}
        events = [
            {"type": "session.created", "session": {"model": "example-model"}},
            {"type": "session.updated", "session": updates},
            {"type": "input_audio_buffer.committed"},
            {"type": "response.done", "response": {"status": "completed", "output": []}},
            {"type": "session.updated", "session": {"temperature": 0, "max_tokens": 128}},
            {"type": "conversation.item.created"},
            {"type": "response.done", "response": {"status": "completed", "output": [{"type": "message", "role": "assistant", "content": [{"type": "text", "text": "B"}]}]}},
        ]
        socket = SimpleNamespace(send=lambda value: sent.append(json.loads(value)), recv=lambda: json.dumps(events.pop(0)), settimeout=lambda value: None, close=lambda: None)
        websocket = SimpleNamespace(create_connection=lambda *args, **kwargs: socket, WebSocketTimeoutException=TimeoutError)
        image = SimpleNamespace(save=lambda buffer, **kwargs: buffer.write(b"image"))
        obj = realtime.Adapter.__new__(realtime.Adapter)
        obj.config, obj.endpoint, obj.model, obj.key, obj.timeout = {}, "wss://api.example.invalid", "example-model", "not-a-live-key", 10
        with patch.dict("sys.modules", {"websocket": websocket, "native": SimpleNamespace(CausalFrames=lambda job, fps: [(image, 0), (image, 1)])}), \
             patch.object(realtime, "check_identity"), patch("PIL.ImageOps.pad", side_effect=lambda value, *args, **kwargs: value), patch.object(realtime.time, "sleep"):
            result = obj.answer_stream({"job_id": "twinpast_0001__A"}, "final question", lambda event: None)
        types = [event["type"] for event in sent]
        query = types.index("conversation.item.create")
        self.assertEqual(types[:query].count("input_image_buffer.append"), 2)
        self.assertNotIn("final question", json.dumps(sent[:query]))
        self.assertEqual(result["response"], "B")
        self.assertFalse(result["source_audio"])


if __name__ == "__main__":
    unittest.main()
