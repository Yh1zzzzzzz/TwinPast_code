"""Google Files upload followed by native generateContent video inference."""
import time
from urllib.parse import quote, urlsplit

from twinpast.data import DataError, require
from twinpast.video import complete_video
from .http_video import Adapter as HTTPAdapter


class Adapter(HTTPAdapter):
    def answer_stream(self, job, prompt, emit):
        path = complete_video(job)
        headers = {"x-goog-api-key": self.key}
        started = self.request("POST", self.base + "/upload/v1beta/files", headers={**headers,
                               "X-Goog-Upload-Protocol": "resumable", "X-Goog-Upload-Command": "start",
                               "X-Goog-Upload-Header-Content-Length": str(path.stat().st_size),
                               "X-Goog-Upload-Header-Content-Type": "video/mp4"},
                               json={"file": {"display_name": "video.mp4"}})
        upload = started.headers.get("X-Goog-Upload-URL")
        require(isinstance(upload, str), "Provider returned no resumable upload URL")
        parsed = urlsplit(upload)
        origin = urlsplit(self.base)
        require((parsed.scheme, parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)) ==
                (origin.scheme, origin.hostname, origin.port or (443 if origin.scheme == "https" else 80))
                and not parsed.username and not parsed.password and not parsed.fragment,
                "Upload origin differs from API origin")
        with path.open("rb") as stream:
            uploaded = self.request("POST", upload, headers={"X-Goog-Upload-Offset": "0",
                                      "X-Goog-Upload-Command": "upload, finalize"}, data=stream).json()["file"]
        name = uploaded["name"]
        require(name.startswith("files/") and all(c.isalnum() or c in "/-_" for c in name), "Invalid uploaded file name")
        try:
            deadline = time.monotonic() + self.config.get("processing_timeout", 900)
            while uploaded.get("state") != "ACTIVE":
                require(uploaded.get("state") != "FAILED", "Video processing failed")
                require(time.monotonic() < deadline, "Video processing timed out")
                time.sleep(2)
                uploaded = self.request("GET", self.base + "/v1beta/" + name, headers=headers).json()
            payload = {"contents": [{"role": "user", "parts": [{"text": prompt},
                       {"fileData": {"mimeType": "video/mp4", "fileUri": uploaded["uri"]}}]}],
                       "generationConfig": self.config.get("generation", {"temperature": 0, "maxOutputTokens": 128})}
            raw = self.request("POST", self.base + "/v1beta/models/" + quote(self.model, safe="") + ":generateContent",
                               headers=headers, json=payload).json()
            candidates = raw.get("candidates") or []
            require(bool(candidates), "Provider returned no candidate answer")
            text = "".join(p.get("text", "") for p in candidates[0].get("content", {}).get("parts", []) if not p.get("thought", False))
            emit({"type": "input", "sampling": "native_video"})
            return {"response": text, "model": raw.get("modelVersion", self.model), "usage": raw.get("usageMetadata")}
        finally:
            if not self.config.get("keep_upload", False):
                try:
                    self.request("DELETE", self.base + "/v1beta/" + name, headers=headers)
                except DataError:
                    emit({"type": "upload_cleanup_failed"})

