"""Native video or sampled-image HTTP clients with environment-only credentials."""
import base64
import io
import os
from urllib.parse import urlsplit

from twinpast.data import DataError, require
from twinpast.video import complete_video, sampled_images


def endpoint(value):
    require(isinstance(value, str), "Set base_url in the model configuration")
    try:
        parsed = urlsplit(value)
        parsed.port
    except ValueError:
        raise DataError("Invalid API base_url") from None
    require(parsed.scheme == "https" or (parsed.scheme == "http" and parsed.hostname in ("127.0.0.1", "localhost")),
            "API base_url must use HTTPS")
    require(parsed.hostname and not parsed.username and not parsed.password and not parsed.query and not parsed.fragment,
            "API base_url cannot contain credentials or query parameters")
    return value.rstrip("/")


class Adapter:
    def __init__(self, configuration):
        import requests
        self.config = configuration
        self.base = endpoint(configuration.get("base_url"))
        self.model = configuration.get("model")
        require(isinstance(self.model, str) and bool(self.model), "Set model in the API configuration")
        self.key = os.environ.get("TWINPAST_API_KEY")
        require(bool(self.key), "Set TWINPAST_API_KEY")
        require(isinstance(configuration.get("trust_env", False), bool), "trust_env must be a boolean")
        self.session = requests.Session()
        self.session.trust_env = bool(configuration.get("trust_env", False))
        self.timeout = configuration.get("timeout", 900)

    def close(self):
        self.session.close()

    def request(self, method, url, **kwargs):
        try:
            response = self.session.request(method, url, timeout=self.timeout, allow_redirects=False, **kwargs)
        except Exception:
            raise DataError("API transport failed; completion status is unknown and no request was retried") from None
        if not 200 <= response.status_code < 300:
            raise DataError("API returned HTTP {}".format(response.status_code))
        return response

    def content(self, job, prompt, responses=False):
        mode = self.config.get("input_mode", "video")
        require(mode in ("video", "query_only", "uniform64", "recent4", "latest"), "Unknown API input_mode")
        if mode == "query_only":
            images, metadata = [], {"sampling": "query_only"}
        elif mode in ("uniform64", "recent4", "latest"):
            images, metadata = sampled_images(job, mode, self.config.get("pixel_cap", 151200))
        else:
            path = complete_video(job)
            limit = self.config.get("max_video_bytes", 512 * 1024 * 1024)
            require(path.stat().st_size <= limit, "Video exceeds configured request size limit")
            uri = "data:video/mp4;base64," + base64.b64encode(path.read_bytes()).decode("ascii")
            item = {"type": "input_video", "video_url": uri} if responses else {"type": "video_url", "video_url": {"url": uri}}
            return ([{"type": "input_text" if responses else "text", "text": prompt}, item], {"sampling": "native_video"})
        content = [{"type": "input_text" if responses else "text", "text": prompt}]
        for image in images:
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=95)
            uri = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
            content.append({"type": "input_image", "image_url": uri} if responses else {"type": "image_url", "image_url": {"url": uri}})
        return content, metadata

    def answer_stream(self, job, prompt, emit):
        transport = self.config.get("transport", "chat")
        require(transport in ("chat", "responses"), "Unknown API transport")
        responses = transport == "responses"
        content, metadata = self.content(job, prompt, responses)
        payload = {"model": self.model, "input" if responses else "messages": [{"role": "user", "content": content}]}
        generation = self.config.get("generation", {"temperature": 0, "max_output_tokens" if responses else "max_tokens": 128})
        require(isinstance(generation, dict) and not set(generation) & {"model", "messages", "input", "stream"},
                "generation cannot replace model inputs")
        payload.update(generation)
        raw = self.request("POST", self.base + ("/responses" if responses else "/chat/completions"),
                           json=payload, headers={"Authorization": "Bearer " + self.key}).json()
        if responses:
            text = "\n".join(part.get("text", "") for item in raw.get("output", []) if item.get("type") == "message"
                             for part in item.get("content", []) if part.get("type") == "output_text")
        else:
            choices = raw.get("choices") or []
            require(bool(choices), "Provider returned no answer choices")
            text = choices[0].get("message", {}).get("content")
            if isinstance(text, list):
                text = "\n".join(p.get("text", "") for p in text if p.get("type") in ("text", "output_text"))
        require(isinstance(text, str), "Provider did not return a text answer")
        emit({"type": "input", **metadata})
        return {"response": text, "model": raw.get("model", self.model), "usage": raw.get("usage"), "sampling": metadata}

