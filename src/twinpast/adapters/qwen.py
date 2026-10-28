"""Qwen3-VL image-sequence and query-only controls."""
from pathlib import Path

from twinpast.video import sampled_images


class Adapter:
    def __init__(self, configuration):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        self.config = configuration
        path = Path(configuration["model_root"]) / configuration.get("checkpoint", "Qwen3-VL-8B")
        self.processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(path, local_files_only=True,
                       torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="auto").eval()

    def answer_stream(self, job, prompt, emit):
        import torch
        mode = self.config.get("input_mode", "uniform64")
        images, sampling = ([], {"sampling": "query_only"}) if mode == "query_only" else sampled_images(job, mode, self.config.get("pixel_cap", 151200))
        content = [{"type": "text", "text": prompt}] + [{"type": "image"} for _ in images]
        text = self.processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
        kwargs = {"text": [text], "return_tensors": "pt", "padding": False}
        if images:
            kwargs["images"] = images
        batch = self.processor(**kwargs).to(self.model.get_input_embeddings().weight.device)
        with torch.inference_mode():
            output = self.model.generate(**batch, do_sample=False, max_new_tokens=self.config.get("max_new_tokens", 128))
        ids = output[0, batch["input_ids"].shape[1]:]
        emit({"type": "sampling", **sampling})
        return {"response": self.processor.batch_decode([ids], skip_special_tokens=True)[0].strip(), "sampling": sampling}
