import os
# os.environ["HF_HOME"] = "./weights-cache"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from threading import Thread
import time
from transformers import TextIteratorStreamer
import torch
from utils import maybe_download_with_pget
from cog import BasePredictor, ConcatenateIterator, Input
model_id='hamel/hc-mistral-qlora-6'

DEFAULT_MAX_NEW_TOKENS = os.environ.get("DEFAULT_MAX_NEW_TOKENS", 5000)
DEFAULT_TEMPERATURE = os.environ.get("DEFAULT_TEMPERATURE", 0.7)
DEFAULT_TOP_P = os.environ.get("DEFAULT_TOP_P", 0.95)
DEFAULT_TOP_K = os.environ.get("DEFAULT_TOP_K", 50)

TORCH_DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_MODEL_ID = "mistralai/Mistral-7B-v0.1"

def prompt(nlq, cols):
    return f"""[INST] <<SYS>>
Honeycomb AI suggests queries based on user input and candidate columns.
<</SYS>>

User Input: {nlq}

Candidate Columns: {cols}
[/INST]
"""

WEIGHTS_URL = "https://weights.replicate.delivery/default/mistral-7b-0.1"
REMOTE_FILES = [
    "config.json",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
    "model.safetensors.index.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json"
]

class Predictor(BasePredictor):
    task = "text-generation"
    base_model_id = BASE_MODEL_ID
    gcp_bucket_weights = True
    cache_dir = "./hf-cache"
    torch_dtype = "bf16"

    def setup(self):
        print("Starting setup")

        if self.gcp_bucket_weights:
            start = time.time()
            maybe_download_with_pget(
                self.base_model_id, 
                WEIGHTS_URL, 
                REMOTE_FILES
            )
            print(f"downloading weights took {time.time() - start:.3f}s")
        
        # os.environ["TRANSFORMERS_CACHE"] = self.cache_dir
        global TextIteratorStreamer
        from transformers import (
            AutoTokenizer,
            TextIteratorStreamer,
        )
        from peft import AutoPeftModelForCausalLM

        # resolve torch dtype from string.
        torch_dtype = TORCH_DTYPE_MAP[self.torch_dtype]
        self.model = AutoPeftModelForCausalLM.from_pretrained(model_id).cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def predict(
        self,
        nlq: str,
        cols: str,
        max_new_tokens: int = Input(
            description="The maximum number of tokens the model should generate as output.",
            default=DEFAULT_MAX_NEW_TOKENS,
        )
    ) -> ConcatenateIterator:
        _prompt = prompt(nlq, cols)
        print(f'\n=================\nThis is the prompt:\n{_prompt}\n')
        inputs = self.tokenizer(
            [_prompt], return_tensors="pt", truncation=True, return_token_type_ids=False
        ).to(DEVICE)
        streamer = TextIteratorStreamer(self.tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)
        generate_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        t = Thread(target=self.model.generate, kwargs=generate_kwargs)
        t.start()

        for text in streamer:
            yield text


if __name__ == "__main__":
    p = Predictor()
    p.setup()
    for text in p.predict(
        nlq="Exception count by exception and caller", 
        cols=['error', 'exception.message', 'exception.type', 'exception.stacktrace', 'SampleRate', 'name', 'db.user', 'type', 'duration_ms', 'db.name', 'service.name', 'http.method', 'db.system', 'status_code', 'db.operation', 'library.name', 'process.pid', 'net.transport', 'messaging.system', 'rpc.system', 'http.target', 'db.statement', 'library.version', 'status_message', 'parent_name', 'aws.region', 'process.command', 'rpc.method', 'span.kind', 'serializer.name', 'net.peer.name', 'rpc.service', 'http.scheme', 'process.runtime.name', 'serializer.format', 'serializer.renderer', 'net.peer.port', 'process.runtime.version', 'http.status_code', 'telemetry.sdk.language', 'trace.parent_id', 'process.runtime.description', 'span.num_events', 'messaging.destination', 'net.peer.ip', 'trace.trace_id', 'telemetry.instrumentation_library', 'trace.span_id', 'span.num_links', 'meta.signal_type', 'http.route'],
        max_new_tokens=1024
    ):
        print(text, end="")
