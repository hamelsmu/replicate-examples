import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
import torch
from cog import BasePredictor
from vllm import LLM

def prompt(nlq, cols):
    return f"""Honeycomb is an observability platform that allows you to write queries to inspect trace data. You are an assistant that takes a natural language query (NLQ) and a list of valid columns and produce a Honeycomb query.

### Instruction:

NLQ: "{nlq}"

Columns: {cols}

### Response:
"""

class Predictor(BasePredictor):
        
    def setup(self):
        self.llm = LLM(model_id='parlance-labs/hc-mistral-alpaca-merged', tensor_parallel_size=torch.cuda.device_count())

    def predict(self, nlq: str, cols: str) -> str:        
        _p = prompt(nlq, cols)
        out = self.llm.generate(_p)
        return out.strip().strip('"')

if __name__ == "__main__":
    p = Predictor()
    p.setup()
    return p.predict(
        nlq="Exception count by exception and caller", 
        cols=['error', 'exception.message', 'exception.type', 'exception.stacktrace', 'SampleRate', 'name', 'db.user', 'type', 'duration_ms', 'db.name', 'service.name', 'http.method', 'db.system', 'status_code', 'db.operation', 'library.name', 'process.pid', 'net.transport', 'messaging.system', 'rpc.system', 'http.target', 'db.statement', 'library.version', 'status_message', 'parent_name', 'aws.region', 'process.command', 'rpc.method', 'span.kind', 'serializer.name', 'net.peer.name', 'rpc.service', 'http.scheme', 'process.runtime.name', 'serializer.format', 'serializer.renderer', 'net.peer.port', 'process.runtime.version', 'http.status_code', 'telemetry.sdk.language', 'trace.parent_id', 'process.runtime.description', 'span.num_events', 'messaging.destination', 'net.peer.ip', 'trace.trace_id', 'telemetry.instrumentation_library', 'trace.span_id', 'span.num_links', 'meta.signal_type', 'http.route'],
    )
