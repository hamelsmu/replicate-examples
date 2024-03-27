import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
import torch
from uuid import uuid4
from cog import BasePredictor
from vllm import LLM, SamplingParams
from vllm import AsyncEngineArgs, AsyncLLMEngine
import time

MODEL_ID = 'parlance-labs/hc-mistral-alpaca-merged-awq'
MAX_TOKENS=2500

PROMPT_TEMPLATE = """Honeycomb is an observability platform that allows you to write queries to inspect trace data. You are an assistant that takes a natural language query (NLQ) and a list of valid columns and produce a Honeycomb query.

### Instruction:

NLQ: "{nlq}"

Columns: {cols}

### Response:
"""

class Predictor(BasePredictor):
        
    async def setup(self):
        n_gpus = torch.cuda.device_count()
        # you usually would let these be set by the user, but for this specific model, I know the best settings.
        self.sampling_params = SamplingParams(stop_token_ids=[2], temperature=0, ignore_eos=True, max_tokens=2500)

        ENGINE_ARGS = AsyncEngineArgs(
                        tensor_parallel_size=n_gpus,
                        quantization="AWQ"
                        model=MODEL_ID,
                        max_model_len=MAX_TOKENS
        )
        self.engine = AsyncLLMEngine.from_engine_args(ENGINE_ARGS)

        # self.llm = LLM(model='parlance-labs/hc-mistral-alpaca-merged-awq', 
        #                tensor_parallel_size=n_gpus, quantization="AWQ")

    async def predict(self, nlq: str, cols: str) -> str:
        start = time.time()
        request_id = uuid4().hex        
        _p = PROMPT_TEMPLATE.format(nlq=nlq, cols=cols)
        results_generator = self.engine.generate(_p, sampling_params=self.sampling_params, request_id=request_id, use_tqdm=False)
        async for request_output in results_generator:
            final_output = request_output
        return final_output.outputs[0].text.strip().strip('"')
