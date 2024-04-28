import os, random, asyncio, time, json
from typing import AsyncIterator, List, Union
from cog import BasePredictor, Input, ConcatenateIterator
from vllm import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams
# from vllm.model_executor.guided_decoding.outlines_logits_processors import JSONLogitsProcessor
import torch
from utils import maybe_download_with_pget
from typing import Optional
from functools import partial

from dotenv import load_dotenv
load_dotenv()

MODEL_ID = os.getenv("MODEL_ID")
WEIGHTS_URL = os.getenv("COG_WEIGHTS")
REMOTE_FILES = [
    "config.json",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json"
]
# LLama 3 Prompt
# https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/
PROMPT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{sys_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

<question>{prompt}</question><|eot_id|><|start_header_id|>assistant<|end_header_id|>

{prefill}"""

FUNC_PROMPT = """You are a hepful assistant. You are given a question inside <question> tags and a set of possible functions inside <function-definitions> tags.  
Calling these functions are optional. Carefully consider the question and determine if one or more functions can be used to answer the question. Place your thoughts and reasoning behind your decision in <function-thoughts> tags.
If the given question lacks the parameters required by the function, point it out in <function-thoughts> tags. Below is a list of function definitions:
<function-definitions>
{funcs}
</function-definitions>

If you wish to call a particular function, specify the name of the function and any arguments in a way that conforms to that function's schema inside <function-call> tags.
Function calls should be in this format: <function-thoughts>Calling func1 would be helpful because of ...</function-thoughts><function-call>[func1(params_name=params_value, params_name2=params_value2...), func2(params)]</function-call>, WITHOUT any answer.
If you do not wish to call any functions, say so in the <function-thoughts> tags followed by <function-call>None</function-call><answer>...</answer>

If and only if NO function calls are made, answer the question to the best of your ability inside <answer> tags.  If you are unsure of the answer, say so in <answer> tags.
"""

DEFAULT_MAX_NEW_TOKENS = os.getenv("DEFAULT_MAX_NEW_TOKENS", 1024)
DEFAULT_TEMPERATURE = os.getenv("DEFAULT_TEMPERATURE", 0.5)
DEFAULT_TOP_P = os.getenv("DEFAULT_TOP_P", 0.9)
DEFAULT_TOP_K = os.getenv("DEFAULT_TOP_K", 50)
DEFAULT_PRESENCE_PENALTY = os.getenv("DEFAULT_PRESENCE_PENALTY", 0.0)  # 1.15
DEFAULT_FREQUENCY_PENALTY = os.getenv("DEFAULT_FREQUENCY_PENALTY", 0.0)  # 0.2


def format_tools(tools):
    result = ""
    for index, tool in enumerate(tools, start=1):
        result += f"<function-{index}>\n"
        result += json.dumps(tool, indent=4)
        result += f"\n</function-{index}>\n\n"
    return result


class VLLMPipeline:
    """
    A simplified inference engine that runs inference w/ vLLM
    """

    def __init__(self, *args, **kwargs) -> None:
        args = AsyncEngineArgs(*args, **kwargs)
        self.engine = AsyncLLMEngine.from_engine_args(args)
        self.tokenizer = (
            self.engine.engine.tokenizer.tokenizer 
            if hasattr(self.engine.engine.tokenizer, "tokenizer") 
            else self.engine.engine.tokenizer
        )

    async def generate_stream(
        self, prompt: str, sampling_params: SamplingParams
    ) -> AsyncIterator[str]:
        results_generator = self.engine.generate(
            prompt, sampling_params, str(random.random())
            )
        async for generated_text in results_generator:
            yield generated_text

    async def __call__(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        stop_sequences: Union[str, List[str]] = '<|eot_id|>',
        stop_token_ids: List[int] = None,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        incremental_generation: bool = True,
        tools: Optional[List[str]] = None,
    ) -> str:
        """
        Given a prompt, runs generation on the language model with vLLM.
        """
        if top_k is None or top_k == 0:
            top_k = -1

        stop_token_ids = stop_token_ids or []
        stop_token_ids.append(self.tokenizer.eos_token_id)

        if isinstance(stop_sequences, str) and stop_sequences != "":
            stop = [stop_sequences]
        elif isinstance(stop_sequences, list) and len(stop_sequences) > 0:
            stop = stop_sequences
        else:
            stop = []

        for tid in stop_token_ids:
            stop.append(self.tokenizer.decode(tid))

        sampling_params = SamplingParams(
            n=1,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            use_beam_search=False,
            stop=stop,
            max_tokens=max_new_tokens,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )

        generation_length = 0

        async for request_output in self.generate_stream(prompt, sampling_params):
            assert len(request_output.outputs) == 1
            generated_text = request_output.outputs[0].text
            if incremental_generation:
                yield generated_text[generation_length:]
            else:
                yield generated_text
            generation_length = len(generated_text)


class Predictor(BasePredictor):
    async def setup(self):
        n_gpus = torch.cuda.device_count()
        start = time.time()
        await maybe_download_with_pget(
            MODEL_ID, WEIGHTS_URL, REMOTE_FILES
        )
        print(f"downloading weights took {time.time() - start:.3f}s")
        self.llm = VLLMPipeline(
            tensor_parallel_size=n_gpus,
            model=MODEL_ID,
            quantization="AWQ",
            dtype="auto",
            trust_remote_code=True
            # max_model_len=MAX_TOKENS
        )

    async def predict(
        self,
        prompt: str,
        max_new_tokens: int = Input(
            description="The maximum number of tokens the model should generate as output.",
            default=DEFAULT_MAX_NEW_TOKENS,
        ),
        temperature: float = Input(
            description="The value used to modulate the next token probabilities.", default=DEFAULT_TEMPERATURE
        ),
        top_p: float = Input(
            description="A probability threshold for generating the output. If < 1.0, only keep the top tokens with cumulative probability >= top_p (nucleus filtering). Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751).",
            default=DEFAULT_TOP_P,
        ),
        top_k: int = Input(
            description="The number of highest probability tokens to consider for generating the output. If > 0, only keep the top k tokens with highest probability (top-k filtering).",
            default=DEFAULT_TOP_K,
        ),
        presence_penalty: float = Input(
            description="Presence penalty",
            default=DEFAULT_PRESENCE_PENALTY,
        ),
        frequency_penalty: float = Input(
            description="Frequency penalty",
            default=DEFAULT_FREQUENCY_PENALTY,
        ),
        prompt_template: str = Input(
            description="The template used to format the prompt. The input prompt is inserted into the template using the `{prompt}` placeholder.",
            default=PROMPT_TEMPLATE,
        ),
        tools: str = Input(
            description="Tools in the form of JSON schema to potentially call.",
            default=None,
        )
    ) -> ConcatenateIterator[str]:
        start = time.time()
        prefill = ""
        if tools:
            try: tools = json.loads(tools)
            except Exception as e: raise ValueError(f"Tools must be a valid JSON schema: {e}")
            sys_prompt = FUNC_PROMPT.format(funcs=format_tools(tools))
            prefill = "<function-thoughts>"
        else:
            sys_prompt = "You are a helpful assistant.  Your job is to assist the user with the given question."

        final_prompt = prompt_template.format(prompt=prompt, sys_prompt=sys_prompt, prefill=prefill)
        # print(f'final_prompt:\n\n{final_prompt}\n\n', '==='*10)

        generate = self.llm(
            prompt=final_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
        )
        async for text in generate:
            yield text
        print(f"generation took {time.time() - start:.3f}s")


async def main():
    from ex_funcs import tools_good_json, tools_bad_json
    from asyncio.exceptions import CancelledError
    import logging
    logger = logging.getLogger()
    logger.setLevel(logging.CRITICAL)

    p = Predictor()
    await p.setup()
    tst_p = partial(p.predict, max_new_tokens=1024, temperature=0.8, top_p=0.95, top_k=50, presence_penalty=1.0, frequency_penalty=0.2, prompt_template=PROMPT_TEMPLATE)
    
    def async_print(text):
        try: print(text, end="")
        except CancelledError as e: pass

    try:
        print('\n---Test Case 1: Tools NOT Relevant---\n')
        async for text in tst_p(prompt="What is the first letter of the alphabet?", tools=tools_good_json): 
            async_print(text)

        print('\n---Test Case 2: Tools ARE Relevant---\n')
        async for text in tst_p(prompt="How many Japenese Yen are there in 1000 USD?", tools=tools_good_json): 
            async_print(text)

        print('\n---Test Case 3: Tools ARE Relevant but not enough info---\n')
        async for text in tst_p(prompt="Can you help me add a Matt Zemuda to my address book?", tools=tools_good_json): 
            async_print(text)

        print('\n---Test Case 4: Tools ARE Relevant---\n')
        async for text in tst_p(prompt="Can you help me add Matt Zemuda to my address book?  His email address is mattt@replicate.com", tools=tools_good_json): 
            async_print(text)

        print('\n---Test Case 5: No tools provided---\n')
        async for text in tst_p(prompt="Write a very short sentence about SEO.", tools=None): 
            async_print(text)

        print('\n---Test Case 6: Improper JSON schema---\n')
        try: 
            async for text in tst_p(prompt="How many Japenese Yen are there in 1000 USD?", tools=tools_bad_json): ...
        except ValueError as e: print(f"Success! Proper Error Raised: {e}")
    
    except CancelledError: pass

if __name__ == "__main__":
    asyncio.run(main())