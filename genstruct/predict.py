import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
import torch
from cog import BasePredictor
from vllm import LLM, SamplingParams

def prompt(title, content):
    return f"""[[[Title]]] {title}

[[[Content]]] {content}

The following is an interaction between a user and an AI assistant that is related to the above text.

[[[User]]] """

class Predictor(BasePredictor):
        
    def setup(self):
        n_gpus = torch.cuda.device_count()
        self.llm = LLM(model='NousResearch/Genstruct-7B', 
                       tensor_parallel_size=n_gpus)

    def predict(self, title: str, content: str, temp:float=0.0, max_tokens:int=2000) -> str:        
        _p = prompt(title, content)
        sampling_params = SamplingParams(temperature=temp, ignore_eos=True, max_tokens=max_tokens)
        out = self.llm.generate(_p, sampling_params=sampling_params, use_tqdm=False)
        return out[0].outputs[0].text

if __name__ == "__main__":
    p = Predictor()
    p.setup()
    p.predict(
        title="Law of large numbers", 
        content="In probability theory, the law of large numbers (LLN) is a mathematical theorem that states that the average of the results obtained from a large number of independent and identical random samples converges to the true value, if it exists.[1] More formally, the LLN states that given a sample of independent and identically distributed values, the sample mean converges to the true mean.",
    )
