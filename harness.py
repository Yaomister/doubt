import torch
from tqdm import tqdm
from lm_eval.api.model import LM
from types import SimpleNamespace
from experiment import generate, device
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_model
from transformers import AutoModel, AutoTokenizer
from experiment import generate, device, _save_last_layer_weights




@register_model("doubt")
class DoubtLM(LM):
    def __init__(self, model_path = "GSAI-ML/LLaDA-8B-Base", method="baseline", epsilon=0.005, random=False, steps=128, generation_length=512, block_length=512, **kwargs):
        super().__init__()
        self.deferred = self.probed = 0
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype = torch.bfloat16).eval().requires_grad_(False).to(device)

        # make sure we can take the gradient of the last layer weights
        for w in self.model.model.transformer.blocks[-1].parameters():
            w.requires_grad_(True)


        self.model.model.transformer.blocks[-1].register_forward_pre_hook(_save_last_layer_weights, with_kwargs=True)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.args = SimpleNamespace(epsilon = float(epsilon), random = bool(random), method = method, steps = int(steps), generation_length = int(generation_length), block_length = int(block_length), output = "./results")


    def generate_until(self, requests):
        answers = []

        for request in tqdm(requests, desc="generating..."):
            prompt_text, generation_kwargs = request.args
            prompt = torch.tensor([self.tokenizer(prompt_text)['input_ids']], device=device)

            generated_answer, deferred, probed, _ = generate(self.model, prompt, self.args)

            self.deferred += deferred
            self.probed += probed

            stop_tokens = generation_kwargs.get("until", [])

            decoded_answer = self.tokenizer.decode(generated_answer[0][prompt.shape[1]:], skip_special_tokens=False)

            for stop_sequence in stop_tokens:
                if stop_sequence in decoded_answer:
                    decoded_answer = decoded_answer.split(stop_sequence)[0]

            answers.append(decoded_answer)
            self.cache_hook.add_partial("generate_until", request.args, decoded_answer)


        return answers
    

    def loglikelihood(self, requests):
        raise NotImplementedError("eval_doubt.py only supports generation tasks")
 
    def loglikelihood_rolling(self, requests):
        raise NotImplementedError("eval_doubt.py only supports generation tasks")



if __name__ == "__main__":
    cli_evaluate()