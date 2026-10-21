import torch
from tqdm import tqdm
from lm_eval.api.model import LM
from types import SimpleNamespace
from experiment import generate, device
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.registry import register_model
from transformers import AutoModel, AutoTokenizer

MODEL_PATHS = {
    "llada": "GSAI-ML/LLaDA-8B-Base",
    "dream": "Dream-org/Dream-v0-Base-7B",
}

MASK_IDS = {
    "llada": 126336,
    "dream": 151666,
}


@register_model("doubt")
class DoubtLM(LM):
    def __init__(
        self,
        model_type="llada",
        model_path=None,
        method="top",
        epsilon=0.005,
        random=False,
        steps=128,
        generation_length=512,
        block_length=512,
        decay=0.95,
        threshold=3,
        apply_core=False,
        **kwargs
    ):
        super().__init__()
        self.deferred = self.probed = 0
        assert model_type in MODEL_PATHS, "invalid model type."
        model_path = model_path or MODEL_PATHS[model_type]
        self.model = (
            AutoModel.from_pretrained(
                model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
            )
            .eval()
            .requires_grad_(False)
            .to(device)
        )

        # make sure we can take the gradient of the last layer weights
        if model_type == "dream":
            last_layer = self.model.model.layers[-1]
        else:
            last_layer = self.model.model.transformer.blocks[-1]
        for w in last_layer.parameters():
            w.requires_grad_(True)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.args = SimpleNamespace(
            model_type=model_type,
            mask_id=MASK_IDS[model_type],
            epsilon=float(epsilon),
            random=bool(random),
            method=method,
            steps=int(steps),
            generation_length=int(generation_length),
            block_length=int(block_length),
            decay=float(decay),
            threshold=float(threshold),
            apply_core=str(apply_core).lower() in ("true", "1"),
            task=None,
            output="./results",
        )

    def generate_until(self, requests):
        """Generate until EOT."""
        answers = []

        for request in tqdm(requests, desc="generating..."):
            prompt_text, generation_kwargs = request.args
            self.args.task = getattr(request, "task_name", None)
            prompt = torch.tensor(
                [self.tokenizer(prompt_text)["input_ids"]], device=device
            )

            generated_answer = generate(self.model, prompt, self.args)

            stop_tokens = generation_kwargs.get("until", [])

            decoded_answer = self.tokenizer.decode(
                generated_answer[0][prompt.shape[1] :], skip_special_tokens=True
            )

            for stop_sequence in stop_tokens:
                if stop_sequence in decoded_answer:
                    decoded_answer = decoded_answer.split(stop_sequence)[0]

            answers.append(decoded_answer)
            self.cache_hook.add_partial("generate_until", request.args, decoded_answer)

        return answers

    def loglikelihood(self, requests):
        """Ignore"""
        raise NotImplementedError("eval_doubt.py only supports generation tasks")

    def loglikelihood_rolling(self, requests):
        """Ignore"""
        raise NotImplementedError("eval_doubt.py only supports generation tasks")


if __name__ == "__main__":
    cli_evaluate()
