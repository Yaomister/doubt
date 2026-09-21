import re
import os
import json
import torch
import argparse
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

"""
DOUBT: Directed Opposition in the denoising loop of diffusion language models.
"""

device = "cuda" if torch.cuda.is_available() else "cpu"

# For LLaDA
generation_length = 256
block_size = 32
MASK_ID = 126336

def spearman(x, y):
    n = len(x)
    d = x.argsort().argsort().float() - y.argsort().argsort().float()
    return float(1 - 6 * d.pow(2).sum() / (n * (n**2 - 1)))

def _get_args():
    """Arguments for the experiment."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=False, default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--method", type=str, required=False, default="rethink", choices=["baseline", "rethink", "pressure", "core"])
    parser.add_argument("--epsilon", type=float, required=False, default=0.005)
    parser.add_argument("--random", action="store_true", required=False, default=False)
    parser.add_argument("--output", required=False, default="./results")
    parser.add_argument("--dataset", required=False, default="gsm8k")
    parser.add_argument("--smoke-test", required=False, action="store_true")
    parser.add_argument("--steps", type=int, default=128)
 
    return parser.parse_args()


def core(model, x, z, prompt_length, args):
    already_written = x != MASK_ID
    already_written[:, :prompt_length] = False

    if int(already_written.sum()) < 4:
        return

    top = z.float().softmax(-1).topk(2, -1).values

    margin = (top[...,0] - top[...,1]).masked_fill(~already_written, 1e30)
    # this is the number of margins to look for
    m = min(32, int(already_written.sum()))

    substitute = torch.zeros_like(already_written)
    substitute[0, margin[0].topk(m, largest=False).indices] = True

    saved = x[substitute]
    xp = x.masked_fill(substitute, MASK_ID)
    with torch.no_grad():
        z_core = model(xp).logits

    core = -torch.log_softmax(z_core[substitute].float(),  -1)

    return substitute, core.gather(1, saved[:, None]).squeeze(1), z_core[substitute].argmax(-1)


def _load_model(args):
    """Load the model."""
    model = AutoModel.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.requires_grad_(False)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True
    )
    return model, tokenizer

def pick_top(p, allowed, amount_to_pick):
    amount_to_pick = min(int(allowed.sum()), int(amount_to_pick))
    selected = torch.zeros_like(allowed)
    if amount_to_pick:
        selected[0, p.masked_fill(~allowed, -1e30)[0].topk(amount_to_pick).indices] = True
    return selected
    

def rethink(model, x, candidates, original, args):
    """Nudge the weights in the direction that shrinks the distance between the first and second most confident logits."""
    # save the original weights
    weights = list(model.model.transformer.blocks[-1].parameters())
    for p in weights:
        p.requires_grad_(True)

    with torch.enable_grad():
        z = model(x).logits[0, candidates].float()
        top = z.topk(2).values
        # we negate the margins because we want the direction that decreases the difference between the top 2 logits
        nudge_directions = torch.autograd.grad((top[0] - top[1]), weights)

    old_weights = [p.detach().clone() for p in weights]

    with torch.no_grad():
        for p, d in zip(weights, nudge_directions):
            if args.random:
                # add random noise
                d = torch.randn_like(d)
            # apply the adversarial nudge to the last layer weights
            p -= (d / d.norm().clamp_min(1e-12)) * args.epsilon * p.norm()
        
        # retain the ones that remained the same after the nudge
        zz = model(x).logits[0, candidates]
        rival = zz.clone()
        rival[original] = -1e30

         # copying back the old weights
        for p, s in zip(weights, old_weights):
            p.copy_(s)

        # if > 0 it flipped
        return float(rival.max() - zz[original])


def generate(model, prompt):

    os.makedirs(args.output, exist_ok=True)

    hidden_dimension = model.get_input_embeddings().weight.shape[1]

    prompt_length = prompt.shape[1]
    x = torch.full((1, prompt_length + generation_length), MASK_ID, dtype=torch.long, device=device)
    x[:, :prompt_length] = prompt

    carry = torch.zeros(1, prompt_length + generation_length, hidden_dimension, device=device)

    changed_n, total, t = 0, 0, 0
    agree = []

    blocks = generation_length // block_size
    steps_per_block = args.steps // blocks

    for block in range(blocks):
        start_index = prompt_length + block_size * block
        end_index = prompt_length + block_size * (block + 1)
        for step in range(steps_per_block):
            candidates = x == MASK_ID
            candidates[:, :start_index], candidates[:, end_index:] = False, False
            blanks_per_step = -(-int(candidates.sum()) // (steps_per_block - step))

            with torch.no_grad():
                z = model(x).logits

            y = z.argmax(-1)
            p = z.float().softmax(-1).max(-1).values

            pick = pick_top(p, candidates, blanks_per_step)

            if args.method == "rethink" and step < steps_per_block - 1:
                for pos in pick[0].nonzero().flatten().tolist():
                    total += 1
                    if rethink(model, x, pos,  y[0, pos], args) > 0:
                        # this means it chose a different token after rethinking
                        pick[0, pos] = False
                        changed_n += 1

            x[pick] = y[pick]
            t += 1

            if args.method in ("core") and t % 8 == 0 and 0.25 <= t / args.steps < 0.75:
                out = core(model, x, z, prompt_length, args)
                if out:
                    tested, core_score, replacement = out
                    if args.method == "core":
                        # overwrite the token the model wants back least
                        worst = int(core_score.argmax())
                        x[0, int(tested[0].nonzero()[worst])] = replacement[worst]
                    else:
                        theirs = tested[0].nonzero().flatten().tolist()
                        mine = torch.tensor([rethink(model, x, s, x[0, s], args) for s in theirs], device=device)
                        if core_score.std() > 1e-4 and mine.std() > 1e-4:
                            agree.append((spearman(core_score, mine),
                                        float(core_score.argmax() == mine.argmax())))

    return x, changed_n, total, agree

def prepare_question(dataset_name, row):
    if dataset_name == "gsm8k":
        question = row['question']
    elif dataset_name == "math":
        question = row["problem"]
    elif dataset_name == "humaneval":
        question = f"Complete this function. Return the full function in a ```python block.\n\n```python\n{row['prompt']}```"
    elif dataset_name == "mbpp":
        question = "You are an expert Python programmer, and here is your task: " + row["text"] + "\nYour code should pass these tests:\n\n" + "\n".join(row["test_list"])
    else:
        raise ValueError()

    formatted = tokenizer.apply_chat_template([{"role": "user", "content": question}], add_generation_prompt=True, tokenize=False)
    token_ids = tokenizer(formatted, add_special_tokens=False, return_tensors="pt").input_ids.to(device)

    return token_ids


if __name__ == "__main__":
    args = _get_args()
    model, tokenizer = _load_model(args)

    records =[]

    # running a smaller test to make sure everything works
    if args.smoke_test:
        dataset = load_dataset("openai/gsm8k", "main", split="test")
        amount_question = 50
        hits = changes = candidates = 0
        rho = []
        for i in range(amount_question):
            answer = dataset[i]["answer"].split("####")[-1].strip().replace(",", "")
            input = prepare_question("gsm8k", dataset[i])
    
            out, f, c, ag = generate(model, input)

            # check if answers match
            text = tokenizer.decode(out[0, input.shape[1]:], skip_special_tokens=True)
            nums = re.findall(r"-?\d+", (text.split("####")[-1] if "####" in text else text).replace(",", ""))

            hits += bool(nums) and nums[0 if "####" in text else -1] == answer
            changes, candidates, rho = changes + f, candidates + c, rho + ag

            msg = f"{i + 1}/{amount_question}  acc={hits / (i + 1):.3f}"
            if candidates:
                msg += f"fell={changes / max(candidates, 1):.2f}"
            if rho:
                msg += f"rho={sum(r for r, _ in rho) / len(rho):+.2f}  same={sum(s for _, s in rho) / len(rho):.2f}"

            print(msg, flush=True)
            records.append({"i": i, "correct": bool(nums) and nums[0 if "####" in text else -1] == answer,
                "acc": hits / (i + 1),
                "deferral": changes / candidates if candidates else None,
                "rho": sum(r for r, _ in rho) / len(rho) if rho else None})
            
            with open(f"{args.output}/smoke_test.json", "w") as fh:
                json.dump({"args": vars(args), "records": records}, fh, indent=1)
    else:
        datasets = {
            "gsm8k": "openai/gsm8k",
            "math" : "HuggingFaceH4/MATH-500",
            "humaneval": "openai/openai_humaneval",
            "mbpp": "google-research-datasets/mbpp"
        }

        dataset = load_dataset(datasets[args.dataset], "main" if args.dataset == "gsm8k" else None, split="test")
        
        for i in range(len(dataset)):
            input = prepare_question(args.dataset, dataset[i])

            res, changed, total, _ = generate(model, input)
            text = tokenizer.decode(res[0, input.shape[1]:], skip_special_tokens=True)
            records.append({"i": i, "output": text, "deferred": changed, "total": total})

            with open(f'{args.output}/results_{args.model.split('/')[-1]}_{args.method}_{args.epsilon}_{args.dataset}_steps{args.steps}{'_random' if args.random else ''}.json', "w") as f:
                json.dump(records, f)