import re
import os
import json
import torch
import argparse
from shared import datasets_lookup
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

"""
DOUBT: Directed Opposition in the denoising loop of diffusion language models.
"""

device = "cuda" if torch.cuda.is_available() else "cpu"

# For LLaDA
MASK_ID = 126336


def spearman(x, y):
    """calculate the separman correlation between two distributions."""
    n = len(x)
    d = x.argsort().argsort().float() - y.argsort().argsort().float()
    return float(1 - 6 * d.pow(2).sum() / (n * (n**2 - 1)))


def _get_args():
    """Arguments for the experiment."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, required=False, default="GSAI-ML/LLaDA-8B-Base"
    )
    parser.add_argument(
        "--method",
        type=str,
        required=False,
        default="rethink",
        choices=["baseline", "rethink", "margin", "core"],
    )
    parser.add_argument("--epsilon", type=float, required=False, default=0.005)
    parser.add_argument("--random", action="store_true", required=False, default=False)
    parser.add_argument("--output", required=False, default="./results")
    parser.add_argument("--dataset", required=False, default="gsm8k")
    parser.add_argument("--smoke-test", required=False, action="store_true")
    parser.add_argument("--steps", type=int, default=128)

    return parser.parse_args()


def core(model, x, z, prompt_length, args):
    """CoRE."""

    # the positions that are already unmasked that we can pick from
    already_written = x != MASK_ID
    # cannot pick from the prompt
    already_written[:, :prompt_length] = False

    if int(already_written.sum()) < 4:
        return

    # the top most confident tokens
    top = z.float().softmax(-1).topk(2, -1).values

    # the margin between the top 2 tokens
    margin = (top[..., 0] - top[..., 1]).masked_fill(~already_written, 1e30)

    # this is the number of margins to look for
    m = min(32, int(already_written.sum()))

    substitute = torch.zeros_like(already_written)

    # the list the top m margins to remask to see if they're context brittle
    substitute[0, margin[0].topk(m, largest=False).indices] = True

    # save the previously chosen logits to compare at the end
    saved = x[substitute]

    # remask them
    xp = x.masked_fill(substitute, MASK_ID)

    # get the new logits after the remasking
    with torch.no_grad():
        z_core = model(xp).logits

    core = -torch.log_softmax(z_core[substitute].float(), -1)

    # return the old choices and the new choices
    return (
        substitute,
        core.gather(1, saved[:, None]).squeeze(1),
        z_core[substitute].argmax(-1),
    )


def _load_model(args):
    """Load the model."""
    model = AutoModel.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.bfloat16, device_map="auto"
    )

    # freeze the model
    model.requires_grad_(False)
    for w in model.model.transformer.blocks[-1].parameters():
        w.requires_grad_(True)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    return model, tokenizer


def pick_top(p, allowed, amount_to_pick):
    """pick the top k values form the list of allowed positions."""
    amount_to_pick = min(int(allowed.sum()), int(amount_to_pick))
    selected = torch.zeros_like(allowed)
    if amount_to_pick:
        selected[0, p.masked_fill(~allowed, -1e30)[0].topk(amount_to_pick).indices] = (
            True
        )
    return selected


def rethink(model, z, candidates, args):
    """Nudge the weights in the direction that shrinks the distance between the first and second most confident logits."""
    # save the original weights
    weights = list(model.model.transformer.blocks[-1].parameters())

    # get the top logits
    top = z[0, candidates].topk(2).values

    m = top[0] - top[1]

    grads = torch.autograd.grad(m, weights, retain_graph=True)

    with torch.no_grad():
        drop = sum(
            (w.float().abs() * g.float().abs()).sum() for w, g in zip(weights, grads)
        )
        print(f"ratio {float(m.sum() / drop)}")
        return float(args.epsilon * drop - m)


def generate(model, prompt, args):
    os.makedirs(args.output, exist_ok=True)

    prompt_length = prompt.shape[1]
    x = torch.full(
        (1, prompt_length + args.generation_length),
        MASK_ID,
        dtype=torch.long,
        device=device,
    )
    x[:, :prompt_length] = prompt

    changed_n, total, t = 0, 0, 0
    agree = []

    # number of blocks
    blocks = args.generation_length // args.block_length
    # number of steps per block
    steps_per_block = args.steps // blocks

    for block in range(blocks):
        # the start and end index of the block
        start_index = prompt_length + args.block_length * block
        end_index = prompt_length + args.block_length * (block + 1)

        # repeat per decoding step
        for step in range(steps_per_block):
            # the ones we can unmask
            candidates = x == MASK_ID
            candidates[:, :start_index], candidates[:, end_index:] = False, False
            blanks_per_step = -(-int(candidates.sum()) // (steps_per_block - step))

            with torch.set_grad_enabled(args.method == "rethink"):
                output = model(x)
                z = output.logits
            y = z.detach().argmax(-1)
            p = z.detach().float().softmax(-1).max(-1).values

            if args.method == "baseline":
                pick = pick_top(p, candidates, blanks_per_step)
            elif args.method == "margin":
                top2 = z.detach().float().softmax(-1).topk(2, -1).values
                margins = top2[..., 0] - top2[..., 1]
                pick = pick_top(margins, candidates, blanks_per_step)
            else:
                pick = pick_top(p, candidates, blanks_per_step)
                if (
                    args.method == "rethink"
                    and step < steps_per_block - 1
                    and 0.25 < t / args.steps < 0.75
                ):
                    limit = 3 * blanks_per_step
                    amount_probed = 0
                    pick = torch.zeros_like(candidates)
                    for pos in (
                        p[0]
                        .masked_fill(~candidates[0], -1e30)
                        .argsort(descending=True)[: int(candidates.sum())]
                        .tolist()
                    ):
                        if pick.sum() == blanks_per_step or amount_probed > limit:
                            break
                        amount_probed += 1
                        total += 1
                        if rethink(model, z, pos, args) <= 0:
                            # this means that it survived the shake
                            pick[0, pos] = True
                        else:
                            changed_n += 1

                    missing = blanks_per_step - pick.sum()
                    if missing >= 0:
                        pick |= pick_top(p, candidates & ~pick, missing)
            x[pick] = y[pick]

            if (
                args.method in ("core",)
                and t % 8 == 0
                and 0.25 <= t / args.steps < 0.75
            ):
                out = core(model, x, z, prompt_length, args)
                if out:
                    tested, core_score, replacement = out
                    if args.method == "core":
                        # overwrite the token the model wants back least
                        worst = int(core_score.argmax())
                        x[0, int(tested[0].nonzero()[worst])] = replacement[worst]
                    else:
                        theirs = tested[0].nonzero().flatten().tolist()
                        mine = torch.tensor(
                            [rethink(model, x, z, s, x[0, s], args) for s in theirs],
                            device=device,
                        )
                        if core_score.std() > 1e-4 and mine.std() > 1e-4:
                            agree.append(
                                (
                                    spearman(core_score, mine),
                                    float(core_score.argmax() == mine.argmax()),
                                )
                            )

            t = t + 1

    return x, changed_n, total, agree


def prepare_question(dataset_name, row):
    if dataset_name == "gsm8k":
        question = row["question"]
    elif dataset_name == "math":
        question = row["problem"]
    elif dataset_name == "humaneval":
        question = f"Complete this function. Return the full function in a ```python block.\n\n```python\n{row['prompt']}```"
    elif dataset_name == "mbpp":
        question = (
            "You are an expert Python programmer, and here is your task: "
            + row["text"]
            + "\nYour code should pass these tests:\n\n"
            + "\n".join(row["test_list"])
        )
    else:
        raise ValueError()

    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        add_generation_prompt=True,
        tokenize=False,
    )
    token_ids = tokenizer(
        formatted, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(device)

    return token_ids


if __name__ == "__main__":
    args = _get_args()
    model, tokenizer = _load_model(args)

    records = []

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
            text = tokenizer.decode(out[0, input.shape[1] :], skip_special_tokens=True)
            nums = re.findall(
                r"-?\d+",
                (text.split("####")[-1] if "####" in text else text).replace(",", ""),
            )

            hits += bool(nums) and nums[0 if "####" in text else -1] == answer
            changes, candidates, rho = changes + f, candidates + c, rho + ag

            msg = f"{i + 1}/{amount_question}  acc={hits / (i + 1):.3f}"
            if candidates:
                msg += f"fell={changes / max(candidates, 1):.2f}"
            if rho:
                msg += f"rho={sum(r for r, _ in rho) / len(rho):+.2f}  same={sum(s for _, s in rho) / len(rho):.2f}"

            print(msg, flush=True)
            records.append(
                {
                    "i": i,
                    "correct": bool(nums)
                    and nums[0 if "####" in text else -1] == answer,
                    "acc": hits / (i + 1),
                    "deferral": changes / candidates if candidates else None,
                    "rho": sum(r for r, _ in rho) / len(rho) if rho else None,
                }
            )

            with open(f"{args.output}/smoke_test.json", "w") as fh:
                json.dump({"args": vars(args), "records": records}, fh, indent=1)
    else:
        dataset = load_dataset(
            datasets_lookup[args.dataset],
            "main" if args.dataset == "gsm8k" else None,
            split="test",
        )

        for i in range(len(dataset)):
            input = prepare_question(args.dataset, dataset[i])

            res, changed, total, _ = generate(model, input)
            text = tokenizer.decode(res[0, input.shape[1] :], skip_special_tokens=True)
            records.append(
                {"i": i, "output": text, "deferred": changed, "total": total}
            )

            with open(
                f"{args.output}/results_{args.model.split('/')[-1]}_{args.method}_{args.epsilon}_{args.dataset}_steps{args.steps}{'_random' if args.random else ''}.json",
                "w",
            ) as f:
                json.dump(records, f)
