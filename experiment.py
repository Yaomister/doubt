import re
import torch
import argparse
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

"""
DOUBT: Directed Opposition in the Denoising Loop of diffusion language models.
"""

device = "cuda" if torch.cuda.is_available() else "cpu"


# need to be model specific
amount_questions = 50
generation_length = 256
denoising_steps = 128
block_size = 32
MASK_ID = 126336
carry_weight = 0.5

epsilon = 0.05
delta = None
norm = 0.0

MODE = "rethink"
RANDOM = False


def spearman(a, b):
    ra, rb = a.argsort().argsort().float(), b.argsort().argsort().float()
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float(ra @ rb / (ra.norm() * rb.norm() + 1e-12))


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=False, default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--method", required=False, default="rethink",
                        choices=["baseline", "rethink", "pressure", "core", "compare"])
    parser.add_argument("--epsilon", required=False, type=float, default=0.05)
    parser.add_argument("--random", action="store_true", help="control: undirected push of the same size")

    return parser.parse_args()


def probe(model, x, z, prompt_length, zeros):
    written = x != MASK_ID
    written[:, :prompt_length] = False
    # bail if theres not enough written
    if int(written.sum()) < 4:
        return None
    # take the difference ebtween the top 2
    p2 = z.float().softmax(-1).topk(2, -1).values
    # fill unwritten tokens with very large numbers (so the margin is huge and blank slots are never picked)
    margin = (p2[..., 0] - p2[..., 1]).masked_fill(~written, 1e30)
    m = min(32, int(written.sum()))
    sub = torch.zeros_like(written)
    sub[0, margin[0].topk(m, largest=False).indices] = True
    # remember the words before blanking them
    cur = x[sub]
    xp = x.masked_fill(sub, MASK_ID)
    zc = forward(model, xp)

    # how much does the model still want the token sitting there (big = it does not)
    core = -torch.log_softmax(zc[sub].float(), -1).gather(1, cur[:, None]).squeeze(1)
    if MODE == "core":
        # third value is what the model would write instead
        return sub, core, zc[sub].argmax(-1), None
    # same 32 positions, but stressed with a push aimed at the words sitting there
    d, _, _ = nudge(model, xp, sub, zeros, target=x)
    za = forward(model, xp, d)
    mine = -torch.log_softmax(za[sub].float(), -1).gather(1, cur[:, None]).squeeze(1)
    return sub, core, zc[sub].argmax(-1), mine


def load_model(args):
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

    def hook(module, input, output):
        global norm
        h = output[0] if isinstance(output, tuple) else output
        norm = h.detach().float().norm(dim=-1).median()   # dim=-1, not norm(-1): that is the p=-1 norm
        if delta is None:
            return None
        h = h + delta.to(h.dtype)
        return (h, ) + tuple(output[1:]) if isinstance(output, tuple) else h

    blocks = model.model.transformer.blocks
    blocks[len(blocks) // 2].register_forward_hook(hook)

    return model, tokenizer


def forward(model, x, d=None):
    global delta
    delta = d
    with torch.no_grad():
        z = model(x).logits
    delta = None
    return z


def nudge(model, x, candidates, start, target=None):
    """nudge the hidden states in the direction that shrinks the distance between the first and second most confident logits."""
    global delta
    d = start.detach().requires_grad_(True)
    delta = d
    with torch.enable_grad():
        z = model(x).logits
        y = z.argmax(-1) if target is None else target
        zz, yy = z[candidates].float(), y[candidates]
        first = zz.gather(1, torch.unsqueeze(yy, 1)).squeeze(1)
        second = zz.topk(2, dim=1).values
        rival = torch.where(second[:, 0] > first, second[:, 0], second[:, 1])
        lead = first - rival
        # negative because we want to supress it
        (g, ) = torch.autograd.grad(-lead.sum(), d)
    delta = None
    r = epsilon * norm
    if RANDOM:
        g = torch.randn_like(g)
    d = d.detach() + r * g / g.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return d * (r / d.norm(dim=-1, keepdim=True).clamp_min(1e-12)).clamp(max=1), z.detach(), y


def pick_top(p, allowed, amount_to_pick):
    """Pick the top-k elements."""
    amount_to_pick = min(int(amount_to_pick), int(allowed.sum()))
    selected = torch.zeros_like(allowed)
    if amount_to_pick:
        selected[0, p.masked_fill(~allowed, -1e30)[0].topk(amount_to_pick).indices] = True
    return selected


def generate(model, prompt, method):

    hidden_dimension = model.get_input_embeddings().weight.shape[1]

    prompt_length = prompt.shape[1]
    x = torch.full((1, prompt_length + generation_length), MASK_ID, dtype=torch.long, device=device)
    x[:, :prompt_length] = prompt

    carry = torch.zeros(1, prompt_length + generation_length, hidden_dimension, device=device)

    changed_n, total, t = 0, 0, 0
    agree = []

    blocks = generation_length // block_size
    steps_per_block = denoising_steps // blocks

    for block in range(blocks):
        start_index = prompt_length + block_size * block
        end_index = prompt_length + block_size * (block + 1)
        for step in range(steps_per_block):
            candidates = x == MASK_ID
            candidates[:, :start_index], candidates[:, end_index:] = False, False
            blanks_per_step = -(-int(candidates.sum()) // (steps_per_block - step))
            #  denoise
            if method == "rethink":
                # write, then stress test
                d, z, y = nudge(model, x, candidates, torch.zeros_like(carry))
                p = z.float().softmax(-1).gather(-1, y[..., None]).squeeze(-1)
                different = (forward(model, x, d).argmax(-1) != y) & candidates
                changed_n = changed_n + int(different.sum())
                total = total + int(candidates.sum())
                survived = candidates if step == steps_per_block - 1 else candidates & ~different
                pick = pick_top(p, survived, blanks_per_step)
            elif method == "pressure":
                # carry over the doubt from the last step
                carry, z, y = nudge(model, x, candidates, carry_weight * carry)
                pick = pick_top(z.float().softmax(-1).gather(-1, y[..., None]).squeeze(-1), candidates, blanks_per_step)
            else:
                # good old denoising (also used by core and compare)
                z = forward(model, x)
                y = z.argmax(-1)
                pick = pick_top(z.float().softmax(-1).gather(-1, y[..., None]).squeeze(-1), candidates, blanks_per_step)
            # actually unmasking it
            x[pick] = y[pick]
            t += 1

            if method in ("core", "compare") and t % 8 == 0 and 0.25 <= t / denoising_steps < 0.75:
                out = probe(model, x, z, prompt_length, torch.zeros_like(carry))
                if out:
                    tested, core_score, replacement, my_score = out
                    if method == "core":
                        # overwrite the token the model wants back least
                        worst = int(core_score.argmax())
                        x[0, int(tested[0].nonzero()[worst])] = replacement[worst]
                    elif core_score.std() > 1e-4 and my_score.std() > 1e-4:
                        agree.append((spearman(core_score, my_score),
                                      float(core_score.argmax() == my_score.argmax())))

    return x, changed_n, total, agree


if __name__ == "__main__":
    args = get_args()
    MODE, RANDOM, epsilon = args.method, args.random, args.epsilon
    model, tokenizer = load_model(args)
    method = args.method

    dataset = load_dataset("openai/gsm8k", "main", split="test")
    hits = changes = candidates = 0
    rho = []

    for i in range(amount_questions):
        answer = dataset[i]["answer"].split("####")[-1].strip().replace(",", "")
        chat = tokenizer.apply_chat_template([{"role": "user", "content": dataset[i]["question"] + "\nReason step by step, then give the final answer after ####."}], add_generation_prompt=True, tokenize=False)

        ids = tokenizer(chat, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        out, f, c, ag = generate(model, ids, method)
        text = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        # probably move this to a separate decoding function
        nums = re.findall(r"-?\d+", (text.split("####")[-1] if "####" in text else text).replace(",", ""))
        hits += bool(nums) and nums[0 if "####" in text else -1] == answer
        changes, candidates, rho = changes + f, candidates + c, rho + ag
        msg = f"{i + 1}/{amount_questions}  acc={hits / (i + 1):.3f}"
        if candidates:
            msg += f"fell={changes / max(candidates, 1):.2f}"
        if rho:
            msg += f"rho={sum(r for r, _ in rho) / len(rho):+.2f}  same={sum(s for _, s in rho) / len(rho):.2f}"
        print(msg, flush=True)