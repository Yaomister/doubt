import re
import os
import json
import torch
import argparse
from shared import datasets_lookup
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

# For LLaDA
MASK_ID = 126336
DECAY = 0.95
THRESH = 3

tokenizer = AutoTokenizer.from_pretrained(
    "GSAI-ML/LLaDA-8B-Base", trust_remote_code=True
)


def core(model, x, z, prompt_length):
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


def pick_top(p, allowed, amount_to_pick):
    """pick the top k values form the list of allowed positions."""
    amount_to_pick = min(int(allowed.sum()), int(amount_to_pick))
    selected = torch.zeros_like(allowed)
    if amount_to_pick:
        selected[0, p.masked_fill(~allowed, -1e30)[0].topk(amount_to_pick).indices] = (
            True
        )
    return selected


def generate(model, prompt, args):
    """Generate tokens."""
    os.makedirs(args.output, exist_ok=True)

    prompt_length = prompt.shape[1]
    x = torch.full(
        (1, prompt_length + args.generation_length),
        MASK_ID,
        dtype=torch.long,
        device=device,
    )
    x[:, :prompt_length] = prompt

    # number of blocks
    blocks = args.generation_length // args.block_length
    # number of steps per block
    steps_per_block = args.steps // blocks

    with torch.no_grad():

        for block in range(blocks):
            # the start and end index of the block
            start_index = prompt_length + args.block_length * block
            end_index = prompt_length + args.block_length * (block + 1)

            scores = torch.zeros(args.block_length, device=device)
            leaders = torch.full(
                (args.block_length,), -1, dtype=torch.long, device=device
            )

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
                probabilities = z[0, start_index:end_index].detach().float().softmax(-1)
                p = torch.full(x.shape, -float("inf"), device=device)
                p[0, start_index:end_index] = probabilities.max(-1).values

                if args.method == "rethink":
                    top = probabilities.topk(2, -1)
                    gap = top.values[:, 0] - top.values[:, 1]
                    scores = torch.where(
                        leaders == top.indices[:, 0], scores * DECAY + gap, gap
                    )
                    leaders = top.indices[:, 0]
                    ready = candidates[0, start_index:end_index] & (scores > THRESH)
                    if step == steps_per_block - 1:
                        ready = candidates[0, start_index:end_index]
                    x[0, start_index:end_index][ready] = leaders[ready]

                else:
                    pick = pick_top(p, candidates, blanks_per_step)
                    x[pick] = y[pick]

                if (
                    "core" in args.method
                    and t % 8 == 0
                    and 0.25 <= t / args.steps < 0.75
                ):
                    out = core(model, x, z, prompt_length)
                    if out:
                        tested, core_score, replacement = out
                        if args.method == "core":
                            # overwrite the token the model wants back least
                            worst = int(core_score.argmax())
                            x[0, int(tested[0].nonzero()[worst])] = replacement[worst]

                t = t + 1

    return x
