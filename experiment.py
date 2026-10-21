import os
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# For LLaDA (DREAM doesnt' need one)
MASK_ID = 126336


def get_klass_params(args):
    """Look up the klass parameters for the current model and dataset."""

    KLASS_PARAMS = {
        "llada": {
            "math": (0.6, 0.010),
            "gsm8k": (0.6, 0.015),
            "humaneval": (0.9, 0.010),
            "mbpp": (0.7, 0.010),
        },
        "dream": {
            "math": (0.9, 0.005),
            "gsm8k": (0.9, 0.001),
            "humaneval": (0.8, 0.001),
            "mbpp": (0.9, 0.001),
        },
    }

    for dataset in ["gsm8k", "humaneval", "mbpp", "math"]:
        if dataset in args.task:
            return KLASS_PARAMS[args.model_type][dataset]

    raise ValueError()


def forward(model, x, args):
    """Run the model and return logits aligned so position i predicts token i."""

    logits = model(x).logits

    # Dream's logits are shifted by 1
    if args.model_type == "dream":
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

    return logits


def random(x, candidates, amount_to_pick):
    """Randomly choose positions to decode."""

    amount_to_pick = min(int(candidates.sum()), int(amount_to_pick))

    selected = torch.zeros_like(candidates)

    scores = torch.rand(x.shape, device=x.device)
    scores[~candidates] = -1e30

    if amount_to_pick:
        selected[0, scores[0].topk(amount_to_pick).indices] = True

    return selected


def core(model, x, z, prompt_length, args):
    """Context Robust Remasking (CoRE)."""

    # the positions that are already unmasked that we can pick from
    already_written = x != args.mask_id
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
    xp = x.masked_fill(substitute, args.mask_id)

    # get the new logits after the remasking
    with torch.no_grad():
        z_core = forward(model, xp, args)

    core_ = -torch.log_softmax(z_core[substitute].float(), -1)

    # return the old choices and the new choices
    return (
        substitute,
        core_.gather(1, saved[:, None]).squeeze(1),
        z_core[substitute].argmax(-1),
    )


def top_margin(probabilities, candidates, amount_to_pick, start_index):
    """pick the top k values by the top-2 margin from the list of allowed positions."""

    amount_to_pick = min(int(candidates.sum()), int(amount_to_pick))

    selected = torch.zeros_like(candidates)

    top_p = probabilities.topk(2, -1).values

    margins = torch.full(candidates.shape, -float("inf"), device=candidates.device)

    # only set the margin value for the block we're decoding
    margins[0, start_index : start_index + top_p.shape[0]] = top_p[:, 0] - top_p[:, 1]

    if amount_to_pick:
        selected[
            0, margins.masked_fill(~candidates, -1e30)[0].topk(amount_to_pick).indices
        ] = True

    return selected


def top(p, candidates, amount_to_pick):
    """pick the top k values form the list of allowed positions."""

    amount_to_pick = min(int(candidates.sum()), int(amount_to_pick))

    selected = torch.zeros_like(candidates)

    if amount_to_pick:
        selected[
            0, p.masked_fill(~candidates, -1e30)[0].topk(amount_to_pick).indices
        ] = True

    return selected


def generate(model, prompt, args):
    """Generate tokens."""

    assert args.method in [
        "rethink",
        "klass",
        "top",
        "top-margin",
        "random",
    ], "invalid decoding method."

    os.makedirs(args.output, exist_ok=True)

    prompt_length = prompt.shape[1]
    x = torch.full(
        (1, prompt_length + args.generation_length),
        args.mask_id,
        dtype=torch.long,
        device=device,
    )
    x[:, :prompt_length] = prompt

    # number of blocks
    blocks = args.generation_length // args.block_length
    # number of steps per block
    steps_per_block = args.steps // blocks

    t = 0
    with torch.no_grad():

        for block in range(blocks):
            # the start and end index of the block
            start_index = prompt_length + args.block_length * block
            end_index = prompt_length + args.block_length * (block + 1)

            scores = torch.zeros(args.block_length, device=device)
            leaders = torch.full(
                (args.block_length,), -1, dtype=torch.long, device=device
            )

            # KLASS states
            history = 2
            p_prev = None
            kl_buf = torch.full(
                (args.block_length, history), float("inf"), device=device
            )

            # repeat per decoding step
            for step in range(steps_per_block):
                # the ones we can unmask
                candidates = x == args.mask_id
                candidates[:, :start_index], candidates[:, end_index:] = False, False
                blanks_per_step = -(-int(candidates.sum()) // (steps_per_block - step))

                with torch.set_grad_enabled(args.method == "rethink"):
                    z = forward(model, x, args)
                y = z.detach().argmax(-1)
                probabilities = z[0, start_index:end_index].detach().float().softmax(-1)
                p = torch.full(x.shape, -float("inf"), device=device)
                p[0, start_index:end_index] = probabilities.max(-1).values

                if args.method == "rethink":
                    top2 = probabilities.topk(2, -1)
                    gap = top2.values[:, 0] - top2.values[:, 1]
                    scores = torch.where(
                        leaders == top2.indices[:, 0], scores * args.decay + gap, gap
                    )
                    leaders = top2.indices[:, 0]
                    ready = candidates[0, start_index:end_index] & (
                        scores > args.threshold
                    )

                    # fallback for when we're running out of steps
                    if step == steps_per_block - 1:
                        ready = candidates[0, start_index:end_index]

                    x[0, start_index:end_index][ready] = leaders[ready]

                elif args.method == "klass":
                    tau, epsilon = get_klass_params(args)

                    if p_prev is None:
                        kl = torch.full(
                            (args.block_length,), float("inf"), device=device
                        )
                    else:
                        kl = torch.nn.functional.kl_div(
                            p_prev.clamp_min(1e-30).log(),
                            probabilities,
                            reduction="none",
                        ).sum(-1)

                    p_prev = probabilities

                    kl_buf = torch.roll(kl_buf, shifts=-1, dims=-1)
                    kl_buf[..., -1] = kl

                    stable = torch.zeros_like(candidates)
                    stable[0, start_index:end_index] = (kl_buf < epsilon).all(-1)

                    ready = stable & (p > tau) & candidates

                    # fallback for when we're running out of steps
                    if step == steps_per_block - 1:
                        ready = candidates

                    x[ready] = y[ready]

                else:
                    if args.method == "top-margin":
                        pick = top_margin(
                            probabilities, candidates, blanks_per_step, start_index
                        )
                    elif args.method == "random":
                        pick = random(x, candidates, blanks_per_step)
                    elif args.method == "top":
                        pick = top(p, candidates, blanks_per_step)

                    x[pick] = y[pick]

                if args.apply_core and t % 8 == 0 and 0.25 <= t / args.steps < 0.75:
                    out = core(model, x, z, prompt_length, args)
                    if out:
                        tested, core_score, replacement = out
                        if args.method == "core":
                            # overwrite the token the model wants back least
                            worst = int(core_score.argmax())
                            x[0, int(tested[0].nonzero()[worst])] = replacement[worst]

                t = t + 1

    return x
