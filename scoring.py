from argparse import ArgumentParser


def _get_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument("--input", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = _get_args()
    if args.dataset == "gsm8k":
        pass
    elif args.dataset == "math":
        pass
    elif args.dataset == "humaneval":
        pass
    elif args.dataset == "mbpp":
        pass

    