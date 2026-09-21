import re
import os
import glob
import json
import tempfile
import contextlib
import numpy as np
import multiprocessing
from datasets import load_dataset
from shared import datasets_lookup
from argparse import ArgumentParser
from pandas import DataFrame, Series
from math_verify import parse, verify


def _get_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument("--input", required=True)
    return parser.parse_args()

# the regex for the expected method names
re_name_expected   = re.compile(r"\b([A-Za-z_]\w*)\s*\(")

# the regex for the generated method names
re_name_generated = re.compile(r"^def\s+(\w+)", re.M)

# in case the generated methods are wrapped in some Python method, so we skip those.
to_skip   = {"assert","math","isclose","abs","len","set","sorted","list","tuple",
          "str","int","float","round","all","any","print","type","dict","map","filter"}

def match_signature(completion, row):
    """Add a statement to the end of the generated code to equate the two method name aliases."""

    expected = next((n for t in row["test_list"] for n in re_name_expected.findall(t) if n not in to_skip), None)
    defined  = set(re_name_generated.findall(completion))
    if not expected or expected in defined or len(defined) != 1:
        return completion
    return completion + f"\n{expected} = {next(iter(defined))}\n"

def build_humaneval(problem, code):
     """Build the HumanEval prompt."""
     return problem["prompt"] + code + "\n" + problem["test"] + "\n" + f"check({problem['entry_point']})"
        
def build_mbpp(problem, code):
    """Build the MBPP prompt."""
    parts = [code]
    if problem.get("test_setup_code"):
        parts.append(problem["test_setup_code"])
    parts.extend(problem['test_list'])
    return "\n".join(parts)

@contextlib.contextmanager
def create_tempdir():
    with tempfile.TemporaryDirectory() as dirname:
        with chdir(dirname):
            yield dirname

@contextlib.contextmanager
def chdir(root):
    if root == ".":
        yield
        return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    except BaseException as exc:
        raise exc
    finally:
        os.chdir(cwd)

def check_correctness(program, results):
    """Check whether the generated process ran."""
    with create_tempdir():
        try:
            exec_globals = {}
            exec(program, exec_globals)
            results.put('passed')
            print("passed")
        except TimeoutError:
            print("timed out")
            results.put("timed out")
        except BaseException as e:
            print('failed')
            results.put(f"failed: {type(e).__name__}: {e}")
        

def execute(program, task_id, timeout):
    """Run the generated code in subprocesses."""

    manager = multiprocessing.Manager()
    
    q = manager.Queue()

    p = multiprocessing.Process(target=check_correctness, args=(program, q))
    p.start()
    p.join(timeout= timeout + 1)

    if p.is_alive():
        p.kill()

    result = q.get() if not q.empty() else "timed out"
    return dict(
        task_id = task_id,
        passed = result == 'passed',
        result = result
    )


if __name__ == "__main__":
    args = _get_args()

    pattern = re.compile(rf"results_(?P<model>.+?)_(?P<method>baseline|rethink|core|compare|pressure)" rf"_(?P<epsilon>[^_]+)_{re.escape(args.dataset)}_steps(?P<steps>\d+)(?P<random>_random)?\.json$"
            )

    df = DataFrame()
    dataset = load_dataset(datasets_lookup[args.dataset], "main" if args.dataset == "gsm8k" else None, split="test")


    for file_name in sorted(os.listdir(args.input)):
        match = pattern.match(file_name)
        if not match:
            continue
        with open(os.path.join(args.input, file_name), "r") as f:
            results = json.load(f)

        method = match['method']
        epsilon = float(match['epsilon'])
        steps = int(match['steps'])
        is_random = match['random'] is not None

        correct = []
        for instance in results:
            if args.dataset == "gsm8k":
                is_correct = verify(parse(dataset[instance["i"]]['answer'].split("####")[-1]), parse(instance['output']))
                correct.append(is_correct)
            elif args.dataset == "math":
                is_correct = verify(parse("$" + dataset[instance['i']]['answer'] + "$"), parse(instance['output']))
                correct.append(is_correct)
            elif args.dataset == "humaneval":
                correct.append(execute(build_humaneval(dataset[instance['i']]['prompt'], instance['output'], dataset[instance['i']]['test'], dataset[instance['i']]['entry_point']), instance['i'], 10)['passed'])
            elif args.dataset == "mbpp":
                correct.append(execute(correct.append(execute(build_mbpp(instance['output'], dataset[instance['i']]['test_list'], dataset[instance['i']]['test_setup_code']), instance['i'], 10)['passed'])))
        df[method, epsilon, steps, is_random] = Series(correct, index=[inst["i"] for inst in results])

    print(df)
    