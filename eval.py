import dataclasses
import re
import time
import os
import sys
import math
from pathlib import Path
from typing import Any
from tabulate import tabulate

import torch.cuda

from lib.utils import set_seed
import importlib

module = sys.argv[1]

try:
    task_module = importlib.import_module(f"{module}.task")
    TestSpec = getattr(task_module, "TestSpec", dict)  # default to dict

    submission_module = importlib.import_module(f"{module}.submission")
    custom_kernel = getattr(submission_module, "custom_kernel", None)

    reference_module = importlib.import_module(f"{module}.reference")
    generate_input = getattr(reference_module, "generate_input", None)
    check_implementation = getattr(reference_module, "check_implementation", None)

    if not custom_kernel:
        print("Error: `custom_kernel` not found in submission.py!")
        sys.exit(1)
    if not generate_input or not check_implementation:
        print(
            "Error: `generate_input` or `check_implementation` not found in reference.py!"
        )
        sys.exit(1)

except ModuleNotFoundError as e:
    print(f"Error: {e}")
    sys.exit(1)


WARMUP_RUNS = 10
TIMED_RUNS = 100


class PopcornOutput:
    def __init__(self, fd: int):
        self.file = os.fdopen(fd, "w")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()

    def print(self, *args, **kwargs):
        print(*args, **kwargs, file=self.file, flush=True)

    def log(self, key, value):
        self.print(f"{key}: {value}")


@dataclasses.dataclass
class TestCase:
    args: dict
    spec: str


def get_test_cases(file_name: str) -> list[TestCase]:
    try:
        content = Path(file_name).read_text()
    except Exception as E:
        print(f"Could not open test file`{file_name}`: {E}", file=sys.stderr)
        exit(113)

    tests = []
    lines = content.splitlines()
    match = r"\s*([a-zA-Z]+):\s*([a-zA-Z]+|[+-]?[0-9]+)\s*"
    for line in lines:
        parts = line.split(";")
        case = {}
        for part in parts:
            matched = re.match(match, part)
            if not re.fullmatch(match, part):
                print(f"invalid test case: '{line}': '{part}'", file=sys.stderr)
                exit(113)
            key = matched[1]
            val = matched[2]
            try:
                val = int(val)
            except ValueError:
                pass

            case[key] = val
        tests.append(TestCase(spec=line, args=case))

    return tests


def warm_up(test: TestCase):
    data = generate_input(**test.args)
    start = time.perf_counter()
    while time.perf_counter() - start < 0.2:
        custom_kernel(data)
        torch.cuda.synchronize()


@dataclasses.dataclass
class Stats:
    runs: int
    mean: float
    std: float
    err: float
    best: float
    worst: float
    name: str = ""


def calculate_stats(durations: list[int]):
    """
    Calculate statistical data from a list of durations.

    @param durations: A list of durations in nanoseconds.
    @return: A Stats object containing the number of runs, mean, standard deviation, error, best, and worst durations.
    """
    runs = len(durations)
    total = sum(durations)
    best = min(durations)
    worst = max(durations)

    avg = total / runs
    variance = sum(map(lambda x: (x - avg) ** 2, durations))
    std = math.sqrt(variance / (runs - 1))
    err = std / math.sqrt(runs)

    return Stats(
        runs=runs, mean=avg, std=std, err=err, best=float(best), worst=float(worst)
    )


def print_stats_table(stats_list: list[Stats]):
    """
    Prints a formatted table displaying statistics for each benchmark.

    :param stats_list: List of Stats objects.
    """

    # Prepare table data
    table_data = []
    for stats in stats_list:
        table_data.append(
            [
                stats.name,
                stats.runs,
                f"{stats.mean / 1000:.2f}",  # Convert ns to µs
                f"{stats.best / 1000:.2f}",  # Convert ns to µs
                f"{stats.worst / 1000:.2f}",  # Convert ns to µs
                f"{stats.std:.2f}",
                f"{stats.err:.2f}",
            ]
        )

    # Define table headers
    headers = ["Benchmark", "Runs", "Mean (µs)", "Best (µs)", "Worst (µs)", "Std Dev", "Error"]

    # Print table using `tabulate`
    print(tabulate(table_data, headers=headers, tablefmt="grid"))


def run_testing(logger: PopcornOutput, tests: list[TestCase]):
    """
    Executes the actual test case code and checks for correctness.

    @param logger: A PopcornOutput object used for logging test results.
    @param tests: A list of TestCase objects representing the test cases to be executed.
    @return: An integer representing the exit status: 0 if all tests pass, otherwise 112.
    """
    passed = True
    logger.log("test-count", len(tests))
    for idx, test in enumerate(tests):
        logger.log(f"test.{idx}.spec", test.spec)

        data = generate_input(**test.args)
        torch.cuda.synchronize()
        submission_output = custom_kernel(data)
        torch.cuda.synchronize()
        error = check_implementation(data, submission_output)
        if error:
            logger.log(f"test.{idx}.status", "fail")
            logger.log(f"test.{idx}.error", error)
            passed = False
        else:
            logger.log(f"test.{idx}.status", "pass")

    if passed:
        logger.log("check", "pass")
        return 0
    else:
        logger.log("check", "fail")
        return 112


def benchmark(
    test: TestCase, recheck: bool, max_repeats: int, max_time_ns: float
) -> Stats | Any:
    """
    For a particular test case, check correctness (if applicable) and grab runtime results.

    @param test: TestCase object.
    @param recheck: Flag for whether to explicitly check functional correctness.
    @param max_repeats: Number of trials to repeat.
    @param max_time_ns: Timeout time in nanoseconds.
    @return: A Stats object for this particular benchmark case or an error if the test fails.
    """
    durations = []
    # generate input data once
    data = generate_input(**test.args)
    # first, one obligatory correctness check; also triggers triton compile for the given shape
    output = custom_kernel(data)
    error = check_implementation(data, output)
    if error:
        return error

    # now, do multiple timing runs without further correctness testing
    # there is an upper bound of 100 runs, and a lower bound of 3 runs;
    # otherwise, we repeat until we either measure at least 10 full seconds,
    # or the relative error of the mean is below 1%.

    for i in range(max_repeats):
        if recheck:
            data = generate_input(**test.args)
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        output = custom_kernel(data)
        torch.cuda.synchronize()
        end = time.perf_counter_ns()

        if recheck:
            error = check_implementation(data, output)
            if error:
                return error

        del output
        durations.append(end - start)

        if i > 1:
            stats = calculate_stats(durations)
            if stats.err / stats.mean < 0.01 or stats.mean * stats.runs > max_time_ns:
                break

    return calculate_stats(durations)


def run_benchmarking(logger: PopcornOutput, tests: list[TestCase]):
    """
    Executes benchmarking code for a CUDA Kernel and logs runtimes.

    @param logger: A PopcornOutput object used for logging benchmark results.
    @param tests: A list of TestCase objects representing the test cases to be benchmarked.
    @return: An integer representing the exit status: 0 if all benchmarks pass, otherwise 112.
    """
    warm_up(tests[0])
    passed = True
    logger.log("benchmark-count", len(tests))
    results = []
    for idx, test in enumerate(tests):
        result = benchmark(test, False, 100, 10e9)
        if isinstance(result, Stats):
            result.name = test.spec
            results.append(result)
        else:
            passed = False

            logger.log(f"benchmark.{idx}.status", "fail")
            logger.log(f"benchmark.{idx}.error", result)

    if passed:
        logger.log("check", "pass")
        print_stats_table(results)
        return
    else:
        logger.log("check", "fail")
        return


def main():
    fd = os.getenv("POPCORN_FD") or 1

    if len(sys.argv) < 3:
        sys.exit(1)

    mode = sys.argv[2]
    tests = get_test_cases(
        os.path.join(f"{module}", f"{mode if mode == 'test' else 'benchmark'}.txt")
    )

    with PopcornOutput(int(fd)) as logger:
        seed = os.getenv("POPCORN_SEED")
        seed = int(seed) if seed else 42
        set_seed(seed)

        if mode == "test":
            return run_testing(logger, tests)

        if mode == "benchmark":
            return run_benchmarking(logger, tests)

        if mode == "leaderboard":
            warm_up(tests[0])
            result = benchmark(tests[-1], True, 100, 30e9)
            if isinstance(result, Stats):
                logger.log("benchmark-count", 1)
                logger.log("check", "pass")
                result.name = tests[-1].spec
                print_stats_table([result])
            else:
                logger.log("test-count", 1)
                logger.log("test.0.status", "fail")
                logger.log(
                    "test.0.error", str(result)
                )  # TODO: Make sure result implements __str__?



if __name__ == "__main__":
    sys.exit(main())
