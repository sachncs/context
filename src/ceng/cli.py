"""Command line interface: `ceng compress | verify | convert | bench`."""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys
from collections.abc import Sequence
from typing import TextIO

from ceng import Context, errors
from ceng import runtime as runtime_lib
from ceng.bench import base as bench_base
from ceng.bench import offline, runner
from ceng.compression import Budget, Overflow
from ceng.evolution import Evolver
from ceng.internals import runner as sync_runner

EXIT_OK = 0
EXIT_FAILED_CHECK = 1
EXIT_USAGE = 2
EXIT_ERROR = 3
DEFAULT_RESULTS_DIR = pathlib.Path("bench/results")
MESSAGES_FORMAT = "messages"


def parse_option(text: str) -> tuple[str, object]:
    """Parses `key=value`; the value is JSON when possible, else a string.

    Raises:
        ConfigError: If there is no `=`.
    """
    key, separator, raw = text.partition("=")
    if not separator or not key:
        raise errors.ConfigError(f"option must look like key=value: {text!r}")
    try:
        return key, json.loads(raw)
    except ValueError:
        return key, raw


def collect_options(
    pairs: Sequence[str], inline_json: str | None
) -> dict[str, object]:
    """Merges `--options-json` with repeated `--option key=value`."""
    options: dict[str, object] = {}
    if inline_json:
        try:
            loaded = json.loads(inline_json)
        except ValueError as exc:
            raise errors.ConfigError(
                f"--options-json is not JSON: {exc}"
            ) from exc
        if not isinstance(loaded, dict):
            raise errors.ConfigError("--options-json must be an object")
        options.update(loaded)
    options.update(parse_option(pair) for pair in pairs)
    return options


def read_text(source: str, stdin: TextIO) -> str:
    """Reads a file, or stdin for `-`."""
    if source == "-":
        return stdin.read()
    try:
        return pathlib.Path(source).read_text(encoding="utf-8")
    except OSError as exc:
        raise errors.ConfigError(f"cannot read {source}: {exc}") from exc


def load_context(
    source: str,
    fmt: str,
    runtime: runtime_lib.Runtime,
    stdin: TextIO,
) -> Context:
    """Loads a context from messages JSON or a registered codec format."""
    if fmt != MESSAGES_FORMAT:
        return Context.load(source, fmt, runtime)
    try:
        document = json.loads(read_text(source, stdin))
    except ValueError as exc:
        raise errors.ValidationError(f"input is not JSON: {exc}") from exc
    raw = document.get("messages") if isinstance(document, dict) else document
    if not isinstance(raw, list):
        raise errors.ValidationError(
            "input must be a list of messages or {'messages': [...]}"
        )
    return Context.from_dicts(raw, runtime)


def write_context(
    context: Context, destination: str | None, fmt: str, stdout: TextIO
) -> None:
    """Writes a context as messages JSON or through a codec."""
    if fmt != MESSAGES_FORMAT:
        if not destination or destination == "-":
            raise errors.ConfigError(f"--to {fmt} needs an --output path")
        context.save(destination, fmt)
        return
    payload = json.dumps(context.to_dicts(), indent=2, ensure_ascii=False)
    if destination and destination != "-":
        pathlib.Path(destination).write_text(payload + "\n", encoding="utf-8")
    else:
        stdout.write(payload + "\n")


def build_parser() -> argparse.ArgumentParser:
    """Builds the argument parser."""
    parser = argparse.ArgumentParser(prog="ceng", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    def add_io(sub: argparse.ArgumentParser) -> None:
        sub.add_argument(
            "input", help="messages JSON file, '-' for stdin, or a bundle"
        )
        sub.add_argument(
            "--from", dest="source_format", default=MESSAGES_FORMAT
        )
        sub.add_argument(
            "--option", action="append", default=[], metavar="KEY=VALUE"
        )
        sub.add_argument("--options-json", help="options as a JSON object")

    compress = commands.add_parser("compress", help="compress a context")
    add_io(compress)
    compress.add_argument("--method", default="ppa")
    compress.add_argument("--budget", type=int, required=True)
    compress.add_argument(
        "--overflow", choices=("raise", "truncate"), default="raise"
    )
    compress.add_argument("--to", dest="target_format", default=MESSAGES_FORMAT)
    compress.add_argument("-o", "--output")

    verify = commands.add_parser("verify", help="verify a context")
    add_io(verify)
    verify.add_argument("--method", default="fits")

    convert = commands.add_parser("convert", help="convert between formats")
    convert.add_argument("input")
    convert.add_argument("output")
    convert.add_argument("--from", dest="source_format", required=True)
    convert.add_argument("--to", dest="target_format", required=True)

    bench = commands.add_parser("bench", help="run a benchmark")
    bench.add_argument("name", help="benchmark name, or 'all'")
    bench.add_argument("--limit", type=int, default=30)
    bench.add_argument("--model")
    bench.add_argument("--evolve", type=int, default=0, metavar="EPOCHS")
    bench.add_argument("--offline", action="store_true")
    bench.add_argument(
        "--results-dir", type=pathlib.Path, default=DEFAULT_RESULTS_DIR
    )
    return parser


def run_compress(
    args: argparse.Namespace, stdin: TextIO, stdout: TextIO, stderr: TextIO
) -> int:
    """Implements `ceng compress`."""
    options = collect_options(args.option, args.options_json)
    with runtime_lib.Runtime.from_env() as runtime:
        context = load_context(args.input, args.source_format, runtime, stdin)
        budget = Budget(args.budget, Overflow(args.overflow))
        result = context.compress(args.method, budget=budget, **options)
        write_context(result, args.output, args.target_format, stdout)
    report = result.report
    if report is not None:
        stderr.write(
            f"{report.method}: {report.original_tokens} -> "
            f"{report.final_tokens} tokens "
            f"({report.ratio:.0%}), {len(report.steps)} steps, "
            f"{report.usage.total_tokens} LLM tokens\n"
        )
    return EXIT_OK


def run_verify(args: argparse.Namespace, stdin: TextIO, stdout: TextIO) -> int:
    """Implements `ceng verify`."""
    options = collect_options(args.option, args.options_json)
    with runtime_lib.Runtime.from_env() as runtime:
        context = load_context(args.input, args.source_format, runtime, stdin)
        verdict = context.verify(args.method, **options)
    stdout.write(json.dumps(dataclasses.asdict(verdict), indent=2) + "\n")
    return EXIT_OK if verdict.passed else EXIT_FAILED_CHECK


def run_convert(args: argparse.Namespace, stdout: TextIO) -> int:
    """Implements `ceng convert`."""
    with runtime_lib.Runtime.from_env(
        {"CENG_BACKEND": "scripted", "CENG_CACHE_DIR": ""}
    ) as runtime:
        context = Context.load(args.input, args.source_format, runtime)
        write_context(context, args.output, args.target_format, stdout)
    return EXIT_OK


def run_bench(args: argparse.Namespace, stdout: TextIO) -> int:
    """Implements `ceng bench`."""
    names = (
        bench_base.Benchmark.registry.names()
        if args.name == "all"
        else [args.name]
    )
    exit_code = EXIT_OK
    for name in names:
        benchmark = bench_base.Benchmark.registry.get(name)()
        samples = benchmark.load_samples(args.limit)
        if args.offline:
            runtime = offline.wiring_runtime(samples)
        else:
            runtime = runtime_lib.Runtime.from_env()
            if args.model:
                runtime = dataclasses.replace(runtime, model=args.model)
        playbook = benchmark.seed_playbook()
        if args.evolve:
            playbook = (
                Evolver(runtime)
                .evolve(playbook, samples, benchmark, epochs=args.evolve)
                .playbook
            )
        arms = (runner.Arm("baseline"), runner.Arm("ceng", playbook))
        result = sync_runner.run_sync(
            runner.Runner(runtime).arun(benchmark, samples, arms)
        )
        path = result.write(args.results_dir)
        stdout.write(result.to_markdown() + f"\nwrote {path}\n")
        if all(arm.backend_errors == result.samples for arm in result.arms):
            exit_code = EXIT_ERROR
    return exit_code


def main(
    argv: Sequence[str] | None = None,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Runs the CLI and returns the process exit code.

    Exit codes: 0 success, 1 failed verification, 2 usage/configuration
    error, 3 runtime failure.
    """
    stdin, stdout, stderr = (
        stdin or sys.stdin,
        stdout or sys.stdout,
        stderr or sys.stderr,
    )
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else EXIT_USAGE
    try:
        if args.command == "compress":
            return run_compress(args, stdin, stdout, stderr)
        if args.command == "verify":
            return run_verify(args, stdin, stdout)
        if args.command == "convert":
            return run_convert(args, stdout)
        return run_bench(args, stdout)
    except errors.ConfigError as exc:
        stderr.write(f"ceng: configuration error: {exc}\n")
        return EXIT_USAGE
    except errors.CengError as exc:
        stderr.write(f"ceng: {type(exc).__name__}: {exc}\n")
        return EXIT_ERROR
