import io
import json

import pytest

from ceng import cli


@pytest.fixture(autouse=True)
def offline_env(monkeypatch, tmp_path):
    monkeypatch.setenv("CENG_BACKEND", "scripted")
    monkeypatch.setenv("CENG_CACHE_DIR", "")
    monkeypatch.chdir(tmp_path)


def run(*argv, stdin=""):
    out, err = io.StringIO(), io.StringIO()
    code = cli.main(list(argv), io.StringIO(stdin), out, err)
    return code, out.getvalue(), err.getvalue()


MESSAGES = [
    {"role": "system", "content": "be brief"},
    {"role": "user", "content": " ".join(f"word{i}" for i in range(400))},
]


def write_input(tmp_path, data=MESSAGES):
    path = tmp_path / "in.json"
    path.write_text(json.dumps(data))
    return str(path)


def test_compress_truncate_to_stdout(tmp_path):
    code, out, err = run(
        "compress",
        write_input(tmp_path),
        "--method",
        "truncate",
        "--budget",
        "50",
    )
    assert code == 0
    messages = json.loads(out)
    assert messages[0]["content"] == "be brief" and len(messages) == 2
    assert "truncate: " in err and "tokens" in err


def test_compress_wrapped_messages_stdin_and_output_file(tmp_path):
    out_path = tmp_path / "out.json"
    code, out, _ = run(
        "compress",
        "-",
        "--method",
        "truncate",
        "--budget",
        "50",
        "-o",
        str(out_path),
        "--option",
        "keep=head",
        stdin=json.dumps({"messages": MESSAGES}),
    )
    assert code == 0 and out == ""
    assert json.loads(out_path.read_text())[1]["content"].startswith("word0")


def test_compress_to_okf_and_convert_roundtrip(tmp_path):
    bundle = tmp_path / "bundle"
    code, _, _ = run(
        "compress",
        write_input(tmp_path),
        "--method",
        "extractive",
        "--budget",
        "100",
        "--overflow",
        "truncate",
        "--to",
        "okf",
        "-o",
        str(bundle),
    )
    assert code == 0 and (bundle / "index.md").exists()
    code, _, _ = run(
        "convert",
        str(bundle),
        str(tmp_path / "c.json"),
        "--from",
        "okf",
        "--to",
        "json",
    )
    assert code == 0
    assert "messages" in json.loads((tmp_path / "c.json").read_text())


def test_compress_okf_needs_output(tmp_path):
    code, _, err = run(
        "compress",
        write_input(tmp_path),
        "--method",
        "truncate",
        "--budget",
        "50",
        "--to",
        "okf",
    )
    assert code == 2 and "--output" in err


def test_budget_error_is_runtime_failure(tmp_path):
    code, _, err = run(
        "compress",
        write_input(tmp_path),
        "--method",
        "window",
        "--budget",
        "5",
        "--option",
        "min_messages=5",
    )
    assert code == 3 and "BudgetExceededError" in err


def test_verify_exit_codes(tmp_path):
    path = write_input(tmp_path)
    code, out, _ = run(
        "verify", path, "--method", "fits", "--option", "tokens=100000"
    )
    assert code == 0 and json.loads(out)["passed"] is True
    code, out, _ = run(
        "verify", path, "--method", "fits", "--options-json", '{"tokens": 5}'
    )
    assert code == 1 and json.loads(out)["passed"] is False


@pytest.mark.parametrize(
    "argv",
    [
        ["compress", "{in}", "--method", "nope", "--budget", "10"],
        ["compress", "{in}", "--budget", "10", "--option", "noequals"],
        ["compress", "{in}", "--budget", "10", "--options-json", "[1]"],
        ["compress", "{in}", "--budget", "10", "--options-json", "{bad"],
        ["compress", "/no/such/file.json", "--budget", "10"],
    ],
)
def test_configuration_errors_exit_2(tmp_path, argv):
    argv = [a.replace("{in}", write_input(tmp_path)) for a in argv]
    code, _, err = run(*argv)
    assert code == 2 and "configuration error" in err


def test_invalid_input_exit_3(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"not": "messages"}')
    code, _, err = run("compress", str(bad), "--budget", "10")
    assert code == 3 and "ValidationError" in err
    bad.write_text("nope")
    assert run("compress", str(bad), "--budget", "10")[0] == 3


def test_argparse_errors_return_codes():
    assert run()[0] == 2
    assert run("--help")[0] == 0


def test_option_value_parsing():
    assert cli.parse_option("a=3") == ("a", 3)
    assert cli.parse_option("a=true") == ("a", True)
    assert cli.parse_option("a=hello") == ("a", "hello")
    assert cli.parse_option("a=[1,2]") == ("a", [1, 2])


def test_bench_offline_shows_delta_only_with_playbook(tmp_path):
    results = tmp_path / "results"
    code, out, _ = run(
        "bench",
        "formula",
        "--offline",
        "--limit",
        "5",
        "--results-dir",
        str(results),
    )
    assert code == 0
    assert "| baseline | 0.000" in out and "| ceng | 1.000" in out
    assert list(results.glob("formula-*.json")) and list(
        results.glob("formula-*.md")
    )


def test_bench_all_offline(tmp_path):
    code, out, _ = run(
        "bench",
        "all",
        "--offline",
        "--limit",
        "3",
        "--results-dir",
        str(tmp_path / "r"),
    )
    assert code == 0
    for name in ("finer", "formula", "ddxplus"):
        assert f"# {name} - ceng benchmark" in out


def test_bench_unknown_name(tmp_path):
    code, _, err = run("bench", "nope", "--offline")
    assert code == 2 and "unknown benchmark" in err


def test_bench_evolve_runs_with_scripted_model(tmp_path):
    # ScriptedBackend echoes the prompt, so evolution parses nothing useful
    # but must complete without crashing and still produce a report.
    code, out, _ = run(
        "bench",
        "ddxplus",
        "--limit",
        "2",
        "--evolve",
        "1",
        "--results-dir",
        str(tmp_path / "r"),
    )
    assert code == 0 and "Measured by ceng" in out
