# Local models with vLLM

Small local models have 4-32k windows, so "send the whole document" is not an
option. That is where Foveate matters most.

## Serve a model (Linux CPU in Docker, works on macOS via colima)

```bash
colima start --cpu 4 --memory 10        # macOS only; Docker Desktop also works
docker compose -f docker/vllm-cpu/compose.yml up -d
```

The compose file runs `vllm/vllm-openai-cpu` with `openbmb/MiniCPM5-1B`,
`--max-model-len 8192` and a cached model volume. Change the model with
`VLLM_MODEL=...` and the window with `VLLM_MAX_LEN=...`. The first start
downloads the weights (about 2 GB for the 1B model).

Then use it like any OpenAI-compatible endpoint:

```bash
export FOVEATE_BACKEND=openai
export FOVEATE_BASE_URL=http://localhost:8000/v1
export FOVEATE_MODEL=openbmb/MiniCPM5-1B
export FOVEATE_CONTEXT_WINDOW=8192
export OPENAI_API_KEY=local
```

```python
from foveate import Document, Foveator, Runtime

with Runtime.from_env() as runtime:
    answer = Foveator(runtime, budget=3500).ask("...", [Document.load("report.pdf")])
```

## In-process engine

On a Linux machine with vLLM installed (`pip install "foveate[vllm]"`):

```python
from foveate import Runtime
from foveate.backends import VLLMBackend

runtime = Runtime(
    backend=VLLMBackend(max_model_len=8192),
    model="openbmb/MiniCPM5-1B",
)
```

Keep the call under `if __name__ == "__main__":` (vLLM spawns worker
processes). On a CPU-only machine also pass `gpu_memory_utilization=0.5`: on
the CPU backend it is the fraction of RAM to reserve, and the default fails when
other programs use memory. `docker/vllm-cpu/inprocess.py` is a working example.

vLLM publishes no macOS wheels; on a Mac use the Docker server above.

## What to expect from small models

* Reasoning models such as MiniCPM5 print `<think>...</think>`; Foveate removes
  it before parsing and caches the visible answer only.
* CPU generation is slow (about 12 tokens/s for the 1B model on 4 cores).
  Budget accordingly, and set `max_concurrency=1` on the backend.
* Small models find the right value but often skip the citation JSON field.
  Foveate marks such answers `grounded=False` (it never pretends), and
  `ungrounded="abstain"` turns them into "not found" if you prefer. See
  [benchmarks](../benchmarks.md) for measured rates.

## Tests

`FOVEATE_TEST_VLLM_URL=http://localhost:8000/v1 pytest tests/local` runs the
suite against your server. It is skipped when the variable is unset.
