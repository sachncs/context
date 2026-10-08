# vLLM on CPU (Docker)

Runs a small model so you can try Foveate locally, and tests the OpenAI-compatible
server path and the in-process `VLLMBackend`. vLLM publishes no macOS wheels, so
on a Mac use Docker (colima or Docker Desktop). Verified with vLLM 0.31.0,
`openbmb/MiniCPM5-1B`, on arm64 Linux in colima (4 CPUs, 10 GB).

## Server

```bash
colima start --cpu 4 --memory 10       # macOS; skip if Docker Desktop is running
docker compose -f docker/compose.yml up -d
curl localhost:8000/v1/models
```

`VLLM_MODEL` and `VLLM_MAX_LEN` override the model and window. The first start
downloads the weights into the `hf-cache` volume.

```bash
export FOVEATE_TEST_VLLM_URL=http://localhost:8000/v1
pytest tests/local
```

## In-process engine

Stop the server first (the engine needs the memory), then:

```bash
docker run --rm -v "$PWD":/repo:ro -v foveate-vllm_hf-cache:/models -e HF_HOME=/models \
  --shm-size 4g --entrypoint bash vllm/vllm-openai-cpu:latest \
  -c "cp -r /repo /tmp/src && pip install -q /tmp/src && python /repo/docker/inprocess.py"
```

Notes from running it:

* On the CPU backend `gpu_memory_utilization` is the fraction of **RAM**
  reserved; the default 0.92 fails when other processes use memory. Pass
  `VLLMBackend(gpu_memory_utilization=0.5, ...)`.
* vLLM starts worker processes, so the calling script needs an
  `if __name__ == "__main__":` guard.
* MiniCPM5 prints `<think>...</think>`; Foveate removes it.
