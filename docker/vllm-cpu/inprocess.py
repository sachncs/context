"""Exercises the in-process VLLMBackend inside the vLLM CPU image.

    docker run --rm -v "$PWD":/repo -v vllm-cpu_hf-cache:/models -e HF_HOME=/models \
        --entrypoint bash vllm/vllm-openai-cpu:latest \
        -c "cp -r /repo /tmp/src && pip install -q /tmp/src && python /repo/docker/vllm-cpu/inprocess.py"
"""

import asyncio
import os

from foveate import Message, Role, Runtime
from foveate.backends import VLLMBackend
from foveate.cache import NullCache

MODEL = os.environ.get("VLLM_MODEL", "openbmb/MiniCPM5-1B")


async def main() -> None:
    runtime = Runtime(
        backend=VLLMBackend(
            max_model_len=4096,
            dtype="bfloat16",
            trust_remote_code=True,
            gpu_memory_utilization=0.5,  # fraction of RAM on the CPU backend
        ),
        model=MODEL,
        cache=NullCache(),
    )
    reply = await runtime.complete(
        (Message(Role.USER, "What is 12*12? Reply with only the number."),),
        source="inprocess",
        namespace="inprocess",
        max_tokens=1024,
    )
    print("REPLY:", reply.text)
    print("USAGE:", reply.usage)
    assert "144" in reply.text and "<think>" not in reply.text


if __name__ == "__main__":  # vLLM spawns worker processes
    asyncio.run(main())
