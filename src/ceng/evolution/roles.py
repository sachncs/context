"""The three ACE roles as classes: Generator, Reflector, Curator."""

from __future__ import annotations

import dataclasses
import enum
import json
import re
from collections.abc import Mapping

from ceng import errors, prompts
from ceng import messages as messages_lib
from ceng import runtime as runtime_lib
from ceng import usage as usage_lib
from ceng.evolution import grading, operations
from ceng.evolution import playbook as playbook_lib
from ceng.evolution import prompts as ace_prompts

MAX_COMPLETION_TOKENS = 4096
FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def extract_json(text: str) -> dict[str, object]:
    """Parses a JSON object out of model text, tolerating code fences.

    Raises:
        ValidationError: If no JSON object can be recovered.
    """
    cleaned = FENCE.sub("", text.strip())
    candidates = [cleaned]
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if 0 <= start < end:
        candidates.append(cleaned[start : end + 1])
    for candidate in candidates:
        try:
            loaded = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(loaded, dict):
            return loaded
    raise errors.ValidationError(
        f"no JSON object in model output: {text[:80]!r}"
    )


@dataclasses.dataclass(frozen=True, slots=True)
class Reply:
    """A parsed role response plus accounting.

    Attributes:
        data: Parsed JSON object.
        usage: Provider-reported tokens.
        cached: Whether the answer came from the cache.
    """

    data: Mapping[str, object]
    usage: usage_lib.Usage
    cached: bool


@dataclasses.dataclass(frozen=True)
class Stage:
    """Shared plumbing: render a template, call the LLM, parse JSON."""

    async def ask(
        self,
        runtime: runtime_lib.Runtime,
        template: prompts.PromptTemplate,
        **values: str,
    ) -> Reply:
        """Calls the model with `template` and parses its JSON reply.

        The request fingerprint covers the full rendered prompt, so cache
        keys always reflect the playbook, question and reflection.

        Raises:
            BackendError: If the backend fails after its retries.
            ValidationError: If the reply is not a JSON object.
        """
        completion = await runtime.complete(
            (
                messages_lib.Message(
                    messages_lib.Role.USER, template.render_user(**values)
                ),
            ),
            source="evolution",
            namespace=f"evolution:{template.fingerprint}",
            max_tokens=MAX_COMPLETION_TOKENS,
        )
        return Reply(
            extract_json(completion.text), completion.usage, completion.cached
        )


def string_list(value: object) -> tuple[str, ...]:
    """Coerces a JSON list into a tuple of strings (else empty)."""
    if not isinstance(value, list):
        return ()
    return tuple(str(item) for item in value if isinstance(item, (str, int)))


@dataclasses.dataclass(frozen=True, slots=True)
class GeneratorOutput:
    """What the Generator produced.

    Attributes:
        reasoning: Chain of thought.
        bullet_ids: Playbook bullets the model says it used.
        final_answer: The answer to grade.
        reply: Accounting for the call.
    """

    reasoning: str
    bullet_ids: tuple[str, ...]
    final_answer: str
    reply: Reply


@dataclasses.dataclass(frozen=True)
class Generator(Stage):
    """Answers a question using the playbook."""

    template: prompts.PromptTemplate = ace_prompts.GENERATOR_PROMPT

    async def run(
        self,
        runtime: runtime_lib.Runtime,
        playbook: playbook_lib.Playbook,
        sample: grading.Sample,
        reflection: str = "(empty)",
    ) -> GeneratorOutput:
        """Generates an answer for `sample`."""
        reply = await self.ask(
            runtime,
            self.template,
            playbook=playbook.render(),
            reflection=reflection,
            question=sample.question,
            context=sample.context,
        )
        data = reply.data
        return GeneratorOutput(
            reasoning=str(data.get("reasoning") or ""),
            bullet_ids=string_list(data.get("bullet_ids")),
            final_answer=str(data.get("final_answer") or "").strip(),
            reply=reply,
        )


class Tag(str, enum.Enum):
    """Reflector verdict on one bullet."""

    HELPFUL = "helpful"
    HARMFUL = "harmful"
    NEUTRAL = "neutral"


@dataclasses.dataclass(frozen=True, slots=True)
class Reflection:
    """What the Reflector concluded.

    Attributes:
        insight: Key lesson (falls back to the error identification).
        tags: Verdict per bullet id.
        reply: Accounting for the call.
    """

    insight: str
    tags: tuple[tuple[str, Tag], ...]
    reply: Reply


@dataclasses.dataclass(frozen=True)
class Reflector(Stage):
    """Diagnoses a wrong (or unverified) answer and tags used bullets."""

    with_truth: prompts.PromptTemplate = ace_prompts.REFLECTOR_PROMPT
    without_truth: prompts.PromptTemplate = ace_prompts.REFLECTOR_PROMPT_NO_GT

    async def run(
        self,
        runtime: runtime_lib.Runtime,
        playbook: playbook_lib.Playbook,
        sample: grading.Sample,
        answer: GeneratorOutput,
        feedback: str,
        use_ground_truth: bool,
    ) -> Reflection:
        """Reflects on `answer`."""
        used = (
            "\n".join(
                b.render()
                for bid in answer.bullet_ids
                if (b := playbook.get(bid)) is not None
            )
            or "(generator referenced no bullets)"
        )
        values = {
            "question": sample.question,
            "reasoning_trace": answer.reasoning,
            "predicted_answer": answer.final_answer,
            "environment_feedback": feedback,
            "bullets_used": used,
        }
        if use_ground_truth:
            reply = await self.ask(
                runtime, self.with_truth, ground_truth=sample.target, **values
            )
        else:
            reply = await self.ask(runtime, self.without_truth, **values)
        data = reply.data
        tags = []
        raw_tags = data.get("bullet_tags")
        for item in raw_tags if isinstance(raw_tags, list) else []:
            if not isinstance(item, Mapping):
                continue
            ident = (
                item.get("id") or item.get("bullet_id") or item.get("bullet")
            )
            try:
                tag = Tag(str(item.get("tag") or "neutral").lower())
            except ValueError:
                tag = Tag.NEUTRAL
            if ident:
                tags.append((str(ident), tag))
        insight = str(
            data.get("key_insight") or data.get("error_identification") or ""
        ).strip()
        return Reflection(insight, tuple(tags), reply)


@dataclasses.dataclass(frozen=True, slots=True)
class CuratorOutput:
    """What the Curator proposed.

    Attributes:
        ops: Typed operations in proposal order.
        reply: Accounting for the call.
    """

    ops: tuple[operations.CuratorOp, ...]
    reply: Reply


@dataclasses.dataclass(frozen=True)
class Curator(Stage):
    """Proposes playbook edits from accumulated reflections."""

    with_truth: prompts.PromptTemplate = ace_prompts.CURATOR_PROMPT
    without_truth: prompts.PromptTemplate = ace_prompts.CURATOR_PROMPT_NO_GT

    async def run(
        self,
        runtime: runtime_lib.Runtime,
        playbook: playbook_lib.Playbook,
        sample: grading.Sample,
        reflection: str,
        step: int,
        total_steps: int,
        token_budget: int,
        use_ground_truth: bool,
    ) -> CuratorOutput:
        """Proposes operations for the current playbook."""
        reply = await self.ask(
            runtime,
            self.with_truth if use_ground_truth else self.without_truth,
            token_budget=str(token_budget),
            current_step=str(step),
            total_samples=str(total_steps),
            playbook_stats=playbook.stats(),
            recent_reflection=reflection,
            current_playbook=playbook.render(),
            question_context=sample.context or sample.question,
        )
        raw = reply.data.get("operations")
        parsed = (
            [operations.CuratorOp.parse(op) for op in raw]
            if isinstance(raw, list)
            else []
        )
        return CuratorOutput(
            tuple(op for op in parsed if op is not None), reply
        )
