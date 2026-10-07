"""The Evolver: Generator -> Reflector -> Curator over a dataset."""

from __future__ import annotations

import dataclasses
import json
import logging
import pathlib
from collections.abc import Sequence

from ceng import errors
from ceng import runtime as runtime_lib
from ceng import usage as usage_lib
from ceng.evolution import grading, roles
from ceng.evolution import playbook as playbook_lib
from ceng.internals import runner
from ceng.okf import model as okf_model

logger = logging.getLogger("ceng.evolution")
NO_ERROR_REFLECTION = "(model produced correct answer; no error to reflect on)"
RETRY_NOTE = (
    "\n(Attempt {attempt}: the previous reflection had no key insight.)"
)


@dataclasses.dataclass(frozen=True, slots=True)
class EvolverConfig:
    """Hyper-parameters (defaults follow the ACE paper's best sweep).

    Attributes:
        max_reflector_rounds: Reflection attempts per wrong answer.
        playbook_token_budget: Playbook size ceiling, enforced by trimming.
        curator_frequency: Run the Curator every N steps (1 = every step).
        use_ground_truth: Grade answers and reflect only on mistakes; when
            False no grading happens and every answer is reflected on.
        fail_fast: Re-raise backend errors instead of skipping the step.
    """

    max_reflector_rounds: int = 5
    playbook_token_budget: int = 80_000
    curator_frequency: int = 1
    use_ground_truth: bool = True
    fail_fast: bool = False

    def __post_init__(self) -> None:
        if self.max_reflector_rounds < 1 or self.curator_frequency < 1:
            raise errors.ConfigError(
                "max_reflector_rounds and curator_frequency must be >= 1"
            )
        if self.playbook_token_budget < 1:
            raise errors.ConfigError("playbook_token_budget must be >= 1")


@dataclasses.dataclass(frozen=True, slots=True)
class StepStats:
    """Measurements for one sample.

    Attributes:
        epoch: Zero-based pass over the data.
        step: One-based position within the pass.
        correct: Grade of the generated answer; None when ungraded or the
            step failed before an answer existed.
        bullets_added: Bullets the Curator added.
        bullets_removed: Bullets removed by operations.
        reflector_rounds: Reflection attempts made.
        parse_failures: Role replies that were not valid JSON.
        backend_errors: Backend failures (not model mistakes).
        usage: Tokens spent.
        cache_hits: LLM calls served from the cache.
    """

    epoch: int
    step: int
    correct: bool | None = None
    bullets_added: int = 0
    bullets_removed: int = 0
    reflector_rounds: int = 0
    parse_failures: int = 0
    backend_errors: int = 0
    usage: usage_lib.Usage = dataclasses.field(default_factory=usage_lib.Usage)
    cache_hits: int = 0


@dataclasses.dataclass(frozen=True, slots=True)
class EvolutionResult:
    """Outcome of an evolution run.

    Attributes:
        playbook: The evolved playbook.
        steps: Per-sample statistics in execution order.
    """

    playbook: playbook_lib.Playbook
    steps: tuple[StepStats, ...]

    @property
    def accuracy(self) -> float | None:
        """Returns the share of graded steps answered correctly."""
        graded = [s.correct for s in self.steps if s.correct is not None]
        return sum(graded) / len(graded) if graded else None


@dataclasses.dataclass(frozen=True, slots=True)
class Checkpoint:
    """On-disk progress so a long run can resume after a crash.

    Attributes:
        path: JSON file location.
    """

    path: pathlib.Path

    def save(self, playbook: playbook_lib.Playbook, completed: int) -> None:
        """Atomically records the playbook and completed step count."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".tmp")
        temporary.write_text(
            json.dumps(
                {"completed": completed, "playbook": playbook.to_json()}
            ),
            encoding="utf-8",
        )
        temporary.replace(self.path)

    def load(self) -> tuple[playbook_lib.Playbook, int] | None:
        """Returns `(playbook, completed)` or None if no checkpoint exists.

        Raises:
            ValidationError: If the file is unreadable or malformed.
        """
        if not self.path.exists():
            return None
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return playbook_lib.Playbook.from_json(data["playbook"]), int(
                data["completed"]
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise errors.ValidationError(
                f"bad checkpoint {self.path}: {exc}"
            ) from exc


@dataclasses.dataclass(frozen=True)
class Evolver:
    """Evolves a playbook by learning from a dataset.

    Attributes:
        runtime: Backend, cache and tokenizer.
        config: Hyper-parameters.
        generator: Answering stage.
        reflector: Diagnosis stage.
        curator: Editing stage.
    """

    runtime: runtime_lib.Runtime
    config: EvolverConfig = dataclasses.field(default_factory=EvolverConfig)
    generator: roles.Generator = dataclasses.field(
        default_factory=roles.Generator
    )
    reflector: roles.Reflector = dataclasses.field(
        default_factory=roles.Reflector
    )
    curator: roles.Curator = dataclasses.field(default_factory=roles.Curator)

    def evolve(
        self,
        playbook: playbook_lib.Playbook | None,
        samples: Sequence[grading.Sample],
        grader: grading.Grader,
        *,
        epochs: int = 1,
        checkpoint: pathlib.Path | None = None,
    ) -> EvolutionResult:
        """Synchronous form of `aevolve`; see there for details."""
        return runner.run_sync(
            self.aevolve(
                playbook, samples, grader, epochs=epochs, checkpoint=checkpoint
            )
        )

    async def aevolve(
        self,
        playbook: playbook_lib.Playbook | None,
        samples: Sequence[grading.Sample],
        grader: grading.Grader,
        *,
        epochs: int = 1,
        checkpoint: pathlib.Path | None = None,
    ) -> EvolutionResult:
        """Runs the loop for `epochs` passes over `samples`.

        Args:
            playbook: Starting playbook; None starts empty.
            samples: Training samples.
            grader: Decides correctness and produces feedback.
            epochs: Passes over the data (>= 1).
            checkpoint: If given, progress is saved after every step and an
                existing checkpoint is resumed.

        Returns:
            The evolved playbook and per-step statistics (steps skipped by
            a resume are not repeated in `steps`).

        Raises:
            ConfigError: If `epochs` < 1.
            BackendError: Only when `config.fail_fast` is set.
        """
        if epochs < 1:
            raise errors.ConfigError("epochs must be >= 1")
        current = playbook or playbook_lib.Playbook()
        store = Checkpoint(checkpoint) if checkpoint else None
        skip = 0
        if store is not None and (saved := store.load()) is not None:
            current, skip = saved
        stats: list[StepStats] = []
        pending: list[str] = []
        position = 0
        for epoch in range(epochs):
            for step, sample in enumerate(samples, start=1):
                position += 1
                if position <= skip:
                    continue
                current, step_stats = await self.step(
                    current, sample, grader, epoch, step, len(samples), pending
                )
                stats.append(step_stats)
                if store is not None:
                    store.save(current, position)
        return EvolutionResult(current, tuple(stats))

    async def step(
        self,
        playbook: playbook_lib.Playbook,
        sample: grading.Sample,
        grader: grading.Grader,
        epoch: int,
        step: int,
        total: int,
        pending: list[str],
    ) -> tuple[playbook_lib.Playbook, StepStats]:
        """Processes one sample, isolating per-step failures."""
        counters = Counters()
        try:
            return await self.process(
                playbook, sample, grader, epoch, step, total, pending, counters
            )
        except errors.BackendError as exc:
            if self.config.fail_fast:
                raise
            logger.warning("step %d failed: %s", step, exc)
            counters.backend_errors += 1
        except errors.ValidationError as exc:
            logger.warning("step %d reply unusable: %s", step, exc)
            counters.parse_failures += 1
        return playbook, counters.freeze(epoch, step, None)

    async def process(
        self,
        playbook: playbook_lib.Playbook,
        sample: grading.Sample,
        grader: grading.Grader,
        epoch: int,
        step: int,
        total: int,
        pending: list[str],
        counters: Counters,
    ) -> tuple[playbook_lib.Playbook, StepStats]:
        """Runs Generator, optional Reflector and Curator for one sample."""
        config = self.config
        answer = await self.generator.run(self.runtime, playbook, sample)
        counters.record(answer.reply)
        correct: bool | None = None
        if config.use_ground_truth:
            correct = grader.is_correct(answer.final_answer, sample.target)
        if correct:
            playbook = tag(playbook, answer.bullet_ids, roles.Tag.HELPFUL)
        else:
            feedback = (
                grader.feedback(answer.final_answer, sample.target)
                if config.use_ground_truth
                else ""
            )
            playbook = await self.reflect(
                playbook, sample, answer, feedback, pending, counters
            )
        if step % config.curator_frequency == 0:
            playbook = await self.curate(
                playbook, sample, step, total, pending, counters
            )
        return playbook, counters.freeze(epoch, step, correct)

    async def reflect(
        self,
        playbook: playbook_lib.Playbook,
        sample: grading.Sample,
        answer: roles.GeneratorOutput,
        feedback: str,
        pending: list[str],
        counters: Counters,
    ) -> playbook_lib.Playbook:
        """Reflects until an insight emerges or rounds run out."""
        for attempt in range(self.config.max_reflector_rounds):
            note = feedback
            if attempt:
                note += RETRY_NOTE.format(attempt=attempt + 1)
            reflection = await self.reflector.run(
                self.runtime,
                playbook,
                sample,
                answer,
                note,
                self.config.use_ground_truth,
            )
            counters.record(reflection.reply)
            counters.reflector_rounds += 1
            for bullet_id, verdict in reflection.tags:
                playbook = tag(playbook, (bullet_id,), verdict)
            if reflection.insight:
                pending.append(reflection.insight)
                break
        return playbook

    async def curate(
        self,
        playbook: playbook_lib.Playbook,
        sample: grading.Sample,
        step: int,
        total: int,
        pending: list[str],
        counters: Counters,
    ) -> playbook_lib.Playbook:
        """Applies the Curator's operations and trims to budget."""
        reflection = "\n".join(pending) or NO_ERROR_REFLECTION
        output = await self.curator.run(
            self.runtime,
            playbook,
            sample,
            reflection,
            step,
            total,
            self.config.playbook_token_budget,
            self.config.use_ground_truth,
        )
        counters.record(output.reply)
        pending.clear()
        now = okf_model.now_iso()
        for op in output.ops:
            outcome = op.apply(playbook, now)
            playbook = outcome.playbook
            counters.added += outcome.added
            counters.removed += outcome.removed
        return playbook.trim(
            self.runtime.tokenizer, self.config.playbook_token_budget
        )


def tag(
    playbook: playbook_lib.Playbook,
    bullet_ids: Sequence[str],
    verdict: roles.Tag,
) -> playbook_lib.Playbook:
    """Increments helpful/harmful counters for the named bullets."""
    if verdict is roles.Tag.NEUTRAL:
        return playbook
    now = okf_model.now_iso()
    for bullet_id in dict.fromkeys(bullet_ids):
        bullet = playbook.get(bullet_id)
        if bullet is None:
            continue
        field = (
            "helpful_count" if verdict is roles.Tag.HELPFUL else "harmful_count"
        )
        playbook = playbook.replace(
            dataclasses.replace(
                bullet, **{field: getattr(bullet, field) + 1}, updated_at=now
            )
        )
    return playbook


@dataclasses.dataclass(slots=True)
class Counters:
    """Mutable accumulator turned into a frozen `StepStats`."""

    added: int = 0
    removed: int = 0
    reflector_rounds: int = 0
    parse_failures: int = 0
    backend_errors: int = 0
    cache_hits: int = 0
    usage: usage_lib.Usage = dataclasses.field(default_factory=usage_lib.Usage)

    def record(self, reply: roles.Reply) -> None:
        """Accounts for one LLM reply."""
        self.usage = self.usage + reply.usage
        self.cache_hits += int(reply.cached)

    def freeze(self, epoch: int, step: int, correct: bool | None) -> StepStats:
        """Returns the immutable statistics."""
        return StepStats(
            epoch=epoch,
            step=step,
            correct=correct,
            bullets_added=self.added,
            bullets_removed=self.removed,
            reflector_rounds=self.reflector_rounds,
            parse_failures=self.parse_failures,
            backend_errors=self.backend_errors,
            usage=self.usage,
            cache_hits=self.cache_hits,
        )
