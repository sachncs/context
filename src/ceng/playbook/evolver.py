"""The Evolver: Generator -> Reflector -> Curator loop.

Mirrors ``ace.ACE.run(mode='offline')`` from the upstream repo. Three
LLM calls per training sample: generate a candidate, judge it
against the answer, add the lesson back into the playbook.

Defaults match the paper's recommended hyperparameter sweep:

- ``max_reflector_rounds`` = 5      (paper §A.6 best)
- ``dedup_threshold`` = 0.90         (paper §A.6 best)
- ``playbook_token_budget`` = 80,000 (paper Appendix F default)
- ``curator_frequency`` = 1          (paper setting — run curator every step)

The Reflector and Curator are LLM-backed; the deterministic merge
happens locally via :meth:`Playbook.merge`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from ceng.cache import NAMESPACE_SUMMARIZE, Cache, make_key
from ceng.playbook import Bullet, Playbook, empty_playbook, render_playbook
from ceng.playbook.prompts import (
    build_curator_messages,
    build_generator_messages,
    build_reflector_messages,
)


@dataclass
class EvolverConfig:
    """Hyperparameters for :meth:`Evolver.run`."""

    max_reflector_rounds: int = 5
    dedup_threshold: float = 0.90
    playbook_token_budget: int = 80_000
    curator_frequency: int = 1
    use_ground_truth: bool = True


@dataclass
class EvolverStepStats:
    bullets_added: int = 0
    bullets_dropped: int = 0
    reflector_rounds_used: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    backend_call_count: int = 0
    last_reflection_excerpt: str = ""
    last_curated_excerpt: str = ""


@dataclass
class Evolver:
    """Run the ACE Generator->Reflector->Curator loop over ``queries``.

    Args:
        backend: A :class:`ceng.backends.Backend`. Defaults to the
            active backend.
        llm: Model id used for every Generator / Reflector / Curator call.
        cache_dir: Path for caching summarisation calls. Empty
            string disables caching.
        config: :class:`EvolverConfig` with the paper's recommended
            defaults.
    """

    backend: Any = None
    llm: str = "gpt-4o-mini"
    cache_dir: str = ".ceng/cache"
    config: EvolverConfig = field(default_factory=EvolverConfig)

    def __post_init__(self) -> None:
        if self.backend is None:
            from ceng.backends import get_backend as resolve_backend

            self.backend = resolve_backend()

    def run(
        self,
        playbook: Playbook | None,
        queries: Iterable[dict],
        evaluator: Callable[[str, str, dict], str],
        max_iterations: int = 5,
    ) -> tuple[Playbook, list[EvolverStepStats]]:
        """Run the loop for ``max_iterations`` rounds.

        Args:
            playbook: starting :class:`Playbook`. ``None`` seeds with
                the seven default ACE sections (empty).
            queries: iterable of ``{"question": ..., "context": ...,
                "ground_truth": ...}`` dicts. ``ground_truth`` is
                optional when ``config.use_ground_truth=False``.
            evaluator: callable ``(question, model_answer, sample) ->
                env_feedback_str``. Compares the model's answer to the
                sample's ground truth (or just inspects the answer
                when GT isn't available) and returns a short string
                describing the gap. Used as Reflector's
                ``environment_feedback``.
            max_iterations: number of ``for query in queries`` passes
                over the dataset. Paper §A.6 finds 1-5 epochs are
                sufficient.

        Returns:
            (evolved ``Playbook``, list of per-step stats). The
            ``stats[-1]`` element summarises the final iteration.
        """
        pb = playbook if playbook is not None else empty_playbook()
        queries = list(queries)
        cache = Cache(cache_dir=self.cache_dir) if self.cache_dir else None
        all_stats: list[EvolverStepStats] = []
        for iteration in range(max_iterations):
            for step, sample in enumerate(queries, start=1):
                stats = self._one_step(
                    pb,
                    sample,
                    step=step,
                    total_steps=len(queries),
                    iteration=iteration,
                    evaluator=evaluator,
                    cache=cache,
                )
                all_stats.append(stats)
                if step % self.config.curator_frequency == 0:
                    pb.trim_to_token_budget(
                        budget=self.config.playbook_token_budget
                    )
        return pb, all_stats

    def _one_step(
        self,
        pb: Playbook,
        sample: dict,
        *,
        step: int,
        total_steps: int,
        iteration: int,
        evaluator,
        cache: Cache | None,
    ) -> EvolverStepStats:
        question = sample["question"]
        context = sample.get("context", "")
        ground_truth = sample.get("ground_truth")
        env_feedback = ""
        bullets_added_this_step = 0
        bullets_dropped = 0
        generator_msgs = build_generator_messages(
            render_playbook(pb),
            reflection="(empty)",
            question=question,
            context=context,
        )
        gen_text, gen_cache_hit = self._call_with_cache(
            namespace=NAMESPACE_SUMMARIZE,
            key=make_key(
                {
                    "op": "playbook_generate",
                    "model": self.llm,
                    "playbook_sha256": _sha256(render_playbook(pb)),
                    "question_sha256": _sha256(question),
                    "iteration": iteration,
                    "step": step,
                }
            ),
            messages=generator_msgs,
            cache=cache,
        )
        parsed_gen = _extract_json(gen_text, {"reasoning": "", "bullet_ids": [], "final_answer": ""})
        bullets_used_ids = parsed_gen.get("bullet_ids") or []
        answer = parsed_gen.get("final_answer", "") or ""

        if self.config.use_ground_truth and ground_truth is not None:
            env_feedback = evaluator(question, answer, sample)

        stats = EvolverStepStats()
        stats.backend_call_count = 1
        if gen_cache_hit:
            stats.cache_hits += 1
        else:
            stats.cache_misses += 1

        rounds_used = 0
        reflection_text = ""
        # Reflector runs only on FAILED samples — the paper finds
        # reflection on successful answers contributes nothing useful
        # because there is no error to introspect. We still update
        # bullet counts (helpful++) if the Reflector tags a bullet the
        # Generator used and the answer was correct.
        if self.config.use_ground_truth and ground_truth is not None and answer != ground_truth:
            current_answer = answer
            current_feedback = env_feedback or f"model answer was {answer!r}; expected {ground_truth!r}"
            for r in range(self.config.max_reflector_rounds):
                rounds_used += 1
                stats.backend_call_count += 1
                reflector_msgs = build_reflector_messages(
                    question=question,
                    reasoning_trace=parsed_gen.get("reasoning", ""),
                    predicted_answer=current_answer,
                    ground_truth=ground_truth,
                    environment_feedback=current_feedback,
                    bullets_used=_render_used_bullets(pb, bullets_used_ids),
                    use_ground_truth=True,
                )
                ref_text, ref_hit = self._call_with_cache(
                    namespace=NAMESPACE_SUMMARIZE,
                    key=make_key(
                        {
                            "op": "playbook_reflect",
                            "model": self.llm,
                            "iter": iteration,
                            "step": step,
                            "round": r,
                        }
                    ),
                    messages=reflector_msgs,
                    cache=cache,
                )
                if ref_hit:
                    stats.cache_hits += 1
                else:
                    stats.cache_misses += 1
                parsed_ref = _extract_json(
                    ref_text,
                    {
                        "reasoning": "",
                        "error_identification": "",
                        "root_cause_analysis": "",
                        "correct_approach": "",
                        "key_insight": "",
                        "bullet_tags": [],
                    },
                )
                reflection_text = (
                    parsed_ref.get("key_insight")
                    or parsed_ref.get("error_identification")
                    or ""
                ).strip()
                stats.last_reflection_excerpt = reflection_text[:200]

                bullet_tags = parsed_ref.get("bullet_tags") or []
                for tag in bullet_tags:
                    bid = tag.get("id") or tag.get("bullet")
                    if not bid:
                        continue
                    bullet = pb.bullets.get(bid)
                    if bullet is None:
                        continue
                    t = (tag.get("tag") or "neutral").lower()
                    if t == "helpful":
                        pb.bullets[bid] = Bullet(
                            id=bullet.id,
                            section=bullet.section,
                            content=bullet.content,
                            helpful_count=bullet.helpful_count + 1,
                            harmful_count=bullet.harmful_count,
                            updated_at=_now_iso(),
                        )
                    elif t == "harmful":
                        pb.bullets[bid] = Bullet(
                            id=bullet.id,
                            section=bullet.section,
                            content=bullet.content,
                            helpful_count=bullet.helpful_count,
                            harmful_count=bullet.harmful_count + 1,
                            updated_at=_now_iso(),
                        )

                if reflection_text and current_feedback:
                    break

        stats.reflector_rounds_used = rounds_used

        # Curator runs every step (not gated on whether the model was
        # wrong) per upstream ACE. On a correct sample we still pass
        # the empty reflection and let the Curator emit nothing — its
        # MERGE step still tidies the playbook. This matches the
        # paper's ``curator_frequency=1`` default.
        stats.backend_call_count += 1
        curator_reflection = reflection_text or "(model produced correct answer; no error to reflect on)"
        curator_msgs = build_curator_messages(
            token_budget=self.config.playbook_token_budget,
            current_step=step,
            total_samples=total_steps,
            playbook_stats=_render_playbook_stats(pb),
            recent_reflection=curator_reflection,
            current_playbook=render_playbook(pb),
            question_context=context or question,
            use_ground_truth=self.config.use_ground_truth,
        )
        cur_text, cur_hit = self._call_with_cache(
            namespace=NAMESPACE_SUMMARIZE,
            key=make_key(
                {
                    "op": "playbook_curate",
                    "model": self.llm,
                    "iter": iteration,
                    "step": step,
                    "reflection_sha256": _sha256(curator_reflection),
                }
            ),
            messages=curator_msgs,
            cache=cache,
        )
        if cur_hit:
            stats.cache_hits += 1
        else:
            stats.cache_misses += 1
        parsed_cur = _extract_json(
            cur_text, {"reasoning": "", "operations": []}
        )
        bullets_added_this_step, bullets_dropped_cur, excerpt = (
            _apply_curator_operations(pb, parsed_cur.get("operations") or [])
        )
        stats.last_curated_excerpt = excerpt
        bullets_dropped += bullets_dropped_cur

        stats.bullets_added = bullets_added_this_step
        stats.bullets_dropped = bullets_dropped
        return stats

    def _call_with_cache(
        self,
        *,
        namespace: str,
        key: str,
        messages: list[dict],
        cache: Cache | None,
    ) -> tuple[str, bool]:
        if cache is not None:
            cached = cache.get(namespace, key)
            if cached is not None:
                return str(cached["text"]), True
        kw = {"temperature": 0.0, "max_tokens": 4096}
        text = self.backend.complete(messages=messages, model=self.llm, **kw)
        if cache is not None:
            cache.set(namespace, key, {"text": text})
        return text, False


def _extract_json(text: str, fallback: dict[str, Any]) -> dict[str, Any]:
    """Try to parse ``text`` as JSON; fall back to ``fallback`` on failure."""
    if not text:
        return dict(fallback)
    try:
        loaded = json.loads(text)
        if isinstance(loaded, dict):
            return loaded
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            loaded = json.loads(m.group(0))
            if isinstance(loaded, dict):
                return loaded
        except json.JSONDecodeError:
            pass
    return dict(fallback)


def _apply_curator_operations(
    playbook: Playbook, ops: list[dict]
) -> tuple[int, int, str]:
    """Apply the Curator's ``operations`` list to ``playbook`` in place.

    Supports all four ACE operations:

    * ``ADD``: add a new bullet (delegates to :meth:`Playbook.merge` so
      dedup still applies).
    * ``UPDATE``: mutate an existing bullet's content by id.
    * ``MERGE``: collapse two near-duplicate bullets — the higher
      ``net_score`` wins; the lower is deleted.
    * ``DELETE``: remove a bullet by id (e.g. retract a harmful one).

    Returns ``(added, dropped, excerpt)`` where ``excerpt`` is a
    short ``; ``-joined summary of the first few operations for
    stats display.
    """
    added = 0
    dropped = 0
    excerpt_bits: list[str] = []
    now = _now_iso()
    for i, op in enumerate(ops):
        if not isinstance(op, dict):
            continue
        kind = (op.get("type") or "").upper()
        if kind == "ADD":
            section = (op.get("section") or "others").strip().lower()
            content = (op.get("content") or "").strip()
            if not content:
                continue
            bullet = Bullet(
                id=f"cng-{i:05d}",
                section=section,
                content=content,
                created_at=now,
                updated_at=now,
            )
            before = len(playbook.bullets)
            playbook.merge([bullet])
            if len(playbook.bullets) > before:
                added += 1
            excerpt_bits.append(f"ADD:{section}:{bullet.id}")
        elif kind == "UPDATE":
            target_id = op.get("id") or op.get("bullet_id")
            content = (op.get("content") or "").strip()
            if not target_id or not content:
                continue
            existing = playbook.bullets.get(target_id)
            if existing is None:
                continue
            playbook.bullets[target_id] = Bullet(
                id=existing.id,
                section=existing.section,
                content=content,
                helpful_count=existing.helpful_count,
                harmful_count=existing.harmful_count,
                created_at=existing.created_at,
                updated_at=now,
            )
            excerpt_bits.append(f"UPDATE:{target_id}")
        elif kind == "MERGE":
            keep_id = op.get("keep_id") or op.get("id")
            drop_id = op.get("drop_id") or op.get("other_id")
            if not keep_id or not drop_id or keep_id == drop_id:
                continue
            keep = playbook.bullets.get(keep_id)
            drop = playbook.bullets.get(drop_id)
            if keep is None or drop is None:
                continue
            winner = keep if keep.net_score >= drop.net_score else drop
            playbook.bullets[winner.id] = Bullet(
                id=winner.id,
                section=winner.section,
                content=winner.content,
                helpful_count=winner.helpful_count + drop.helpful_count,
                harmful_count=winner.harmful_count + drop.harmful_count,
                created_at=winner.created_at,
                updated_at=now,
            )
            if drop.id != winner.id:
                del playbook.bullets[drop.id]
                dropped += 1
            excerpt_bits.append(f"MERGE:{keep_id}<-{drop_id}")
        elif kind == "DELETE":
            target_id = op.get("id") or op.get("bullet_id")
            if not target_id:
                continue
            if target_id in playbook.bullets:
                del playbook.bullets[target_id]
                dropped += 1
                excerpt_bits.append(f"DELETE:{target_id}")
    return added, dropped, "; ".join(excerpt_bits[:3])


def _render_used_bullets(playbook: Playbook, bullet_ids: list[str]) -> str:
    """Render only the bullets the generator referenced, for the Reflector."""
    lines: list[str] = []
    for bid in bullet_ids:
        b = playbook.bullets.get(bid)
        if b:
            lines.append(
                f"[{b.id}] helpful={b.helpful_count} harmful={b.harmful_count} :: {b.content}"
            )
    return "\n".join(lines) if lines else "(generator referenced no bullets)"


def _render_playbook_stats(playbook: Playbook) -> str:
    by_section: dict[str, int] = {s: 0 for s in playbook.sections_in_order}
    for b in playbook.bullets.values():
        by_section.setdefault(b.section, 0)
        by_section[b.section] += 1
    parts = [f"total={len(playbook.bullets)}"]
    for s, c in by_section.items():
        parts.append(f"{s}={c}")
    return ", ".join(parts)


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _sha256(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()
