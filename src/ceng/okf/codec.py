"""Persist contexts as OKF bundles.

Layout:
    index.md              type ceng/bundle-index: links every message
    report.md             type ceng/report: compression report as JSON
    messages/NNNN-role.md type ceng/message: one message per concept
"""

from __future__ import annotations

import json
import pathlib

from ceng import errors, persistence
from ceng import messages as messages_lib
from ceng.compression import report as report_lib
from ceng.okf import bundle as bundle_lib
from ceng.okf import model

BUNDLE_INDEX = "ceng/bundle-index"
MESSAGE = "ceng/message"
REPORT = "ceng/report"
FENCE = "```"


@persistence.Codec.registry.register("okf")
class OkfCodec(persistence.Codec):
    """Writes a snapshot as an OKF v0.1 bundle directory."""

    def write(self, snapshot: persistence.Snapshot, path: pathlib.Path) -> None:
        concepts = []
        links = []
        for position, message in enumerate(snapshot.messages):
            relative = f"messages/{position:04d}-{message.role.value}.md"
            extra: dict[str, object] = {
                "role": message.role.value,
                "position": position,
            }
            if message.name is not None:
                extra["name"] = message.name
            concepts.append(
                model.Concept(
                    model.Frontmatter(
                        type=MESSAGE,
                        title=f"{position}: {message.role.value}",
                        timestamp=model.now_iso(),
                        extra=extra,
                    ),
                    message.content,
                ).at(relative)
            )
            links.append(f"- [{position}: {message.role.value}]({relative})")
        if snapshot.report is not None:
            payload = json.dumps(
                snapshot.report.to_mapping(), indent=2, sort_keys=True
            )
            concepts.append(
                model.Concept(
                    model.Frontmatter(type=REPORT, title="Compression report"),
                    f"{FENCE}json\n{payload}\n{FENCE}",
                ).at("report.md")
            )
            links.append("- [Compression report](report.md)")
        extra_index: dict[str, object] = {
            "okf_version": model.OKF_VERSION,
            "message_count": len(snapshot.messages),
        }
        for key, value in snapshot.metadata.items():
            extra_index[f"meta_{key}"] = value
        concepts.append(
            model.Concept(
                model.Frontmatter(
                    type=BUNDLE_INDEX,
                    title="Context",
                    timestamp=model.now_iso(),
                    extra=extra_index,
                ),
                "\n".join(links),
            ).at(model.RESERVED_INDEX)
        )
        bundle_lib.Bundle.of(concepts).write(path)

    def read(self, path: pathlib.Path) -> persistence.Snapshot:
        bundle = bundle_lib.Bundle.read(path)
        index = bundle.find(model.RESERVED_INDEX)
        if index is None or index.frontmatter.type != BUNDLE_INDEX:
            raise errors.ValidationError(f"{path} is not a ceng context bundle")
        ordered = sorted(
            bundle.of_type(MESSAGE),
            key=lambda c: int(c.frontmatter.extra.get("position", 0)),
        )
        messages = []
        for concept in ordered:
            extra = concept.frontmatter.extra
            try:
                role = messages_lib.Role(extra["role"])
            except (KeyError, ValueError) as exc:
                raise errors.ValidationError(
                    f"{concept.path}: invalid role"
                ) from exc
            name = extra.get("name")
            messages.append(
                messages_lib.Message(
                    role, concept.body, name if isinstance(name, str) else None
                )
            )
        expected = index.frontmatter.extra.get("message_count")
        if expected is not None and expected != len(messages):
            raise errors.ValidationError(
                f"bundle lists {expected} messages but holds {len(messages)}"
            )
        report = None
        stored = bundle.find("report.md")
        if stored is not None:
            report = report_lib.CompressionReport.from_mapping(
                self.extract_json(stored.body)
            )
        metadata = {
            key.removeprefix("meta_"): str(value)
            for key, value in index.frontmatter.extra.items()
            if key.startswith("meta_")
        }
        return persistence.Snapshot(tuple(messages), report, metadata)

    @staticmethod
    def extract_json(body: str) -> dict[str, object]:
        """Returns the JSON object inside the first fenced block of `body`."""
        start = body.find(f"{FENCE}json\n")
        end = body.rfind(FENCE)
        if start == -1 or end <= start:
            raise errors.ValidationError("report has no JSON block")
        try:
            loaded = json.loads(body[start + len(FENCE) + 5 : end])
        except ValueError as exc:
            raise errors.ValidationError(f"invalid report JSON: {exc}") from exc
        if not isinstance(loaded, dict):
            raise errors.ValidationError("report JSON must be an object")
        return loaded
