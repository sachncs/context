"""Persistence codecs for contexts."""

from __future__ import annotations

import abc
import dataclasses
import json
import pathlib
from collections.abc import Mapping

from ceng import errors
from ceng import messages as messages_lib
from ceng.compression import report as report_lib
from ceng.internals import registry


@dataclasses.dataclass(frozen=True, slots=True)
class Snapshot:
    """The persistable parts of a `Context` (everything except services).

    Attributes:
        messages: Conversation messages.
        report: Compression provenance, if any.
        metadata: String annotations.
    """

    messages: tuple[messages_lib.Message, ...]
    report: report_lib.CompressionReport | None = None
    metadata: Mapping[str, str] = dataclasses.field(default_factory=dict)


class Codec(abc.ABC):
    """Reads and writes snapshots in one on-disk format."""

    registry: registry.Registry[type[Codec]] = registry.Registry("format")

    @abc.abstractmethod
    def write(self, snapshot: Snapshot, path: pathlib.Path) -> None:
        """Writes `snapshot` to `path`, replacing previous content.

        Raises:
            ValidationError: If the snapshot cannot be represented.
        """

    @abc.abstractmethod
    def read(self, path: pathlib.Path) -> Snapshot:
        """Reads a snapshot from `path`.

        Raises:
            ValidationError: If the content is malformed.
        """


@Codec.registry.register("json")
class JsonCodec(Codec):
    """Single JSON file: `{"messages": [...], "report": {...}, ...}`."""

    def write(self, snapshot: Snapshot, path: pathlib.Path) -> None:
        document = {
            "messages": [m.to_mapping() for m in snapshot.messages],
            "report": snapshot.report.to_mapping() if snapshot.report else None,
            "metadata": dict(snapshot.metadata),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(
            json.dumps(document, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary.replace(path)

    def read(self, path: pathlib.Path) -> Snapshot:
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
            messages = tuple(
                messages_lib.Message.from_mapping(m)
                for m in document["messages"]
            )
            raw_report = document.get("report")
            report = (
                report_lib.CompressionReport.from_mapping(raw_report)
                if raw_report
                else None
            )
            metadata = {
                str(k): str(v) for k, v in document.get("metadata", {}).items()
            }
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise errors.ValidationError(f"cannot read {path}: {exc}") from exc
        return Snapshot(messages, report, metadata)
