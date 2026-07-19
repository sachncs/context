"""Logging facade for the :mod:`ceng` package.

Importable as ``ceng.compress.log`` and as ``ceng.log`` via the
top-level re-export. Single shared logger so callers can do
``logging.getLogger("ceng")`` and integrate with their own handlers.
"""

from __future__ import annotations

import logging

CENG_LOGGER_NAME = "ceng"

logger = logging.getLogger(CENG_LOGGER_NAME)


def configure_logging(level: int = logging.WARNING) -> None:
    """Attach a single :class:`logging.StreamHandler` to the ceng logger.

    Idempotent: replaces any previously configured ceng handler so
    repeated calls don't multiply handlers. Disabled by default
    because ceng is a library; callers opt in.
    """
    logger.setLevel(level)
    for handler in list(logger.handlers):
        if getattr(handler, "_ceng_owned", False):
            logger.removeHandler(handler)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s ceng %(levelname)s %(message)s")
    )
    handler._ceng_owned = True  # type: ignore[attr-defined]
    logger.addHandler(handler)
    logger.propagate = False
