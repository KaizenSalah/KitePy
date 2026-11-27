"""
Structured, rank-aware logging for KitePy.

Supports:
- Distributed rank-0 filtering (prevents duplicate logs across DDP/FSDP workers)
- Configurable log levels via set_log_level() or KITEPY_LOG_LEVEL env var
- Optional JSON-lines structured log file output for institutional/enterprise pipelines
"""

from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional


def _get_global_rank() -> int:
    for env_var in ("RANK", "ACCELERATE_PROCESS_INDEX", "SLURM_PROCID"):
        val = os.environ.get(env_var)
        if val is not None:
            try:
                return int(val)
            except ValueError:
                pass
    return 0


class _RankZeroFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return _get_global_rank() == 0


class _ConsoleRankZeroFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return _get_global_rank() == 0 and not getattr(record, "structured_only", False)


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra_metrics = getattr(record, "metrics", None)
        if isinstance(extra_metrics, dict):
            payload["metrics"] = extra_metrics
        return json.dumps(payload, default=str)


def _attach_file_handler(logger: logging.Logger, filepath: str, formatter: logging.Formatter):
    resolved = str(Path(filepath).resolve())
    for existing in logger.handlers:
        if isinstance(existing, logging.FileHandler) and getattr(existing, "baseFilename", None) == resolved:
            return
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(filepath, encoding="utf-8")
    handler.setFormatter(formatter)
    handler.addFilter(_RankZeroFilter())
    logger.addHandler(handler)


def get_logger(
    name: str = "kitepy",
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    json_file: Optional[str] = None,
) -> logging.Logger:
    """Return a rank-aware KitePy logger."""
    root_logger = logging.getLogger("kitepy")
    if not root_logger.handlers:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        stream_handler.addFilter(_ConsoleRankZeroFilter())
        root_logger.addHandler(stream_handler)
        root_logger.propagate = False

    logger = logging.getLogger(name)
    resolved_level = str(level or os.environ.get("KITEPY_LOG_LEVEL") or "INFO").upper()
    target_level = logging.CRITICAL + 10 if resolved_level == "SILENT" else getattr(logging, resolved_level, logging.INFO)
    logger.setLevel(target_level)
    if name == "kitepy" or name.startswith("kitepy."):
        root_logger.setLevel(target_level)
        if name != "kitepy":
            logger.propagate = True
    elif not logger.handlers:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        stream_handler.addFilter(_RankZeroFilter())
        logger.addHandler(stream_handler)
        logger.propagate = False

    # Route file/JSON handlers through parent "kitepy" logger when under "kitepy.*"
    # so planner, engine, and guard events all land in the same structured log.
    target_handler_logger = root_logger if name.startswith("kitepy.") else logger

    if log_file and _get_global_rank() == 0:
        _attach_file_handler(
            target_handler_logger,
            log_file,
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"),
        )

    if json_file and _get_global_rank() == 0:
        _attach_file_handler(target_handler_logger, json_file, _JsonFormatter())

    return logger


def set_log_level(level: str):
    """Set global KitePy log level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'SILENT')."""
    logger = get_logger("kitepy", level=level)
    lvl = level.upper()
    if lvl == "SILENT":
        logger.setLevel(logging.CRITICAL + 10)
    else:
        logger.setLevel(getattr(logging, lvl, logging.INFO))


__all__ = ["get_logger", "set_log_level"]
