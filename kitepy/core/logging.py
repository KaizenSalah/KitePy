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


def get_logger(
    name: str = "kitepy",
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    json_file: Optional[str] = None,
) -> logging.Logger:
    """Return a rank-aware KitePy logger."""
    logger = logging.getLogger(name)
    resolved_level = (level or os.environ.get("KITEPY_LOG_LEVEL", "INFO")).upper()
    if resolved_level == "SILENT":
        logger.setLevel(logging.CRITICAL + 10)
    else:
        logger.setLevel(getattr(logging, resolved_level, logging.INFO))

    if not logger.handlers:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        stream_handler.addFilter(_RankZeroFilter())
        logger.addHandler(stream_handler)
        logger.propagate = False

    if log_file and _get_global_rank() == 0:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(fh)

    if json_file and _get_global_rank() == 0:
        Path(json_file).parent.mkdir(parents=True, exist_ok=True)
        jh = logging.FileHandler(json_file, encoding="utf-8")
        jh.setFormatter(_JsonFormatter())
        logger.addHandler(jh)

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
