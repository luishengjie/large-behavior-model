"""Utilities for configuring logging."""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path


def configure_logging(
    script_name: str,
    log_level: str = "INFO",
    *,
    log_dir: str | Path = "logs",
) -> Path:
    """Log to terminal and timestamped file."""

    normalized_level = log_level.upper()
    valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    if normalized_level not in valid_levels:
        raise ValueError(f"Unsupported log level: {log_level!r}.")

    safe_name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", script_name).strip("._")
    if not safe_name:
        raise ValueError("script_name must contain at least one valid character.")

    output_dir = Path(log_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_path = output_dir / f"{safe_name}_{timestamp}.log"
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    logging.basicConfig(
        level=getattr(logging, normalized_level),
        handlers=[console_handler, file_handler],
        force=True,
    )
    return log_path
