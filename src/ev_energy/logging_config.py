"""JSON logging to stdout: one JSON object per log line."""
import logging
import sys

from pythonjsonlogger.json import JsonFormatter


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:  # don't add a second handler on re-import
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            JsonFormatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger
