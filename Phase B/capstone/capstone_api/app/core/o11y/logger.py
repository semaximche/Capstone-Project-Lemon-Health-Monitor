import sys
from functools import partial
from typing import Any, TypedDict

from loguru import logger

from app.core.context.log_context import get_log_context


class HandlerConfig(TypedDict, total=False):
    """Type definition for loguru handler configuration."""

    sink: Any  # Can be str, Path, TextIO, or callable
    format: str
    serialize: bool
    backtrace: bool
    diagnose: bool
    catch: bool
    enqueue: bool
    colorize: bool


def patch_context(record, service_name: str, extra: dict[str, Any] | None = None):
    context = get_log_context()

    record_extra = record["extra"]
    if extra and isinstance(extra, dict):
        # This is first to make sure the extra is not override the mandatory fields
        record_extra.update(extra)

    record_extra["service_name"] = service_name
    if context and isinstance(context, dict):
        record_extra["context"] = context


def init_logger(
        service_name: str,
        serialize: bool = False,
        extra: dict[str, Any] | None = None
):
    patch_context_func = partial(patch_context, service_name=service_name, extra=extra)

    handler_config: HandlerConfig = {
        "sink": sys.stdout,
        "serialize": serialize,
        "backtrace": True,
        "diagnose": True,
        "catch": True,
        "enqueue": False,
        "colorize": not serialize,
    }
    log_format = "{message}" if serialize else "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level> | <level>{extra}</level>"
    if log_format is not None:
        handler_config["format"] = log_format
    logger.configure(
        handlers=[handler_config],
        patcher=patch_context_func,
    )
    logger.info("Initializing logger")
