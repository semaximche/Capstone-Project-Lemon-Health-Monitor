from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Optional, cast

LOG_CONTEXT: ContextVar[Optional[dict]] = cast(
    ContextVar[Optional[dict]], ContextVar("LOG_CONTEXT", default=None)
)


def get_log_context() -> dict | None:
    return LOG_CONTEXT.get()


def set_log_context(context: dict | None):
    return LOG_CONTEXT.set(context)


def reset_log_context(token: Token[dict | None]) -> None:
    LOG_CONTEXT.reset(token)


def update_log_context(**kwargs) -> None:
    context = get_log_context()
    if not context:
        context = {}
    context.update(kwargs)
    set_log_context(context)


@contextmanager
def global_log_context(context: dict):
    token = set_log_context(context=context)
    context_var = get_log_context()
    try:
        yield context_var
    finally:
        reset_log_context(token=token)
