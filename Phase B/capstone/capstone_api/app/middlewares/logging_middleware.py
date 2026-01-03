from time import perf_counter
from fastapi import Request
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware
from app.core.context.log_context import update_log_context

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):

        if request.scope["type"] == "websocket":
            return await call_next(request)


        start_time = perf_counter()

        request_headers = dict(request.headers)
        for key in ["authorization", "cookie", "cookies", "x-api-key"]:
            request_headers.pop(key, None)


        context = {
            "app": {
                "title": request.app.title,
                "version": request.app.version,
            },
            "request": {
                "url": request.url,
                "path": request.url.path,
                "method": request.method,
                "headers": request_headers,
                "host": request.client.host if request.client else None,
                "port": request.client.port if request.client else None,
            },
        }
        update_log_context(**context)


        logger.info("Incoming request")
        response = await call_next(request)

        process_time = perf_counter() - start_time
        response.headers["process-time"] = str(round(process_time, 3))
        logger.info(
            "Outgoing response",
            response={
                "status_code": response.status_code,
                "headers": dict(response.headers),
            },
            process_time=process_time,
        )

        return response
