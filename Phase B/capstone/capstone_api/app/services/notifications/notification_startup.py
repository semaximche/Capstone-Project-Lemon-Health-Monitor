from fastapi import FastAPI
from contextlib import asynccontextmanager
import asyncio
from app.services.notifications.notification_consumer import consume_notifications
from app.services.notifications.notification_service import handle_notification

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.
    Starts the notification consumer on startup,
    and cancels it on shutdown.
    """
    # Start background consumer task
    consumer_task = asyncio.create_task(
        consume_notifications(handle_notification)
    )
    print("Notification consumer started in background...")

    try:
        yield
    finally:
        # Cancel background task on shutdown
        consumer_task.cancel()
        print("Shutting down notification consumer...")
