import aio_pika
import json
from app.settings import settings

async def consume_notifications(on_message):

    connection = await aio_pika.connect_robust(
        host=settings.queue_host,
        login=settings.queue_user,
        password=settings.queue_password,
    )
    channel = await connection.channel()


    exchange = await channel.declare_exchange(
        settings.events_exchange,
        aio_pika.ExchangeType.TOPIC,
        durable=True
    )
    queue = await channel.declare_queue(
        settings.notifications_queue_name,
        durable=True
    )
    await queue.bind(exchange, routing_key="analysis.completed")
    print("Notification consumer is ready... waiting for messages")

    async with queue.iterator() as queue_iter:
        async for message in queue_iter:
            async with message.process():  # auto-acknowledges if no exception
                payload = json.loads(message.body)
                # Call your processing function
                await on_message(payload)
