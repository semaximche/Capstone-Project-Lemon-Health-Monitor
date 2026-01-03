import pika
import json
from inference.app.settings import settings


class RabbitMQPublisher:
    def __init__(self, host=settings.queue_host):
        self.host = host
        self.credentials = pika.PlainCredentials(
            settings.queue_user,
            settings.queue_password
        )

    def publish_event(self, routing_key: str, payload):
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                host=self.host,
                credentials=self.credentials
            )
        )

        channel = connection.channel()

        # Declare exchange ONCE (safe to call multiple times)
        channel.exchange_declare(
            exchange=settings.events_exchange,
            exchange_type="topic",
            durable=True
        )

        channel.basic_publish(
            exchange=settings.events_exchange,
            routing_key=routing_key,
            body=json.dumps(payload),
            properties=pika.BasicProperties(
                delivery_mode=2  # make message persistent
            )
        )

        connection.close()


publisher = RabbitMQPublisher()
