import pika
import json
from app.settings import settings


class RabbitMQPublisher:

    def __init__(self, host=settings.queue_host, queue_name=settings.queue_name):
        """
        Initializes the publisher with connection parameters.
        """
        self.host = host
        self.queue_name = queue_name


    def publish_job(self,job):
        credentials = pika.PlainCredentials(settings.queue_user, settings.queue_password)
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(self.host, credentials=credentials)
        )

        channel = connection.channel()
        channel.queue_declare(queue=self.queue_name)
        channel.basic_publish(
            exchange='',
            routing_key=self.queue_name,
            body=json.dumps(job)
        )
        connection.close()

publisher = RabbitMQPublisher()
