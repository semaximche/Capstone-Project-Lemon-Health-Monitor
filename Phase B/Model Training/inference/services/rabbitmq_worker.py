import pika
import json
import base64
import tempfile
import os
from .health_analyzer import health_analyzer


def process_image(image_bytes: bytes):
    """Save image temporarily and run analyzer."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name

    # Analyze using  model
    results = health_analyzer.analyze(tmp_path)
    print("analysis results:")
    print(results)
    # Delete temp file
    os.remove(tmp_path)

    return results


def callback(ch, method, properties, body: bytes):
    print("Received task from RabbitMQ")

    # Body is JSON with base64 image
    data = json.loads(body)
    img_b64 = data["image_path"]

    # Convert back to bytes
    image_bytes = base64.b64decode(img_b64)

    # Run the detection
    results = process_image(image_bytes)

    print("Analysis complete:")
    print(results)

    ch.basic_ack(delivery_tag=method.delivery_tag)


def start_worker():
    """
    Connects to RabbitMQ, declares the queue, and starts consuming messages.
    """
    try:
        # Establish a blocking connection to the RabbitMQ broker
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(host="localhost")
        )
        channel = connection.channel()

        channel.queue_declare(queue="disease_jobs")

        print("Worker is waiting for tasks. To exit press CTRL+C")
        channel.basic_qos(prefetch_count=1)

        channel.basic_consume(queue="disease_jobs", on_message_callback=callback)

        # Start the main message loop
        channel.start_consuming()

    except pika.exceptions.AMQPConnectionError:
        print("\n[!!!] Failed to connect to RabbitMQ. Please ensure the server is running on localhost.")
    except KeyboardInterrupt:
        print("\nWorker stopped by user.")
        if 'connection' in locals() and connection.is_open:
            connection.close()


