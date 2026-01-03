import uuid
import pika
import json
import base64
import tempfile
from inference.app.analysis.health_analyzer import health_analyzer
from inference.app.db.db import get_db
from inference.app.crud.analysis import analysis_crud
from inference.app.storage.storage_service import storage_service
from inference.app.settings import settings
from inference.app.llm_model.gemini_model import llm_generator
from inference.app.services.publisher import publisher


def process_image(image_bytes: bytes, user_id: str):
    """Save image temporarily, run analyzer, upload, and save to DB."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name

    # Analyze using AI model
    results = health_analyzer.analyze(tmp_path)
    llm_summary = llm_generator.generate_report(results)

    print("LLM summary:", llm_summary)

    object_id = uuid.uuid4()
    object_name = f"users/{user_id}/analysis/{object_id}.jpg"
    presigned_url = storage_service.upload_file(
        object_name=object_name, source_path=tmp_path, extension=".jpg"
    )

    print("Presigned URL:", presigned_url)

    # Save to DB
    db_gen = get_db()
    db = next(db_gen)
    try:
        new_analysis = {
            "user_id": user_id,
            "presigned_url": str(presigned_url),
            "description": str(results),
            "summary": str(llm_summary),
        }
        analysis_id = analysis_crud.create(db, new_analysis)
        print(f"Inserted analysis with ID {analysis_id}")
        return analysis_id
    except Exception as e:
        print(f"Error inserting analysis: {e}")
    finally:
        db.close()


def callback(ch, method, properties, body: bytes):
    print("Received task from RabbitMQ")

    try:
        data = json.loads(body)
        if data.get("type") != "analysis.requested":
            print("Skipping invalid message:", data)
            ch.basic_ack(delivery_tag=method.delivery_tag)
            return

        img_b64 = data["image"]
        user_id = data["user_id"]
        image_bytes = base64.b64decode(img_b64)

        # Process image and save analysis
        analysis_id = process_image(image_bytes, user_id)

        # Publish "analysis.completed" event
        completed_event = {
            "type": "analysis.completed",
            "user_id": user_id,
            "analysis_id": str(analysis_id),
        }
        publisher.publish_event(
            routing_key="analysis.completed", payload=completed_event
        )
        print(f"Published analysis.completed for analysis ID {analysis_id}")

        ch.basic_ack(delivery_tag=method.delivery_tag)
        print("Analysis complete")

    except Exception as e:
        print("Error processing message:", e)
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)


def start_worker():
    """Connects to RabbitMQ, declares queues, and starts consuming messages."""
    try:
        credentials = pika.PlainCredentials(settings.queue_user, settings.queue_password)
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=settings.queue_host, credentials=credentials)
        )
        channel = connection.channel()

        # Declare exchange and queue
        channel.exchange_declare(
            exchange=settings.events_exchange, exchange_type="topic", durable=True
        )
        channel.queue_declare(queue=settings.analysis_queue_name, durable=True)
        channel.queue_bind(
            exchange=settings.events_exchange,
            queue=settings.analysis_queue_name,
            routing_key="analysis.requested",
        )

        print("Worker waiting for tasks...")
        channel.basic_qos(prefetch_count=1)
        channel.basic_consume(
            queue=settings.analysis_queue_name, on_message_callback=callback
        )
        channel.start_consuming()

    except pika.exceptions.AMQPConnectionError:
        print(f"[!!!] Failed to connect to RabbitMQ at {settings.queue_host}.")
    except KeyboardInterrupt:
        print("\nWorker stopped by user.")
        if 'connection' in locals() and connection.is_open:
            connection.close()
