import uuid
import pika
import json
import base64
import tempfile
from inference.app.analysis.health_analyzer import health_analyzer
from inference.app.db.db import get_db
from inference.app.crud.analysis import analysis_crud
from inference.app.storage.storage_service import storage_service


def process_image(image_bytes: bytes,user_id:str):
    """Save image temporarily and run analyzer."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name

    # Analyze using  AI model
    results = health_analyzer.analyze(tmp_path)
    object_id = uuid.uuid4()

    presigned_url =  storage_service.upload_file(object_name=f"users\\{user_id}\\analysis\\{object_id}",source_path=tmp_path,extension=".jpg")
    #modify results to our needs and get real URL

    print("pre signed url")
    print(presigned_url)
    db_gen = get_db()
    db = next(db_gen)
    try:
        new_analysis = {
            "presigned_url": str(presigned_url),
            "description": str(results),
        }
        analysis_crud.create(db,new_analysis)
        print(f"Inserted analysis")

    except Exception as e:
        print(f"Error inserting analysis: {e}")
        db.close()

    finally:
        db.close()

    return results


def callback(ch, method, properties, body: bytes):

    print("Received task from RabbitMQ")
    data = json.loads(body)
    img_b64 = data["image"]
    user_id = data["user_id"]

    image_bytes = base64.b64decode(img_b64)

    # Run the detection
    process_image(image_bytes,user_id)

    print("Analysis complete:")
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


