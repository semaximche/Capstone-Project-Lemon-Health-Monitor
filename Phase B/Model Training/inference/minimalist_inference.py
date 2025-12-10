from ultralytics import YOLO
from services.rabbitmq_worker import start_worker


def main():
    start_worker()


if __name__ == '__main__':
    main()