from ultralytics import YOLO

DATASET_PATH='input/yolo_dataset_leafdetection/data.yaml'

TRAINING_DEVICE='cuda'
TRAINING_EPOCHS=350
TRAINING_WORKERS=3

def main():
    model = YOLO('yolo11n.pt')

    model.train(data=DATASET_PATH,
                epochs=TRAINING_EPOCHS,
                imgsz=1024,
                device=TRAINING_DEVICE,
                batch=8,
                workers=TRAINING_WORKERS,
                )

if __name__ == "__main__":
    main()
