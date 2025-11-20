import os
from ultralytics import YOLO

def main():
    model = YOLO('yolo11n.pt')

    model.train(data='./leaf_detection_dataset/yolo_dataset_leafdetection/data.yaml',
                epochs=10,
                imgsz=1024,
                device='cuda',
                batch=-1,
                workers=3,
                )

if __name__ == "__main__":
    main()
