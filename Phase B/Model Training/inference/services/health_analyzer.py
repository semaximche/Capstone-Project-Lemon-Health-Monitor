import torch
import torchvision.transforms as transforms
from ultralytics import YOLO
import cv2
import PIL.Image
import numpy as np  # Still needed for potential future use, though not directly used in analyze now
import base64  # Not needed, but kept for context consistency
import io  # Not needed, but kept for context consistency
from typing import List, Dict, Any


class LemonHealthAnalyzer:
    """
    A class to perform two-stage inference (Detection + Classification)
    on lemon leaf images using YOLO and EfficientNet models.
    """

    # Model and Inference Configuration
    CONFIDENCE_THRESHOLD = 0.3
    INPUT_SIZE = 224
    MEAN = [0.6368, 0.7232, 0.5855]
    STD = [0.3056, 0.2899, 0.3960]
    CLASS_NAMES = ['Anthracnose', 'Bacterial Blight', 'Citrus Canker', 'Curl Virus',
                   'Deficiency Leaf', 'Dry Leaf', 'Healthy Leaf', 'Sooty Mould', 'Spider Mites']

    # File Paths
    YOLO_WEIGHTS_PATH = 'inference/yolo100epochs.pt'
    EFFICIENTNET_WEIGHTS_PATH = 'inference/efficientnet50epochs.pth'

    def __init__(self):
        """Initializes the device, models, and image transforms."""

        # 1. Device Setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # 2. Transform Setup
        self.test_transform = transforms.Compose([
            transforms.Resize(self.INPUT_SIZE),
            transforms.CenterCrop(self.INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(self.MEAN, self.STD)
        ])

        # 3. Model Loading
        self.yolo_model = self._load_yolo_model()
        self.efficientnet_model = self._load_efficientnet_model()

    def _load_yolo_model(self) -> YOLO:
        """Loads the YOLO detection model."""
        try:
            return YOLO(self.YOLO_WEIGHTS_PATH)
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            raise

    def _load_efficientnet_model(self) -> torch.nn.Module:
        """Loads the EfficientNet classification model."""
        try:
            model = torch.load(
                self.EFFICIENTNET_WEIGHTS_PATH,
                weights_only=False,
                map_location=self.device
            )
            model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            print(f"Error loading EfficientNet model: {e}")
            raise

    def analyze(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Runs the two-stage analysis on a single input image path. 🔎

        :param image_path: Path to the image file (e.g., 'path/to/leaf.jpg').
        :return: A list of dictionaries containing detection boxes and
                 EfficientNet classification results for each detected leaf.
        """

        # --- Revert: Load Image from Path using OpenCV ---
        img = cv2.imread(image_path)

        if img is None:
            raise FileNotFoundError(f"Could not load image at path: {image_path}")

        # -----------------------------------------------

        # 1. Run YOLO Detection (YOLO can often take the path directly, but we use the loaded image here for consistency)
        print(f"Running YOLO detection on {image_path}...")

        # Note: Using the original image path here ensures YOLO's internal preprocessing is leveraged if needed.
        # However, using the loaded `img` array is also common. We'll use the path as in your original successful code:
        results = self.yolo_model.predict(image_path, conf=self.CONFIDENCE_THRESHOLD, verbose=False)

        final_results = []

        # 2. Iterate over detected boxes and run EfficientNet Classification
        print(f"Found {len(results[0].boxes)} objects. Running classification...")
        for i, box in enumerate(results[0].boxes.xyxy):
            # Get coordinates and crop the image using the `img` array loaded earlier
            x1, y1, x2, y2 = map(int, box[:4])
            cropped_image = img[y1:y2, x1:x2]

            # Convert OpenCV (BGR) crop to PIL (RGB) for torchvision transforms
            pilimg = PIL.Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

            # Apply transforms and prepare for model input
            readyimg = self.test_transform(pilimg)
            readyimg = readyimg.unsqueeze(0)  # Add batch dimension

            # Move image to device and run EfficientNet inference
            readyimg = readyimg.to(self.device)

            with torch.no_grad():
                self.efficientnet_model.eval()
                outputs = self.efficientnet_model(readyimg)

            # Post-process outputs
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            confidence, predicted_class_idx = torch.max(probabilities, 0)

            # Create a structured result entry
            result_entry = {
                "box": [x1, y1, x2, y2],
                "yolo_conf": float(results[0].boxes.conf[i]),
                "efficientnet_class_name": self.CLASS_NAMES[predicted_class_idx.item()],
                "efficientnet_confidence": confidence.item(),
            }
            final_results.append(result_entry)

        return final_results



health_analyzer = LemonHealthAnalyzer()