from google import genai
from inference.app.settings import settings
from typing import List, Dict


class PlantDiseaseReportGenerator:
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash",
    ):
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def _format_detections(self, detections: List[Dict]) -> str:
        """
        Convert detections into a readable markdown-style string for the prompt.
        """
        formatted = ""
        for d in detections:
            formatted += (
                f"\n- **{d['efficientnet_class_name']}** "
                f"- YOLO Confidence: {d['yolo_conf'] * 100:.2f}%, "
                f"EfficientNet Confidence: {d['efficientnet_confidence'] * 100:.2f}% "
                f"(Bounding Box: {d['box']})"
            )
        return formatted

    def _build_prompt(self, detections: List[Dict]) -> str:
        detection_results = self._format_detections(detections)

        return f"""
Hello! I'm here to help you understand the results of your citrus plant disease detection.

The analysis is based on:
- **YOLOv8** for object detection
- **EfficientNet** for disease classification

### Confidence Guidelines
- **High Confidence (Above 80%)** → Take immediate action (e.g., fungicides, pesticides)
- **Moderate Confidence (60%–80%)** → Monitor closely and treat if symptoms progress
- **Low Confidence (Below 60%)** → Further inspection recommended

### Detection Results
{detection_results}

Please summarize the findings clearly and provide practical recommendations per disease type.keep all the summary no more than 10 sentences
"""

    def generate_report(self, detections) -> str:
        """
        Generate a natural-language report based on model detections.
        """
        if not detections:
            return "No detections were provided. Unable to generate a report."

        prompt = self._build_prompt(detections)

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )

        return response.text


llm_generator = PlantDiseaseReportGenerator(
    api_key=settings.gemini_api_key
)

