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
We are using visual model outputs on lemon tree images to detect and classify diseases. We want to give our users conservative analysis based on the model outputs to help them better treat and manage their lemon trees and lemon tree orchards.
The analysis results are based on images taken from a camera, the images are taken through a YOLOv11 model to designate boxes of leaves, these boxes are taken through an EfficientNetV2 classifier to guess and detect disease and symptoms that the leaf is experiencing. In the results each element in the array describes one leaf detected.

### Output guidelines
 - Base your answers on percentages of leaves experiencing a certain symptom or disease.
 - If 75% or more of the leaves are classified as healthy it is fair to assume the plant is largely healthy too.
 - If you see an overwhelming minority classification of one disease and symptom at around 30% then it is fair to assume the model is confident and that the tree is most likely experiencing these. You should give guidelines specific to the diagnosis on how to treat the plant and monitor for further inspections.
 - If you see many different sporadic classifications of various diseases such as Canker Citrus, Anthracnose, Souty Mould and such it is fair to assume the model is not confident in one disease and therefor you should not advise to treat any one disease.

### Analysis Results
{detection_results}

Summaries the results findings and offer recommendations to the user on how to continue treating their lemon tree including guidelines to further monitor the plant health. Keep your analysis brief, no more than 4 paragraphs and avoid mentioning your guidelines instead focusing on natural language in your response.
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

