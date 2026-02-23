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
        ...
        {detection_results}
        """

    def generate_report(self, detections) -> str:
        if not detections:
            return "No detections were provided. Unable to generate a report."

        prompt = self._build_prompt(detections)

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
            )

            return response.text

        except Exception as e:
            error_message = str(e).lower()

            # Handle token/quota exhaustion
            if "quota" in error_message or "resource exhausted" in error_message or "429" in error_message:
                return (
                    "The AI analysis service is temporarily unavailable due to "
                    "token or quota limits being reached. "
                    "Please try again later."
                )

            # Handle other unexpected errors gracefully
            return (
                "An unexpected error occurred while generating the analysis report. "
                "Please try again later."
            )


llm_generator = PlantDiseaseReportGenerator(
    api_key=settings.gemini_api_key
)