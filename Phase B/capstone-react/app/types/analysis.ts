// UI display data
interface ResultsDisplay {
  image: string;
  classifications: Array<AnalysisBox>;
}

interface AnalysisBox {
  box: number[];          // 4 coordinate points in pixels
  yolo_conf: number;      // yolo confidence for leaf detection
  disease_class: string;  // efficientnet disease classification string
  disease_conf: number;   // efficientnet disease confidence
}