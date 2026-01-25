// UI display data
export interface ResultsDisplay {
  image: string | undefined;
  classifications: Array<AnalysisBox> | null;
}

export interface AnalysisBox {
  box: number[];          // 4 coordinate points in pixels
  yolo_conf: number;      // yolo confidence for leaf detection
  efficientnet_class_name: string;  // efficientnet disease classification string
  efficientnet_confidence: number;   // efficientnet disease confidence
}