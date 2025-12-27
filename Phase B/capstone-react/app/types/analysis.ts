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

/**
 * Sample:
 * [{'box': [447, 290, 611, 395],
 * 'yolo_conf': 0.8707272410392761,
 * 'efficientnet_class_name': 'Curl Virus',
 * 'efficientnet_confidence': 0.5073761940002441},
 * 
 * {'box': [187, 91, 251, 187],
 * 'yolo_conf': 0.8486918807029724,
 * 'efficientnet_class_name': 'Anthracnose',
 * 'efficientnet_confidence': 0.9901493787765503},
 * 
 * {'box': [349, 69, 433, 160],
 * 'yolo_conf': 0.7968596816062927,
 * 'efficientnet_class_name': 'Citrus Canker',
 * 'efficientnet_confidence': 0.9999996423721313},
 * 
 * {'box': [0, 236, 95, 402],
 * 'yolo_conf': 0.7298312783241272,
 * 'efficientnet_class_name': 'Anthracnose',
 * 'efficientnet_confidence': 0.8804681897163391},
 * 
 * {'box': [242, 98, 294, 214],
 * 'yolo_conf': 0.7213596105575562,
 * 'efficientnet_class_name': 'Citrus Canker',
 * 'efficientnet_confidence': 0.9317632913589478},
 * 
 * {'box': [408, 206, 578, 298],
 * 'yolo_conf': 0.6756537556648254,
 * 'efficientnet_class_name': 'Healthy Leaf',
 * 'efficientnet_confidence': 0.4842284023761749},
 * 
 * {'box': [152, 208, 265, 358],
 * 'yolo_conf': 0.6190729141235352,
 * 'efficientnet_class_name': 'Dry Leaf',
 * 'efficientnet_confidence': 0.9994090795516968},
 * 
 * {'box': [143, 0, 327, 92],
 * 'yolo_conf': 0.5168587565422058,
 * 'efficientnet_class_name': 'Anthracnose',
 * 'efficientnet_confidence': 0.7569513916969299},
 * 
 * {'box': [319, 0, 407, 65],
 * 'yolo_conf': 0.498854398727417,
 * 'efficientnet_class_name': 'Anthracnose',
 * 'efficientnet_confidence': 0.9111652970314026},
 * 
 * {'box': [0, 362, 66, 407],
 * 'yolo_conf': 0.3878941833972931,
 * 'efficientnet_class_name': 'Anthracnose',
 * 'efficientnet_confidence': 0.9179883599281311}]
 */